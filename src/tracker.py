"""Tracker: associates detections across frames and maintains track history."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from src.config import TrackerConfig
from src.models import DecodedFrame, Detection, DetectionList, PositionSample, TrackRecord

logger = logging.getLogger(__name__)

# Physical constants
_KM_PER_AU = 1.496e8          # km per AU
_ARCSEC_PER_RADIAN = 206265.0  # arcseconds per radian
# Approximate angular diameter of a 1 km object at 1 AU in arcsec
# Used for distance estimation: dist_au = ref_size_km / (angular_diam_arcsec / _ARCSEC_PER_RADIAN * _KM_PER_AU)
_REF_SIZE_KM = 1.0             # reference asteroid diameter in km


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union for two bounding boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _make_kalman_filter() -> KalmanFilter:
    """Create a 6-state Kalman filter: [cx, cy, vx, vy, w, h]."""
    kf = KalmanFilter(dim_x=6, dim_z=4)

    # Transition matrix (constant velocity model)
    kf.F = np.array([
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=float)

    # Measurement matrix (observe centroid and size)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=float)

    # Measurement noise covariance
    kf.R = np.eye(4) * 10.0

    # Process noise covariance
    kf.Q = np.eye(6) * 1.0

    # Initial state covariance
    kf.P = np.eye(6) * 100.0

    return kf


class _Track:
    """Internal track state."""

    def __init__(self, track_id: str, detection: Detection, frame: DecodedFrame) -> None:
        self.track_id = track_id
        self.created_at = _utcnow()
        self.updated_at = _utcnow()
        self.frames_since_last_detection = 0
        self.history: list[PositionSample] = []
        self.neo_catalogue_id: str | None = None
        self.neo_name: str | None = None

        # Initialise Kalman filter
        self.kf = _make_kalman_filter()
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)
        self.kf.x = np.array([[cx], [cy], [0.0], [0.0], [w], [h]])

    def predict(self) -> tuple[float, float, float, float]:
        """Run KF prediction step; return predicted (cx, cy, w, h)."""
        self.kf.predict()
        cx, cy, _, _, w, h = self.kf.x.flatten()
        return cx, cy, w, h

    def update(self, detection: Detection) -> None:
        """Update KF with a new measurement."""
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)
        z = np.array([[cx], [cy], [w], [h]])
        self.kf.update(z)

    def predicted_bbox(self) -> tuple[int, int, int, int]:
        """Return predicted bounding box from current KF state."""
        cx, cy, _, _, w, h = self.kf.x.flatten()
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return x1, y1, x2, y2

    def to_track_record(self, status: str) -> TrackRecord:
        return TrackRecord(
            track_id=self.track_id,
            status=status,  # type: ignore[arg-type]
            created_at=self.created_at,
            updated_at=self.updated_at,
            frames_since_last_detection=self.frames_since_last_detection,
            history=list(self.history),
            neo_catalogue_id=self.neo_catalogue_id,
            neo_name=self.neo_name,
        )


class Tracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: dict[str, _Track] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: DetectionList, frame: DecodedFrame) -> list[TrackRecord]:
        """Associate detections with existing tracks; return updated track list."""
        # 1. Predict next position for all existing tracks
        for track in self._tracks.values():
            track.predict()

        # 2. Build IoU cost matrix between existing tracks and new detections
        track_ids = list(self._tracks.keys())
        matched_track_ids: set[str] = set()
        matched_det_indices: set[int] = set()

        if track_ids and detections:
            cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=float)
            for i, tid in enumerate(track_ids):
                pred_bbox = self._tracks[tid].predicted_bbox()
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = 1.0 - _iou(pred_bbox, det.bbox)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                iou_val = 1.0 - cost_matrix[r, c]
                if iou_val >= self.config.min_iou_threshold:
                    tid = track_ids[r]
                    track = self._tracks[tid]
                    track.update(detections[c])
                    track.frames_since_last_detection = 0
                    track.updated_at = _utcnow()
                    matched_track_ids.add(tid)
                    matched_det_indices.add(c)

        # 3. Increment lost counter for unmatched tracks
        for tid in track_ids:
            if tid not in matched_track_ids:
                self._tracks[tid].frames_since_last_detection += 1

        # 4. Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                new_id = self._allocate_id()
                new_track = _Track(new_id, det, frame)
                self._tracks[new_id] = new_track

        # 5. Append position samples to all active tracks
        for tid, track in self._tracks.items():
            if track.frames_since_last_detection == 0:
                # Track was matched this frame — record position sample
                sample = self._build_position_sample(track, frame)
                track.history.append(sample)
                # Enforce rolling history limit
                if len(track.history) > self.config.max_history:
                    track.history = track.history[-self.config.max_history:]

        # 6. Determine status and collect results
        results: list[TrackRecord] = []
        to_remove: list[str] = []

        for tid, track in self._tracks.items():
            if track.frames_since_last_detection > self.config.max_lost_frames:
                results.append(track.to_track_record("lost"))
                to_remove.append(tid)
            else:
                results.append(track.to_track_record("active"))

        for tid in to_remove:
            del self._tracks[tid]

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _allocate_id(self) -> str:
        tid = f"TRK-{self._next_id:03d}"
        self._next_id += 1
        return tid

    def _build_position_sample(self, track: _Track, frame: DecodedFrame) -> PositionSample:
        """Build a PositionSample from the current KF state."""
        cx, cy, vx, vy, w, h = track.kf.x.flatten()
        bbox_w = max(1.0, w)
        bbox_h = max(1.0, h)

        x1 = int(cx - bbox_w / 2)
        y1 = int(cy - bbox_h / 2)
        x2 = int(cx + bbox_w / 2)
        y2 = int(cy + bbox_h / 2)
        bbox = (x1, y1, x2, y2)

        approx_flag = self.config.fov_scale_arcsec_per_pixel is None

        # Velocity in km/s (requires >= 5 samples for meaningful estimate)
        velocity_kms: float | None = None
        if len(track.history) >= 4 and self.config.fov_scale_arcsec_per_pixel is not None:
            # Speed in pixels/frame → arcsec/frame → km/frame → km/s
            # We assume 1 frame = 1 second for simplicity (no frame rate in config)
            speed_px_per_frame = float(np.hypot(vx, vy))
            speed_arcsec_per_frame = speed_px_per_frame * self.config.fov_scale_arcsec_per_pixel
            # arcsec/frame * (1 radian / 206265 arcsec) * dist_au * _KM_PER_AU
            # Without known distance, use a nominal 1 AU
            speed_rad_per_frame = speed_arcsec_per_frame / _ARCSEC_PER_RADIAN
            velocity_kms = speed_rad_per_frame * 1.0 * _KM_PER_AU  # nominal 1 AU

        # Angular diameter in arcsec
        angular_diameter_arcsec: float | None = None
        if self.config.plate_scale_arcsec_per_pixel is not None:
            angular_diameter_arcsec = bbox_w * self.config.plate_scale_arcsec_per_pixel

        # Distance in AU (from angular size + reference model)
        distance_au: float | None = None
        if angular_diameter_arcsec is not None and angular_diameter_arcsec > 0:
            # dist_au = ref_size_km / (angular_diam_rad * _KM_PER_AU)
            angular_diam_rad = angular_diameter_arcsec / _ARCSEC_PER_RADIAN
            distance_au = _REF_SIZE_KM / (angular_diam_rad * _KM_PER_AU)

        return PositionSample(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            centroid_x=float(cx),
            centroid_y=float(cy),
            bbox=bbox,
            velocity_px=(float(vx), float(vy)),
            velocity_kms=velocity_kms,
            angular_diameter_arcsec=angular_diameter_arcsec,
            distance_au=distance_au,
            approx_flag=approx_flag,
        )
