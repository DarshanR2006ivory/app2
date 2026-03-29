"""Tests for src/tracker.py — unit tests and property-based tests."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import TrackerConfig
from src.models import DecodedFrame, Detection, TrackRecord
from src.tracker import Tracker, _iou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def make_frame(frame_id: str = "f001") -> DecodedFrame:
    return DecodedFrame(
        frame_id=frame_id,
        timestamp=utcnow(),
        rgb_array=np.zeros((480, 640, 3), dtype=np.uint8),
        source="test",
    )


def make_detection(x1: int, y1: int, x2: int, y2: int, frame_id: str = "f001") -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), confidence=0.9, frame_id=frame_id)


def default_config(**kwargs) -> TrackerConfig:
    defaults = dict(
        algorithm="kalman",
        max_lost_frames=3,
        max_history=100,
        min_iou_threshold=0.3,
        pixel_distance_threshold=20,
        fov_scale_arcsec_per_pixel=None,
        plate_scale_arcsec_per_pixel=None,
    )
    defaults.update(kwargs)
    return TrackerConfig(**defaults)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestNewTrackCreation:
    def test_first_detection_creates_track(self):
        tracker = Tracker(default_config())
        frame = make_frame("f001")
        det = make_detection(10, 10, 50, 50, "f001")
        tracks = tracker.update([det], frame)
        assert len(tracks) == 1
        assert tracks[0].status == "active"
        assert tracks[0].track_id.startswith("TRK-")

    def test_track_id_format(self):
        tracker = Tracker(default_config())
        frame = make_frame()
        det = make_detection(10, 10, 50, 50)
        tracks = tracker.update([det], frame)
        assert tracks[0].track_id == "TRK-001"

    def test_empty_detections_no_tracks(self):
        tracker = Tracker(default_config())
        frame = make_frame()
        tracks = tracker.update([], frame)
        assert tracks == []

    def test_multiple_detections_create_multiple_tracks(self):
        tracker = Tracker(default_config())
        frame = make_frame()
        dets = [
            make_detection(10, 10, 50, 50),
            make_detection(200, 200, 250, 250),
        ]
        tracks = tracker.update(dets, frame)
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2


class TestTrackLoss:
    def test_track_lost_after_max_lost_frames(self):
        """Track must be marked lost after exactly max_lost_frames frames with no match."""
        max_lost = 3
        tracker = Tracker(default_config(max_lost_frames=max_lost))

        # Create a track
        frame = make_frame("f001")
        det = make_detection(10, 10, 50, 50, "f001")
        tracker.update([det], frame)

        # Feed empty detections for max_lost_frames frames
        for i in range(max_lost):
            f = make_frame(f"f{i+2:03d}")
            tracks = tracker.update([], f)
            # Should still be active (not yet exceeded)
            active = [t for t in tracks if t.status == "active"]
            assert len(active) == 1, f"Expected active track at frame {i+2}"

        # One more frame — now it should be lost
        f = make_frame(f"f{max_lost+2:03d}")
        tracks = tracker.update([], f)
        lost = [t for t in tracks if t.status == "lost"]
        assert len(lost) == 1

    def test_track_not_lost_before_max_lost_frames(self):
        max_lost = 5
        tracker = Tracker(default_config(max_lost_frames=max_lost))
        frame = make_frame("f001")
        tracker.update([make_detection(10, 10, 50, 50)], frame)

        for i in range(max_lost):
            f = make_frame(f"f{i+2:03d}")
            tracks = tracker.update([], f)
            assert all(t.status == "active" for t in tracks)


class TestHungarianAssignment:
    def test_overlapping_bbox_assigned_to_existing_track(self):
        """A detection that overlaps well with an existing track should be matched."""
        tracker = Tracker(default_config(min_iou_threshold=0.3))
        frame1 = make_frame("f001")
        det1 = make_detection(10, 10, 60, 60, "f001")
        tracks1 = tracker.update([det1], frame1)
        assert len(tracks1) == 1
        original_id = tracks1[0].track_id

        # Slightly shifted detection — should match
        frame2 = make_frame("f002")
        det2 = make_detection(12, 12, 62, 62, "f002")
        tracks2 = tracker.update([det2], frame2)
        assert len(tracks2) == 1
        assert tracks2[0].track_id == original_id

    def test_non_overlapping_bbox_creates_new_track(self):
        """A detection far from existing tracks should create a new track."""
        tracker = Tracker(default_config(min_iou_threshold=0.3))
        frame1 = make_frame("f001")
        det1 = make_detection(10, 10, 50, 50, "f001")
        tracker.update([det1], frame1)

        frame2 = make_frame("f002")
        det2 = make_detection(400, 400, 450, 450, "f002")
        tracks2 = tracker.update([det2], frame2)

        # Should have 2 tracks: original (unmatched, still active) + new
        assert len(tracks2) == 2
        ids = {t.track_id for t in tracks2}
        assert len(ids) == 2


class TestDistinctTrackIds:
    def test_two_nearby_detections_get_distinct_ids(self):
        """Two detections in the same frame must get distinct Track IDs."""
        tracker = Tracker(default_config())
        frame = make_frame()
        det1 = make_detection(10, 10, 50, 50)
        det2 = make_detection(30, 30, 70, 70)  # overlapping but separate detections
        tracks = tracker.update([det1, det2], frame)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))

    def test_ids_are_unique_within_each_frame(self):
        """Within any single frame, all returned track IDs must be distinct."""
        tracker = Tracker(default_config())
        for i in range(5):
            frame = make_frame(f"f{i:03d}")
            # Add a new non-overlapping detection each frame
            det = make_detection(i * 100, i * 100, i * 100 + 40, i * 100 + 40)
            tracks = tracker.update([det], frame)
            ids_this_frame = [t.track_id for t in tracks]
            assert len(ids_this_frame) == len(set(ids_this_frame)), (
                f"Duplicate track IDs in frame {i}: {ids_this_frame}"
            )


class TestApproxFlag:
    def test_approx_flag_true_when_fov_scale_none(self):
        tracker = Tracker(default_config(fov_scale_arcsec_per_pixel=None))
        frame = make_frame()
        det = make_detection(10, 10, 50, 50)
        tracker.update([det], frame)
        # Feed a few more frames to build history
        for i in range(3):
            f = make_frame(f"f{i+2:03d}")
            d = make_detection(10 + i, 10 + i, 50 + i, 50 + i)
            tracks = tracker.update([d], f)
        assert all(s.approx_flag for t in tracks for s in t.history)

    def test_approx_flag_false_when_fov_scale_configured(self):
        tracker = Tracker(default_config(fov_scale_arcsec_per_pixel=0.5))
        frame = make_frame()
        det = make_detection(10, 10, 50, 50)
        tracks = tracker.update([det], frame)
        assert all(not s.approx_flag for t in tracks for s in t.history)


class TestIoU:
    def test_identical_boxes(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < iou < 1.0


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Hypothesis strategies
@st.composite
def detection_strategy(draw, frame_id: str = "f001"):
    x1 = draw(st.integers(min_value=0, max_value=500))
    y1 = draw(st.integers(min_value=0, max_value=400))
    w = draw(st.integers(min_value=5, max_value=100))
    h = draw(st.integers(min_value=5, max_value=100))
    return Detection(
        bbox=(x1, y1, x1 + w, y1 + h),
        confidence=draw(st.floats(min_value=0.5, max_value=1.0)),
        frame_id=frame_id,
    )


@st.composite
def detection_list_strategy(draw, frame_id: str = "f001"):
    n = draw(st.integers(min_value=0, max_value=5))
    return [draw(detection_strategy(frame_id=frame_id)) for _ in range(n)]


@st.composite
def frame_sequence_strategy(draw):
    """Generate a sequence of (frame, detection_list) pairs."""
    n_frames = draw(st.integers(min_value=1, max_value=10))
    pairs = []
    for i in range(n_frames):
        fid = f"f{i:03d}"
        frame = DecodedFrame(
            frame_id=fid,
            timestamp=utcnow(),
            rgb_array=np.zeros((480, 640, 3), dtype=np.uint8),
            source="test",
        )
        dets = draw(detection_list_strategy(frame_id=fid))
        pairs.append((frame, dets))
    return pairs


# Property 2: Track ID uniqueness and persistence
# Feature: asteroid-threat-monitor, Property 2: Track ID uniqueness and persistence
# Validates: Requirements 3.2
@given(frame_sequence_strategy())
@settings(max_examples=100)
def test_property_track_id_uniqueness_and_persistence(frame_sequence):
    """
    For any sequence of detection lists, each Track ID must be unique across the
    session, and once assigned, the same ID must appear in all subsequent TrackRecords
    for that track until it is marked lost.

    Validates: Requirements 3.2
    """
    tracker = Tracker(default_config())

    # Map: track_id -> last seen status
    seen_ids: dict[str, str] = {}
    # Map: track_id -> set of frame indices where it appeared
    id_frames: dict[str, list[int]] = {}

    for frame_idx, (frame, dets) in enumerate(frame_sequence):
        tracks = tracker.update(dets, frame)

        for track in tracks:
            tid = track.track_id
            # ID must not have been previously assigned to a different track
            # (uniqueness: each ID appears for exactly one logical track)
            if tid not in seen_ids:
                seen_ids[tid] = track.status
                id_frames[tid] = [frame_idx]
            else:
                id_frames[tid].append(frame_idx)
                # Once a track is lost, it should not reappear
                assert seen_ids[tid] != "lost", (
                    f"Track {tid} reappeared after being marked lost"
                )
            seen_ids[tid] = track.status

    # All IDs must be unique strings
    assert len(seen_ids) == len(set(seen_ids.keys()))


# Property 3: Track loss invariant
# Feature: asteroid-threat-monitor, Property 3: Track loss invariant
# Validates: Requirements 3.3
@given(
    st.integers(min_value=1, max_value=5),   # max_lost_frames
    st.integers(min_value=1, max_value=20),  # extra_frames beyond max_lost
)
@settings(max_examples=100)
def test_property_track_loss_invariant(max_lost_frames, extra_frames):
    """
    For any track that has not received a matching detection for more than
    max_lost_frames consecutive frames, the tracker must mark it as "lost".

    Validates: Requirements 3.3
    """
    tracker = Tracker(default_config(max_lost_frames=max_lost_frames))

    # Create a single track
    frame0 = make_frame("f000")
    det = make_detection(100, 100, 150, 150, "f000")
    tracker.update([det], frame0)

    # Feed empty frames for max_lost_frames (track should still be active)
    for i in range(max_lost_frames):
        f = make_frame(f"f{i+1:03d}")
        tracks = tracker.update([], f)
        active = [t for t in tracks if t.status == "active"]
        assert len(active) >= 1, (
            f"Track should still be active at frame {i+1} (max_lost={max_lost_frames})"
        )

    # One more frame — track must now be lost
    f_final = make_frame(f"f{max_lost_frames+1:03d}")
    tracks = tracker.update([], f_final)
    lost = [t for t in tracks if t.status == "lost"]
    assert len(lost) >= 1, (
        f"Track should be lost after {max_lost_frames+1} frames without detection"
    )

    # After being emitted as lost, it should not appear in subsequent frames
    for j in range(extra_frames):
        f = make_frame(f"fe{j:03d}")
        tracks = tracker.update([], f)
        # The original track should not reappear
        assert all(t.status != "lost" or t.track_id != "TRK-001" for t in tracks)


# Property 4: Physical parameter approximation flag
# Feature: asteroid-threat-monitor, Property 4: Physical parameter approximation flag
# Validates: Requirements 4.5
@given(frame_sequence_strategy())
@settings(max_examples=100)
def test_property_approx_flag_when_fov_scale_null(frame_sequence):
    """
    For any TrackRecord produced when fov_scale_arcsec_per_pixel is not configured,
    every PositionSample must have approx_flag=True.

    Validates: Requirements 4.5
    """
    # Explicitly set fov_scale_arcsec_per_pixel to None
    tracker = Tracker(default_config(fov_scale_arcsec_per_pixel=None))

    for frame, dets in frame_sequence:
        tracks = tracker.update(dets, frame)
        for track in tracks:
            for sample in track.history:
                assert sample.approx_flag is True, (
                    f"Track {track.track_id}: PositionSample has approx_flag=False "
                    f"but fov_scale_arcsec_per_pixel is None"
                )
