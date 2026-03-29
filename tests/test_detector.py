"""Tests for Detector (Task 3, sub-tasks 3.1 and 3.2).

Sub-task 3.1 — Property test for detection confidence filtering (Property 1)
  Feature: asteroid-threat-monitor, Property 1: Detection confidence filtering
  Validates: Requirements 2.2

Sub-task 3.2 — Unit tests for detector
  - Empty DetectionList returned on blank frame
  - Confidence threshold filtering at boundary values (0.0, 0.5, 1.0)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import DetectorConfig
from src.detector import Detector
from src.models import DecodedFrame, Detection, DetectionList


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    model_type: str = "yolov8",
    model_path: str = "./models/fake.pt",
    confidence_threshold: float = 0.5,
) -> DetectorConfig:
    return DetectorConfig(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        input_resolution=[640, 640],
    )


def _make_frame(frame_id: str = "frame_001") -> DecodedFrame:
    return DecodedFrame(
        frame_id=frame_id,
        timestamp=datetime.now(tz=timezone.utc),
        rgb_array=np.zeros((480, 640, 3), dtype=np.uint8),
        source="./data/frames",
    )


def _make_detector_with_mock_model(
    confidence_threshold: float = 0.5,
    raw_detections: DetectionList | None = None,
) -> Detector:
    """Create a Detector whose model loading is bypassed and _run_inference is mocked."""
    cfg = _make_config(confidence_threshold=confidence_threshold)
    with patch.object(Detector, "_load_model", return_value=None):
        detector = Detector(cfg)
    # Provide a non-None sentinel so detect() doesn't short-circuit on model=None
    detector._model = MagicMock()
    if raw_detections is not None:
        detector._run_inference = MagicMock(return_value=raw_detections)
    else:
        detector._run_inference = MagicMock(return_value=[])
    return detector


# ---------------------------------------------------------------------------
# Sub-task 3.2 — Unit tests
# ---------------------------------------------------------------------------

class TestDetectorUnit:
    def test_empty_list_on_blank_frame_no_model(self) -> None:
        """When no model is loaded, detect() returns an empty DetectionList."""
        cfg = _make_config()
        with patch.object(Detector, "_load_model", return_value=None):
            detector = Detector(cfg)
        # _model stays None
        result = detector.detect(_make_frame())
        assert result == []

    def test_empty_list_when_inference_returns_nothing(self) -> None:
        """detect() returns [] when the model produces no raw detections."""
        detector = _make_detector_with_mock_model(
            confidence_threshold=0.5,
            raw_detections=[],
        )
        result = detector.detect(_make_frame())
        assert result == []

    def test_threshold_0_0_passes_all_detections(self) -> None:
        """With threshold=0.0, all detections (including conf=0.0) are returned."""
        raw = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.0, frame_id="f"),
            Detection(bbox=(0, 0, 10, 10), confidence=0.3, frame_id="f"),
            Detection(bbox=(0, 0, 10, 10), confidence=1.0, frame_id="f"),
        ]
        detector = _make_detector_with_mock_model(confidence_threshold=0.0, raw_detections=raw)
        result = detector.detect(_make_frame())
        assert len(result) == 3

    def test_threshold_0_5_filters_below(self) -> None:
        """With threshold=0.5, detections with conf < 0.5 are excluded."""
        raw = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.3, frame_id="f"),
            Detection(bbox=(0, 0, 10, 10), confidence=0.5, frame_id="f"),  # boundary: included
            Detection(bbox=(0, 0, 10, 10), confidence=0.8, frame_id="f"),
        ]
        detector = _make_detector_with_mock_model(confidence_threshold=0.5, raw_detections=raw)
        result = detector.detect(_make_frame())
        assert len(result) == 2
        assert all(d.confidence >= 0.5 for d in result)

    def test_threshold_1_0_only_perfect_confidence(self) -> None:
        """With threshold=1.0, only detections with conf==1.0 are returned."""
        raw = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.99, frame_id="f"),
            Detection(bbox=(0, 0, 10, 10), confidence=1.0, frame_id="f"),
        ]
        detector = _make_detector_with_mock_model(confidence_threshold=1.0, raw_detections=raw)
        result = detector.detect(_make_frame())
        assert len(result) == 1
        assert result[0].confidence == 1.0

    def test_threshold_1_0_returns_empty_when_none_perfect(self) -> None:
        """With threshold=1.0 and no perfect-confidence detections, returns []."""
        raw = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.99, frame_id="f"),
        ]
        detector = _make_detector_with_mock_model(confidence_threshold=1.0, raw_detections=raw)
        result = detector.detect(_make_frame())
        assert result == []

    def test_returned_detections_carry_correct_frame_id(self) -> None:
        """Detections returned by detect() preserve the frame_id from inference."""
        raw = [Detection(bbox=(5, 5, 15, 15), confidence=0.9, frame_id="frame_042")]
        detector = _make_detector_with_mock_model(confidence_threshold=0.5, raw_detections=raw)
        frame = _make_frame(frame_id="frame_042")
        result = detector.detect(frame)
        assert result[0].frame_id == "frame_042"


# ---------------------------------------------------------------------------
# Sub-task 3.1 — Property test (Property 1)
# ---------------------------------------------------------------------------

# Feature: asteroid-threat-monitor, Property 1: Detection confidence filtering
# Validates: Requirements 2.2

@st.composite
def detection_strategy(draw: st.DrawFn) -> Detection:
    """Generate a Detection with a random confidence in [0, 1]."""
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    x1 = draw(st.integers(min_value=0, max_value=600))
    y1 = draw(st.integers(min_value=0, max_value=440))
    x2 = draw(st.integers(min_value=x1, max_value=640))
    y2 = draw(st.integers(min_value=y1, max_value=480))
    return Detection(bbox=(x1, y1, x2, y2), confidence=confidence, frame_id="prop_frame")


@given(
    raw_detections=st.lists(detection_strategy(), min_size=0, max_size=20),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_1_confidence_filtering(
    raw_detections: DetectionList,
    threshold: float,
) -> None:
    """**Validates: Requirements 2.2**

    Property 1: For any decoded frame and any confidence threshold in [0, 1],
    every detection returned by the detector must have a confidence score >= the
    configured threshold.
    """
    detector = _make_detector_with_mock_model(
        confidence_threshold=threshold,
        raw_detections=raw_detections,
    )
    frame = _make_frame()
    result = detector.detect(frame)

    # Core property: every returned detection meets the threshold
    for detection in result:
        assert detection.confidence >= threshold, (
            f"Detection with confidence {detection.confidence} was returned "
            f"but threshold is {threshold}"
        )

    # Completeness: no detection that should pass was dropped
    expected_count = sum(1 for d in raw_detections if d.confidence >= threshold)
    assert len(result) == expected_count, (
        f"Expected {expected_count} detections above threshold {threshold}, "
        f"got {len(result)}"
    )
