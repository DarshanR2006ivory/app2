"""Tests for ThreatClassifier — unit tests and property-based tests.

Feature: asteroid-threat-monitor
Property 7: Threat classification rule correctness
Property 8: Threat level change recording
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import ThreatClassifierConfig
from src.models import ForecastStep, PositionSample, PredictionRecord, ThreatRecord, TrackRecord
from src.threat_classifier import ThreatClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _default_config() -> ThreatClassifierConfig:
    return ThreatClassifierConfig(
        dangerous_distance_au=0.002,
        dangerous_velocity_kms=10.0,
        hazardous_distance_au=0.05,
    )


def _make_track(track_id: str = "TRK-001", velocity_kms: float | None = None) -> TrackRecord:
    now = _utcnow()
    sample = PositionSample(
        frame_id="frame_001",
        timestamp=now,
        centroid_x=100.0,
        centroid_y=200.0,
        bbox=(90, 190, 110, 210),
        velocity_px=(1.0, 0.5),
        velocity_kms=velocity_kms,
        angular_diameter_arcsec=None,
        distance_au=None,
        approx_flag=True,
    )
    return TrackRecord(
        track_id=track_id,
        status="active",
        created_at=now,
        updated_at=now,
        frames_since_last_detection=0,
        history=[sample],
    )


def _make_prediction(track_id: str, closest_approach_au: float) -> PredictionRecord:
    now = _utcnow()
    return PredictionRecord(
        track_id=track_id,
        computed_at=now,
        horizon_hours=72.0,
        forecast_steps=[
            ForecastStep(time_offset_hours=1.0, position_au=(1.0, 0.0, 0.0), confidence_interval_au=0.001)
        ],
        closest_approach_au=closest_approach_au,
        closest_approach_time=now + timedelta(hours=24),
        intersects_earth_corridor=closest_approach_au <= 0.05,
        model_used="orbital",
    )


# ---------------------------------------------------------------------------
# Unit tests — classification rules
# ---------------------------------------------------------------------------

class TestClassificationRules:
    def test_dangerous_when_close_and_fast(self):
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=15.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.001)
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Dangerous"

    def test_safe_when_close_but_slow(self):
        """Close approach but velocity <= threshold → not Dangerous, check other rules."""
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=5.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.001)
        # closest_approach_au=0.001 < 0.002 but velocity=5 <= 10, so not Dangerous
        # 0.001 < 0.002 so not in [0.002, 0.05] range → Safe
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Safe"

    def test_potentially_hazardous_in_range(self):
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=5.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.03)
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Potentially_Hazardous"

    def test_safe_when_far(self):
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=20.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.1)
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Safe"

    def test_no_velocity_data_cannot_be_dangerous(self):
        """Without velocity data, Dangerous classification is impossible."""
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=None)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.001)
        result = classifier.classify(track, prediction)
        assert result.threat_level != "Dangerous"

    # Boundary conditions at exactly 0.002 AU
    def test_boundary_0002_au_is_potentially_hazardous(self):
        """At exactly 0.002 AU, should be Potentially_Hazardous (lower bound of range)."""
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=20.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.002)
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Potentially_Hazardous"

    def test_just_below_0002_au_with_high_velocity_is_dangerous(self):
        """Just below 0.002 AU with high velocity → Dangerous."""
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=15.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.0019)
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Dangerous"

    # Boundary conditions at exactly 0.05 AU
    def test_boundary_005_au_is_potentially_hazardous(self):
        """At exactly 0.05 AU, should be Potentially_Hazardous (upper bound of range)."""
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=5.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.05)
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Potentially_Hazardous"

    def test_just_above_005_au_is_safe(self):
        """Just above 0.05 AU → Safe."""
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=5.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.0501)
        result = classifier.classify(track, prediction)
        assert result.threat_level == "Safe"


# ---------------------------------------------------------------------------
# Unit tests — change tracking
# ---------------------------------------------------------------------------

class TestChangeTracking:
    def test_first_classification_has_no_previous_level(self):
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=5.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.03)
        result = classifier.classify(track, prediction)
        assert result.previous_threat_level is None
        assert result.changed_at is None

    def test_changed_at_is_none_when_level_unchanged(self):
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=5.0)
        prediction = _make_prediction("TRK-001", closest_approach_au=0.03)
        classifier.classify(track, prediction)
        result2 = classifier.classify(track, prediction)
        assert result2.changed_at is None
        assert result2.previous_threat_level == "Potentially_Hazardous"

    def test_changed_at_is_non_null_when_level_changes(self):
        classifier = ThreatClassifier(_default_config())
        track = _make_track(velocity_kms=5.0)

        # First: Potentially_Hazardous
        pred1 = _make_prediction("TRK-001", closest_approach_au=0.03)
        classifier.classify(track, pred1)

        # Second: Safe (level changes)
        track2 = _make_track(velocity_kms=5.0)
        pred2 = _make_prediction("TRK-001", closest_approach_au=0.1)
        result2 = classifier.classify(track2, pred2)

        assert result2.threat_level == "Safe"
        assert result2.previous_threat_level == "Potentially_Hazardous"
        assert result2.changed_at is not None

    def test_previous_level_recorded_correctly_on_change(self):
        classifier = ThreatClassifier(_default_config())
        track_dangerous = _make_track(velocity_kms=15.0)
        pred_dangerous = _make_prediction("TRK-001", closest_approach_au=0.001)
        classifier.classify(track_dangerous, pred_dangerous)

        track_safe = _make_track(velocity_kms=5.0)
        pred_safe = _make_prediction("TRK-001", closest_approach_au=0.1)
        result = classifier.classify(track_safe, pred_safe)

        assert result.previous_threat_level == "Dangerous"
        assert result.threat_level == "Safe"
        assert result.changed_at is not None

    def test_tracks_are_independent(self):
        """Different track IDs maintain independent state."""
        classifier = ThreatClassifier(_default_config())

        track_a = _make_track("TRK-A", velocity_kms=5.0)
        pred_a = _make_prediction("TRK-A", closest_approach_au=0.03)
        classifier.classify(track_a, pred_a)

        track_b = _make_track("TRK-B", velocity_kms=5.0)
        pred_b = _make_prediction("TRK-B", closest_approach_au=0.03)
        result_b = classifier.classify(track_b, pred_b)

        # TRK-B has no prior history, so previous_threat_level should be None
        assert result_b.previous_threat_level is None
        assert result_b.changed_at is None


# ---------------------------------------------------------------------------
# Property 7: Threat classification rule correctness
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# ---------------------------------------------------------------------------

@st.composite
def classification_inputs(draw):
    """Generate (track, prediction) pairs with controlled velocity and distance."""
    closest_approach_au = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    velocity_kms = draw(st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ))
    track = _make_track(track_id="TRK-PROP", velocity_kms=velocity_kms)
    prediction = _make_prediction("TRK-PROP", closest_approach_au=closest_approach_au)
    return track, prediction, closest_approach_au, velocity_kms


@given(data=classification_inputs())
@settings(max_examples=100)
def test_property7_threat_classification_rule_correctness(data):
    """
    Property 7: Threat classification rule correctness
    For any PredictionRecord and TrackRecord, the ThreatRecord must satisfy exactly
    the documented rules:
      - Dangerous iff closest_approach_au < 0.002 AND velocity_kms > 10
      - Potentially_Hazardous iff 0.002 <= closest_approach_au <= 0.05
      - Safe iff closest_approach_au > 0.05
    Every input must produce exactly one of the three levels.
    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    """
    track, prediction, closest_au, velocity_kms = data
    config = _default_config()
    classifier = ThreatClassifier(config)
    result = classifier.classify(track, prediction)

    # Must produce exactly one of the three levels
    assert result.threat_level in ("Safe", "Potentially_Hazardous", "Dangerous")

    # Verify rule correctness
    if closest_au < config.dangerous_distance_au and velocity_kms is not None and velocity_kms > config.dangerous_velocity_kms:
        assert result.threat_level == "Dangerous", (
            f"Expected Dangerous for au={closest_au}, v={velocity_kms}, got {result.threat_level}"
        )
    elif config.dangerous_distance_au <= closest_au <= config.hazardous_distance_au:
        assert result.threat_level == "Potentially_Hazardous", (
            f"Expected Potentially_Hazardous for au={closest_au}, got {result.threat_level}"
        )
    else:
        assert result.threat_level == "Safe", (
            f"Expected Safe for au={closest_au}, v={velocity_kms}, got {result.threat_level}"
        )


# ---------------------------------------------------------------------------
# Property 8: Threat level change recording
# Validates: Requirements 6.5
# ---------------------------------------------------------------------------

_THREAT_LEVELS = ["Safe", "Potentially_Hazardous", "Dangerous"]


def _prediction_for_level(track_id: str, level: str, config: ThreatClassifierConfig) -> PredictionRecord:
    """Create a prediction that will produce the given threat level."""
    if level == "Dangerous":
        au = config.dangerous_distance_au / 2  # well below threshold
    elif level == "Potentially_Hazardous":
        au = (config.dangerous_distance_au + config.hazardous_distance_au) / 2
    else:  # Safe
        au = config.hazardous_distance_au * 2
    return _make_prediction(track_id, closest_approach_au=au)


def _track_for_level(track_id: str, level: str, config: ThreatClassifierConfig) -> TrackRecord:
    """Create a track with velocity appropriate for the given threat level."""
    if level == "Dangerous":
        velocity = config.dangerous_velocity_kms * 2  # well above threshold
    else:
        velocity = config.dangerous_velocity_kms / 2  # below threshold
    return _make_track(track_id=track_id, velocity_kms=velocity)


@given(
    level_sequence=st.lists(
        st.sampled_from(_THREAT_LEVELS),
        min_size=2,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_property8_threat_level_change_recording(level_sequence: list[str]):
    """
    Property 8: Threat level change recording
    For any sequence of ThreatRecords for the same track where the threat level
    changes, the new ThreatRecord must record the previous level, the new level,
    and a non-null changed_at timestamp. When the level does not change,
    changed_at must be None.
    Validates: Requirements 6.5
    """
    config = _default_config()
    classifier = ThreatClassifier(config)
    track_id = "TRK-SEQ"

    previous_level: str | None = None
    for level in level_sequence:
        track = _track_for_level(track_id, level, config)
        prediction = _prediction_for_level(track_id, level, config)
        result = classifier.classify(track, prediction)

        assert result.threat_level in ("Safe", "Potentially_Hazardous", "Dangerous")

        if previous_level is None:
            # First classification: no previous level, no change timestamp
            assert result.previous_threat_level is None
            assert result.changed_at is None
        elif previous_level != result.threat_level:
            # Level changed: must record previous level and non-null changed_at
            assert result.previous_threat_level == previous_level, (
                f"Expected previous_threat_level={previous_level}, got {result.previous_threat_level}"
            )
            assert result.changed_at is not None, (
                f"changed_at must be non-null when level changes from {previous_level} to {result.threat_level}"
            )
        else:
            # Level unchanged: changed_at must be None
            assert result.changed_at is None, (
                f"changed_at must be None when level stays {result.threat_level}"
            )

        previous_level = result.threat_level
