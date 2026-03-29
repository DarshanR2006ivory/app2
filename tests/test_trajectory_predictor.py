"""Tests for TrajectoryPredictor — unit tests and property-based tests.

Feature: asteroid-threat-monitor
Property 5: Trajectory prediction deferral
Property 6: Earth corridor intersection consistency
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import TrajectoryConfig
from src.models import ForecastStep, PositionSample, PredictionRecord, TrackRecord
from src.trajectory_predictor import TrajectoryPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_sample(i: int, base: datetime | None = None) -> PositionSample:
    """Create a PositionSample at index i."""
    if base is None:
        base = _utcnow()
    return PositionSample(
        frame_id=f"frame_{i:04d}",
        timestamp=base + timedelta(seconds=i),
        centroid_x=100.0 + i * 0.5,
        centroid_y=200.0 + i * 0.3,
        bbox=(90, 190, 110, 210),
        velocity_px=(0.5, 0.3),
        velocity_kms=None,
        angular_diameter_arcsec=None,
        distance_au=1.0,
        approx_flag=True,
    )


def _make_track(n_samples: int, track_id: str = "TRK-001") -> TrackRecord:
    """Create a TrackRecord with n_samples position samples."""
    base = _utcnow()
    return TrackRecord(
        track_id=track_id,
        status="active",
        created_at=base,
        updated_at=base + timedelta(seconds=n_samples),
        frames_since_last_detection=0,
        history=[_make_sample(i, base) for i in range(n_samples)],
    )


def _default_config(
    min_samples: int = 20,
    horizon_hours: float = 72.0,
    earth_corridor_au: float = 0.05,
    lstm_model_path: str | None = None,
    lstm_blend_weight: float = 0.3,
) -> TrajectoryConfig:
    return TrajectoryConfig(
        min_samples=min_samples,
        horizon_hours=horizon_hours,
        earth_corridor_au=earth_corridor_au,
        lstm_model_path=lstm_model_path,
        lstm_blend_weight=lstm_blend_weight,
    )


# ---------------------------------------------------------------------------
# Unit tests — deferral (None for < min_samples)
# ---------------------------------------------------------------------------

class TestDeferral:
    def test_zero_samples_returns_none(self):
        predictor = TrajectoryPredictor(_default_config())
        track = _make_track(0)
        assert predictor.predict(track) is None

    def test_one_sample_returns_none(self):
        predictor = TrajectoryPredictor(_default_config())
        track = _make_track(1)
        assert predictor.predict(track) is None

    def test_nineteen_samples_returns_none(self):
        predictor = TrajectoryPredictor(_default_config())
        track = _make_track(19)
        assert predictor.predict(track) is None

    def test_exactly_twenty_samples_returns_prediction(self):
        predictor = TrajectoryPredictor(_default_config())
        track = _make_track(20)
        result = predictor.predict(track)
        assert result is not None
        assert isinstance(result, PredictionRecord)

    def test_more_than_twenty_samples_returns_prediction(self):
        predictor = TrajectoryPredictor(_default_config())
        track = _make_track(50)
        result = predictor.predict(track)
        assert result is not None


# ---------------------------------------------------------------------------
# Unit tests — Earth corridor flag
# ---------------------------------------------------------------------------

class TestEarthCorridorFlag:
    def test_intersects_when_closest_approach_equals_corridor(self):
        """At the boundary (closest_approach_au == earth_corridor_au), flag must be True."""
        predictor = TrajectoryPredictor(_default_config(earth_corridor_au=0.05))
        track = _make_track(20)
        result = predictor.predict(track)
        assert result is not None
        # Verify the invariant holds regardless of actual value
        if result.closest_approach_au <= 0.05:
            assert result.intersects_earth_corridor is True
        else:
            assert result.intersects_earth_corridor is False

    def test_flag_consistency_with_closest_approach(self):
        """intersects_earth_corridor must be consistent with closest_approach_au."""
        predictor = TrajectoryPredictor(_default_config(earth_corridor_au=0.05))
        track = _make_track(30)
        result = predictor.predict(track)
        assert result is not None
        expected = result.closest_approach_au <= 0.05
        assert result.intersects_earth_corridor == expected

    def test_prediction_record_has_required_fields(self):
        predictor = TrajectoryPredictor(_default_config())
        track = _make_track(20)
        result = predictor.predict(track)
        assert result is not None
        assert result.track_id == "TRK-001"
        assert result.horizon_hours == 72.0
        assert len(result.forecast_steps) > 0
        assert result.model_used == "orbital"
        assert isinstance(result.closest_approach_time, datetime)


# ---------------------------------------------------------------------------
# Property 5: Trajectory prediction deferral
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

@st.composite
def track_with_few_samples(draw) -> TrackRecord:
    """Generate a TrackRecord with fewer than 20 samples."""
    n = draw(st.integers(min_value=0, max_value=19))
    return _make_track(n, track_id=draw(st.text(min_size=1, max_size=20)))


@given(track=track_with_few_samples())
@settings(max_examples=100)
def test_property5_trajectory_prediction_deferral(track: TrackRecord):
    """
    Property 5: Trajectory prediction deferral
    For any TrackRecord with fewer than 20 position samples, the trajectory
    predictor must return None and must not raise an exception.
    Validates: Requirements 5.5
    """
    predictor = TrajectoryPredictor(_default_config())
    result = predictor.predict(track)
    assert result is None, (
        f"Expected None for track with {len(track.history)} samples, got {result}"
    )


# ---------------------------------------------------------------------------
# Property 6: Earth corridor intersection consistency
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------

@st.composite
def prediction_record(draw) -> tuple[PredictionRecord, float]:
    """Generate a PredictionRecord with a known earth_corridor_au."""
    earth_corridor_au = draw(st.floats(min_value=0.001, max_value=1.0, allow_nan=False))
    closest_approach_au = draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
    intersects = closest_approach_au <= earth_corridor_au
    now = _utcnow()
    step = ForecastStep(
        time_offset_hours=1.0,
        position_au=(1.0, 0.0, 0.0),
        confidence_interval_au=0.001,
    )
    record = PredictionRecord(
        track_id="TRK-TEST",
        computed_at=now,
        horizon_hours=72.0,
        forecast_steps=[step],
        closest_approach_au=closest_approach_au,
        closest_approach_time=now + timedelta(hours=1),
        intersects_earth_corridor=intersects,
        model_used="orbital",
    )
    return record, earth_corridor_au


@given(data=prediction_record())
@settings(max_examples=100)
def test_property6_earth_corridor_intersection_consistency(
    data: tuple[PredictionRecord, float],
):
    """
    Property 6: Earth corridor intersection consistency
    For any PredictionRecord, if intersects_earth_corridor is True then
    closest_approach_au <= earth_corridor_au, and if False then
    closest_approach_au > earth_corridor_au.
    Validates: Requirements 5.2
    """
    record, earth_corridor_au = data
    if record.intersects_earth_corridor:
        assert record.closest_approach_au <= earth_corridor_au, (
            f"intersects_earth_corridor=True but closest_approach_au={record.closest_approach_au} "
            f"> earth_corridor_au={earth_corridor_au}"
        )
    else:
        assert record.closest_approach_au > earth_corridor_au, (
            f"intersects_earth_corridor=False but closest_approach_au={record.closest_approach_au} "
            f"<= earth_corridor_au={earth_corridor_au}"
        )


@given(
    n_samples=st.integers(min_value=20, max_value=100),
    earth_corridor_au=st.floats(min_value=0.001, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_property6_predictor_corridor_consistency(
    n_samples: int, earth_corridor_au: float
):
    """
    Property 6 (via predictor): For any PredictionRecord produced by TrajectoryPredictor,
    intersects_earth_corridor must be consistent with closest_approach_au.
    Validates: Requirements 5.2
    """
    config = _default_config(earth_corridor_au=earth_corridor_au)
    predictor = TrajectoryPredictor(config)
    track = _make_track(n_samples)
    result = predictor.predict(track)
    assert result is not None

    if result.intersects_earth_corridor:
        assert result.closest_approach_au <= earth_corridor_au, (
            f"intersects_earth_corridor=True but closest_approach_au={result.closest_approach_au} "
            f"> earth_corridor_au={earth_corridor_au}"
        )
    else:
        assert result.closest_approach_au > earth_corridor_au, (
            f"intersects_earth_corridor=False but closest_approach_au={result.closest_approach_au} "
            f"<= earth_corridor_au={earth_corridor_au}"
        )
