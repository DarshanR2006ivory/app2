"""Shared pytest fixtures for the Asteroid Threat Monitor test suite."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.models import (
    AlertRecord,
    DecodedFrame,
    Detection,
    ForecastStep,
    NeoRecord,
    PositionSample,
    PredictionRecord,
    ThreatRecord,
    TrackRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_frame() -> DecodedFrame:
    return DecodedFrame(
        frame_id="frame_001",
        timestamp=utcnow(),
        rgb_array=np.zeros((480, 640, 3), dtype=np.uint8),
        source="./data/frames",
    )


@pytest.fixture()
def sample_detection() -> Detection:
    return Detection(bbox=(10, 20, 50, 60), confidence=0.85, frame_id="frame_001")


@pytest.fixture()
def sample_position_sample() -> PositionSample:
    return PositionSample(
        frame_id="frame_001",
        timestamp=utcnow(),
        centroid_x=30.0,
        centroid_y=40.0,
        bbox=(10, 20, 50, 60),
        velocity_px=(1.5, -0.5),
        velocity_kms=None,
        angular_diameter_arcsec=None,
        distance_au=None,
        approx_flag=True,
    )


@pytest.fixture()
def sample_track(sample_position_sample: PositionSample) -> TrackRecord:
    return TrackRecord(
        track_id="TRK-001",
        status="active",
        created_at=utcnow(),
        updated_at=utcnow(),
        frames_since_last_detection=0,
        history=[sample_position_sample],
    )


@pytest.fixture()
def sample_forecast_step() -> ForecastStep:
    return ForecastStep(
        time_offset_hours=1.0,
        position_au=(1.0, 0.0, 0.0),
        confidence_interval_au=0.001,
    )


@pytest.fixture()
def sample_prediction(sample_forecast_step: ForecastStep) -> PredictionRecord:
    return PredictionRecord(
        track_id="TRK-001",
        computed_at=utcnow(),
        horizon_hours=72.0,
        forecast_steps=[sample_forecast_step],
        closest_approach_au=0.03,
        closest_approach_time=utcnow(),
        intersects_earth_corridor=True,
        model_used="orbital",
    )


@pytest.fixture()
def sample_threat() -> ThreatRecord:
    return ThreatRecord(
        track_id="TRK-001",
        threat_level="Potentially_Hazardous",
        previous_threat_level=None,
        closest_approach_au=0.03,
        velocity_kms=5.0,
        changed_at=None,
        evaluated_at=utcnow(),
    )


# ---------------------------------------------------------------------------
# Config YAML fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_config_yaml(tmp_path: Path) -> Path:
    """A minimal valid config file (all defaults)."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}\n")
    return cfg


@pytest.fixture()
def full_config_yaml(tmp_path: Path) -> Path:
    """A fully specified valid config file."""
    content = textwrap.dedent("""\
        image_processor:
          source_type: directory
          source_path: ./data/frames
          supported_formats: [fits, jpeg, png, tiff]
          decode_timeout_ms: 50

        detector:
          model_type: yolov8
          model_path: ./models/asteroid_yolov8.pt
          confidence_threshold: 0.5
          input_resolution: [640, 640]

        tracker:
          algorithm: kalman
          max_lost_frames: 10
          max_history: 100
          min_iou_threshold: 0.3
          pixel_distance_threshold: 20

        trajectory_predictor:
          min_samples: 20
          horizon_hours: 72
          earth_corridor_au: 0.05
          lstm_blend_weight: 0.3

        threat_classifier:
          dangerous_distance_au: 0.002
          dangerous_velocity_kms: 10.0
          hazardous_distance_au: 0.05

        alert_manager:
          deduplication_window_seconds: 300
          retry_attempts: 3
          retry_interval_seconds: 10

        data_store:
          backend: sqlite
          sqlite_path: ./data/asteroid_monitor.db
          retention_days: 90

        api_client:
          enabled: false
          nasa_neows_api_key: DEMO_KEY
          fetch_interval_hours: 1
          max_requests_per_hour: 10
          match_tolerance_au: 0.01

        dashboard:
          refresh_interval_seconds: 1
          port: 8501
    """)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(content)
    return cfg
