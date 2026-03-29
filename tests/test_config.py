"""Unit and property-based tests for src/config.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import (
    AppConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Unit tests — Task 1.3
# ---------------------------------------------------------------------------

class TestConfigMissingFile:
    def test_missing_file_exits_with_code_1(self, tmp_path: Path) -> None:
        """Requirement 11.2: missing config file → exit code 1."""
        with pytest.raises(SystemExit) as exc_info:
            load_config(tmp_path / "nonexistent.yaml")
        assert exc_info.value.code == 1


class TestConfigMalformedYAML:
    def test_malformed_yaml_exits_with_code_1(self, tmp_path: Path) -> None:
        """Requirement 11.2: malformed YAML → exit code 1."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed bracket\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(bad)
        assert exc_info.value.code == 1

    def test_non_mapping_yaml_exits_with_code_1(self, tmp_path: Path) -> None:
        """Top-level YAML that is not a mapping should exit with code 1."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(bad)
        assert exc_info.value.code == 1


class TestConfigDefaults:
    def test_empty_config_applies_all_defaults(self, minimal_config_yaml: Path) -> None:
        """Requirement 11.4: empty config file → all documented defaults applied."""
        cfg = load_config(minimal_config_yaml)
        assert isinstance(cfg, AppConfig)
        # image_processor defaults
        assert cfg.image_processor.source_type == "directory"
        assert cfg.image_processor.decode_timeout_ms == 50
        # detector defaults
        assert cfg.detector.confidence_threshold == 0.5
        assert cfg.detector.input_resolution == [640, 640]
        # tracker defaults
        assert cfg.tracker.algorithm == "kalman"
        assert cfg.tracker.max_lost_frames == 10
        assert cfg.tracker.max_history == 100
        assert cfg.tracker.min_iou_threshold == 0.3
        assert cfg.tracker.fov_scale_arcsec_per_pixel is None
        # trajectory_predictor defaults
        assert cfg.trajectory_predictor.min_samples == 20
        assert cfg.trajectory_predictor.horizon_hours == 72.0
        assert cfg.trajectory_predictor.earth_corridor_au == 0.05
        assert cfg.trajectory_predictor.lstm_blend_weight == 0.3
        # threat_classifier defaults
        assert cfg.threat_classifier.dangerous_distance_au == 0.002
        assert cfg.threat_classifier.dangerous_velocity_kms == 10.0
        assert cfg.threat_classifier.hazardous_distance_au == 0.05
        # alert_manager defaults
        assert cfg.alert_manager.deduplication_window_seconds == 300
        assert cfg.alert_manager.retry_attempts == 3
        assert cfg.alert_manager.email.enabled is False
        assert cfg.alert_manager.sms.enabled is False
        # data_store defaults
        assert cfg.data_store.backend == "sqlite"
        assert cfg.data_store.retention_days == 90
        # api_client defaults
        assert cfg.api_client.enabled is False
        assert cfg.api_client.nasa_neows_api_key == "DEMO_KEY"
        # dashboard defaults
        assert cfg.dashboard.refresh_interval_seconds == 1
        assert cfg.dashboard.port == 8501

    def test_full_config_loads_without_error(self, full_config_yaml: Path) -> None:
        """A fully specified valid config should load without error."""
        cfg = load_config(full_config_yaml)
        assert cfg.detector.confidence_threshold == 0.5


class TestConfigValidation:
    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    def test_invalid_confidence_threshold_above_1(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "detector:\n  confidence_threshold: 1.5\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_confidence_threshold_below_0(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "detector:\n  confidence_threshold: -0.1\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_max_lost_frames_zero(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "tracker:\n  max_lost_frames: 0\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_max_history_zero(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "tracker:\n  max_history: 0\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_iou_threshold_above_1(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "tracker:\n  min_iou_threshold: 1.1\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_lstm_blend_weight_above_1(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "trajectory_predictor:\n  lstm_blend_weight: 1.5\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_data_store_backend(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "data_store:\n  backend: mysql\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_source_type(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "image_processor:\n  source_type: ftp\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1

    def test_invalid_tracker_algorithm(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "tracker:\n  algorithm: deep_sort\n")
        with pytest.raises(SystemExit) as exc_info:
            load_config(p)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Property 12: Configuration defaults applied — Task 1.1
# Validates: Requirements 11.4
# ---------------------------------------------------------------------------

# Strategy: generate a subset of optional keys to omit from the config
_OPTIONAL_KEYS = [
    "image_processor", "detector", "tracker", "trajectory_predictor",
    "threat_classifier", "alert_manager", "data_store", "api_client", "dashboard",
]


@st.composite
def partial_config_yaml(draw: st.DrawFn, tmp_path_factory) -> Path:  # type: ignore[type-arg]
    """Generate a YAML file with a random subset of top-level sections omitted."""
    keys_to_include = draw(st.lists(st.sampled_from(_OPTIONAL_KEYS), unique=True))
    lines: list[str] = []
    if "detector" in keys_to_include:
        threshold = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        lines.append(f"detector:\n  confidence_threshold: {threshold}\n")
    if "tracker" in keys_to_include:
        mlf = draw(st.integers(min_value=1, max_value=50))
        lines.append(f"tracker:\n  max_lost_frames: {mlf}\n")
    tmp = tmp_path_factory.mktemp("prop12")
    p = tmp / "config.yaml"
    p.write_text("\n".join(lines) if lines else "{}\n")
    return p


@settings(max_examples=100)
@given(st.data())
def test_property_12_config_defaults_applied(tmp_path_factory, data: st.DataObject) -> None:
    """Feature: asteroid-threat-monitor, Property 12: Configuration defaults applied.

    For any config file that omits one or more optional parameters, the loaded
    configuration object must contain the documented default value for each
    omitted parameter.

    Validates: Requirements 11.4
    """
    # Build a config that omits a random subset of sections
    sections_to_omit = data.draw(
        st.lists(st.sampled_from(_OPTIONAL_KEYS), min_size=1, unique=True)
    )
    tmp = tmp_path_factory.mktemp("prop12")
    lines: list[str] = []
    # Include only sections NOT in the omit list, with valid values
    if "detector" not in sections_to_omit:
        lines.append("detector:\n  confidence_threshold: 0.7\n")
    if "tracker" not in sections_to_omit:
        lines.append("tracker:\n  max_lost_frames: 5\n")
    p = tmp / "config.yaml"
    p.write_text("\n".join(lines) if lines else "{}\n")

    cfg = load_config(p)

    # Verify defaults for omitted sections
    if "detector" in sections_to_omit:
        assert cfg.detector.confidence_threshold == 0.5
        assert cfg.detector.input_resolution == [640, 640]
        assert cfg.detector.model_type == "yolov8"
    if "tracker" in sections_to_omit:
        assert cfg.tracker.max_lost_frames == 10
        assert cfg.tracker.max_history == 100
        assert cfg.tracker.min_iou_threshold == 0.3
        assert cfg.tracker.fov_scale_arcsec_per_pixel is None
    if "trajectory_predictor" in sections_to_omit:
        assert cfg.trajectory_predictor.min_samples == 20
        assert cfg.trajectory_predictor.horizon_hours == 72.0
        assert cfg.trajectory_predictor.earth_corridor_au == 0.05
        assert cfg.trajectory_predictor.lstm_blend_weight == 0.3
    if "threat_classifier" in sections_to_omit:
        assert cfg.threat_classifier.dangerous_distance_au == 0.002
        assert cfg.threat_classifier.dangerous_velocity_kms == 10.0
        assert cfg.threat_classifier.hazardous_distance_au == 0.05
    if "alert_manager" in sections_to_omit:
        assert cfg.alert_manager.deduplication_window_seconds == 300
        assert cfg.alert_manager.retry_attempts == 3
    if "data_store" in sections_to_omit:
        assert cfg.data_store.backend == "sqlite"
        assert cfg.data_store.retention_days == 90
    if "api_client" in sections_to_omit:
        assert cfg.api_client.enabled is False
        assert cfg.api_client.nasa_neows_api_key == "DEMO_KEY"
    if "dashboard" in sections_to_omit:
        assert cfg.dashboard.refresh_interval_seconds == 1
        assert cfg.dashboard.port == 8501


# ---------------------------------------------------------------------------
# Property 13: Configuration validation rejects invalid values — Task 1.2
# Validates: Requirements 11.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    confidence=st.one_of(
        st.floats(max_value=-0.001, allow_nan=False),
        st.floats(min_value=1.001, allow_nan=False, allow_infinity=False),
    )
)
def test_property_13_invalid_confidence_threshold_rejected(
    tmp_path_factory, confidence: float
) -> None:
    """Feature: asteroid-threat-monitor, Property 13: Configuration validation rejects invalid values.

    For any confidence_threshold outside [0, 1], the system must exit with
    a non-zero status code.

    Validates: Requirements 11.3
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(f"detector:\n  confidence_threshold: {confidence}\n")
        name = f.name
    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(name)
        assert exc_info.value.code != 0
    finally:
        os.unlink(name)


@settings(max_examples=100)
@given(max_lost_frames=st.integers(max_value=0))
def test_property_13_invalid_max_lost_frames_rejected(
    tmp_path_factory, max_lost_frames: int
) -> None:
    """Feature: asteroid-threat-monitor, Property 13: Configuration validation rejects invalid values.

    For any max_lost_frames < 1, the system must exit with a non-zero status code.

    Validates: Requirements 11.3
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(f"tracker:\n  max_lost_frames: {max_lost_frames}\n")
        name = f.name
    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(name)
        assert exc_info.value.code != 0
    finally:
        os.unlink(name)


@settings(max_examples=100)
@given(
    iou=st.one_of(
        st.floats(max_value=-0.001, allow_nan=False),
        st.floats(min_value=1.001, allow_nan=False, allow_infinity=False),
    )
)
def test_property_13_invalid_iou_threshold_rejected(tmp_path_factory, iou: float) -> None:
    """Feature: asteroid-threat-monitor, Property 13: Configuration validation rejects invalid values.

    For any min_iou_threshold outside [0, 1], the system must exit with
    a non-zero status code.

    Validates: Requirements 11.3
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(f"tracker:\n  min_iou_threshold: {iou}\n")
        name = f.name
    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(name)
        assert exc_info.value.code != 0
    finally:
        os.unlink(name)
