"""Configuration loader: reads YAML, applies defaults, validates ranges/types."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImageProcessorConfig:
    source_type: str = "directory"
    source_path: str = "./data/frames"
    supported_formats: list[str] = field(default_factory=lambda: ["fits", "jpeg", "png", "tiff"])
    decode_timeout_ms: int = 50


@dataclass
class DetectorConfig:
    model_type: str = "yolov8"
    model_path: str = "./models/asteroid_yolov8.pt"
    confidence_threshold: float = 0.5
    input_resolution: list[int] = field(default_factory=lambda: [640, 640])


@dataclass
class TrackerConfig:
    algorithm: str = "kalman"
    max_lost_frames: int = 10
    max_history: int = 100
    min_iou_threshold: float = 0.3
    pixel_distance_threshold: int = 20
    fov_scale_arcsec_per_pixel: float | None = None
    plate_scale_arcsec_per_pixel: float | None = None


@dataclass
class TrajectoryConfig:
    min_samples: int = 20
    horizon_hours: float = 72.0
    earth_corridor_au: float = 0.05
    lstm_model_path: str | None = None
    lstm_blend_weight: float = 0.3


@dataclass
class ThreatClassifierConfig:
    dangerous_distance_au: float = 0.002
    dangerous_velocity_kms: float = 10.0
    hazardous_distance_au: float = 0.05


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_host: str = "smtp.example.com"
    smtp_port: int = 587
    sender: str = "alerts@example.com"
    recipients: list[str] = field(default_factory=list)
    username: str = ""
    password: str = ""


@dataclass
class SmsConfig:
    enabled: bool = False
    provider: str = "twilio"
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    to_numbers: list[str] = field(default_factory=list)


@dataclass
class AlertConfig:
    deduplication_window_seconds: int = 300
    retry_attempts: int = 3
    retry_interval_seconds: int = 10
    email: EmailConfig = field(default_factory=EmailConfig)
    sms: SmsConfig = field(default_factory=SmsConfig)


@dataclass
class DataStoreConfig:
    backend: str = "sqlite"
    sqlite_path: str = "./data/asteroid_monitor.db"
    postgresql_dsn: str = ""
    retention_days: int = 90


@dataclass
class ApiConfig:
    enabled: bool = False
    nasa_neows_api_key: str = "DEMO_KEY"
    fetch_interval_hours: float = 1.0
    max_requests_per_hour: int = 10
    match_tolerance_au: float = 0.01


@dataclass
class DashboardConfig:
    refresh_interval_seconds: int = 1
    port: int = 8501


@dataclass
class AppConfig:
    image_processor: ImageProcessorConfig = field(default_factory=ImageProcessorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    trajectory_predictor: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    threat_classifier: ThreatClassifierConfig = field(default_factory=ThreatClassifierConfig)
    alert_manager: AlertConfig = field(default_factory=AlertConfig)
    data_store: DataStoreConfig = field(default_factory=DataStoreConfig)
    api_client: ApiConfig = field(default_factory=ApiConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d: dict, key: str, default: Any = None) -> Any:
    return d.get(key, default)


def _validate_range(value: float, lo: float, hi: float, field_name: str) -> None:
    if not (lo <= value <= hi):
        logger.error("Config validation error: '%s' = %s is outside valid range [%s, %s]",
                     field_name, value, lo, hi)
        sys.exit(1)


def _validate_min(value: float, lo: float, field_name: str) -> None:
    if value < lo:
        logger.error("Config validation error: '%s' = %s must be >= %s", field_name, value, lo)
        sys.exit(1)


def _validate_choice(value: str, choices: list[str], field_name: str) -> None:
    if value not in choices:
        logger.error("Config validation error: '%s' = '%s' must be one of %s",
                     field_name, value, choices)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _parse_image_processor(raw: dict) -> ImageProcessorConfig:
    cfg = ImageProcessorConfig()
    cfg.source_type = _get(raw, "source_type", cfg.source_type)
    cfg.source_path = _get(raw, "source_path", cfg.source_path)
    cfg.supported_formats = _get(raw, "supported_formats", cfg.supported_formats)
    cfg.decode_timeout_ms = int(_get(raw, "decode_timeout_ms", cfg.decode_timeout_ms))
    _validate_choice(cfg.source_type, ["directory", "video", "rtsp", "http"],
                     "image_processor.source_type")
    _validate_min(cfg.decode_timeout_ms, 1, "image_processor.decode_timeout_ms")
    return cfg


def _parse_detector(raw: dict) -> DetectorConfig:
    cfg = DetectorConfig()
    cfg.model_type = _get(raw, "model_type", cfg.model_type)
    cfg.model_path = _get(raw, "model_path", cfg.model_path)
    cfg.confidence_threshold = float(_get(raw, "confidence_threshold", cfg.confidence_threshold))
    cfg.input_resolution = _get(raw, "input_resolution", cfg.input_resolution)
    _validate_choice(cfg.model_type, ["yolov8", "yolov5", "onnx", "pytorch"],
                     "detector.model_type")
    _validate_range(cfg.confidence_threshold, 0.0, 1.0, "detector.confidence_threshold")
    return cfg


def _parse_tracker(raw: dict) -> TrackerConfig:
    cfg = TrackerConfig()
    cfg.algorithm = _get(raw, "algorithm", cfg.algorithm)
    cfg.max_lost_frames = int(_get(raw, "max_lost_frames", cfg.max_lost_frames))
    cfg.max_history = int(_get(raw, "max_history", cfg.max_history))
    cfg.min_iou_threshold = float(_get(raw, "min_iou_threshold", cfg.min_iou_threshold))
    cfg.pixel_distance_threshold = int(_get(raw, "pixel_distance_threshold",
                                            cfg.pixel_distance_threshold))
    cfg.fov_scale_arcsec_per_pixel = _get(raw, "fov_scale_arcsec_per_pixel",
                                          cfg.fov_scale_arcsec_per_pixel)
    cfg.plate_scale_arcsec_per_pixel = _get(raw, "plate_scale_arcsec_per_pixel",
                                            cfg.plate_scale_arcsec_per_pixel)
    _validate_choice(cfg.algorithm, ["kalman", "optical_flow"], "tracker.algorithm")
    _validate_min(cfg.max_lost_frames, 1, "tracker.max_lost_frames")
    _validate_min(cfg.max_history, 1, "tracker.max_history")
    _validate_range(cfg.min_iou_threshold, 0.0, 1.0, "tracker.min_iou_threshold")
    return cfg


def _parse_trajectory(raw: dict) -> TrajectoryConfig:
    cfg = TrajectoryConfig()
    cfg.min_samples = int(_get(raw, "min_samples", cfg.min_samples))
    cfg.horizon_hours = float(_get(raw, "horizon_hours", cfg.horizon_hours))
    cfg.earth_corridor_au = float(_get(raw, "earth_corridor_au", cfg.earth_corridor_au))
    cfg.lstm_model_path = _get(raw, "lstm_model_path", cfg.lstm_model_path)
    cfg.lstm_blend_weight = float(_get(raw, "lstm_blend_weight", cfg.lstm_blend_weight))
    _validate_min(cfg.min_samples, 1, "trajectory_predictor.min_samples")
    _validate_range(cfg.lstm_blend_weight, 0.0, 1.0, "trajectory_predictor.lstm_blend_weight")
    return cfg


def _parse_threat_classifier(raw: dict) -> ThreatClassifierConfig:
    cfg = ThreatClassifierConfig()
    cfg.dangerous_distance_au = float(_get(raw, "dangerous_distance_au",
                                           cfg.dangerous_distance_au))
    cfg.dangerous_velocity_kms = float(_get(raw, "dangerous_velocity_kms",
                                            cfg.dangerous_velocity_kms))
    cfg.hazardous_distance_au = float(_get(raw, "hazardous_distance_au",
                                           cfg.hazardous_distance_au))
    return cfg


def _parse_alert(raw: dict) -> AlertConfig:
    cfg = AlertConfig()
    cfg.deduplication_window_seconds = int(_get(raw, "deduplication_window_seconds",
                                                cfg.deduplication_window_seconds))
    cfg.retry_attempts = int(_get(raw, "retry_attempts", cfg.retry_attempts))
    cfg.retry_interval_seconds = int(_get(raw, "retry_interval_seconds",
                                          cfg.retry_interval_seconds))
    email_raw = _get(raw, "email", {}) or {}
    cfg.email = EmailConfig(
        enabled=bool(_get(email_raw, "enabled", False)),
        smtp_host=_get(email_raw, "smtp_host", "smtp.example.com"),
        smtp_port=int(_get(email_raw, "smtp_port", 587)),
        sender=_get(email_raw, "sender", "alerts@example.com"),
        recipients=_get(email_raw, "recipients", []),
        username=_get(email_raw, "username", ""),
        password=_get(email_raw, "password", ""),
    )
    sms_raw = _get(raw, "sms", {}) or {}
    cfg.sms = SmsConfig(
        enabled=bool(_get(sms_raw, "enabled", False)),
        provider=_get(sms_raw, "provider", "twilio"),
        account_sid=_get(sms_raw, "account_sid", ""),
        auth_token=_get(sms_raw, "auth_token", ""),
        from_number=_get(sms_raw, "from_number", ""),
        to_numbers=_get(sms_raw, "to_numbers", []),
    )
    _validate_min(cfg.retry_attempts, 0, "alert_manager.retry_attempts")
    return cfg


def _parse_data_store(raw: dict) -> DataStoreConfig:
    cfg = DataStoreConfig()
    cfg.backend = _get(raw, "backend", cfg.backend)
    cfg.sqlite_path = _get(raw, "sqlite_path", cfg.sqlite_path)
    cfg.postgresql_dsn = _get(raw, "postgresql_dsn", cfg.postgresql_dsn) or ""
    cfg.retention_days = int(_get(raw, "retention_days", cfg.retention_days))
    _validate_choice(cfg.backend, ["sqlite", "postgresql"], "data_store.backend")
    _validate_min(cfg.retention_days, 1, "data_store.retention_days")
    return cfg


def _parse_api_client(raw: dict) -> ApiConfig:
    cfg = ApiConfig()
    cfg.enabled = bool(_get(raw, "enabled", cfg.enabled))
    cfg.nasa_neows_api_key = _get(raw, "nasa_neows_api_key", cfg.nasa_neows_api_key)
    cfg.fetch_interval_hours = float(_get(raw, "fetch_interval_hours", cfg.fetch_interval_hours))
    cfg.max_requests_per_hour = int(_get(raw, "max_requests_per_hour", cfg.max_requests_per_hour))
    cfg.match_tolerance_au = float(_get(raw, "match_tolerance_au", cfg.match_tolerance_au))
    _validate_min(cfg.fetch_interval_hours, 0.0, "api_client.fetch_interval_hours")
    _validate_min(cfg.max_requests_per_hour, 1, "api_client.max_requests_per_hour")
    return cfg


def _parse_dashboard(raw: dict) -> DashboardConfig:
    cfg = DashboardConfig()
    cfg.refresh_interval_seconds = int(_get(raw, "refresh_interval_seconds",
                                            cfg.refresh_interval_seconds))
    cfg.port = int(_get(raw, "port", cfg.port))
    _validate_min(cfg.refresh_interval_seconds, 1, "dashboard.refresh_interval_seconds")
    _validate_min(cfg.port, 1, "dashboard.port")
    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> AppConfig:
    """Load, validate, and return AppConfig from a YAML file.

    Exits with code 1 on missing file, malformed YAML, or invalid values.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        logger.error("Malformed YAML in configuration file '%s': %s", config_path, exc)
        sys.exit(1)

    if not isinstance(raw, dict):
        logger.error("Configuration file '%s' must contain a YAML mapping at the top level",
                     config_path)
        sys.exit(1)

    app = AppConfig(
        image_processor=_parse_image_processor(raw.get("image_processor", {}) or {}),
        detector=_parse_detector(raw.get("detector", {}) or {}),
        tracker=_parse_tracker(raw.get("tracker", {}) or {}),
        trajectory_predictor=_parse_trajectory(raw.get("trajectory_predictor", {}) or {}),
        threat_classifier=_parse_threat_classifier(raw.get("threat_classifier", {}) or {}),
        alert_manager=_parse_alert(raw.get("alert_manager", {}) or {}),
        data_store=_parse_data_store(raw.get("data_store", {}) or {}),
        api_client=_parse_api_client(raw.get("api_client", {}) or {}),
        dashboard=_parse_dashboard(raw.get("dashboard", {}) or {}),
    )
    return app
