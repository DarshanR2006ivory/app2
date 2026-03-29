"""Data models (dataclasses) forming the internal data contract between modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np


@dataclass
class DecodedFrame:
    frame_id: str
    timestamp: datetime
    rgb_array: np.ndarray  # shape (H, W, 3), dtype uint8
    source: str


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                # [0.0, 1.0]
    frame_id: str


# Type alias
DetectionList = list[Detection]


@dataclass
class PositionSample:
    frame_id: str
    timestamp: datetime
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    velocity_px: tuple[float, float]          # (vx, vy) pixels/frame
    velocity_kms: float | None                # km/s; None if scale unknown
    angular_diameter_arcsec: float | None
    distance_au: float | None
    approx_flag: bool                         # True when FOV scale not configured
    # Extended physical parameters (optional, injected by pipeline/simulator)
    estimated_mass_kg: float | None = None    # kg, derived from size + density model
    diameter_km: float | None = None          # estimated diameter in km
    shape: str | None = None                  # geometry: "spherical" | "elongated" | "irregular"
    rotation_period_hours: float | None = None  # estimated spin period


@dataclass
class TrackRecord:
    track_id: str
    status: Literal["active", "lost"]
    created_at: datetime
    updated_at: datetime
    frames_since_last_detection: int
    history: list[PositionSample] = field(default_factory=list)  # rolling, max 100
    neo_catalogue_id: str | None = None
    neo_name: str | None = None


@dataclass
class ForecastStep:
    time_offset_hours: float
    position_au: tuple[float, float, float]   # (x, y, z) heliocentric AU
    confidence_interval_au: float


@dataclass
class PredictionRecord:
    track_id: str
    computed_at: datetime
    horizon_hours: float
    forecast_steps: list[ForecastStep]
    closest_approach_au: float
    closest_approach_time: datetime
    intersects_earth_corridor: bool
    model_used: Literal["orbital", "lstm", "blended"]


@dataclass
class ThreatRecord:
    track_id: str
    threat_level: Literal["Safe", "Potentially_Hazardous", "Dangerous"]
    previous_threat_level: str | None
    closest_approach_au: float
    velocity_kms: float | None
    changed_at: datetime | None   # None if level unchanged
    evaluated_at: datetime


@dataclass
class AlertRecord:
    alert_id: str                   # UUID string
    track_id: str
    threat_level: str
    closest_approach_au: float
    velocity_kms: float | None
    created_at: datetime
    channels: list[str]             # e.g. ["visual", "email", "sms"]
    delivery_status: dict[str, str] # channel -> "sent" | "failed" | "suppressed"


@dataclass
class NeoRecord:
    neo_id: str                              # NASA catalogue ID
    name: str
    absolute_magnitude: float
    estimated_diameter_km: tuple[float, float]  # (min, max)
    is_potentially_hazardous: bool
    close_approach_date: date
    miss_distance_au: float
    relative_velocity_kms: float
    fetched_at: datetime
