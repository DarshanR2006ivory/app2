"""Threat classifier: applies classification rules to prediction results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from src.config import ThreatClassifierConfig
from src.models import PredictionRecord, ThreatRecord, TrackRecord


class ThreatClassifier:
    def __init__(self, config: ThreatClassifierConfig) -> None:
        self.config = config
        # Maps track_id -> last known threat level
        self._last_threat_level: dict[str, str] = {}

    def classify(self, track: TrackRecord, prediction: PredictionRecord) -> ThreatRecord:
        """Return ThreatRecord based on prediction and configured thresholds."""
        closest_au = prediction.closest_approach_au

        # Derive velocity_kms from the most recent position sample with a value
        velocity_kms: float | None = None
        for sample in reversed(track.history):
            if sample.velocity_kms is not None:
                velocity_kms = sample.velocity_kms
                break

        # Apply classification rules
        if (
            closest_au < self.config.dangerous_distance_au
            and velocity_kms is not None
            and velocity_kms > self.config.dangerous_velocity_kms
        ):
            threat_level: Literal["Safe", "Potentially_Hazardous", "Dangerous"] = "Dangerous"
        elif self.config.dangerous_distance_au <= closest_au <= self.config.hazardous_distance_au:
            threat_level = "Potentially_Hazardous"
        else:
            threat_level = "Safe"

        # Determine previous level and change tracking
        previous = self._last_threat_level.get(track.track_id)
        now = datetime.now(tz=timezone.utc)
        changed_at: datetime | None = now if (previous is not None and previous != threat_level) else None

        # Update internal state
        self._last_threat_level[track.track_id] = threat_level

        return ThreatRecord(
            track_id=track.track_id,
            threat_level=threat_level,
            previous_threat_level=previous,
            closest_approach_au=closest_au,
            velocity_kms=velocity_kms,
            changed_at=changed_at,
            evaluated_at=now,
        )
