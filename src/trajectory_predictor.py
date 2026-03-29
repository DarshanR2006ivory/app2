"""Trajectory predictor: forecasts future asteroid positions from track history."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from src.config import TrajectoryConfig
from src.models import ForecastStep, PredictionRecord, TrackRecord

logger = logging.getLogger(__name__)

# AU per hour conversion factor (Earth orbital speed ~29.78 km/s)
_AU_PER_KM = 1.0 / 1.496e8  # 1 AU in km
_HOURS_PER_STEP = 1.0


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class TrajectoryPredictor:
    def __init__(self, config: TrajectoryConfig) -> None:
        self.config = config
        self._lstm_model = None
        if config.lstm_model_path is not None:
            self._lstm_model = self._load_lstm(config.lstm_model_path)

    def _load_lstm(self, path: str):
        """Load a PyTorch LSTM model from path. Returns None on failure."""
        try:
            import torch  # type: ignore
            model = torch.load(path, map_location="cpu")
            model.eval()
            return model
        except Exception as exc:
            logger.warning("Failed to load LSTM model from '%s': %s", path, exc)
            return None

    def predict(self, track: TrackRecord) -> PredictionRecord | None:
        """Return PredictionRecord, or None if track has fewer than min_samples."""
        n_samples = len(track.history)
        if n_samples < self.config.min_samples:
            logger.info(
                "Deferring prediction for track %s: only %d samples (need %d)",
                track.track_id,
                n_samples,
                self.config.min_samples,
            )
            return None

        now = _utcnow()
        forecast_steps, closest_approach_au, closest_approach_time = self._compute_orbital_forecast(
            track, now
        )

        model_used: str = "orbital"

        if self._lstm_model is not None:
            lstm_steps = self._compute_lstm_corrections(track, forecast_steps)
            if lstm_steps is not None:
                forecast_steps = lstm_steps
                model_used = "blended"
                # Recompute closest approach from blended forecast
                closest_approach_au, closest_approach_time = self._find_closest_approach(
                    forecast_steps, now
                )

        intersects = closest_approach_au <= self.config.earth_corridor_au

        return PredictionRecord(
            track_id=track.track_id,
            computed_at=now,
            horizon_hours=self.config.horizon_hours,
            forecast_steps=forecast_steps,
            closest_approach_au=closest_approach_au,
            closest_approach_time=closest_approach_time,
            intersects_earth_corridor=intersects,
            model_used=model_used,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Orbital mechanics (simplified linear extrapolation)
    # ------------------------------------------------------------------

    def _compute_orbital_forecast(
        self, track: TrackRecord, now: datetime
    ) -> tuple[list[ForecastStep], float, datetime]:
        """
        Simplified orbital forecast using linear extrapolation of pixel positions
        converted to angular coordinates, then mapped to AU-scale positions.

        Returns (forecast_steps, closest_approach_au, closest_approach_time).
        """
        history = track.history
        n = len(history)

        # Build time series of pixel positions
        times_s = np.array(
            [(s.timestamp - history[0].timestamp).total_seconds() for s in history],
            dtype=float,
        )
        xs = np.array([s.centroid_x for s in history], dtype=float)
        ys = np.array([s.centroid_y for s in history], dtype=float)

        # Fit linear trend (least squares) for x and y independently
        if times_s[-1] > 0:
            vx = float(np.polyfit(times_s, xs, 1)[0])  # px/s
            vy = float(np.polyfit(times_s, ys, 1)[0])
        else:
            vx, vy = 0.0, 0.0

        # Current position (last sample)
        x0 = xs[-1]
        y0 = ys[-1]

        # Estimate distance in AU from track (use last sample if available, else default)
        last = history[-1]
        distance_au = last.distance_au if last.distance_au is not None else 1.0

        # Plate scale: arcsec/pixel (use config if available, else 1.0 arcsec/px)
        plate_scale = 1.0  # arcsec/pixel fallback

        # Convert pixel velocity to angular velocity (arcsec/s)
        ang_vx = vx * plate_scale  # arcsec/s
        ang_vy = vy * plate_scale

        # Convert angular velocity to AU/s using small-angle approximation
        # 1 arcsec at distance d AU ≈ d * (1/206265) AU
        au_per_arcsec = distance_au / 206265.0
        vel_x_au_s = ang_vx * au_per_arcsec
        vel_y_au_s = ang_vy * au_per_arcsec

        # Starting position in AU (heliocentric approximation: place at distance_au from origin)
        pos_x0 = distance_au
        pos_y0 = 0.0
        pos_z0 = 0.0

        # Compute residuals for confidence interval
        predicted_xs = x0 + vx * (times_s - times_s[-1])
        residuals = xs - predicted_xs
        residual_std_px = float(np.std(residuals)) if len(residuals) > 1 else 0.0
        confidence_au = max(residual_std_px * au_per_arcsec * plate_scale, 1e-6)

        # Generate forecast steps
        n_steps = max(1, int(self.config.horizon_hours / _HOURS_PER_STEP))
        forecast_steps: list[ForecastStep] = []
        for i in range(1, n_steps + 1):
            t_hours = i * _HOURS_PER_STEP
            t_s = t_hours * 3600.0
            px = pos_x0 + vel_x_au_s * t_s
            py = pos_y0 + vel_y_au_s * t_s
            pz = pos_z0
            forecast_steps.append(
                ForecastStep(
                    time_offset_hours=t_hours,
                    position_au=(px, py, pz),
                    confidence_interval_au=confidence_au * math.sqrt(i),
                )
            )

        closest_approach_au, closest_approach_time = self._find_closest_approach(
            forecast_steps, now
        )
        return forecast_steps, closest_approach_au, closest_approach_time

    def _find_closest_approach(
        self, forecast_steps: list[ForecastStep], now: datetime
    ) -> tuple[float, datetime]:
        """Find the step with minimum distance to Earth (approximated at origin)."""
        if not forecast_steps:
            return 1.0, now

        # Earth position approximation: 1 AU from Sun along x-axis
        earth_x, earth_y, earth_z = 1.0, 0.0, 0.0

        min_dist = float("inf")
        min_time = now
        for step in forecast_steps:
            px, py, pz = step.position_au
            dist = math.sqrt(
                (px - earth_x) ** 2 + (py - earth_y) ** 2 + (pz - earth_z) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                min_time = now + timedelta(hours=step.time_offset_hours)

        return min_dist, min_time

    # ------------------------------------------------------------------
    # LSTM blending
    # ------------------------------------------------------------------

    def _compute_lstm_corrections(
        self, track: TrackRecord, orbital_steps: list[ForecastStep]
    ) -> list[ForecastStep] | None:
        """Apply LSTM delta corrections blended with orbital forecast."""
        try:
            import torch  # type: ignore

            history = track.history
            # Build normalised input: last N samples of (centroid_x, centroid_y)
            n_input = min(len(history), 20)
            samples = history[-n_input:]
            xs = np.array([s.centroid_x for s in samples], dtype=np.float32)
            ys = np.array([s.centroid_y for s in samples], dtype=np.float32)

            # Normalise
            x_mean, x_std = xs.mean(), xs.std() + 1e-8
            y_mean, y_std = ys.mean(), ys.std() + 1e-8
            xs_norm = (xs - x_mean) / x_std
            ys_norm = (ys - y_mean) / y_std

            seq = np.stack([xs_norm, ys_norm], axis=1)  # (n_input, 2)
            tensor = torch.tensor(seq).unsqueeze(0)  # (1, n_input, 2)

            with torch.no_grad():
                deltas = self._lstm_model(tensor)  # expected shape: (1, n_steps, 3)
                deltas = deltas.squeeze(0).numpy()  # (n_steps, 3)

            w = self.config.lstm_blend_weight
            blended: list[ForecastStep] = []
            for i, step in enumerate(orbital_steps):
                if i < len(deltas):
                    dx, dy, dz = deltas[i]
                else:
                    dx, dy, dz = 0.0, 0.0, 0.0
                ox, oy, oz = step.position_au
                blended.append(
                    ForecastStep(
                        time_offset_hours=step.time_offset_hours,
                        position_au=(ox + dx * w, oy + dy * w, oz + dz * w),
                        confidence_interval_au=step.confidence_interval_au,
                    )
                )
            return blended
        except Exception as exc:
            logger.warning("LSTM blending failed: %s", exc)
            return None
