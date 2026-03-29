"""API client: fetches NASA NeoWs data and annotates matching tracks."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import date, datetime, timezone
from typing import Any

import requests

from src.config import ApiConfig
from src.data_store import DataStore
from src.models import NeoRecord

logger = logging.getLogger(__name__)

_NEOWS_URL = "https://api.nasa.gov/neo/rest/v1/feed"


class ApiClient:
    def __init__(self, config: ApiConfig, data_store: DataStore) -> None:
        self.config = config
        self.data_store = data_store

        # Rate-limit tracking: timestamps of requests made in the current hour window
        self._request_timestamps: deque[float] = deque()
        self._lock = threading.Lock()

        # Cached NEO records from last successful fetch
        self._cached_neos: list[NeoRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_scheduled_fetch(self) -> None:
        """Run the fetch loop in the current thread (intended for background thread use).

        Fetches NeoWs data at the configured interval indefinitely.
        """
        while True:
            self._fetch_and_process()
            time.sleep(self.config.fetch_interval_hours * 3600)

    def start_background(self) -> threading.Thread:
        """Start the scheduled fetch loop in a daemon background thread."""
        t = threading.Thread(target=self.run_scheduled_fetch, daemon=True, name="api-client")
        t.start()
        return t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_process(self) -> None:
        """Perform one fetch cycle: get NEOs, upsert, match tracks."""
        if not self.config.enabled:
            return

        neos = self._fetch_neos()
        if neos is not None:
            self._cached_neos = neos
        else:
            # API failure — use cached data
            neos = self._cached_neos

        for neo in neos:
            self.data_store.upsert_neo(neo)

        self._match_tracks(neos)

    def _fetch_neos(self) -> list[NeoRecord] | None:
        """Fetch today's NEO feed from NASA NeoWs.

        Returns a list of NeoRecord on success, or None on failure.
        """
        if not self._check_rate_limit():
            logger.warning(
                "API rate limit reached (%d req/hr). Skipping fetch.",
                self.config.max_requests_per_hour,
            )
            return None

        today = date.today().isoformat()
        params = {
            "start_date": today,
            "end_date": today,
            "api_key": self.config.nasa_neows_api_key,
        }

        try:
            self._record_request()
            response = requests.get(_NEOWS_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)
        except requests.RequestException as exc:
            logger.error("NASA NeoWs API request failed: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error fetching NeoWs data: %s", exc)
            return None

    def _check_rate_limit(self) -> bool:
        """Return True if we can make another request without exceeding the rate limit."""
        with self._lock:
            now = time.monotonic()
            # Remove timestamps older than 1 hour
            cutoff = now - 3600.0
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()
            return len(self._request_timestamps) < self.config.max_requests_per_hour

    def _record_request(self) -> None:
        """Record that a request was made right now."""
        with self._lock:
            self._request_timestamps.append(time.monotonic())

    def _parse_response(self, data: dict[str, Any]) -> list[NeoRecord]:
        """Parse the NeoWs JSON response into a list of NeoRecord."""
        records: list[NeoRecord] = []
        fetched_at = datetime.now(tz=timezone.utc)

        near_earth_objects = data.get("near_earth_objects", {})
        for date_str, neo_list in near_earth_objects.items():
            try:
                approach_date = date.fromisoformat(date_str)
            except ValueError:
                approach_date = date.today()

            for neo_data in neo_list:
                try:
                    record = self._parse_neo_entry(neo_data, approach_date, fetched_at)
                    if record is not None:
                        records.append(record)
                except Exception as exc:
                    neo_id = neo_data.get("id", "unknown")
                    logger.warning("Failed to parse NEO entry %s: %s", neo_id, exc)

        return records

    def _parse_neo_entry(
        self,
        neo_data: dict[str, Any],
        approach_date: date,
        fetched_at: datetime,
    ) -> NeoRecord | None:
        """Parse a single NEO entry from the API response."""
        neo_id = neo_data.get("id", "")
        name = neo_data.get("name", "")
        absolute_magnitude = float(neo_data.get("absolute_magnitude_h", 0.0))
        is_hazardous = bool(neo_data.get("is_potentially_hazardous_asteroid", False))

        # Estimated diameter
        diameter_data = neo_data.get("estimated_diameter", {})
        km_data = diameter_data.get("kilometers", {})
        diameter_min = float(km_data.get("estimated_diameter_min", 0.0))
        diameter_max = float(km_data.get("estimated_diameter_max", 0.0))

        # Close approach data — pick the first entry
        close_approaches = neo_data.get("close_approach_data", [])
        miss_distance_au = 0.0
        relative_velocity_kms = 0.0
        if close_approaches:
            ca = close_approaches[0]
            miss_dist = ca.get("miss_distance", {})
            miss_distance_au = float(miss_dist.get("astronomical", 0.0))
            rel_vel = ca.get("relative_velocity", {})
            relative_velocity_kms = float(rel_vel.get("kilometers_per_second", 0.0))
            # Use the close approach date from the entry if available
            ca_date_str = ca.get("close_approach_date", "")
            if ca_date_str:
                try:
                    approach_date = date.fromisoformat(ca_date_str)
                except ValueError:
                    pass

        return NeoRecord(
            neo_id=neo_id,
            name=name,
            absolute_magnitude=absolute_magnitude,
            estimated_diameter_km=(diameter_min, diameter_max),
            is_potentially_hazardous=is_hazardous,
            close_approach_date=approach_date,
            miss_distance_au=miss_distance_au,
            relative_velocity_kms=relative_velocity_kms,
            fetched_at=fetched_at,
        )

    def _match_tracks(self, neos: list[NeoRecord]) -> None:
        """Match active tracks to NEOs by distance and annotate matching TrackRecords."""
        if not neos:
            return

        try:
            active_tracks = self.data_store.get_active_tracks()
        except Exception as exc:
            logger.error("Failed to retrieve active tracks for NEO matching: %s", exc)
            return

        for track in active_tracks:
            # Get the most recent distance_au from track history
            track_distance_au = self._get_track_distance(track)
            if track_distance_au is None:
                continue

            best_neo: NeoRecord | None = None
            best_diff = float("inf")

            for neo in neos:
                diff = abs(track_distance_au - neo.miss_distance_au)
                if diff <= self.config.match_tolerance_au and diff < best_diff:
                    best_diff = diff
                    best_neo = neo

            if best_neo is not None:
                track.neo_catalogue_id = best_neo.neo_id
                track.neo_name = best_neo.name
                # Persist the annotation back to the data store
                try:
                    self._annotate_track(track)
                except Exception as exc:
                    logger.error(
                        "Failed to annotate track %s with NEO %s: %s",
                        track.track_id,
                        best_neo.neo_id,
                        exc,
                    )

    def _get_track_distance(self, track: "Any") -> float | None:
        """Return the most recent distance_au from a track's history, or None."""
        for sample in reversed(track.history):
            if sample.distance_au is not None:
                return sample.distance_au
        return None

    def _annotate_track(self, track: "Any") -> None:
        """Persist NEO annotation on a track record via the data store."""
        from src.models import ThreatRecord

        # We need to call upsert_track but we don't have prediction/threat here.
        # Use a minimal threat record to satisfy the interface.
        now = datetime.now(tz=timezone.utc)
        minimal_threat = ThreatRecord(
            track_id=track.track_id,
            threat_level="Safe",
            previous_threat_level=None,
            closest_approach_au=0.0,
            velocity_kms=None,
            changed_at=None,
            evaluated_at=now,
        )
        self.data_store.upsert_track(track, None, minimal_threat)
