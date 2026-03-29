"""Unit tests for ApiClient (Task 11.1)."""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.api_client import ApiClient
from src.config import ApiConfig, DataStoreConfig
from src.data_store import DataStore
from src.models import NeoRecord, PositionSample, TrackRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_config(**kwargs) -> ApiConfig:
    defaults = dict(
        enabled=True,
        nasa_neows_api_key="TEST_KEY",
        fetch_interval_hours=1.0,
        max_requests_per_hour=10,
        match_tolerance_au=0.01,
    )
    defaults.update(kwargs)
    return ApiConfig(**defaults)


def _make_data_store() -> DataStore:
    cfg = DataStoreConfig(backend="sqlite", sqlite_path=":memory:")
    return DataStore(cfg)


def _make_neo_response(neo_id: str = "12345", miss_distance_au: float = 0.05) -> dict:
    """Build a minimal NeoWs-style API response."""
    today = date.today().isoformat()
    return {
        "near_earth_objects": {
            today: [
                {
                    "id": neo_id,
                    "name": f"NEO {neo_id}",
                    "absolute_magnitude_h": 20.5,
                    "is_potentially_hazardous_asteroid": False,
                    "estimated_diameter": {
                        "kilometers": {
                            "estimated_diameter_min": 0.1,
                            "estimated_diameter_max": 0.3,
                        }
                    },
                    "close_approach_data": [
                        {
                            "close_approach_date": today,
                            "miss_distance": {"astronomical": str(miss_distance_au)},
                            "relative_velocity": {"kilometers_per_second": "15.0"},
                        }
                    ],
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Test: API failure logs error and continues without interrupting pipeline
# ---------------------------------------------------------------------------

class TestApiFailure:
    def test_connection_error_logs_and_uses_cache(self, caplog):
        """On requests.ConnectionError, log error and fall back to cached data."""
        config = _make_config()
        ds = _make_data_store()
        client = ApiClient(config, ds)

        # Pre-populate cache with one NEO
        cached_neo = NeoRecord(
            neo_id="cached-1",
            name="Cached NEO",
            absolute_magnitude=18.0,
            estimated_diameter_km=(0.1, 0.2),
            is_potentially_hazardous=False,
            close_approach_date=date.today(),
            miss_distance_au=0.05,
            relative_velocity_kms=10.0,
            fetched_at=_utcnow(),
        )
        client._cached_neos = [cached_neo]

        with patch("requests.get", side_effect=requests.ConnectionError("unreachable")):
            import logging
            with caplog.at_level(logging.ERROR, logger="src.api_client"):
                client._fetch_and_process()

        # Error should be logged
        assert any("API request failed" in r.message or "request failed" in r.message
                   for r in caplog.records), "Expected error log on API failure"

        # Cached NEO should have been upserted into the data store
        stored = ds.get_neo_catalogue()
        assert len(stored) == 1
        assert stored[0].neo_id == "cached-1"

    def test_http_error_logs_and_uses_cache(self, caplog):
        """On HTTP 500, log error and fall back to cached data."""
        config = _make_config()
        ds = _make_data_store()
        client = ApiClient(config, ds)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with patch("requests.get", return_value=mock_response):
            import logging
            with caplog.at_level(logging.ERROR, logger="src.api_client"):
                client._fetch_and_process()

        assert any("API request failed" in r.message or "request failed" in r.message
                   for r in caplog.records)

    def test_api_failure_does_not_raise(self):
        """API failure must not propagate an exception (pipeline must continue)."""
        config = _make_config()
        ds = _make_data_store()
        client = ApiClient(config, ds)

        with patch("requests.get", side_effect=requests.ConnectionError("unreachable")):
            # Should not raise
            client._fetch_and_process()

    def test_disabled_client_skips_fetch(self):
        """When enabled=False, _fetch_and_process should do nothing."""
        config = _make_config(enabled=False)
        ds = _make_data_store()
        client = ApiClient(config, ds)

        with patch("requests.get") as mock_get:
            client._fetch_and_process()
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Rate limit not exceeded across multiple scheduled fetches
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_blocks_excess_requests(self):
        """After max_requests_per_hour requests, further requests are blocked."""
        config = _make_config(max_requests_per_hour=3)
        ds = _make_data_store()
        client = ApiClient(config, ds)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = _make_neo_response()

        request_count = 0

        def counting_get(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            return mock_response

        with patch("requests.get", side_effect=counting_get):
            # Make 5 fetch attempts; only 3 should actually call requests.get
            for _ in range(5):
                client._fetch_and_process()

        assert request_count == 3, (
            f"Expected exactly 3 HTTP requests (rate limit), got {request_count}"
        )

    def test_rate_limit_resets_after_one_hour(self):
        """Requests older than 1 hour are not counted against the rate limit."""
        config = _make_config(max_requests_per_hour=2)
        ds = _make_data_store()
        client = ApiClient(config, ds)

        # Manually inject old timestamps (> 1 hour ago)
        old_time = time.monotonic() - 3700  # 1 hour + 100 seconds ago
        client._request_timestamps.append(old_time)
        client._request_timestamps.append(old_time)

        # Should still be allowed since old timestamps are expired
        assert client._check_rate_limit() is True

    def test_rate_limit_check_counts_recent_requests(self):
        """Recent requests within the hour window count against the limit."""
        config = _make_config(max_requests_per_hour=2)
        ds = _make_data_store()
        client = ApiClient(config, ds)

        # Record 2 recent requests
        client._record_request()
        client._record_request()

        # Should be at limit now
        assert client._check_rate_limit() is False

    def test_rate_limit_warning_logged_when_exceeded(self, caplog):
        """A warning is logged when the rate limit prevents a fetch."""
        config = _make_config(max_requests_per_hour=1)
        ds = _make_data_store()
        client = ApiClient(config, ds)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = _make_neo_response()

        import logging
        with patch("requests.get", return_value=mock_response):
            with caplog.at_level(logging.WARNING, logger="src.api_client"):
                client._fetch_and_process()  # first fetch — allowed
                client._fetch_and_process()  # second fetch — rate limited

        assert any("rate limit" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: NEO parsing and upsert
# ---------------------------------------------------------------------------

class TestNeoParsingAndUpsert:
    def test_successful_fetch_upserts_neos(self):
        """A successful API response results in NEOs being stored."""
        config = _make_config()
        ds = _make_data_store()
        client = ApiClient(config, ds)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = _make_neo_response(neo_id="99999", miss_distance_au=0.03)

        with patch("requests.get", return_value=mock_response):
            client._fetch_and_process()

        stored = ds.get_neo_catalogue()
        assert len(stored) == 1
        assert stored[0].neo_id == "99999"
        assert abs(stored[0].miss_distance_au - 0.03) < 1e-9

    def test_parse_response_extracts_fields(self):
        """_parse_response correctly extracts all NeoRecord fields."""
        config = _make_config()
        ds = _make_data_store()
        client = ApiClient(config, ds)

        data = _make_neo_response(neo_id="42", miss_distance_au=0.07)
        records = client._parse_response(data)

        assert len(records) == 1
        r = records[0]
        assert r.neo_id == "42"
        assert r.name == "NEO 42"
        assert r.absolute_magnitude == 20.5
        assert r.estimated_diameter_km == (0.1, 0.3)
        assert r.is_potentially_hazardous is False
        assert abs(r.miss_distance_au - 0.07) < 1e-9
        assert abs(r.relative_velocity_kms - 15.0) < 1e-9


# ---------------------------------------------------------------------------
# Test: Track matching / annotation
# ---------------------------------------------------------------------------

class TestTrackMatching:
    def _make_track_with_distance(self, track_id: str, distance_au: float) -> TrackRecord:
        sample = PositionSample(
            frame_id="f1",
            timestamp=_utcnow(),
            centroid_x=100.0,
            centroid_y=100.0,
            bbox=(0, 0, 10, 10),
            velocity_px=(0.0, 0.0),
            velocity_kms=None,
            angular_diameter_arcsec=None,
            distance_au=distance_au,
            approx_flag=False,
        )
        return TrackRecord(
            track_id=track_id,
            status="active",
            created_at=_utcnow(),
            updated_at=_utcnow(),
            frames_since_last_detection=0,
            history=[sample],
        )

    def test_track_annotated_when_within_tolerance(self):
        """Track is annotated with NEO id/name when distance matches within tolerance."""
        config = _make_config(match_tolerance_au=0.01)
        ds = _make_data_store()
        client = ApiClient(config, ds)

        neo = NeoRecord(
            neo_id="neo-1",
            name="Test NEO",
            absolute_magnitude=18.0,
            estimated_diameter_km=(0.1, 0.2),
            is_potentially_hazardous=False,
            close_approach_date=date.today(),
            miss_distance_au=0.05,
            relative_velocity_kms=10.0,
            fetched_at=_utcnow(),
        )

        track = self._make_track_with_distance("TRK-001", 0.055)  # within 0.01 AU

        ds.upsert_neo(neo)

        # Persist the track first so get_active_tracks returns it
        from src.models import ThreatRecord
        threat = ThreatRecord(
            track_id="TRK-001",
            threat_level="Safe",
            previous_threat_level=None,
            closest_approach_au=0.055,
            velocity_kms=None,
            changed_at=None,
            evaluated_at=_utcnow(),
        )
        ds.upsert_track(track, None, threat)

        client._match_tracks([neo])

        # Verify annotation was persisted to the data store
        active_tracks = ds.get_active_tracks()
        annotated = next((t for t in active_tracks if t.track_id == "TRK-001"), None)
        assert annotated is not None
        assert annotated.neo_catalogue_id == "neo-1"
        assert annotated.neo_name == "Test NEO"

    def test_track_not_annotated_when_outside_tolerance(self):
        """Track is NOT annotated when distance difference exceeds tolerance."""
        config = _make_config(match_tolerance_au=0.01)
        ds = _make_data_store()
        client = ApiClient(config, ds)

        neo = NeoRecord(
            neo_id="neo-2",
            name="Far NEO",
            absolute_magnitude=18.0,
            estimated_diameter_km=(0.1, 0.2),
            is_potentially_hazardous=False,
            close_approach_date=date.today(),
            miss_distance_au=0.05,
            relative_velocity_kms=10.0,
            fetched_at=_utcnow(),
        )

        track = self._make_track_with_distance("TRK-002", 0.10)  # 0.05 AU away — outside tolerance

        from src.models import ThreatRecord
        threat = ThreatRecord(
            track_id="TRK-002",
            threat_level="Safe",
            previous_threat_level=None,
            closest_approach_au=0.10,
            velocity_kms=None,
            changed_at=None,
            evaluated_at=_utcnow(),
        )
        ds.upsert_track(track, None, threat)

        client._match_tracks([neo])

        assert track.neo_catalogue_id is None
        assert track.neo_name is None
