"""Tests for DataStore: unit tests and property-based test (Property 11)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import DataStoreConfig
from src.data_store import DataStore
from src.models import (
    AlertRecord,
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

def _in_memory_store() -> DataStore:
    cfg = DataStoreConfig(backend="sqlite", sqlite_path=":memory:", retention_days=90)
    return DataStore(cfg)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_position_sample(frame_id: str = "f001", ts: datetime | None = None) -> PositionSample:
    return PositionSample(
        frame_id=frame_id,
        timestamp=ts or _utcnow(),
        centroid_x=100.0,
        centroid_y=200.0,
        bbox=(10, 20, 50, 60),
        velocity_px=(1.0, -0.5),
        velocity_kms=5.0,
        angular_diameter_arcsec=None,
        distance_au=0.03,
        approx_flag=False,
    )


def _make_track(
    track_id: str = "TRK-001",
    status: str = "active",
    history: list[PositionSample] | None = None,
) -> TrackRecord:
    now = _utcnow()
    return TrackRecord(
        track_id=track_id,
        status=status,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
        frames_since_last_detection=0,
        history=history or [_make_position_sample()],
    )


def _make_threat(track_id: str = "TRK-001") -> ThreatRecord:
    return ThreatRecord(
        track_id=track_id,
        threat_level="Safe",
        previous_threat_level=None,
        closest_approach_au=0.1,
        velocity_kms=5.0,
        changed_at=None,
        evaluated_at=_utcnow(),
    )


def _make_alert(track_id: str = "TRK-001") -> AlertRecord:
    return AlertRecord(
        alert_id=str(uuid.uuid4()),
        track_id=track_id,
        threat_level="Potentially_Hazardous",
        closest_approach_au=0.03,
        velocity_kms=5.0,
        created_at=_utcnow(),
        channels=["visual"],
        delivery_status={"visual": "sent"},
    )


def _make_neo(neo_id: str = "NEO-001") -> NeoRecord:
    return NeoRecord(
        neo_id=neo_id,
        name="Asteroid Test",
        absolute_magnitude=18.5,
        estimated_diameter_km=(0.1, 0.3),
        is_potentially_hazardous=True,
        close_approach_date=date(2025, 6, 1),
        miss_distance_au=0.04,
        relative_velocity_kms=12.3,
        fetched_at=_utcnow(),
    )


def _make_prediction(track_id: str = "TRK-001") -> PredictionRecord:
    return PredictionRecord(
        track_id=track_id,
        computed_at=_utcnow(),
        horizon_hours=72.0,
        forecast_steps=[
            ForecastStep(time_offset_hours=1.0, position_au=(1.0, 0.0, 0.0), confidence_interval_au=0.001)
        ],
        closest_approach_au=0.03,
        closest_approach_time=_utcnow(),
        intersects_earth_corridor=True,
        model_used="orbital",
    )


# ---------------------------------------------------------------------------
# Unit tests: TrackRecord round-trip
# ---------------------------------------------------------------------------

class TestTrackRoundTrip:
    def test_upsert_and_get_active_track(self) -> None:
        store = _in_memory_store()
        track = _make_track()
        threat = _make_threat()
        store.upsert_track(track, None, threat)

        active = store.get_active_tracks()
        assert len(active) == 1
        assert active[0].track_id == "TRK-001"
        assert active[0].status == "active"

    def test_position_samples_persisted(self) -> None:
        store = _in_memory_store()
        samples = [_make_position_sample(f"f{i:03d}") for i in range(3)]
        track = _make_track(history=samples)
        store.upsert_track(track, None, _make_threat())

        history = store.get_track_history("TRK-001")
        assert len(history) == 3

    def test_upsert_idempotent_for_same_frame(self) -> None:
        """Calling upsert_track twice with the same samples should not duplicate them."""
        store = _in_memory_store()
        track = _make_track()
        threat = _make_threat()
        store.upsert_track(track, None, threat)
        store.upsert_track(track, None, threat)

        history = store.get_track_history("TRK-001")
        assert len(history) == 1  # no duplicates

    def test_get_track_history_limit(self) -> None:
        store = _in_memory_store()
        samples = [_make_position_sample(f"f{i:03d}") for i in range(10)]
        track = _make_track(history=samples)
        store.upsert_track(track, None, _make_threat())

        history = store.get_track_history("TRK-001", limit=5)
        assert len(history) == 5


# ---------------------------------------------------------------------------
# Unit tests: closed_at set when track marked lost
# ---------------------------------------------------------------------------

class TestClosedAt:
    def test_closed_at_set_when_lost(self) -> None:
        store = _in_memory_store()
        track = _make_track(status="active")
        store.upsert_track(track, None, _make_threat())

        # Mark as lost
        lost_track = _make_track(status="lost")
        store.upsert_track(lost_track, None, _make_threat())

        # Lost tracks should not appear in get_active_tracks
        active = store.get_active_tracks()
        assert all(t.track_id != "TRK-001" for t in active)

        # Verify closed_at is set in DB
        with store._Session() as session:
            from src.data_store import TrackORM
            row = session.get(TrackORM, "TRK-001")
            assert row is not None
            assert row.closed_at is not None

    def test_closed_at_not_set_for_active(self) -> None:
        store = _in_memory_store()
        track = _make_track(status="active")
        store.upsert_track(track, None, _make_threat())

        with store._Session() as session:
            from src.data_store import TrackORM
            row = session.get(TrackORM, "TRK-001")
            assert row is not None
            assert row.closed_at is None


# ---------------------------------------------------------------------------
# Unit tests: retention query
# ---------------------------------------------------------------------------

class TestRetention:
    def test_retention_excludes_old_records(self) -> None:
        store = _in_memory_store()

        # Insert a track with created_at > 90 days ago (should be excluded)
        old_time = _utcnow() - timedelta(days=100)
        old_track = TrackRecord(
            track_id="TRK-OLD",
            status="active",
            created_at=old_time,
            updated_at=old_time,
            frames_since_last_detection=0,
            history=[_make_position_sample()],
        )
        store.upsert_track(old_track, None, _make_threat("TRK-OLD"))

        # Insert a recent track
        recent_track = _make_track("TRK-NEW")
        store.upsert_track(recent_track, None, _make_threat("TRK-NEW"))

        active = store.get_active_tracks()
        track_ids = [t.track_id for t in active]
        assert "TRK-OLD" not in track_ids
        assert "TRK-NEW" in track_ids


# ---------------------------------------------------------------------------
# Unit tests: AlertRecord round-trip
# ---------------------------------------------------------------------------

class TestAlertRoundTrip:
    def test_insert_and_get_alert(self) -> None:
        store = _in_memory_store()
        # Need a track first (FK)
        track = _make_track()
        store.upsert_track(track, None, _make_threat())

        alert = _make_alert()
        store.insert_alert(alert)

        since = _utcnow() - timedelta(seconds=10)
        alerts = store.get_alerts(since)
        assert len(alerts) == 1
        assert alerts[0].alert_id == alert.alert_id
        assert alerts[0].track_id == "TRK-001"
        assert alerts[0].channels == ["visual"]
        assert alerts[0].delivery_status == {"visual": "sent"}

    def test_get_alerts_filters_by_since(self) -> None:
        store = _in_memory_store()
        track = _make_track()
        store.upsert_track(track, None, _make_threat())

        old_alert = AlertRecord(
            alert_id=str(uuid.uuid4()),
            track_id="TRK-001",
            threat_level="Safe",
            closest_approach_au=0.1,
            velocity_kms=None,
            created_at=_utcnow() - timedelta(hours=2),
            channels=["visual"],
            delivery_status={"visual": "sent"},
        )
        new_alert = _make_alert()
        store.insert_alert(old_alert)
        store.insert_alert(new_alert)

        since = _utcnow() - timedelta(minutes=5)
        alerts = store.get_alerts(since)
        assert len(alerts) == 1
        assert alerts[0].alert_id == new_alert.alert_id

    def test_insert_alert_idempotent(self) -> None:
        store = _in_memory_store()
        track = _make_track()
        store.upsert_track(track, None, _make_threat())

        alert = _make_alert()
        store.insert_alert(alert)
        store.insert_alert(alert)  # second insert should be ignored

        since = _utcnow() - timedelta(seconds=10)
        alerts = store.get_alerts(since)
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# Unit tests: NeoRecord round-trip
# ---------------------------------------------------------------------------

class TestNeoRoundTrip:
    def test_upsert_and_get_neo(self) -> None:
        store = _in_memory_store()
        neo = _make_neo()
        store.upsert_neo(neo)

        catalogue = store.get_neo_catalogue()
        assert len(catalogue) == 1
        assert catalogue[0].neo_id == "NEO-001"
        assert catalogue[0].name == "Asteroid Test"
        assert catalogue[0].estimated_diameter_km == (0.1, 0.3)
        assert catalogue[0].is_potentially_hazardous is True

    def test_upsert_neo_updates_existing(self) -> None:
        store = _in_memory_store()
        neo = _make_neo()
        store.upsert_neo(neo)

        updated = NeoRecord(
            neo_id="NEO-001",
            name="Updated Name",
            absolute_magnitude=20.0,
            estimated_diameter_km=(0.2, 0.5),
            is_potentially_hazardous=False,
            close_approach_date=date(2025, 7, 1),
            miss_distance_au=0.06,
            relative_velocity_kms=8.0,
            fetched_at=_utcnow(),
        )
        store.upsert_neo(updated)

        catalogue = store.get_neo_catalogue()
        assert len(catalogue) == 1
        assert catalogue[0].name == "Updated Name"
        assert catalogue[0].is_potentially_hazardous is False


# ---------------------------------------------------------------------------
# Unit tests: prediction persistence
# ---------------------------------------------------------------------------

class TestPredictionPersistence:
    def test_prediction_stored_with_track(self) -> None:
        store = _in_memory_store()
        track = _make_track()
        prediction = _make_prediction()
        threat = _make_threat()
        store.upsert_track(track, prediction, threat)

        with store._Session() as session:
            from src.data_store import PredictionORM
            rows = session.query(PredictionORM).filter_by(track_id="TRK-001").all()
            assert len(rows) == 1
            assert rows[0].model_used == "orbital"
            assert rows[0].intersects_earth_corridor is True


# ---------------------------------------------------------------------------
# Property-based test: Property 11 — Data store round-trip
# **Validates: Requirements 9.1**
# ---------------------------------------------------------------------------

# Strategies
_track_id_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-"),
    min_size=1,
    max_size=20,
)

_timestamp_st = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.none(),  # naive datetimes for SQLite
)

_position_sample_st = st.builds(
    PositionSample,
    frame_id=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
    timestamp=_timestamp_st,
    centroid_x=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    centroid_y=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    bbox=st.tuples(
        st.integers(min_value=0, max_value=500),
        st.integers(min_value=0, max_value=500),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
    ),
    velocity_px=st.tuples(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ),
    velocity_kms=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    angular_diameter_arcsec=st.none(),
    distance_au=st.none(),
    approx_flag=st.booleans(),
)


def _unique_samples(samples: list[PositionSample]) -> list[PositionSample]:
    """Deduplicate by frame_id to avoid DB constraint issues."""
    seen: set[str] = set()
    result = []
    for s in samples:
        if s.frame_id not in seen:
            seen.add(s.frame_id)
            result.append(s)
    return result


@given(
    track_id=_track_id_st,
    status=st.sampled_from(["active", "lost"]),
    samples=st.lists(_position_sample_st, min_size=0, max_size=10),
)
@settings(max_examples=100)
def test_property_11_data_store_round_trip(
    track_id: str,
    status: str,
    samples: list[PositionSample],
) -> None:
    """
    Feature: asteroid-threat-monitor, Property 11: Data store round-trip

    For any TrackRecord written to the data store, reading it back by track_id
    must return a record with identical track_id, status, and the same number
    of position samples.

    **Validates: Requirements 9.1**
    """
    unique_samples = _unique_samples(samples)
    # Use a recent naive datetime so it falls within the 90-day retention window
    now = datetime.utcnow()

    track = TrackRecord(
        track_id=track_id,
        status=status,  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
        frames_since_last_detection=0,
        history=unique_samples,
    )
    threat = ThreatRecord(
        track_id=track_id,
        threat_level="Safe",
        previous_threat_level=None,
        closest_approach_au=0.1,
        velocity_kms=None,
        changed_at=None,
        evaluated_at=now,
    )

    store = _in_memory_store()
    store.upsert_track(track, None, threat)

    # Read back via get_track_history
    history = store.get_track_history(track_id)
    assert len(history) == len(unique_samples)

    # Read back via get_active_tracks (only for active tracks within retention)
    if status == "active":
        active = store.get_active_tracks()
        matching = [t for t in active if t.track_id == track_id]
        assert len(matching) == 1
        assert matching[0].track_id == track_id
        assert matching[0].status == status
        assert len(matching[0].history) == len(unique_samples)
