"""Data store: persists all records and provides query interface for dashboard."""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import DataStoreConfig
from src.models import (
    AlertRecord,
    ForecastStep,
    NeoRecord,
    PositionSample,
    PredictionRecord,
    ThreatRecord,
    TrackRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM Base and Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class TrackORM(Base):
    __tablename__ = "tracks"

    track_id = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    neo_catalogue_id = Column(String, nullable=True)
    neo_name = Column(String, nullable=True)


class PositionSampleORM(Base):
    __tablename__ = "position_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String, nullable=False)
    frame_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    centroid_x = Column(Float, nullable=False)
    centroid_y = Column(Float, nullable=False)
    bbox_x1 = Column(Integer, nullable=True)
    bbox_y1 = Column(Integer, nullable=True)
    bbox_x2 = Column(Integer, nullable=True)
    bbox_y2 = Column(Integer, nullable=True)
    velocity_px_x = Column(Float, nullable=True)
    velocity_px_y = Column(Float, nullable=True)
    velocity_kms = Column(Float, nullable=True)
    angular_diameter_arcsec = Column(Float, nullable=True)
    distance_au = Column(Float, nullable=True)
    approx_flag = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("idx_position_samples_track_id", "track_id"),
        Index("idx_position_samples_timestamp", "timestamp"),
    )


class PredictionORM(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String, nullable=False)
    computed_at = Column(DateTime, nullable=False)
    horizon_hours = Column(Float, nullable=False)
    closest_approach_au = Column(Float, nullable=False)
    closest_approach_time = Column(DateTime, nullable=False)
    intersects_earth_corridor = Column(Boolean, nullable=False)
    model_used = Column(String, nullable=False)
    forecast_json = Column(Text, nullable=False)


class ThreatHistoryORM(Base):
    __tablename__ = "threat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String, nullable=False)
    threat_level = Column(String, nullable=False)
    previous_level = Column(String, nullable=True)
    closest_approach_au = Column(Float, nullable=False)
    velocity_kms = Column(Float, nullable=True)
    changed_at = Column(DateTime, nullable=True)
    evaluated_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_threat_history_track_id", "track_id"),
    )


class AlertORM(Base):
    __tablename__ = "alerts"

    alert_id = Column(String, primary_key=True)
    track_id = Column(String, nullable=False)
    threat_level = Column(String, nullable=False)
    closest_approach_au = Column(Float, nullable=False)
    velocity_kms = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False)
    channels = Column(Text, nullable=False)       # JSON array
    delivery_status = Column(Text, nullable=False)  # JSON object

    __table_args__ = (
        Index("idx_alerts_created_at", "created_at"),
    )


class NeoCatalogueORM(Base):
    __tablename__ = "neo_catalogue"

    neo_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    absolute_magnitude = Column(Float, nullable=True)
    diameter_min_km = Column(Float, nullable=True)
    diameter_max_km = Column(Float, nullable=True)
    is_potentially_hazardous = Column(Boolean, nullable=False)
    close_approach_date = Column(String, nullable=True)  # stored as ISO date string
    miss_distance_au = Column(Float, nullable=True)
    relative_velocity_kms = Column(Float, nullable=True)
    fetched_at = Column(DateTime, nullable=False)


# ---------------------------------------------------------------------------
# Retry queue item
# ---------------------------------------------------------------------------

class _RetryItem:
    def __init__(self, record_type: str, record_id: str, data: Any) -> None:
        self.record_type = record_type
        self.record_id = record_id
        self.data = data


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------

class DataStore:
    def __init__(self, config: DataStoreConfig) -> None:
        self.config = config
        self._retry_queue: queue.Queue[_RetryItem] = queue.Queue()
        self._lock = threading.Lock()

        if config.backend == "postgresql":
            if not config.postgresql_dsn:
                raise ValueError("postgresql_dsn must be set when backend is 'postgresql'")
            url = config.postgresql_dsn
        else:
            # SQLite — support :memory: for tests
            if config.sqlite_path == ":memory:":
                url = "sqlite:///:memory:"
            else:
                url = f"sqlite:///{config.sqlite_path}"

        connect_args: dict = {}
        if config.backend == "sqlite":
            connect_args["check_same_thread"] = False

        self._engine = create_engine(url, connect_args=connect_args)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def upsert_track(
        self,
        track: TrackRecord,
        prediction: PredictionRecord | None,
        threat: ThreatRecord,
    ) -> None:
        """Persist track, its latest position samples, optional prediction, and threat entry."""
        try:
            with self._Session() as session:
                self._upsert_track_orm(session, track)
                self._insert_new_position_samples(session, track)
                if prediction is not None:
                    self._insert_prediction(session, prediction)
                self._insert_threat_history(session, threat)
                session.commit()
        except Exception as exc:
            logger.error(
                "DataStore write failure for track %s: %s", track.track_id, exc
            )
            self._retry_queue.put(
                _RetryItem("track", track.track_id, (track, prediction, threat))
            )

    def insert_alert(self, alert: AlertRecord) -> None:
        """Persist an alert record."""
        try:
            with self._Session() as session:
                existing = session.get(AlertORM, alert.alert_id)
                if existing is None:
                    orm = AlertORM(
                        alert_id=alert.alert_id,
                        track_id=alert.track_id,
                        threat_level=alert.threat_level,
                        closest_approach_au=alert.closest_approach_au,
                        velocity_kms=alert.velocity_kms,
                        created_at=_strip_tz(alert.created_at),
                        channels=json.dumps(alert.channels),
                        delivery_status=json.dumps(alert.delivery_status),
                    )
                    session.add(orm)
                session.commit()
        except Exception as exc:
            logger.error(
                "DataStore write failure for alert %s: %s", alert.alert_id, exc
            )
            self._retry_queue.put(_RetryItem("alert", alert.alert_id, alert))

    def upsert_neo(self, neo: NeoRecord) -> None:
        """Persist or update a NEO catalogue entry."""
        try:
            with self._Session() as session:
                existing = session.get(NeoCatalogueORM, neo.neo_id)
                if existing is None:
                    orm = _neo_to_orm(neo)
                    session.add(orm)
                else:
                    existing.name = neo.name
                    existing.absolute_magnitude = neo.absolute_magnitude
                    existing.diameter_min_km = neo.estimated_diameter_km[0]
                    existing.diameter_max_km = neo.estimated_diameter_km[1]
                    existing.is_potentially_hazardous = neo.is_potentially_hazardous
                    existing.close_approach_date = neo.close_approach_date.isoformat()
                    existing.miss_distance_au = neo.miss_distance_au
                    existing.relative_velocity_kms = neo.relative_velocity_kms
                    existing.fetched_at = _strip_tz(neo.fetched_at)
                session.commit()
        except Exception as exc:
            logger.error(
                "DataStore write failure for NEO %s: %s", neo.neo_id, exc
            )
            self._retry_queue.put(_RetryItem("neo", neo.neo_id, neo))

    # ------------------------------------------------------------------
    # Public read methods
    # ------------------------------------------------------------------

    def get_active_tracks(self) -> list[TrackRecord]:
        """Return all tracks with status='active' within the retention window."""
        cutoff = _utcnow() - timedelta(days=self.config.retention_days)
        with self._Session() as session:
            rows = (
                session.query(TrackORM)
                .filter(
                    TrackORM.status == "active",
                    TrackORM.created_at >= _strip_tz(cutoff),
                )
                .all()
            )
            return [_orm_to_track(row, session) for row in rows]

    def get_track_history(self, track_id: str, limit: int = 100) -> list[PositionSample]:
        """Return the most recent `limit` position samples for a track."""
        with self._Session() as session:
            rows = (
                session.query(PositionSampleORM)
                .filter(PositionSampleORM.track_id == track_id)
                .order_by(PositionSampleORM.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [_orm_to_position_sample(r) for r in reversed(rows)]

    def get_alerts(self, since: datetime) -> list[AlertRecord]:
        """Return all alerts created at or after `since`."""
        with self._Session() as session:
            rows = (
                session.query(AlertORM)
                .filter(AlertORM.created_at >= _strip_tz(since))
                .order_by(AlertORM.created_at.asc())
                .all()
            )
            return [_orm_to_alert(r) for r in rows]

    def get_neo_catalogue(self) -> list[NeoRecord]:
        """Return all NEO catalogue entries."""
        with self._Session() as session:
            rows = session.query(NeoCatalogueORM).all()
            return [_orm_to_neo(r) for r in rows]

    # ------------------------------------------------------------------
    # Retry queue access (for testing / background workers)
    # ------------------------------------------------------------------

    @property
    def retry_queue(self) -> queue.Queue[_RetryItem]:
        return self._retry_queue

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsert_track_orm(self, session: Session, track: TrackRecord) -> None:
        existing = session.get(TrackORM, track.track_id)
        closed_at: datetime | None = None
        if track.status == "lost":
            # Preserve existing closed_at if already set, otherwise set now
            if existing and existing.closed_at:
                closed_at = existing.closed_at
            else:
                closed_at = _strip_tz(_utcnow())

        if existing is None:
            orm = TrackORM(
                track_id=track.track_id,
                status=track.status,
                created_at=_strip_tz(track.created_at),
                updated_at=_strip_tz(track.updated_at),
                closed_at=closed_at,
                neo_catalogue_id=track.neo_catalogue_id,
                neo_name=track.neo_name,
            )
            session.add(orm)
        else:
            existing.status = track.status
            existing.updated_at = _strip_tz(track.updated_at)
            existing.closed_at = closed_at
            existing.neo_catalogue_id = track.neo_catalogue_id
            existing.neo_name = track.neo_name

    def _insert_new_position_samples(self, session: Session, track: TrackRecord) -> None:
        """Insert position samples that are not yet in the DB (identified by frame_id + track_id)."""
        existing_frame_ids: set[str] = set()
        rows = (
            session.query(PositionSampleORM.frame_id)
            .filter(PositionSampleORM.track_id == track.track_id)
            .all()
        )
        existing_frame_ids = {r.frame_id for r in rows}

        for sample in track.history:
            if sample.frame_id not in existing_frame_ids:
                orm = PositionSampleORM(
                    track_id=track.track_id,
                    frame_id=sample.frame_id,
                    timestamp=_strip_tz(sample.timestamp),
                    centroid_x=sample.centroid_x,
                    centroid_y=sample.centroid_y,
                    bbox_x1=sample.bbox[0],
                    bbox_y1=sample.bbox[1],
                    bbox_x2=sample.bbox[2],
                    bbox_y2=sample.bbox[3],
                    velocity_px_x=sample.velocity_px[0],
                    velocity_px_y=sample.velocity_px[1],
                    velocity_kms=sample.velocity_kms,
                    angular_diameter_arcsec=sample.angular_diameter_arcsec,
                    distance_au=sample.distance_au,
                    approx_flag=sample.approx_flag,
                )
                session.add(orm)

    def _insert_prediction(self, session: Session, prediction: PredictionRecord) -> None:
        forecast_data = [
            {
                "time_offset_hours": s.time_offset_hours,
                "position_au": list(s.position_au),
                "confidence_interval_au": s.confidence_interval_au,
            }
            for s in prediction.forecast_steps
        ]
        orm = PredictionORM(
            track_id=prediction.track_id,
            computed_at=_strip_tz(prediction.computed_at),
            horizon_hours=prediction.horizon_hours,
            closest_approach_au=prediction.closest_approach_au,
            closest_approach_time=_strip_tz(prediction.closest_approach_time),
            intersects_earth_corridor=prediction.intersects_earth_corridor,
            model_used=prediction.model_used,
            forecast_json=json.dumps(forecast_data),
        )
        session.add(orm)

    def _insert_threat_history(self, session: Session, threat: ThreatRecord) -> None:
        orm = ThreatHistoryORM(
            track_id=threat.track_id,
            threat_level=threat.threat_level,
            previous_level=threat.previous_threat_level,
            closest_approach_au=threat.closest_approach_au,
            velocity_kms=threat.velocity_kms,
            changed_at=_strip_tz(threat.changed_at) if threat.changed_at else None,
            evaluated_at=_strip_tz(threat.evaluated_at),
        )
        session.add(orm)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _strip_tz(dt: datetime) -> datetime:
    """Remove timezone info for SQLite compatibility (store as naive UTC)."""
    if dt is None:
        return dt
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _orm_to_track(row: TrackORM, session: Session) -> TrackRecord:
    samples = (
        session.query(PositionSampleORM)
        .filter(PositionSampleORM.track_id == row.track_id)
        .order_by(PositionSampleORM.timestamp.asc())
        .all()
    )
    history = [_orm_to_position_sample(s) for s in samples]
    return TrackRecord(
        track_id=row.track_id,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
        updated_at=row.updated_at,
        frames_since_last_detection=0,
        history=history,
        neo_catalogue_id=row.neo_catalogue_id,
        neo_name=row.neo_name,
    )


def _orm_to_position_sample(row: PositionSampleORM) -> PositionSample:
    return PositionSample(
        frame_id=row.frame_id,
        timestamp=row.timestamp,
        centroid_x=row.centroid_x,
        centroid_y=row.centroid_y,
        bbox=(row.bbox_x1 or 0, row.bbox_y1 or 0, row.bbox_x2 or 0, row.bbox_y2 or 0),
        velocity_px=(row.velocity_px_x or 0.0, row.velocity_px_y or 0.0),
        velocity_kms=row.velocity_kms,
        angular_diameter_arcsec=row.angular_diameter_arcsec,
        distance_au=row.distance_au,
        approx_flag=bool(row.approx_flag),
    )


def _orm_to_alert(row: AlertORM) -> AlertRecord:
    return AlertRecord(
        alert_id=row.alert_id,
        track_id=row.track_id,
        threat_level=row.threat_level,
        closest_approach_au=row.closest_approach_au,
        velocity_kms=row.velocity_kms,
        created_at=row.created_at,
        channels=json.loads(row.channels),
        delivery_status=json.loads(row.delivery_status),
    )


def _orm_to_neo(row: NeoCatalogueORM) -> NeoRecord:
    from datetime import date as date_type
    close_approach = (
        date_type.fromisoformat(row.close_approach_date)
        if row.close_approach_date
        else date_type.today()
    )
    return NeoRecord(
        neo_id=row.neo_id,
        name=row.name,
        absolute_magnitude=row.absolute_magnitude or 0.0,
        estimated_diameter_km=(row.diameter_min_km or 0.0, row.diameter_max_km or 0.0),
        is_potentially_hazardous=bool(row.is_potentially_hazardous),
        close_approach_date=close_approach,
        miss_distance_au=row.miss_distance_au or 0.0,
        relative_velocity_kms=row.relative_velocity_kms or 0.0,
        fetched_at=row.fetched_at,
    )


def _neo_to_orm(neo: NeoRecord) -> NeoCatalogueORM:
    return NeoCatalogueORM(
        neo_id=neo.neo_id,
        name=neo.name,
        absolute_magnitude=neo.absolute_magnitude,
        diameter_min_km=neo.estimated_diameter_km[0],
        diameter_max_km=neo.estimated_diameter_km[1],
        is_potentially_hazardous=neo.is_potentially_hazardous,
        close_approach_date=neo.close_approach_date.isoformat(),
        miss_distance_au=neo.miss_distance_au,
        relative_velocity_kms=neo.relative_velocity_kms,
        fetched_at=_strip_tz(neo.fetched_at),
    )
