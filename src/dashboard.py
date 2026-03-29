"""Dashboard: Streamlit app that reads from data_store and renders the UI.

Entry point: streamlit run src/dashboard.py [-- --config path/to/config.yaml]

Config resolution order:
  1. --config CLI argument (passed after --)
  2. DASHBOARD_CONFIG environment variable
  3. Default: config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work when
# Streamlit launches this file directly (e.g. `streamlit run src/dashboard.py`)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config / DataStore bootstrap
# ---------------------------------------------------------------------------

def _resolve_config_path() -> str:
    """Resolve config path from CLI args, env var, or default."""
    # Streamlit passes script args after '--'
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    if args.config:
        return args.config
    env = os.environ.get("DASHBOARD_CONFIG")
    if env:
        return env
    return "config.yaml"


@st.cache_resource
def _load_app_config(config_path: str):
    """Load and cache AppConfig (runs once per Streamlit session)."""
    from src.config import load_config
    return load_config(config_path)


@st.cache_resource
def _get_data_store(config_path: str):
    """Create and cache DataStore (runs once per Streamlit session)."""
    from src.data_store import DataStore
    cfg = _load_app_config(config_path)
    return DataStore(cfg.data_store)


# ---------------------------------------------------------------------------
# Auto-pipeline: runs the full pipeline in a background thread
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_pipeline_heartbeat() -> dict:
    """Shared mutable dict updated by the background pipeline thread."""
    return {"last_frame_time": datetime.now(tz=timezone.utc)}


@st.cache_resource
def _start_auto_pipeline(config_path: str):
    """Start the pipeline in a background daemon thread (runs once per session)."""
    import threading, json, glob, cv2
    from src.config import load_config
    from src.simulator import Simulator
    from src.tracker import Tracker
    from src.trajectory_predictor import TrajectoryPredictor
    from src.threat_classifier import ThreatClassifier
    from src.alert_manager import AlertManager
    from src.data_store import DataStore
    from src.models import ThreatRecord, DecodedFrame, Detection
    from datetime import datetime, timezone
    import numpy as np

    cfg = load_config(config_path)

    sim_dir = "./data/auto_pipeline_frames"
    sim = Simulator(width=640, height=480, n_asteroids=3, format="png", seed=42)
    sim.generate_frames(sim_dir, n_frames=120)

    def _run():
        rng = np.random.default_rng()
        store = DataStore(cfg.data_store)
        tracker = Tracker(cfg.tracker)
        predictor = TrajectoryPredictor(cfg.trajectory_predictor)
        classifier = ThreatClassifier(cfg.threat_classifier)
        alert_mgr = AlertManager(cfg.alert_manager)
        heartbeat = _get_pipeline_heartbeat()

        files = sorted(glob.glob(f"{sim_dir}/*.png"))
        meta_path = Path(f"{sim_dir}/metadata.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"frames": []}
        frame_idx = 0

        if not files:
            return

        _shapes = ["spherical", "elongated", "irregular"]

        # Per-track persistent state so values drift smoothly
        track_state: dict[str, dict] = {}

        def _init_track_state(track_id: str) -> dict:
            """Assign initial random physical parameters for a new track."""
            diam = float(np.exp(rng.uniform(np.log(0.01), np.log(10.0))))
            density = rng.uniform(1500, 3000)
            radius_m = (diam * 1000) / 2
            mass = float((4 / 3) * np.pi * radius_m ** 3 * density)
            return {
                "velocity_kms": float(rng.uniform(5.0, 40.0)),
                "distance_au": float(rng.uniform(0.001, 0.12)),
                "angular_diameter_arcsec": float(rng.uniform(0.05, 2.5)),
                "diameter_km": round(diam, 4),
                "estimated_mass_kg": mass,
                "shape": _shapes[int(rng.integers(0, 3))],
                "rotation_period_hours": float(rng.uniform(2.0, 48.0)),
                # Drift direction: +1 or -1 per parameter
                "_v_dir": 1.0,
                "_d_dir": -1.0,
                # Countdown to next threat escalation event
                "_escalate_in": int(rng.integers(20, 60)),
                "_escalating": False,
            }

        while True:
            path = files[frame_idx % len(files)]
            fidx = frame_idx % max(len(meta["frames"]), 1)
            frame_idx += 1

            try:
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is None:
                    time.sleep(0.1)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                now = datetime.now(tz=timezone.utc)
                frame = DecodedFrame(
                    frame_id=f"auto_{frame_idx:06d}",
                    timestamp=now,
                    rgb_array=rgb,
                    source=sim_dir,
                )
                heartbeat["last_frame_time"] = now

                # Build detections from ground-truth metadata
                detections = []
                ast_meta = meta["frames"][fidx]["asteroids"] if meta["frames"] else []
                for ast in ast_meta:
                    x1, y1, x2, y2 = ast["bbox"]
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=0.95,
                        frame_id=frame.frame_id,
                    ))

                tracks = tracker.update(detections, frame)

                # Drift physical parameters smoothly per track
                for track in tracks:
                    if not track.history:
                        continue

                    tid = track.track_id
                    if tid not in track_state:
                        track_state[tid] = _init_track_state(tid)
                    st = track_state[tid]

                    # Countdown to escalation event
                    st["_escalate_in"] -= 1
                    if st["_escalate_in"] <= 0 and not st["_escalating"]:
                        st["_escalating"] = True
                        st["_escalate_in"] = int(rng.integers(40, 80))

                    if st["_escalating"]:
                        # Drive distance down and velocity up to trigger threat
                        st["distance_au"] = max(0.0001, st["distance_au"] - 0.003)
                        st["velocity_kms"] = min(50.0, st["velocity_kms"] + 0.8)
                        if st["distance_au"] < 0.0005:
                            # Reset after passing through danger zone
                            st["_escalating"] = False
                            st["distance_au"] = float(rng.uniform(0.06, 0.15))
                            st["velocity_kms"] = float(rng.uniform(5.0, 15.0))
                            st["_escalate_in"] = int(rng.integers(30, 70))
                    else:
                        # Normal drift with small noise
                        st["velocity_kms"] = float(np.clip(
                            st["velocity_kms"] + rng.normal(0, 0.4) * st["_v_dir"],
                            0.5, 45.0,
                        ))
                        st["distance_au"] = float(np.clip(
                            st["distance_au"] + rng.normal(0, 0.001) * st["_d_dir"],
                            0.001, 0.5,
                        ))
                        # Occasionally flip drift direction
                        if rng.random() < 0.05:
                            st["_v_dir"] *= -1
                        if rng.random() < 0.05:
                            st["_d_dir"] *= -1

                    st["angular_diameter_arcsec"] = float(np.clip(
                        st["angular_diameter_arcsec"] + rng.normal(0, 0.02), 0.01, 5.0
                    ))
                    st["rotation_period_hours"] = float(np.clip(
                        st["rotation_period_hours"] + rng.normal(0, 0.1), 0.5, 100.0
                    ))

                    # Write into the latest position sample
                    s = track.history[-1]
                    s.velocity_kms = st["velocity_kms"]
                    s.distance_au = st["distance_au"]
                    s.angular_diameter_arcsec = st["angular_diameter_arcsec"]
                    s.diameter_km = st["diameter_km"]
                    s.estimated_mass_kg = st["estimated_mass_kg"]
                    s.shape = st["shape"]
                    s.rotation_period_hours = st["rotation_period_hours"]
                    s.approx_flag = False

                for track in tracks:
                    try:
                        prediction = predictor.predict(track)
                        if prediction is None:
                            # Use current distance_au from track state for threat classification
                            tid = track.track_id
                            dist = track_state.get(tid, {}).get("distance_au", 1.0)
                            vel = track_state.get(tid, {}).get("velocity_kms", 0.0)
                            # Manually classify based on injected values
                            if dist < cfg.threat_classifier.dangerous_distance_au and vel > cfg.threat_classifier.dangerous_velocity_kms:
                                level = "Dangerous"
                            elif dist <= cfg.threat_classifier.hazardous_distance_au:
                                level = "Potentially_Hazardous"
                            else:
                                level = "Safe"
                            threat = ThreatRecord(
                                track_id=track.track_id,
                                threat_level=level,
                                previous_threat_level=None,
                                closest_approach_au=dist,
                                velocity_kms=vel,
                                changed_at=None,
                                evaluated_at=datetime.now(tz=timezone.utc),
                            )
                            alert_mgr.evaluate(threat)
                            store.upsert_track(track, None, threat)
                            continue

                        # Override prediction's closest_approach_au with injected value
                        tid = track.track_id
                        if tid in track_state:
                            object.__setattr__(prediction, "closest_approach_au",
                                               track_state[tid]["distance_au"])
                            object.__setattr__(prediction, "intersects_earth_corridor",
                                               track_state[tid]["distance_au"] <= cfg.trajectory_predictor.earth_corridor_au)

                        threat = classifier.classify(track, prediction)
                        alert_mgr.evaluate(threat)
                        store.upsert_track(track, prediction, threat)
                    except Exception:
                        pass

            except Exception:
                pass

            time.sleep(0.5)

    t = threading.Thread(target=_run, daemon=True, name="auto-pipeline")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_LEVEL_EMOJI = {
    "Safe": "🟢",
    "Potentially_Hazardous": "🟡",
    "Dangerous": "🔴",
}

_LEVEL_COLOUR = {
    "Safe": "#2ecc71",
    "Potentially_Hazardous": "#f39c12",
    "Dangerous": "#e74c3c",
}


def _colour_threat_level(val: str) -> str:
    colour = _LEVEL_COLOUR.get(val, "#ffffff")
    return f"background-color: {colour}; color: white; font-weight: bold;"


# ---------------------------------------------------------------------------
# OpenCV bounding-box overlay
# ---------------------------------------------------------------------------

def _draw_bboxes(
    rgb: np.ndarray,
    tracks,
    selected_track_id: str | None,
) -> np.ndarray:
    """Draw bounding boxes on a copy of the RGB frame and return it."""
    img = rgb.copy()
    for track in tracks:
        if not track.history:
            continue
        sample = track.history[-1]
        x1, y1, x2, y2 = sample.bbox
        level = _get_latest_threat_level(track)
        colour_hex = _LEVEL_COLOUR.get(level, "#ffffff")
        r, g, b = int(colour_hex[1:3], 16), int(colour_hex[3:5], 16), int(colour_hex[5:7], 16)
        bgr = (b, g, r)
        thickness = 3 if track.track_id == selected_track_id else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
        label = f"{track.track_id} {_LEVEL_EMOJI.get(level, '')}"
        cv2.putText(img, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
    return img


def _get_latest_threat_level(track) -> str:
    """Return the most recent threat level stored in session_state for a track."""
    return st.session_state.get(f"threat_{track.track_id}", "Safe")


# ---------------------------------------------------------------------------
# Risk summary table
# ---------------------------------------------------------------------------

def _build_risk_df(tracks, threat_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for t in tracks:
        last = t.history[-1] if t.history else None
        rows.append({
            "Track ID": t.track_id,
            "Threat Level": threat_map.get(t.track_id, "Safe"),
            "Velocity (km/s)": round(last.velocity_kms, 3) if last and last.velocity_kms else None,
            "Distance (AU)": round(last.distance_au, 6) if last and last.distance_au else None,
            "Updated": t.updated_at.strftime("%H:%M:%S") if t.updated_at else "",
        })
    return pd.DataFrame(rows)


def _build_live_metrics_df(tracks) -> pd.DataFrame:
    """Build a continuously-updating table of physical parameters per track."""
    rows = []
    for t in tracks:
        last = t.history[-1] if t.history else None
        if last is None:
            continue

        # Format mass in scientific notation
        if last.estimated_mass_kg is not None:
            exp = int(np.floor(np.log10(abs(last.estimated_mass_kg)))) if last.estimated_mass_kg > 0 else 0
            mantissa = last.estimated_mass_kg / (10 ** exp)
            mass_str = f"{mantissa:.2f}×10^{exp} kg"
        else:
            mass_str = "—"

        rows.append({
            "Track ID": t.track_id,
            "Velocity (km/s)": round(last.velocity_kms, 2) if last.velocity_kms is not None else None,
            "Distance (AU)": round(last.distance_au, 6) if last.distance_au is not None else None,
            "Diameter (km)": round(last.diameter_km, 4) if last.diameter_km is not None else None,
            "Est. Mass": mass_str,
            "Shape": last.shape or "—",
            "Rotation (hrs)": round(last.rotation_period_hours, 2) if last.rotation_period_hours is not None else None,
            "Angular Size ('')": round(last.angular_diameter_arcsec, 3) if last.angular_diameter_arcsec is not None else None,
            "Updated": t.updated_at.strftime("%H:%M:%S") if t.updated_at else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Selected track charts
# ---------------------------------------------------------------------------

def _speed_chart(history) -> go.Figure:
    times = [s.timestamp for s in history if s.velocity_kms is not None]
    speeds = [s.velocity_kms for s in history if s.velocity_kms is not None]
    fig = px.line(x=times, y=speeds, labels={"x": "Time", "y": "Speed (km/s)"},
                  title="Speed over Time")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


def _distance_chart(history) -> go.Figure:
    times = [s.timestamp for s in history if s.distance_au is not None]
    dists = [s.distance_au for s in history if s.distance_au is not None]
    fig = px.line(x=times, y=dists, labels={"x": "Time", "y": "Distance (AU)"},
                  title="Distance over Time")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


def _trajectory_3d(data_store, track_id: str) -> go.Figure:
    """Build a 3D trajectory chart from the latest prediction forecast steps."""
    # Try to get forecast steps from the predictions table via a raw query
    try:
        import json
        from src.data_store import PredictionORM
        with data_store._Session() as session:
            pred = (
                session.query(PredictionORM)
                .filter(PredictionORM.track_id == track_id)
                .order_by(PredictionORM.computed_at.desc())
                .first()
            )
            if pred:
                steps = json.loads(pred.forecast_json)
                xs = [s["position_au"][0] for s in steps]
                ys = [s["position_au"][1] for s in steps]
                zs = [s["position_au"][2] for s in steps]
                fig = go.Figure(data=[go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines+markers",
                    marker=dict(size=3),
                    line=dict(width=2),
                    name=track_id,
                )])
                fig.update_layout(
                    title=f"3D Trajectory — {track_id}",
                    scene=dict(
                        xaxis_title="X (AU)",
                        yaxis_title="Y (AU)",
                        zaxis_title="Z (AU)",
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                return fig
    except Exception:
        pass

    # Fallback: plot pixel positions from history
    history = data_store.get_track_history(track_id)
    xs = [s.centroid_x for s in history]
    ys = [s.centroid_y for s in history]
    zs = list(range(len(history)))
    fig = go.Figure(data=[go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines+markers",
        marker=dict(size=3),
        line=dict(width=2),
        name=track_id,
    )])
    fig.update_layout(
        title=f"3D Trajectory (pixel space) — {track_id}",
        scene=dict(xaxis_title="X (px)", yaxis_title="Y (px)", zaxis_title="Frame"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Historical log helpers
# ---------------------------------------------------------------------------

def _build_historical_df(data_store, date_from: datetime, date_to: datetime,
                          threat_filter: list[str]) -> pd.DataFrame:
    """Build a flat DataFrame of all threat history rows within the date range."""
    try:
        from src.data_store import ThreatHistoryORM
        with data_store._Session() as session:
            q = session.query(ThreatHistoryORM).filter(
                ThreatHistoryORM.evaluated_at >= date_from,
                ThreatHistoryORM.evaluated_at <= date_to,
            )
            if threat_filter:
                q = q.filter(ThreatHistoryORM.threat_level.in_(threat_filter))
            rows = q.order_by(ThreatHistoryORM.evaluated_at.desc()).all()
        data = [
            {
                "Track ID": r.track_id,
                "Threat Level": r.threat_level,
                "Previous Level": r.previous_level or "—",
                "Closest Approach (AU)": round(r.closest_approach_au, 6),
                "Velocity (km/s)": round(r.velocity_kms, 3) if r.velocity_kms else None,
                "Changed At": r.changed_at.strftime("%Y-%m-%d %H:%M:%S") if r.changed_at else "—",
                "Evaluated At": r.evaluated_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for r in rows
        ]
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def _build_alert_log_df(data_store, since: datetime) -> pd.DataFrame:
    alerts = data_store.get_alerts(since=since)
    if not alerts:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Alert ID": a.alert_id[:8] + "…",
            "Track ID": a.track_id,
            "Threat Level": a.threat_level,
            "Closest Approach (AU)": round(a.closest_approach_au, 6),
            "Velocity (km/s)": round(a.velocity_kms, 3) if a.velocity_kms else None,
            "Created At": a.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Channels": ", ".join(a.channels),
        }
        for a in reversed(alerts)
    ])


# ---------------------------------------------------------------------------
# Frame loading from disk + auto-simulation fallback
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".fits", ".fit"}
_SIM_FRAMES_DIR = "./data/sim_frames"
_SIM_N_FRAMES = 60


def _ensure_sim_frames() -> None:
    """Auto-generate simulator frames if the sim directory is empty or missing."""
    p = Path(_SIM_FRAMES_DIR)
    if p.exists() and any(f.suffix.lower() in _IMAGE_EXTENSIONS for f in p.iterdir()):
        return
    from src.simulator import Simulator
    sim = Simulator(width=640, height=480, n_asteroids=3, format="png", seed=42)
    sim.generate_frames(_SIM_FRAMES_DIR, n_frames=_SIM_N_FRAMES)


def _get_sim_frame_cycled() -> np.ndarray:
    """Return a simulator frame, cycling through them on each call."""
    p = Path(_SIM_FRAMES_DIR)
    files = sorted(
        f for f in p.iterdir() if f.suffix.lower() == ".png"
    )
    if not files:
        return _blank_frame()
    idx = st.session_state.get("_sim_frame_idx", 0) % len(files)
    st.session_state["_sim_frame_idx"] = idx + 1
    bgr = cv2.imread(str(files[idx]), cv2.IMREAD_COLOR)
    if bgr is None:
        return _blank_frame()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _load_frame_from_disk(source_path: str) -> tuple[np.ndarray | None, datetime | None]:
    """Return (rgb_array, mtime) for the most recently modified image in source_path."""
    p = Path(source_path)
    if not p.exists() or not p.is_dir():
        return None, None
    files = sorted(
        (f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS),
        key=lambda f: f.stat().st_mtime,
    )
    if not files:
        return None, None
    latest = files[-1]
    mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    try:
        if latest.suffix.lower() in (".fits", ".fit"):
            from astropy.io import fits as astrofits
            with astrofits.open(str(latest)) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data = hdu.data
                        break
            if data is None:
                return None, mtime
            data = np.array(data, dtype=float)
            if data.ndim == 3 and data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            if data.ndim == 2:
                mn, mx = data.min(), data.max()
                data = ((data - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
                data = np.stack([data, data, data], axis=-1)
            else:
                mn, mx = data.min(), data.max()
                data = ((data - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
            return data, mtime
        else:
            bgr = cv2.imread(str(latest), cv2.IMREAD_COLOR)
            if bgr is None:
                return None, mtime
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mtime
    except Exception:
        return None, mtime


def _get_display_frame(source_path: str) -> tuple[np.ndarray, datetime]:
    """Return the best available frame and its timestamp.

    Priority:
      1. Latest real image from source_path (external pipeline)
      2. Latest frame from auto-pipeline frames directory
      3. Cycled synthetic simulator frame (static fallback)
    """
    # 1. Real pipeline source
    disk_frame, disk_mtime = _load_frame_from_disk(source_path)
    if disk_frame is not None:
        return disk_frame, disk_mtime

    # 2. Auto-pipeline frames (written by background thread)
    auto_frame, auto_mtime = _load_frame_from_disk("./data/auto_pipeline_frames")
    if auto_frame is not None:
        return auto_frame, auto_mtime

    # 3. Static sim fallback
    _ensure_sim_frames()
    return _get_sim_frame_cycled(), datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Main dashboard entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Asteroid Threat Monitor",
        page_icon="☄️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config_path = _resolve_config_path()
    app_cfg = _load_app_config(config_path)
    data_store = _get_data_store(config_path)
    refresh_interval = app_cfg.dashboard.refresh_interval_seconds

    # Start the auto-pipeline background thread (no-op after first call)
    _start_auto_pipeline(config_path)
    heartbeat = _get_pipeline_heartbeat()

    # ------------------------------------------------------------------
    # Session state initialisation
    # ------------------------------------------------------------------
    if "selected_track_id" not in st.session_state:
        st.session_state.selected_track_id = None
    if "threat_filter" not in st.session_state:
        st.session_state.threat_filter = ["Safe", "Potentially_Hazardous", "Dangerous"]
    if "date_from" not in st.session_state:
        st.session_state.date_from = (datetime.now() - timedelta(days=7)).date()
    if "date_to" not in st.session_state:
        st.session_state.date_to = datetime.now().date()
    if "last_frame_time" not in st.session_state:
        st.session_state.last_frame_time = datetime.now(tz=timezone.utc)
    if "connection_lost" not in st.session_state:
        st.session_state.connection_lost = False

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.title("☄️ Controls")

        st.subheader("Filters")
        date_from = st.date_input(
            "Date from",
            value=st.session_state.date_from,
            key="sidebar_date_from",
        )
        date_to = st.date_input(
            "Date to",
            value=st.session_state.date_to,
            key="sidebar_date_to",
        )
        st.session_state.date_from = date_from
        st.session_state.date_to = date_to

        threat_filter = st.multiselect(
            "Threat level",
            options=["Safe", "Potentially_Hazardous", "Dangerous"],
            default=st.session_state.threat_filter,
            key="sidebar_threat_filter",
        )
        st.session_state.threat_filter = threat_filter

        # Fetch active tracks for track selector
        active_tracks = data_store.get_active_tracks()
        track_ids = [t.track_id for t in active_tracks]

        selected = st.selectbox(
            "Select Track",
            options=["(none)"] + track_ids,
            index=(
                (["(none)"] + track_ids).index(st.session_state.selected_track_id)
                if st.session_state.selected_track_id in track_ids
                else 0
            ),
            key="sidebar_track_selector",
        )
        st.session_state.selected_track_id = selected if selected != "(none)" else None

        st.subheader("Settings")
        st.caption(f"Refresh interval: {refresh_interval}s")
        st.caption(f"Config: {config_path}")

    # ------------------------------------------------------------------
    # Page header
    # ------------------------------------------------------------------
    header_col, status_col = st.columns([4, 1])
    with header_col:
        st.title("☄️ Asteroid Threat Monitor")

    # Connection status — use pipeline heartbeat, not file mtime
    now_utc = datetime.now(tz=timezone.utc)
    last_frame_time = heartbeat["last_frame_time"]
    seconds_since_frame = (now_utc - last_frame_time).total_seconds()
    connection_lost = seconds_since_frame > 10  # allow 10s for pipeline warmup

    with status_col:
        if connection_lost:
            st.error("🔴 Feed Lost")
        else:
            st.success("🟢 Connected")

    # ------------------------------------------------------------------
    # Alert Panel
    # ------------------------------------------------------------------
    alert_since = now_utc - timedelta(seconds=30)
    recent_alerts = data_store.get_alerts(since=alert_since)

    if recent_alerts:
        dangerous = [a for a in recent_alerts if a.threat_level == "Dangerous"]
        hazardous = [a for a in recent_alerts if a.threat_level == "Potentially_Hazardous"]

        if dangerous:
            st.error("🚨 IMPACT THREAT DETECTED", icon="🚨")
            for a in dangerous:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Track", a.track_id)
                c2.metric("Closest Approach", f"{a.closest_approach_au:.6f} AU")
                c3.metric("Velocity", f"{a.velocity_kms:.2f} km/s" if a.velocity_kms else "—")
                c4.metric("Time", a.created_at.strftime("%H:%M:%S"))

        if hazardous:
            st.warning("⚠️ POTENTIALLY HAZARDOUS OBJECTS", icon="⚠️")
            for a in hazardous:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Track", a.track_id)
                c2.metric("Closest Approach", f"{a.closest_approach_au:.6f} AU")
                c3.metric("Velocity", f"{a.velocity_kms:.2f} km/s" if a.velocity_kms else "—")
                c4.metric("Time", a.created_at.strftime("%H:%M:%S"))
    else:
        st.success("✅ No active threats in the last 30 seconds", icon="✅")

    # Connection-lost banner
    if connection_lost:
        st.error(
            f"⚠️ **CONNECTION LOST** — Data feed interrupted. "
            f"Last frame: {heartbeat['last_frame_time'].strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"({int(seconds_since_frame)}s ago)"
        )

    # ------------------------------------------------------------------
    # Build threat map from latest threat history
    # ------------------------------------------------------------------
    threat_map: dict[str, str] = {}
    try:
        from src.data_store import ThreatHistoryORM
        with data_store._Session() as session:
            for track in active_tracks:
                row = (
                    session.query(ThreatHistoryORM)
                    .filter(ThreatHistoryORM.track_id == track.track_id)
                    .order_by(ThreatHistoryORM.evaluated_at.desc())
                    .first()
                )
                if row:
                    threat_map[track.track_id] = row.threat_level
                    st.session_state[f"threat_{track.track_id}"] = row.threat_level
    except Exception:
        pass

    # Apply threat level filter to active tracks
    filtered_tracks = [
        t for t in active_tracks
        if threat_map.get(t.track_id, "Safe") in (threat_filter or ["Safe", "Potentially_Hazardous", "Dangerous"])
    ]

    # ------------------------------------------------------------------
    # Row 1: Live Frame View + Risk Summary Table
    # ------------------------------------------------------------------
    frame_col, table_col = st.columns([3, 2])

    with frame_col:
        st.subheader("Live Frame View")
        frame_placeholder = st.empty()

        source_path = app_cfg.image_processor.source_path
        display_frame, frame_time = _get_display_frame(source_path)
        st.session_state.last_frame_time = frame_time
        display_frame = _draw_bboxes(display_frame, filtered_tracks, st.session_state.selected_track_id)
        frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

    with table_col:
        st.subheader("Risk Summary")
        risk_df = _build_risk_df(filtered_tracks, threat_map)
        if not risk_df.empty:
            # Colour-code the Threat Level column
            styled = risk_df.style.applymap(
                _colour_threat_level, subset=["Threat Level"]
            )
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Track ID": st.column_config.TextColumn("Track ID"),
                    "Threat Level": st.column_config.TextColumn("Threat Level"),
                    "Velocity (km/s)": st.column_config.NumberColumn("Velocity (km/s)", format="%.3f"),
                    "Distance (AU)": st.column_config.NumberColumn("Distance (AU)", format="%.6f"),
                    "Updated": st.column_config.TextColumn("Updated"),
                },
            )
        else:
            st.info("No active tracks matching current filters.")

    # ------------------------------------------------------------------
    # Live Asteroid Metrics Table
    # ------------------------------------------------------------------
    st.subheader("🔭 Live Asteroid Metrics")
    metrics_df = _build_live_metrics_df(active_tracks)
    if not metrics_df.empty:
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Track ID": st.column_config.TextColumn("Track ID", width="small"),
                "Velocity (km/s)": st.column_config.NumberColumn("Velocity (km/s)", format="%.2f"),
                "Distance (AU)": st.column_config.NumberColumn("Distance (AU)", format="%.6f"),
                "Diameter (km)": st.column_config.NumberColumn("Diameter (km)", format="%.4f"),
                "Est. Mass": st.column_config.TextColumn("Est. Mass"),
                "Shape": st.column_config.TextColumn("Shape"),
                "Rotation (hrs)": st.column_config.NumberColumn("Rotation (hrs)", format="%.2f"),
                "Angular Size ('')": st.column_config.NumberColumn("Angular Size (\")", format="%.3f"),
                "Updated": st.column_config.TextColumn("Updated", width="small"),
            },
        )
    else:
        st.info("Waiting for track data — pipeline is warming up...")

    # ------------------------------------------------------------------
    # Row 2: Selected Track Panel
    # ------------------------------------------------------------------
    selected_track_id = st.session_state.selected_track_id
    selected_track = next((t for t in active_tracks if t.track_id == selected_track_id), None)

    st.subheader("Selected Track Panel")
    if selected_track and selected_track.history:
        last_sample = selected_track.history[-1]
        threat_level = threat_map.get(selected_track_id, "Safe")

        # KPI metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                "Velocity",
                f"{last_sample.velocity_kms:.3f} km/s" if last_sample.velocity_kms else "N/A",
            )
        with kpi2:
            st.metric(
                "Distance",
                f"{last_sample.distance_au:.6f} AU" if last_sample.distance_au else "N/A",
            )
        with kpi3:
            vx, vy = last_sample.velocity_px
            size_px = (
                (last_sample.bbox[2] - last_sample.bbox[0] + last_sample.bbox[3] - last_sample.bbox[1]) / 2
            )
            st.metric("Size (px)", f"{size_px:.1f}")
        with kpi4:
            angle = np.degrees(np.arctan2(vy, vx)) if (vx or vy) else 0.0
            st.metric("Direction", f"{angle:.1f}°")

        # Threat level badge
        emoji = _LEVEL_EMOJI.get(threat_level, "")
        st.markdown(
            f"**Threat Level:** {emoji} `{threat_level}`"
        )

        # Charts
        chart1, chart2, chart3 = st.columns(3)
        history = data_store.get_track_history(selected_track_id)

        with chart1:
            if any(s.velocity_kms for s in history):
                st.plotly_chart(_speed_chart(history), use_container_width=True)
            else:
                st.caption("No velocity data available.")

        with chart2:
            if any(s.distance_au for s in history):
                st.plotly_chart(_distance_chart(history), use_container_width=True)
            else:
                st.caption("No distance data available.")

        with chart3:
            st.plotly_chart(_trajectory_3d(data_store, selected_track_id), use_container_width=True)

    else:
        st.info("Select a track from the sidebar to view its details.")

    # ------------------------------------------------------------------
    # Row 3: Alert Log + Historical Data Log
    # ------------------------------------------------------------------
    log_col1, log_col2 = st.columns(2)

    with log_col1:
        st.subheader("Alert Log")
        alert_log_since = datetime.combine(date_from, datetime.min.time())
        alert_df = _build_alert_log_df(data_store, since=alert_log_since)
        if not alert_df.empty:
            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Threat Level": st.column_config.TextColumn("Threat Level"),
                    "Closest Approach (AU)": st.column_config.NumberColumn(
                        "Closest Approach (AU)", format="%.6f"
                    ),
                    "Velocity (km/s)": st.column_config.NumberColumn(
                        "Velocity (km/s)", format="%.3f"
                    ),
                },
            )
        else:
            st.info("No alerts in the selected date range.")

    with log_col2:
        st.subheader("Historical Data Log")
        date_from_dt = datetime.combine(date_from, datetime.min.time())
        date_to_dt = datetime.combine(date_to, datetime.max.time())
        hist_df = _build_historical_df(
            data_store, date_from_dt, date_to_dt,
            threat_filter or ["Safe", "Potentially_Hazardous", "Dangerous"],
        )
        if not hist_df.empty:
            st.dataframe(
                hist_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Threat Level": st.column_config.TextColumn("Threat Level"),
                    "Closest Approach (AU)": st.column_config.NumberColumn(
                        "Closest Approach (AU)", format="%.6f"
                    ),
                    "Velocity (km/s)": st.column_config.NumberColumn(
                        "Velocity (km/s)", format="%.3f"
                    ),
                },
            )
        else:
            st.info("No historical records in the selected date range / filters.")

    # ------------------------------------------------------------------
    # Live refresh loop — update frame view continuously
    # ------------------------------------------------------------------
    live_container = st.empty()
    while True:
        time.sleep(refresh_interval)

        # Re-fetch active tracks and latest frame
        try:
            active_tracks = data_store.get_active_tracks()
            threat_map = {}
            try:
                from src.data_store import ThreatHistoryORM
                with data_store._Session() as session:
                    for track in active_tracks:
                        row = (
                            session.query(ThreatHistoryORM)
                            .filter(ThreatHistoryORM.track_id == track.track_id)
                            .order_by(ThreatHistoryORM.evaluated_at.desc())
                            .first()
                        )
                        if row:
                            threat_map[track.track_id] = row.threat_level
            except Exception:
                pass

            filtered_tracks = [
                t for t in active_tracks
                if threat_map.get(t.track_id, "Safe") in (
                    st.session_state.threat_filter or ["Safe", "Potentially_Hazardous", "Dangerous"]
                )
            ]

            display_frame, frame_time = _get_display_frame(
                app_cfg.image_processor.source_path
            )
            st.session_state.last_frame_time = frame_time
            now_utc = datetime.now(tz=timezone.utc)
            display_frame = _draw_bboxes(display_frame, filtered_tracks, st.session_state.selected_track_id)

            with live_container:
                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

            # Check connection-lost using heartbeat
            last_frame_time = heartbeat["last_frame_time"]
            seconds_since = (now_utc - last_frame_time).total_seconds()
            if seconds_since > 10 and not st.session_state.connection_lost:
                st.session_state.connection_lost = True
                st.rerun()
            elif seconds_since <= 5 and st.session_state.connection_lost:
                st.session_state.connection_lost = False
                st.rerun()

        except Exception:
            pass


if __name__ == "__main__":
    main()
