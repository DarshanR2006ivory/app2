"""Main pipeline entry point and CLI."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime, timezone

from src.alert_manager import AlertManager
from src.api_client import ApiClient
from src.config import load_config
from src.data_store import DataStore
from src.detector import Detector
from src.image_processor import ImageProcessor
from src.models import ThreatRecord
from src.threat_classifier import ThreatClassifier
from src.tracker import Tracker
from src.trajectory_predictor import TrajectoryPredictor

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asteroid Threat Monitor")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--module",
        choices=[
            "image_processor", "detector", "tracker",
            "trajectory_predictor", "threat_classifier",
            "alert_manager", "data_store", "api_client", "dashboard",
        ],
        help="Run a single module independently for testing",
    )
    return parser


# ---------------------------------------------------------------------------
# Independent module runners (for --module flag)
# ---------------------------------------------------------------------------

def _run_image_processor(config) -> None:
    """Run image_processor independently: print each decoded frame."""
    processor = ImageProcessor(config.image_processor)
    try:
        frame_count = 0
        while True:
            frame = processor.next_frame()
            if frame is None:
                break
            frame_count += 1
            logger.info(
                "Frame %d: id=%s timestamp=%s shape=%s source=%s",
                frame_count,
                frame.frame_id,
                frame.timestamp.isoformat(),
                frame.rgb_array.shape,
                frame.source,
            )
        logger.info("image_processor: processed %d frames", frame_count)
    finally:
        processor.close()


def _run_detector(config) -> None:
    """Run detector independently: decode one frame and run detection."""
    processor = ImageProcessor(config.image_processor)
    detector = Detector(config.detector)
    try:
        frame = processor.next_frame()
        if frame is None:
            logger.info("detector: no frames available from source")
            return
        detections = detector.detect(frame)
        logger.info(
            "detector: frame=%s detections=%d",
            frame.frame_id,
            len(detections),
        )
        for i, det in enumerate(detections):
            logger.info(
                "  [%d] bbox=%s confidence=%.4f",
                i,
                det.bbox,
                det.confidence,
            )
    finally:
        processor.close()


def _run_tracker(config) -> None:
    """Run tracker independently: process all frames and print track summaries."""
    processor = ImageProcessor(config.image_processor)
    detector = Detector(config.detector)
    tracker = Tracker(config.tracker)
    try:
        frame_count = 0
        while True:
            frame = processor.next_frame()
            if frame is None:
                break
            frame_count += 1
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            logger.info(
                "tracker: frame=%s detections=%d tracks=%d",
                frame.frame_id,
                len(detections),
                len(tracks),
            )
        logger.info("tracker: processed %d frames", frame_count)
    finally:
        processor.close()


def _run_trajectory_predictor(config) -> None:
    """Run trajectory_predictor independently: process frames and print predictions."""
    processor = ImageProcessor(config.image_processor)
    detector = Detector(config.detector)
    tracker = Tracker(config.tracker)
    predictor = TrajectoryPredictor(config.trajectory_predictor)
    try:
        frame_count = 0
        while True:
            frame = processor.next_frame()
            if frame is None:
                break
            frame_count += 1
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            for track in tracks:
                prediction = predictor.predict(track)
                if prediction is not None:
                    logger.info(
                        "trajectory_predictor: track=%s closest_approach_au=%.6f intersects=%s",
                        track.track_id,
                        prediction.closest_approach_au,
                        prediction.intersects_earth_corridor,
                    )
        logger.info("trajectory_predictor: processed %d frames", frame_count)
    finally:
        processor.close()


def _run_threat_classifier(config) -> None:
    """Run threat_classifier independently: process frames and print threat levels."""
    processor = ImageProcessor(config.image_processor)
    detector = Detector(config.detector)
    tracker = Tracker(config.tracker)
    predictor = TrajectoryPredictor(config.trajectory_predictor)
    classifier = ThreatClassifier(config.threat_classifier)
    try:
        frame_count = 0
        while True:
            frame = processor.next_frame()
            if frame is None:
                break
            frame_count += 1
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            for track in tracks:
                prediction = predictor.predict(track)
                if prediction is not None:
                    threat = classifier.classify(track, prediction)
                    logger.info(
                        "threat_classifier: track=%s level=%s closest_au=%.6f",
                        threat.track_id,
                        threat.threat_level,
                        threat.closest_approach_au,
                    )
        logger.info("threat_classifier: processed %d frames", frame_count)
    finally:
        processor.close()


def _run_alert_manager(config) -> None:
    """Run alert_manager independently: process frames and print generated alerts."""
    processor = ImageProcessor(config.image_processor)
    detector = Detector(config.detector)
    tracker = Tracker(config.tracker)
    predictor = TrajectoryPredictor(config.trajectory_predictor)
    classifier = ThreatClassifier(config.threat_classifier)
    alert_mgr = AlertManager(config.alert_manager)
    try:
        frame_count = 0
        while True:
            frame = processor.next_frame()
            if frame is None:
                break
            frame_count += 1
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            for track in tracks:
                prediction = predictor.predict(track)
                if prediction is not None:
                    threat = classifier.classify(track, prediction)
                    alert_mgr.evaluate(threat)
        alerts = alert_mgr.get_alerts()
        logger.info(
            "alert_manager: processed %d frames, generated %d alerts",
            frame_count,
            len(alerts),
        )
        for alert in alerts:
            logger.info(
                "  alert_id=%s track=%s level=%s channels=%s",
                alert.alert_id,
                alert.track_id,
                alert.threat_level,
                alert.channels,
            )
    finally:
        processor.close()
        alert_mgr.shutdown(wait=False)


def _run_data_store(config) -> None:
    """Run data_store independently: show active tracks from the store."""
    store = DataStore(config.data_store)
    tracks = store.get_active_tracks()
    logger.info("data_store: %d active tracks", len(tracks))
    for track in tracks:
        logger.info(
            "  track_id=%s status=%s samples=%d",
            track.track_id,
            track.status,
            len(track.history),
        )


def _run_api_client(config) -> None:
    """Run api_client independently: perform one fetch cycle."""
    store = DataStore(config.data_store)
    client = ApiClient(config.api_client, store)
    logger.info("api_client: running one fetch cycle")
    client._fetch_and_process()
    neos = store.get_neo_catalogue()
    logger.info("api_client: %d NEO records in catalogue", len(neos))


def _run_dashboard(config) -> None:
    """Run dashboard independently via streamlit."""
    import subprocess

    logger.info("dashboard: launching Streamlit on port %d", config.dashboard.port)
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", "src/dashboard.py",
            "--server.port", str(config.dashboard.port),
        ],
        check=False,
    )


_MODULE_RUNNERS = {
    "image_processor": _run_image_processor,
    "detector": _run_detector,
    "tracker": _run_tracker,
    "trajectory_predictor": _run_trajectory_predictor,
    "threat_classifier": _run_threat_classifier,
    "alert_manager": _run_alert_manager,
    "data_store": _run_data_store,
    "api_client": _run_api_client,
    "dashboard": _run_dashboard,
}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config) -> None:
    """Run the full asteroid threat monitor pipeline until stream is exhausted."""
    # Initialise all modules
    image_processor = ImageProcessor(config.image_processor)
    detector = Detector(config.detector)
    tracker = Tracker(config.tracker)
    predictor = TrajectoryPredictor(config.trajectory_predictor)
    classifier = ThreatClassifier(config.threat_classifier)
    alert_mgr = AlertManager(config.alert_manager)
    store = DataStore(config.data_store)

    # Start api_client background thread if enabled
    api_thread = None
    if config.api_client.enabled:
        api_client = ApiClient(config.api_client, store)
        api_thread = api_client.start_background()
        logger.info("api_client background thread started")

    frame_count = 0
    try:
        while True:
            # Fetch next frame; None signals stream exhausted
            try:
                frame = image_processor.next_frame()
            except Exception:
                logger.error(
                    "Unhandled exception in image_processor.next_frame():\n%s",
                    traceback.format_exc(),
                )
                continue

            if frame is None:
                logger.info("Stream exhausted after %d frames — exiting cleanly", frame_count)
                break

            frame_count += 1

            # Run each pipeline stage; catch and log any unhandled exception,
            # then continue to the next frame (Requirement 12.3)
            try:
                detections = detector.detect(frame)
            except Exception:
                logger.error(
                    "Unhandled exception in detector.detect() for frame %s:\n%s",
                    frame.frame_id,
                    traceback.format_exc(),
                )
                continue

            try:
                tracks = tracker.update(detections, frame)
            except Exception:
                logger.error(
                    "Unhandled exception in tracker.update() for frame %s:\n%s",
                    frame.frame_id,
                    traceback.format_exc(),
                )
                continue

            for track in tracks:
                try:
                    prediction = predictor.predict(track)
                    if prediction is None:
                        # Not enough samples yet; persist track with no prediction
                        # We still need a minimal threat record for data_store
                        minimal_threat = ThreatRecord(
                            track_id=track.track_id,
                            threat_level="Safe",
                            previous_threat_level=None,
                            closest_approach_au=float("inf"),
                            velocity_kms=None,
                            changed_at=None,
                            evaluated_at=datetime.now(tz=timezone.utc),
                        )
                        store.upsert_track(track, None, minimal_threat)
                        continue

                    threat = classifier.classify(track, prediction)
                    alert_mgr.evaluate(threat)
                    store.upsert_track(track, prediction, threat)

                except Exception:
                    logger.error(
                        "Unhandled exception processing track %s in frame %s:\n%s",
                        track.track_id,
                        frame.frame_id,
                        traceback.format_exc(),
                    )
                    # Continue to next track / next frame

    finally:
        image_processor.close()
        alert_mgr.shutdown(wait=True)
        logger.info("Pipeline shut down cleanly")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    # load_config exits with code 1 on missing/invalid config (Requirement 11.1, 11.2)
    config = load_config(args.config)

    if args.module:
        runner = _MODULE_RUNNERS[args.module]
        runner(config)
        return

    run_pipeline(config)


if __name__ == "__main__":
    main()
