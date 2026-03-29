"""Unit tests for src/main.py — pipeline entry point and CLI."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.main import build_parser, run_pipeline, _MODULE_RUNNERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(frame_id: str = "f001"):
    from src.models import DecodedFrame
    return DecodedFrame(
        frame_id=frame_id,
        timestamp=datetime.now(tz=timezone.utc),
        rgb_array=np.zeros((100, 100, 3), dtype=np.uint8),
        source="test",
    )


def _make_config():
    from src.config import AppConfig
    return AppConfig()


# ---------------------------------------------------------------------------
# CLI / argparse tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.config == "config.yaml"
        assert args.module is None

    def test_config_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--config", "my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_module_flag_valid(self):
        parser = build_parser()
        for module in [
            "image_processor", "detector", "tracker",
            "trajectory_predictor", "threat_classifier",
            "alert_manager", "data_store", "api_client", "dashboard",
        ]:
            args = parser.parse_args(["--module", module])
            assert args.module == module

    def test_module_flag_invalid(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--module", "nonexistent_module"])


# ---------------------------------------------------------------------------
# main() — config loading
# ---------------------------------------------------------------------------

class TestMainConfigLoading:
    def test_missing_config_exits_1(self, tmp_path):
        """main() must exit with code 1 when config file is missing."""
        from src.main import main
        with patch("sys.argv", ["main", "--config", str(tmp_path / "missing.yaml")]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_malformed_config_exits_1(self, tmp_path):
        """main() must exit with code 1 when config YAML is malformed."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text(": invalid: yaml: [")
        from src.main import main
        with patch("sys.argv", ["main", "--config", str(bad_config)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# run_pipeline — stream exhaustion
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_exits_cleanly_when_stream_exhausted(self, tmp_path):
        """Pipeline exits cleanly (no exception) when next_frame() returns None."""
        config = _make_config()
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector") as MockDet,
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor") as MockTP,
            patch("src.main.ThreatClassifier") as MockTC,
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore") as MockDS,
        ):
            ip = MockIP.return_value
            ip.next_frame.return_value = None  # stream immediately exhausted

            am = MockAM.return_value
            am.shutdown = MagicMock()

            # Should complete without raising
            run_pipeline(config)

            ip.close.assert_called_once()
            am.shutdown.assert_called_once_with(wait=True)

    def test_pipeline_processes_frames(self, tmp_path):
        """Pipeline calls each module in order for each frame."""
        config = _make_config()
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        frame = _make_frame()

        from src.models import TrackRecord, PredictionRecord, ThreatRecord, ForecastStep

        track = TrackRecord(
            track_id="TRK-001",
            status="active",
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
            frames_since_last_detection=0,
            history=[],
        )
        prediction = PredictionRecord(
            track_id="TRK-001",
            computed_at=datetime.now(tz=timezone.utc),
            horizon_hours=72.0,
            forecast_steps=[
                ForecastStep(time_offset_hours=1.0, position_au=(1.0, 0.0, 0.0), confidence_interval_au=0.001)
            ],
            closest_approach_au=0.1,
            closest_approach_time=datetime.now(tz=timezone.utc),
            intersects_earth_corridor=False,
            model_used="orbital",
        )
        threat = ThreatRecord(
            track_id="TRK-001",
            threat_level="Safe",
            previous_threat_level=None,
            closest_approach_au=0.1,
            velocity_kms=None,
            changed_at=None,
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector") as MockDet,
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor") as MockTP,
            patch("src.main.ThreatClassifier") as MockTC,
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore") as MockDS,
        ):
            ip = MockIP.return_value
            # Return one frame then None
            ip.next_frame.side_effect = [frame, None]

            det = MockDet.return_value
            det.detect.return_value = []

            trk = MockTrk.return_value
            trk.update.return_value = [track]

            tp = MockTP.return_value
            tp.predict.return_value = prediction

            tc = MockTC.return_value
            tc.classify.return_value = threat

            am = MockAM.return_value
            am.shutdown = MagicMock()

            ds = MockDS.return_value
            ds.upsert_track = MagicMock()

            run_pipeline(config)

            det.detect.assert_called_once_with(frame)
            trk.update.assert_called_once_with([], frame)
            tp.predict.assert_called_once_with(track)
            tc.classify.assert_called_once_with(track, prediction)
            am.evaluate.assert_called_once_with(threat)
            ds.upsert_track.assert_called_once_with(track, prediction, threat)

    def test_pipeline_continues_on_detector_exception(self, tmp_path):
        """Unhandled detector exception is logged and pipeline continues to next frame."""
        config = _make_config()
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        frame1 = _make_frame("f001")
        frame2 = _make_frame("f002")

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector") as MockDet,
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor"),
            patch("src.main.ThreatClassifier"),
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore"),
        ):
            ip = MockIP.return_value
            ip.next_frame.side_effect = [frame1, frame2, None]

            det = MockDet.return_value
            # First frame raises, second returns empty list
            det.detect.side_effect = [RuntimeError("boom"), []]

            trk = MockTrk.return_value
            trk.update.return_value = []

            MockAM.return_value.shutdown = MagicMock()

            # Should not raise
            run_pipeline(config)

            # detect called twice (once per frame)
            assert det.detect.call_count == 2

    def test_pipeline_continues_on_tracker_exception(self, tmp_path):
        """Unhandled tracker exception is logged and pipeline continues to next frame."""
        config = _make_config()
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        frame = _make_frame()

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector") as MockDet,
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor"),
            patch("src.main.ThreatClassifier"),
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore"),
        ):
            ip = MockIP.return_value
            ip.next_frame.side_effect = [frame, None]

            det = MockDet.return_value
            det.detect.return_value = []

            trk = MockTrk.return_value
            trk.update.side_effect = RuntimeError("tracker exploded")

            MockAM.return_value.shutdown = MagicMock()

            # Should not raise
            run_pipeline(config)

    def test_pipeline_continues_on_per_track_exception(self, tmp_path):
        """Unhandled per-track exception is logged and pipeline continues."""
        config = _make_config()
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        frame = _make_frame()

        from src.models import TrackRecord

        track = TrackRecord(
            track_id="TRK-001",
            status="active",
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
            frames_since_last_detection=0,
            history=[],
        )

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector") as MockDet,
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor") as MockTP,
            patch("src.main.ThreatClassifier"),
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore"),
        ):
            ip = MockIP.return_value
            ip.next_frame.side_effect = [frame, None]

            det = MockDet.return_value
            det.detect.return_value = []

            trk = MockTrk.return_value
            trk.update.return_value = [track]

            tp = MockTP.return_value
            tp.predict.side_effect = RuntimeError("predictor exploded")

            MockAM.return_value.shutdown = MagicMock()

            # Should not raise
            run_pipeline(config)

    def test_api_client_started_when_enabled(self, tmp_path):
        """api_client background thread is started when api_client.enabled is True."""
        config = _make_config()
        config.api_client.enabled = True
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector"),
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor"),
            patch("src.main.ThreatClassifier"),
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore"),
            patch("src.main.ApiClient") as MockApiClient,
        ):
            MockIP.return_value.next_frame.return_value = None
            MockTrk.return_value.update.return_value = []
            MockAM.return_value.shutdown = MagicMock()

            mock_api = MockApiClient.return_value
            mock_api.start_background = MagicMock(return_value=MagicMock())

            run_pipeline(config)

            mock_api.start_background.assert_called_once()

    def test_api_client_not_started_when_disabled(self, tmp_path):
        """api_client background thread is NOT started when api_client.enabled is False."""
        config = _make_config()
        config.api_client.enabled = False
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector"),
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor"),
            patch("src.main.ThreatClassifier"),
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore"),
            patch("src.main.ApiClient") as MockApiClient,
        ):
            MockIP.return_value.next_frame.return_value = None
            MockTrk.return_value.update.return_value = []
            MockAM.return_value.shutdown = MagicMock()

            run_pipeline(config)

            MockApiClient.assert_not_called()

    def test_no_prediction_stores_minimal_threat(self, tmp_path):
        """When predictor returns None, a minimal Safe ThreatRecord is stored."""
        config = _make_config()
        config.data_store.sqlite_path = str(tmp_path / "test.db")

        frame = _make_frame()

        from src.models import TrackRecord

        track = TrackRecord(
            track_id="TRK-001",
            status="active",
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
            frames_since_last_detection=0,
            history=[],
        )

        with (
            patch("src.main.ImageProcessor") as MockIP,
            patch("src.main.Detector") as MockDet,
            patch("src.main.Tracker") as MockTrk,
            patch("src.main.TrajectoryPredictor") as MockTP,
            patch("src.main.ThreatClassifier") as MockTC,
            patch("src.main.AlertManager") as MockAM,
            patch("src.main.DataStore") as MockDS,
        ):
            MockIP.return_value.next_frame.side_effect = [frame, None]
            MockDet.return_value.detect.return_value = []
            MockTrk.return_value.update.return_value = [track]
            MockTP.return_value.predict.return_value = None  # no prediction yet
            MockAM.return_value.shutdown = MagicMock()

            ds = MockDS.return_value
            ds.upsert_track = MagicMock()

            run_pipeline(config)

            # upsert_track called with prediction=None and a Safe threat
            assert ds.upsert_track.call_count == 1
            call_args = ds.upsert_track.call_args
            stored_track, stored_prediction, stored_threat = call_args[0]
            assert stored_prediction is None
            assert stored_threat.threat_level == "Safe"
            # alert_manager.evaluate should NOT be called (no real prediction)
            MockAM.return_value.evaluate.assert_not_called()


# ---------------------------------------------------------------------------
# Module runners registered
# ---------------------------------------------------------------------------

class TestModuleRunners:
    def test_all_modules_have_runners(self):
        expected = {
            "image_processor", "detector", "tracker",
            "trajectory_predictor", "threat_classifier",
            "alert_manager", "data_store", "api_client", "dashboard",
        }
        assert set(_MODULE_RUNNERS.keys()) == expected

    def test_all_runners_are_callable(self):
        for name, runner in _MODULE_RUNNERS.items():
            assert callable(runner), f"Runner for '{name}' is not callable"
