# Implementation Plan: Asteroid Threat Monitor

## Overview

Incremental Python implementation of the full pipeline: config → data models → image_processor → detector → tracker → trajectory_predictor → threat_classifier → alert_manager → data_store → api_client → dashboard → main pipeline → simulator. Each task wires into the previous, ending with a fully integrated system.

## Tasks

- [x] 1. Project structure, data models, and config loader
  - Create `src/` package layout: `config.py`, `models.py`, `image_processor.py`, `detector.py`, `tracker.py`, `trajectory_predictor.py`, `threat_classifier.py`, `alert_manager.py`, `data_store.py`, `api_client.py`, `dashboard.py`, `main.py`
  - Create `tests/` directory with `conftest.py`
  - Define all dataclasses in `src/models.py`: `DecodedFrame`, `Detection`, `DetectionList`, `PositionSample`, `TrackRecord`, `PredictionRecord`, `ForecastStep`, `ThreatRecord`, `AlertRecord`, `NeoRecord`
  - Implement `src/config.py`: load YAML via PyYAML, validate all fields against documented ranges/types, apply defaults, exit with code 1 on missing file or invalid value
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2_

  - [ ]* 1.1 Write property test for configuration defaults (Property 12)
    - **Property 12: Configuration defaults applied**
    - **Validates: Requirements 11.4**

  - [ ]* 1.2 Write property test for configuration validation rejection (Property 13)
    - **Property 13: Configuration validation rejects invalid values**
    - **Validates: Requirements 11.3**

  - [x]* 1.3 Write unit tests for config loader
    - Test missing file exits with code 1
    - Test malformed YAML exits with code 1
    - Test all documented defaults are applied when fields are omitted

- [x] 2. Implement `image_processor`
  - Implement `ImageProcessor` class in `src/image_processor.py`
  - Support `source_type`: `directory` (sorted by name/timestamp), `video`, `rtsp`/`http` (OpenCV VideoCapture), `fits` (astropy.io.fits → numpy)
  - `next_frame()` returns `DecodedFrame | None`; on decode failure log error + frame ID and skip
  - Enforce 50 ms decode timeout; process batch frames in chronological order
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.1 Write unit tests for image_processor
    - Test each supported format decodes to correct RGB array shape
    - Test corrupt file logs error and skips without halting
    - Test batch mode processes frames in chronological order

- [x] 3. Implement `detector`
  - Implement `Detector` class in `src/detector.py`
  - Support model loading: YOLOv8/v5 via `ultralytics`, ONNX via `onnxruntime`, PyTorch via `torch.load`
  - `detect(frame)` returns `DetectionList` filtered by `confidence_threshold`; return empty list on no detections
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.1 Write property test for detection confidence filtering (Property 1)
    - **Property 1: Detection confidence filtering**
    - **Validates: Requirements 2.2**

  - [ ]* 3.2 Write unit tests for detector
    - Test empty DetectionList returned on blank frame
    - Test confidence threshold filtering at boundary values (0.0, 0.5, 1.0)

- [x] 4. Implement `tracker`
  - Implement `Tracker` class in `src/tracker.py`
  - Kalman Filter state `[cx, cy, vx, vy, w, h]` using `filterpy.kalman.KalmanFilter` with transition/measurement matrices from design
  - Hungarian algorithm via `scipy.optimize.linear_sum_assignment` on IoU cost matrix; reject pairs below `min_iou_threshold`
  - Assign unique persistent Track IDs; mark tracks lost after `max_lost_frames`; maintain rolling history up to `max_history`
  - Compute 2D velocity vector (px/frame → km/s), angular diameter (arcsec), distance (AU); set `approx_flag=True` when `fov_scale_arcsec_per_pixel` is null
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 4.1 Write property test for Track ID uniqueness and persistence (Property 2)
    - **Property 2: Track ID uniqueness and persistence**
    - **Validates: Requirements 3.2**

  - [ ]* 4.2 Write property test for track loss invariant (Property 3)
    - **Property 3: Track loss invariant**
    - **Validates: Requirements 3.3**

  - [ ]* 4.3 Write property test for physical parameter approximation flag (Property 4)
    - **Property 4: Physical parameter approximation flag**
    - **Validates: Requirements 4.5**

  - [ ]* 4.4 Write unit tests for tracker
    - Test new track creation on first detection
    - Test track loss after exactly `max_lost_frames` frames with no match
    - Test Hungarian assignment correctness with overlapping bounding boxes
    - Test two nearby detections get distinct Track IDs

- [x] 5. Checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement `trajectory_predictor`
  - Implement `TrajectoryPredictor` class in `src/trajectory_predictor.py`
  - Return `None` (no exception) when track has fewer than 20 samples
  - Convert pixel positions to angular coordinates using plate scale; fit Keplerian orbit via `scipy.optimize.minimize` or poliastro; propagate with Kepler's equation
  - Compute `closest_approach_au` and `closest_approach_time` using astropy solar system ephemeris
  - Set `intersects_earth_corridor = closest_approach_au <= earth_corridor_au`
  - Derive confidence intervals from orbit-fit residuals
  - If `lstm_model_path` configured: load PyTorch LSTM, compute delta corrections, blend with `lstm_blend_weight`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 6.1 Write property test for trajectory prediction deferral (Property 5)
    - **Property 5: Trajectory prediction deferral**
    - **Validates: Requirements 5.5**

  - [ ]* 6.2 Write property test for Earth corridor intersection consistency (Property 6)
    - **Property 6: Earth corridor intersection consistency**
    - **Validates: Requirements 5.2**

  - [ ]* 6.3 Write unit tests for trajectory_predictor
    - Test None returned for track with 0, 1, and 19 samples
    - Test Earth corridor flag set correctly at boundary (exactly `earth_corridor_au`)

- [x] 7. Implement `threat_classifier`
  - Implement `ThreatClassifier` class in `src/threat_classifier.py`
  - Apply classification rules from design using configurable thresholds (`dangerous_distance_au`, `dangerous_velocity_kms`, `hazardous_distance_au`)
  - Record `previous_threat_level`, `changed_at` (non-null on change, None on no change) in `ThreatRecord`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 7.1 Write property test for threat classification rule correctness (Property 7)
    - **Property 7: Threat classification rule correctness**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

  - [ ]* 7.2 Write property test for threat level change recording (Property 8)
    - **Property 8: Threat level change recording**
    - **Validates: Requirements 6.5**

  - [ ]* 7.3 Write unit tests for threat_classifier
    - Test boundary conditions at exactly 0.002 AU and 0.05 AU
    - Test `changed_at` is None when level unchanged, non-null when changed

- [x] 8. Implement `alert_manager`
  - Implement `AlertManager` class in `src/alert_manager.py`
  - Generate `AlertRecord` (UUID alert_id) for Potentially_Hazardous and Dangerous threats
  - Map threat levels to severity: Potentially_Hazardous → "medium", Dangerous → "high"
  - In-memory deduplication dict `{(track_id, threat_level): last_alert_utc}`; suppress within `deduplication_window_seconds`
  - Dispatch email via `smtplib` (SMTP with TLS) when `email.enabled`; dispatch SMS via Twilio when `sms.enabled` (Dangerous only)
  - Retry failed deliveries up to `retry_attempts` times with `retry_interval_seconds` delay using a background `ThreadPoolExecutor`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ]* 8.1 Write property test for alert deduplication (Property 9)
    - **Property 9: Alert deduplication**
    - **Validates: Requirements 7.7**

  - [ ]* 8.2 Write property test for alert severity mapping (Property 10)
    - **Property 10: Alert severity mapping**
    - **Validates: Requirements 7.3**

  - [ ]* 8.3 Write unit tests for alert_manager
    - Test deduplication window boundary (just inside vs. just outside window)
    - Test retry logic: mock delivery failure, assert retried up to 3 times
    - Test SMS only sent for Dangerous, not Potentially_Hazardous

- [x] 9. Implement `data_store`
  - Implement `DataStore` class in `src/data_store.py` using SQLAlchemy
  - Define ORM models matching the SQL schema from design (tracks, position_samples, predictions, threat_history, alerts, neo_catalogue)
  - Support SQLite (default) and PostgreSQL backends via `data_store.backend` config
  - Implement `upsert_track`, `get_active_tracks`, `get_track_history`, `get_alerts`, `get_neo_catalogue`
  - Finalise track with `closed_at` when status becomes "lost"; enforce 90-day retention query
  - On write failure: log failure + record ID, queue for retry
  - Create all indexes defined in design schema
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 9.1 Write property test for data store round-trip (Property 11)
    - **Property 11: Data store round-trip**
    - **Validates: Requirements 9.1**

  - [ ]* 9.2 Write unit tests for data_store
    - Test round-trip write/read for each record type (TrackRecord, AlertRecord, NeoRecord)
    - Test `closed_at` set when track marked lost
    - Test retention query excludes records older than 90 days

- [x] 10. Checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement `api_client`
  - Implement `ApiClient` class in `src/api_client.py`
  - Fetch NASA NeoWs feed via `requests` at `fetch_interval_hours` interval in a background thread
  - Parse response into `NeoRecord` list and upsert into `data_store.neo_catalogue`
  - Match tracks to NEOs within `match_tolerance_au`; annotate `TrackRecord.neo_catalogue_id` and `neo_name`
  - Respect `max_requests_per_hour` rate limit; on API failure log and use cached data
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 11.1 Write unit tests for api_client
    - Test API failure logs error and continues without interrupting pipeline (mock requests)
    - Test rate limit not exceeded across multiple scheduled fetches

- [x] 12. Implement `dashboard`
  - Implement `src/dashboard.py` as a Streamlit app (entry point: `streamlit run src/dashboard.py`)
  - Live frame view: `st.empty()` updated in `while True` loop with `time.sleep(refresh_interval_seconds)`; draw bounding box overlays with OpenCV before passing to `st.image()`
  - Risk summary table: `st.dataframe()` with colour-coded Threat_Level column (green/amber/red)
  - Sidebar: date range, threat level filter, track ID selector via `st.sidebar` widgets; state in `st.session_state`
  - Selected track panel: `st.metric()` for velocity/distance/size KPIs; `st.plotly_chart()` for speed/distance time-series and 3D trajectory
  - Alert banner: `st.error()` / `st.warning()` at top of main area for active alerts
  - Historical log: `st.dataframe()` with column config, filterable
  - Connection-lost banner when data feed interrupted > 5 seconds
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [x] 13. Implement `main` pipeline and CLI
  - Implement `src/main.py` with the main loop from design: `image_processor → detector → tracker → trajectory_predictor → threat_classifier → alert_manager → data_store`
  - Load config at startup; exit with code 1 on missing/invalid config
  - Catch unhandled module exceptions: log full stack trace, continue next frame
  - Implement CLI via `argparse`: `--config` flag; `--module` flag to run each module independently for testing
  - Start `api_client` background thread if `api_client.enabled`
  - _Requirements: 11.1, 11.2, 12.3, 12.4_

- [x] 14. Implement simulator / sample data generator
  - Implement `src/simulator.py`: generate synthetic `DecodedFrame` sequences with configurable asteroid trajectories
  - Support generating FITS and PNG frames with injected asteroid bounding boxes at known positions
  - Output a directory of frames usable as `source_type: directory` input for end-to-end testing
  - _Requirements: 1.1, 1.4, 1.5_

- [x] 15. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)`
- All 13 correctness properties from the design are covered by property sub-tasks
- Checkpoints at tasks 5, 10, and 15 ensure incremental validation
