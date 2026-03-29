# Requirements Document

## Introduction

The Asteroid Threat Monitor is a full-stack AI-powered application that ingests continuous image streams from telescope feeds or datasets, detects and tracks asteroids using computer vision and deep learning, predicts their trajectories using physics-based and ML models, classifies threat levels, and presents all findings through an interactive real-time dashboard. The system is designed to support planetary defense researchers and space agencies in identifying potentially hazardous near-Earth objects (NEOs) early and reliably.

---

## Glossary

- **System**: The Asteroid Threat Monitor application as a whole.
- **Image_Processor**: The module responsible for ingesting image frames and running object detection.
- **Detector**: The deep learning model (YOLO or CNN) used to locate asteroids within a single frame.
- **Tracker**: The module that links detections across sequential frames using Kalman Filter or optical flow.
- **Trajectory_Predictor**: The module that forecasts future asteroid paths using orbital mechanics and/or ML regression.
- **Threat_Classifier**: The module that assigns a threat level to a tracked asteroid based on distance, speed, and trajectory.
- **Alert_Manager**: The module that evaluates threat classifications and dispatches notifications.
- **Dashboard**: The frontend UI built with Streamlit that visualises tracking data, graphs, and risk indicators.
- **Data_Store**: The persistence layer (SQLite or PostgreSQL) that stores processed asteroid records.
- **API_Client**: The optional module that fetches supplementary data from public space APIs such as NASA NeoWs.
- **NEO**: Near-Earth Object — any asteroid or comet whose orbit brings it within 1.3 AU of the Sun.
- **Threat_Level**: A categorical classification of an asteroid's danger: Safe, Potentially_Hazardous, or Dangerous.
- **Frame**: A single image from the telescope feed or dataset.
- **Track**: A time-ordered sequence of detections belonging to the same physical asteroid.
- **Confidence_Score**: A numeric value in [0, 1] representing the Detector's certainty that a bounding box contains an asteroid.

---

## Requirements

### Requirement 1: Image Ingestion

**User Story:** As a planetary defense analyst, I want the system to accept both real-time and batch image inputs, so that I can monitor live telescope feeds and replay historical datasets.

#### Acceptance Criteria

1. THE Image_Processor SHALL accept image streams delivered as a directory of sequentially named files, a video file, or a live RTSP/HTTP stream URL.
2. WHEN an image frame is received, THE Image_Processor SHALL decode it into a standardised RGB array within 50 ms of receipt.
3. IF an incoming frame cannot be decoded, THEN THE Image_Processor SHALL log the error with the frame identifier and skip to the next frame without halting the pipeline.
4. THE Image_Processor SHALL support FITS, JPEG, PNG, and TIFF image formats.
5. WHILE operating in batch mode, THE Image_Processor SHALL process frames in chronological order based on file timestamp or sequence number.

---

### Requirement 2: Asteroid Detection

**User Story:** As a planetary defense analyst, I want the system to detect asteroids in each frame using a trained deep learning model, so that I can identify objects of interest automatically.

#### Acceptance Criteria

1. WHEN a decoded frame is available, THE Detector SHALL produce a list of bounding boxes, each with a Confidence_Score, within 200 ms per frame on reference hardware.
2. THE Detector SHALL filter out detections with a Confidence_Score below a configurable threshold (default: 0.5).
3. THE Detector SHALL support loading weights from YOLO (v5 or v8) and custom CNN model files in ONNX or PyTorch format.
4. IF no asteroids are detected in a frame, THEN THE Detector SHALL return an empty detection list and log the frame as processed.
5. THE Detector SHALL expose a configurable input resolution parameter to balance speed and accuracy.

---

### Requirement 3: Multi-Object Tracking

**User Story:** As a planetary defense analyst, I want the system to track each detected asteroid across sequential frames, so that I can observe continuous movement and build a trajectory history.

#### Acceptance Criteria

1. WHEN detections are available for a frame, THE Tracker SHALL associate each detection with an existing Track or create a new Track, using a Kalman Filter or optical flow algorithm.
2. THE Tracker SHALL assign a unique, persistent Track ID to each asteroid for the lifetime of the observation session.
3. WHILE a Track has not received a matching detection for more than a configurable number of consecutive frames (default: 10), THE Tracker SHALL mark the Track as lost.
4. THE Tracker SHALL maintain a rolling history of at least 100 frame positions per Track for downstream analysis.
5. IF two detections in the same frame are within a configurable pixel distance threshold, THEN THE Tracker SHALL treat them as separate objects and assign distinct Track IDs.

---

### Requirement 4: Physical Parameter Estimation

**User Story:** As a planetary defense analyst, I want the system to estimate each asteroid's velocity, direction, distance, and size from tracking data, so that I have the physical parameters needed for threat assessment.

#### Acceptance Criteria

1. WHEN a Track contains at least 5 position samples, THE Tracker SHALL compute the asteroid's 2D velocity vector in pixels per frame and convert it to an estimated km/s value using the known or estimated field-of-view scale.
2. THE Tracker SHALL estimate the asteroid's angular diameter in arc-seconds from its bounding box dimensions and the instrument's known plate scale.
3. THE Tracker SHALL derive a distance estimate in AU using the asteroid's apparent magnitude or angular size combined with a configurable reference model.
4. WHEN physical parameters are updated, THE Tracker SHALL emit a structured data record containing Track ID, timestamp, position, velocity, direction, estimated size, and estimated distance.
5. IF the field-of-view scale is not configured, THEN THE Tracker SHALL flag all derived physical values as approximate and include a warning in the data record.

---

### Requirement 5: Trajectory Prediction

**User Story:** As a planetary defense analyst, I want the system to predict each asteroid's future path, so that I can determine whether it poses a collision risk to Earth.

#### Acceptance Criteria

1. WHEN a Track contains at least 20 position samples, THE Trajectory_Predictor SHALL compute a predicted path for the next configurable time horizon (default: 72 hours) using an orbital mechanics model.
2. THE Trajectory_Predictor SHALL determine whether the predicted path intersects Earth's orbital corridor, defined as within 0.05 AU of Earth's position at the predicted time.
3. WHERE an LSTM time-series model is configured, THE Trajectory_Predictor SHALL use it to refine the orbital mechanics prediction and output a blended forecast.
4. THE Trajectory_Predictor SHALL output a confidence interval for the predicted position at each forecast step.
5. IF the Track history contains fewer than 20 samples, THEN THE Trajectory_Predictor SHALL defer prediction and log the Track ID with the current sample count.
6. THE Trajectory_Predictor SHALL re-compute predictions whenever new position samples are added to a Track.

---

### Requirement 6: Threat Classification

**User Story:** As a planetary defense analyst, I want each tracked asteroid to be assigned a threat level, so that I can prioritise my response actions.

#### Acceptance Criteria

1. WHEN trajectory prediction results are available for a Track, THE Threat_Classifier SHALL assign one of three Threat_Levels: Safe, Potentially_Hazardous, or Dangerous.
2. THE Threat_Classifier SHALL classify an asteroid as Dangerous when the predicted closest approach distance is less than 0.002 AU and the estimated velocity exceeds 10 km/s.
3. THE Threat_Classifier SHALL classify an asteroid as Potentially_Hazardous when the predicted closest approach distance is between 0.002 AU and 0.05 AU, regardless of velocity.
4. THE Threat_Classifier SHALL classify an asteroid as Safe when the predicted closest approach distance exceeds 0.05 AU.
5. WHEN a Threat_Level changes for an existing Track, THE Threat_Classifier SHALL record the previous level, the new level, and the UTC timestamp of the change.
6. THE Threat_Classifier SHALL re-evaluate the Threat_Level whenever the Trajectory_Predictor updates the prediction for a Track.

---

### Requirement 7: Alert System

**User Story:** As a planetary defense analyst, I want the system to trigger alerts when a hazardous asteroid is detected, so that I can take timely action.

#### Acceptance Criteria

1. WHEN THE Threat_Classifier assigns a Threat_Level of Potentially_Hazardous or Dangerous to a Track, THE Alert_Manager SHALL generate an alert containing the Track ID, Threat_Level, estimated closest approach distance, estimated velocity, and UTC timestamp.
2. THE Alert_Manager SHALL display the alert as a visual notification on the Dashboard within 2 seconds of the Threat_Level assignment.
3. THE Alert_Manager SHALL map Threat_Levels to severity levels: Potentially_Hazardous maps to medium severity, Dangerous maps to high severity.
4. WHERE email notification is configured, THE Alert_Manager SHALL send an email alert to all configured recipients within 30 seconds of a Threat_Level of Potentially_Hazardous or Dangerous being assigned.
5. WHERE SMS notification is configured, THE Alert_Manager SHALL send an SMS alert to all configured recipients within 30 seconds of a Threat_Level of Dangerous being assigned.
6. IF an alert delivery attempt fails, THEN THE Alert_Manager SHALL retry delivery up to 3 times with a 10-second interval before logging the failure.
7. THE Alert_Manager SHALL suppress duplicate alerts for the same Track ID and Threat_Level within a configurable deduplication window (default: 300 seconds).

---

### Requirement 8: Dashboard UI

**User Story:** As a planetary defense analyst, I want a real-time dashboard, so that I can monitor all tracked asteroids, their parameters, and risk status at a glance.

#### Acceptance Criteria

1. THE Dashboard SHALL be implemented using Streamlit as the primary dashboard framework.
2. THE Dashboard SHALL display a live 2D visualisation of all active Tracks overlaid on the current telescope frame, updating at a minimum of 1 frame per second using Streamlit's `st.empty()` or `st.image()` components with auto-refresh.
3. THE Dashboard SHALL render time-series graphs of speed, distance, and trajectory deviation for each selected Track using Streamlit-compatible charting (e.g., Plotly via `st.plotly_chart`).
4. THE Dashboard SHALL display a risk status indicator for each active Track using colour coding: green for Safe, amber for Potentially_Hazardous, and red for Dangerous.
5. THE Dashboard SHALL provide a historical data log view showing all processed Tracks, their physical parameters, and Threat_Level history, filterable by date range and Threat_Level using Streamlit sidebar widgets.
6. WHEN a user selects a Track on the visualisation, THE Dashboard SHALL display the full parameter panel for that Track including velocity, direction, size, distance, and predicted path.
7. THE Dashboard SHALL provide a 3D trajectory visualisation view for any selected Track using Plotly rendered via `st.plotly_chart`.
8. IF the data feed is interrupted for more than 5 seconds, THEN THE Dashboard SHALL display a prominent connection-lost warning and the timestamp of the last received frame.

---

### Requirement 9: Data Persistence

**User Story:** As a planetary defense analyst, I want all processed asteroid data to be stored persistently, so that I can review historical observations and audit system decisions.

#### Acceptance Criteria

1. THE Data_Store SHALL persist every Track record including Track ID, all position samples, physical parameter estimates, Threat_Level history, and alert records.
2. THE Data_Store SHALL support both SQLite (for single-node deployments) and PostgreSQL (for multi-user deployments) as configurable backends.
3. WHEN a Track is marked as lost, THE Data_Store SHALL finalise the Track record with a closed timestamp.
4. THE Data_Store SHALL retain all Track records for a minimum of 90 days before they are eligible for archival or deletion.
5. IF a write operation to the Data_Store fails, THEN THE System SHALL log the failure with the affected record identifier and queue the record for retry.

---

### Requirement 10: External Data Integration

**User Story:** As a planetary defense analyst, I want the system to optionally enrich asteroid data with public space agency datasets, so that I can cross-reference detections with known NEO catalogues.

#### Acceptance Criteria

1. WHERE NASA NeoWs API integration is configured, THE API_Client SHALL query the NASA NeoWs feed at a configurable interval (default: 1 hour) and store the results in the Data_Store.
2. WHEN a Track's estimated orbital parameters match a known NEO within a configurable tolerance, THE API_Client SHALL annotate the Track record with the NEO's catalogue identifier and official name.
3. IF the external API is unreachable, THEN THE API_Client SHALL log the failure and continue operating using locally cached data without interrupting the detection pipeline.
4. THE API_Client SHALL respect the API provider's rate limits and SHALL NOT exceed the configured maximum requests per hour.

---

### Requirement 11: Configuration Management

**User Story:** As a system operator, I want all tunable parameters to be managed through a single configuration file, so that I can adapt the system to different instruments and deployment environments without modifying source code.

#### Acceptance Criteria

1. THE System SHALL load all configurable parameters from a single YAML or JSON configuration file at startup.
2. WHEN the configuration file is absent or malformed, THE System SHALL log a descriptive error and exit with a non-zero status code.
3. THE System SHALL validate all configuration values against defined ranges and types at startup and SHALL reject invalid values with a descriptive error message.
4. WHERE a configurable parameter is not present in the configuration file, THE System SHALL apply the documented default value.

---

### Requirement 12: Modular Architecture

**User Story:** As a developer, I want the system to be organised into clearly separated modules, so that I can maintain, test, and extend each component independently.

#### Acceptance Criteria

1. THE System SHALL organise source code into distinct modules: image_processor, detector, tracker, trajectory_predictor, threat_classifier, alert_manager, dashboard, data_store, and api_client.
2. THE System SHALL define a documented internal data contract (schema) for records passed between modules.
3. WHEN a module raises an unhandled exception, THE System SHALL log the exception with a full stack trace and continue processing subsequent frames where recovery is possible.
4. THE System SHALL expose a command-line interface that allows each module to be run and tested independently.
