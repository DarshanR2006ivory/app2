# Asteroid Threat Monitor

An AI-powered pipeline that ingests telescope image streams, detects and tracks asteroids using computer vision and deep learning, predicts trajectories using orbital mechanics, classifies threat levels, and presents findings through a real-time Streamlit dashboard.

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Config file

A `config.yaml` is already provided in the project root. Edit it as needed — all parameters not listed will use documented defaults. See `src/config.py` for the full schema.

### 2. Run the dashboard

```bash
streamlit run src/dashboard.py -- --config config.yaml
```

Then open http://localhost:8501 in your browser.

The dashboard automatically generates synthetic asteroid frames on first launch if no real frames are present — no pipeline required to see it working immediately.

### 3. Generate sample data (optional)

To manually generate synthetic frames for the pipeline:

```bash
python -m src.simulator --output ./data/sim_frames --n-frames 50 --format png
```

Options:
- `--format` — `png`, `fits`, or `both`
- `--n-asteroids` — number of synthetic asteroids (default: 1)
- `--width` / `--height` — frame dimensions (default: 640x480)
- `--seed` — random seed for reproducibility (default: 42)

### 4. Run the full pipeline (optional)

```bash
python -m src.main --config config.yaml
```

Run this alongside the dashboard in a separate terminal. The pipeline writes processed data to the SQLite database and the dashboard reads from it live.

## Dashboard Features

- **Live Frame View** — cycles through real or auto-generated synthetic frames with bounding box overlays colour-coded by threat level (green / amber / red)
- **Risk Summary Table** — all active tracks with threat level, velocity, and distance
- **Selected Track Panel** — velocity, distance, size, and direction KPIs; speed and distance time-series charts; 3D trajectory visualisation
- **Alert Banner** — prominent warnings for Potentially Hazardous and Dangerous tracks
- **Historical Log** — filterable by date range and threat level
- **Connection Status** — shows feed lost banner if no new frames arrive for more than 5 seconds

## Running Modules Independently

Each module can be run and tested in isolation using the `--module` flag:

```bash
python -m src.main --config config.yaml --module image_processor
python -m src.main --config config.yaml --module detector
python -m src.main --config config.yaml --module tracker
python -m src.main --config config.yaml --module trajectory_predictor
python -m src.main --config config.yaml --module threat_classifier
python -m src.main --config config.yaml --module alert_manager
python -m src.main --config config.yaml --module data_store
python -m src.main --config config.yaml --module api_client
```

## Running Tests

```bash
pytest tests/ -q
```

## Project Structure

```
src/
  config.py               # YAML config loader and validation
  models.py               # Shared dataclasses (internal data contracts)
  image_processor.py      # Frame ingestion (directory, video, RTSP, FITS)
  detector.py             # Deep learning object detection (YOLO, ONNX, PyTorch)
  tracker.py              # Kalman filter multi-object tracking
  trajectory_predictor.py # Orbital mechanics trajectory forecasting
  threat_classifier.py    # Rule-based threat level classification
  alert_manager.py        # Alert generation and dispatch (email, SMS)
  data_store.py           # SQLite/PostgreSQL persistence (SQLAlchemy)
  api_client.py           # NASA NeoWs API integration
  dashboard.py            # Streamlit real-time dashboard
  simulator.py            # Synthetic frame generator for testing
  main.py                 # Pipeline entry point and CLI

tests/
  conftest.py             # Shared pytest fixtures
  test_config.py
  test_image_processor.py
  test_detector.py
  test_tracker.py
  test_trajectory_predictor.py
  test_threat_classifier.py
  test_alert_manager.py
  test_data_store.py
  test_api_client.py
  test_main.py
  test_simulator.py
```

## Notes

- The detector requires a trained model file (YOLOv8, ONNX, or PyTorch). Without one it logs a warning and returns empty detections — the rest of the pipeline still runs.
- The dashboard and pipeline are separate processes. Run them in two terminals pointing at the same `sqlite_path`.
- NASA NeoWs API integration is disabled by default. Enable it in config with `api_client.enabled: true` and set your `nasa_neows_api_key`.
- Email and SMS alerts are disabled by default. Configure them under `alert_manager.email` and `alert_manager.sms` in `config.yaml`.
