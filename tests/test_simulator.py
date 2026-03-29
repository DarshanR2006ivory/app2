"""Unit tests for src/simulator.py."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.simulator import AsteroidConfig, Simulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(**kwargs) -> Simulator:
    return Simulator(width=64, height=48, **kwargs)


# ---------------------------------------------------------------------------
# generate_frames — basic output
# ---------------------------------------------------------------------------

class TestGenerateFramesPNG:
    def test_returns_correct_number_of_paths(self, tmp_path):
        sim = _make_sim(format="png")
        paths = sim.generate_frames(str(tmp_path), n_frames=5)
        assert len(paths) == 5

    def test_png_files_exist(self, tmp_path):
        sim = _make_sim(format="png")
        paths = sim.generate_frames(str(tmp_path), n_frames=3)
        for p in paths:
            assert Path(p).exists(), f"Missing file: {p}"

    def test_png_files_are_named_sequentially(self, tmp_path):
        sim = _make_sim(format="png")
        paths = sim.generate_frames(str(tmp_path), n_frames=3)
        names = [Path(p).name for p in paths]
        assert names == ["frame_000000.png", "frame_000001.png", "frame_000002.png"]

    def test_png_readable_as_image(self, tmp_path):
        import cv2
        sim = _make_sim(format="png")
        paths = sim.generate_frames(str(tmp_path), n_frames=1)
        img = cv2.imread(paths[0])
        assert img is not None
        assert img.shape == (48, 64, 3)


class TestGenerateFramesFITS:
    def test_returns_correct_number_of_paths(self, tmp_path):
        sim = _make_sim(format="fits")
        paths = sim.generate_frames(str(tmp_path), n_frames=4)
        assert len(paths) == 4

    def test_fits_files_exist(self, tmp_path):
        sim = _make_sim(format="fits")
        paths = sim.generate_frames(str(tmp_path), n_frames=2)
        for p in paths:
            assert Path(p).exists()

    def test_fits_readable(self, tmp_path):
        from astropy.io import fits
        sim = _make_sim(format="fits")
        paths = sim.generate_frames(str(tmp_path), n_frames=1)
        with fits.open(paths[0]) as hdul:
            data = hdul[0].data
        assert data is not None
        assert data.shape == (48, 64)

    def test_fits_data_dtype_float(self, tmp_path):
        from astropy.io import fits
        sim = _make_sim(format="fits")
        paths = sim.generate_frames(str(tmp_path), n_frames=1)
        with fits.open(paths[0]) as hdul:
            data = hdul[0].data
        assert np.issubdtype(data.dtype, np.floating)


class TestGenerateFramesBoth:
    def test_both_format_doubles_file_count(self, tmp_path):
        sim = _make_sim(format="both")
        paths = sim.generate_frames(str(tmp_path), n_frames=3)
        # 3 frames × 2 formats = 6 files
        assert len(paths) == 6

    def test_both_format_produces_png_and_fits(self, tmp_path):
        sim = _make_sim(format="both")
        paths = sim.generate_frames(str(tmp_path), n_frames=2)
        exts = {Path(p).suffix for p in paths}
        assert ".png" in exts
        assert ".fits" in exts


# ---------------------------------------------------------------------------
# Metadata JSON
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_file_created(self, tmp_path):
        sim = _make_sim(format="png")
        sim.generate_frames(str(tmp_path), n_frames=3)
        assert (tmp_path / "metadata.json").exists()

    def test_metadata_frame_count(self, tmp_path):
        sim = _make_sim(format="png")
        sim.generate_frames(str(tmp_path), n_frames=5)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["n_frames"] == 5
        assert len(meta["frames"]) == 5

    def test_metadata_asteroid_positions_present(self, tmp_path):
        ast = AsteroidConfig(x=32, y=24, vx=1.0, vy=0.5)
        sim = Simulator(width=64, height=48, format="png", asteroids=[ast])
        sim.generate_frames(str(tmp_path), n_frames=3)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        for frame in meta["frames"]:
            assert len(frame["asteroids"]) == 1
            a = frame["asteroids"][0]
            assert "centroid_x" in a
            assert "centroid_y" in a
            assert "bbox" in a
            assert len(a["bbox"]) == 4

    def test_metadata_ground_truth_linear_motion(self, tmp_path):
        """Asteroid positions should follow linear trajectory."""
        ast = AsteroidConfig(x=10.0, y=10.0, vx=2.0, vy=1.0, width=6, height=6)
        sim = Simulator(width=64, height=48, format="png", asteroids=[ast])
        sim.generate_frames(str(tmp_path), n_frames=5)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        for i, frame in enumerate(meta["frames"]):
            a = frame["asteroids"][0]
            expected_x = 10.0 + 2.0 * i
            expected_y = 10.0 + 1.0 * i
            # Positions may be clamped to frame bounds
            assert abs(a["centroid_x"] - expected_x) < 1.0 or a["centroid_x"] == pytest.approx(
                max(3.0, min(61.0, expected_x)), abs=1.0
            )

    def test_metadata_dimensions_recorded(self, tmp_path):
        sim = _make_sim(format="png")
        sim.generate_frames(str(tmp_path), n_frames=1)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["width"] == 64
        assert meta["height"] == 48


# ---------------------------------------------------------------------------
# AsteroidConfig / trajectory
# ---------------------------------------------------------------------------

class TestAsteroidConfig:
    def test_explicit_asteroids_used(self, tmp_path):
        asteroids = [
            AsteroidConfig(x=20, y=20, vx=1.0, vy=0.0, width=10, height=10, brightness=200),
            AsteroidConfig(x=40, y=30, vx=-1.0, vy=0.5, width=8, height=8, brightness=180),
        ]
        sim = Simulator(width=64, height=48, format="png", asteroids=asteroids)
        paths = sim.generate_frames(str(tmp_path), n_frames=2)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["n_asteroids"] == 2
        for frame in meta["frames"]:
            assert len(frame["asteroids"]) == 2

    def test_default_asteroids_count(self, tmp_path):
        sim = Simulator(width=64, height=48, n_asteroids=3, format="png")
        assert len(sim.asteroids) == 3

    def test_position_clamped_to_frame(self):
        """Asteroid starting near edge should be clamped."""
        ast = AsteroidConfig(x=0, y=0, vx=-10.0, vy=-10.0, width=10, height=10)
        sim = Simulator(width=64, height=48, format="png", asteroids=[ast])
        positions = sim._compute_positions(frame_idx=5)
        cx, cy = positions[0]
        assert cx >= ast.width / 2
        assert cy >= ast.height / 2

    def test_seed_reproducibility(self):
        sim1 = Simulator(width=64, height=48, n_asteroids=2, seed=7)
        sim2 = Simulator(width=64, height=48, n_asteroids=2, seed=7)
        for a1, a2 in zip(sim1.asteroids, sim2.asteroids):
            assert a1.x == a2.x
            assert a1.y == a2.y
            assert a1.vx == a2.vx

    def test_different_seeds_differ(self):
        sim1 = Simulator(width=64, height=48, n_asteroids=1, seed=1)
        sim2 = Simulator(width=64, height=48, n_asteroids=1, seed=99)
        # Very unlikely to be identical
        assert sim1.asteroids[0].x != sim2.asteroids[0].x or \
               sim1.asteroids[0].y != sim2.asteroids[0].y


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------

class TestOutputDirectory:
    def test_creates_output_dir_if_absent(self, tmp_path):
        new_dir = tmp_path / "nested" / "output"
        assert not new_dir.exists()
        sim = _make_sim(format="png")
        sim.generate_frames(str(new_dir), n_frames=1)
        assert new_dir.exists()

    def test_zero_frames_produces_empty_list(self, tmp_path):
        sim = _make_sim(format="png")
        paths = sim.generate_frames(str(tmp_path), n_frames=0)
        assert paths == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_png(self, tmp_path):
        from src.simulator import main
        main(["--output", str(tmp_path), "--n-frames", "3", "--format", "png",
              "--width", "32", "--height", "24"])
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) == 3

    def test_cli_fits(self, tmp_path):
        from src.simulator import main
        main(["--output", str(tmp_path), "--n-frames", "2", "--format", "fits",
              "--width", "32", "--height", "24"])
        fits_files = list(tmp_path.glob("*.fits"))
        assert len(fits_files) == 2

    def test_cli_both(self, tmp_path):
        from src.simulator import main
        main(["--output", str(tmp_path), "--n-frames", "2", "--format", "both",
              "--width", "32", "--height", "24"])
        assert len(list(tmp_path.glob("*.png"))) == 2
        assert len(list(tmp_path.glob("*.fits"))) == 2

    def test_cli_metadata_written(self, tmp_path):
        from src.simulator import main
        main(["--output", str(tmp_path), "--n-frames", "2", "--format", "png",
              "--width", "32", "--height", "24"])
        assert (tmp_path / "metadata.json").exists()
