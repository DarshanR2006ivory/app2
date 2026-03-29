"""Simulator / sample data generator for the Asteroid Threat Monitor.

Generates synthetic image frames (PNG and/or FITS) with injected asteroid
bounding boxes at known positions following configurable trajectories.

Output directories are compatible with ``source_type: directory`` in the
image processor config, enabling end-to-end pipeline testing.

CLI usage::

    python -m src.simulator --output ./data/sim_frames --n-frames 50 --format png
    python -m src.simulator --output ./data/sim_frames --n-frames 50 --format fits
    python -m src.simulator --output ./data/sim_frames --n-frames 50 --format both
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asteroid trajectory dataclass
# ---------------------------------------------------------------------------

@dataclass
class AsteroidConfig:
    """Configuration for a single synthetic asteroid."""
    x: float          # starting centroid x in pixels
    y: float          # starting centroid y in pixels
    vx: float         # velocity in pixels/frame (x direction)
    vy: float         # velocity in pixels/frame (y direction)
    width: int = 20   # bounding box width in pixels
    height: int = 20  # bounding box height in pixels
    brightness: int = 200  # 0-255


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

class Simulator:
    """Generate synthetic telescope frames with asteroid trajectories.

    Parameters
    ----------
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    n_asteroids:
        Number of asteroids to inject (uses default trajectories when
        ``asteroids`` is not provided).
    format:
        Output format: ``"png"``, ``"fits"``, or ``"both"``.
    asteroids:
        Explicit list of :class:`AsteroidConfig` objects.  When *None*,
        ``n_asteroids`` default trajectories are generated automatically.
    seed:
        Random seed for reproducible default trajectory generation.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        n_asteroids: int = 1,
        format: Literal["png", "fits", "both"] = "png",
        asteroids: list[AsteroidConfig] | None = None,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.format = format

        if asteroids is not None:
            self.asteroids = asteroids
        else:
            rng = np.random.default_rng(seed)
            self.asteroids = self._default_asteroids(n_asteroids, rng)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_frames(self, output_dir: str, n_frames: int) -> list[str]:
        """Generate *n_frames* synthetic frames and write them to *output_dir*.

        Returns a list of file paths (relative to *output_dir*) in frame order.
        A ``metadata.json`` file is also written to *output_dir* containing
        frame filenames and ground-truth asteroid positions per frame.

        Parameters
        ----------
        output_dir:
            Directory to write frames into (created if absent).
        n_frames:
            Number of frames to generate.

        Returns
        -------
        list[str]
            Absolute paths to the generated frame files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_paths: list[str] = []
        metadata: list[dict] = []

        for frame_idx in range(n_frames):
            # Compute asteroid positions for this frame
            positions = self._compute_positions(frame_idx)

            frame_entry: dict = {
                "frame_index": frame_idx,
                "asteroids": [],
            }

            frame_paths: list[str] = []

            if self.format in ("png", "both"):
                png_path = out / f"frame_{frame_idx:06d}.png"
                self._write_png(png_path, positions)
                frame_paths.append(str(png_path))
                frame_entry["png_file"] = png_path.name

            if self.format in ("fits", "both"):
                fits_path = out / f"frame_{frame_idx:06d}.fits"
                self._write_fits(fits_path, positions)
                frame_paths.append(str(fits_path))
                frame_entry["fits_file"] = fits_path.name

            # Record ground-truth bounding boxes
            for ast, (cx, cy) in zip(self.asteroids, positions):
                x1 = int(cx - ast.width / 2)
                y1 = int(cy - ast.height / 2)
                x2 = int(cx + ast.width / 2)
                y2 = int(cy + ast.height / 2)
                frame_entry["asteroids"].append({
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "bbox": [x1, y1, x2, y2],
                    "width": ast.width,
                    "height": ast.height,
                    "brightness": ast.brightness,
                    "vx": ast.vx,
                    "vy": ast.vy,
                })

            metadata.append(frame_entry)
            all_paths.extend(frame_paths)

        # Write metadata JSON
        meta_path = out / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_frames": n_frames,
                    "width": self.width,
                    "height": self.height,
                    "format": self.format,
                    "n_asteroids": len(self.asteroids),
                    "frames": metadata,
                },
                fh,
                indent=2,
            )
        logger.info("Simulator: wrote metadata to %s", meta_path)

        return all_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_asteroids(self, n: int, rng: np.random.Generator) -> list[AsteroidConfig]:
        """Generate *n* asteroids with random-but-sensible default trajectories."""
        margin = min(50, self.width // 4, self.height // 4)
        max_bbox = max(4, min(30, self.width // 8, self.height // 8))
        min_bbox = max(2, max_bbox // 2)

        asteroids = []
        for _ in range(n):
            x_lo = margin
            x_hi = max(x_lo + 1, self.width - margin)
            y_lo = margin
            y_hi = max(y_lo + 1, self.height - margin)
            x = float(rng.integers(x_lo, x_hi))
            y = float(rng.integers(y_lo, y_hi))
            vx = float(rng.uniform(-3.0, 3.0))
            vy = float(rng.uniform(-3.0, 3.0))
            w = int(rng.integers(min_bbox, max_bbox + 1))
            h = int(rng.integers(min_bbox, max_bbox + 1))
            brightness = int(rng.integers(150, 255))
            asteroids.append(AsteroidConfig(x=x, y=y, vx=vx, vy=vy,
                                            width=w, height=h, brightness=brightness))
        return asteroids

    def _compute_positions(self, frame_idx: int) -> list[tuple[float, float]]:
        """Return (cx, cy) for each asteroid at *frame_idx* (linear motion)."""
        positions = []
        for ast in self.asteroids:
            cx = ast.x + ast.vx * frame_idx
            cy = ast.y + ast.vy * frame_idx
            # Clamp to frame bounds so the asteroid stays visible
            cx = max(ast.width / 2, min(self.width - ast.width / 2, cx))
            cy = max(ast.height / 2, min(self.height - ast.height / 2, cy))
            positions.append((cx, cy))
        return positions

    def _make_background(self) -> np.ndarray:
        """Create a dark starfield background (uint8, shape H×W)."""
        bg = np.zeros((self.height, self.width), dtype=np.uint8)
        # Scatter a few faint stars
        rng = np.random.default_rng(0)
        n_stars = int(self.width * self.height * 0.001)
        xs = rng.integers(0, self.width, size=n_stars)
        ys = rng.integers(0, self.height, size=n_stars)
        brightness = rng.integers(20, 80, size=n_stars).astype(np.uint8)
        bg[ys, xs] = brightness
        return bg

    def _write_png(self, path: Path, positions: list[tuple[float, float]]) -> None:
        """Write a PNG frame with asteroid blobs drawn as filled ellipses."""
        import cv2  # lazy import

        # Grayscale background → convert to BGR for drawing
        bg = self._make_background()
        frame_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        for ast, (cx, cy) in zip(self.asteroids, positions):
            center = (int(round(cx)), int(round(cy)))
            axes = (max(1, ast.width // 2), max(1, ast.height // 2))
            color = (ast.brightness, ast.brightness, ast.brightness)
            cv2.ellipse(frame_bgr, center, axes, 0, 0, 360, color, -1)

        cv2.imwrite(str(path), frame_bgr)

    def _write_fits(self, path: Path, positions: list[tuple[float, float]]) -> None:
        """Write a FITS frame with Gaussian blobs at asteroid positions."""
        from astropy.io import fits  # lazy import

        data = self._make_background().astype(np.float32)

        for ast, (cx, cy) in zip(self.asteroids, positions):
            sigma_x = ast.width / 4.0
            sigma_y = ast.height / 4.0
            # Bounding region to update (avoid iterating full frame)
            x_lo = max(0, int(cx) - ast.width * 2)
            x_hi = min(self.width, int(cx) + ast.width * 2 + 1)
            y_lo = max(0, int(cy) - ast.height * 2)
            y_hi = min(self.height, int(cy) + ast.height * 2 + 1)

            xs = np.arange(x_lo, x_hi, dtype=np.float32)
            ys = np.arange(y_lo, y_hi, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            gaussian = ast.brightness * np.exp(
                -(((xx - cx) ** 2) / (2 * sigma_x ** 2) +
                  ((yy - cy) ** 2) / (2 * sigma_y ** 2))
            )
            data[y_lo:y_hi, x_lo:x_hi] = np.clip(
                data[y_lo:y_hi, x_lo:x_hi] + gaussian, 0, 255
            )

        hdu = fits.PrimaryHDU(data)
        hdu.header["COMMENT"] = "Synthetic asteroid frame generated by src.simulator"
        hdu.writeto(str(path), overwrite=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.simulator",
        description="Generate synthetic asteroid frames for end-to-end testing.",
    )
    parser.add_argument(
        "--output", default="./data/sim_frames",
        help="Output directory for generated frames (default: ./data/sim_frames)",
    )
    parser.add_argument(
        "--n-frames", type=int, default=50,
        help="Number of frames to generate (default: 50)",
    )
    parser.add_argument(
        "--format", choices=["png", "fits", "both"], default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--n-asteroids", type=int, default=1,
        help="Number of synthetic asteroids to inject (default: 1)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Frame width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Frame height in pixels (default: 480)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible trajectory generation (default: 42)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    sim = Simulator(
        width=args.width,
        height=args.height,
        n_asteroids=args.n_asteroids,
        format=args.format,
        seed=args.seed,
    )

    logger.info(
        "Simulator: generating %d frames (%s) with %d asteroid(s) → %s",
        args.n_frames, args.format, args.n_asteroids, args.output,
    )
    paths = sim.generate_frames(args.output, args.n_frames)
    logger.info("Simulator: wrote %d file(s) to %s", len(paths), args.output)


if __name__ == "__main__":
    main()
