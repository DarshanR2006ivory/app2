"""Unit tests for ImageProcessor (Task 2, sub-task 2.1)."""

from __future__ import annotations

import logging
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.config import ImageProcessorConfig
from src.image_processor import ImageProcessor
from src.models import DecodedFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> ImageProcessorConfig:
    defaults = dict(
        source_type="directory",
        source_path="./data/frames",
        supported_formats=["fits", "jpeg", "png", "tiff"],
        decode_timeout_ms=50,
    )
    defaults.update(kwargs)
    return ImageProcessorConfig(**defaults)


def _write_png(path: Path, h: int = 8, w: int = 8) -> None:
    """Write a minimal valid RGB PNG file."""
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 100  # some non-zero red channel
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _write_fits(path: Path, h: int = 8, w: int = 8) -> None:
    """Write a minimal valid FITS file with a 2-D image."""
    from astropy.io import fits

    data = np.arange(h * w, dtype=np.float32).reshape(h, w)
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(str(path), overwrite=True)


# ---------------------------------------------------------------------------
# Directory source – PNG
# ---------------------------------------------------------------------------

class TestDirectorySourcePNG:
    def test_decodes_png_to_rgb_shape(self, tmp_path: Path) -> None:
        """PNG file in directory mode decodes to (H, W, 3) uint8 array."""
        _write_png(tmp_path / "frame_001.png", h=16, w=24)
        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        frame = proc.next_frame()
        proc.close()

        assert frame is not None
        assert isinstance(frame, DecodedFrame)
        assert frame.rgb_array.shape == (16, 24, 3)
        assert frame.rgb_array.dtype == np.uint8

    def test_frame_id_is_filename(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "frame_001.png")
        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        frame = proc.next_frame()
        proc.close()

        assert frame is not None
        assert frame.frame_id == "frame_001.png"

    def test_returns_none_when_exhausted(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "a.png")
        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        proc.next_frame()  # consume the only frame
        frame = proc.next_frame()
        proc.close()

        assert frame is None

    def test_batch_chronological_order(self, tmp_path: Path) -> None:
        """Frames are returned in chronological order (sorted by mtime/name)."""
        import time

        for name in ["frame_001.png", "frame_002.png", "frame_003.png"]:
            _write_png(tmp_path / name)
            time.sleep(0.01)  # ensure distinct mtimes

        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        ids = []
        while (f := proc.next_frame()) is not None:
            ids.append(f.frame_id)
        proc.close()

        assert ids == sorted(ids)

    def test_corrupt_file_skipped_without_halting(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt image file is skipped; the next valid frame is returned."""
        (tmp_path / "bad_001.png").write_bytes(b"not a valid image")
        _write_png(tmp_path / "good_002.png")

        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        with caplog.at_level(logging.ERROR, logger="src.image_processor"):
            proc = ImageProcessor(cfg)
            frame = proc.next_frame()
            proc.close()

        # The corrupt file should have been logged
        assert any("bad_001.png" in r.message for r in caplog.records)
        # The good frame should still be returned
        assert frame is not None
        assert frame.frame_id == "good_002.png"

    def test_all_corrupt_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If all files are corrupt, next_frame() returns None without raising."""
        (tmp_path / "bad.png").write_bytes(b"garbage")
        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        with caplog.at_level(logging.ERROR, logger="src.image_processor"):
            proc = ImageProcessor(cfg)
            frame = proc.next_frame()
            proc.close()

        assert frame is None
        assert len(caplog.records) >= 1


# ---------------------------------------------------------------------------
# FITS source
# ---------------------------------------------------------------------------

class TestFITSSource:
    def test_decodes_fits_to_rgb_shape(self, tmp_path: Path) -> None:
        """FITS file decodes to (H, W, 3) uint8 array."""
        _write_fits(tmp_path / "image_001.fits", h=10, w=12)
        cfg = _make_config(source_type="fits", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        frame = proc.next_frame()
        proc.close()

        assert frame is not None
        assert frame.rgb_array.shape == (10, 12, 3)
        assert frame.rgb_array.dtype == np.uint8

    def test_fits_corrupt_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt FITS file is skipped; error is logged."""
        (tmp_path / "bad.fits").write_bytes(b"not fits data")
        _write_fits(tmp_path / "good.fits", h=4, w=4)

        cfg = _make_config(source_type="fits", source_path=str(tmp_path))
        with caplog.at_level(logging.ERROR, logger="src.image_processor"):
            proc = ImageProcessor(cfg)
            frame = proc.next_frame()
            proc.close()

        assert any("bad.fits" in r.message for r in caplog.records)
        assert frame is not None
        assert frame.frame_id == "good.fits"

    def test_fits_batch_chronological_order(self, tmp_path: Path) -> None:
        """FITS frames are returned in chronological order."""
        import time

        for name in ["a.fits", "b.fits", "c.fits"]:
            _write_fits(tmp_path / name)
            time.sleep(0.01)

        cfg = _make_config(source_type="fits", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        ids = []
        while (f := proc.next_frame()) is not None:
            ids.append(f.frame_id)
        proc.close()

        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# DecodedFrame contract
# ---------------------------------------------------------------------------

class TestDecodedFrameContract:
    def test_frame_has_utc_timestamp(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "frame.png")
        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        frame = proc.next_frame()
        proc.close()

        assert frame is not None
        assert frame.timestamp.tzinfo is not None

    def test_frame_source_matches_config(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "frame.png")
        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        frame = proc.next_frame()
        proc.close()

        assert frame is not None
        assert frame.source == str(tmp_path)

    def test_close_makes_next_frame_return_none(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "frame.png")
        cfg = _make_config(source_type="directory", source_path=str(tmp_path))
        proc = ImageProcessor(cfg)
        proc.close()
        frame = proc.next_frame()

        assert frame is None
