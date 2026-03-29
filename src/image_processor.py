"""Image processor: ingests image sources and decodes frames to RGB arrays."""

from __future__ import annotations

import logging
import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from src.config import ImageProcessorConfig
from src.models import DecodedFrame

logger = logging.getLogger(__name__)

# Supported image file extensions for directory mode
_IMAGE_EXTENSIONS = {".fits", ".fit", ".jpeg", ".jpg", ".png", ".tiff", ".tif"}


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _decode_with_timeout(fn, timeout_s: float):
    """Run fn() in a thread; return its result or raise TimeoutError."""
    result = [None]
    exc = [None]

    def _run():
        try:
            result[0] = fn()
        except Exception as e:  # noqa: BLE001
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"Decode exceeded {timeout_s * 1000:.0f} ms timeout")
    if exc[0] is not None:
        raise exc[0]
    return result[0]


# ---------------------------------------------------------------------------
# Source-specific iterators
# ---------------------------------------------------------------------------

def _iter_directory(source_path: str) -> Iterator[tuple[str, datetime, Path]]:
    """Yield (frame_id, timestamp, path) for image files sorted by name."""
    root = Path(source_path)
    files = sorted(
        (p for p in root.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    for p in files:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        yield p.name, mtime, p


def _decode_image_file(path: Path) -> np.ndarray:
    """Decode a JPEG/PNG/TIFF file to an RGB uint8 array via OpenCV."""
    import cv2  # lazy import

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2.imread returned None for {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_fits_file(path: Path) -> np.ndarray:
    """Decode a FITS file to an RGB uint8 array."""
    from astropy.io import fits  # lazy import

    with fits.open(str(path)) as hdul:
        # Find the first image HDU with data
        data = None
        for hdu in hdul:
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                data = hdu.data
                break
        if data is None:
            raise ValueError(f"No image data found in FITS file {path}")

    data = np.array(data, dtype=float)

    # Handle multi-channel FITS (e.g. shape (3, H, W) or (H, W, 3))
    if data.ndim == 3:
        if data.shape[0] == 3:
            # (3, H, W) → (H, W, 3)
            data = np.transpose(data, (1, 2, 0))
        elif data.shape[2] != 3:
            # Take first channel
            data = data[:, :, 0]

    if data.ndim == 2:
        # Grayscale → replicate to RGB
        mn, mx = data.min(), data.max()
        if mx > mn:
            data = (data - mn) / (mx - mn) * 255.0
        else:
            data = np.zeros_like(data)
        data = data.astype(np.uint8)
        data = np.stack([data, data, data], axis=-1)
    else:
        # Normalise each channel to [0, 255]
        mn, mx = data.min(), data.max()
        if mx > mn:
            data = (data - mn) / (mx - mn) * 255.0
        else:
            data = np.zeros_like(data)
        data = data.astype(np.uint8)

    return data


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig) -> None:
        self.config = config
        self._timeout_s = config.decode_timeout_ms / 1000.0
        self._source_type = config.source_type.lower()
        self._source_path = config.source_path

        # State for directory mode
        self._dir_iter: Iterator[tuple[str, datetime, Path]] | None = None

        # State for video / rtsp / http mode
        self._cap = None  # cv2.VideoCapture
        self._frame_index: int = 0

        self._closed = False
        self._init_source()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_source(self) -> None:
        if self._source_type == "directory":
            self._dir_iter = _iter_directory(self._source_path)
        elif self._source_type in ("video", "rtsp", "http"):
            import cv2  # lazy import

            self._cap = cv2.VideoCapture(self._source_path)
            if not self._cap.isOpened():
                logger.error(
                    "ImageProcessor: failed to open video source '%s'", self._source_path
                )
        elif self._source_type == "fits":
            # FITS directory: same as directory but only .fits files
            self._dir_iter = _iter_directory(self._source_path)
        else:
            logger.error(
                "ImageProcessor: unsupported source_type '%s'", self._source_type
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_frame(self) -> DecodedFrame | None:
        """Return next decoded frame, or None when stream is exhausted."""
        if self._closed:
            return None

        if self._source_type == "directory":
            return self._next_directory_frame()
        elif self._source_type == "fits":
            return self._next_fits_frame()
        elif self._source_type in ("video", "rtsp", "http"):
            return self._next_video_frame()
        else:
            return None

    def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Source-specific frame fetchers
    # ------------------------------------------------------------------

    def _next_directory_frame(self) -> DecodedFrame | None:
        """Iterate over image files in the directory (non-FITS)."""
        if self._dir_iter is None:
            return None

        while True:
            try:
                frame_id, timestamp, path = next(self._dir_iter)
            except StopIteration:
                return None

            # Skip FITS files in plain directory mode; they are handled separately
            if path.suffix.lower() in (".fits", ".fit"):
                continue

            try:
                rgb = _decode_with_timeout(
                    lambda p=path: _decode_image_file(p), self._timeout_s
                )
            except TimeoutError:
                logger.error(
                    "ImageProcessor: decode timeout for frame '%s'", frame_id
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ImageProcessor: failed to decode frame '%s': %s", frame_id, exc
                )
                continue

            return DecodedFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                rgb_array=rgb,
                source=self._source_path,
            )

    def _next_fits_frame(self) -> DecodedFrame | None:
        """Iterate over FITS files in the source directory."""
        if self._dir_iter is None:
            return None

        while True:
            try:
                frame_id, timestamp, path = next(self._dir_iter)
            except StopIteration:
                return None

            # In fits mode, only process FITS files
            if path.suffix.lower() not in (".fits", ".fit"):
                continue

            try:
                rgb = _decode_with_timeout(
                    lambda p=path: _decode_fits_file(p), self._timeout_s
                )
            except TimeoutError:
                logger.error(
                    "ImageProcessor: decode timeout for FITS frame '%s'", frame_id
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ImageProcessor: failed to decode FITS frame '%s': %s", frame_id, exc
                )
                continue

            return DecodedFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                rgb_array=rgb,
                source=self._source_path,
            )

    def _next_video_frame(self) -> DecodedFrame | None:
        """Read the next frame from a VideoCapture source."""
        if self._cap is None or not self._cap.isOpened():
            return None

        import cv2  # lazy import

        self._frame_index += 1
        frame_id = f"frame_{self._frame_index:06d}"

        try:
            def _read(cap=self._cap):
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    return None
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            rgb = _decode_with_timeout(_read, self._timeout_s)
        except TimeoutError:
            logger.error(
                "ImageProcessor: decode timeout for video frame '%s'", frame_id
            )
            return self._next_video_frame()  # skip and try next
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ImageProcessor: failed to decode video frame '%s': %s", frame_id, exc
            )
            return self._next_video_frame()  # skip and try next

        if rgb is None:
            # Stream exhausted
            return None

        return DecodedFrame(
            frame_id=frame_id,
            timestamp=_utcnow(),
            rgb_array=rgb,
            source=self._source_path,
        )
