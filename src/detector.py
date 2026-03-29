"""Detector: runs deep learning model on decoded frames to produce bounding boxes."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.config import DetectorConfig
from src.models import DecodedFrame, Detection, DetectionList

logger = logging.getLogger(__name__)


class Detector:
    """Runs a deep learning model on decoded frames and returns filtered detections.

    Supports:
    - YOLOv8/v5 via ``ultralytics``
    - ONNX via ``onnxruntime``
    - PyTorch via ``torch.load``
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model: Any = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_type = self.config.model_type
        model_path = self.config.model_path

        if model_type in ("yolov8", "yolov5"):
            self._model = self._load_yolo(model_path, model_type)
        elif model_type == "onnx":
            self._model = self._load_onnx(model_path)
        elif model_type == "pytorch":
            self._model = self._load_pytorch(model_path)
        else:
            logger.error("Unknown model_type '%s'; no model loaded.", model_type)

    def _load_yolo(self, model_path: str, model_type: str) -> Any:
        try:
            from ultralytics import YOLO  # type: ignore[import]
            logger.info("Loading %s model from %s via ultralytics", model_type, model_path)
            return YOLO(model_path)
        except ImportError:
            logger.warning("ultralytics not installed; falling back to torch.hub for %s", model_type)
        except Exception as exc:
            logger.error("Failed to load YOLO model from '%s': %s", model_path, exc)
            return None

        # Fallback: torch.hub
        try:
            import torch  # type: ignore[import]
            hub_name = "ultralytics/yolov5" if model_type == "yolov5" else "ultralytics/yolov8"
            logger.info("Loading %s via torch.hub from %s", model_type, model_path)
            return torch.hub.load(hub_name, "custom", path=model_path, force_reload=False)
        except Exception as exc:
            logger.error("Failed to load YOLO model via torch.hub from '%s': %s", model_path, exc)
            return None

    def _load_onnx(self, model_path: str) -> Any:
        try:
            import onnxruntime as ort  # type: ignore[import]
            logger.info("Loading ONNX model from %s", model_path)
            return ort.InferenceSession(model_path)
        except Exception as exc:
            logger.error("Failed to load ONNX model from '%s': %s", model_path, exc)
            return None

    def _load_pytorch(self, model_path: str) -> Any:
        try:
            import torch  # type: ignore[import]
            logger.info("Loading PyTorch model from %s", model_path)
            return torch.load(model_path, map_location="cpu")
        except Exception as exc:
            logger.error("Failed to load PyTorch model from '%s': %s", model_path, exc)
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: DecodedFrame) -> DetectionList:
        """Run inference on *frame* and return detections above confidence threshold.

        Returns an empty list when no model is loaded, the frame produces no
        detections, or all detections fall below the configured threshold.
        """
        if self._model is None:
            logger.debug("No model loaded; returning empty DetectionList for frame %s", frame.frame_id)
            return []

        raw_detections = self._run_inference(frame)
        threshold = self.config.confidence_threshold
        filtered = [d for d in raw_detections if d.confidence >= threshold]

        if not filtered:
            logger.debug("No detections above threshold %.2f for frame %s", threshold, frame.frame_id)

        return filtered

    def _run_inference(self, frame: DecodedFrame) -> DetectionList:
        """Dispatch to the appropriate inference backend."""
        model_type = self.config.model_type

        if model_type in ("yolov8", "yolov5"):
            return self._infer_yolo(frame)
        elif model_type == "onnx":
            return self._infer_onnx(frame)
        elif model_type == "pytorch":
            return self._infer_pytorch(frame)

        return []

    def _infer_yolo(self, frame: DecodedFrame) -> DetectionList:
        try:
            results = self._model(frame.rgb_array, verbose=False)
            detections: DetectionList = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        frame_id=frame.frame_id,
                    ))
            return detections
        except Exception as exc:
            logger.error("YOLO inference failed for frame %s: %s", frame.frame_id, exc)
            return []

    def _infer_onnx(self, frame: DecodedFrame) -> DetectionList:
        try:
            import onnxruntime as ort  # type: ignore[import]
            h, w = self.config.input_resolution
            img = _preprocess(frame.rgb_array, (h, w))
            input_name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {input_name: img})
            return _parse_onnx_output(outputs, frame.frame_id)
        except Exception as exc:
            logger.error("ONNX inference failed for frame %s: %s", frame.frame_id, exc)
            return []

    def _infer_pytorch(self, frame: DecodedFrame) -> DetectionList:
        try:
            import torch  # type: ignore[import]
            h, w = self.config.input_resolution
            img = _preprocess(frame.rgb_array, (h, w))
            tensor = torch.from_numpy(img)
            with torch.no_grad():
                outputs = self._model(tensor)
            return _parse_pytorch_output(outputs, frame.frame_id)
        except Exception as exc:
            logger.error("PyTorch inference failed for frame %s: %s", frame.frame_id, exc)
            return []


# ------------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------------

def _preprocess(rgb_array: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resize and normalise an RGB array to a model-ready float32 tensor (1, 3, H, W)."""
    import cv2  # type: ignore[import]
    h, w = resolution
    resized = cv2.resize(rgb_array, (w, h))
    # HWC → CHW, normalise to [0, 1]
    chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...]  # add batch dim


def _parse_onnx_output(outputs: list[Any], frame_id: str) -> DetectionList:
    """Parse generic ONNX detection output (shape [1, N, 6]: x1,y1,x2,y2,conf,cls)."""
    detections: DetectionList = []
    if not outputs:
        return detections
    preds = outputs[0]
    if preds is None or preds.size == 0:
        return detections
    for row in np.squeeze(preds, axis=0):
        if row.shape[0] < 5:
            continue
        x1, y1, x2, y2, conf = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])
        detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, frame_id=frame_id))
    return detections


def _parse_pytorch_output(outputs: Any, frame_id: str) -> DetectionList:
    """Parse generic PyTorch detection output (tensor shape [N, 6]: x1,y1,x2,y2,conf,cls)."""
    import torch  # type: ignore[import]
    detections: DetectionList = []
    if outputs is None:
        return detections
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    if outputs is None or (hasattr(outputs, "numel") and outputs.numel() == 0):
        return detections
    arr = outputs.cpu().numpy() if hasattr(outputs, "cpu") else np.array(outputs)
    for row in arr:
        if len(row) < 5:
            continue
        x1, y1, x2, y2, conf = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])
        detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, frame_id=frame_id))
    return detections
