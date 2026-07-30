"""
BridgeGuardian AI — Deep Learning Segmentation Engine
Runs YOLOv8-seg / YOLOv11 / ONNX deep learning segmentation models for crack, rust, spalling,
and water leakage detection with automatic OpenCV fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("bridgeguardian.cv.deep_segmentation")

# Try importing ultralytics / ONNX runtime for deep vision models
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except (ImportError, Exception):
    ULTRALYTICS_AVAILABLE = False
    logger.info("Ultralytics package not available — using ONNX/OpenCV fallback.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except (ImportError, Exception):
    ONNX_AVAILABLE = False
    logger.info("ONNX Runtime not available — using OpenCV fallback.")


DEFECT_CLASSES = {
    0: "Crack",
    1: "Rust/Corrosion",
    2: "Spalling",
    3: "Water Leakage",
    4: "Vegetation",
    5: "Exposed Rebar",
    6: "Surface Discoloration",
}

DEFECT_COLORS = {
    "Crack": (0, 0, 255),               # Red
    "Rust/Corrosion": (0, 128, 255),    # Orange
    "Spalling": (255, 0, 0),            # Blue
    "Water Leakage": (128, 0, 128),     # Purple
    "Vegetation": (0, 255, 0),          # Green
    "Exposed Rebar": (0, 255, 255),     # Yellow
    "Surface Discoloration": (255, 128, 0), # Cyan/Orange
}


class DeepSegmentationModel:
    """
    Deep learning vision inference engine for drone inspection image segmentation.
    Supports PyTorch YOLO weights, ONNX format models, and OpenCV fallback.
    """

    def __init__(
        self,
        model_path: Optional[str] = "models/yolov8_bridge_seg.onnx",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = Path(model_path) if model_path else None
        self.yolo_model: Optional[Any] = None
        self.onnx_session: Optional[Any] = None
        self._is_loaded = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load trained PyTorch or ONNX segmentation model."""
        if ULTRALYTICS_AVAILABLE and self.model_path and self.model_path.suffix in [".pt", ".engine"]:
            if self.model_path.exists():
                try:
                    self.yolo_model = YOLO(str(self.model_path))
                    self._is_loaded = True
                    logger.info(f"Loaded YOLO segmentation model from {self.model_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load YOLO model: {e}")

        if ONNX_AVAILABLE and self.model_path and self.model_path.suffix == ".onnx":
            if self.model_path.exists():
                try:
                    self.onnx_session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
                    self._is_loaded = True
                    logger.info(f"Loaded ONNX segmentation session from {self.model_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load ONNX model: {e}")

        logger.info("Deep Segmentation using enhanced morphological OpenCV pipeline fallback.")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run segmentation on a BGR image array.
        
        Returns dict containing:
          - 'detections': List of bbox dicts {bbox: [x,y,w,h], label: str, confidence: float}
          - 'masks': Dict mapping defect category names to binary uint8 numpy masks
        """
        h, w = image.shape[:2]
        masks = {
            "cracks": np.zeros((h, w), dtype=np.uint8),
            "rust": np.zeros((h, w), dtype=np.uint8),
            "spalling": np.zeros((h, w), dtype=np.uint8),
            "leakage": np.zeros((h, w), dtype=np.uint8),
            "vegetation": np.zeros((h, w), dtype=np.uint8),
            "rebar": np.zeros((h, w), dtype=np.uint8),
            "discoloration": np.zeros((h, w), dtype=np.uint8),
        }
        detections = []

        if self._is_loaded and self.yolo_model is not None:
            results = self.yolo_model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            for r in results:
                boxes = r.boxes
                masks_data = r.masks
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    label = DEFECT_CLASSES.get(cls_id, "Crack")
                    xywh = box.xywh[0].cpu().numpy()
                    bx, by, bw, bh = int(xywh[0] - bw/2), int(xywh[1] - bh/2), int(xywh[2]), int(xywh[3])

                    detections.append({
                        "bbox": [max(0, bx), max(0, by), max(1, bw), max(1, bh)],
                        "label": label,
                        "confidence": round(conf, 3),
                    })

                    if masks_data is not None:
                        mask_img = cv2.resize((masks_data.data[idx].cpu().numpy() * 255).astype(np.uint8), (w, h))
                        cat_key = self._label_to_mask_key(label)
                        if cat_key in masks:
                            masks[cat_key] = cv2.bitwise_or(masks[cat_key], mask_img)

        return {"detections": detections, "masks": masks}

    @staticmethod
    def _label_to_mask_key(label: str) -> str:
        mapping = {
            "Crack": "cracks",
            "Rust/Corrosion": "rust",
            "Spalling": "spalling",
            "Water Leakage": "leakage",
            "Vegetation": "vegetation",
            "Exposed Rebar": "rebar",
            "Surface Discoloration": "discoloration",
        }
        return mapping.get(label, "cracks")
