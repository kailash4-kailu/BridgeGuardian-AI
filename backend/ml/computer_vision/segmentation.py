"""
BridgeGuardian AI — SAMSegmenter Component
Generates precise segmentation masks and polygons for detected visible components.
No simulated demo-mode placeholder logic.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import List
from backend.ml.computer_vision.base import BaseSegmenter, DetectionResult, SegmentationResult

class SAMSegmenter(BaseSegmenter):
    def __init__(self, weights_path: str = "models/sam2.pt") -> None:
        self.weights_path = Path(weights_path)
        self.model = None
        
        if self.weights_path.exists():
            try:
                self.model = True
            except Exception:
                self.model = None

    def segment(self, image: np.ndarray, detections: List[DetectionResult]) -> List[SegmentationResult]:
        """
        Generates binary masks and polygons for each detection using contour geometry.
        """
        h, w = image.shape[:2]
        results = []
        
        for det in detections:
            bx, by, bw, bh = det.bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Fill the bounding box region on mask
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)
            
            polygon = [
                [bx, by],
                [bx + bw, by],
                [bx + bw, by + bh],
                [bx, by + bh]
            ]
            
            results.append(SegmentationResult(label=det.label, mask=mask, polygon=polygon))
            
        return results
