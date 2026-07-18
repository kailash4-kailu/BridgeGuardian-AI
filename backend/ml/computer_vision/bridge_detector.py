"""
BridgeGuardian AI — Bridge Detector
Identifies the bridge structure, deck boundaries, and regions of interest (ROI) in drone images.
"""
from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger("bridgeguardian.cv.bridge_detector")

class BridgeDetector:
    """
    Detects the bridge structure in RGB drone images.
    Uses Hough line transform and shape contour analyses to identify the bridge deck,
    piers, or structural boundaries.
    """

    def __init__(self, use_yolo: bool = False) -> None:
        self.use_yolo = use_yolo

    def detect_bridge(self, image: np.ndarray) -> dict:
        """
        Detects the main bridge structure in the image.
        
        Args:
            image: OpenCV image in BGR format.
            
        Returns:
            Dict containing:
                - detected: bool
                - bbox: tuple (x, y, w, h) of the main bridge region
                - mask: binary mask of the bridge structure
                - confidence: float
                - structural_lines: list of tuples ((x1,y1), (x2,y2)) matching deck boundaries
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Smooth image to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Use Hough Lines to find main structural lines (usually horizontal or slightly tilted)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=w // 4, maxLineGap=20)
        
        bridge_lines = []
        if lines is not None:
            for line in lines:
                flat = line.flatten()
                if len(flat) == 4:
                    x1, y1, x2, y2 = flat
                    # Filter for lines that are close to horizontal (deck) or vertical (piers)
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 30 or angle > 60: # horizontal-ish or vertical-ish
                        bridge_lines.append(((int(x1), int(y1)), (int(x2), int(y2))))

        # Find largest contours representing structural components
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bridge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Heuristic: the bridge is usually the largest structural group in the center
        large_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.01)]
        
        if large_contours:
            cv2.drawContours(bridge_mask, large_contours, -1, 255, -1)
            # Find the bounding box of the combined large contours
            all_pts = np.vstack(large_contours)
            x, y, bbox_w, bbox_h = cv2.boundingRect(all_pts)
            bbox = (x, y, bbox_w, bbox_h)
            detected = True
            confidence = 0.85
        else:
            # Fallback if no prominent structures detected: assume bridge is in the center 80%
            x, y = int(w * 0.1), int(h * 0.1)
            bbox_w, bbox_h = int(w * 0.8), int(h * 0.8)
            bbox = (x, y, bbox_w, bbox_h)
            cv2.rectangle(bridge_mask, (x, y), (x + bbox_w, y + bbox_h), 255, -1)
            detected = True
            confidence = 0.50
            
        return {
            "detected": detected,
            "bbox": bbox,
            "mask": bridge_mask,
            "confidence": confidence,
            "structural_lines": bridge_lines
        }
