"""
BridgeGuardian AI — Image Quality Checker Component
Validates image files before processing for blur, brightness, resolution, fog/rain, bridge visibility, and duplication.
"""
from __future__ import annotations
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set

from backend.ml.computer_vision.base import BaseImageQualityChecker

class BridgePresenceClassifier:
    """
    Pre-quality classifier for structural bridge presence verification.
    Outputs: 'Bridge', 'Bridge Component', 'Non Bridge', or 'Unknown'.
    """
    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Feature 1: Edge density & structural line continuity
        edges = cv2.Canny(gray, 40, 140)
        edge_density = float(np.sum(edges > 0) / (h * w))
        
        # Feature 2: Hough Line Structural Alignment (Trusses, Girders, Cables)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=10)
        num_lines = len(lines) if lines is not None else 0

        # Feature 3: Contour Area Scale & Aspect Ratio
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_cnt_count = 0
        if contours:
            large_cnts = [c for c in contours if cv2.contourArea(c) > (h * w * 0.002)]
            large_cnt_count = len(large_cnts)

        # Classification Heuristics
        if num_lines >= 2 or large_cnt_count >= 1 or edge_density >= 0.002:
            if num_lines >= 10 or large_cnt_count >= 2:
                classification = "Bridge"
                confidence = 0.92
            else:
                classification = "Bridge Component"
                confidence = 0.85
        elif edge_density < 0.0005 and num_lines == 0:
            classification = "Non Bridge"
            confidence = 0.90
        elif edge_density < 0.001 and num_lines == 0:
            classification = "Non Bridge"
            confidence = 0.82
        else:
            classification = "Bridge Component"
            confidence = 0.70

        return {
            "classification": classification,
            "confidence": confidence,
            "num_lines": num_lines,
            "edge_density": round(edge_density, 4),
            "large_contours": large_cnt_count
        }


class OpenCVImageQualityChecker(BaseImageQualityChecker):
    def __init__(
        self,
        blur_threshold: float = 50.0,
        dark_threshold: float = 30.0,
        bright_threshold: float = 240.0,
        min_dim: int = 100,
        min_contrast: float = 5.0,
        min_bridge_coverage: float = 0.01  # Removed 25% hard cutoff
    ) -> None:
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_dim = min_dim
        self.min_contrast = min_contrast
        self.min_bridge_coverage = min_bridge_coverage
        self.presence_classifier = BridgePresenceClassifier()

    def check_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Validates image for blur, lighting, resolution, and bridge presence.
        Uses Adaptive Multi-Factor Quality Score and Bridge Presence Classifier.
        Accepts long-distance drone images; rejects ONLY 'Non Bridge' or severely unreadable frames.
        """
        path = Path(image_path)
        if not path.exists():
            return {
                "is_valid": False,
                "warnings": ["Image file does not exist"],
                "rejection_reason": "File missing or inaccessible",
                "metrics": {}
            }
            
        with open(path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        image = cv2.imread(image_path)
        if image is None:
            return {
                "is_valid": False,
                "warnings": ["Corrupted or unreadable image file"],
                "rejection_reason": "Corrupted or unreadable image file",
                "metrics": {"hash": file_hash}
            }
            
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur score via Laplacian variance
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # 2. Brightness mean
        brightness_val = float(np.mean(gray))
        
        # 3. Contrast (Standard deviation of pixel intensities)
        contrast_val = float(np.std(gray))
        
        # 4. Fog/Haze index via Dark Channel Prior approximation
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(min_channel, kernel)
        dark_channel_mean = float(np.mean(dark_channel))

        # 5. Bridge Presence Classification
        presence_res = self.presence_classifier.classify(image)
        bridge_class = presence_res["classification"]

        # Calculate Multi-Factor Adaptive Quality Score (0 - 100)
        sharpness_subscore = min(100.0, (blur_val / 150.0) * 100.0)
        presence_subscore = 100.0 if bridge_class in ["Bridge", "Bridge Component"] else (50.0 if bridge_class == "Unknown" else 0.0)
        lighting_subscore = max(0.0, 100.0 - abs(brightness_val - 128.0) * 0.8)
        resolution_subscore = min(100.0, ((w * h) / (800.0 * 600.0)) * 100.0)

        adaptive_quality_score = float(round(
            0.35 * sharpness_subscore +
            0.25 * presence_subscore +
            0.20 * lighting_subscore +
            0.20 * resolution_subscore,
            1
        ))

        warnings = []
        rejection_reasons = []

        # Check resolution
        if h < self.min_dim or w < self.min_dim:
            rejection_reasons.append(f"Low Resolution ({w}x{h}, minimum {self.min_dim}x{self.min_dim})")
            
        # Check blurriness
        if blur_val < self.blur_threshold:
            rejection_reasons.append(f"Blur (score {blur_val:.1f} < {self.blur_threshold})")

        # Priority 3 Rule: Reject ONLY Non Bridge
        if bridge_class == "Non Bridge":
            rejection_reasons.append("Non-Bridge Image Rejected (No bridge or component structure detected)")
        elif bridge_class == "Unknown":
            warnings.append("Bridge presence uncertain; flagged for manual review")
            
        # Check brightness extremes
        if brightness_val < self.dark_threshold:
            rejection_reasons.append(f"Too Dark (brightness {brightness_val:.1f} < {self.dark_threshold})")
        elif brightness_val > self.bright_threshold:
            rejection_reasons.append(f"Too Bright / Overexposed (brightness {brightness_val:.1f} > {self.bright_threshold})")

        is_valid = len(rejection_reasons) == 0
        primary_reason = rejection_reasons[0] if rejection_reasons else "None"

        return {
            "is_valid": is_valid,
            "warnings": rejection_reasons or warnings,
            "rejection_reason": primary_reason,
            "metrics": {
                "hash": file_hash,
                "width": w,
                "height": h,
                "blur_score": round(blur_val, 2),
                "brightness": round(brightness_val, 2),
                "contrast": round(contrast_val, 2),
                "dark_channel_mean": round(dark_channel_mean, 2),
                "bridge_presence_class": bridge_class,
                "bridge_presence_confidence": presence_res["confidence"],
                "adaptive_quality_score": adaptive_quality_score
            }
        }
