"""
BridgeGuardian AI — OpenCVImageQualityChecker Component
Validates image files before processing for blur, brightness, resolution, and duplication.
"""
from __future__ import annotations
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
from backend.ml.computer_vision.base import BaseImageQualityChecker

class OpenCVImageQualityChecker(BaseImageQualityChecker):
    def __init__(
        self,
        blur_threshold: float = 100.0,
        dark_threshold: float = 50.0,
        bright_threshold: float = 220.0,
        min_dim: int = 300
    ) -> None:
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_dim = min_dim

    def check_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Check if the image is blurry, too dark, too bright, or very low resolution.
        Returns a dict with 'is_valid', 'metrics', and list of 'warnings'.
        """
        path = Path(image_path)
        if not path.exists():
            return {
                "is_valid": False,
                "warnings": [f"Image file does not exist at '{image_path}'"],
                "metrics": {}
            }
            
        # Read file bytes to calculate hash
        with open(path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        # Load image in grayscale
        image = cv2.imread(image_path)
        if image is None:
            return {
                "is_valid": False,
                "warnings": ["Failed to load image (corrupted or unsupported format)"],
                "metrics": {"hash": file_hash}
            }
            
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate stats
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_val = float(np.mean(gray))
        
        warnings = []
        
        # Check resolution
        if h < self.min_dim or w < self.min_dim:
            warnings.append(f"Very low resolution: {w}x{h} (minimum size {self.min_dim}x{self.min_dim})")
            
        # Check blurriness
        if blur_val < self.blur_threshold:
            warnings.append(f"Image is blurry (variance {blur_val:.1f} < threshold {self.blur_threshold})")
            
        # Check brightness
        if brightness_val < self.dark_threshold:
            warnings.append(f"Image is too dark (average brightness {brightness_val:.1f} < threshold {self.dark_threshold})")
        elif brightness_val > self.bright_threshold:
            warnings.append(f"Image is too bright (average brightness {brightness_val:.1f} > threshold {self.bright_threshold})")
            
        is_valid = len(warnings) == 0
        
        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "metrics": {
                "hash": file_hash,
                "width": w,
                "height": h,
                "blur_score": round(blur_val, 2),
                "brightness": round(brightness_val, 2)
            }
        }
