"""
BridgeGuardian AI — Image Quality Enhancer
Applies CLAHE contrast enhancement, exposure correction, bilateral adaptive denoising,
and unsharp mask sharpening for drone inspection imagery.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger("bridgeguardian.cv.image_enhancer")


class ImageEnhancer:
    """
    Automated preprocessing pipeline for drone inspection imagery:
      - CLAHE contrast enhancement on LAB color space L-channel
      - Exposure & gamma correction for shadow and glare mitigation
      - Adaptive bilateral denoising to preserve crisp crack boundaries
      - High-frequency unsharp mask sharpening for hairline crack extraction
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        denoise_d: int = 5,
        sharpen_strength: float = 0.5,
    ) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
        self.denoise_d = denoise_d
        self.sharpen_strength = sharpen_strength

    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Enhance BGR image quality for downstream defect segmentation.
        
        Args:
            image: BGR numpy image array.
            
        Returns:
            Tuple of (enhanced_image, applied_actions_dict)
        """
        if image is None or image.size == 0:
            return image, {}

        enhanced = image.copy()
        actions = {"clahe": False, "exposure_corrected": False, "denoised": False, "sharpened": False}

        # 1. Exposure and Brightness normalization
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 80.0 or mean_brightness > 200.0:
            # Apply gamma correction
            gamma = 1.4 if mean_brightness < 80.0 else 0.7
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
            actions["exposure_corrected"] = True

        # 2. CLAHE Contrast Enhancement in LAB space
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        actions["clahe"] = True

        # 3. Bilateral Adaptive Denoising (Preserves sharp crack edges)
        try:
            enhanced = cv2.bilateralFilter(enhanced, d=self.denoise_d, sigmaColor=50, sigmaSpace=50)
            actions["denoised"] = True
        except Exception as e:
            logger.warning(f"Bilateral filtering skipped: {e}")

        # 4. Unsharp Mask Sharpening
        if self.sharpen_strength > 0:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.0 + self.sharpen_strength, gaussian, -self.sharpen_strength, 0)
            actions["sharpened"] = True

        return enhanced, actions
