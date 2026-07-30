"""
BridgeGuardian AI — Image Measurements
Estimates defect dimensions (length, width) and structural parameters (bridge tilt) in physical units.
"""
from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger("bridgeguardian.cv.image_measurements")

class ImageMeasurements:
    """
    Translates pixel-level detections into physical measurements (mm, degrees).
    """

    def __init__(self, default_pixel_to_mm: float = 0.5) -> None:
        """
        Args:
            default_pixel_to_mm: Conversion factor (mm per pixel).
                                 E.g., at 5m distance, 1 pixel is ~0.5mm.
        """
        self.pixel_to_mm = default_pixel_to_mm

    def estimate_crack_dimensions(self, crack_mask: np.ndarray) -> tuple[float, float]:
        """
        Estimates the approximate total length and maximum width of cracks in mm.
        
        Uses the distance transform to find the thickness of crack regions, and
        contour lengths for the crack extent.
        
        Returns:
            Tuple (approximate_length_mm, approximate_width_mm)
        """
        if np.sum(crack_mask > 0) == 0:
            return 0.0, 0.0

        # Estimate Length: Sum of arc lengths of all crack contours
        contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_length_px = 0.0
        for c in contours:
            length = cv2.arcLength(c, False)
            # Since contours wrap around the crack, the skeleton length is roughly half the perimeter
            total_length_px += (length / 2.0)

        total_length_mm = round(total_length_px * self.pixel_to_mm, 2)

        # Estimate Width: Use Distance Transform to find max thickness
        dist_transform = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
        _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
        
        # Max thickness (diameter) is twice the distance transform radius
        max_width_px = max_val * 2.0
        max_width_mm = round(max_width_px * self.pixel_to_mm, 2)
        
        # Ensure values are realistic (width should not be larger than length if there are cracks)
        if total_length_mm > 0 and max_width_mm == 0:
            max_width_mm = 0.1

        return float(total_length_mm), float(max_width_mm)

    def estimate_bridge_tilt(self, structural_lines: list[tuple[tuple[int, int], tuple[int, int]]]) -> float:
        """
        Estimates the approximate bridge tilt in degrees based on the orientation
        of horizontal-ish structural lines.
        
        Returns:
            Tilt angle in degrees relative to the horizontal (0°).
        """
        if not structural_lines:
            return 0.0

        angles = []
        for (x1, y1), (x2, y2) in structural_lines:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            # Calculate angle in degrees
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            # Since we only want deviation from horizontal (0 degrees),
            # normalize angle to [-45, 45]
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
                
            if abs(angle) < 45: # Horizontal-ish structural element
                angles.append(angle)

        if not angles:
            return 0.0

        # Average horizontal tilt deviation
        avg_tilt = float(np.mean(angles))
        return float(round(avg_tilt, 2))

    def calculate_defect_area_m2(self, mask: np.ndarray) -> float:
        """
        Calculates physical defect area in square meters (m^2) using pixel_to_mm conversion.
        """
        pixel_count = int(np.sum(mask > 0))
        # 1 px = pixel_to_mm (mm) -> area of 1 px in m^2 is (pixel_to_mm * 1e-3)^2
        area_per_px_m2 = (self.pixel_to_mm * 1e-3) ** 2
        total_area_m2 = round(pixel_count * area_per_px_m2, 4)
        return float(total_area_m2)

    def calculate_severity_score(
        self,
        masks: dict[str, np.ndarray],
        crack_width_mm: float,
        total_image_pixels: int,
    ) -> tuple[float, str]:
        """
        Computes a continuous defect severity index (0.0 to 100.0) and severity rating.
        """
        total_pixels = max(1, total_image_pixels)
        crack_pct = (np.sum(masks.get("cracks", np.zeros(1)) > 0) / total_pixels) * 100.0
        rust_pct = (np.sum(masks.get("rust", np.zeros(1)) > 0) / total_pixels) * 100.0
        spalling_pct = (np.sum(masks.get("spalling", np.zeros(1)) > 0) / total_pixels) * 100.0
        rebar_pct = (np.sum(masks.get("rebar", np.zeros(1)) > 0) / total_pixels) * 100.0

        # Weighted severity index formula
        score = (
            (crack_pct * 3.5)
            + (rust_pct * 2.0)
            + (spalling_pct * 4.0)
            + (rebar_pct * 5.0)
            + (min(10.0, crack_width_mm) * 3.0)
        )
        score = float(round(min(100.0, score), 1))

        if score >= 60.0 or rebar_pct > 0.5 or crack_width_mm >= 4.0:
            rating = "Critical"
        elif score >= 35.0 or spalling_pct > 1.0 or crack_width_mm >= 2.0:
            rating = "Severe"
        elif score >= 15.0 or rust_pct > 2.0 or crack_width_mm >= 0.8:
            rating = "Moderate"
        else:
            rating = "Minor"

        return score, rating

    def generate_density_heatmap(self, masks: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        """
        Generates a smooth kernel density heatmap (0-255 uint8) from all defect masks.
        """
        h, w = shape[:2]
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for m in masks.values():
            if m is not None and m.shape[:2] == (h, w):
                union_mask = cv2.bitwise_or(union_mask, m)

        if np.sum(union_mask > 0) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        blur_k = max(21, (min(h, w) // 15) | 1)
        density = cv2.GaussianBlur(union_mask.astype(float), (blur_k, blur_k), 0)
        max_d = np.max(density)

        if max_d > 0:
            density_norm = (density / max_d * 255.0).astype(np.uint8)
        else:
            density_norm = np.zeros((h, w), dtype=np.uint8)

        return density_norm

