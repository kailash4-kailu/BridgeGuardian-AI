"""
BridgeGuardian AI — Damage Detector
Detects surface defects (cracks, rust, spalling, vegetation, leakage, missing bolts, loose components, guard rail damage, surface deformation).
"""
from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger("bridgeguardian.cv.damage_detector")

class DamageDetector:
    """
    Detects and segments visible surface damage on bridge elements.
    Uses multi-spectral thresholding, edge-contour mapping, and morphology.
    """

    def __init__(self, use_yolo: bool = False) -> None:
        self.use_yolo = use_yolo

    def detect_all_damage(self, image: np.ndarray, bridge_info: dict) -> dict:
        """
        Runs multiple specialized visual detectors for various damage types.
        
        Args:
            image: OpenCV BGR image.
            bridge_info: Output of BridgeDetector.
            
        Returns:
            Dict containing overlays, bounding boxes, masks, and percentages for defects.
        """
        h, w = image.shape[:2]
        bridge_mask = bridge_info.get("mask")
        if bridge_mask is None or np.sum(bridge_mask > 0) < (h * w * 0.02):
            bridge_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Initialize masks
        crack_mask = np.zeros((h, w), dtype=np.uint8)
        rust_mask = np.zeros((h, w), dtype=np.uint8)
        spalling_mask = np.zeros((h, w), dtype=np.uint8)
        vegetation_mask = np.zeros((h, w), dtype=np.uint8)
        leakage_mask = np.zeros((h, w), dtype=np.uint8)
        guardrail_mask = np.zeros((h, w), dtype=np.uint8)
        deformation_mask = np.zeros((h, w), dtype=np.uint8)
        
        bboxes = [] # List of dicts: {"label": str, "bbox": [x, y, w, h], "confidence": float}
        
        # 1. Vegetation Detection (HSV Green filter)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        vegetation_mask = cv2.bitwise_and(green_mask, bridge_mask)
        
        # 2. Rust / Corrosion Detection (HSV Orange/Reddish-Brown filter)
        # Rust has characteristic brown/red/orange colors
        lower_rust1 = np.array([0, 30, 30])
        upper_rust1 = np.array([30, 255, 255])
        lower_rust2 = np.array([160, 30, 30])
        upper_rust2 = np.array([180, 255, 255])
        rust_mask1 = cv2.inRange(hsv, lower_rust1, upper_rust1)
        rust_mask2 = cv2.inRange(hsv, lower_rust2, upper_rust2)
        rust_mask_all = cv2.bitwise_or(rust_mask1, rust_mask2)
        rust_mask = cv2.bitwise_and(rust_mask_all, bridge_mask)
        
        # Clean rust mask using morphological opening/closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Crack Detection
        # Bilateral filter to preserve edges while smoothing concrete texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding to find local dark pixels (cracks)
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )
        
        # Apply Canny to find edges
        edges = cv2.Canny(filtered, 30, 100)
        
        # Intersect adaptive threshold and edges to get thin crack candidates
        crack_candidates = cv2.bitwise_and(thresh, edges)
        crack_mask = cv2.bitwise_and(crack_candidates, bridge_mask)
        
        # Keep raw crack candidates to preserve fine lines
        crack_mask = crack_mask
        
        # 4. Water Leakage Stains (Dark vertical streaks)
        # Thresholding for dark regions
        _, dark_thresh = cv2.threshold(filtered, 80, 255, cv2.THRESH_BINARY_INV)
        # Apply vertical kernel morph to extract vertical lines
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        leakage_candidates = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, vert_kernel)
        leakage_mask = cv2.bitwise_and(leakage_candidates, bridge_mask)
        
        # 5. Surface Spalling (Irregular concrete textures)
        # Concrete spalling shows as loss of surface and high local variation
        # Calculate Local Variance using a Sobel filter to find highly textured, rough areas
        sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobel_x, sobel_y)
        sobel_uint8 = np.uint8(np.clip(sobel, 0, 255))
        # Spalling has high edge density but isn't as thin as cracks
        _, rough_texture = cv2.threshold(sobel_uint8, 80, 255, cv2.THRESH_BINARY)
        # Exclude cracks from spalling mask
        spalling_candidates = cv2.subtract(rough_texture, crack_mask)
        spalling_mask = cv2.bitwise_and(spalling_candidates, bridge_mask)
        spalling_mask = cv2.morphologyEx(spalling_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        
        # Usually appear as high-contrast structural circles/squares on bridge joints
        # Look for small circular contours on adaptive threshold to get filled area
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        missing_bolts_count = 0
        loose_components_count = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if 10 <= area <= 200:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Circular shapes represent missing bolt holes or loose bolts
                if 0.7 < circularity <= 1.0:
                    x, y, cw, ch = cv2.boundingRect(c)
                    # Check if near bridge structures
                    if bridge_mask[y, x] > 0:
                        missing_bolts_count += 1
                        bboxes.append({
                            "label": "Missing Bolt",
                            "bbox": [int(x), int(y), int(cw), int(ch)],
                            "confidence": 0.82
                        })
                elif 0.3 <= circularity <= 0.7:
                    # Irregular small components
                    x, y, cw, ch = cv2.boundingRect(c)
                    if bridge_mask[y, x] > 0:
                        loose_components_count += 1
                        bboxes.append({
                            "label": "Loose Component",
                            "bbox": [int(x), int(y), int(cw), int(ch)],
                            "confidence": 0.74
                        })
                        
        # 7. Guard Rail Damage & Surface Deformation
        # Detect long horizontal guard rails or deck lines, and measure deviation
        lines = bridge_info.get("structural_lines", [])
        guard_rail_issues = False
        deformation_issues = False
        
        # If any major structural lines are highly segmented or deviate from straight path,
        # flag as guard rail damage or surface deformation.
        # Find horizontal lines in the top 30% of the bridge bounding box for guardrail
        bx, by, bw, bh = bridge_info.get("bbox", (0, 0, w, h))
        top_y_limit = by + int(bh * 0.35)
        
        for line in lines:
            (x1, y1), (x2, y2) = line
            # Guardrail check
            if y1 < top_y_limit and y2 < top_y_limit:
                # If guardrail is broken/highly sloped or short
                line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 10 and line_len < w // 5:
                    guard_rail_issues = True
                    cv2.line(guardrail_mask, (x1, y1), (x2, y2), 255, 3)
                    bboxes.append({
                        "label": "Guard Rail Damage",
                        "bbox": [min(x1, x2), min(y1, y2), abs(x2 - x1) + 4, abs(y2 - y1) + 4],
                        "confidence": 0.78
                    })
            else:
                # Surface deformation check (large deviation in bridge deck flatness)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 2 < angle < 15: # slightly tilted but should be flat
                    deformation_issues = True
                    cv2.line(deformation_mask, (x1, y1), (x2, y2), 255, 4)
                    bboxes.append({
                        "label": "Surface Deformation",
                        "bbox": [min(x1, x2), min(y1, y2), abs(x2 - x1) + 4, abs(y2 - y1) + 4],
                        "confidence": 0.71
                    })

        # Calculate area percentages relative to the bridge region
        bridge_area_pixels = np.sum(bridge_mask > 0)
        if bridge_area_pixels == 0:
            bridge_area_pixels = h * w
            
        def get_pct(mask):
            return float(round((np.sum(mask > 0) / bridge_area_pixels) * 100, 2))

        # Build bounding boxes for large defect areas
        for label, mask, conf in [("Rust/Corrosion", rust_mask, 0.88), 
                                  ("Spalling", spalling_mask, 0.79), 
                                  ("Water Leakage", leakage_mask, 0.81), 
                                  ("Vegetation", vegetation_mask, 0.85)]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area > (h * w * 0.002): # only box significant areas
                    x, y, cw, ch = cv2.boundingRect(c)
                    bboxes.append({
                        "label": label,
                        "bbox": [int(x), int(y), int(cw), int(ch)],
                        "confidence": conf
                    })

        # Also add bounding boxes for major cracks
        crack_cnts, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in crack_cnts:
            if cv2.arcLength(c, True) > 30:
                x, y, cw, ch = cv2.boundingRect(c)
                bboxes.append({
                    "label": "Crack",
                    "bbox": [int(x), int(y), int(cw), int(ch)],
                    "confidence": 0.90
                })

        return {
            "masks": {
                "cracks": crack_mask,
                "rust": rust_mask,
                "spalling": spalling_mask,
                "vegetation": vegetation_mask,
                "leakage": leakage_mask,
                "guardrail": guardrail_mask,
                "deformation": deformation_mask,
            },
            "percentages": {
                "crack_density": get_pct(crack_mask),
                "corrosion_percent": get_pct(rust_mask),
                "spalling_percent": get_pct(spalling_mask),
                "vegetation_percent": get_pct(vegetation_mask),
                "leakage_percent": get_pct(leakage_mask),
                "guardrail_percent": get_pct(guardrail_mask),
                "deformation_percent": get_pct(deformation_mask),
            },
            "counts": {
                "missing_bolts": missing_bolts_count,
                "loose_components": loose_components_count,
            },
            "flags": {
                "crack_presence": bool(np.sum(crack_mask > 0) > 10),
                "spalling_presence": bool(np.sum(spalling_mask > 0) > 50),
                "leakage_presence": bool(np.sum(leakage_mask > 0) > 50),
                "guardrail_damage": guard_rail_issues,
                "surface_deformation": deformation_issues,
            },
            "bboxes": bboxes
        }
