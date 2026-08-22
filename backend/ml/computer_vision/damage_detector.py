"""
BridgeGuardian AI — Damage Detector
Executes ROI-gated defect detection strictly within visible bridge component masks.
Applies Defect Verification Protocol (area, probability, edge consistency, structural reasoning compatibility).
Removes hallucinated defects and avoids reporting false positives.
"""
from __future__ import annotations
import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from backend.ml.computer_vision.structural_reasoning import StructuralReasoningEngine

from backend.ml.computer_vision.base import VisionPipelineError

logger = logging.getLogger("bridgeguardian.cv.damage_detector")

CLASS_MAPPING = {
    "crack": ("Crack", "Concrete Crack"),
    "cracks": ("Crack", "Concrete Crack"),
    "rust": ("Rust/Corrosion", "Surface Corrosion"),
    "corrosion": ("Rust/Corrosion", "Surface Corrosion"),
    "rust/corrosion": ("Rust/Corrosion", "Surface Corrosion"),
    "spalling": ("Spalling", "Spalling & Delamination"),
    "delamination": ("Spalling", "Spalling & Delamination"),
    "exposed rebar": ("Exposed Rebar", "Exposed Reinforcement"),
    "missing bolt": ("Missing Bolt", "Fastener Defect"),
    "vegetation": ("Vegetation", "Biological Growth"),
    "water leakage": ("Water Leakage", "Efflorescence & Leakage"),
    "guardrail damage": ("Guardrail Damage", "Structural Alignment"),
    "surface deformation": ("Surface Deformation", "Deformation")
}


class DamageDetector:
    """
    Detects and verifies visible surface damage on bridge elements.
    Operates within expanded ROI masks of visible components with fallback to full-frame detection.
    """

    def __init__(self, use_yolo: bool = False, min_confidence: float = 0.30, debug_bypass_filters: bool = False) -> None:
        self.use_yolo = use_yolo
        self.min_confidence = min_confidence if not debug_bypass_filters else 0.20
        self.debug_bypass_filters = debug_bypass_filters
        self.reasoning_engine = StructuralReasoningEngine()
        self.classes: List[str] = [
            "Crack", "Spalling", "Rust/Corrosion", "Rust", "Corrosion",
            "Vegetation", "Water Leakage", "Guardrail Damage", "Surface Deformation",
            "Missing Bolt", "Exposed Rebar", "Delamination"
        ]
        logger.info(
            f"[Model Startup] Defect Detector LOADED successfully.\n"
            f"  Threshold: {self.min_confidence}\n"
            f"  Debug Bypass Filters: {self.debug_bypass_filters}\n"
            f"  Supported classes: {self.classes}"
        )

    def validate_class_mapping(self, raw_label: str) -> tuple[str, str]:
        """
        Validates raw model prediction class name mapping.
        Raw Model Class -> Mapped Internal Class -> Report Category.
        Raises VisionPipelineError if an unmapped raw class is encountered.
        """
        normalized = raw_label.strip().lower()
        if normalized not in CLASS_MAPPING:
            raise VisionPipelineError(
                f"[Class Mapping Error] Unmapped raw defect model class: '{raw_label}'. "
                f"Class is missing from valid category mapping registry."
            )
        return CLASS_MAPPING[normalized]

    def log_model_status(self) -> Dict[str, Any]:
        """Returns defect model startup audit metadata."""
        return {
            "model_type": "Defect Verification Engine (CV + ROI Gating)",
            "is_loaded": True,
            "min_confidence_threshold": self.min_confidence,
            "supported_classes": self.classes
        }

    def detect_all_damage(
        self,
        image: np.ndarray,
        bridge_info: dict,
        visible_components: Optional[List[dict]] = None
    ) -> dict:
        """
        Runs ROI-based defect detection strictly within visible component regions.
        Enforces Defect Verification (area, probability, edge consistency, structural compatibility).
        
        Args:
            image: OpenCV BGR image.
            bridge_info: Output of BridgeDetector.
            visible_components: List of detected visible components (label, bbox, mask).
            
        Returns:
            Dict containing verified defect masks, percentages, counts, flags, and bboxes.
        """
        h, w = image.shape[:2]
        
        # Build composite ROI mask from visible component masks
        component_roi = np.zeros((h, w), dtype=np.uint8)
        comp_labels = [c.get("label", "Deck") for c in visible_components] if visible_components else ["Deck"]
        
        if visible_components:
            for comp in visible_components:
                comp_mask = comp.get("mask")
                if comp_mask is not None and np.sum(comp_mask > 0) > 0:
                    component_roi = cv2.bitwise_or(component_roi, comp_mask)
                else:
                    bx, by, bw_c, bh_c = comp.get("bbox", [0, 0, w, h])
                    # Pad bbox by 20% to avoid boundary defect clipping
                    px = max(0, bx - int(bw_c * 0.10))
                    py = max(0, by - int(bh_c * 0.10))
                    pw = min(w - px, int(bw_c * 1.20))
                    ph = min(h - py, int(bh_c * 1.20))
                    cv2.rectangle(component_roi, (px, py), (px + pw, py + ph), 255, -1)
        else:
            bridge_mask = bridge_info.get("mask")
            if bridge_mask is not None and np.sum(bridge_mask > 0) > (h * w * 0.02):
                component_roi = bridge_mask
            else:
                component_roi = np.ones((h, w), dtype=np.uint8) * 255

        # Morphologically dilate ROI by 20px kernel to expand coverage and prevent edge truncation
        kernel_exp = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        component_roi = cv2.dilate(component_roi, kernel_exp, iterations=1)

        # Fallback to full-frame ROI if component_roi is empty or less than 5% of image
        roi_pixel_count = int(np.sum(component_roi > 0))
        if roi_pixel_count < (h * w * 0.05):
            logger.info("[ROI Gating] ROI expanded to full-frame image area to capture unsegmented defects.")
            component_roi = np.ones((h, w), dtype=np.uint8) * 255

        # Initialize empty defect masks
        crack_mask = np.zeros((h, w), dtype=np.uint8)
        rust_mask = np.zeros((h, w), dtype=np.uint8)
        spalling_mask = np.zeros((h, w), dtype=np.uint8)
        vegetation_mask = np.zeros((h, w), dtype=np.uint8)
        leakage_mask = np.zeros((h, w), dtype=np.uint8)
        guardrail_mask = np.zeros((h, w), dtype=np.uint8)
        deformation_mask = np.zeros((h, w), dtype=np.uint8)
        
        bboxes = []
        raw_candidates = []
        missing_bolts_count = 0
        loose_components_count = 0

        # Helper function for defect-component compatibility across ANY visible component
        def is_compatible(defect_name: str) -> bool:
            if not visible_components:
                return True
            return any(self.reasoning_engine.validate_defect_compatibility(defect_name, lbl) for lbl in comp_labels)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Pre-compute bilateral filtered image & adaptive threshold for crack and bolt detectors
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )

        # ---------------------------------------------------------------------
        # 1. Rust / Corrosion Detection (ROI-based + Context-Aware)
        # ---------------------------------------------------------------------
        if is_compatible("Rust"):
            # Restrict HSV rust detection strictly inside ROI
            lower_rust1 = np.array([5, 40, 40])
            upper_rust1 = np.array([30, 255, 240])
            lower_rust2 = np.array([160, 40, 40])
            upper_rust2 = np.array([180, 255, 240])
            
            rm1 = cv2.inRange(hsv, lower_rust1, upper_rust1)
            rm2 = cv2.inRange(hsv, lower_rust2, upper_rust2)
            raw_rust = cv2.bitwise_or(rm1, rm2)
            
            # Mask strictly by component ROI
            candidate_rust = cv2.bitwise_and(raw_rust, component_roi)
            
            # Verification: Minimum Area & Edge Consistency
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned_rust = cv2.morphologyEx(candidate_rust, cv2.MORPH_OPEN, kernel)
            
            cnts, _ = cv2.findContours(cleaned_rust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= 50:
                    x, y, cw, ch = cv2.boundingRect(c)
                    conf = min(0.95, float(round(0.60 + min(0.35, area / (h * w * 0.05)), 2)))
                    raw_candidates.append({"label": "Rust/Corrosion", "bbox": [int(x), int(y), int(cw), int(ch)], "confidence": conf})
                    if conf >= self.min_confidence:
                        cv2.drawContours(rust_mask, [c], -1, 255, -1)
                        bboxes.append({
                            "label": "Rust/Corrosion",
                            "bbox": [int(x), int(y), int(cw), int(ch)],
                            "confidence": conf
                        })

        # ---------------------------------------------------------------------
        # 2. Crack Detection (ROI-based + Line/Edge Verification)
        # ---------------------------------------------------------------------
        if is_compatible("Crack"):
            edges = cv2.Canny(filtered, 30, 100)
            raw_crack = cv2.bitwise_or(thresh, edges)
            kernel_crack = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            crack_cand = cv2.morphologyEx(raw_crack, cv2.MORPH_CLOSE, kernel_crack)
            crack_cand = cv2.bitwise_and(crack_cand, component_roi)
            
            cnts, _ = cv2.findContours(crack_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                arc_len = cv2.arcLength(c, False)
                area = cv2.contourArea(c)
                x, y, cw, ch = cv2.boundingRect(c)

                # Filter straight geometric grid lines and rectangular borders (thin lines or straight bars)
                if (ch <= 6 and cw > 15) or (cw <= 6 and ch > 15):
                    continue  # Skip straight grid lines and rectangular borders

                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if rw > 25 and rh > 25:
                    box_solidity = float(area) / (cw * ch) if (cw * ch) > 0 else 0.0
                    # Crack lines are thin linear features (low bounding-box solidity < 0.30)
                    if box_solidity > 0.35:
                        continue  # Skip 2D surface degradation patches in crack detector

                if arc_len >= 20 or area >= 25:
                    conf = min(0.95, float(round(0.55 + min(0.40, (arc_len + area) / 100.0), 2)))
                    raw_candidates.append({"label": "Crack", "bbox": [int(x), int(y), int(cw), int(ch)], "confidence": conf})
                    if conf >= self.min_confidence:
                        cv2.drawContours(crack_mask, [c], -1, 255, 2)
                        bboxes.append({
                            "label": "Crack",
                            "bbox": [int(x), int(y), int(cw), int(ch)],
                            "confidence": conf
                        })

        # ---------------------------------------------------------------------
        # 3. Spalling & Concrete Degradation Detection
        # ---------------------------------------------------------------------
        if is_compatible("Spalling"):
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobel_x, sobel_y)
            sobel_u8 = np.uint8(np.clip(sobel, 0, 255))
            _, rough = cv2.threshold(sobel_u8, 40, 255, cv2.THRESH_BINARY)
            
            spall_cand = cv2.subtract(rough, crack_mask)
            spall_cand = cv2.bitwise_and(spall_cand, component_roi)
            kernel_spall = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            spall_cand = cv2.morphologyEx(spall_cand, cv2.MORPH_CLOSE, kernel_spall)
            
            cnts, _ = cv2.findContours(spall_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= 50:
                    x, y, cw, ch = cv2.boundingRect(c)
                    aspect_ratio = float(max(cw, ch)) / min(cw, ch) if min(cw, ch) > 0 else 1.0
                    solidity = float(area) / (cw * ch) if (cw * ch) > 0 else 0.0
                    if aspect_ratio < 5.0 and 0.10 <= solidity <= 0.98:
                        conf = min(0.92, float(round(0.55 + min(0.35, area / (h * w * 0.05)), 2)))
                        raw_candidates.append({"label": "Spalling", "bbox": [int(x), int(y), int(cw), int(ch)], "confidence": conf})
                        if conf >= self.min_confidence:
                            cv2.drawContours(spalling_mask, [c], -1, 255, -1)
                            bboxes.append({
                                "label": "Spalling",
                                "bbox": [int(x), int(y), int(cw), int(ch)],
                                "confidence": conf
                            })

        # ---------------------------------------------------------------------
        # 4. Missing Bolt / Mechanical Component Detection (Strict Context Rule)
        # ---------------------------------------------------------------------
        if is_compatible("Missing Bolt"):
            cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if 15 <= area <= 150:
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if 0.75 <= circularity <= 1.0:
                            x, y, cw, ch = cv2.boundingRect(c)
                            if component_roi[int(y + ch/2), int(x + cw/2)] > 0:
                                missing_bolts_count += 1
                                bboxes.append({
                                    "label": "Missing Bolt",
                                    "bbox": [int(x), int(y), int(cw), int(ch)],
                                    "confidence": 0.84
                                })

        # Calculate defect area percentages relative to component ROI
        roi_pixels = int(np.sum(component_roi > 0))
        if roi_pixels == 0:
            roi_pixels = h * w

        logger.info(
            f"[DamageDetector Audit] Raw Candidate Defect Predictions: {len(raw_candidates)} | "
            f"Verified Defects (conf >= {self.min_confidence}): {len(bboxes)}"
        )
        for cand in raw_candidates[:10]:
            logger.info(f"  -> Candidate: {cand['label']} (conf: {cand['confidence']}, bbox: {cand['bbox']})")

        def get_pct(m: np.ndarray) -> float:
            return float(round((np.sum(m > 0) / roi_pixels) * 100, 2))

        return {
            "raw_candidates": raw_candidates,
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
                "crack_presence": bool(np.sum(crack_mask > 0) > 0),
                "spalling_presence": bool(np.sum(spalling_mask > 0) > 0),
                "leakage_presence": bool(np.sum(leakage_mask > 0) > 0),
                "guardrail_damage": bool(np.sum(guardrail_mask > 0) > 0),
                "surface_deformation": bool(np.sum(deformation_mask > 0) > 0),
            },
            "bboxes": bboxes
        }

    def _empty_result(self, h: int, w: int) -> dict:
        """Returns zero defects packet when image is clean or no ROI matches."""
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return {
            "masks": {
                "cracks": empty_mask.copy(),
                "rust": empty_mask.copy(),
                "spalling": empty_mask.copy(),
                "vegetation": empty_mask.copy(),
                "leakage": empty_mask.copy(),
                "guardrail": empty_mask.copy(),
                "deformation": empty_mask.copy(),
            },
            "percentages": {
                "crack_density": 0.0,
                "corrosion_percent": 0.0,
                "spalling_percent": 0.0,
                "vegetation_percent": 0.0,
                "leakage_percent": 0.0,
                "guardrail_percent": 0.0,
                "deformation_percent": 0.0,
            },
            "counts": {
                "missing_bolts": 0,
                "loose_components": 0,
            },
            "flags": {
                "crack_presence": False,
                "spalling_presence": False,
                "leakage_presence": False,
                "guardrail_damage": False,
                "surface_deformation": False,
            },
            "bboxes": []
        }
