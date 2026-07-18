"""
BridgeGuardian AI — Image Feature Extractor
Extracts damage statistics and dimensional measurements from drone images.
"""
from __future__ import annotations

import logging
import cv2
import numpy as np

from backend.ml.computer_vision.bridge_detector import BridgeDetector
from backend.ml.computer_vision.damage_detector import DamageDetector
from backend.ml.computer_vision.image_measurements import ImageMeasurements
from backend.ml.computer_vision.base import BaseFeatureExtractor, DetectionResult, SegmentationResult

logger = logging.getLogger("bridgeguardian.cv.feature_extractor")

class ImageFeatureExtractor:
    """
    Orchestrates the computer vision pipeline to extract structural and defect features
    from drone images.
    """

    def __init__(self, use_yolo: bool = False, pixel_to_mm: float = 0.5) -> None:
        self.bridge_detector = BridgeDetector(use_yolo=use_yolo)
        self.damage_detector = DamageDetector(use_yolo=use_yolo)
        self.measurements = ImageMeasurements(default_pixel_to_mm=pixel_to_mm)

    def extract_features(self, image_path: str) -> tuple[dict, dict]:
        """
        Loads an image, processes it, and extracts feature metrics and raw masks.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {image_path}")

        h, w = image.shape[:2]
        
        # 1. Detect Bridge Structure
        bridge_info = self.bridge_detector.detect_bridge(image)
        bridge_mask = bridge_info["mask"]

        # 2. Detect all Damage
        damage_info = self.damage_detector.detect_all_damage(image, bridge_info)
        masks = damage_info["masks"]
        percentages = damage_info["percentages"]
        counts = damage_info["counts"]
        bboxes = damage_info["bboxes"]

        # 3. Compute Physical Measurements
        crack_len_mm, crack_width_mm = self.measurements.estimate_crack_dimensions(masks["cracks"])
        tilt_deg = self.measurements.estimate_bridge_tilt(bridge_info["structural_lines"])

        # 4. Compute overall damage area percentage (union of all defect masks)
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for m in masks.values():
            union_mask = cv2.bitwise_or(union_mask, m)
            
        bridge_area_pixels = np.sum(bridge_mask > 0)
        if bridge_area_pixels == 0:
            bridge_area_pixels = h * w
            
        total_damage_pixels = np.sum(cv2.bitwise_and(union_mask, bridge_mask) > 0)
        damage_area_percent = float(round((total_damage_pixels / bridge_area_pixels) * 100, 2))

        features = {
            "crack_density": percentages["crack_density"],
            "crack_length": crack_len_mm,
            "crack_width": crack_width_mm,
            "corrosion_percent": percentages["corrosion_percent"],
            "spalling_percent": percentages["spalling_percent"],
            "vegetation_percent": percentages["vegetation_percent"],
            "leakage_percent": percentages["leakage_percent"],
            "tilt_angle": tilt_deg,
            "missing_components": int(counts["missing_bolts"] + counts["loose_components"]),
            "damage_area_percent": damage_area_percent
        }

        # Keep raw objects for drawing overlays later
        raw_results = {
            "image": image,
            "bridge_info": bridge_info,
            "damage_info": damage_info,
            "masks": masks,
            "bboxes": bboxes,
            "features": features
        }

        return features, raw_results


class OpenCVFeatureExtractor(BaseFeatureExtractor):
    def extract_features(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        segments: List[SegmentationResult],
        pixel_to_mm: float,
    ) -> Dict[str, Any]:
        """
        Extracts dimensional measurements from masks and bboxes.
        Classifies defect severity levels.
        """
        h, w = image.shape[:2]
        
        # Identify bridge component mask (union of structural elements)
        struct_labels = {"Girder", "Deck", "Pier", "Bearing", "Expansion Joint", "Guard Rail", "Connection Plate"}
        bridge_mask = np.zeros((h, w), dtype=np.uint8)
        
        for seg in segments:
            if seg.label in struct_labels:
                bridge_mask = cv2.bitwise_or(bridge_mask, seg.mask)
                
        bridge_area_pixels = int(np.sum(bridge_mask > 0))
        if bridge_area_pixels == 0:
            # Fallback if no structural component detected: use full image
            bridge_area_pixels = h * w
            
        # Initialize stats
        cracks_masks = []
        rust_masks = []
        spalling_masks = []
        leakage_masks = []
        vegetation_masks = []
        guardrail_defect_masks = []
        
        missing_bolts = 0
        missing_nuts = 0
        loose_connections = 0
        
        for seg in segments:
            if seg.label == "Crack":
                cracks_masks.append(seg.mask)
            elif seg.label in ("Rust", "Corrosion"):
                rust_masks.append(seg.mask)
            elif seg.label == "Spalling":
                spalling_masks.append(seg.mask)
            elif seg.label == "Water Leakage":
                leakage_masks.append(seg.mask)
            elif seg.label == "Vegetation":
                vegetation_masks.append(seg.mask)
            elif seg.label == "Guard Rail Damage":
                guardrail_defect_masks.append(seg.mask)
                
        for det in detections:
            if det.label == "Missing Bolt":
                missing_bolts += 1
            elif det.label == "Missing Nut":
                missing_nuts += 1
            elif det.label == "Loose Connection":
                loose_connections += 1

        # 1. Crack calculations
        crack_count = len(cracks_masks)
        crack_lengths = []
        crack_widths = []
        crack_pixels = 0
        
        for mask in cracks_masks:
            crack_pixels += np.sum(mask > 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                # Estimate length (perimeter / 2)
                arc_len = cv2.arcLength(c, False)
                length_mm = (arc_len / 2.0) * pixel_to_mm
                if length_mm > 0:
                    crack_lengths.append(length_mm)
                    # Width = area / length
                    area = cv2.contourArea(c)
                    area_mm2 = area * (pixel_to_mm ** 2)
                    width_mm = area_mm2 / length_mm if length_mm > 0 else 0.0
                    crack_widths.append(max(0.1, width_mm))
                    
        avg_crack_len = float(np.mean(crack_lengths)) if crack_lengths else 0.0
        max_crack_len = float(np.max(crack_lengths)) if crack_lengths else 0.0
        avg_crack_width = float(np.mean(crack_widths)) if crack_widths else 0.0
        max_crack_width = float(np.max(crack_widths)) if crack_widths else 0.0
        crack_density = float(round((crack_pixels / bridge_area_pixels) * 100, 4))

        # 2. Area percentages
        def get_area_pct(masks_list):
            if not masks_list:
                return 0.0
            union = np.zeros((h, w), dtype=np.uint8)
            for m in masks_list:
                union = cv2.bitwise_or(union, m)
            return float(round((np.sum(union > 0) / bridge_area_pixels) * 100, 4))
            
        corrosion_pct = get_area_pct(rust_masks)
        spalling_pct = get_area_pct(spalling_masks)
        leakage_pct = get_area_pct(leakage_masks)
        vegetation_pct = get_area_pct(vegetation_masks)
        
        # 3. Overall surface damage percent (union of all defects)
        all_defects_mask = np.zeros((h, w), dtype=np.uint8)
        defect_labels = {"Crack", "Rust", "Corrosion", "Spalling", "Water Leakage", "Vegetation", "Guard Rail Damage"}
        for seg in segments:
            if seg.label in defect_labels:
                all_defects_mask = cv2.bitwise_or(all_defects_mask, seg.mask)
        surface_damage_pct = float(round((np.sum(all_defects_mask > 0) / bridge_area_pixels) * 100, 4))

        # 4. Bridge Tilt estimation
        tilt_angle = 0.0
        # Find structural lines for Deck or Girder to estimate slope
        for seg in segments:
            if seg.label in ("Deck", "Girder", "Guard Rail"):
                contours, _ = cv2.findContours(seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if len(c) >= 5:
                        (cx, cy), (ma, mi), angle = cv2.fitEllipse(c)
                        if angle > 90:
                            angle = angle - 180
                        tilt_angle = float(round(angle, 2))
                        break

        # 5. Severity Classifications
        defects_list = []
        for idx, det in enumerate(detections):
            if det.label in struct_labels:
                continue
                
            bx, by, bw, bh = det.bbox
            severity = "Minor"
            
            if det.label == "Crack":
                width = avg_crack_width
                if width > 4.0:
                    severity = "Critical"
                elif width > 2.0:
                    severity = "Severe"
                elif width > 0.8:
                    severity = "Moderate"
            elif det.label in ("Rust", "Corrosion"):
                area_mm2 = (bw * bh) * (pixel_to_mm ** 2)
                if area_mm2 > 50000:
                    severity = "Critical"
                elif area_mm2 > 10000:
                    severity = "Severe"
                elif area_mm2 > 2000:
                    severity = "Moderate"
            elif det.label == "Spalling":
                area_mm2 = (bw * bh) * (pixel_to_mm ** 2)
                if area_mm2 > 80000:
                    severity = "Critical"
                elif area_mm2 > 25000:
                    severity = "Severe"
                elif area_mm2 > 5000:
                    severity = "Moderate"
            elif det.label in ("Missing Bolt", "Missing Nut"):
                severity = "Severe"
            elif det.label == "Loose Connection":
                severity = "Moderate"
            elif det.label == "Guard Rail Damage":
                severity = "Severe"
                
            defects_list.append({
                "defect_index": idx,
                "type": det.label,
                "bbox": det.bbox,
                "confidence": float(round(det.confidence, 4)),
                "severity": severity,
                "measurements": {
                    "width_mm": float(round(max_crack_width, 2)) if det.label == "Crack" else 0.0,
                    "length_mm": float(round(max_crack_len, 2)) if det.label == "Crack" else 0.0,
                    "area_pct": float(round((bw * bh) / bridge_area_pixels * 100, 2))
                }
            })

        # Collect detected structural components
        detected_components = list({det.label for det in detections if det.label in struct_labels})

        features = {
            "crack_count": crack_count,
            "crack_density": crack_density,
            "avg_crack_length": avg_crack_len,
            "max_crack_length": max_crack_len,
            "avg_crack_width": avg_crack_width,
            "max_crack_width": max_crack_width,
            "corrosion_percent": corrosion_pct,
            "spalling_percent": spalling_pct,
            "leakage_percent": leakage_pct,
            "missing_bolts": missing_bolts,
            "missing_nuts": missing_nuts,
            "loose_connections": loose_connections,
            "vegetation_percent": vegetation_pct,
            "surface_damage_percent": surface_damage_pct,
            "bridge_tilt": tilt_angle,
            "components": detected_components,
            "defects": defects_list
        }
        
        return features
