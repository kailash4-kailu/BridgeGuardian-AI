"""
BridgeGuardian AI — Visualization Component
Generates styled bounding box overlays, transparent segmentation masks, and density heatmaps.
"""
from __future__ import annotations
import base64
import cv2
import numpy as np
from typing import Dict, List, Tuple
from backend.ml.computer_vision.base import DetectionResult, SegmentationResult

class Visualizer:
    @staticmethod
    def draw_bboxes(image: np.ndarray, detections: List[DetectionResult], severities: List[str]) -> np.ndarray:
        """Draw bounding boxes with labels, confidence, and severity colors."""
        img = image.copy()
        
        # Severity colors (BGR)
        colors = {
            "Critical": (0, 0, 255),    # Red
            "Severe": (0, 128, 255),    # Orange
            "Moderate": (0, 255, 255),  # Yellow
            "Minor": (0, 255, 0),       # Green
            "Component": (255, 0, 0)    # Blue for structural components
        }
        
        struct_labels = {"Girder", "Deck", "Pier", "Bearing", "Expansion Joint", "Guard Rail", "Connection Plate"}
        
        for det, severity in zip(detections, severities):
            bx, by, bw, bh = det.bbox
            
            # Map color based on type/severity
            if det.label in struct_labels:
                color = colors["Component"]
                label_text = det.label
            else:
                color = colors.get(severity, colors["Moderate"])
                label_text = f"{det.label} [{severity}] ({det.confidence:.2f})"
                
            # Draw box
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
            
            # Draw text label background
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (bx, max(0, by - th - 5)), (bx + tw, by), color, -1)
            
            # Draw text label
            cv2.putText(
                img, label_text, (bx, max(12, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            
        return img

    @staticmethod
    def draw_segmentation(image: np.ndarray, segments: List[SegmentationResult], severities: List[str]) -> np.ndarray:
        """Draw filled transparent segmentation masks for detected defects."""
        h, w = image.shape[:2]
        img = image.copy()
        
        # Color mask layer
        color_mask = np.zeros_like(image)
        
        colors = {
            "Critical": (0, 0, 255),
            "Severe": (0, 128, 255),
            "Moderate": (0, 255, 255),
            "Minor": (0, 255, 0),
            "Component": (150, 50, 50) # Dark blue/gray for components
        }
        
        struct_labels = {"Girder", "Deck", "Pier", "Bearing", "Expansion Joint", "Guard Rail", "Connection Plate"}
        
        any_mask_drawn = False
        
        for seg, severity in zip(segments, severities):
            if seg.mask is None or np.sum(seg.mask) == 0:
                continue
                
            color = colors["Component"] if seg.label in struct_labels else colors.get(severity, colors["Moderate"])
            
            # Fill the mask on color layer
            color_mask[seg.mask > 0] = color
            any_mask_drawn = True
            
        if any_mask_drawn:
            # Blend original and colored masks
            mask_indices = np.any(color_mask > 0, axis=2)
            img[mask_indices] = cv2.addWeighted(color_mask, 0.4, image, 0.6, 0)[mask_indices]
            
        return img

    @staticmethod
    def draw_heatmap(image: np.ndarray, segments: List[SegmentationResult]) -> np.ndarray:
        """Draw density damage heatmaps by blurring the union of defect masks."""
        h, w = image.shape[:2]
        img = image.copy()
        
        defect_labels = {"Crack", "Rust", "Corrosion", "Spalling", "Water Leakage", "Vegetation", "Guard Rail Damage", "Expansion Joint Damage"}
        union_mask = np.zeros((h, w), dtype=np.uint8)
        
        for seg in segments:
            if seg.label in defect_labels and seg.mask is not None:
                union_mask = cv2.bitwise_or(union_mask, seg.mask)
                
        if np.sum(union_mask > 0) > 0:
            # Apply Gaussian Blur to create a density gradient
            blur_size = max(21, (min(h, w) // 15) | 1)
            density = cv2.GaussianBlur(union_mask.astype(float), (blur_size, blur_size), 0)
            
            # Normalize to 0-255
            max_d = np.max(density)
            if max_d > 0:
                density = (density / max_d * 255).astype(np.uint8)
                # Apply Colormap Jet
                heatmap_color = cv2.applyColorMap(density, cv2.COLORMAP_JET)
                # Blend with original
                img = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
                
        return img

    @staticmethod
    def to_base64_src(image: np.ndarray) -> str:
        """Convert BGR image to base64 data URI."""
        _, buffer = cv2.imencode(".jpg", image)
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
