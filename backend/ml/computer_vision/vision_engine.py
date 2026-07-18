"""
BridgeGuardian AI — Vision AI Engine Orchestrator
Coordinates quality control, object detection, instance segmentation, features, and visualizations.
"""
from __future__ import annotations
import logging
import cv2
from pathlib import Path
from typing import Any, Dict, List
from backend.ml.computer_vision.base import (
    BaseDetector,
    BaseSegmenter,
    BaseFeatureExtractor,
    BaseImageQualityChecker,
)
from backend.ml.computer_vision.visualization import Visualizer

logger = logging.getLogger("bridgeguardian.cv.vision_engine")

class VisionEngine:
    def __init__(
        self,
        detector: BaseDetector,
        segmenter: BaseSegmenter,
        extractor: BaseFeatureExtractor,
        quality_checker: BaseImageQualityChecker
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.extractor = extractor
        self.quality_checker = quality_checker

    def process_images(self, image_paths: List[str], pixel_to_mm: float = 0.5) -> List[Dict[str, Any]]:
        """
        Orchestrates batch image processing.
        Gracefully handles corrupted frames or missing assets on a per-image basis.
        """
        results = []
        static_dir = Path("backend/static/processed")
        static_dir.mkdir(parents=True, exist_ok=True)
        
        for path_str in image_paths:
            path = Path(path_str)
            img_name = path.name
            
            logger.info(f"Vision Engine processing image: {img_name}")
            
            # 1. Quality Check
            try:
                quality_res = self.quality_checker.check_quality(path_str)
            except Exception as e:
                logger.error(f"Quality check failed for {img_name}: {e}")
                results.append({
                    "image_path": path_str,
                    "image_name": img_name,
                    "is_valid": False,
                    "warnings": [f"Quality check runtime error: {str(e)}"],
                    "metrics": {}
                })
                continue
                
            if not quality_res["is_valid"]:
                logger.warning(f"Image {img_name} failed quality check: {quality_res['warnings']}")
                results.append({
                    "image_path": path_str,
                    "image_name": img_name,
                    "is_valid": False,
                    "warnings": quality_res["warnings"],
                    "metrics": quality_res["metrics"]
                })
                continue
                
            # 2. Load Image & Run Inference
            try:
                image = cv2.imread(path_str)
                if image is None:
                    raise ValueError("Failed to load image matrix (cv2.imread returned None)")
                    
                # Run YOLO Detector (Stage 1)
                detections = self.detector.detect(image, path_str)
                
                # Run SAM Segmenter (Stage 2)
                segments = self.segmenter.segment(image, detections)
                
                # Run OpenCV Feature Extractor (Stage 3)
                features = self.extractor.extract_features(image, detections, segments, pixel_to_mm)
                
                # 3. Generate Visual Overlays
                # Match severities to detections
                severities = []
                struct_labels = {"Girder", "Deck", "Pier", "Bearing", "Expansion Joint", "Guard Rail", "Connection Plate"}
                defect_items = [d for d in features["defects"]]
                
                for det in detections:
                    if det.label in struct_labels:
                        severities.append("Component")
                    else:
                        # find matching severity
                        matched = next((d for d in defect_items if d["type"] == det.label and d["bbox"] == det.bbox), None)
                        severities.append(matched["severity"] if matched else "Moderate")
                        
                vis_bboxes = Visualizer.draw_bboxes(image, detections, severities)
                vis_segs = Visualizer.draw_segmentation(image, segments, severities)
                vis_heatmap = Visualizer.draw_heatmap(image, segments)
                
                # Save base64 strings
                b64_results = {
                    "original": Visualizer.to_base64_src(image),
                    "bboxes": Visualizer.to_base64_src(vis_bboxes),
                    "segmentation": Visualizer.to_base64_src(vis_segs),
                    "heatmap": Visualizer.to_base64_src(vis_heatmap)
                }
                
                # Save processed copies to disk for PDF generator
                img_id = path.stem
                saved_paths = {}
                for key, vis_img in [("original", image), ("bboxes", vis_bboxes), ("segmentation", vis_segs), ("heatmap", vis_heatmap)]:
                    save_path = static_dir / f"{img_id}_{key}.jpg"
                    cv2.imwrite(str(save_path), vis_img)
                    saved_paths[key] = str(save_path)
                    
                results.append({
                    "image_path": path_str,
                    "image_name": img_name,
                    "is_valid": True,
                    "warnings": [],
                    "metrics": quality_res["metrics"],
                    "features": features,
                    "visualizations": b64_results,
                    "saved_paths": saved_paths
                })
                
            except Exception as e:
                logger.error(f"Inference pipeline failed on {img_name}: {e}", exc_info=True)
                results.append({
                    "image_path": path_str,
                    "image_name": img_name,
                    "is_valid": False,
                    "warnings": [f"Inference failure: {str(e)}"],
                    "metrics": quality_res["metrics"]
                })
                
        return results
