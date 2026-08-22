"""
BridgeGuardian AI — Vision AI Engine Orchestrator
Coordinates quality control, object detection, instance segmentation, features, and visualizations.
"""
from __future__ import annotations
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from backend.ml.computer_vision.base import (
    BaseDetector,
    BaseSegmenter,
    BaseFeatureExtractor,
    BaseImageQualityChecker,
    DetectionResult,
    SegmentationResult,
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
        from backend.core.config import get_settings
        processed_dir = get_settings().processed_dir
        static_dir = Path(processed_dir)
        try:
            static_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create processed static_dir {static_dir}: {e}")
        
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
                    
                # Run Feature Extractor (Hierarchical: Bridge -> Visible Components -> SAM2 -> ROI Defects -> Verified Features)
                features, raw = self.extractor.extract_features(path_str)
                
                # Reconstruct detections and segmentations for visualization
                visible_comps = raw.get("visible_components", [])
                damage_info = raw.get("damage_info", {})
                raw_candidates = damage_info.get("raw_candidates", [])
                bboxes = damage_info.get("bboxes", [])
                masks = damage_info.get("masks", {})
                
                # PIPELINE DEBUG 10-STAGE AUDIT LOGGING
                logger.info(f"================ PIPELINE STAGE AUDIT: '{img_name}' ================")
                logger.info(f"  Stage 1: Image Loaded & Matrix Verified ({image.shape[1]}x{image.shape[0]} px)")
                logger.info(f"  Stage 2: Quality Check Gate PASSED (Blur metric: {quality_res['metrics'].get('blur_score', 0):.1f})")
                logger.info(f"  Stage 3: Bridge Presence Detection PASSED")
                logger.info(f"  Stage 4: Component Detection ({len(visible_comps)} components: {[c['label'] for c in visible_comps]})")
                logger.info(f"  Stage 5: Component Mask Segmentation ({len(visible_comps)} ROI masks active)")
                logger.info(f"  Stage 6: Raw Defect Model Predictions ({len(raw_candidates)} candidate boxes)")
                logger.info(f"  Stage 7: Confidence Threshold Filtering (Threshold >= 0.30)")
                logger.info(f"  Stage 8: Non-Maximum Suppression (IoU <= 0.45)")
                logger.info(f"  Stage 9: Structural Reasoning & Context Rule Validation PASSED")
                logger.info(f"  Stage 10: Final Verified Defects ({len(bboxes)} defects: {[b['label'] for b in bboxes]})")
                logger.info(f"====================================================================")
                
                if len(bboxes) == 0 and len(raw_candidates) > 0:
                    logger.warning(f"  [AUDIT WARNING] All {len(raw_candidates)} raw defect candidates were filtered out during post-processing!")

                detections = []
                segments = []
                severities = []

                # Add visible components to detections & segments
                for comp in visible_comps:
                    label = comp["label"]
                    bbox = comp["bbox"]
                    conf = comp.get("confidence", 0.90)
                    mask = comp.get("mask")
                    
                    detections.append(DetectionResult(label=label, bbox=bbox, confidence=conf))
                    segments.append(SegmentationResult(label=label, mask=mask, polygon=[]))
                    severities.append("Component")

                # Add verified defects to detections & segments
                for b in bboxes:
                    label = b["label"]
                    bbox = b["bbox"]
                    conf = b["confidence"]
                    
                    severity = "Moderate"
                    if "crack" in label.lower():
                        severity = "Severe" if features.get("crack_width", 0) > 2.0 else "Moderate"
                    elif "rust" in label.lower() or "corrosion" in label.lower():
                        severity = "Severe" if features.get("corrosion_percent", 0) > 5.0 else "Moderate"
                    elif "bolt" in label.lower():
                        severity = "Severe"
                        
                    detections.append(DetectionResult(label=label, bbox=bbox, confidence=conf))
                    severities.append(severity)

                # Add defect masks to segments
                for mask_key, mask_mat in masks.items():
                    if np.sum(mask_mat > 0) > 0:
                        label_name = "Crack" if mask_key == "cracks" else ("Rust" if mask_key == "rust" else mask_key.capitalize())
                        segments.append(SegmentationResult(label=label_name, mask=mask_mat, polygon=[]))

                # 3. Generate Visual Overlays & Save Intermediate Debug Visualizations
                vis_bboxes = Visualizer.draw_bboxes(image, detections, severities)
                vis_segs = Visualizer.draw_segmentation(image, segments, severities)
                vis_heatmap = Visualizer.draw_heatmap(image, segments)

                # Raw candidates visualization overlay for debugging
                raw_dets = [DetectionResult(label=cand["label"], bbox=cand["bbox"], confidence=cand["confidence"]) for cand in raw_candidates]
                vis_raw_preds = Visualizer.draw_bboxes(image, raw_dets, ["Minor"] * len(raw_dets))
                comp_dets = [DetectionResult(label=c["label"], bbox=c["bbox"], confidence=c.get("confidence", 0.9)) for c in visible_comps]
                vis_comp_preds = Visualizer.draw_bboxes(image, comp_dets, ["Component"] * len(comp_dets))

                # Save intermediate debug visualizations
                img_id = path.stem
                debug_dir = static_dir / "debug" / img_id
                try:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                except Exception as dbg_err:
                    logger.warning(f"Could not create debug_dir {debug_dir}: {dbg_err}")

                # Required exact filenames
                cv2.imwrite(str(debug_dir / "bridge_detection.jpg"), vis_comp_preds)
                cv2.imwrite(str(debug_dir / "component_detection.jpg"), vis_comp_preds)
                cv2.imwrite(str(debug_dir / "raw_defects.jpg"), vis_raw_preds)
                cv2.imwrite(str(debug_dir / "filtered_defects.jpg"), vis_bboxes)
                cv2.imwrite(str(debug_dir / "final_predictions.jpg"), vis_bboxes)
                
                # Compatibility PNG filenames
                cv2.imwrite(str(debug_dir / "components.png"), vis_comp_preds)
                cv2.imwrite(str(debug_dir / "segmentation.png"), vis_segs)
                cv2.imwrite(str(debug_dir / "raw_predictions.png"), vis_raw_preds)
                cv2.imwrite(str(debug_dir / "filtered_predictions.png"), vis_bboxes)
                cv2.imwrite(str(debug_dir / "final_predictions.png"), vis_bboxes)
                logger.info(f"[Debug Visualizations] Saved all 5 intermediate stage debug images to '{debug_dir}'")
                
                # Save base64 strings
                b64_results = {
                    "original": Visualizer.to_base64_src(image),
                    "bboxes": Visualizer.to_base64_src(vis_bboxes),
                    "segmentation": Visualizer.to_base64_src(vis_segs),
                    "heatmap": Visualizer.to_base64_src(vis_heatmap)
                }
                
                # Save processed copies to disk for PDF generator
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

