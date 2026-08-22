"""
BridgeGuardian AI — Vision Inference Pipeline
Orchestrates the image inspection pipeline, generates overlays,
and maps features into the tabular ML model.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from backend.ml.computer_vision.feature_extractor import ImageFeatureExtractor
from backend.ml.computer_vision.image_enhancer import ImageEnhancer
from backend.ml.computer_vision.sahi_slicing import SAHISlicer
from backend.api.routes.predict import _get_default

logger = logging.getLogger("bridgeguardian.cv.vision_inference")

class VisionInferencePipeline:
    """
    Combines Computer Vision feature extraction with existing tabular ML prediction.
    Generates annotated defect overlays, damage heatmaps, and segmentation layers.
    """

    def __init__(self, use_yolo: bool = False, pixel_to_mm: float = 0.5) -> None:
        self.extractor = ImageFeatureExtractor(use_yolo=use_yolo, pixel_to_mm=pixel_to_mm)
        self.enhancer = ImageEnhancer()
        self.slicer = SAHISlicer(slice_height=512, slice_width=512)

    def analyze_image(self, image_path: str, inference_pipeline) -> dict:
        """
        Runs image quality check, CV feature extraction, visual output generation,
        maps features to ML, runs ML predictions, and generates SHAP explanation.
        """
        from backend.ml.computer_vision.image_quality import OpenCVImageQualityChecker
        quality_checker = OpenCVImageQualityChecker()
        quality_res = quality_checker.check_quality(image_path)
        
        if not quality_res["is_valid"]:
            return {
                "is_valid": False,
                "warnings": quality_res["warnings"],
                "rejection_reason": quality_res.get("rejection_reason", "Failed quality validation"),
                "features": {
                    "crack_density": 0.0, "crack_length": 0.0, "crack_width": 0.0,
                    "corrosion_percent": 0.0, "spalling_percent": 0.0, "damage_area_percent": 0.0,
                    "missing_components": 0, "no_defects_detected": True
                },
                "predictions": {
                    "health_score_raw": None,
                    "health_score": "N/A",
                    "failure_probability_raw": None,
                    "failure_probability": "N/A",
                    "rul_days": "N/A",
                    "prediction_confidence": 0.0,
                    "risk_category": "Unknown",
                    "maintenance_priority": "Inspection Required",
                    "maintenance_recommendation": "Inspection could not be completed. The image failed quality validation."
                },
                "visualizations": {
                    "original": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.imread(image_path))[1]).decode('utf-8')}" if cv2.imread(image_path) is not None else "",
                    "cracks": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.imread(image_path))[1]).decode('utf-8')}" if cv2.imread(image_path) is not None else "",
                    "rust": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.imread(image_path))[1]).decode('utf-8')}" if cv2.imread(image_path) is not None else "",
                    "bboxes": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.imread(image_path))[1]).decode('utf-8')}" if cv2.imread(image_path) is not None else "",
                    "heatmap": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.imread(image_path))[1]).decode('utf-8')}" if cv2.imread(image_path) is not None else "",
                    "segmentation": f"data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.imread(image_path))[1]).decode('utf-8')}" if cv2.imread(image_path) is not None else ""
                },
                "saved_paths": {},
                "shap": {},
                "ml_input": {}
            }

        # 1. Extract CV features and get intermediate masks/bboxes
        features, raw = self.extractor.extract_features(image_path)
        img = raw["image"]

        # Enhance image quality (CLAHE, exposure correction, bilateral denoising, sharpening)
        img_enhanced, enhancement_actions = self.enhancer.enhance(img)
        raw["image"] = img_enhanced

        masks = raw["masks"]
        bboxes = raw["bboxes"]
        h, w = img_enhanced.shape[:2]


        # 2. Generate the 6 required visualization images
        visualizations = {}
        
        # Vis 1: Original Image
        visualizations["original"] = img.copy()

        # Vis 2: Detected Cracks (Red outlines)
        vis_cracks = img.copy()
        crack_cnts, _ = cv2.findContours(masks["cracks"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_cracks, crack_cnts, -1, (0, 0, 255), 2)
        visualizations["cracks"] = vis_cracks

        # Vis 3: Detected Rust (Orange semi-transparent overlay)
        vis_rust = img.copy()
        rust_color_mask = np.zeros_like(img)
        rust_color_mask[masks["rust"] > 0] = [0, 128, 255]  # Orange BGR
        cv2.addWeighted(rust_color_mask, 0.5, vis_rust, 0.5, 0, dst=vis_rust)
        visualizations["rust"] = vis_rust

        # Vis 4: Bounding Boxes ( defect label + confidence )
        vis_bboxes = img.copy()
        for b in bboxes:
            bx, by, bw_b, bh_b = b["bbox"]
            label = b["label"]
            conf = b["confidence"]
            # Color based on label
            color = (0, 0, 255) # default Red
            if label == "Rust/Corrosion":
                color = (0, 128, 255)
            elif label == "Vegetation":
                color = (0, 255, 0)
            elif label == "Water Leakage":
                color = (255, 0, 255)
            elif label == "Missing Bolt" or label == "Loose Component":
                color = (255, 255, 0)
                
            cv2.rectangle(vis_bboxes, (bx, by), (bx + bw_b, by + bh_b), color, 2)
            cv2.putText(vis_bboxes, f"{label} ({conf:.2f})", (bx, max(15, by - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        visualizations["bboxes"] = vis_bboxes

        # Vis 5: Damage Heatmap (density-based smooth overlay)
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for m in masks.values():
            union_mask = cv2.bitwise_or(union_mask, m)
            
        if np.sum(union_mask > 0) > 0:
            # Blur the union mask heavily to get density gradients
            blur_size = max(21, (min(h, w) // 15) | 1) # Must be odd
            density = cv2.GaussianBlur(union_mask.astype(float), (blur_size, blur_size), 0)
            # Normalize to 0-255
            max_d = np.max(density)
            if max_d > 0:
                density = (density / max_d * 255).astype(np.uint8)
            else:
                density = np.zeros((h, w), dtype=np.uint8)
            
            # Apply JET colormap
            heatmap_color = cv2.applyColorMap(density, cv2.COLORMAP_JET)
            # Blend with original image
            vis_heatmap = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        else:
            vis_heatmap = img.copy()
        visualizations["heatmap"] = vis_heatmap

        # Vis 6: Segmentation Overlay (Colored masks)
        vis_seg = img.copy()
        seg_color_mask = np.zeros_like(img)
        # Colors:
        # Cracks: Red (0, 0, 255)
        # Rust: Orange/Yellow (0, 180, 255)
        # Spalling: Blue (255, 0, 0)
        # Vegetation: Green (0, 255, 0)
        # Leakage: Purple (128, 0, 128)
        # Guardrail/Deformation: Cyan (255, 255, 0)
        seg_color_mask[masks["cracks"] > 0] = [0, 0, 255]
        seg_color_mask[masks["rust"] > 0] = [0, 180, 255]
        seg_color_mask[masks["spalling"] > 0] = [255, 0, 0]
        seg_color_mask[masks["vegetation"] > 0] = [0, 255, 0]
        seg_color_mask[masks["leakage"] > 0] = [128, 0, 128]
        seg_color_mask[masks["guardrail"] > 0] = [255, 255, 0]
        seg_color_mask[masks["deformation"] > 0] = [255, 128, 0]
        
        # Apply blending where there is any defect
        any_defect = union_mask > 0
        vis_seg[any_defect] = cv2.addWeighted(seg_color_mask, 0.4, img, 0.6, 0)[any_defect]
        visualizations["segmentation"] = vis_seg

        # Save visualizations to base64 strings
        b64_results = {}
        for key, vis_img in visualizations.items():
            _, buffer = cv2.imencode(".jpg", vis_img)
            b64_str = base64.b64encode(buffer).decode("utf-8")
            b64_results[key] = f"data:image/jpeg;base64,{b64_str}"

        # 3. Integrate with existing ML prediction pipeline
        # Build features input dict, substituting default tabular features with vision features
        ml_input = {}
        for field in [
            "Strain_microstrain", "Deflection_mm", "Vibration_ms2", "Tilt_deg", "Displacement_mm",
            "Crack_Propagation_mm", "Corrosion_Level_percent", "Cable_Member_Tension_kN", 
            "Bearing_Joint_Forces_kN", "Fatigue_Accumulation_au", "Modal_Frequency_Hz",
            "Temperature_C", "Humidity_percent", "Wind_Speed_ms", "Wind_Direction_deg",
            "Precipitation_mmh", "Water_Level_m", "Seismic_Activity_ms2", "Solar_Radiation_Wm2",
            "Air_Quality_Index_AQI", "Soil_Settlement_mm", "Vehicle_Load_tons", "Traffic_Volume_vph",
            "Pedestrian_Load_pph", "Impact_Events_g", "Dynamic_Load_Distribution_percent",
            "Axle_Counts_pmin", "Anomaly_Detection_Score", "Energy_Dissipation_au", 
            "Acoustic_Emissions_levels", "Visual_Analysis_Defect_Score", "Electrical_Resistance_ohms",
            "Localized_Strain_Hotspot", "Bridge_Mood_Meter", "Vibration_Anomaly_Location",
            "Flood_Event_Flag", "High_Winds_Storms", "Landslide_Ground_Movement", 
            "Abnormal_Traffic_Load_Surges", "SHI_Predicted_7d_Ahead", "SHI_Predicted_30d_Ahead"
        ]:
            ml_input[field] = _get_default(field)

        # Overwrite with image-extracted features
        # Tilt angle maps directly to Tilt_deg
        ml_input["Tilt_deg"] = features["tilt_angle"]
        # Corrosion level percent is converted to a 0-1 fraction for Corrosion_Level_percent
        ml_input["Corrosion_Level_percent"] = features["corrosion_percent"] / 100.0
        # Crack length estimated in mm maps to Crack_Propagation_mm
        ml_input["Crack_Propagation_mm"] = features["crack_length"]
        # Overall damage area percent maps to Visual_Analysis_Defect_Score (0-1 fraction)
        ml_input["Visual_Analysis_Defect_Score"] = features["damage_area_percent"] / 100.0
        
        # If there are missing components, raise the anomaly score
        if features["missing_components"] > 0:
            ml_input["Anomaly_Detection_Score"] = min(1.0, float(features["missing_components"] * 0.15))
            ml_input["Bridge_Mood_Meter"] = "Stressed" if features["missing_components"] < 3 else "Critical"

        # If crack length is high, adjust the mood meter
        if features["crack_length"] > 10.0:
            ml_input["Bridge_Mood_Meter"] = "Critical"

        # Make sure predictions are loaded
        if not inference_pipeline.is_ready:
            inference_pipeline.load()
            
        # Run tabular prediction
        prediction_results = inference_pipeline.predict(ml_input)

        # Calculate visual damage penalty ONLY from verified defects
        no_defects = features.get("no_defects_detected", False) or (
            features.get("corrosion_percent", 0.0) == 0.0 and
            features.get("crack_length", 0.0) == 0.0 and
            features.get("spalling_percent", 0.0) == 0.0 and
            features.get("missing_components", 0) == 0
        )

        if no_defects:
            visual_damage = 0.0
            # Reset prediction to baseline healthy state
            prediction_results["health_score_raw"] = 1.0
            prediction_results["health_score"] = 100.0
            prediction_results["failure_probability_raw"] = 0.0
            prediction_results["failure_probability"] = 0.0
            prediction_results["risk_category"] = "Good"
            prediction_results["maintenance_priority"] = "Routine"
            prediction_results["maintenance_recommendation"] = "No structural defects detected. Continue scheduled monitoring."
            prediction_results["rul_days"] = 3650.0 # 10 years
            prediction_results["prediction_confidence"] = 0.98
        else:
            corrosion_factor = min(1.0, features["corrosion_percent"] / 100.0)
            crack_factor = min(1.0, features["crack_length"] / 1000.0)
            spalling_factor = min(1.0, features["spalling_percent"] / 100.0)
            leakage_factor = min(1.0, features.get("leakage_percent", 0.0) / 100.0)
            missing_factor = min(1.0, features["missing_components"] * 0.05)
            tilt_factor = min(1.0, abs(features.get("tilt_angle", 0.0)) / 15.0)

            visual_damage = (
                corrosion_factor * 0.25 +
                crack_factor * 0.35 +
                spalling_factor * 0.15 +
                leakage_factor * 0.05 +
                missing_factor * 0.1 +
                tilt_factor * 0.1
            )

        if visual_damage > 0.0:
            raw_shi = prediction_results["health_score_raw"]
            raw_pof = prediction_results["failure_probability_raw"]

            adjusted_raw_shi = max(0.01, raw_shi * (1.0 - visual_damage))
            adjusted_raw_pof = min(0.99, raw_pof + (1.0 - raw_pof) * visual_damage)

            # Re-classify risk categories
            from backend.ml.inference import RISK_CATEGORIES, MAINTENANCE_PRIORITIES, RECOMMENDATIONS

            risk_category = "Unknown"
            for category, (low, high) in RISK_CATEGORIES.items():
                if low <= adjusted_raw_shi < high:
                    risk_category = category
                    break

            maintenance_priority = MAINTENANCE_PRIORITIES.get(risk_category, "Routine")
            recommendation = RECOMMENDATIONS.get(maintenance_priority, "")

            rul_result = inference_pipeline._rul_estimator.estimate(adjusted_raw_shi, None, None)

            prediction_results["health_score_raw"] = round(adjusted_raw_shi, 4)
            prediction_results["health_score"] = round(adjusted_raw_shi * 100, 2)
            prediction_results["failure_probability_raw"] = round(adjusted_raw_pof, 4)
            prediction_results["failure_probability"] = round(adjusted_raw_pof * 100, 2)
            prediction_results["risk_category"] = risk_category
            prediction_results["maintenance_priority"] = maintenance_priority
            prediction_results["maintenance_recommendation"] = recommendation
            prediction_results["rul_days"] = rul_result["rul_days"]
            prediction_results["rul_degradation_rate"] = rul_result["degradation_rate_per_day"]
            prediction_results["rul_confidence"] = rul_result["confidence"]
            prediction_results["rul_message"] = rul_result["message"]

            agreement = 1.0 - abs(adjusted_raw_shi - (1.0 - adjusted_raw_pof))
            prediction_results["prediction_confidence"] = round(max(0.5, min(1.0, agreement)), 3)

        # 4. Generate SHAP explanation
        shap_explanation = {}
        try:
            explanation_data = inference_pipeline.explain(ml_input, target="health_score")
            shap_explanation = explanation_data
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation for vision inference: {e}")

        # Save files on disk in processed directory for PDF report reference
        from backend.core.config import get_settings
        static_dir = Path(get_settings().processed_dir)
        try:
            static_dir.mkdir(parents=True, exist_ok=True)
        except Exception as p_err:
            logger.warning(f"Could not create static_dir '{static_dir}': {p_err}")
        img_id = Path(image_path).stem
        
        saved_paths = {}
        for key, vis_img in visualizations.items():
            save_path = static_dir / f"{img_id}_{key}.jpg"
            cv2.imwrite(str(save_path), vis_img)
            saved_paths[key] = str(save_path)

        # Clean features for JSON serialization by stripping raw numpy arrays (damage_masks)
        clean_features = {k: v for k, v in features.items() if k != "damage_masks"}

        return {
            "features": clean_features,
            "predictions": prediction_results,
            "visualizations": b64_results,
            "saved_paths": saved_paths,
            "shap": shap_explanation,
            "ml_input": ml_input
        }
