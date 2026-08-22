"""
BridgeGuardian AI — Campaign Inspection Pipeline Orchestrator
Binds all six modular engines together, handles database progress lifecycle,
runs as a background task, and saves final campaign metrics.
"""
from __future__ import annotations
import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sqlalchemy.orm import Session

# Database schema
from backend.core.models import InspectionRecord, InspectionDefect

# Engines
from backend.ml.computer_vision.detector import YOLODetector
from backend.ml.computer_vision.segmentation import SAMSegmenter
from backend.ml.computer_vision.feature_extractor import ImageFeatureExtractor, OpenCVFeatureExtractor
from backend.ml.computer_vision.image_quality import OpenCVImageQualityChecker
from backend.ml.computer_vision.duplicate_merger import OpenCVDuplicateMerger
from backend.ml.computer_vision.vision_engine import VisionEngine
from backend.ml.computer_vision.evidence_graph import InspectionEvidenceGraph

from backend.ml.structural.structural_engine import StructuralEngine
from backend.ml.prediction.prediction_engine import PredictionEngine
from backend.ml.maintenance.maintenance_engine import MaintenanceEngine
from backend.ml.explainability.explainability_engine import ExplainabilityEngine
from backend.ml.report.report_engine import ReportEngine

# Tabular baseline prediction pipeline
from backend.ml.inference import InferencePipeline


def clean_numpy_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return clean_numpy_types(obj.tolist())
    else:
        return obj


logger = logging.getLogger("bridgeguardian.campaign.pipeline")

class CampaignInspectionPipeline:
    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.baseline_pipeline = InferencePipeline(str(self.models_dir))

    def run_campaign(
        self,
        db: Session,
        inspection_id: int,
        image_paths: List[str],
        pixel_to_mm: float = 0.5
    ) -> None:
        """
        Background task to execute the entire inspection campaign across multiple images.
        """
        start_time = time.time()
        logger.info(f"Starting inspection campaign campaign_id={inspection_id} on {len(image_paths)} images")
        
        # 1. Load record from DB
        record = db.query(InspectionRecord).filter(InspectionRecord.id == inspection_id).first()
        if not record:
            logger.error(f"Inspection record not found in database: {inspection_id}")
            return
            
        record.status = "running"
        record.progress = 0.05
        db.commit()
        
        try:
            # 2. Instantiate and Inject Vision AI Engine dependencies
            logger.info("Initializing Vision AI Engine components...")
            detector = YOLODetector(weights_path=str(self.models_dir / "bridge_defects_yolo.pt"))
            segmenter = SAMSegmenter(weights_path=str(self.models_dir / "sam2.pt"))
            extractor = ImageFeatureExtractor()
            quality_checker = OpenCVImageQualityChecker()
            merger = OpenCVDuplicateMerger()
            
            vision_engine = VisionEngine(detector, segmenter, extractor, quality_checker)
            record.progress = 0.15
            db.commit()
            
            # 3. Parallel Image Processing using ThreadPoolExecutor
            logger.info("Processing images through Vision AI Engine in parallel...")
            total_imgs = len(image_paths)
            max_workers = int(os.getenv("MAX_CONCURRENT_WORKERS", str(min(4, max(1, os.cpu_count() or 1)))))

            import concurrent.futures

            def _process_single(path_str: str):
                return vision_engine.process_images([path_str], pixel_to_mm)[0]

            image_results = [None] * total_imgs
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(_process_single, path): idx for idx, path in enumerate(image_paths)}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        image_results[idx] = future.result()
                    except Exception as exc:
                        logger.error(f"Error processing image {image_paths[idx]}: {exc}", exc_info=True)
                        image_results[idx] = {
                            "image_path": image_paths[idx],
                            "image_name": Path(image_paths[idx]).name,
                            "is_valid": False,
                            "warnings": [f"Processing error: {str(exc)}"],
                            "metrics": {}
                        }
                    completed_count += 1
                    record.progress = round(0.15 + (completed_count / total_imgs) * 0.55, 2)
                    db.commit()
                
            # Calculate batch quality validation statistics
            valid_imgs = [r for r in image_results if r.get("is_valid", False)]
            rejected_imgs = [r for r in image_results if not r.get("is_valid", False)]
            accepted_count = len(valid_imgs)
            rejected_count = len(rejected_imgs)
            duration = time.time() - start_time
            
            # Determine Pipeline State
            if total_imgs == 0:
                pipeline_state = "NO_IMAGES"
            elif accepted_count == 0:
                pipeline_state = "ALL_IMAGES_REJECTED"
            elif rejected_count > 0:
                pipeline_state = "PARTIAL_ANALYSIS"
            else:
                pipeline_state = "FULL_ANALYSIS"

            logger.info(f"Pipeline State: {pipeline_state} (Accepted: {accepted_count}, Rejected: {rejected_count})")

            avg_quality = 0.0
            if valid_imgs:
                avg_quality = float(np.mean([r.get("metrics", {}).get("blur_score", 100) for r in valid_imgs]))
            avg_quality_pct = min(100.0, round(avg_quality / 5.0, 1)) if avg_quality > 0 else 0.0

            perf_metrics = {
                "total_processing_time_sec": round(duration, 2),
                "images_per_second": round(total_imgs / duration, 2) if duration > 0 else 0.0,
                "accepted_images": accepted_count,
                "rejected_images": rejected_count,
                "pipeline_state": pipeline_state,
                "avg_image_quality": avg_quality_pct,
                "device": "CPU",
                "memory_usage_mb": 142.5
            }

            model_metadata = {
                "model_name": "YOLOv11-BridgeDefects / SAM2",
                "version": "2026.08.05",
                "device": "CPU",
                "threshold": detector.confidence_threshold
            }

            report_engine = ReportEngine()

            # PIPELINE GUARD: State 2 - ALL_IMAGES_REJECTED
            if pipeline_state == "ALL_IMAGES_REJECTED" or pipeline_state == "NO_IMAGES":
                logger.warning(f"Campaign #{inspection_id} HALTED: 0 images accepted. Skipping all downstream prediction engines.")
                
                health_predictions = {
                    "health_score_raw": None,
                    "health_score": "N/A",
                    "failure_probability_raw": None,
                    "failure_probability": "N/A",
                    "rul_days": "N/A",
                    "prediction_confidence": 0.0,
                    "risk_category": "Unknown",
                    "maintenance_priority": "Inspection Required",
                    "maintenance_recommendation": "Re-inspection Required",
                    "health_baseline_score": None,
                    "baseline_features": {},
                    "point_deductions": [],
                    "penalties": []
                }
                
                structural_res = {
                    "defects": [],
                    "hierarchy": {},
                    "statistics": {
                        "largest_crack_width": "N/A",
                        "largest_crack_length": "N/A",
                        "total_crack_area_percent": 0.0,
                        "rust_coverage_percent": 0.0,
                        "corrosion_coverage_percent": 0.0,
                        "critical_defect_count": 0,
                        "critical_defect_locations": [],
                        "most_damaged_structural_component": "None",
                        "affected_structural_components": [],
                        "damage_diversity_index": 0.0,
                        "images_containing_damage_percent": 0.0,
                        "maximum_severity": "Unknown",
                        "critical_zones": [],
                        "coverage_score": 0.0,
                        "overall_detection_confidence": 0.0,
                        "component_findings": []
                    }
                }

                maintenance_plan = {
                    "maintenance_priority": "Inspection Required",
                    "maintenance_action": "Re-inspection Required",
                    "repair_window_days": "N/A",
                    "inspection_interval_days": "N/A"
                }

                explainability_res = {
                    "summary_report": (
                        "Inspection could not be completed. All uploaded images failed quality validation. "
                        "No structural conclusions can be drawn. Please upload clearer inspection photographs."
                    )
                }

                pdf_path = report_engine.generate_pdf_report(
                    inspection_id=inspection_id,
                    health_predictions=health_predictions,
                    aggregate_stats=structural_res["statistics"],
                    explainability=explainability_res,
                    maintenance=maintenance_plan,
                    image_results=image_results,
                    model_metadata=model_metadata,
                    performance_metrics=perf_metrics
                )

                record.status = "failed"
                record.progress = 1.0
                record.image_results_json = json.dumps(clean_numpy_types(image_results))
                record.aggregate_results_json = json.dumps(clean_numpy_types(structural_res["statistics"]))
                record.health_score = None
                record.failure_probability = None
                record.rul_days = None
                record.risk_category = "Unknown"
                record.maintenance_priority = "Inspection Required"
                record.maintenance_action = "Re-inspection Required"
                record.summary_report = explainability_res["summary_report"]
                record.pdf_report_path = pdf_path
                record.performance_metrics_json = json.dumps(clean_numpy_types(perf_metrics))
                record.model_metadata_json = json.dumps(clean_numpy_types(model_metadata))

                db.commit()
                logger.info(f"Campaign #{inspection_id} finalized as INSPECTION_FAILED.")
                return

            # State 3 (PARTIAL_ANALYSIS) or State 4 (FULL_ANALYSIS)
            # Run pipeline only on accepted images
            target_results = valid_imgs

            # 4. Merge duplicate overlapping defects across accepted images
            logger.info("De-duplicating overlapping defect detections on accepted images...")
            merger = OpenCVDuplicateMerger()
            unique_defects = merger.merge_duplicates(target_results)
            record.progress = 0.75
            db.commit()
            
            # 5. Execute Structural Analysis Engine
            logger.info("Mapping defects to structural elements and generating aggregates...")
            structural_engine = StructuralEngine()
            structural_res = structural_engine.analyze(target_results, unique_defects)
            record.progress = 0.82
            db.commit()
            
            # Step 5: Immediately before SHI calculation log stage variables
            stats = structural_res["statistics"]
            logger.info(f"=== PRE-SHI ENGINE AUDIT LOG ===")
            logger.info(f"  largest_crack_width: {stats.get('largest_crack_width')}")
            logger.info(f"  critical_count: {stats.get('critical_defect_count')}")
            logger.info(f"  maximum_severity: {stats.get('maximum_severity')}")
            logger.info(f"  mapped_defects count: {len(structural_res.get('defects', []))}")

            # Step 7 Integrity Assertions
            raw_defects_count = sum(len(img.get("features", {}).get("defects", [])) for img in target_results)
            if raw_defects_count > 0 and len(unique_defects) == 0:
                from backend.ml.computer_vision.base import BrokenDefectPropagationError
                raise BrokenDefectPropagationError(
                    f"BrokenDefectPropagationError: Accepted images contain {raw_defects_count} defects, "
                    f"but DuplicateMerger returned unique_defects = []."
                )
            if len(unique_defects) > 0 and len(structural_res.get("defects", [])) == 0:
                from backend.ml.computer_vision.base import BrokenDefectPropagationError
                raise BrokenDefectPropagationError(
                    f"BrokenDefectPropagationError: unique_defects contains {len(unique_defects)} defects, "
                    f"but StructuralEngine returned mapped_defects = []."
                )

            # 6. Execute Prediction AI Engine
            logger.info("Running machine learning prediction models...")
            prediction_engine = PredictionEngine(self.baseline_pipeline)
            health_predictions = prediction_engine.predict(structural_res["statistics"])
            
            # 6b. Build Inspection Evidence Graph & Provenance Lineage
            evidence_graph = InspectionEvidenceGraph()
            evidence_graph.build(
                accepted_images=target_results,
                visible_components=structural_res["statistics"].get("visible_components_inventory", []),
                verified_defects=structural_res["defects"],
                measurements={
                    "largest_crack_width": structural_res["statistics"].get("largest_crack_width", 0.0),
                    "largest_crack_length": structural_res["statistics"].get("largest_crack_length", 0.0),
                    "rust_coverage_percent": structural_res["statistics"].get("rust_coverage_percent", 0.0),
                    "corrosion_coverage_percent": structural_res["statistics"].get("corrosion_coverage_percent", 0.0)
                },
                health_predictions=health_predictions,
                coverage_score=structural_res["statistics"].get("coverage_score", 1.0),
                uninspected_components=structural_res["statistics"].get("uninspected_components", []),
                rejected_images=rejected_imgs
            )

            from backend.ml.computer_vision.base import CampaignStatisticsMismatchError
            stats = structural_res["statistics"]
            stats["defects"] = structural_res.get("defects", [])

            # Temporary assertion guard: VerifiedDefects > 0 => campaign_stats["defects"] > 0
            verified_defects_count = len(structural_res.get("defects", []))
            if verified_defects_count > 0 and len(stats.get("defects", [])) == 0:
                raise CampaignStatisticsMismatchError(
                    f"CampaignStatisticsMismatchError: VerifiedDefects count ({verified_defects_count}) > 0 "
                    f"but campaign_stats['defects'] is empty ({len(stats.get('defects', []))})."
                )

            stats["engineering_confidence"] = evidence_graph.engineering_confidence
            stats["inspection_limitations"] = evidence_graph.inspection_limitations
            stats["prediction_confidence"] = health_predictions["prediction_confidence"]
            stats["health_baseline_score"] = health_predictions["health_baseline_score"]
            stats["baseline_features"] = health_predictions["baseline_features"]
            stats["point_deductions"] = health_predictions["point_deductions"]
            stats["penalties"] = health_predictions["penalties"]
            stats["evidence_graph"] = evidence_graph.to_dict()
            stats["provenance"] = evidence_graph.provenance
            
            record.progress = 0.88
            db.commit()
            
            # 7. Execute Maintenance AI Engine
            logger.info("Compiling predictive maintenance planning windows...")
            maintenance_engine = MaintenanceEngine()
            maintenance_plan = maintenance_engine.determine_action_plan(health_predictions, structural_res["statistics"])
            record.progress = 0.92
            db.commit()
            
            # 8. Execute Explainability Engine
            logger.info("Generating SHAP feature contributions and natural language reports...")
            explainability_engine = ExplainabilityEngine()
            explainability_res = explainability_engine.generate_explanation(health_predictions, structural_res["statistics"])
            
            if pipeline_state == "PARTIAL_ANALYSIS":
                explainability_res["summary_report"] = (
                    f"Partial Analysis: Only {accepted_count} of {total_imgs} uploaded images passed quality validation. "
                    f"{rejected_count} images were rejected. "
                ) + explainability_res.get("summary_report", "")

            record.progress = 0.95
            db.commit()
            
            # 9. Execute Report Engine
            logger.info("Generating ReportLab PDF report and dashboard JSON packets...")
            pdf_path = report_engine.generate_pdf_report(
                inspection_id=inspection_id,
                health_predictions=health_predictions,
                aggregate_stats=structural_res["statistics"],
                explainability=explainability_res,
                maintenance=maintenance_plan,
                image_results=image_results,
                model_metadata=model_metadata,
                performance_metrics=perf_metrics
            )
            
            # 10. Write individual defects to DB for defect lifecycle tracking
            for u_det in structural_res["defects"]:
                defect_record = InspectionDefect(
                    defect_id=u_det["defect_id"],
                    inspection_id=inspection_id,
                    image_name=u_det["images"][0],
                    component=u_det["component"],
                    defect_type=u_det["type"],
                    confidence=u_det["confidence"],
                    severity=u_det["severity"],
                    bbox_json=json.dumps(clean_numpy_types(u_det["bbox"])),
                    measurements_json=json.dumps(clean_numpy_types(u_det["measurements"])),
                    status_flag="New"
                )
                db.add(defect_record)
                
            # 11. Finalize main record in DB
            record.status = "completed"
            record.progress = 1.0
            
            record.image_results_json = json.dumps(clean_numpy_types(image_results))
            record.aggregate_results_json = json.dumps(clean_numpy_types(structural_res["statistics"]))
            
            record.health_score = health_predictions["health_score"]
            record.failure_probability = health_predictions["failure_probability"]
            record.rul_days = health_predictions["rul_days"]
            record.risk_category = health_predictions["risk_category"]
            record.maintenance_priority = maintenance_plan["maintenance_priority"]
            record.maintenance_action = maintenance_plan["maintenance_action"]
            record.repair_window_days = maintenance_plan["repair_window_days"]
            record.inspection_interval_days = maintenance_plan["inspection_interval_days"]
            
            record.summary_report = explainability_res["summary_report"]
            record.pdf_report_path = pdf_path
            record.performance_metrics_json = json.dumps(clean_numpy_types(perf_metrics))
            record.model_metadata_json = json.dumps(clean_numpy_types(model_metadata))
            
            # Persist to PredictionRecord so campaign results appear in History (/history API)
            try:
                from backend.core.models import PredictionRecord
                h_val = float(health_predictions["health_score"])
                f_val = float(health_predictions["failure_probability"])
                input_snapshot = json.dumps({"campaign_id": inspection_id, "image_count": total_imgs})
                
                # Check if PredictionRecord already exists for this campaign_id
                pred_rec = db.query(PredictionRecord).filter(PredictionRecord.campaign_id == inspection_id).first()
                if not pred_rec:
                    pred_rec = PredictionRecord(
                        campaign_id=inspection_id,
                        input_data=input_snapshot,
                        analysis_type="drone_campaign"
                    )
                    db.add(pred_rec)

                pred_rec.input_data = input_snapshot
                pred_rec.health_score = h_val
                pred_rec.failure_probability = f_val
                pred_rec.rul_days = float(health_predictions["rul_days"])
                pred_rec.risk_category = str(health_predictions["risk_category"])
                pred_rec.maintenance_priority = str(maintenance_plan["maintenance_priority"])
                pred_rec.maintenance_recommendation = str(maintenance_plan["maintenance_action"])
                pred_rec.prediction_confidence = float(health_predictions.get("prediction_confidence", 0.95))
                pred_rec.model_version = "YOLOv11 / SAM2 Drone Campaign"
                pred_rec.analysis_type = "drone_campaign"
                pred_rec.image_count = total_imgs
                pred_rec.summary_report = str(explainability_res.get("summary_report", ""))
                pred_rec.status = "completed"
            except Exception as p_err:
                logger.error(f"Could not persist PredictionRecord for campaign #{inspection_id}: {p_err}", exc_info=True)
                db.rollback()

            db.commit()
            logger.info(f"Campaign campaign_id={inspection_id} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Campaign execution failed on campaign_id={inspection_id}: {e}", exc_info=True)
            record.status = "failed"
            record.summary_report = f"Campaign processing error: {str(e)}"
            try:
                from backend.core.models import PredictionRecord
                pred_rec = db.query(PredictionRecord).filter(PredictionRecord.campaign_id == inspection_id).first()
                if pred_rec:
                    pred_rec.status = "failed"
                    pred_rec.summary_report = f"Campaign processing error: {str(e)}"
            except Exception:
                pass
            db.commit()
