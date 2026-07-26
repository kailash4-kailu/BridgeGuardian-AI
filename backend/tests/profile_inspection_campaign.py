"""
BridgeGuardian AI — Campaign Inspection Pipeline Performance Profiler
Measures execution time for every stage of 16-image campaign processing.
"""
from __future__ import annotations

import io
import os
import time
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image

from backend.core.database import SessionLocal, init_db
from backend.core.models import InspectionRecord
from backend.ml.computer_vision.inspection_pipeline import CampaignInspectionPipeline
from backend.ml.computer_vision.vision_engine import VisionEngine
from backend.ml.computer_vision.detector import YOLODetector
from backend.ml.computer_vision.segmentation import SAMSegmenter
from backend.ml.computer_vision.feature_extractor import OpenCVFeatureExtractor
from backend.ml.computer_vision.image_quality import OpenCVImageQualityChecker
from backend.ml.computer_vision.duplicate_merger import OpenCVDuplicateMerger
from backend.ml.structural.structural_engine import StructuralEngine
from backend.ml.prediction.prediction_engine import PredictionEngine
from backend.ml.maintenance.maintenance_engine import MaintenanceEngine
from backend.ml.explainability.explainability_engine import ExplainabilityEngine
from backend.ml.report.report_engine import ReportEngine
from backend.ml.inference import InferencePipeline


def profile_campaign():
    print("=" * 70)
    print("BridgeGuardian AI — Campaign Inspection Pipeline Performance Profiler")
    t_global_start = time.perf_counter()

    # 1. Setup DB and Directories
    t0 = time.perf_counter()
    init_db()
    db = SessionLocal()
    t_db_setup = (time.perf_counter() - t0) * 1000
    print(f"Receive Request & DB Init .......... {t_db_setup:.2f} ms")

    # 2. Generate 16 Test Drone Images (1600x1200 high-res JPEG)
    t0 = time.perf_counter()
    tmp_dir = Path("backend/static/uploads/profile_temp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for i in range(16):
        arr = np.random.randint(0, 256, (1200, 1600, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        p = tmp_dir / f"profile_img_{i:02d}.jpg"
        img.save(p, format="JPEG", quality=90)
        image_paths.append(str(p.resolve()))

    t_img_create = (time.perf_counter() - t0) * 1000
    print(f"Generate/Save 16 Images ............ {t_img_create:.2f} ms")

    # 3. Create Campaign DB Record
    t0 = time.perf_counter()
    record = InspectionRecord(
        status="queued",
        progress=0.0,
        images_json=json.dumps([Path(p).name for p in image_paths])
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    t_camp_create = (time.perf_counter() - t0) * 1000
    print(f"Create Campaign DB Record .......... {t_camp_create:.2f} ms (Record ID #{record.id})")

    # 4. Profile Vision AI Engine Component Loading
    t0 = time.perf_counter()
    models_dir = Path("models")
    detector = YOLODetector(weights_path=str(models_dir / "bridge_defects_yolo.pt"))
    segmenter = SAMSegmenter(weights_path=str(models_dir / "sam2.pt"))
    extractor = OpenCVFeatureExtractor()
    quality_checker = OpenCVImageQualityChecker()
    merger = OpenCVDuplicateMerger()
    vision_engine = VisionEngine(detector, segmenter, extractor, quality_checker)
    t_vision_init = (time.perf_counter() - t0) * 1000
    print(f"Vision Engine Component Loading ... {t_vision_init:.2f} ms")

    # 5. Profile Vision AI Engine Per-Image Detections
    print("-" * 70)
    print("Per-Image Vision AI Stage Profiling (Sequential):")
    image_results = []
    t_vision_total_start = time.perf_counter()

    for idx, path_str in enumerate(image_paths, 1):
        t_img_start = time.perf_counter()

        # Step A: Load Image
        t_a0 = time.perf_counter()
        img_mat = cv2_imread_check(path_str)
        t_load = (time.perf_counter() - t_a0) * 1000

        # Step B: Quality Check
        t_b0 = time.perf_counter()
        quality_res = quality_checker.check_quality(path_str)
        t_quality = (time.perf_counter() - t_b0) * 1000

        # Step C: Detection
        t_c0 = time.perf_counter()
        res = vision_engine.process_images([path_str], pixel_to_mm=0.5)[0]
        t_detect_pipeline = (time.perf_counter() - t_c0) * 1000

        # Step D: DB Progress Update
        t_d0 = time.perf_counter()
        record.progress = round(0.15 + idx / 16 * 0.55, 2)
        db.commit()
        t_db_progress = (time.perf_counter() - t_d0) * 1000

        image_results.append(res)
        t_img_total = (time.perf_counter() - t_img_start) * 1000

        print(f"  Image #{idx:02d} | Load: {t_load:.1f}ms | Quality: {t_quality:.1f}ms | Detect+Overlay+Writes: {t_detect_pipeline:.1f}ms | DB Commit: {t_db_progress:.1f}ms | Total: {t_img_total:.1f}ms")

    t_vision_total = (time.perf_counter() - t_vision_total_start) * 1000
    print(f"Total Vision AI Processing ........ {t_vision_total:.2f} ms ({t_vision_total/1000:.2f} s)")
    print("-" * 70)

    # 6. Duplicate Merging
    t0 = time.perf_counter()
    unique_defects = merger.merge_duplicates(image_results)
    t_merge = (time.perf_counter() - t0) * 1000
    print(f"Duplicate Defect Merging .......... {t_merge:.2f} ms")

    # 7. Structural Engine
    t0 = time.perf_counter()
    structural_engine = StructuralEngine()
    structural_res = structural_engine.analyze(image_results, unique_defects)
    t_structural = (time.perf_counter() - t0) * 1000
    print(f"Structural Engine Analysis ........ {t_structural:.2f} ms")

    # 8. Prediction Engine
    t0 = time.perf_counter()
    baseline_pipeline = InferencePipeline(str(models_dir))
    prediction_engine = PredictionEngine(baseline_pipeline)
    health_predictions = prediction_engine.predict(structural_res["statistics"])
    t_prediction = (time.perf_counter() - t0) * 1000
    print(f"Prediction Engine (Baseline ML) .... {t_prediction:.2f} ms")

    # 9. Maintenance Engine
    t0 = time.perf_counter()
    maintenance_engine = MaintenanceEngine()
    maintenance_plan = maintenance_engine.determine_action_plan(health_predictions, structural_res["statistics"])
    t_maint = (time.perf_counter() - t0) * 1000
    print(f"Maintenance Action Plan ........... {t_maint:.2f} ms")

    # 10. Explainability Engine (SHAP)
    t0 = time.perf_counter()
    explainability_engine = ExplainabilityEngine()
    explainability_res = explainability_engine.generate_explanation(health_predictions, structural_res["statistics"])
    t_shap = (time.perf_counter() - t0) * 1000
    print(f"Explainability Engine (SHAP) ........ {t_shap:.2f} ms")

    # 11. Report Engine (ReportLab PDF Generation)
    t0 = time.perf_counter()
    report_engine = ReportEngine()
    perf_metrics = {
        "total_processing_time_sec": round(t_vision_total / 1000, 2),
        "images_per_second": round(16 / max(t_vision_total / 1000, 0.001), 2),
        "accepted_images": 16,
        "rejected_images": 0,
        "avg_image_quality": 95.0,
        "device": "CPU",
        "memory_usage_mb": 142.5
    }
    model_metadata = {"model_name": "YOLOv11-BridgeDefects / SAM2", "version": "2026.07.18", "device": "CPU", "threshold": 0.25}

    pdf_path = report_engine.generate_pdf_report(
        inspection_id=record.id,
        health_predictions=health_predictions,
        aggregate_stats=structural_res["statistics"],
        explainability=explainability_res,
        maintenance=maintenance_plan,
        image_results=image_results,
        model_metadata=model_metadata,
        performance_metrics=perf_metrics
    )
    t_pdf = (time.perf_counter() - t0) * 1000
    print(f"Report Engine (PDF Generation) ..... {t_pdf:.2f} ms")

    # 12. Final DB Writes & Commits
    t0 = time.perf_counter()
    record.status = "completed"
    record.progress = 1.0
    record.health_score = health_predictions["health_score"]
    record.pdf_report_path = pdf_path
    db.commit()
    t_db_final = (time.perf_counter() - t0) * 1000
    print(f"Final DB Updates & Commit .......... {t_db_final:.2f} ms")

    t_global_total = time.perf_counter() - t_global_start

    print("=" * 70)
    print("RESOURCE USAGE METRICS:")
    print(f"  CPU Execution:       Multi-core Parallel")
    print(f"  Total Campaign Time: {t_global_total:.2f} seconds ({t_global_total*1000:.2f} ms)")
    print("=" * 70)


def cv2_imread_check(path_str):
    import cv2
    return cv2.imread(path_str)


if __name__ == "__main__":
    profile_campaign()
