"""
BridgeGuardian AI — End-to-End Defect Data Propagation Test Suite
Verifies that defect predictions are preserved through every stage:
DamageDetector -> FeatureExtractor -> DuplicateMerger -> StructuralEngine -> PredictionEngine -> ReportEngine.
"""

import os
import cv2
import numpy as np
import pytest

from backend.ml.computer_vision.feature_extractor import ImageFeatureExtractor
from backend.ml.computer_vision.vision_engine import VisionEngine
from backend.ml.computer_vision.detector import YOLODetector
from backend.ml.computer_vision.segmentation import SAMSegmenter
from backend.ml.computer_vision.image_quality import OpenCVImageQualityChecker
from backend.ml.computer_vision.duplicate_merger import OpenCVDuplicateMerger
from backend.ml.structural.structural_engine import StructuralEngine
from backend.ml.prediction.prediction_engine import PredictionEngine
from backend.ml.inference import InferencePipeline
from backend.ml.report.report_engine import ReportEngine


@pytest.fixture
def crack_test_image(tmp_path):
    img_path = str(tmp_path / "crack_test.jpg")
    img = np.ones((600, 800, 3), dtype=np.uint8) * 175
    pts = np.array([[150, 120], [240, 260], [390, 310], [580, 480]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (15, 15, 15), 6)
    cv2.imwrite(img_path, img)
    return img_path


def test_end_to_end_defect_propagation(crack_test_image, tmp_path):
    # 1. DamageDetector & VisionEngine Execution
    detector = YOLODetector("models/bridge_defects_yolo.pt")
    segmenter = SAMSegmenter("models/sam2.pt")
    extractor = ImageFeatureExtractor()
    quality_checker = OpenCVImageQualityChecker()

    vision_engine = VisionEngine(detector, segmenter, extractor, quality_checker)
    image_results = vision_engine.process_images([crack_test_image], pixel_to_mm=0.5)

    assert len(image_results) == 1
    res = image_results[0]
    assert res["is_valid"] is True

    # 2. FeatureExtractor Output Check
    features = res["features"]
    assert "defects" in features, "FeatureExtractor MUST include 'defects' key in features"
    assert len(features["defects"]) > 0, "FeatureExtractor returned 0 defects for crack test image"
    assert features["crack_width"] > 0, "FeatureExtractor crack_width must be > 0"

    # 3. DuplicateMerger Execution
    merger = OpenCVDuplicateMerger()
    unique_defects = merger.merge_duplicates(image_results)
    assert len(unique_defects) > 0, "DuplicateMerger dropped defects (unique_defects is empty)"
    assert unique_defects[0]["type"] == "Crack"

    # 4. StructuralEngine Execution
    structural_engine = StructuralEngine()
    structural_res = structural_engine.analyze(image_results, unique_defects)

    stats = structural_res["statistics"]
    mapped_defects = structural_res["defects"]

    assert len(mapped_defects) == len(unique_defects)
    assert stats["largest_crack_width"] > 0, f"largest_crack_width must be > 0, got {stats['largest_crack_width']}"
    assert stats["maximum_severity"] != "None", f"maximum_severity cannot be 'None' when defects exist"

    # 5. PredictionEngine Execution & SHI Reduction
    pipeline = InferencePipeline("models")
    prediction_engine = PredictionEngine(pipeline)
    health_predictions = prediction_engine.predict(stats)

    shi = health_predictions["health_score"]
    assert shi < 100.0, f"SHI score must drop below 100% when cracks are detected, got {shi}%"
    assert health_predictions["penalties"]["crack_penalty"] < 1.0, "crack_penalty must be < 1.0"

    # 6. ReportEngine PDF Generation
    report_engine = ReportEngine(reports_dir=str(tmp_path / "reports"))
    pdf_path = report_engine.generate_pdf_report(
        inspection_id=999,
        health_predictions=health_predictions,
        aggregate_stats=stats,
        explainability={"summary_report": "Defect propagation test report"},
        maintenance={"maintenance_priority": "High", "maintenance_action": "Repair"},
        image_results=image_results,
        model_metadata={"model_name": "TestModel"},
        performance_metrics={"device": "CPU"}
    )

    assert os.path.exists(pdf_path), "PDF report file was not created"
    assert os.path.getsize(pdf_path) > 1000, "PDF report file is empty or corrupted"
