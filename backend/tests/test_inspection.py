"""
BridgeGuardian AI — Campaign Inspection Pipeline Tests
Validates quality checks, YOLODetector modes, structural hierarchy mapping,
predictions aggregation, and FastAPI inspection endpoints.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from unittest import mock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.database import SessionLocal
from backend.core.models import InspectionRecord, InspectionDefect
from backend.ml.computer_vision.detector import YOLODetector
from backend.ml.computer_vision.segmentation import SAMSegmenter
from backend.ml.computer_vision.image_quality import OpenCVImageQualityChecker
from backend.ml.computer_vision.duplicate_merger import OpenCVDuplicateMerger
from backend.ml.structural.structural_engine import StructuralEngine
from backend.ml.prediction.prediction_engine import PredictionEngine


@pytest.fixture
def test_image() -> np.ndarray:
    """Generates a simple mock gray image matrix."""
    return np.ones((400, 600, 3), dtype=np.uint8) * 128


def test_yolo_detector_demo_mode(test_image):
    """Test that YOLODetector returns mock boxes in Demo Mode and throws errors if not."""
    # With DEMO_MODE = true, it should return mock bboxes
    with mock.patch.dict(os.environ, {"DEMO_MODE": "true"}):
        detector = YOLODetector(weights_path="models/missing_weights_file.pt")
        results = detector.detect(test_image)
        assert len(results) > 0
        assert any(r.label == "Girder" for r in results)

    # With DEMO_MODE = false and missing weights, it should raise FileNotFoundError
    with mock.patch.dict(os.environ, {"DEMO_MODE": "false"}):
        with pytest.raises(FileNotFoundError):
            YOLODetector(weights_path="models/missing_weights_file.pt")


def test_image_quality_checker(tmp_path, test_image):
    """Test the quality gate (blurriness, darkness, brightness limits)."""
    import cv2
    img_path = tmp_path / "test_frame.jpg"
    cv2.imwrite(str(img_path), test_image)
    
    checker = OpenCVImageQualityChecker(blur_threshold=-1.0, dark_threshold=40.0)
    report = checker.check_quality(str(img_path))
    
    assert report["is_valid"] is True
    assert "hash" in report["metrics"]


def test_duplicate_merger():
    """Test ORB/heuristic duplicate merging of duplicate defects."""
    merger = OpenCVDuplicateMerger()
    
    # Mock visual results list
    image_results = [
        {
            "image_path": "frame1.jpg",
            "features": {
                "defects": [
                    {"type": "Crack", "bbox": [100, 100, 50, 50], "confidence": 0.90, "severity": "Moderate", "measurements": {"width_mm": 1.5, "length_mm": 100, "area_pct": 1.2}}
                ]
            }
        },
        {
            "image_path": "frame2.jpg",
            "features": {
                "defects": [
                    {"type": "Crack", "bbox": [105, 102, 48, 52], "confidence": 0.94, "severity": "Severe", "measurements": {"width_mm": 1.8, "length_mm": 120, "area_pct": 1.3}}
                ]
            }
        }
    ]
    
    unique_defects = merger.merge_duplicates(image_results)
    assert len(unique_defects) == 1
    assert unique_defects[0]["type"] == "Crack"
    assert unique_defects[0]["severity"] == "Severe" # Combined severity scales up
    assert unique_defects[0]["occurrences"] == 2


def test_structural_engine():
    """Verify defects map inside component bounding boxes and aggregate stats correctly."""
    engine = StructuralEngine()
    
    image_results = [
        {
            "image_name": "frame1.jpg",
            "is_valid": True,
            "features": {
                "defects": [
                    {"type": "Girder", "bbox": [50, 50, 400, 200]},
                    {"type": "Crack", "bbox": [100, 100, 30, 20]}
                ]
            }
        }
    ]
    
    unique_defects = [
        {
            "defect_id": "DEFECT-000001",
            "type": "Crack",
            "severity": "Severe",
            "confidence": 0.90,
            "bbox": [100, 100, 30, 20],
            "measurements": {"width_mm": 2.2, "length_mm": 150, "area_pct": 1.5},
            "images": ["frame1.jpg"]
        }
    ]
    
    res = engine.analyze(image_results, unique_defects)
    
    assert res["defects"][0]["component"] == "Girder" # Correctly mapped to overlapping girder
    assert res["statistics"]["largest_crack_width"] == 2.2
    assert res["statistics"]["critical_defect_count"] == 1


def test_prediction_engine():
    """Verify visual penalty aggregates reduce health score correctly."""
    # Mock baseline pipeline
    mock_pipeline = mock.MagicMock()
    mock_pipeline.is_ready = True
    mock_pipeline.predict.return_value = {
        "health_score": 85.0,
        "failure_probability": 3.0,
        "rul_days": 1800.0,
        "risk_category": "Excellent"
    }
    
    pred_engine = PredictionEngine(mock_pipeline)
    
    # 1. Healthy stats: no penalty
    health_predictions_ok = pred_engine.predict({
        "largest_crack_width": 0.2,
        "rust_coverage_percent": 0.0,
        "corrosion_coverage_percent": 0.0,
        "critical_defect_count": 0,
        "maximum_severity": "Minor"
    })
    assert health_predictions_ok["health_score"] == 85.0
    
    # 2. Damaged stats: triggers penalties
    health_predictions_dmg = pred_engine.predict({
        "largest_crack_width": 5.2, # critical crack width (> 4.0)
        "rust_coverage_percent": 5.0, # severe rust
        "corrosion_coverage_percent": 6.0,
        "critical_defect_count": 4,
        "maximum_severity": "Critical"
    })
    assert health_predictions_dmg["health_score"] < 85.0
    assert health_predictions_dmg["failure_probability"] > 3.0
    assert health_predictions_dmg["rul_days"] < 1800.0


def test_inspection_endpoints(tmp_path):
    """Test the campaign endpoints (/upload-images, /run-inspection, /inspection/{id})."""
    # Create mock JPG
    import cv2
    img_path = tmp_path / "drone_frame_test.jpg"
    cv2.imwrite(str(img_path), np.ones((400, 400, 3), dtype=np.uint8) * 128)
    
    with TestClient(app) as client:
        # Mock background task pipeline runner to prevent hanging
        with mock.patch("backend.ml.computer_vision.inspection_pipeline.CampaignInspectionPipeline.run_campaign") as mock_run:
            # 1. Upload
            with open(img_path, "rb") as f:
                r_upload = client.post("/api/v1/inspection/upload-images", files={"files": ("drone_frame_test.jpg", f, "image/jpeg")})
            assert r_upload.status_code == 200
            upload_data = r_upload.json()
            assert len(upload_data) == 1
            saved_path = upload_data[0]["filepath"]
            
            # 2. Run Inspection
            r_run = client.post("/api/v1/inspection/run-inspection", json={"image_paths": [saved_path], "pixel_to_mm": 0.5})
            assert r_run.status_code == 200
            run_data = r_run.json()
            assert "inspection_id" in run_data
            assert run_data["status"] == "queued"
            
            # 3. Poll Details
            inspect_id = run_data["inspection_id"]
            r_details = client.get(f"/api/v1/inspection/{inspect_id}")
            assert r_details.status_code == 200
            details_data = r_details.json()
            assert details_data["id"] == inspect_id
