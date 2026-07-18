"""
BridgeGuardian AI — Computer Vision Inspection Pipeline Tests
Verifies the CV pipeline components, feature extractor, and HTTP endpoints.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.main import app
from backend.ml.computer_vision.bridge_detector import BridgeDetector
from backend.ml.computer_vision.damage_detector import DamageDetector
from backend.ml.computer_vision.image_measurements import ImageMeasurements
from backend.ml.computer_vision.feature_extractor import ImageFeatureExtractor
from backend.ml.computer_vision.vision_inference import VisionInferencePipeline

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def init_test_db():
    from backend.core.database import init_db
    init_db()

@pytest.fixture
def synthetic_bridge_image(tmp_path) -> Path:
    """Generates a synthetic bridge image with cracks, rust, vegetation, and a tilt."""
    img_h, img_w = 400, 600
    # Concrete gray background
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 180
    
    # Draw bridge span (dark gray rectangle)
    cv2.rectangle(img, (50, 150), (550, 280), (100, 100, 100), -1)
    
    # Draw horizontal structural boundaries (Hough line candidates)
    cv2.line(img, (50, 150), (550, 150), (40, 40, 40), 2)
    cv2.line(img, (50, 280), (550, 280), (40, 40, 40), 2)
    
    # Draw Cracks: black jagged lines
    cv2.line(img, (120, 160), (130, 200), (20, 20, 20), 2)
    cv2.line(img, (130, 200), (145, 240), (20, 20, 20), 2)
    
    # Draw Rust: reddish-brown orange spots (HSV approx H=10, S=200, V=180 -> BGR approx B=30, G=100, R=200)
    cv2.circle(img, (300, 200), 15, (25, 95, 205), -1)
    cv2.circle(img, (315, 210), 10, (25, 95, 205), -1)
    
    # Draw Vegetation: green patch (HSV approx H=60, S=180, V=150 -> BGR approx B=30, G=160, R=40)
    cv2.circle(img, (450, 160), 12, (30, 170, 40), -1)
    
    # Draw Missing Bolt Holes: small high-contrast dark circles (radius 6 for clean contour area)
    cv2.circle(img, (80, 180), 6, (10, 10, 10), -1)
    cv2.circle(img, (80, 200), 6, (10, 10, 10), -1)
    
    img_path = tmp_path / "test_bridge.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


def test_bridge_detector(synthetic_bridge_image):
    img = cv2.imread(str(synthetic_bridge_image))
    detector = BridgeDetector()
    results = detector.detect_bridge(img)
    
    assert results["detected"] is True
    assert results["confidence"] >= 0.5
    assert len(results["bbox"]) == 4
    assert results["mask"].shape == img.shape[:2]


def test_damage_detector(synthetic_bridge_image):
    img = cv2.imread(str(synthetic_bridge_image))
    detector = BridgeDetector()
    bridge_info = detector.detect_bridge(img)
    
    damage_detector = DamageDetector()
    results = damage_detector.detect_all_damage(img, bridge_info)
    
    assert "masks" in results
    assert "percentages" in results
    assert "counts" in results
    assert "bboxes" in results
    
    # Vegetation, rust, and cracks should be non-zero
    assert results["percentages"]["vegetation_percent"] > 0
    assert results["percentages"]["corrosion_percent"] > 0
    assert results["percentages"]["crack_density"] > 0
    assert results["counts"]["missing_bolts"] > 0


def test_measurements(synthetic_bridge_image):
    img = cv2.imread(str(synthetic_bridge_image))
    detector = BridgeDetector()
    bridge_info = detector.detect_bridge(img)
    
    damage_detector = DamageDetector()
    damage_results = damage_detector.detect_all_damage(img, bridge_info)
    
    measurer = ImageMeasurements()
    crack_len, crack_width = measurer.estimate_crack_dimensions(damage_results["masks"]["cracks"])
    assert crack_len > 0
    assert crack_width > 0
    
    tilt = measurer.estimate_bridge_tilt(bridge_info["structural_lines"])
    assert isinstance(tilt, float)


def test_feature_extractor(synthetic_bridge_image):
    extractor = ImageFeatureExtractor()
    features, raw = extractor.extract_features(str(synthetic_bridge_image))
    
    expected_keys = {
        "crack_density", "crack_length", "crack_width", "corrosion_percent",
        "spalling_percent", "vegetation_percent", "leakage_percent",
        "tilt_angle", "missing_components", "damage_area_percent"
    }
    assert set(features.keys()) == expected_keys
    assert features["corrosion_percent"] > 0.0
    assert features["crack_density"] > 0.0


def test_vision_endpoints(synthetic_bridge_image):
    # Ensure a pipeline model exists in mock state or training state (inference_pipeline gets mocked/loaded)
    # We can load the global pipeline
    from backend.main import inference_pipeline
    if not inference_pipeline.is_ready:
        try:
            inference_pipeline.load()
        except Exception:
            pytest.skip("Tabular ML models not trained, skipping API endpoints integration test.")
            
    # 1. Test POST /upload-image
    with open(synthetic_bridge_image, "rb") as f:
        file_content = f.read()
        
    response = client.post(
        "/api/v1/vision/upload-image",
        files={"files": ("test_bridge.jpg", io.BytesIO(file_content), "image/jpeg")}
    )
    assert response.status_code == 200
    upload_data = response.json()
    assert len(upload_data) == 1
    image_id = upload_data[0]["image_id"]
    assert image_id.endswith("test_bridge.jpg")

    # 2. Test POST /vision-predict
    predict_response = client.post(
        "/api/v1/vision/vision-predict",
        json={"image_id": image_id, "pixel_to_mm": 0.5}
    )
    assert predict_response.status_code == 200
    pred_data = predict_response.json()
    assert "prediction_id" in pred_data
    assert "features" in pred_data
    assert "predictions" in pred_data
    assert "visualizations" in pred_data
    assert "original" in pred_data["visualizations"]
    assert "segmentation" in pred_data["visualizations"]
    prediction_id = pred_data["prediction_id"]

    # 3. Test GET /generate-report
    report_response = client.get(
        f"/api/v1/vision/generate-report?image_id={image_id}&prediction_id={prediction_id}"
    )
    assert report_response.status_code == 200
    assert report_response.headers["content-type"] == "application/pdf"
    assert len(report_response.content) > 1000  # Non-trivial PDF output
