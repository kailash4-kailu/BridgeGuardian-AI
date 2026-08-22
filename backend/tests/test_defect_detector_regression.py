"""
BridgeGuardian AI — Test Suite: Defect Detector Regression & Pipeline Audit
Automatically tests defect detection on synthetic and sample defect images (cracks, rust, spalling).
Asserts that raw candidate defect count > 0 and verified defect count > 0.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from backend.ml.computer_vision.damage_detector import DamageDetector
from backend.ml.computer_vision.bridge_detector import BridgeDetector
from backend.ml.computer_vision.detector import YOLODetector
from backend.ml.computer_vision.vision_engine import VisionEngine
from backend.ml.computer_vision.base import BaseFeatureExtractor, BaseImageQualityChecker


@pytest.fixture
def crack_test_image(tmp_path):
    """Creates a sample concrete deck image with a distinct dark crack."""
    file_path = tmp_path / "sample_concrete_crack.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 180
    # Draw prominent dark crack paths
    pts = np.array([[120, 100], [200, 240], [350, 280], [520, 440]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (10, 10, 10), 6)
    cv2.imwrite(str(file_path), img)
    return str(file_path)


@pytest.fixture
def rust_test_image(tmp_path):
    """Creates a sample steel girder image with distinct reddish-orange rust patch."""
    file_path = tmp_path / "sample_steel_rust.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 140
    # Draw prominent rust patch (BGR reddish-brown color)
    cv2.rectangle(img, (250, 200), (450, 380), (25, 60, 180), -1)
    cv2.imwrite(str(file_path), img)
    return str(file_path)


def test_crack_detection_regression(crack_test_image):
    """
    Test 1: Ensures concrete crack image produces raw defect candidates and verified defect bounding boxes.
    """
    image = cv2.imread(crack_test_image)
    detector = DamageDetector(min_confidence=0.30)
    bridge_detector = BridgeDetector()
    bridge_info = bridge_detector.detect_bridge(image)

    res = detector.detect_all_damage(image, bridge_info, visible_components=[{"label": "Deck", "bbox": [0,0,800,600]}])

    raw_cands = res.get("raw_candidates", [])
    verified_bboxes = res.get("bboxes", [])

    assert len(raw_cands) > 0, "Raw candidate defect count must be > 0 for crack image"
    assert len(verified_bboxes) > 0, "Verified defect count must be > 0 for crack image"
    assert any(b["label"] == "Crack" for b in verified_bboxes)


def test_rust_detection_regression(rust_test_image):
    """
    Test 2: Ensures steel rust image produces raw defect candidates and verified rust bounding boxes.
    """
    image = cv2.imread(rust_test_image)
    detector = DamageDetector(min_confidence=0.30)
    bridge_detector = BridgeDetector()
    bridge_info = bridge_detector.detect_bridge(image)

    res = detector.detect_all_damage(image, bridge_info, visible_components=[{"label": "Steel Girder", "bbox": [0,0,800,600]}])

    raw_cands = res.get("raw_candidates", [])
    verified_bboxes = res.get("bboxes", [])

    assert len(raw_cands) > 0, "Raw candidate defect count must be > 0 for rust image"
    assert len(verified_bboxes) > 0, "Verified defect count must be > 0 for rust image"
    assert any(b["label"] == "Rust/Corrosion" for b in verified_bboxes)


def test_model_startup_class_audit():
    """
    Test 3: Audits model startup and class name compatibility.
    """
    comp_det = YOLODetector()
    damage_det = DamageDetector()

    comp_status = comp_det.log_model_status()
    damage_status = damage_det.log_model_status()

    assert comp_status["is_loaded"] is True
    assert damage_status["is_loaded"] is True
    assert "Crack" in damage_status["supported_classes"]
    assert "Rust/Corrosion" in damage_status["supported_classes"]
    assert "Spalling" in damage_status["supported_classes"]
