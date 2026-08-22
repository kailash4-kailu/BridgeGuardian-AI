"""
BridgeGuardian AI — Test Suite: Final Vision Pipeline Audit & Benchmark Dataset Validation
Tests Priorities 1 through 8:
- Acceptance of long-shot drone imagery (removal of 25% bridge area cutoff)
- BridgePresenceClassifier & Adaptive Quality Score
- Defect Detection Recall on Crack, Rust, and Spalling images
- Rejection of Random Non-Bridge Images (smooth walls, sky, roads)
- Regression Guard Fail-Build assertions
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from backend.ml.computer_vision.image_quality import OpenCVImageQualityChecker, BridgePresenceClassifier
from backend.ml.computer_vision.damage_detector import DamageDetector
from backend.ml.computer_vision.bridge_detector import BridgeDetector
from backend.ml.computer_vision.feature_extractor import ImageFeatureExtractor


@pytest.fixture
def long_shot_drone_bridge_image(tmp_path):
    """Creates a synthetic long-distance drone bridge image where structure occupies ~15% of frame."""
    file_path = tmp_path / "long_shot_drone_bridge.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 190
    # Draw distant truss bridge structure (15% coverage)
    cv2.rectangle(img, (200, 350), (600, 420), (50, 50, 50), -1)
    for x in range(200, 600, 20):
        cv2.line(img, (x, 350), (x + 15, 420), (20, 20, 20), 2)
        cv2.line(img, (x + 15, 350), (x, 420), (20, 20, 20), 2)
    cv2.imwrite(str(file_path), img)
    return str(file_path)


@pytest.fixture
def crack_closeup_image(tmp_path):
    """Creates a sample concrete deck crack close-up photo."""
    file_path = tmp_path / "crack_closeup.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 175
    # Add dark jagged crack line
    pts = np.array([[150, 120], [240, 260], [390, 310], [580, 480]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (15, 15, 15), 6)
    cv2.imwrite(str(file_path), img)
    return str(file_path)


@pytest.fixture
def rust_corrosion_image(tmp_path):
    """Creates a steel girder rust/corrosion sample photo."""
    file_path = tmp_path / "rust_corrosion.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 140
    # Draw rust patch (reddish-brown color in BGR)
    cv2.rectangle(img, (220, 180), (480, 390), (20, 60, 190), -1)
    cv2.imwrite(str(file_path), img)
    return str(file_path)


@pytest.fixture
def spalling_image(tmp_path):
    """Creates a concrete spalling degradation sample photo."""
    file_path = tmp_path / "spalling_concrete.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 170
    # Draw dark irregular spalled region with rough texture
    cv2.ellipse(img, (400, 300), (100, 70), 20, 0, 360, (50, 50, 50), -1)
    noise = np.random.randint(0, 90, (140, 200, 3), dtype=np.uint8)
    img[230:370, 300:500] = cv2.addWeighted(img[230:370, 300:500], 0.4, noise, 0.6, 0)
    cv2.imwrite(str(file_path), img)
    return str(file_path)


@pytest.fixture
def healthy_bridge_image(tmp_path):
    """Creates a clean healthy bridge component image."""
    file_path = tmp_path / "healthy_deck.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 170
    cv2.imwrite(str(file_path), img)
    return str(file_path)


@pytest.fixture
def random_non_bridge_wall_image(tmp_path):
    """Creates a smooth featureless wall image with no bridge structure."""
    file_path = tmp_path / "random_interior_wall.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 210  # Plain flat wall
    cv2.imwrite(str(file_path), img)
    return str(file_path)


# -----------------------------------------------------------------------------
# PRIORITY 2 & 6: Acceptance of Long-Distance Drone Images & Adaptive Scoring
# -----------------------------------------------------------------------------
def test_long_shot_drone_bridge_accepted(long_shot_drone_bridge_image):
    """
    Priority 2 & 6: Asserts that long-distance drone shots (<25% bridge area) are ACCEPTED
    using Adaptive Inspection Quality Scoring.
    """
    checker = OpenCVImageQualityChecker()
    res = checker.check_quality(long_shot_drone_bridge_image)

    assert res["is_valid"] is True, f"Long-distance drone shot must be accepted! Warnings: {res['warnings']}"
    assert res["metrics"]["bridge_presence_class"] in ["Bridge", "Bridge Component"]
    assert res["metrics"]["adaptive_quality_score"] > 40.0


# -----------------------------------------------------------------------------
# PRIORITY 3: Bridge Presence Classifier & Non-Bridge Rejection
# -----------------------------------------------------------------------------
def test_random_wall_rejected_as_non_bridge(random_non_bridge_wall_image):
    """
    Priority 3: Asserts that random non-bridge wall photos are REJECTED as 'Non Bridge'.
    """
    checker = OpenCVImageQualityChecker()
    res = checker.check_quality(random_non_bridge_wall_image)

    assert res["is_valid"] is False, "Random non-bridge wall image must be rejected!"
    assert res["metrics"]["bridge_presence_class"] == "Non Bridge"
    assert any("Non-Bridge" in w for w in res["warnings"])


# -----------------------------------------------------------------------------
# PRIORITY 1 & 8: Defect Detector Recall & Regression Guard
# -----------------------------------------------------------------------------
def test_crack_detection_recall(crack_closeup_image):
    """
    Priority 1 & 8: Asserts that known crack image produces verified crack detections.
    Build fails if 0 defects detected.
    """
    img = cv2.imread(crack_closeup_image)
    detector = DamageDetector(min_confidence=0.30)
    bridge_detector = BridgeDetector()
    bridge_info = bridge_detector.detect_bridge(img)

    res = detector.detect_all_damage(img, bridge_info, visible_components=[{"label": "Deck", "bbox": [0,0,800,600]}])
    verified = res.get("bboxes", [])

    assert len(verified) > 0, "REGRESSION FAIL: Known crack image produced 0 verified defects!"
    assert any(b["label"] == "Crack" for b in verified)


def test_rust_detection_recall(rust_corrosion_image):
    """
    Priority 1 & 8: Asserts that known rust image produces verified rust/corrosion detections.
    Build fails if 0 defects detected.
    """
    img = cv2.imread(rust_corrosion_image)
    detector = DamageDetector(min_confidence=0.30)
    bridge_detector = BridgeDetector()
    bridge_info = bridge_detector.detect_bridge(img)

    res = detector.detect_all_damage(img, bridge_info, visible_components=[{"label": "Steel Girder", "bbox": [0,0,800,600]}])
    verified = res.get("bboxes", [])

    assert len(verified) > 0, "REGRESSION FAIL: Known rust image produced 0 verified defects!"
    assert any(b["label"] == "Rust/Corrosion" for b in verified)


def test_spalling_detection_recall(spalling_image):
    """
    Priority 1 & 8: Asserts that known spalling image produces verified spalling detections.
    Build fails if 0 defects detected.
    """
    img = cv2.imread(spalling_image)
    detector = DamageDetector(min_confidence=0.30)
    bridge_detector = BridgeDetector()
    bridge_info = bridge_detector.detect_bridge(img)

    res = detector.detect_all_damage(img, bridge_info, visible_components=[{"label": "Pier", "bbox": [0,0,800,600]}])
    verified = res.get("bboxes", [])

    assert len(verified) > 0, "REGRESSION FAIL: Known spalling image produced 0 verified defects!"
    assert any(b["label"] == "Spalling" for b in verified)


def test_healthy_bridge_zero_false_positives(healthy_bridge_image):
    """
    Priority 1 & 8: Asserts that a clean healthy bridge image produces zero false defects.
    """
    extractor = ImageFeatureExtractor(pixel_to_mm=0.5)
    features, raw = extractor.extract_features(healthy_bridge_image)

    verified = raw["damage_info"]["bboxes"]
    assert len(verified) == 0, f"REGRESSION FAIL: Clean healthy bridge produced false defect predictions: {verified}"
    assert features["no_defects_detected"] is True
