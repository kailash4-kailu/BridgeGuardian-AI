"""
BridgeGuardian AI — Isolated Single Image Defect Detector Test CLI
Runs defect detection directly on an isolated image file outside the full web application pipeline.
Prints model loading status, supported class names, raw candidate predictions, and verified defects.
"""
import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.ml.computer_vision.detector import YOLODetector
from backend.ml.computer_vision.damage_detector import DamageDetector
from backend.ml.computer_vision.bridge_detector import BridgeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_single_image_defect_detector")


def run_isolated_test(image_path: str, min_confidence: float = 0.30) -> None:
    logger.info(f"========== Isolated Defect Detector Test ==========")
    logger.info(f"Target Image: {image_path}")
    logger.info(f"Confidence Threshold: {min_confidence}")

    if not Path(image_path).exists():
        logger.error(f"Image path does not exist: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image matrix: {image_path}")
        return

    h, w = image.shape[:2]
    logger.info(f"Image Dimensions: {w}x{h} px")

    # 1. Audit Model Startup
    comp_detector = YOLODetector(confidence_threshold=min_confidence)
    comp_status = comp_detector.log_model_status()
    logger.info(f"[Model Audit] Component Model Status: {comp_status}")

    damage_detector = DamageDetector(min_confidence=min_confidence)
    damage_status = damage_detector.log_model_status()
    logger.info(f"[Model Audit] Defect Model Status: {damage_status}")

    # 2. Step 1: Detect Bridge & Components
    bridge_detector = BridgeDetector()
    bridge_info = bridge_detector.detect_bridge(image)
    comp_detections = comp_detector.detect(image, image_path)

    visible_components = [
        {"label": det.label, "bbox": det.bbox, "confidence": det.confidence, "mask": None}
        for det in comp_detections
    ]
    logger.info(f"[Stage 1] Detected Components ({len(visible_components)}): {[c['label'] for c in visible_components]}")

    # 3. Step 2: Run Damage Detector
    damage_results = damage_detector.detect_all_damage(image, bridge_info, visible_components)

    raw_candidates = damage_results.get("raw_candidates", [])
    verified_defects = damage_results.get("bboxes", [])

    logger.info(f"========== Pipeline Audit Results ==========")
    logger.info(f"Raw Candidates Count: {len(raw_candidates)}")
    for i, cand in enumerate(raw_candidates, 1):
        logger.info(f"  Raw #{i}: {cand['label']} | Conf: {cand['confidence']:.2f} | BBox: {cand['bbox']}")

    logger.info(f"Verified Defects Count (Conf >= {min_confidence}): {len(verified_defects)}")
    for i, defect in enumerate(verified_defects, 1):
        logger.info(f"  Verified #{i}: {defect['label']} | Conf: {defect['confidence']:.2f} | BBox: {defect['bbox']}")

    # 4. Save visualization
    out_dir = Path("backend/static/debug/isolated_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"test_{Path(image_path).stem}.png"

    vis_img = image.copy()
    for d in verified_defects:
        bx, by, bw_c, bh_c = d["bbox"]
        cv2.rectangle(vis_img, (bx, by), (bx + bw_c, by + bh_c), (0, 0, 255), 2)
        cv2.putText(vis_img, f"{d['label']} {d['confidence']:.2f}", (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(str(out_file), vis_img)
    logger.info(f"Visualization saved to: {out_file}")
    logger.info(f"==================================================")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "test_crack_image.jpg"
        # Create a mock synthetic crack image for CLI execution if none specified
        h, w = 600, 800
        mock_img = np.ones((h, w, 3), dtype=np.uint8) * 180
        # Draw explicit dark jagged line representing a crack
        cv2.polylines(mock_img, [np.array([[100, 100], [200, 250], [350, 280], [500, 420]], dtype=np.int32)], False, (20, 20, 20), 4)
        cv2.imwrite(target, mock_img)

    run_isolated_test(target)
