"""
BridgeGuardian AI — Standalone Isolated Defect Model Runner CLI
Runs defect model inference directly on single images outside the web application pipeline.
Prints model file path, class count, class names, raw candidate predictions, and verified defects.
"""
import sys
import logging
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.ml.computer_vision.detector import YOLODetector
from backend.ml.computer_vision.damage_detector import DamageDetector
from backend.ml.computer_vision.bridge_detector import BridgeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_isolated_defect_model")


def run_isolated_inference(image_path: str, min_confidence: float = 0.30, debug_mode: bool = False) -> None:
    logger.info("================ STANDALONE DEFECT MODEL INFERENCE ================")
    logger.info(f"Target Image: '{image_path}'")
    logger.info(f"Confidence Threshold: {min_confidence}")
    logger.info(f"Debug Bypass Filters: {debug_mode}")

    if not Path(image_path).exists():
        logger.error(f"Target image path does not exist: '{image_path}'")
        return

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image matrix: '{image_path}'")
        return

    h, w = image.shape[:2]
    logger.info(f"Image Matrix: {w}x{h} px")

    # 1. Startup Model Verification Audit
    comp_detector = YOLODetector(confidence_threshold=min_confidence)
    comp_status = comp_detector.log_model_status()
    logger.info(f"[Startup Audit] Component Model Metadata:\n{comp_status}")

    damage_detector = DamageDetector(min_confidence=min_confidence, debug_bypass_filters=debug_mode)
    damage_status = damage_detector.log_model_status()
    logger.info(f"[Startup Audit] Defect Model Metadata:\n{damage_status}")

    # 2. Run Component & Damage Detector
    bridge_detector = BridgeDetector()
    bridge_info = bridge_detector.detect_bridge(image)
    comp_detections = comp_detector.detect(image, image_path)

    visible_components = [
        {"label": det.label, "bbox": det.bbox, "confidence": det.confidence, "mask": None}
        for det in comp_detections
    ]
    logger.info(f"[Stage 1] Visible Components ({len(visible_components)}): {[c['label'] for c in visible_components]}")

    damage_results = damage_detector.detect_all_damage(image, bridge_info, visible_components)

    raw_candidates = damage_results.get("raw_candidates", [])
    verified_defects = damage_results.get("bboxes", [])

    logger.info("---------------- Raw Model Candidate Output Dumps ----------------")
    logger.info(f"Total Raw Candidates: {len(raw_candidates)}")
    for i, cand in enumerate(raw_candidates, 1):
        bx, by, bw, bh = cand["bbox"]
        area = bw * bh
        logger.info(
            f"  Candidate #{i}: Class='{cand['label']}' | "
            f"Confidence={cand['confidence']:.2f} | BBox={cand['bbox']} | Area={area} px"
        )

    logger.info("---------------- Verified Defects (Post-Filtering) ----------------")
    logger.info(f"Total Verified Defects: {len(verified_defects)}")
    for i, defect in enumerate(verified_defects, 1):
        bx, by, bw, bh = defect["bbox"]
        area = bw * bh
        logger.info(
            f"  Verified #{i}: Class='{defect['label']}' | "
            f"Confidence={defect['confidence']:.2f} | BBox={defect['bbox']} | Area={area} px"
        )

    # 3. Save Visual Overlay Output
    out_dir = Path("backend/static/debug/isolated_standalone")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"standalone_{Path(image_path).stem}.png"

    vis_img = image.copy()
    for d in verified_defects:
        bx, by, bw, bh = d["bbox"]
        cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(vis_img, f"{d['label']} {d['confidence']:.2f}", (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(str(out_file), vis_img)
    logger.info(f"Visual overlay output saved to: '{out_file.resolve()}'")
    logger.info("==================================================================")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "test_crack_sample.jpg"
    if not Path(target).exists():
        h, w = 600, 800
        mock_img = np.ones((h, w, 3), dtype=np.uint8) * 180
        pts = np.array([[100, 120], [220, 240], [380, 290], [550, 460]], dtype=np.int32)
        cv2.polylines(mock_img, [pts], False, (10, 10, 10), 6)
        cv2.imwrite(target, mock_img)

    run_isolated_inference(target)
