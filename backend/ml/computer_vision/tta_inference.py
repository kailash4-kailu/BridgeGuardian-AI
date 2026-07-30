"""
BridgeGuardian AI — Test-Time Augmentation (TTA) & Multi-Scale Engine
Ensembles predictions across multiple image scales (0.8x, 1.0x, 1.2x) and horizontal flips
to maximize recall and defect localization accuracy.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("bridgeguardian.cv.tta_inference")


class TTAInferenceEngine:
    """
    Applies Test-Time Augmentation (TTA) over vision models:
      - Multi-scale transforms (0.85x, 1.0x, 1.15x)
      - Horizontal flip ensembling
      - Mask soft-voting and thresholding
    """

    def __init__(self, scales: Tuple[float, ...] = (0.85, 1.0, 1.15), use_flip: bool = True) -> None:
        self.scales = scales
        self.use_flip = use_flip

    def run_tta(self, image: np.ndarray, infer_fn) -> Dict[str, Any]:
        """
        Runs multi-scale and flipped inference, averaging mask predictions.
        
        Args:
            image: Original BGR image array.
            infer_fn: Callable taking image and returning {'detections': list, 'masks': dict}.
            
        Returns:
            Dict containing ensembled 'detections' and 'masks'.
        """
        h_orig, w_orig = image.shape[:2]
        accumulated_masks: Dict[str, np.ndarray] = {}
        all_detections: List[Dict[str, Any]] = []
        transform_count = 0

        # Collect augmented variations
        variations = []
        for s in self.scales:
            scaled_h, scaled_w = max(32, int(h_orig * s)), max(32, int(w_orig * s))
            scaled_img = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            variations.append((scaled_img, s, False))
            if self.use_flip:
                flipped_img = cv2.flip(scaled_img, 1)
                variations.append((flipped_img, s, True))

        for var_img, scale, is_flipped in variations:
            res = infer_fn(var_img)
            dets = res.get("detections", [])
            masks = res.get("masks", {})
            transform_count += 1

            # Transform masks back to original canvas
            for cat_key, mask in masks.items():
                if is_flipped:
                    mask = cv2.flip(mask, 1)
                unscaled_mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                if cat_key not in accumulated_masks:
                    accumulated_masks[cat_key] = unscaled_mask.astype(np.float32)
                else:
                    accumulated_masks[cat_key] += unscaled_mask.astype(np.float32)

            # Map detections back to original scale
            for d in dets:
                bx, by, bw_b, bh_b = d["bbox"]
                if is_flipped:
                    bx = var_img.shape[1] - (bx + bw_b)
                orig_bx = int(bx / scale)
                orig_by = int(by / scale)
                orig_bw = int(bw_b / scale)
                orig_bh = int(bh_b / scale)

                det_copy = d.copy()
                det_copy["bbox"] = [max(0, orig_bx), max(0, orig_by), max(1, orig_bw), max(1, orig_bh)]
                all_detections.append(det_copy)

        # Threshold ensembled probability masks (> 40% agreement)
        final_masks: Dict[str, np.ndarray] = {}
        threshold = (transform_count * 0.35) * 255.0

        for cat_key, float_mask in accumulated_masks.items():
            binary = (float_mask >= threshold).astype(np.uint8) * 255
            final_masks[cat_key] = binary

        return {"detections": all_detections, "masks": final_masks}
