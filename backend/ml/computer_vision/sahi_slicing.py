"""
BridgeGuardian AI — SAHI (Slicing Aided Hyper Inference) Engine
Slices ultra-high-resolution drone imagery into overlapping patches, runs fine-grained tile inference
to detect tiny hairline cracks and distant defects, and stitches results back together with NMS.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("bridgeguardian.cv.sahi_slicing")


class SAHISlicer:
    """
    SAHI engine for ultra-high-resolution drone image slicing and stitched patch inference.
    Prevents downscaling artifact loss for hairline cracks and small defects.
    """

    def __init__(
        self,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        iou_threshold: float = 0.4,
    ) -> None:
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.iou_threshold = iou_threshold

    def slice_and_infer(self, image: np.ndarray, infer_fn) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
        """
        Slices image, calls infer_fn(tile_bgr) on each tile, and merges predictions.
        
        Args:
            image: Full-resolution BGR image.
            infer_fn: Callable taking tile image and returning dict with 'detections' and 'masks'.
            
        Returns:
            Tuple of (merged_detections_list, stitched_masks_dict)
        """
        h, w = image.shape[:2]
        
        # If image is smaller than tile dimensions, run single inference directly
        if h <= self.slice_height and w <= self.slice_width:
            res = infer_fn(image)
            return res.get("detections", []), res.get("masks", {})

        step_y = max(64, int(self.slice_height * (1.0 - self.overlap_height_ratio)))
        step_x = max(64, int(self.slice_width * (1.0 - self.overlap_width_ratio)))

        all_detections = []
        stitched_masks: Dict[str, np.ndarray] = {}

        y_coords = list(range(0, h - self.slice_height + 1, step_y))
        if y_coords[-1] + self.slice_height < h:
            y_coords.append(h - self.slice_height)

        x_coords = list(range(0, w - self.slice_width + 1, step_x))
        if x_coords[-1] + self.slice_width < w:
            x_coords.append(w - self.slice_width)

        for y in y_coords:
            for x in x_coords:
                tile = image[y : y + self.slice_height, x : x + self.slice_width]
                res = infer_fn(tile)
                tile_dets = res.get("detections", [])
                tile_masks = res.get("masks", {})

                # Translate tile bounding boxes to global coordinates
                for det in tile_dets:
                    bx, by, bw_b, bh_b = det["bbox"]
                    det_copy = det.copy()
                    det_copy["bbox"] = [x + bx, y + by, bw_b, bh_b]
                    all_detections.append(det_copy)

                # Combine tile masks to full-size canvas
                for cat_key, tile_mask in tile_masks.items():
                    if cat_key not in stitched_masks:
                        stitched_masks[cat_key] = np.zeros((h, w), dtype=np.uint8)
                    stitched_masks[cat_key][y : y + self.slice_height, x : x + self.slice_width] = cv2.bitwise_or(
                        stitched_masks[cat_key][y : y + self.slice_height, x : x + self.slice_width], tile_mask
                    )

        # Apply Non-Maximum Suppression (NMS) to global bounding boxes
        merged_detections = self._nms_bboxes(all_detections)
        return merged_detections, stitched_masks

    def _nms_bboxes(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Applies Non-Maximum Suppression over overlapping slice bounding boxes."""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections]) # [x, y, w, h]
        scores = np.array([d["confidence"] for d in detections])
        labels = [d["label"] for d in detections]

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # Same label filter
            same_label = np.array([labels[idx] == labels[i] for idx in order[1:]])
            suppress = np.where((iou > self.iou_threshold) & same_label)[0]

            order = np.delete(order, np.concatenate(([0], suppress + 1)))

        return [detections[k] for k in keep]
