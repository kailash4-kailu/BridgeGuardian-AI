"""
BridgeGuardian AI — Industrial SAHI Slicing & Quality Gate Module
Implements Slicing Aided Hyper Inference (SAHI) over 4K/20MP aerial drone imagery
with Laplacian variance blur rejection and CLAHE contrast enhancement.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple


class SAHITiler:
    """
    Slices high-resolution drone images into overlapping tiles to preserve
    sub-millimeter micro-crack resolution for deep learning inference.
    """

    def __init__(
        self,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        min_blur_variance: float = 100.0,
    ) -> None:
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.min_blur_variance = min_blur_variance

    def compute_laplacian_blur_variance(self, image: np.ndarray) -> float:
        """
        Calculates Laplacian variance to measure spatial edge sharpness.
        Variance < min_blur_variance indicates motion blur or out-of-focus capture.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def apply_clahe_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to illuminate shadowed bridge under-decks and pier regions.
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def slice_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Slices image into overlapping patches.

        Returns:
            Dict containing slices (list of image patches), slice bounding boxes,
            image metadata, and blur quality evaluation.
        """
        h_img, w_img = image.shape[:2]
        blur_var = self.compute_laplacian_blur_variance(image)
        is_blurred = bool(blur_var < self.min_blur_variance)

        step_h = int(self.slice_height * (1.0 - self.overlap_height_ratio))
        step_w = int(self.slice_width * (1.0 - self.overlap_width_ratio))

        slices = []
        slice_bboxes = []

        for y in range(0, max(1, h_img - self.slice_height + step_h), step_h):
            for x in range(0, max(1, w_img - self.slice_width + step_w), step_w):
                y_end = min(y + self.slice_height, h_img)
                x_end = min(x + self.slice_width, w_img)
                y_start = max(0, y_end - self.slice_height)
                x_start = max(0, x_end - self.slice_width)

                slice_patch = image[y_start:y_end, x_start:x_end]
                slices.append(slice_patch)
                slice_bboxes.append((x_start, y_start, x_end, y_end))

        return {
            "slices": slices,
            "slice_bboxes": slice_bboxes,
            "original_shape": (h_img, w_img),
            "blur_variance": round(blur_var, 2),
            "is_blurred": is_blurred,
            "slice_count": len(slices),
        }

    def map_slice_predictions_to_global(
        self,
        slice_predictions: List[List[Dict[str, Any]]],
        slice_bboxes: List[Tuple[int, int, int, int]],
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Maps slice-level detection bounding boxes and polygon masks back to original image space
        and performs Non-Maximum Suppression (NMS) to merge overlapping predictions.
        """
        global_detections = []

        for preds, (x_off, y_off, _, _) in zip(slice_predictions, slice_bboxes):
            for p in preds:
                box = p["bbox"]  # [x_min, y_min, x_max, y_max]
                global_box = [
                    box[0] + x_off,
                    box[1] + y_off,
                    box[2] + x_off,
                    box[3] + y_off,
                ]

                global_pred = {
                    "bbox": global_box,
                    "confidence": p["confidence"],
                    "defect_class": p["defect_class"],
                }

                if "segmentation" in p and p["segmentation"]:
                    poly = p["segmentation"]
                    global_poly = [[pt[0] + x_off, pt[1] + y_off] for pt in poly]
                    global_pred["segmentation"] = global_poly

                global_detections.append(global_pred)

        # Apply basic greedy NMS
        if not global_detections:
            return []

        global_detections.sort(key=lambda d: d["confidence"], reverse=True)
        keep = []

        while global_detections:
            current = global_detections.pop(0)
            keep.append(current)
            global_detections = [
                d for d in global_detections if self._compute_iou(current["bbox"], d["bbox"]) < iou_threshold
            ]

        return keep

    @staticmethod
    def _compute_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0.0
