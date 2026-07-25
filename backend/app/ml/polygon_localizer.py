"""
BridgeGuardian AI — Homography Polygon Structural Localization Engine
Transforms 2D pixel detection polygons into 3D structural template coordinates (H 3x3)
and executes Point-in-Polygon (PIP) ray-casting alignment against bridge components.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class PolygonLocalizer:
    """
    Maps 2D pixel defect segmentation polygons to structural component template coordinates
    using a 3x3 Planar Homography matrix H.
    """

    def __init__(self, homography_matrix: Optional[np.ndarray] = None) -> None:
        if homography_matrix is not None:
            self.H = homography_matrix
        else:
            # Default identity transformation matrix
            self.H = np.eye(3, dtype=np.float64)

    def set_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> None:
        """
        Computes 3x3 Homography matrix H from 4 or more source-destination point pairs.
        """
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            self.H = H

    def transform_polygon(self, polygon_pts: List[List[float]]) -> List[List[float]]:
        """
        Transforms pixel coordinate polygon [[x1,y1], [x2,y2], ...] via H 3x3 matrix.
        """
        if not polygon_pts:
            return []

        pts = np.array(polygon_pts, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2).tolist()

    def align_defect_to_component(
        self,
        defect_polygon: List[List[float]],
        component_polygons: Dict[str, List[List[float]]],
    ) -> Tuple[Optional[str], float]:
        """
        Calculates Intersection over Area (IoA) between transformed defect polygon
        and structural component reference polygons.

        Returns:
            Tuple of (assigned_component_code, intersection_over_area)
        """
        if not defect_polygon or not component_polygons:
            return None, 0.0

        transformed_poly = self.transform_polygon(defect_polygon)
        defect_cnt = np.array(transformed_poly, dtype=np.int32)
        defect_area = cv2.contourArea(defect_cnt)

        if defect_area <= 0:
            return None, 0.0

        best_component = None
        max_ioa = 0.0

        for comp_code, comp_pts in component_polygons.items():
            comp_cnt = np.array(comp_pts, dtype=np.int32)
            
            # Compute spatial intersection using OpenCV mask operations
            x_max = max(int(np.max(defect_cnt[:, 0])), int(np.max(comp_cnt[:, 0]))) + 10
            y_max = max(int(np.max(defect_cnt[:, 1])), int(np.max(comp_cnt[:, 1]))) + 10
            
            mask_def = np.zeros((y_max, x_max), dtype=np.uint8)
            mask_comp = np.zeros((y_max, x_max), dtype=np.uint8)

            cv2.drawContours(mask_def, [defect_cnt], -1, 255, -1)
            cv2.drawContours(mask_comp, [comp_cnt], -1, 255, -1)

            intersection = cv2.bitwise_and(mask_def, mask_comp)
            inter_area = float(cv2.countNonZero(intersection))

            ioa = inter_area / defect_area
            if ioa > max_ioa:
                max_ioa = ioa
                best_component = comp_code

        # Assign component if IoA >= 0.30 threshold
        if max_ioa >= 0.30:
            return best_component, round(max_ioa, 4)
        
        return None, round(max_ioa, 4)
