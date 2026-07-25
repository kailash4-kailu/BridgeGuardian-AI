"""
BridgeGuardian AI — GIS Spatial Defect Heatmap Engine
Calculates spatial defect density matrices using Gaussian Kernel Density Estimation (KDE)
and generates severity-weighted intensity heatmaps for Leaflet/GIS maps.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np


class GISHeatmapEngine:
    """
    Computes spatial defect kernel density estimation (KDE) over normalized 2D structural planes.
    """

    SEVERITY_WEIGHTS: Dict[str, float] = {
        "MINOR": 0.25,
        "MODERATE": 0.55,
        "SEVERE": 0.85,
        "CRITICAL": 1.00,
    }

    def __init__(self, grid_size: int = 50, bandwidth: float = 0.15) -> None:
        self.grid_size = grid_size
        self.bandwidth = bandwidth

    def generate_density_heatmap(
        self, detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates 2D spatial heatmap density grid and severity points.

        Args:
            detections: List of detection dicts containing bbox or polygon coordinates and severity.

        Returns:
            Dict containing grid dimensions, density matrix, and max intensity point.
        """
        if not detections:
            return {
                "grid_size": self.grid_size,
                "max_density": 0.0,
                "density_grid": [[0.0] * self.grid_size for _ in range(self.grid_size)],
                "defect_points": [],
            }

        # Extract normalized centers (x, y) in range [0, 1]
        points = []
        for d in detections:
            bbox = d.get("bbox", [0, 0, 100, 100])  # [x_min, y_min, x_max, y_max]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            # Normalize coordinates assuming reference frame dimensions
            ref_w = max(bbox[2], 1000.0)
            ref_h = max(bbox[3], 1000.0)
            
            nx = max(0.0, min(1.0, cx / ref_w))
            ny = max(0.0, min(1.0, cy / ref_h))
            
            severity = d.get("severity_level", "MINOR").upper()
            weight = self.SEVERITY_WEIGHTS.get(severity, 0.25)
            
            points.append((nx, ny, weight, severity))

        # Evaluate Gaussian KDE matrix
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        x_coords = np.linspace(0.0, 1.0, self.grid_size)
        y_coords = np.linspace(0.0, 1.0, self.grid_size)

        for px, py, w, _ in points:
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    dist_sq = (x - px) ** 2 + (y - py) ** 2
                    kernel_val = w * math.exp(-dist_sq / (2.0 * (self.bandwidth ** 2)))
                    grid[i, j] += kernel_val

        max_val = float(np.max(grid)) if len(points) > 0 else 0.0
        normalized_grid = (grid / max_val).tolist() if max_val > 0 else grid.tolist()

        formatted_points = [
            {"x": round(px, 4), "y": round(py, 4), "weight": w, "severity": s}
            for px, py, w, s in points
        ]

        return {
            "grid_size": self.grid_size,
            "max_density": round(max_val, 4),
            "density_grid": normalized_grid,
            "defect_points": formatted_points,
        }
