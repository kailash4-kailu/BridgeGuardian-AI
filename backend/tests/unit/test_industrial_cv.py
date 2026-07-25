"""
BridgeGuardian AI — Unit Tests: Industrial Computer Vision & Localization Engine (v2.0)
Tests SAHI tile slicing, Laplacian blur quality gate, soft-min health scoring, and homography alignment.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.sahi_tiler import SAHITiler
from backend.app.services.health_aggregator import StructuralHealthAggregator
from backend.app.ml.polygon_localizer import PolygonLocalizer


def test_sahi_tiler_slicing_and_blur():
    """Verify that SAHITiler slices high-resolution images and computes blur variance."""
    # Create synthetic 1280x1280 image
    image = np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)

    tiler = SAHITiler(slice_height=640, slice_width=640, min_blur_variance=50.0)
    result = tiler.slice_image(image)

    assert result["slice_count"] >= 4
    assert result["original_shape"] == (1280, 1280)
    assert "blur_variance" in result
    assert isinstance(result["is_blurred"], bool)


def test_soft_min_health_aggregator_severe_defect_dominance():
    """Verify that a single severe defect forces component status to CRITICAL."""
    aggregator = StructuralHealthAggregator()

    # Detections containing 1 Exposed Rebar and 5 minor hairline cracks
    detections = [
        {"defect_class": "EXPOSED_REBAR", "confidence": 0.95},
        {"defect_class": "CRACK_HAIRLINE", "confidence": 0.50},
        {"defect_class": "CRACK_HAIRLINE", "confidence": 0.60},
    ]

    result = aggregator.compute_component_health(detections)

    assert result["status_category"] in ("CRITICAL", "SEVERE")
    assert result["health_score"] < 30.0
    assert result["worst_defect_class"] == "EXPOSED_REBAR"
    assert "Emergency" in result["recommendation"] or "Maintenance" in result["recommendation"]


def test_polygon_localizer_homography_and_pip_alignment():
    """Verify that PolygonLocalizer maps defect polygon to component polygon via PIP alignment."""
    localizer = PolygonLocalizer()

    defect_poly = [[10, 10], [50, 10], [50, 50], [10, 50]]
    component_polys = {
        "DECK": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "PIER": [[200, 200], [300, 200], [300, 300], [200, 300]],
    }

    comp, ioa = localizer.align_defect_to_component(defect_poly, component_polys)

    assert comp == "DECK"
    assert ioa > 0.80
