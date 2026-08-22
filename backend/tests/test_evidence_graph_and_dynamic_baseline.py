"""
BridgeGuardian AI — Test Suite: Evidence Graph & Dynamic Baseline Validation
Verifies no assumed components, dynamic 100% SHI for clean inspections,
evidence graph provenance lineage, and non-fabricated executive summary.
"""
import json
import pathlib
import tempfile
import numpy as np
import cv2
import pytest

from backend.ml.structural.structural_engine import StructuralEngine
from backend.ml.prediction.prediction_engine import PredictionEngine
from backend.ml.explainability.explainability_engine import ExplainabilityEngine
from backend.ml.computer_vision.evidence_graph import InspectionEvidenceGraph
from backend.ml.inference import InferencePipeline


@pytest.fixture
def clean_accepted_images(tmp_path):
    """Creates mock sharp accepted image results with detected Suspension Cable & Tower ONLY."""
    file_path = tmp_path / "clean_suspension.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 160
    cv2.imwrite(str(file_path), img)

    return [
        {
            "image_path": str(file_path),
            "image_name": "clean_suspension.jpg",
            "is_valid": True,
            "warnings": [],
            "metrics": {"blur_score": 120.0, "bridge_coverage_pct": 72.5},
            "features": {
                "visible_components": ["Suspension Cable", "Tower"],
                "defects": []
            }
        }
    ]


def test_no_assumed_components_in_inventory(clean_accepted_images):
    """
    Test 1: Predefined bridge templates are eliminated.
    Detected components (Suspension Cable, Tower) appear as 'No Visible Defect Observed',
    and unobserved ontology components appear as 'Not Inspected'.
    """
    engine = StructuralEngine()
    result = engine.analyze(clean_accepted_images, unique_defects=[])
    stats = result["statistics"]

    detected_findings = [f for f in stats["component_findings"] if f["status"] != "Not Inspected"]
    component_names = [f["component"] for f in detected_findings]
    
    assert "Suspension Cable" in component_names
    assert "Tower" in component_names

    for finding in detected_findings:
        assert finding["status"] in ("No Visible Defect Observed", "Verified Healthy")
        assert finding["cracks"] == "No"
        assert finding["rust"] == "No"


def test_dynamic_clean_inspection_baseline(clean_accepted_images):
    """
    Test 2: Clean inspection with 0 defects returns >90% SHI for high coverage,
    and 3650.0 RUL days baseline, NOT hardcoded 85.75 SHI or 2.42% failure prob.
    """
    struct_engine = StructuralEngine()
    struct_res = struct_engine.analyze(clean_accepted_images, unique_defects=[])
    stats = struct_res["statistics"]
    stats["coverage_score"] = 0.95  # 95% coverage for full verification

    pipeline = InferencePipeline("models")
    pred_engine = PredictionEngine(pipeline)
    health_predictions = pred_engine.predict(stats)

    assert health_predictions["health_score"] == 100.0
    assert health_predictions["failure_probability"] <= 0.05
    assert health_predictions["rul_days"] >= 3650.0
    assert health_predictions["risk_category"] == "Verified Healthy"


def test_evidence_graph_and_provenance_lineage(clean_accepted_images):
    """
    Test 3: Inspection Evidence Graph compiles lineage tracking for all metrics.
    """
    struct_engine = StructuralEngine()
    struct_res = struct_engine.analyze(clean_accepted_images, unique_defects=[])
    stats = struct_res["statistics"]
    stats["coverage_score"] = 0.95

    pipeline = InferencePipeline("models")
    pred_engine = PredictionEngine(pipeline)
    health_predictions = pred_engine.predict(stats)

    graph = InspectionEvidenceGraph()
    graph.build(
        accepted_images=clean_accepted_images,
        visible_components=stats.get("visible_components_inventory", []),
        verified_defects=[],
        measurements={"largest_crack_width": 0.0},
        health_predictions=health_predictions,
        coverage_score=0.95
    )

    prov = graph.provenance
    assert "shi_provenance" in prov
    assert "failure_probability_provenance" in prov
    assert "rul_provenance" in prov
    assert "coverage_provenance" in prov


def test_executive_summary_non_fabrication(clean_accepted_images):
    """
    Test 4: Executive summary explicitly states no visible structural deterioration was identified
    when 0 defects exist, and never fabricates 'minor visual degradation'.
    """
    struct_engine = StructuralEngine()
    struct_res = struct_engine.analyze(clean_accepted_images, unique_defects=[])
    stats = struct_res["statistics"]
    stats["coverage_score"] = 0.95

    pipeline = InferencePipeline("models")
    pred_engine = PredictionEngine(pipeline)
    health_predictions = pred_engine.predict(stats)

    explain_engine = ExplainabilityEngine()
    explanation = explain_engine.generate_explanation(health_predictions, stats)

    summary = explanation["summary_report"]
    assert "No visible structural deterioration was identified" in summary
    assert "minor visual degradation" not in summary
