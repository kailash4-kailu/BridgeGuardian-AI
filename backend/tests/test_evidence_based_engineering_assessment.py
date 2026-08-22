"""
BridgeGuardian AI — Test Suite: Evidence-Based Engineering Assessment
Verifies 4 component states, coverage-weighted uncertainty penalties,
evidence-bounded executive summary statements, and inspection limitations compilation.
"""
import pytest
import numpy as np
import cv2

from backend.ml.structural.structural_engine import StructuralEngine
from backend.ml.prediction.prediction_engine import PredictionEngine
from backend.ml.explainability.explainability_engine import ExplainabilityEngine
from backend.ml.computer_vision.evidence_graph import InspectionEvidenceGraph
from backend.ml.inference import InferencePipeline


@pytest.fixture
def partial_coverage_images(tmp_path):
    """Creates mock sharp accepted image results representing 40% surface coverage."""
    file_path = tmp_path / "deck_partial.jpg"
    img = np.ones((600, 800, 3), dtype=np.uint8) * 180
    cv2.imwrite(str(file_path), img)

    return [
        {
            "image_path": str(file_path),
            "image_name": "deck_partial.jpg",
            "is_valid": True,
            "warnings": [],
            "metrics": {"blur_score": 140.0, "bridge_coverage_pct": 40.0},
            "features": {
                "visible_components": ["Deck"],
                "defects": []
            }
        }
    ]


def test_four_component_engineering_states(partial_coverage_images):
    """
    Test 1: Component findings assign 'No Visible Defect Observed' when 0 defects detected
    on partial coverage (<90%), and 'Not Inspected' for unobserved ontology elements (Bearing, Substructure).
    Must NEVER label partial coverage as 'Healthy'.
    """
    engine = StructuralEngine()
    result = engine.analyze(partial_coverage_images, unique_defects=[])
    findings = result["statistics"]["component_findings"]

    deck_finding = next(f for f in findings if f["component"] == "Deck")
    assert deck_finding["status"] == "No Visible Defect Observed"
    assert deck_finding["status"] != "Healthy"
    assert deck_finding["status"] != "Verified Healthy"

    uninspected_names = [f["component"] for f in findings if f["status"] == "Not Inspected"]
    assert "Bearing" in uninspected_names or "Substructure" in uninspected_names or "Expansion Joint" in uninspected_names


def test_coverage_weighted_uncertainty_penalty(partial_coverage_images):
    """
    Test 2: At 40% surface coverage with 0 defects detected, SHI MUST NOT be 100%.
    Uncertainty penalty = (1.0 - 0.40) * 15.0 = 9.0% -> SHI should be ~91.0%,
    and failure probability must report uncertified/uncertainty warning.
    """
    struct_engine = StructuralEngine()
    struct_res = struct_engine.analyze(partial_coverage_images, unique_defects=[])
    stats = struct_res["statistics"]
    stats["coverage_score"] = 0.40  # Force 40% coverage

    pipeline = InferencePipeline("models")
    pred_engine = PredictionEngine(pipeline)
    health_predictions = pred_engine.predict(stats)

    assert health_predictions["health_score"] == 91.0
    assert health_predictions["health_score"] < 100.0
    assert health_predictions["risk_category"] == "No Visible Defect Observed"


def test_evidence_bounded_executive_summary(partial_coverage_images):
    """
    Test 3: Executive summary explicitly states approximately 40% was observed,
    unassessed regions exist, and cannot certify entire bridge as defect-free.
    Must NEVER claim 'Bridge is healthy' or 'Optimal condition'.
    """
    struct_engine = StructuralEngine()
    struct_res = struct_engine.analyze(partial_coverage_images, unique_defects=[])
    stats = struct_res["statistics"]
    stats["coverage_score"] = 0.40

    pipeline = InferencePipeline("models")
    pred_engine = PredictionEngine(pipeline)
    health_predictions = pred_engine.predict(stats)

    explain_engine = ExplainabilityEngine()
    explanation = explain_engine.generate_explanation(health_predictions, stats)

    summary = explanation["summary_report"]
    assert "Approximately 40% of the bridge structure was observed" in summary
    assert "Therefore this inspection cannot certify the entire bridge as defect-free" in summary
    assert "All observed structural components remain in optimal condition" not in summary


def test_inspection_limitations_extraction(partial_coverage_images):
    """
    Test 4: Inspection Evidence Graph extracts engineering confidence and limitations packet.
    """
    struct_engine = StructuralEngine()
    struct_res = struct_engine.analyze(partial_coverage_images, unique_defects=[])
    stats = struct_res["statistics"]
    stats["coverage_score"] = 0.40

    pipeline = InferencePipeline("models")
    pred_engine = PredictionEngine(pipeline)
    health_predictions = pred_engine.predict(stats)

    graph = InspectionEvidenceGraph()
    graph.build(
        accepted_images=partial_coverage_images,
        visible_components=stats.get("visible_components_inventory", []),
        verified_defects=[],
        measurements={},
        health_predictions=health_predictions,
        coverage_score=0.40,
        uninspected_components=["Bearing", "Substructure"],
        rejected_images=[]
    )

    limitations = graph.inspection_limitations
    assert limitations["estimated_surface_coverage_pct"] == 40.0
    assert "Bearing" in limitations["uninspected_components"]
    assert "uncertified_disclaimer" in limitations
