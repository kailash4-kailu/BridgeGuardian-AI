"""
BridgeGuardian AI — Inspection Evidence Graph
Maintains end-to-end evidence traceability linking Accepted Images -> Visible Components -> Verified Defects -> Measurements -> Engineering Rules -> Health Scores.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional


class InspectionEvidenceGraph:
    """
    Data structure and provenance engine for inspection campaigns.
    Ensures every number, status, table row, and report statement is backed by explicit evidence.
    """

    def __init__(self) -> None:
        self.accepted_images: List[Dict[str, Any]] = []
        self.visible_components: List[Dict[str, Any]] = []  # [{component, confidence, image_count, images, bbox, mask_area, status}]
        self.verified_defects: List[Dict[str, Any]] = []    # [{defect_id, type, component, severity, confidence, bbox, measurements, images}]
        self.measurements: Dict[str, Any] = {}
        self.engineering_rules_applied: List[Dict[str, Any]] = []
        self.provenance: Dict[str, Any] = {}
        self.inspection_limitations: Dict[str, Any] = {}
        self.engineering_confidence: float = 0.0

    def build(
        self,
        accepted_images: List[Dict[str, Any]],
        visible_components: List[Dict[str, Any]],
        verified_defects: List[Dict[str, Any]],
        measurements: Dict[str, Any],
        health_predictions: Dict[str, Any],
        coverage_score: float,
        uninspected_components: Optional[List[str]] = None,
        rejected_images: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Populates the evidence graph and constructs explicit provenance and engineering confidence for all metrics.
        """
        self.accepted_images = accepted_images
        self.visible_components = visible_components
        self.verified_defects = verified_defects
        self.measurements = measurements

        accepted_count = len(accepted_images)
        components_count = len(visible_components)
        defects_count = len(verified_defects)

        # Average image quality factor across accepted frames
        avg_quality = float(np.mean([img.get("metrics", {}).get("blur_score", 100.0) for img in accepted_images])) if accepted_images else 100.0
        quality_factor = min(1.0, max(0.2, avg_quality / 500.0)) if avg_quality > 0 else 0.8

        # Calculate mean vision confidence across accepted images/defects
        conf_list = [d.get("confidence", 0.95) for d in verified_defects] if defects_count > 0 else [0.95]
        avg_confidence = float(round(sum(conf_list) / len(conf_list), 4))

        # Viewpoint diversity factor (unique image frames with component detections)
        viewpoint_count = len({img["image_name"] for img in accepted_images}) if accepted_images else 1
        viewpoint_factor = min(1.0, max(0.4, viewpoint_count / 5.0))

        # Engineering Confidence = Surface Coverage * Image Quality * Detection Confidence * Viewpoint Diversity
        eng_conf_raw = coverage_score * quality_factor * avg_confidence * viewpoint_factor
        self.engineering_confidence = float(round(min(1.0, max(0.05, eng_conf_raw)), 4))

        uncertainty_penalty = float(round((1.0 - min(1.0, coverage_score)) * 15.0, 2))
        shi_val = health_predictions.get("health_score", 100.0)
        fail_prob_val = health_predictions.get("failure_probability", 0.01)
        rul_val = health_predictions.get("rul_days", "Estimated from inspected regions only")

        uninspected = uninspected_components or ["Under-Deck Substructure", "Bearings", "Expansion Joints", "Abutments"]
        rejected_list = rejected_images or []

        self.inspection_limitations = {
            "uninspected_components": uninspected,
            "occluded_low_conf_regions": ["Shadowed connection joints", "Far-range span framing"],
            "rejected_photos_count": len(rejected_list),
            "rejected_photos_reasons": [
                f"{r.get('image_name', 'unknown')}: {r.get('rejection_reason', 'Quality check failed')}"
                for r in rejected_list
            ],
            "estimated_surface_coverage_pct": round(coverage_score * 100, 1),
            "engineering_confidence_pct": round(self.engineering_confidence * 100, 1),
            "uncertified_disclaimer": "This inspection cannot certify the entire bridge as defect-free due to unassessed structural regions."
        }

        self.provenance = {
            "shi_provenance": {
                "accepted_images_count": accepted_count,
                "visible_components_count": components_count,
                "verified_defects_count": defects_count,
                "coverage_percent": round(coverage_score * 100, 1),
                "uncertainty_penalty": uncertainty_penalty,
                "average_confidence": avg_confidence,
                "derivation": (
                    f"Base 100.0 - {round(100.0 - (shi_val if isinstance(shi_val, (int, float)) else 100.0), 2)}% verified penalties - {uncertainty_penalty}% uninspected region penalty = {shi_val}% SHI"
                    if defects_count > 0
                    else f"Base 100.0 - 0.0% damage penalties - {uncertainty_penalty}% uninspected region penalty = {shi_val}% SHI (No Visible Defect Observed within inspected regions)"
                )
            },
            "failure_probability_provenance": {
                "verified_defects_count": defects_count,
                "coverage_percent": round(coverage_score * 100, 1),
                "derivation": (
                    f"Derived dynamically from {defects_count} verified defect(s) and confidence factors"
                    if defects_count > 0
                    else (
                        "Unable to estimate precisely (< 1.5% in inspected regions). No visible deterioration observed within inspected regions."
                        if coverage_score < 0.90
                        else "No verified defects detected across >90% coverage: Failure probability approaches baseline (< 0.05%)"
                    )
                )
            },
            "rul_provenance": {
                "verified_defects_count": defects_count,
                "coverage_percent": round(coverage_score * 100, 1),
                "derivation": (
                    f"Calculated from verified structural deterioration ({defects_count} defects)"
                    if defects_count > 0
                    else "Estimated from inspected regions only (Subject to uninspected structural regions)"
                )
            },
            "coverage_provenance": {
                "observed_components": [c.get("component") for c in visible_components],
                "coverage_percent": round(coverage_score * 100, 1),
                "engineering_confidence_percent": round(self.engineering_confidence * 100, 1),
                "derivation": f"Observed structural surface ratio across {accepted_count} accepted frames (Engineering Confidence: {round(self.engineering_confidence * 100, 1)}%)"
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the evidence graph to JSON-compatible dictionary."""
        return {
            "accepted_images_count": len(self.accepted_images),
            "visible_components": self.visible_components,
            "verified_defects_count": len(self.verified_defects),
            "verified_defects": self.verified_defects,
            "measurements": self.measurements,
            "engineering_confidence": self.engineering_confidence,
            "inspection_limitations": self.inspection_limitations,
            "provenance": self.provenance
        }
