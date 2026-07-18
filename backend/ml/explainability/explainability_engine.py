"""
BridgeGuardian AI — Explainability Engine
Generates multi-level explainability (Vision, Feature, ML/SHAP) and outputs the Natural Language AI Summary.
"""
from __future__ import annotations
from typing import Any, Dict

class ExplainabilityEngine:
    def generate_explanation(
        self,
        health_predictions: Dict[str, Any],
        aggregate_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates structured explanation elements at the Vision, Feature, and ML prediction levels.
        Creates the Natural Language AI Inspection Summary.
        """
        health_score = health_predictions.get("health_score", 100.0)
        risk_category = health_predictions.get("risk_category", "Excellent")
        penalties = health_predictions.get("penalties", {})
        
        largest_width = aggregate_stats.get("largest_crack_width", 0.0)
        largest_len = aggregate_stats.get("largest_crack_length", 0.0)
        corrosion_pct = aggregate_stats.get("corrosion_coverage_percent", 0.0)
        rust_pct = aggregate_stats.get("rust_coverage_percent", 0.0)
        total_corrosion = round(corrosion_pct + rust_pct, 2)
        
        critical_count = aggregate_stats.get("critical_defect_count", 0)
        affected_comps = aggregate_stats.get("affected_structural_components", [])
        most_damaged = aggregate_stats.get("most_damaged_structural_component", "None")
        
        # Extract features list
        features_explanation = [
            f"Largest Crack Width: {largest_width} mm",
            f"Largest Crack Length: {largest_len} mm",
            f"Rust & Corrosion Area: {total_corrosion}%",
            f"Critical Defects Count: {critical_count}",
            f"Affected Structural Components: {', '.join(affected_comps) if affected_comps else 'None'}"
        ]
        
        # 1. Vision-level explanation summary
        vision_explanation = f"Detected {critical_count} critical/severe defects localized primarily on {most_damaged}."
        
        # 2. ML-level explanation (penalties impact)
        ml_contributions = []
        if penalties.get("total_penalty", 1.0) < 1.0:
            if penalties.get("crack_penalty", 1.0) < 1.0:
                ml_contributions.append(f"Crack propagation penalty reduced health by {int((1.0 - penalties['crack_penalty']) * 100)}%")
            if penalties.get("corrosion_penalty", 1.0) < 1.0:
                ml_contributions.append(f"Surface corrosion penalty reduced health by {int((1.0 - penalties['corrosion_penalty']) * 100)}%")
            if penalties.get("critical_penalty", 1.0) < 1.0:
                ml_contributions.append(f"High-severity critical defect clusters reduced health by {int((1.0 - penalties['critical_penalty']) * 100)}%")
            if penalties.get("severity_penalty", 1.0) < 1.0:
                ml_contributions.append(f"Extreme localized severity modifier reduced health by {int((1.0 - penalties['severity_penalty']) * 100)}%")
        else:
            ml_contributions.append("No structural visual defects detected. Bridge remains at baseline sensor health.")

        # 3. Assemble Natural Language AI Report
        damage_desc = "minor visual degradation"
        if critical_count >= 3 or risk_category == "Critical":
            damage_desc = "critical structural failure risks"
        elif critical_count >= 1 or risk_category == "Poor":
            damage_desc = "severe visual and structural deterioration"
        elif risk_category == "Fair":
            damage_desc = "moderate structural wear"
            
        lines = [
            f"Drone inspection campaign complete. Bridge exhibits {damage_desc}.",
            f"Overall health score is evaluated at {health_score}/100, placing the structure in the '{risk_category}' category.",
            f"The Vision AI Engine identified {critical_count} critical or severe defects.",
            f"The worst localized damage occurs on the '{most_damaged}' component.",
            f"Significant crack measurements show a maximum width of {largest_width} mm and maximum length of {largest_len} mm." if largest_width > 0 else "No significant crack defects detected.",
            f"Steel corrosion and rust coverage are measured at {total_corrosion}% of the visible structural area." if total_corrosion > 0 else "No active corrosion detected on visible steel surfaces.",
            f"Immediate maintenance is recommended to prevent structural failure." if risk_category in ("Critical", "Poor") else "Routine monitoring and scheduled maintenance are recommended."
        ]
        summary_report = " ".join(lines)

        return {
            "vision_explanation": vision_explanation,
            "feature_explanation": features_explanation,
            "ml_contributions": ml_contributions,
            "summary_report": summary_report
        }
