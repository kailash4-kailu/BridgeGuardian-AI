"""
BridgeGuardian AI — Non-Linear Soft-Min Structural Health Aggregation Engine
Implements worst-defect dominance penalty math to guarantee severe defects force component status to Critical/Repair.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List


class StructuralHealthAggregator:
    """
    Computes bridge component structural health index H(c) in range [0.0, 100.0]
    using a non-linear worst-defect soft-min penalty equation.
    """

    SEVERITY_WEIGHTS: Dict[str, float] = {
        "CRACK_HAIRLINE": 0.15,
        "CRACK_MODERATE": 0.45,
        "CRACK_SEVERE": 0.85,
        "RUST_MINOR": 0.20,
        "RUST_SEVERE": 0.70,
        "SPALLING_MODERATE": 0.55,
        "SPALLING_SEVERE": 0.90,
        "EXPOSED_REBAR": 0.98,
        "BEARING_DISPLACEMENT": 0.92,
    }

    def compute_component_health(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes component health score H(c) and maps to categorical status.

        Formula:
            H(c) = 100 * (1.0 - max(w_i * p_i)) * prod(1.0 - 0.10 * w_i * p_i)
        """
        if not detections:
            return {
                "health_score": 100.0,
                "status_category": "HEALTHY",
                "worst_defect_class": None,
                "max_penalty": 0.0,
                "recommendation": "Routine Visual Monitoring",
            }

        max_penalty = 0.0
        worst_defect = None
        multiplicative_product = 1.0

        for d in detections:
            cls_name = d.get("defect_class", "CRACK_HAIRLINE").upper()
            confidence = float(d.get("confidence", 1.0))
            weight = self.SEVERITY_WEIGHTS.get(cls_name, 0.35)

            effective_penalty = weight * confidence

            if effective_penalty > max_penalty:
                max_penalty = effective_penalty
                worst_defect = cls_name

            # Secondary multiplicative factor
            multiplicative_product *= (1.0 - 0.10 * effective_penalty)

        # Soft-min non-linear health score equation
        health_score = 100.0 * (1.0 - max_penalty) * multiplicative_product
        health_score = max(0.0, min(100.0, round(health_score, 2)))

        # Categorical Status Mapping
        if health_score < 30.0 or max_penalty >= 0.85:
            status = "CRITICAL"
            recommendation = "Immediate Emergency Structural Repair Required"
        elif health_score < 55.0 or max_penalty >= 0.60:
            status = "SEVERE"
            recommendation = "Schedule Maintenance & Load Limit Restriction"
        elif health_score < 75.0 or max_penalty >= 0.35:
            status = "MODERATE"
            recommendation = "Plan Target Inspection within 30 Days"
        elif health_score < 90.0:
            status = "MINOR"
            recommendation = "Routine Maintenance Scheduling"
        else:
            status = "HEALTHY"
            recommendation = "Routine Annual Monitoring"

        return {
            "health_score": health_score,
            "status_category": status,
            "worst_defect_class": worst_defect,
            "max_penalty": round(max_penalty, 4),
            "recommendation": recommendation,
        }
