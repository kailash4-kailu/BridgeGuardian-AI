"""
BridgeGuardian AI — Maintenance AI Engine
Determines actionable maintenance strategies, repair urgency, and follow-up schedules.
"""
from __future__ import annotations
from typing import Any, Dict

class MaintenanceEngine:
    def determine_action_plan(
        self,
        health_predictions: Dict[str, Any],
        aggregate_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Derives action, priority, and schedules based on ML outputs and maximum defect severity.
        """
        risk_cat = health_predictions.get("risk_category", "Excellent")
        max_severity = aggregate_stats.get("maximum_severity", "Minor")
        critical_count = aggregate_stats.get("critical_defect_count", 0)
        
        # Urgency Priority Mapping
        # Priority can scale higher than the standard ML risk category if there is an extreme visual defect
        priority = "Low"
        if risk_cat == "Critical" or max_severity == "Critical" or critical_count >= 3:
            priority = "Critical"
        elif risk_cat == "Poor" or max_severity == "Severe" or critical_count >= 1:
            priority = "High"
        elif risk_cat == "Fair" or max_severity == "Moderate":
            priority = "Medium"
        elif risk_cat in ("Good", "Excellent"):
            priority = "Low"
            
        # Action Plan Rules
        if priority == "Critical":
            action = "Replace"
            repair_window = 7  # 7 days
            next_inspect = 30  # Re-inspect in 1 month
        elif priority == "High":
            action = "Repair"
            repair_window = 30  # 30 days
            next_inspect = 90  # Re-inspect in 3 months
        elif priority == "Medium":
            action = "Repair"
            repair_window = 90  # 3 months
            next_inspect = 180  # Re-inspect in 6 months
        else: # Low priority
            action = "Monitor"
            repair_window = 365  # 1 year
            next_inspect = 365  # Re-inspect in 12 months

        return {
            "maintenance_action": action,
            "maintenance_priority": priority,
            "repair_window_days": repair_window,
            "inspection_interval_days": next_inspect,
            "recommendation_summary": self._generate_recommendation(action, priority, repair_window, next_inspect)
        }

    def _generate_recommendation(self, action: str, priority: str, window: int, interval: int) -> str:
        if priority == "Critical":
            return (
                f"🚨 URGENT ACTION REQUIRED: Scheduled for immediate structural {action.lower()} within {window} days. "
                f"Deploy load restrictions or immediate structural shoring. Re-inspection interval set to {interval} days."
            )
        elif priority == "High":
            return (
                f"⚠️ HIGH PRIORITY: Planned structural {action.lower()} is recommended within {window} days. "
                f"Close monitoring of active cracks and localized corrosion. Re-inspection interval scheduled for {interval} days."
            )
        elif priority == "Medium":
            return (
                f"📋 MODERATE PRIORITY: Schedule structural {action.lower()} within {window} days. "
                f"Perform maintenance checks on seals and bearings. Re-inspect bridge status in {interval} days."
            )
        else:
            return (
                f"🟢 ROUTINE MAINTENANCE: Standard structural monitoring set. "
                f"No immediate repairs needed. Continue regular visual inspections every {interval} days."
            )
