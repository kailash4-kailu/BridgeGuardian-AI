"""
BridgeGuardian AI — Structural Reasoning Engine
Enforces context-aware rules between bridge components and candidate defects.
Validates structural compatibility and rejects impossible defect predictions.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Set, Any, Optional

logger = logging.getLogger("bridgeguardian.cv.structural_reasoning")

class StructuralReasoningEngine:
    """
    Rule engine enforcing physical and domain compatibility constraints
    between bridge structural components and detected surface defects.
    """

    # Allowed defects per structural component class
    COMPONENT_DEFECT_RULES: Dict[str, Set[str]] = {
        "Suspension Cable": {
            "Rust", "Corrosion", "Rust/Corrosion", "Broken Strands",
            "Paint Peeling", "Cable Tension Loss"
        },
        "Cable": {
            "Rust", "Corrosion", "Rust/Corrosion", "Broken Strands",
            "Paint Peeling", "Cable Tension Loss"
        },
        "Tower": {
            "Crack", "Spalling", "Erosion", "Scaling", "Water Leakage",
            "Rust", "Corrosion", "Rust/Corrosion", "Paint Peeling"
        },
        "Pylon": {
            "Crack", "Spalling", "Erosion", "Scaling", "Water Leakage",
            "Rust", "Corrosion", "Rust/Corrosion", "Paint Peeling"
        },
        "Deck": {
            "Crack", "Pothole", "Spalling", "Water Leakage", "Vegetation",
            "Surface Deformation", "Erosion", "Scaling"
        },
        "Concrete Deck": {
            "Crack", "Pothole", "Spalling", "Water Leakage", "Vegetation",
            "Surface Deformation", "Erosion", "Scaling"
        },
        "Steel Girder": {
            "Rust", "Corrosion", "Rust/Corrosion", "Fatigue Crack", "Crack",
            "Paint Peeling", "Loose Connection", "Missing Bolt", "Missing Nut"
        },
        "Girder": {
            "Rust", "Corrosion", "Rust/Corrosion", "Fatigue Crack", "Crack",
            "Paint Peeling", "Loose Connection", "Missing Bolt", "Missing Nut"
        },
        "Pier": {
            "Crack", "Spalling", "Erosion", "Scaling", "Water Leakage",
            "Vegetation", "Scour"
        },
        "Bearing": {
            "Bearing Movement", "Corrosion", "Rust", "Rust/Corrosion",
            "Deformation", "Loose Connection"
        },
        "Expansion Joint": {
            "Expansion Joint Damage", "Deformation", "Water Leakage",
            "Debris Accumulation", "Loose Connection", "Crack"
        },
        "Guard Rail": {
            "Guard Rail Damage", "Rust", "Corrosion", "Rust/Corrosion",
            "Deformation", "Paint Peeling"
        },
        "Connection Plate": {
            "Missing Bolt", "Missing Nut", "Loose Connection", "Rust",
            "Corrosion", "Rust/Corrosion", "Crack", "Fatigue Crack"
        }
    }

    # Explicitly forbidden defect-component pairs
    FORBIDDEN_PAIRS: List[tuple[str, str]] = [
        ("Suspension Cable", "Crack"),
        ("Suspension Cable", "Deck Crack"),
        ("Suspension Cable", "Spalling"),
        ("Suspension Cable", "Missing Bolt"),
        ("Suspension Cable", "Missing Nut"),
        ("Suspension Cable", "Expansion Joint Damage"),
        ("Suspension Cable", "Bearing Failure"),
        ("Cable", "Missing Bolt"),
        ("Cable", "Spalling"),
        ("Concrete Deck", "Missing Bolt"),
        ("Deck", "Missing Bolt"),
        ("Pier", "Missing Bolt"),
        ("Steel Girder", "Spalling"),
        ("Steel Girder", "Pothole"),
    ]

    def validate_defect_compatibility(self, defect_label: str, component_label: str) -> bool:
        """
        Validates if a defect label is physically plausible on a given bridge component.
        
        Args:
            defect_label: E.g., "Missing Bolt", "Crack", "Rust", "Spalling"
            component_label: E.g., "Suspension Cable", "Steel Girder", "Deck"
            
        Returns:
            True if defect is compatible with component, False if rejected.
        """
        if not component_label or component_label == "Unknown" or component_label == "Background":
            # If no component label, allow general defects but reject specific mechanical items
            if defect_label in ("Missing Bolt", "Missing Nut", "Expansion Joint Damage"):
                logger.info(f"Rejected defect '{defect_label}' because component is '{component_label}'")
                return False
            return True

        # Check explicit forbidden pairs first
        for forbidden_comp, forbidden_def in self.FORBIDDEN_PAIRS:
            if component_label.lower() in forbidden_comp.lower() and defect_label.lower() in forbidden_def.lower():
                logger.warning(
                    f"Structural Reasoning REJECTED: Impossible defect '{defect_label}' on component '{component_label}'"
                )
                return False

        # Find allowed set for component
        allowed_defects = None
        for comp_key, allowed_set in self.COMPONENT_DEFECT_RULES.items():
            if comp_key.lower() == component_label.lower() or comp_key.lower() in component_label.lower():
                allowed_defects = allowed_set
                break

        if allowed_defects is not None:
            # Normalize label check
            is_allowed = any(
                allowed_def.lower() in defect_label.lower() or defect_label.lower() in allowed_def.lower()
                for allowed_def in allowed_defects
            )
            if not is_allowed:
                logger.warning(
                    f"Structural Reasoning REJECTED: Defect '{defect_label}' not in allowed set for '{component_label}'"
                )
                return False
            return True

        return True

    def filter_incompatible_defects(
        self,
        defects: List[Dict[str, Any]],
        component_label: str
    ) -> List[Dict[str, Any]]:
        """
        Filters a list of defect dictionaries against a target component label.
        """
        valid_defects = []
        for det in defects:
            label = det.get("label") or det.get("type")
            if self.validate_defect_compatibility(label, component_label):
                valid_defects.append(det)
            else:
                logger.info(f"Filtered out incompatible defect '{label}' on component '{component_label}'")
        return valid_defects
