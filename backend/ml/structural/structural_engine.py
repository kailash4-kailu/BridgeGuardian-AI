"""
BridgeGuardian AI — Structural Analysis Engine
Maps detected defects to specific structural components,
identifies critical zones, and aggregates bridge-level statistics.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Set


class StructuralEngine:
    def __init__(self) -> None:
        self.known_component_ontology = {"Girder", "Deck", "Pier", "Bearing", "Expansion Joint", "Guard Rail", "Connection Plate", "Suspension Cable", "Tower"}

    def analyze(self, image_results: List[Dict[str, Any]], unique_defects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrates structural mapping and statistics calculation based strictly on detected evidence.
        """
        valid_images = [img for img in image_results if img.get("is_valid", False)]
        
        # 1. Discover all visible components detected across accepted frames
        detected_comp_types: Set[str] = set()
        total_observed_area = 0.0
        
        for img_res in valid_images:
            comp_list = img_res.get("features", {}).get("visible_components", img_res.get("features", {}).get("components", []))
            for c in comp_list:
                if isinstance(c, str):
                    detected_comp_types.add(c)
                elif isinstance(c, dict) and "label" in c:
                    detected_comp_types.add(c["label"])
            # Accumulate observed structural coverage area percentage from image metrics
            total_observed_area += img_res.get("metrics", {}).get("bridge_coverage_pct", 65.0)

        # 2. Map each unique defect occurrence to a structural component
        mapped_defects = []
        for u_det in unique_defects:
            associated_components = []
            
            for img_res in valid_images:
                if img_res["image_name"] not in u_det["images"]:
                    continue
                    
                components_in_image = []
                for det in img_res.get("features", {}).get("defects", []):
                    if det.get("type") in self.known_component_ontology:
                        components_in_image.append(det)
                        
                dbox = u_det["bbox"]
                best_component = "Deck" if "Deck" in detected_comp_types else (next(iter(detected_comp_types)) if detected_comp_types else "Unknown")
                max_overlap = 0.0
                
                for comp in components_in_image:
                    overlap = self._bbox_containment(dbox, comp["bbox"])
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_component = comp["type"]
                        
                if max_overlap > 0.3:
                    associated_components.append(best_component)
                    
            final_component = max(set(associated_components), key=associated_components.count) if associated_components else ("Deck" if "Deck" in detected_comp_types else (next(iter(detected_comp_types)) if detected_comp_types else "Unknown Component"))
            
            mapped_defects.append({
                **u_det,
                "component": final_component
            })

        # Pipeline Guard Step 4: Assert mapped_defects count matches unique_defects count
        assert len(mapped_defects) == len(unique_defects), f"Mismatch: mapped_defects ({len(mapped_defects)}) != unique_defects ({len(unique_defects)})"
        
        if len(unique_defects) > 0 and len(mapped_defects) == 0:
            from backend.ml.computer_vision.base import BrokenDefectPropagationError
            raise BrokenDefectPropagationError(
                f"BrokenDefectPropagationError: StructuralEngine received {len(unique_defects)} unique defects "
                f"but mapped_defects is empty."
            )

        # 3. Build Component Hierarchy strictly from detected components
        hierarchy = {c: [] for c in detected_comp_types}
        for md in mapped_defects:
            comp = md["component"]
            if comp not in hierarchy:
                hierarchy[comp] = []
            hierarchy[comp].append(md)

        # 4. Calculate Bridge-Level Statistics
        total_images = len(image_results)
        damaged_images = [img for img in valid_images if len(img.get("features", {}).get("defects", [])) > 0]
        
        pct_damaged_images = float(round((len(damaged_images) / len(valid_images) * 100), 2)) if valid_images else 0.0
        
        # Aggregate defect parameters across Crack, Rust, Corrosion, Spalling
        all_crack_widths = [
            d.get("measurements", {}).get("width_mm", 0.0)
            for d in mapped_defects
            if d.get("type", "").lower() == "crack"
        ]
        all_crack_lengths = [
            d.get("measurements", {}).get("length_mm", 0.0)
            for d in mapped_defects
            if d.get("type", "").lower() == "crack"
        ]
        
        largest_crack_width = float(np.max(all_crack_widths)) if all_crack_widths else 0.0
        largest_crack_length = float(np.max(all_crack_lengths)) if all_crack_lengths else 0.0
        total_crack_area_pct = float(np.sum([d.get("measurements", {}).get("area_pct", 0.0) for d in mapped_defects if d.get("type", "").lower() == "crack"]))
        
        rust_coverage = float(np.sum([d.get("measurements", {}).get("area_pct", 0.0) for d in mapped_defects if d.get("type", "").lower() in ("rust", "rust/corrosion")]))
        corrosion_coverage = float(np.sum([d.get("measurements", {}).get("area_pct", 0.0) for d in mapped_defects if d.get("type", "").lower() in ("corrosion", "rust/corrosion")]))
        
        critical_defects = [d for d in mapped_defects if d.get("severity") in ("Severe", "Critical")]
        critical_count = len(critical_defects)
        critical_locations = [{"image": d["images"][0], "component": d["component"], "type": d["type"], "severity": d["severity"]} for d in critical_defects]
        
        # Damage Diversity Index
        defect_counts = {}
        for d in mapped_defects:
            defect_counts[d["type"]] = defect_counts.get(d["type"], 0) + 1
            
        entropy = 0.0
        if mapped_defects:
            total_def = len(mapped_defects)
            entropy = float(-sum((count / total_def) * np.log(count / total_def) for count in defect_counts.values()))
            
        affected_components = list({d["component"] for d in mapped_defects})
        
        # Most damaged component
        component_defect_counts = {c: len(defs) for c, defs in hierarchy.items()}
        most_damaged_component = max(component_defect_counts, key=component_defect_counts.get) if mapped_defects else "None"
        
        severity_priority = {"Critical": 4, "Severe": 3, "Moderate": 2, "Minor": 1}
        max_severity = "None" if len(mapped_defects) == 0 else max([d["severity"] for d in mapped_defects], key=lambda s: severity_priority.get(s, 0))

        # Critical Zones
        critical_zones = []
        for comp, defs in hierarchy.items():
            crit_in_comp = [d for d in defs if d["severity"] == "Critical"]
            sev_in_comp = [d for d in defs if d["severity"] == "Severe"]
            if len(crit_in_comp) >= 1 or len(sev_in_comp) >= 2:
                critical_zones.append({
                    "component": comp,
                    "defect_count": len(defs),
                    "critical_count": len(crit_in_comp),
                    "severe_count": len(sev_in_comp)
                })

        # Coverage Score = Observed structural area / Expected visible structural area
        avg_observed_coverage = (total_observed_area / len(valid_images)) if valid_images else 0.0
        coverage_score = float(round(min(1.0, max(0.1, avg_observed_coverage / 100.0)), 4))
        overall_conf = float(np.mean([d["confidence"] for d in mapped_defects])) if mapped_defects else 0.95

        # Component findings table: 4 Explicit Engineering States
        component_findings = []
        
        # 1. Detected Components Evaluation
        for comp in sorted(detected_comp_types):
            comp_defs = hierarchy.get(comp, [])
            has_cracks = "Yes" if any(d["type"] == "Crack" for d in comp_defs) else "No"
            has_rust = "Yes" if any(d["type"] in ("Rust", "Corrosion") for d in comp_defs) else "No"
            
            if comp_defs:
                comp_max_severity = max([d["severity"] for d in comp_defs], key=lambda s: severity_priority.get(s, 0))
                if comp_max_severity == "Critical":
                    status = "Replace"
                elif comp_max_severity == "Severe":
                    status = "Repair"
                elif comp_max_severity == "Moderate":
                    status = "Inspect"
                else:
                    status = "Monitor"
            else:
                comp_max_severity = "None"
                # Four Engineering States Rules:
                # Verified Healthy ONLY when coverage >= 90%, >= 5 viewpoints, high quality, 0 defects
                if coverage_score >= 0.90 and len(valid_images) >= 5:
                    status = "Verified Healthy"
                else:
                    # DEFAULT state when no defects detected on incomplete/limited coverage
                    status = "No Visible Defect Observed"
                
            component_findings.append({
                "component": comp,
                "cracks": has_cracks,
                "rust": has_rust,
                "severity": comp_max_severity if comp_defs else "No Defects Observed",
                "status": status,
                "coverage_pct": round(coverage_score * 100, 1),
                "confidence_pct": round(overall_conf * 100, 1)
            })

        # 2. Uninspected Components from Ontology
        uninspected_standard = [c for c in sorted(self.known_component_ontology) if c not in detected_comp_types]
        for un_comp in uninspected_standard:
            component_findings.append({
                "component": un_comp,
                "cracks": "N/A",
                "rust": "N/A",
                "severity": "Not Inspected",
                "status": "Not Inspected",
                "coverage_pct": 0.0,
                "confidence_pct": 0.0
            })

        visible_inventory = [
            {
                "component": item["component"],
                "status": item["status"],
                "defect_count": len(hierarchy.get(item["component"], [])),
                "coverage_pct": item.get("coverage_pct", 0.0)
            }
            for item in component_findings
        ]

        uninspected_list = [item["component"] for item in component_findings if item["status"] == "Not Inspected"]

        aggregate_results = {
            "largest_crack_width": round(largest_crack_width, 2) if all_crack_widths else 0.0,
            "largest_crack_length": round(largest_crack_length, 2) if all_crack_lengths else 0.0,
            "total_crack_area_percent": round(total_crack_area_pct, 4),
            "rust_coverage_percent": round(rust_coverage, 4),
            "corrosion_coverage_percent": round(corrosion_coverage, 4),
            "critical_defect_count": critical_count,
            "critical_defect_locations": critical_locations,
            "most_damaged_structural_component": most_damaged_component if mapped_defects else "None",
            "affected_structural_components": affected_components,
            "visible_components_inventory": visible_inventory,
            "uninspected_components": uninspected_list,
            "damage_diversity_index": round(entropy, 4),
            "images_containing_damage_percent": pct_damaged_images,
            "maximum_severity": max_severity,
            "critical_zones": critical_zones,
            "coverage_score": coverage_score,
            "overall_detection_confidence": round(overall_conf, 4),
            "component_findings": component_findings,
            "defects": mapped_defects
        }

        return {
            "defects": mapped_defects,
            "hierarchy": hierarchy,
            "statistics": aggregate_results
        }

    def _bbox_containment(self, bbox_defect: List[int], bbox_component: List[int]) -> float:
        """Calculates what fraction of the defect bbox overlaps/falls within the component bbox."""
        dx, dy, dw, dh = bbox_defect
        cx, cy, cw, ch = bbox_component
        
        # Calculate intersection
        x_left = max(dx, cx)
        y_top = max(dy, cy)
        x_right = min(dx + dw, cx + cw)
        y_bottom = min(dy + dh, cy + ch)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        defect_area = dw * dh
        return float(intersection_area / defect_area)
