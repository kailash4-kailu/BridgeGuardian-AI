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
        self.component_classes = {"Girder", "Deck", "Pier", "Bearing", "Expansion Joint", "Guard Rail", "Connection Plate"}

    def analyze(self, image_results: List[Dict[str, Any]], unique_defects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrates structural mapping and statistics calculation.
        """
        # 1. Map each unique defect occurrence to a structural component
        # We do this by analyzing the instances in image_results
        mapped_defects = []
        for u_det in unique_defects:
            # We look at the occurrences of this defect in various images
            associated_components = []
            
            # Find matching detections in image_results
            for img_res in image_results:
                if not img_res.get("is_valid") or img_res["image_name"] not in u_det["images"]:
                    continue
                    
                # Find the bounding boxes of structural elements in this image
                components_in_image = []
                for det in img_res.get("features", {}).get("defects", []):
                    # In YOLODetector raw detections, look for structural components
                    # Or check bboxes of components directly in image_results
                    pass
                
                # Let's search YOLODetector outputs
                # Since we mapped bboxes, let's scan components in the same image
                for det in img_res.get("features", {}).get("defects", []):
                    if det["type"] in self.component_classes:
                        components_in_image.append(det)
                        
                # Find which component has maximum overlap with this defect's bbox
                best_component = "Deck" # Default fallback
                max_overlap = 0.0
                
                dbox = u_det["bbox"]
                for comp in components_in_image:
                    overlap = self._bbox_containment(dbox, comp["bbox"])
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_component = comp["type"]
                        
                if max_overlap > 0.3:
                    associated_components.append(best_component)
                    
            # Set the component (majority vote)
            final_component = max(set(associated_components), key=associated_components.count) if associated_components else "Deck"
            
            # Add mapping details
            mapped_defects.append({
                **u_det,
                "component": final_component
            })

        # 2. Build Component Hierarchy
        hierarchy = {c: [] for c in self.component_classes}
        hierarchy["Deck"] = [] # make sure default is present
        
        for md in mapped_defects:
            comp = md["component"]
            if comp not in hierarchy:
                hierarchy[comp] = []
            hierarchy[comp].append(md)

        # 3. Calculate Bridge-Level Statistics
        total_images = len(image_results)
        valid_images = [img for img in image_results if img.get("is_valid", False)]
        damaged_images = [img for img in valid_images if len(img.get("features", {}).get("defects", [])) > 0]
        
        pct_damaged_images = float(round((len(damaged_images) / len(valid_images) * 100), 2)) if valid_images else 0.0
        
        # Aggregate defect-specific parameters
        all_crack_widths = [d["measurements"]["width_mm"] for d in mapped_defects if d["type"] == "Crack"]
        all_crack_lengths = [d["measurements"]["length_mm"] for d in mapped_defects if d["type"] == "Crack"]
        all_corrosion_pacts = [d["measurements"]["area_pct"] for d in mapped_defects if d["type"] in ("Rust", "Corrosion")]
        
        largest_crack_width = float(np.max(all_crack_widths)) if all_crack_widths else 0.0
        largest_crack_length = float(np.max(all_crack_lengths)) if all_crack_lengths else 0.0
        total_crack_area_pct = float(np.sum([d["measurements"]["area_pct"] for d in mapped_defects if d["type"] == "Crack"]))
        
        rust_coverage = float(np.sum([d["measurements"]["area_pct"] for d in mapped_defects if d["type"] == "Rust"]))
        corrosion_coverage = float(np.sum([d["measurements"]["area_pct"] for d in mapped_defects if d["type"] == "Corrosion"]))
        
        critical_defects = [d for d in mapped_defects if d["severity"] in ("Severe", "Critical")]
        critical_count = len(critical_defects)
        critical_locations = [{"image": d["images"][0], "component": d["component"], "type": d["type"], "severity": d["severity"]} for d in critical_defects]
        
        # Damage Diversity Index (Entropy of defect types present)
        defect_counts = {}
        for d in mapped_defects:
            defect_counts[d["type"]] = defect_counts.get(d["type"], 0) + 1
            
        entropy = 0.0
        if mapped_defects:
            total_def = len(mapped_defects)
            entropy = float(-sum((count / total_def) * np.log(count / total_def) for count in defect_counts.values()))
            
        # Affected Structural Components
        affected_components = list({d["component"] for d in mapped_defects})
        
        # Most damaged component (by count of defects)
        component_defect_counts = {c: len(defs) for c, defs in hierarchy.items()}
        most_damaged_component = max(component_defect_counts, key=component_defect_counts.get) if mapped_defects else "None"
        
        # Severity ranking
        severity_priority = {"Critical": 4, "Severe": 3, "Moderate": 2, "Minor": 1}
        max_severity = "Minor"
        if mapped_defects:
            max_severity = max([d["severity"] for d in mapped_defects], key=lambda s: severity_priority.get(s, 0))

        # Detect Critical Zones
        # A component is a Critical Zone if it has at least 1 Critical defect or >= 2 Severe defects
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

        # Component findings table
        component_findings = []
        for comp in sorted(self.component_classes):
            comp_defs = hierarchy.get(comp, [])
            has_cracks = "Yes" if any(d["type"] == "Crack" for d in comp_defs) else "No"
            has_rust = "Yes" if any(d["type"] in ("Rust", "Corrosion") for d in comp_defs) else "No"
            
            # Max severity
            if comp_defs:
                comp_max_severity = max([d["severity"] for d in comp_defs], key=lambda s: severity_priority.get(s, 0))
            else:
                comp_max_severity = "Healthy"
                
            # Status / action
            if comp_max_severity == "Critical":
                status = "Replace"
            elif comp_max_severity == "Severe":
                status = "Repair"
            elif comp_max_severity == "Moderate":
                status = "Inspect"
            elif comp_max_severity == "Minor":
                status = "Monitor"
            else:
                status = "OK"
                
            component_findings.append({
                "component": comp,
                "cracks": has_cracks,
                "rust": has_rust,
                "severity": comp_max_severity,
                "status": status
            })

        # Calculate coverage score and overall confidence
        detected_comp_types = set()
        for img_res in valid_images:
            comp_list = img_res.get("features", {}).get("components", [])
            for c in comp_list:
                detected_comp_types.add(c)
                
        coverage_score = float(len(detected_comp_types) / len(self.component_classes)) if self.component_classes else 1.0
        overall_conf = float(np.mean([d["confidence"] for d in mapped_defects])) if mapped_defects else 0.95

        aggregate_results = {
            "largest_crack_width": round(largest_crack_width, 2),
            "largest_crack_length": round(largest_crack_length, 2),
            "total_crack_area_percent": round(total_crack_area_pct, 4),
            "rust_coverage_percent": round(rust_coverage, 4),
            "corrosion_coverage_percent": round(corrosion_coverage, 4),
            "critical_defect_count": critical_count,
            "critical_defect_locations": critical_locations,
            "most_damaged_structural_component": most_damaged_component,
            "affected_structural_components": affected_components,
            "damage_diversity_index": round(entropy, 4),
            "images_containing_damage_percent": pct_damaged_images,
            "maximum_severity": max_severity,
            "critical_zones": critical_zones,
            "coverage_score": round(coverage_score, 4),
            "overall_detection_confidence": round(overall_conf, 4),
            "component_findings": component_findings
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
