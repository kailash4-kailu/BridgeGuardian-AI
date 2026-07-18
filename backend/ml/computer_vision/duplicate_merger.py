"""
BridgeGuardian AI — OpenCVDuplicateMerger Component
Clusters and merges duplicate defect reports across overlapping drone frames using ORB descriptors and spatial heuristics.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from backend.ml.computer_vision.base import BaseDuplicateMerger

class OpenCVDuplicateMerger(BaseDuplicateMerger):
    def __init__(self, match_threshold: float = 0.75, min_match_count: int = 15) -> None:
        self.match_threshold = match_threshold
        self.min_match_count = min_match_count
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def merge_duplicates(self, image_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans all image results, compares defect crops from different images using ORB matching,
        and clusters duplicate defects together.
        Returns a list of unique bridge-level defects.
        """
        unique_defects = []
        defect_id_counter = 1
        
        # Flatten all defects with their source image info
        all_instances = []
        for img_res in image_results:
            img_path = img_res.get("image_path")
            img_name = Path(img_path).name if img_path else "unknown.jpg"
            img_cv = cv2.imread(img_path) if img_path and Path(img_path).exists() else None
            
            for det in img_res.get("features", {}).get("defects", []):
                all_instances.append({
                    "img_name": img_name,
                    "img_cv": img_cv,
                    "defect": det
                })
                
        # Group duplicates
        visited = set()
        
        for i, inst1 in enumerate(all_instances):
            if i in visited:
                continue
            visited.add(i)
            
            import uuid
            defect_id = f"DEFECT-{uuid.uuid4().hex[:6].upper()}"
            
            group = [inst1]
            det1 = inst1["defect"]
            
            # Compare with all subsequent instances
            for j in range(i + 1, len(all_instances)):
                if j in visited:
                    continue
                inst2 = all_instances[j]
                det2 = inst2["defect"]
                
                # Must be of the same defect type to be the same physical defect
                if det1["type"] != det2["type"]:
                    continue
                    
                # Visual match check if images are loaded
                is_duplicate = False
                if inst1["img_cv"] is not None and inst2["img_cv"] is not None:
                    is_duplicate = self._check_visual_similarity(
                        inst1["img_cv"], det1["bbox"],
                        inst2["img_cv"], det2["bbox"]
                    )
                else:
                    # Simple heuristic fallback if images aren't available:
                    # if the coordinates and size are extremely close, treat as duplicate
                    bx1, by1, bw1, bh1 = det1["bbox"]
                    bx2, by2, bw2, bh2 = det2["bbox"]
                    dist = np.sqrt((bx1 - bx2)**2 + (by1 - by2)**2)
                    if dist < 30 and abs(bw1 - bw2) < 20:
                        is_duplicate = True
                        
                if is_duplicate:
                    group.append(inst2)
                    visited.add(j)
            
            # Determine the representative defect info (use the one with highest confidence)
            representative = max(group, key=lambda x: x["defect"]["confidence"])["defect"]
            
            # Aggregate severity (take the maximum severity present in the group)
            severities = [x["defect"]["severity"] for x in group]
            severity_priority = {"Critical": 4, "Severe": 3, "Moderate": 2, "Minor": 1}
            final_severity = max(severities, key=lambda s: severity_priority.get(s, 0))
            
            # Unique defect summary
            unique_defects.append({
                "defect_id": defect_id,
                "type": representative["type"],
                "severity": final_severity,
                "confidence": float(np.mean([x["defect"]["confidence"] for x in group])),
                "bbox": representative["bbox"],
                "measurements": representative["measurements"],
                "images": [x["img_name"] for x in group],
                "occurrences": len(group)
            })
            
        return unique_defects

    def _check_visual_similarity(
        self,
        img1: np.ndarray, bbox1: List[int],
        img2: np.ndarray, bbox2: List[int]
    ) -> bool:
        """Helper to crop defect regions, compute ORB descriptors, and check match density."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Crop defect patches
        crop1 = img1[y1:y1+h1, x1:x1+w1]
        crop2 = img2[y2:y2+h2, x2:x2+w2]
        
        if crop1.size == 0 or crop2.size == 0:
            return False
            
        # Convert to grayscale
        g1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
        
        # Resize to match size for comparison
        g1 = cv2.resize(g1, (100, 100))
        g2 = cv2.resize(g2, (100, 100))
        
        # Find keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(g1, None)
        kp2, des2 = self.orb.detectAndCompute(g2, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            # Fallback: SSIM-like template match
            res = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val > 0.82
            
        # Match descriptors
        try:
            matches = self.bf.match(des1, des2)
            good_matches = [m for m in matches if m.distance < 60]
            ratio = len(good_matches) / max(len(des1), len(des2))
            return ratio > 0.35 # Match density threshold
        except Exception:
            return False
