"""
BridgeGuardian AI — OpenCVDuplicateMerger Component
Clusters and merges duplicate defect reports across overlapping drone frames using spatial IoU pre-filtering and ORB descriptors.
Optimized for high-throughput multi-image campaigns.
"""
from __future__ import annotations
import uuid
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from backend.ml.computer_vision.base import BaseDuplicateMerger


class OpenCVDuplicateMerger(BaseDuplicateMerger):
    def __init__(self, match_threshold: float = 0.75, min_match_count: int = 15) -> None:
        self.match_threshold = match_threshold
        self.min_match_count = min_match_count
        self.orb = cv2.ORB_create(nfeatures=250)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def merge_duplicates(self, image_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans all image results, compares defect crops from different images using spatial pre-filtering
        and fast ORB matching, clustering duplicate defects together.
        """
        unique_defects = []
        all_instances = []

        for img_res in image_results:
            img_path = img_res.get("image_path")
            img_name = Path(img_path).name if img_path else "unknown.jpg"

            for det in img_res.get("features", {}).get("defects", []):
                all_instances.append({
                    "img_name": img_name,
                    "img_path": img_path,
                    "defect": det,
                    "img_cv": None  # Loaded lazily only if spatial candidate matches
                })

        visited = set()

        for i, inst1 in enumerate(all_instances):
            if i in visited:
                continue
            visited.add(i)

            defect_id = f"DEFECT-{uuid.uuid4().hex[:6].upper()}"
            group = [inst1]
            det1 = inst1["defect"]
            bx1, by1, bw1, bh1 = det1["bbox"]

            for j in range(i + 1, len(all_instances)):
                if j in visited:
                    continue
                inst2 = all_instances[j]
                det2 = inst2["defect"]

                type1 = det1.get("type", det1.get("label"))
                type2 = det2.get("type", det2.get("label"))
                if type1 != type2:
                    continue

                bx2, by2, bw2, bh2 = det2["bbox"]
                dist = np.sqrt((bx1 - bx2)**2 + (by1 - by2)**2)

                # Fast spatial filtering: skip if bboxes are far apart
                if dist > 100 or abs(bw1 - bw2) > 60 or abs(bh1 - bh2) > 60:
                    continue

                # Close proximity match
                if dist < 40 and abs(bw1 - bw2) < 25:
                    group.append(inst2)
                    visited.add(j)
                    continue

                # Lazy load images for visual comparison only if close spatially
                if inst1["img_cv"] is None and inst1["img_path"] and Path(inst1["img_path"]).exists():
                    inst1["img_cv"] = cv2.imread(inst1["img_path"])
                if inst2["img_cv"] is None and inst2["img_path"] and Path(inst2["img_path"]).exists():
                    inst2["img_cv"] = cv2.imread(inst2["img_path"])

                is_duplicate = False
                if inst1["img_cv"] is not None and inst2["img_cv"] is not None:
                    is_duplicate = self._check_visual_similarity(
                        inst1["img_cv"], det1["bbox"],
                        inst2["img_cv"], det2["bbox"]
                    )

                if is_duplicate:
                    group.append(inst2)
                    visited.add(j)

            representative = max(group, key=lambda x: x["defect"]["confidence"])["defect"]
            severities = [x["defect"].get("severity", "Moderate") for x in group]
            severity_priority = {"Critical": 4, "Severe": 3, "Moderate": 2, "Minor": 1}
            final_severity = max(severities, key=lambda s: severity_priority.get(s, 0))

            unique_defects.append({
                "defect_id": defect_id,
                "type": representative.get("type", representative.get("label", "Crack")),
                "severity": final_severity,
                "confidence": float(np.mean([x["defect"]["confidence"] for x in group])),
                "bbox": representative["bbox"],
                "measurements": representative.get("measurements", {}),
                "images": [x["img_name"] for x in group],
                "occurrences": len(group)
            })

        import logging
        logger = logging.getLogger("bridgeguardian.cv.duplicate_merger")
        logger.info(f"DuplicateMerger: Input Defects ({len(all_instances)}) -> Merged Defects ({len(unique_defects)})")

        if len(all_instances) > 0 and len(unique_defects) == 0:
            from backend.ml.computer_vision.base import BrokenDefectPropagationError
            raise BrokenDefectPropagationError(
                f"BrokenDefectPropagationError: DuplicateMerger received {len(all_instances)} input defects "
                f"but returned 0 merged unique defects."
            )

        return unique_defects

    def _check_visual_similarity(
        self,
        img1: np.ndarray, bbox1: List[int],
        img2: np.ndarray, bbox2: List[int]
    ) -> bool:
        """Helper to crop defect regions, compute ORB descriptors, and check match density."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        crop1 = img1[y1:y1+h1, x1:x1+w1]
        crop2 = img2[y2:y2+h2, x2:x2+w2]

        if crop1.size == 0 or crop2.size == 0:
            return False

        g1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(g1, (80, 80))
        g2 = cv2.resize(g2, (80, 80))

        kp1, des1 = self.orb.detectAndCompute(g1, None)
        kp2, des2 = self.orb.detectAndCompute(g2, None)

        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            res = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val > 0.85

        try:
            matches = self.bf.match(des1, des2)
            good_matches = [m for m in matches if m.distance < 60]
            ratio = len(good_matches) / max(len(des1), len(des2))
            return ratio > 0.35
        except Exception:
            return False
