"""
BridgeGuardian AI — Base Interfaces for Vision Engine
Provides abstract bases for pluggable detection, segmentation, and feature extraction.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class DetectionResult:
    """Stores the bounding box, class, and confidence for a detected defect/component."""
    def __init__(self, label: str, bbox: List[int], confidence: float) -> None:
        self.label = label      # defect name, e.g. "Crack", "Rust", "Pier", "Girder"
        self.bbox = bbox        # [x, y, w, h] pixel coordinates
        self.confidence = confidence


class SegmentationResult:
    """Stores a binary mask and polygon outline for a segmented defect."""
    def __init__(self, label: str, mask: np.ndarray, polygon: List[List[int]]) -> None:
        self.label = label
        self.mask = mask        # numpy binary mask (uint8) of shape (H, W)
        self.polygon = polygon  # list of points, e.g. [[x1, y1], [x2, y2], ...]


class BaseDetector(ABC):
    """Abstract base for object detection models (e.g. YOLO, RT-DETR)."""
    @abstractmethod
    def detect(self, image: np.ndarray, image_path: str = None) -> List[DetectionResult]:
        pass


class BaseSegmenter(ABC):
    """Abstract base for segmentation models (e.g. SAM2, YOLO Seg)."""
    @abstractmethod
    def segment(self, image: np.ndarray, detections: List[DetectionResult]) -> List[SegmentationResult]:
        pass


class BaseFeatureExtractor(ABC):
    """Abstract base for defect geometry calculations (e.g. length, width, area)."""
    @abstractmethod
    def extract_features(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        segments: List[SegmentationResult],
        pixel_to_mm: float,
    ) -> Dict[str, Any]:
        pass


class BaseImageQualityChecker(ABC):
    """Abstract base for quality check gates."""
    @abstractmethod
    def check_quality(self, image_path: str) -> Dict[str, Any]:
        pass


class BaseDuplicateMerger(ABC):
    """Abstract base for spatial duplicate merging of overlapping drone shots."""
    @abstractmethod
    def merge_duplicates(self, image_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass
