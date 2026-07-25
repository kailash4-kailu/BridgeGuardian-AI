"""
BridgeGuardian AI — Celery Vision Processing Tasks
Background tasks for processing drone damage inspection imagery.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("bridgeguardian.tasks.vision")


def process_drone_image_task(image_path: str, save_annotated: bool = True) -> Dict[str, Any]:
    """Background task to run OpenCV damage detection on a single drone photo."""
    logger.info(f"Processing drone image task: {image_path}")
    from backend.ml.computer_vision.vision_inference import VisionInferencePipeline

    pipeline = VisionInferencePipeline()
    result = pipeline.analyze_image(image_path, save_annotated=save_annotated)
    return result


def process_inspection_campaign_task(image_paths: List[str], campaign_id: int) -> Dict[str, Any]:
    """Background task to run campaign damage detection across multiple drone photos."""
    logger.info(f"Processing inspection campaign {campaign_id} with {len(image_paths)} images")
    from backend.ml.computer_vision.inspection_pipeline import InspectionPipeline

    pipeline = InspectionPipeline()
    summary = pipeline.run_campaign(image_paths=image_paths, campaign_id=campaign_id)
    return summary
