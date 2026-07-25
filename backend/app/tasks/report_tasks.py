"""
BridgeGuardian AI — Celery Report Generation Tasks
Background tasks for generating ReportLab PDF inspection campaign assessment reports.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("bridgeguardian.tasks.report")


def compile_pdf_report_task(
    inspection_id: int,
    health_predictions: Dict[str, Any],
    aggregate_stats: Dict[str, Any],
    explainability: Dict[str, Any],
    maintenance: Dict[str, Any],
    image_results: List[Dict[str, Any]],
    model_metadata: Dict[str, Any],
    performance_metrics: Dict[str, Any],
) -> str:
    """Background task to compile dynamic ReportLab PDF inspection reports."""
    logger.info(f"Compiling PDF report task for inspection ID: {inspection_id}")
    from backend.ml.report.report_engine import ReportEngine

    engine = ReportEngine()
    pdf_path = engine.generate_pdf_report(
        inspection_id=inspection_id,
        health_predictions=health_predictions,
        aggregate_stats=aggregate_stats,
        explainability=explainability,
        maintenance=maintenance,
        image_results=image_results,
        model_metadata=model_metadata,
        performance_metrics=performance_metrics,
    )
    return pdf_path
