"""
BridgeGuardian AI — Celery Asynchronous Tasks
Defines Celery worker tasks for PDF report generation, multi-image drone campaigns, and ML model re-training.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from datetime import datetime

from backend.core.celery_app import celery_app, HAS_CELERY
from backend.core.database import SessionLocal
from backend.core.models import InspectionRecord, InspectionCampaignRecord, TrainingRun, ModelMetadata, ModelRegistryEntry

logger = logging.getLogger("bridgeguardian.tasks")

# If Celery is installed, wrap tasks with @celery_app.task; otherwise export standard functions
def task_wrapper(name: str):
    if HAS_CELERY and celery_app:
        return celery_app.task(name=name, bind=True)
    def dummy_decorator(func):
        return func
    return dummy_decorator


@task_wrapper("generate_pdf_report_task")
def generate_pdf_report_task(self_or_inspection_id, inspection_id: int = None):
    """Asynchronous PDF report compilation task."""
    rec_id = inspection_id if inspection_id is not None else self_or_inspection_id
    logger.info(f"Starting PDF generation task for inspection #{rec_id}")

    db = SessionLocal()
    try:
        inspection = db.query(InspectionRecord).filter(InspectionRecord.id == rec_id).first()
        if not inspection:
            logger.error(f"Inspection record #{rec_id} not found.")
            return {"status": "FAILED", "reason": "Record not found"}

        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        reports_dir = Path("backend/static/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = reports_dir / f"inspection_report_{rec_id}.pdf"

        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"<b>BridgeGuardian AI — Inspection Report #{rec_id}</b>", styles['Heading1']),
            Spacer(1, 12),
            Paragraph(f"Status: {inspection.status}", styles['Normal']),
            Paragraph(f"Health Score: {inspection.health_score or 0.0:.2f}/100", styles['Normal']),
            Paragraph(f"Risk Category: {inspection.risk_category or 'N/A'}", styles['Normal']),
            Paragraph(f"Generated At: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Normal']),
        ]
        doc.build(story)

        inspection.pdf_report_path = f"/static/reports/inspection_report_{rec_id}.pdf"
        inspection.status = "COMPLETED"
        inspection.progress = 100.0
        db.commit()

        logger.info(f"PDF report generated successfully at {pdf_path}")
        return {"status": "SUCCESS", "pdf_path": str(pdf_path)}
    except Exception as exc:
        logger.error(f"Failed to generate PDF for inspection #{rec_id}: {exc}")
        if db:
            db.rollback()
        return {"status": "FAILED", "error": str(exc)}
    finally:
        db.close()


@task_wrapper("process_inspection_campaign_task")
def process_inspection_campaign_task(self_or_campaign_id, campaign_id: str = None):
    """Asynchronous multi-image drone campaign processing task."""
    c_id = campaign_id if campaign_id is not None else self_or_campaign_id
    logger.info(f"Starting campaign processing task for campaign {c_id}")

    db = SessionLocal()
    try:
        campaign = db.query(InspectionCampaignRecord).filter(InspectionCampaignRecord.campaign_id == c_id).first()
        if not campaign:
            logger.error(f"Campaign {c_id} not found.")
            return {"status": "FAILED", "reason": "Campaign not found"}

        campaign.status = "PROCESSING"
        db.commit()

        # Simulate progressive processing steps
        total = campaign.total_images or 1
        for i in range(1, total + 1):
            campaign.processed_images = i
            db.commit()

        campaign.status = "COMPLETED"
        db.commit()
        return {"status": "COMPLETED", "campaign_id": c_id, "processed": total}
    except Exception as exc:
        logger.error(f"Campaign processing error for {c_id}: {exc}")
        if db:
            db.rollback()
        return {"status": "FAILED", "error": str(exc)}
    finally:
        db.close()


@task_wrapper("retrain_model_task")
def retrain_model_task(self_or_run_id, run_id: int = None):
    """Asynchronous ML model re-training task."""
    r_id = run_id if run_id is not None else self_or_run_id
    logger.info(f"Starting ML model re-training task run #{r_id}")

    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == r_id).first()
        if run:
            run.status = "RUNNING"
            db.commit()

        # Run pipeline trainer logic
        from ml_pipeline.train import main as run_trainer
        run_trainer()

        if run:
            run.status = "COMPLETED"
            run.completed_at = datetime.utcnow()
            run.log = "Model training execution completed successfully."
            db.commit()

        return {"status": "COMPLETED", "run_id": r_id}
    except Exception as exc:
        logger.error(f"Model retraining failed for run #{r_id}: {exc}")
        if db:
            run = db.query(TrainingRun).filter(TrainingRun.id == r_id).first()
            if run:
                run.status = "FAILED"
                run.log = str(exc)
                db.commit()
        return {"status": "FAILED", "error": str(exc)}
    finally:
        db.close()
