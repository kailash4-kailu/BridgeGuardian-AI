"""
BridgeGuardian AI — Campaign Inspection API Endpoints
Provides routes for batch upload, run-inspection queue, polling status, and downloading reports.
"""
from __future__ import annotations
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.models import InspectionRecord
from backend.ml.computer_vision.inspection_pipeline import CampaignInspectionPipeline

from backend.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/inspection", tags=["Drone Inspection"])


# Request payload schemas
class RunInspectionRequest(BaseModel):
    image_paths: List[str]
    pixel_to_mm: float = 0.5


@router.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Saves multiple uploaded drone images to the uploads directory.
    Returns names and paths for processing.
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    
    uploaded_files = []
    for file in files:
        # Check support formats
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{suffix}'. Allowed formats: jpg, jpeg, png."
            )
            
        dest_path = upload_dir / file.filename
        
        # Save to disk
        try:
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file '{file.filename}': {str(e)}")
            
        uploaded_files.append({
            "filename": file.filename,
            "filepath": str(dest_path.resolve())
        })
        
    return uploaded_files


@router.post("/run-inspection")
async def run_inspection(
    request: RunInspectionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Initiates a new multi-image inspection campaign and schedules the background task.
    """
    if not request.image_paths:
        raise HTTPException(status_code=400, detail="No image paths provided for inspection.")
        
    # Verify image paths exist
    for p in request.image_paths:
        if not Path(p).exists():
            raise HTTPException(status_code=400, detail=f"Image path does not exist: {p}")
            
    # Create DB record
    record = InspectionRecord(
        status="queued",
        progress=0.0,
        images_json=json.dumps([Path(p).name for p in request.image_paths])
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    # Schedule background process
    pipeline = CampaignInspectionPipeline()
    background_tasks.add_task(
        pipeline.run_campaign,
        db=db,
        inspection_id=record.id,
        image_paths=request.image_paths,
        pixel_to_mm=request.pixel_to_mm
    )
    
    return {
        "inspection_id": record.id,
        "status": record.status,
        "progress": record.progress,
        "created_at": record.created_at.isoformat()
    }


@router.get("/{id}")
async def get_inspection_details(id: int, db: Session = Depends(get_db)):
    """
    Fetches the details and processing progress of a given inspection campaign.
    """
    record = db.query(InspectionRecord).filter(InspectionRecord.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Inspection campaign record not found.")
        
    # Helper to safely load JSON
    def safe_json_load(data):
        if not data:
            return None
        try:
            return json.loads(data)
        except Exception:
            return None

    # Load performance metrics and model metadata
    perf_metrics = safe_json_load(record.performance_metrics_json)
    model_metadata = safe_json_load(record.model_metadata_json)
    agg_stats = safe_json_load(record.aggregate_results_json)
    image_results = safe_json_load(record.image_results_json)
    
    # Vision level and general explanations
    explainability = None
    if record.status == "completed" and record.summary_report:
        # Reconstruct structured explainability from summary and aggregate stats
        explainability = {
            "summary_report": record.summary_report,
            "vision_explanation": f"Identified visual defects located across components.",
            "feature_explanation": [
                f"Max Crack Width: {agg_stats.get('largest_crack_width', 0.0)} mm" if agg_stats else "",
                f"Rust Coverage: {agg_stats.get('rust_coverage_percent', 0.0)}%" if agg_stats else ""
            ],
            "ml_contributions": [
                f"Adjusted Health Score: {record.health_score}%",
                f"Calculated Failure Probability: {record.failure_probability}%"
            ]
        }

    return {
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "status": record.status,
        "progress": record.progress,
        "health_score": record.health_score,
        "failure_probability": record.failure_probability,
        "rul_days": record.rul_days,
        "risk_category": record.risk_category,
        "maintenance_priority": record.maintenance_priority,
        "maintenance_action": record.maintenance_action,
        "repair_window_days": record.repair_window_days,
        "inspection_interval_days": record.inspection_interval_days,
        "summary_report": record.summary_report,
        "explainability": explainability,
        "aggregate_results": agg_stats,
        "image_results": image_results,
        "performance_metrics": perf_metrics,
        "model_metadata": model_metadata
    }


@router.get("/report/{id}")
async def download_report_pdf(id: int, db: Session = Depends(get_db)):
    """
    Downloads the compiled PDF report for a completed inspection campaign.
    """
    record = db.query(InspectionRecord).filter(InspectionRecord.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Inspection campaign record not found.")
        
    if record.status != "completed":
        raise HTTPException(status_code=400, detail=f"PDF report not ready. Current status: {record.status}.")
        
    pdf_path = Path(record.pdf_report_path) if record.pdf_report_path else None
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Compiled PDF report file not found on disk.")
        
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name
    )
