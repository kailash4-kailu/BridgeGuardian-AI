"""
BridgeGuardian AI — Campaign Inspection API Endpoints
Provides routes for batch upload, run-inspection queue, polling status, and downloading reports.
Includes step-by-step diagnostic logging for upload operations.
"""
from __future__ import annotations

import shutil
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.models import InspectionRecord
from backend.ml.computer_vision.inspection_pipeline import CampaignInspectionPipeline
from backend.core.config import get_settings

logger = logging.getLogger("bridgeguardian.inspection")
settings = get_settings()

router = APIRouter(prefix="/inspection", tags=["Drone Inspection"])


class RunInspectionRequest(BaseModel):
    image_paths: List[str]
    pixel_to_mm: float = 0.5


@router.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Saves multiple uploaded drone images to the static uploads directory.
    Validates file formats, sizes, and returns structured metadata immediately.
    """
    t0 = time.perf_counter()
    logger.info("[1] Request received for /upload-images")

    try:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        num_files = len(files) if files else 0
        logger.info(f"[2] Multipart parsed: {num_files} files received")

        if not files or num_files == 0:
            logger.warning("[3] No files provided in request.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided in request."
            )

        logger.info(f"[3] Validating {num_files} files...")
        uploaded_files = []

        for i, file in enumerate(files, 1):
            filename = file.filename or f"uploaded_image_{i}.jpg"
            suffix = Path(filename).suffix.lower()

            if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
                logger.warning(f"[Validation Failed] File {i} '{filename}' unsupported format '{suffix}'")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file format '{suffix}' for '{filename}'."
                )

            dest_path = upload_dir / filename

            logger.info(f"[4] Reading UploadFile {i}/{num_files}: {filename}")
            content = await file.read()

            file_size = len(content)
            if file_size > settings.max_file_size:
                logger.warning(f"[Size Check Failed] File {i} '{filename}' size {file_size} > {settings.max_file_size}")
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File '{filename}' exceeds maximum allowed size of {settings.max_file_size} bytes."
                )

            # Fast synchronous file save
            with open(dest_path, "wb") as buffer:
                buffer.write(content)

            logger.info(f"[5] Saved file {i}/{num_files}: {dest_path} ({file_size} bytes)")

            uploaded_files.append({
                "filename": filename,
                "filepath": str(dest_path.resolve()),
                "url": f"/static/uploads/{filename}"
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[10] Returning HTTP response 200 OK for {num_files} images in {elapsed_ms:.2f}ms")
        return uploaded_files

    except HTTPException as http_exc:
        logger.warning(f"[Upload Exception] HTTP {http_exc.status_code}: {http_exc.detail}")
        raise http_exc
    except Exception as exc:
        logger.error(f"[Upload Error] Unhandled exception in /upload-images: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Server upload error: {str(exc)}"}
        )


@router.post("/run-inspection")
async def run_inspection(
    request: RunInspectionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Initiates a new multi-image inspection campaign and schedules background execution.
    """
    try:
        if not request.image_paths:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image paths provided for inspection.")

        for p in request.image_paths:
            if not Path(p).exists():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image path does not exist: {p}")

        record = InspectionRecord(
            status="queued",
            progress=0.0,
            images_json=json.dumps([Path(p).name for p in request.image_paths])
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        pipeline = CampaignInspectionPipeline()
        background_tasks.add_task(
            pipeline.run_campaign,
            db=db,
            record_id=record.id,
            image_paths=request.image_paths,
            pixel_to_mm=request.pixel_to_mm
        )

        return {
            "message": "Inspection campaign initiated successfully.",
            "record_id": record.id,
            "inspection_id": record.id,
            "id": record.id,
            "status": "queued"
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error in /run-inspection: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Failed to initiate inspection: {str(exc)}"}
        )


@router.get("/{record_id}")
async def get_inspection_status(record_id: int, db: Session = Depends(get_db)):
    """
    Retrieves execution status, progress, and computed metrics for an inspection campaign.
    """
    record = db.query(InspectionRecord).filter(InspectionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Inspection record #{record_id} not found.")

    res = {
        "id": record.id,
        "record_id": record.id,
        "inspection_id": record.id,
        "status": record.status,
        "progress": record.progress,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "pdf_report_path": record.pdf_report_path,
        "health_score": record.health_score,
        "failure_probability": record.failure_probability,
        "rul_days": record.rul_days,
        "risk_category": record.risk_category,
        "maintenance_priority": record.maintenance_priority,
        "summary_report": record.summary_report,
    }

    if record.aggregate_results_json:
        try:
            res["aggregate_results"] = json.loads(record.aggregate_results_json)
        except Exception:
            pass

    return res


@router.get("/report/{record_id}")
async def download_inspection_report(record_id: int, db: Session = Depends(get_db)):
    """
    Downloads compiled PDF inspection report.
    """
    record = db.query(InspectionRecord).filter(InspectionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Inspection record #{record_id} not found.")

    pdf_path = record.pdf_report_path
    if not pdf_path:
        reports_dir = Path("backend/static/reports")
        candidate = reports_dir / f"inspection_report_{record_id}.pdf"
        if candidate.exists():
            pdf_path = str(candidate)

    if not pdf_path or not Path(pdf_path.lstrip("/")).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"PDF report for record #{record_id} not yet generated.")

    clean_path = pdf_path.lstrip("/") if pdf_path.startswith("/") else pdf_path
    return FileResponse(
        path=clean_path,
        media_type="application/pdf",
        filename=f"inspection_report_{record_id}.pdf"
    )
