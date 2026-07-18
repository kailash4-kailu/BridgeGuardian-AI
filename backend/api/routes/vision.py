"""
BridgeGuardian AI — Computer Vision API Routes
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.models import PredictionRecord
from backend.api.routes.predict import get_pipeline
from backend.ml.computer_vision.vision_inference import VisionInferencePipeline

logger = logging.getLogger("bridgeguardian.api.vision")
router = APIRouter(prefix="/vision", tags=["Vision Inspection"])

from backend.core.config import get_settings
import re

settings = get_settings()
UPLOAD_DIR = Path(settings.upload_dir)

def secure_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and invalid characters."""
    name = os.path.basename(filename)
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    if not name or name in ('.', '..'):
        name = "uploaded_file"
    return name

# ── Schemas ────────────────────────────────────────────────────────────── #
class VisionPredictRequest(BaseModel):
    image_id: str = Field(description="Image identifier (filename/path from upload)")
    pixel_to_mm: float = Field(default=0.5, description="Conversion ratio mm/pixel")

class UploadResponseItem(BaseModel):
    image_id: str
    filename: str
    size_bytes: int
    url: str

class VisionPredictResponse(BaseModel):
    prediction_id: int
    timestamp: datetime
    features: dict
    predictions: dict
    visualizations: dict
    shap: dict

# ── Endpoints ──────────────────────────────────────────────────────────── #
@router.post(
    "/upload-image",
    response_model=List[UploadResponseItem],
    summary="Upload drone inspection images",
    description="Accepts one or more drone images, saves them to static storage, and returns metadata.",
)
async def upload_image(files: List[UploadFile] = File(...)) -> List[UploadResponseItem]:
    results = []
    for file in files:
        if not file.filename:
            continue
            
        # Secure filename
        safe_name = secure_filename(file.filename)
        ext = Path(safe_name).suffix.lower()
        
        # Check extensions
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
            
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_name = f"{timestamp_str}_{safe_name}"
        save_path = UPLOAD_DIR / unique_name
        
        try:
            content = await file.read()
            # Double check size limits
            if len(content) > settings.max_upload_size:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_upload_size} bytes."
                )
                
            save_path.write_bytes(content)
            
            # Form static URL (relative to mounted static dir)
            static_url = f"/static/uploads/{unique_name}"
            results.append(UploadResponseItem(
                image_id=unique_name,
                filename=file.filename,
                size_bytes=len(content),
                url=static_url
            ))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}")
            
    return results

@router.post(
    "/vision-predict",
    response_model=VisionPredictResponse,
    summary="Run AI inspection on drone image",
    description="Extracts visual defects, feeds them to the tabular ML model, saves results to database.",
)
async def vision_predict(
    payload: VisionPredictRequest,
    db: Session = Depends(get_db),
    pipeline=Depends(get_pipeline),
) -> VisionPredictResponse:
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail="ML models not trained yet. Deploy/train models first.",
        )
        
    image_path = UPLOAD_DIR / payload.image_id
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded image not found")
        
    try:
        # Initialize vision pipeline
        vision_pipeline = VisionInferencePipeline(pixel_to_mm=payload.pixel_to_mm)
        results = vision_pipeline.analyze_image(str(image_path), pipeline)
        
        # Save inspection predictions to DB
        pred = results["predictions"]
        record = PredictionRecord(
            input_data=json.dumps(results["ml_input"]),
            health_score=pred["health_score_raw"],
            failure_probability=pred["failure_probability_raw"],
            rul_days=pred["rul_days"],
            risk_category=pred["risk_category"],
            maintenance_priority=pred["maintenance_priority"],
            maintenance_recommendation=pred["maintenance_recommendation"],
            prediction_confidence=pred["prediction_confidence"],
            model_version=pred["model_version"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        # Return results including visualizations and DB ID
        return VisionPredictResponse(
            prediction_id=record.id,
            timestamp=datetime.utcnow(),
            features=results["features"],
            predictions=pred,
            visualizations=results["visualizations"],
            shap=results["shap"]
        )
        
    except Exception as e:
        logger.error(f"Vision inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get(
    "/generate-report",
    summary="Download PDF inspection report",
    description="Generates a downloadable PDF report for a given inspection run.",
)
async def generate_report(image_id: str, prediction_id: int, db: Session = Depends(get_db)):
    # Retrieve prediction from DB
    record = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found")
        
    # Paths for images
    original_img_path = UPLOAD_DIR / image_id
    processed_dir = Path(settings.processed_dir)
    annotated_img_path = processed_dir / f"{original_img_path.stem}_segmentation.jpg"
    
    if not original_img_path.exists():
        raise HTTPException(status_code=404, detail="Source image not found on disk")
        
    if not annotated_img_path.exists():
        # Fallback to original image if processed image doesn't exist
        annotated_img_path = original_img_path

    # Generate PDF Report
    pdf_dir = Path(settings.reports_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"Inspection_Report_{prediction_id}.pdf"
    pdf_path = pdf_dir / pdf_filename

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                                rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            leading=28,
            textColor=colors.HexColor('#1E293B'),
            spaceAfter=12
        )
        
        h2_style = ParagraphStyle(
            'Heading2_Custom',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#0F172A'),
            spaceBefore=12,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            'Body_Custom',
            parent=styles['BodyText'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155')
        )
        
        bold_body = ParagraphStyle(
            'Bold_Body',
            parent=body_style,
            fontName='Helvetica-Bold'
        )
        
        center_body = ParagraphStyle(
            'Center_Body',
            parent=body_style,
            alignment=1  # 1 = Center (TA_CENTER)
        )

        # Title
        story.append(Paragraph("BridgeGuardian AI — Inspection Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 15))
        
        # 1. Health Score Section (Grid table)
        health_pct = round(record.health_score * 100, 1)
        failure_pct = round(record.failure_probability * 100, 2)
        
        score_color = "#22C55E" # Green
        if record.risk_category in {"Critical", "Poor"}:
            score_color = "#EF4444" # Red
        elif record.risk_category == "Fair":
            score_color = "#EAB308" # Yellow
            
        summary_data = [
            [
                Paragraph("<b>Structural Health Index (SHI)</b>", body_style),
                Paragraph(f"<font color='{score_color}'><b>{health_pct}%</b></font>", title_style)
            ],
            [
                Paragraph("<b>Failure Probability</b>", body_style),
                Paragraph(f"<b>{failure_pct}%</b>", body_style)
            ],
            [
                Paragraph("<b>Risk Category</b>", body_style),
                Paragraph(f"<b>{record.risk_category}</b>", body_style)
            ],
            [
                Paragraph("<b>Maintenance Priority</b>", body_style),
                Paragraph(f"<b>{record.maintenance_priority}</b>", body_style)
            ]
        ]
        
        summary_table = Table(summary_data, colWidths=[200, 300])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 8),
            ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor('#E2E8F0')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        # 2. Recommendations
        story.append(Paragraph("Maintenance Recommendation", h2_style))
        story.append(Paragraph(record.maintenance_recommendation, body_style))
        story.append(Spacer(1, 15))
        
        # 3. Defect Analysis (from database record input_data if stored)
        # Parse inputs
        inputs = json.loads(record.input_data)
        crack_len = inputs.get("Crack_Propagation_mm", 0.0)
        corr_lvl = inputs.get("Corrosion_Level_percent", 0.0) * 100.0
        defect_score = inputs.get("Visual_Analysis_Defect_Score", 0.0) * 100.0
        tilt = inputs.get("Tilt_deg", 0.0)
        anomaly_score = inputs.get("Anomaly_Detection_Score", 0.0)
        
        defect_data = [
            [Paragraph("<b>Defect Description</b>", bold_body), Paragraph("<b>Estimated Value</b>", bold_body)],
            [Paragraph("Estimated Crack Length", body_style), Paragraph(f"{crack_len:.2f} mm", body_style)],
            [Paragraph("Corrosion Area percentage", body_style), Paragraph(f"{corr_lvl:.2f} %", body_style)],
            [Paragraph("Total Damage Area percentage", body_style), Paragraph(f"{defect_score:.2f} %", body_style)],
            [Paragraph("Approximate Bridge Tilt", body_style), Paragraph(f"{tilt:.2f}°", body_style)],
            [Paragraph("Anomaly Score indicator", body_style), Paragraph(f"{anomaly_score:.2f}", body_style)]
        ]
        defect_table = Table(defect_data, colWidths=[250, 250])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#E2E8F0')),
            ('PADDING', (0,0), (-1,-1), 6),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ]))
        story.append(Paragraph("Quantitative Defect Analysis", h2_style))
        story.append(defect_table)
        story.append(Spacer(1, 20))
        
        # 4. Images Section (Side-by-side or stacked)
        # We size them to fit neatly on a Letter page (width ~ 530pt printable)
        img_w, img_h = 240, 160
        img_data = [
            [
                Image(str(original_img_path), width=img_w, height=img_h),
                Image(str(annotated_img_path), width=img_w, height=img_h)
            ],
            [
                Paragraph("<b>Original Drone Image</b>", center_body),
                Paragraph("<b>Annotated Defects Overlay</b>", center_body)
            ]
        ]
        img_table = Table(img_data, colWidths=[260, 260])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(Paragraph("Visual Evidence Map", h2_style))
        story.append(img_table)
        
        doc.build(story)
        
    except Exception as e:
        logger.error(f"Failed to compile PDF report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="Generated PDF file not found")
        
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=pdf_filename
    )
