"""
BridgeGuardian AI — Campaign Inspection & Analytics REST Endpoints (/campaigns)
Manages drone inspection campaign uploads, SAHI batch inference, spatial heatmaps, and timeline analytics.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.ml.gis_heatmap import GISHeatmapEngine
from backend.app.services.health_aggregator import StructuralHealthAggregator
from backend.core.database import get_db
from backend.core.models import (
    ComponentHealthRecord,
    DefectDetectionRecord,
    DroneImageRecord,
    InspectionCampaignRecord,
)

router = APIRouter()


class CampaignCreateRequest(BaseModel):
    name: str
    bridge_id: str
    total_images: int = 1


class CampaignResponse(BaseModel):
    campaign_id: str
    name: str
    bridge_id: str
    status: str
    total_images: int
    processed_images: int
    created_at: str


@router.post(
    "/campaigns/upload",
    response_model=CampaignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new drone inspection campaign",
    tags=["Inspection Campaigns"],
)
async def create_campaign(
    payload: CampaignCreateRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Creates a new drone inspection campaign record."""
    cid = f"cmp_{uuid.uuid4().hex[:10]}"
    record = InspectionCampaignRecord(
        campaign_id=cid,
        name=payload.name,
        bridge_id=payload.bridge_id,
        status="IN_PROGRESS",
        total_images=payload.total_images,
        processed_images=0,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return CampaignResponse(
        campaign_id=record.campaign_id,
        name=record.name,
        bridge_id=record.bridge_id,
        status=record.status,
        total_images=record.total_images,
        processed_images=record.processed_images,
        created_at=record.created_at.isoformat(),
    )


@router.get(
    "/campaigns/{campaign_id}/heatmap",
    summary="Get spatial defect density heatmap for Leaflet/GIS map",
    tags=["Inspection Campaigns"],
)
async def get_campaign_heatmap(
    campaign_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Generates 2D Gaussian KDE spatial defect heatmap for campaign."""
    detections = (
        db.query(DefectDetectionRecord)
        .filter(DefectDetectionRecord.campaign_id == campaign_id)
        .all()
    )

    formatted_detections = []
    import json
    for d in detections:
        try:
            bbox = json.loads(d.bbox_json)
        except Exception:
            bbox = [10, 10, 100, 100]

        formatted_detections.append({
            "bbox": bbox,
            "severity_level": d.severity_level,
            "confidence": d.confidence,
        })

    engine = GISHeatmapEngine()
    heatmap_data = engine.generate_density_heatmap(formatted_detections)

    return {
        "campaign_id": campaign_id,
        "total_defects": len(detections),
        "heatmap": heatmap_data,
    }


@router.get(
    "/campaigns/{campaign_id}/timeline",
    summary="Get component health timeline analytics",
    tags=["Inspection Campaigns"],
)
async def get_campaign_timeline(
    campaign_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Returns component health evaluation summary across all structural components."""
    health_records = (
        db.query(ComponentHealthRecord)
        .filter(ComponentHealthRecord.campaign_id == campaign_id)
        .all()
    )

    timeline_data = [
        {
            "component_code": r.component_code,
            "health_score": r.health_score,
            "status_category": r.status_category,
            "worst_defect_class": r.worst_defect_class,
            "evaluated_at": r.evaluated_at.isoformat(),
        }
        for r in health_records
    ]

    return {
        "campaign_id": campaign_id,
        "component_health_timeline": timeline_data,
    }
