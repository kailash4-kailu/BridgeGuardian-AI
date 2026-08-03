"""
BridgeGuardian AI — SQLAlchemy ORM Models
Defines database tables for predictions, model metadata, and training runs.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class PredictionRecord(Base):
    """Stores each prediction request and its results."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Input snapshot (JSON serialised)
    input_data: Mapped[str] = mapped_column(Text, nullable=False)

    # Prediction outputs
    health_score: Mapped[float] = mapped_column(Float, nullable=True)
    failure_probability: Mapped[float] = mapped_column(Float, nullable=True)
    rul_days: Mapped[float] = mapped_column(Float, nullable=True)
    risk_category: Mapped[str] = mapped_column(String(50), nullable=True)
    maintenance_priority: Mapped[str] = mapped_column(String(50), nullable=True)
    maintenance_recommendation: Mapped[str] = mapped_column(Text, nullable=True)
    prediction_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    repair_cost_estimate: Mapped[float] = mapped_column(Float, nullable=True)

    # Model metadata & Workflow tagging
    model_version: Mapped[str] = mapped_column(String(50), nullable=True)
    analysis_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="Structural Health")
    campaign_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    summary_report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="completed")

    __table_args__ = (
        Index("ix_prediction_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<PredictionRecord id={self.id} "
            f"health={self.health_score:.3f} "
            f"risk={self.risk_category}>"
        )


class ModelMetadata(Base):
    """Stores metadata for each trained model version."""

    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    target: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    features_count: Mapped[int] = mapped_column(Integer, nullable=True)
    training_rows: Mapped[int] = mapped_column(Integer, nullable=True)

    # Metrics (JSON)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=True)
    hyperparams_json: Mapped[str] = mapped_column(Text, nullable=True)

    is_active: Mapped[int] = mapped_column(Integer, default=0)  # 1 = current best

    def __repr__(self) -> str:
        return f"<ModelMetadata {self.algorithm} v{self.model_version} target={self.target}>"


class TrainingRun(Base):
    """Tracks each training run for audit purposes."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="running")
    log: Mapped[str] = mapped_column(Text, nullable=True)
    best_model: Mapped[str] = mapped_column(String(100), nullable=True)
    dataset_rows: Mapped[int] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} status={self.status}>"


class InspectionRecord(Base):
    """Stores a multi-image drone inspection campaign and its results."""

    __tablename__ = "inspection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    status: Mapped[str] = mapped_column(String(50), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    # Ingested images (JSON list)
    images_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Output results (JSON serialised)
    image_results_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    aggregate_results_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Aggregated health indicators
    health_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    failure_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rul_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    maintenance_priority: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Actionable maintenance fields
    maintenance_action: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    repair_window_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    inspection_interval_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Auditing and logging metadata
    model_metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    performance_metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # AI text summary
    summary_report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Output file
    pdf_report_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    def __repr__(self) -> str:
        return f"<InspectionRecord id={self.id} status={self.status} progress={self.progress}>"


class InspectionDefect(Base):
    """Stores granular, localized defect items extracted from drone images."""

    __tablename__ = "inspection_defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    defect_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    inspection_id: Mapped[int] = mapped_column(Integer, nullable=False)
    image_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Structural component association
    component: Mapped[str] = mapped_column(String(50), nullable=False)
    defect_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Quality metrics
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(50), nullable=False)

    # Spatial coordinates relative to original frame: [x, y, w, h] (JSON list)
    bbox_json: Mapped[str] = mapped_column(String(100), nullable=False)

    # Raw OpenCV geometric measurements (JSON serialized)
    measurements_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Lifecycle tracking flag
    status_flag: Mapped[str] = mapped_column(String(50), default="New")

    def __repr__(self) -> str:
        return f"<InspectionDefect {self.defect_id} type={self.defect_type} component={self.component}>"


class User(Base):
    """User account model for authentication and Role-Based Access Control (RBAC)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(50), default="field_inspector", nullable=False)
    is_active: Mapped[int] = mapped_column(Integer, default=1)

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"


class UserSession(Base):
    """Active user session tracking and token revocation."""

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    token_jti: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    is_revoked: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<UserSession user_id={self.user_id} jti={self.token_jti} revoked={self.is_revoked}>"


class InspectionCampaignRecord(Base):
    """Industrial Drone Inspection Campaign tracking table."""

    __tablename__ = "inspection_campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    campaign_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    bridge_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="IN_PROGRESS")
    total_images: Mapped[int] = mapped_column(Integer, default=0)
    processed_images: Mapped[int] = mapped_column(Integer, default=0)


class DroneImageRecord(Base):
    """Drone Aerial Image Metadata and Quality Metrics table."""

    __tablename__ = "drone_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    campaign_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    laplacian_variance: Mapped[float] = mapped_column(Float, nullable=False)
    is_blurred: Mapped[int] = mapped_column(Integer, default=0)


class DefectDetectionRecord(Base):
    """Defect Segmentation and Bounding Box Detection Record table."""

    __tablename__ = "defect_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    campaign_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    image_id: Mapped[int] = mapped_column(Integer, nullable=False)
    component_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    defect_class: Mapped[str] = mapped_column(String(100), nullable=False)
    severity_level: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    polygon_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ComponentHealthRecord(Base):
    """Bridge Component Structural Health Index Summary table."""

    __tablename__ = "component_health_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    campaign_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    component_code: Mapped[str] = mapped_column(String(50), nullable=False)
    health_score: Mapped[float] = mapped_column(Float, nullable=False)
    status_category: Mapped[str] = mapped_column(String(50), nullable=False)
    worst_defect_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class ModelRegistryEntry(Base):
    """Production Model Registry tracking version, accuracy metrics, dataset version, and deployment status."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    accuracy_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rmse_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dataset_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1.0")
    training_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    deployment_status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")  # active, archived, rolled_back
    is_active: Mapped[int] = mapped_column(Integer, default=1)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<ModelRegistryEntry version={self.model_version} status={self.deployment_status}>"



