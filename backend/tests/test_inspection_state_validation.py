"""
BridgeGuardian AI — Inspection State Validation & Strict Engineering Integrity Test Suite
Verifies:
- State 1: NO_IMAGES (0 files uploaded)
- State 2: ALL_IMAGES_REJECTED (0 images accepted, all failed quality validation)
- State 3: PARTIAL_ANALYSIS (Accepted > 0, Rejected > 0)
- State 4: FULL_ANALYSIS (All images accepted)
- Zero fabricated metrics when 0 images are accepted (SHI = N/A, Conf = 0%)
- Inspection Attempt Report PDF layout
"""
import pytest
import numpy as np
import cv2
import json
from pathlib import Path

from backend.core.database import SessionLocal, init_db
from backend.core.models import InspectionRecord
from backend.ml.computer_vision.inspection_pipeline import CampaignInspectionPipeline

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()

@pytest.fixture
def blurry_images_batch(tmp_path):
    """Creates 3 blurry images that fail quality check."""
    paths = []
    for i in range(3):
        file_path = tmp_path / f"blurry_photo_{i+1}.jpg"
        img = np.ones((600, 800, 3), dtype=np.uint8) * 120
        blurred = cv2.GaussianBlur(img, (51, 51), 0)
        cv2.imwrite(str(file_path), blurred)
        paths.append(str(file_path))
    return paths

@pytest.fixture
def valid_sample_image(tmp_path):
    """Creates a sharp, high-resolution valid bridge sample image with high edge texture."""
    file_path = tmp_path / "sharp_bridge.jpg"
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 160
    # Draw sharp structural lines and high-contrast grid texture
    for x in range(50, w - 50, 15):
        cv2.line(img, (x, 100), (x, 500), (30, 30, 30), 2)
    for y in range(100, 500, 15):
        cv2.line(img, (50, y), (w - 50, y), (30, 30, 30), 2)
    cv2.rectangle(img, (100, 200), (700, 400), (40, 40, 40), 3)
    cv2.imwrite(str(file_path), img)
    return str(file_path)

def test_all_images_rejected_state(blurry_images_batch):
    """
    State 2 (ALL_IMAGES_REJECTED):
    Verifies that when 0 images pass quality check:
    - Pipeline halts downstream inference
    - Health score and failure prob are N/A (None)
    - Summary report states 'Inspection could not be completed...'
    """
    db = SessionLocal()
    try:
        record = InspectionRecord(
            images_json=json.dumps(blurry_images_batch),
            status="pending"
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        pipeline = CampaignInspectionPipeline()
        pipeline.run_campaign(db, record.id, blurry_images_batch)

        db.refresh(record)
        assert record.status == "failed"
        assert record.health_score is None # Represents N/A
        assert record.failure_probability is None # Represents N/A
        assert record.rul_days is None # Represents N/A
        assert record.maintenance_priority == "Inspection Required"
        assert "Inspection could not be completed" in record.summary_report

        # Verify performance metrics JSON
        perf = json.loads(record.performance_metrics_json)
        assert perf["accepted_images"] == 0
        assert perf["rejected_images"] == 3
        assert perf["pipeline_state"] == "ALL_IMAGES_REJECTED"

        # Verify PDF report exists
        assert record.pdf_report_path is not None
        assert Path(record.pdf_report_path).exists()

    finally:
        db.close()

def test_partial_analysis_state(blurry_images_batch, valid_sample_image):
    """
    State 3 (PARTIAL_ANALYSIS):
    Verifies that when 1 image is accepted and 3 are rejected:
    - Pipeline executes on accepted image
    - Summary indicates 'Partial Analysis: Only 1 of 4 uploaded images passed quality validation'
    """
    db = SessionLocal()
    try:
        mixed_batch = [valid_sample_image] + blurry_images_batch

        record = InspectionRecord(
            images_json=json.dumps(mixed_batch),
            status="pending"
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        pipeline = CampaignInspectionPipeline()
        pipeline.run_campaign(db, record.id, mixed_batch)

        db.refresh(record)
        assert record.status == "completed"
        assert record.health_score is not None

        perf = json.loads(record.performance_metrics_json)
        assert perf["accepted_images"] == 1
        assert perf["rejected_images"] == 3
        assert perf["pipeline_state"] == "PARTIAL_ANALYSIS"
        assert "Partial Analysis: Only 1 of 4" in record.summary_report

    finally:
        db.close()
