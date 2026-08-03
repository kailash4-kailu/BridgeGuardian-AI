"""
BridgeGuardian AI — 20 Photos Drone Inspection Campaign Test
Generates 20 realistic bridge inspection photos, executes the complete 6-stage
AI vision & ML analytics pipeline, generates a PDF report, and outputs results.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.database import SessionLocal, init_db
from backend.core.models import InspectionRecord, InspectionDefect
from backend.ml.computer_vision.inspection_pipeline import CampaignInspectionPipeline


def create_realistic_bridge_photo(filename: str, photo_index: int, output_dir: Path) -> str:
    """Generate a high-res (1600x1200) realistic bridge component inspection photo."""
    width, height = 1600, 1200
    
    # Base concrete / steel texture colors
    components = [
        ("Concrete Deck Span", (170, 172, 175), (140, 142, 145)),
        ("Abutment Support Wall", (160, 162, 165), (130, 132, 135)),
        ("Steel Girder Flange", (120, 125, 130), (80, 85, 90)),
        ("Pier Column Base", (180, 182, 184), (150, 152, 155)),
        ("Expansion Joint Assembly", (110, 115, 120), (70, 75, 80)),
        ("Prestressed Cable Member", (130, 135, 140), (90, 95, 100)),
        ("Bearing Pad Seat", (165, 168, 170), (135, 138, 140)),
        ("Concrete Parapet Barrier", (175, 177, 180), (145, 147, 150)),
    ]
    
    comp_name, base_color, dark_color = components[photo_index % len(components)]
    
    # Generate background with subtle noise texture
    np.random.seed(42 + photo_index)
    base_arr = np.ones((height, width, 3), dtype=np.uint8) * np.array(base_color, dtype=np.uint8)
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    img_arr = np.clip(base_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    
    # Draw structural grid lines / joints
    draw.line([(0, height // 3), (width, height // 3)], fill=dark_color, width=4)
    draw.line([(0, 2 * height // 3), (width, 2 * height // 3)], fill=dark_color, width=4)
    draw.line([(width // 2, 0), (width // 2, height)], fill=dark_color, width=3)
    
    # Inject visual defects based on photo index (Cracks, Corrosion, Spalling, Efflorescence)
    if photo_index % 2 == 0:
        # Draw a jagged Crack
        points = []
        cx, cy = np.random.randint(300, 1300), np.random.randint(200, 900)
        for step in range(12):
            cx += np.random.randint(-35, 45)
            cy += np.random.randint(15, 60)
            points.append((cx, cy))
        if len(points) > 1:
            draw.line(points, fill=(30, 30, 35), width=np.random.randint(4, 9))
            
    if photo_index % 3 == 0:
        # Draw a Rust / Corrosion patch
        rx = np.random.randint(400, 1200)
        ry = np.random.randint(300, 800)
        rw, rh = np.random.randint(120, 300), np.random.randint(80, 200)
        draw.ellipse([rx, ry, rx + rw, ry + rh], fill=(165, 75, 25))
        
    if photo_index % 5 == 0:
        # Draw Concrete Spalling defect
        sx = np.random.randint(200, 1000)
        sy = np.random.randint(400, 900)
        sw, sh = np.random.randint(180, 350), np.random.randint(120, 250)
        draw.polygon([(sx, sy), (sx + sw, sy + 30), (sx + sw - 20, sy + sh), (sx + 20, sy + sh - 10)], fill=(90, 88, 85))

    # Add HUD overlay annotation
    draw.rectangle([20, 20, 480, 70], fill=(20, 20, 30))
    draw.text((30, 30), f"DRONE CAM #{photo_index+1:02d} | {comp_name.upper()}", fill=(240, 240, 250))
    
    img_path = output_dir / filename
    img.save(img_path, format="JPEG", quality=92)
    return str(img_path.resolve())


def run_20_photos_inspection():
    print("=" * 75)
    print("BridgeGuardian AI — 20 Photos Drone Inspection Campaign Test")
    print("=" * 75)
    
    init_db()
    db = SessionLocal()
    
    # 1. Generate 20 high-res drone inspection photos
    uploads_dir = Path("backend/static/uploads/campaign_20_photos")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    print("\n[Step 1/4] Generating 20 realistic bridge drone inspection photos...")
    for i in range(20):
        fname = f"bridge_drone_img_{i+1:02d}.jpg"
        p = create_realistic_bridge_photo(fname, i, uploads_dir)
        image_paths.append(p)
        print(f"  Photo #{i+1:02d}: {fname} generated.")
        
    # 2. Register Campaign Record in DB
    print("\n[Step 2/4] Registering inspection campaign record in database...")
    record = InspectionRecord(
        status="queued",
        progress=0.0,
        images_json=json.dumps([Path(p).name for p in image_paths])
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    print(f"  Campaign Record created: ID #{record.id}")

    # 3. Execute 6-Stage Campaign AI Inspection Pipeline
    print(f"\n[Step 3/4] Executing 6-Stage AI Vision & ML Pipeline on 20 photos...")
    t0 = time.time()
    pipeline = CampaignInspectionPipeline(models_dir="models")
    pipeline.run_campaign(db, record.id, image_paths, pixel_to_mm=0.5)
    elapsed = round(time.time() - t0, 2)
    print(f"  AI Inspection Pipeline finished in {elapsed}s.")
    
    # 4. Fetch Campaign Results from DB
    db.refresh(record)
    print("\n[Step 4/4] Campaign Inspection Results Summary:")
    print("-" * 75)
    print(f"  Campaign ID: #{record.id}")
    print(f"  Status: {record.status}")
    print(f"  Health Score: {record.health_score:.2f} / 100")
    print(f"  Failure Probability: {record.failure_probability:.2f}%")
    print(f"  Estimated RUL: {record.rul_days} days")
    print(f"  Risk Category: {record.risk_category}")
    print(f"  Maintenance Priority: {record.maintenance_priority}")
    print(f"  Maintenance Action: {record.maintenance_action}")
    print(f"  Repair Window: {record.repair_window_days} days")
    print(f"  Inspection Interval: {record.inspection_interval_days} days")
    print(f"  PDF Report Path: {record.pdf_report_path}")
    print("-" * 75)

    # Copy PDF report and JSON results to artifacts dir for easy download
    artifacts_dir = Path(r"C:\Users\Acer\.gemini\antigravity-ide\brain\67cf5d1b-35dd-4761-9447-3b18af76a6be")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    copied_pdf = None
    if record.pdf_report_path and os.path.exists(record.pdf_report_path):
        target_pdf = artifacts_dir / f"bridge_20_photos_inspection_report_#{record.id}.pdf"
        shutil.copy(record.pdf_report_path, target_pdf)
        copied_pdf = str(target_pdf)
        print(f"  PDF Report copied to artifact download: {copied_pdf}")
        
    summary_data = {
        "campaign_id": record.id,
        "total_images_processed": 20,
        "processing_time_seconds": elapsed,
        "health_score": record.health_score,
        "failure_probability": record.failure_probability,
        "rul_days": record.rul_days,
        "risk_category": record.risk_category,
        "maintenance_priority": record.maintenance_priority,
        "maintenance_action": record.maintenance_action,
        "repair_window_days": record.repair_window_days,
        "pdf_report_path": copied_pdf or record.pdf_report_path,
        "summary_report": record.summary_report,
    }
    
    with open(artifacts_dir / "inspection_campaign_results_20_photos.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    print(f"  JSON Results saved to: {artifacts_dir / 'inspection_campaign_results_20_photos.json'}")
    print("=" * 75)
    return summary_data


if __name__ == "__main__":
    run_20_photos_inspection()
