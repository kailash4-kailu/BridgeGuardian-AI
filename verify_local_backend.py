"""
BridgeGuardian AI — Local Verification Script
Validates imports, FastAPI app initialization, ML model loading, image processing,
and database persistence before Vercel deployment.
"""
from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

# Force UTF-8 stdout if supported or use ASCII output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
root_dir = Path("c:/Users/Acer/Desktop/bridge")
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def run_verification() -> int:
    print("=" * 60)
    print("BridgeGuardian AI - Local Backend Verification")
    print("=" * 60)

    # 1. Imports
    print("\n[1/6] Testing Core Package Imports...")
    try:
        import cv2
        import numpy as np
        import pandas as pd
        import shap
        import sklearn
        import fastapi
        import uvicorn
        import sqlalchemy

        print(f"  [OK] cv2 version: {cv2.__version__}")
        print(f"  [OK] numpy version: {np.__version__}")
        print(f"  [OK] pandas version: {pd.__version__}")
        print(f"  [OK] shap version: {shap.__version__}")
        print(f"  [OK] sklearn version: {sklearn.__version__}")
        print(f"  [OK] fastapi version: {fastapi.__version__}")
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        traceback.print_exc()
        return 1

    # 2. FastAPI app import
    print("\n[2/6] Testing FastAPI Entry Point (api.index)...")
    try:
        from api.index import app
        print("  [OK] Successfully imported 'app' from api.index")
        print(f"  [OK] API Title: {app.title}")
    except Exception as e:
        print(f"  [FAIL] FastAPI app import failed: {e}")
        traceback.print_exc()
        return 1

    # 3. ML Model Loading
    print("\n[3/6] Testing ML Model Loading...")
    try:
        from backend.ml.inference import InferencePipeline
        models_path = root_dir / "models"
        pipeline = InferencePipeline(models_dir=str(models_path))
        pipeline.load()
        print(f"  [OK] Inference pipeline models loaded (version: {pipeline._model_version})")
    except Exception as e:
        print(f"  [FAIL] ML Model loading failed: {e}")
        traceback.print_exc()
        return 1

    # 4. Test Image Discovery
    print("\n[4/6] Locating Test Image...")
    img_candidates = [
        root_dir / "test_bridge_sample.jpg",
        root_dir / "test_crack_sample.jpg",
        root_dir / "test_crack_image.jpg",
        root_dir / "temp_test.jpg"
    ]
    test_image_path = None
    for cand in img_candidates:
        if cand.exists():
            test_image_path = cand
            break

    if not test_image_path:
        print("  [FAIL] Required test image not found!")
        print("  Checked paths:")
        for cand in img_candidates:
            print(f"   - {cand}")
        return 1

    print(f"  [OK] Using test image: {test_image_path.name} ({test_image_path.resolve()})")

    # 5 & 6. Database & Inspection Execution
    print("\n[5/6] Initializing Database & Executing Inspection Pipeline...")
    from backend.core.database import SessionLocal, init_db
    from backend.core.models import InspectionRecord
    from backend.ml.computer_vision.inspection_pipeline import CampaignInspectionPipeline

    init_db()
    db = SessionLocal()
    record_id = None

    try:
        record = InspectionRecord(
            status="queued",
            progress=0.0,
            images_json=f'["{test_image_path.name}"]'
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        record_id = record.id
        print(f"  [OK] Created test InspectionRecord ID #{record_id}")

        campaign_pipeline = CampaignInspectionPipeline(models_dir=str(models_path))
        campaign_pipeline.run_campaign(
            db=db,
            inspection_id=record_id,
            image_paths=[str(test_image_path)],
            pixel_to_mm=0.5
        )

        db.refresh(record)

        print("\n[6/6] Verifying Inspection Record Data Output...")
        print(f"  - status: {record.status}")
        print(f"  - health_score: {record.health_score}")
        print(f"  - failure_probability: {record.failure_probability}")
        print(f"  - rul_days: {record.rul_days}")
        print(f"  - risk_category: {record.risk_category}")
        print(f"  - maintenance_action: {record.maintenance_action}")

        assert record.status == "completed", f"Expected status 'completed', got '{record.status}'"
        assert record.health_score is not None, "record.health_score is None"
        assert record.failure_probability is not None, "record.failure_probability is None"
        assert record.rul_days is not None, "record.rul_days is None"
        assert record.risk_category is not None, "record.risk_category is None"

        print("\n" + "=" * 60)
        print("SUCCESS: ALL LOCAL VERIFICATION CHECKS PASSED PERFECTLY!")
        print("=" * 60)
        return 0

    except Exception as exc:
        print(f"\n  [FAIL] Inspection pipeline execution failed: {exc}")
        traceback.print_exc()
        return 1

    finally:
        if record_id:
            try:
                rec_to_delete = db.query(InspectionRecord).filter(InspectionRecord.id == record_id).first()
                if rec_to_delete:
                    db.delete(rec_to_delete)
                    db.commit()
                    print(f"  [OK] Cleanup: Removed temporary test record #{record_id}")
            except Exception as clean_err:
                print(f"  [WARN] Note: Cleanup of test record #{record_id} skipped: {clean_err}")
        db.close()

if __name__ == "__main__":
    exit_code = run_verification()
    sys.exit(exit_code)
