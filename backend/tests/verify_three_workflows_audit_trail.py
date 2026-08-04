import sys
import time
import json
from pathlib import Path

# Add workspace root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from backend.main import app
from backend.core.database import SessionLocal, init_db
from backend.core.models import PredictionRecord, InspectionRecord

def run_verification():
    print("=== STARTING END-TO-END AUDIT TRAIL VERIFICATION ===")
    init_db()
    
    # 1. Clean slate for test records
    db = SessionLocal()
    try:
        db.query(PredictionRecord).delete()
        db.query(InspectionRecord).delete()
        db.commit()
        print("Database cleared for clean verification run.")
    finally:
        db.close()

    client = TestClient(app)

    # Ensure inference pipeline is loaded
    from backend.main import inference_pipeline
    inference_pipeline.load()

    # -------------------------------------------------------------
    # 1. Run Structural Health Analysis
    # -------------------------------------------------------------
    print("\n[Step 1] Running Structural Health Analysis...")
    telemetry_payload = {
        "Strain_microstrain": 850.0,
        "Deflection_mm": 18.5,
        "Vibration_ms2": 1.5,
        "Tilt_deg": 0.8,
        "Displacement_mm": 25.0,
        "Crack_Propagation_mm": 0.02,
        "Corrosion_Level_percent": 0.2,
        "Bridge_Mood_Meter": "Healthy"
    }
    resp_predict = client.post("/api/v1/predict", json=telemetry_payload)
    assert resp_predict.status_code == 200, f"Predict failed: {resp_predict.text}"
    predict_data = resp_predict.json()
    print(f"-> Structural Health Success: ID={predict_data.get('prediction_id')}, SHI={predict_data.get('health_score')}")

    # -------------------------------------------------------------
    # 2. Run Single Image Inspection
    # -------------------------------------------------------------
    print("\n[Step 2] Running Single Image Analysis...")
    # Create temporary dummy image
    from PIL import Image
    import io
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    upload_resp = client.post(
        "/api/v1/vision/upload-image",
        files={"files": ("test_single.jpg", buf, "image/jpeg")}
    )
    assert upload_resp.status_code == 200, f"Single image upload failed: {upload_resp.text}"
    upload_data = upload_resp.json()
    image_id = upload_data[0]["image_id"]

    vision_resp = client.post(
        "/api/v1/vision/vision-predict",
        json={"image_id": image_id, "pixel_to_mm": 0.5}
    )
    assert vision_resp.status_code == 200, f"Vision predict failed: {vision_resp.text}"
    vision_data = vision_resp.json()
    print(f"-> Single Image Success: Prediction ID={vision_data.get('prediction_id')}, SHI={vision_data.get('predictions', {}).get('health_score')}")

    # -------------------------------------------------------------
    # 3. Run Drone Campaign Inspection
    # -------------------------------------------------------------
    print("\n[Step 3] Running Drone Campaign Analysis...")
    dummy_files = []
    for i in range(5):
        b = io.BytesIO()
        img.save(b, format="JPEG")
        b.seek(0)
        dummy_files.append(("files", (f"drone_test_{i}.jpg", b, "image/jpeg")))

    campaign_upload_resp = client.post("/api/v1/inspection/upload-images", files=dummy_files)
    assert campaign_upload_resp.status_code == 200, f"Campaign upload failed: {campaign_upload_resp.text}"
    campaign_files = campaign_upload_resp.json()
    paths = [f["filepath"] for f in campaign_files]

    run_resp = client.post("/api/v1/inspection/run-inspection", json={"image_paths": paths, "pixel_to_mm": 0.5})
    assert run_resp.status_code == 200, f"Run campaign failed: {run_resp.text}"
    inspection_id = run_resp.json()["inspection_id"]

    # Synchronously execute campaign pipeline to simulate background execution
    from backend.ml.computer_vision.inspection_pipeline import CampaignInspectionPipeline
    db_bg = SessionLocal()
    try:
        pipeline = CampaignInspectionPipeline()
        pipeline.run_campaign(db=db_bg, inspection_id=inspection_id, image_paths=paths, pixel_to_mm=0.5)
    finally:
        db_bg.close()

    print(f"-> Drone Campaign Success: Campaign ID={inspection_id}")

    # -------------------------------------------------------------
    # 4. Verify History API (/history)
    # -------------------------------------------------------------
    print("\n[Step 4] Querying /history endpoint...")
    history_resp = client.get("/api/v1/history?limit=50&offset=0")
    assert history_resp.status_code == 200, f"History request failed: {history_resp.text}"
    history_data = history_resp.json()
    items = history_data["items"]
    total = history_data["total"]

    print(f"\nTotal History Records Returned: {total}")
    for item in items:
        print(f"  - Record ID={item['id']} | Type='{item['analysis_type']}' | SHI={item['health_score']} | Status='{item['status']}'")

    assert total == 3, f"Expected exactly 3 history records, found {total}"

    # Verify each workflow type is represented
    types_found = [item["analysis_type"] for item in items]
    print(f"\nWorkflow types found in audit trail: {types_found}")
    
    assert "structural_health" in types_found, "Missing 'structural_health' in history"
    assert "single_image" in types_found, "Missing 'single_image' in history"
    assert "drone_campaign" in types_found, "Missing 'drone_campaign' in history"

    print("\n=== VERIFICATION SUCCESSFUL: EXACTLY 3 RECORDS (1 OF EACH WORKFLOW TYPE) PRESENT IN AUDIT TRAIL ===")

if __name__ == "__main__":
    run_verification()
