"""
BridgeGuardian AI — End-to-End Campaign Execution Live Verification
Tests image upload -> campaign trigger -> real-time progress polling -> completion -> PDF generation.
Uses Python Standard Library urllib.request to avoid external dependencies.
"""
from __future__ import annotations

import io
import time
import json
import urllib.request
import urllib.parse
import uuid
from PIL import Image
import numpy as np


def post_multipart(url, files, fields=None, headers=None):
    boundary = uuid.uuid4().hex.encode('utf-8')
    body = bytearray()

    if fields:
        for name, value in fields.items():
            body.extend(b'--' + boundary + b'\r\n')
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode('utf-8'))
            body.extend(str(value).encode('utf-8') + b'\r\n')

    for name, filename, content_type, data in files:
        body.extend(b'--' + boundary + b'\r\n')
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode('utf-8'))
        body.extend(f'Content-Type: {content_type}\r\n\r\n'.encode('utf-8'))
        body.extend(data)
        body.extend(b'\r\n')

    body.extend(b'--' + boundary + b'--\r\n')

    req = urllib.request.Request(url, data=bytes(body), method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary.decode("utf-8")}')
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))


def http_get_json(url, headers=None):
    req = urllib.request.Request(url, method='GET')
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))


def http_post_json(url, payload, headers=None):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))


def verify_full_campaign():
    print("=" * 70)
    print("BridgeGuardian AI — Full Live Drone Campaign End-to-End Verification")
    print("=" * 70)

    base_url = "http://127.0.0.1:8000"
    headers = {"Origin": "http://127.0.0.1:3000"}

    # 1. Generate 8 test drone photos
    files = []
    print("Generating 8 test drone photos...")
    for i in range(8):
        arr = np.random.randint(0, 256, (800, 1000, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        files.append(('files', f"live_drone_{i:02d}.jpg", 'image/jpeg', buf.getvalue()))

    # 2. Upload images
    t0 = time.time()
    print("Uploading 8 drone photos to /api/v1/inspection/upload-images...")
    status, res_upload = post_multipart(f"{base_url}/api/v1/inspection/upload-images", files=files, headers=headers)
    t_upload = (time.time() - t0) * 1000

    print(f"[OK] Upload Status: {status} (took {t_upload:.1f}ms)")
    image_paths = [item["filepath"] for item in res_upload if isinstance(item, dict) and "filepath" in item]

    print(f"[OK] Saved {len(image_paths)} image files for inspection campaign")

    # 3. Trigger Campaign Execution
    print(f"Triggering Campaign Analysis for uploaded images...")
    t0 = time.time()
    status, res_run = http_post_json(
        f"{base_url}/api/v1/inspection/run-inspection",
        {"image_paths": image_paths, "pixel_to_mm": 0.5},
        headers=headers
    )
    record_id = res_run.get("inspection_id") or res_run.get("id")
    print(f"[OK] Run Inspection Trigger Status: {status} | Campaign Record ID #{record_id}")

    # 4. Poll Campaign Progress
    print("Polling Campaign Progress via /api/v1/inspection/{record_id}...")
    for poll in range(60):
        time.sleep(0.5)
        status, rec = http_get_json(f"{base_url}/api/v1/inspection/{record_id}", headers=headers)
        if status == 200:
            st = rec.get("status")
            progress = int((rec.get("progress") or 0) * 100)
            print(f"  Poll #{poll+1:02d} | Status: {st:<10} | Progress: {progress}%")
            if st == "completed":
                t_total = time.time() - t0
                print("=" * 70)
                print(f"SUCCESS: Campaign Inspection #{record_id} COMPLETED in {t_total:.2f}s!")
                print(f"  Health Score:          {rec.get('health_score')}%")
                print(f"  Failure Probability:   {rec.get('failure_probability')}%")
                print(f"  Risk Category:         {rec.get('risk_category')}")
                print(f"  Maintenance Action:    {rec.get('maintenance_priority')}")
                print(f"  PDF Report Path:       {rec.get('pdf_report_path')}")
                print("=" * 70)
                return True
            elif st == "failed":
                print("ERROR: Campaign failed on server.")
                return False

    print("TIMED OUT waiting for campaign completion.")
    return False


if __name__ == "__main__":
    verify_full_campaign()
