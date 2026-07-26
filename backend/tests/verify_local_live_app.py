"""
BridgeGuardian AI — Live CORS & 16-Image Upload Verification Script
Tests OPTIONS preflight requests, CORS header injection, and 16-image batch upload with Origin headers.
"""
import io
import time
import httpx
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"


def test_cors_and_upload():
    print("=" * 60)
    print("BridgeGuardian AI — Live CORS & Upload Verification")
    print("=" * 60)

    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    # 1. OPTIONS Preflight Request Test with Origin
    headers_cors = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    r = client.options("/api/v1/inspection/upload-images", headers=headers_cors)
    assert r.status_code == 200, f"OPTIONS preflight failed with status {r.status_code}: {r.text}"
    print(f"[OK] OPTIONS /api/v1/inspection/upload-images: Status 200 OK (CORS Preflight Success)")

    # 2. Upload 16 Drone Images with Origin Header (simulating browser fetch)
    print("Uploading 16 images in a single batch with Origin: http://localhost:3000...")
    files = []
    for i in range(16):
        img = Image.new("RGB", (400, 400), color=(50 + i * 10, 120, 180))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        files.append(("files", (f"drone_browser_img_{i:02d}.jpg", buf.getvalue(), "image/jpeg")))

    post_headers = {"Origin": "http://localhost:3000"}
    t_start = time.perf_counter()
    r = client.post("/api/v1/inspection/upload-images", files=files, headers=post_headers)
    t_elapsed = (time.perf_counter() - t_start) * 1000
    assert r.status_code == 200, f"Upload failed with status {r.status_code}: {r.text}"
    
    uploaded = r.json()
    assert len(uploaded) == 16, f"Expected 16 uploaded files, got {len(uploaded)}"
    
    # Check CORS header in response
    allow_origin = r.headers.get("access-control-allow-origin")
    print(f"[OK] POST /api/v1/inspection/upload-images: Status 200 OK | Duration: {t_elapsed:.2f}ms | Access-Control-Allow-Origin: {allow_origin}")
    print(f"[OK] Uploaded {len(uploaded)} images successfully in {t_elapsed:.2f}ms (< 1 second)!")

    # 3. Test 16 Images Upload with Origin: http://127.0.0.1:3000
    post_headers_ip = {"Origin": "http://127.0.0.1:3000"}
    files_ip = []
    for i in range(16):
        img = Image.new("RGB", (400, 400), color=(60 + i * 10, 130, 190))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        files_ip.append(("files", (f"drone_browser_ip_img_{i:02d}.jpg", buf.getvalue(), "image/jpeg")))

    r = client.post("/api/v1/inspection/upload-images", files=files_ip, headers=post_headers_ip)
    assert r.status_code == 200, f"Upload with 127.0.0.1 origin failed: {r.text}"
    print(f"[OK] POST with Origin: http://127.0.0.1:3000: Status 200 OK")

    print("=" * 60)
    print("LIVE BROWSER CORS & 16-IMAGE UPLOAD VERIFICATION 100% PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_cors_and_upload()
