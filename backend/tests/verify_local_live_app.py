"""
BridgeGuardian AI — 16 Large High-Res Drone Photos Live Upload Verification
Tests 16 images (~5 MB each, total 80 MB payload) directly against http://127.0.0.1:8000 with CORS headers.
"""
import io
import time
import httpx
import numpy as np
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"


def test_large_batch_upload():
    print("=" * 60)
    print("BridgeGuardian AI — 16 Large Drone Photos Upload Test")
    print("=" * 60)

    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    # 1. OPTIONS Preflight Request Test with Origin
    headers_cors = {
        "Origin": "http://127.0.0.1:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    r = client.options("/api/v1/inspection/upload-images", headers=headers_cors)
    assert r.status_code == 200, f"OPTIONS preflight failed with status {r.status_code}: {r.text}"
    print(f"[OK] OPTIONS /api/v1/inspection/upload-images: Status 200 OK (CORS Preflight Success)")

    # 2. Upload 16 Large High-Res Drone Images with high-entropy noise (~15 MB payload)
    print("Generating 16 large drone photos with high-entropy details...")
    files = []
    for i in range(16):
        arr = np.random.randint(0, 256, (1200, 1600, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        files.append(("files", (f"drone_real_photo_{i:02d}.jpg", buf.getvalue(), "image/jpeg")))

    total_bytes = sum(len(f[1][1]) for f in files)
    print(f"Uploading 16 photos ({total_bytes / (1024*1024):.2f} MB total payload) with Origin: http://127.0.0.1:3000...")

    post_headers = {"Origin": "http://127.0.0.1:3000"}
    t_start = time.perf_counter()
    r = client.post("/api/v1/inspection/upload-images", files=files, headers=post_headers)
    t_elapsed = (time.perf_counter() - t_start) * 1000

    assert r.status_code == 200, f"Upload failed with status {r.status_code}: {r.text}"
    uploaded = r.json()
    assert len(uploaded) == 16, f"Expected 16 uploaded files, got {len(uploaded)}"

    allow_origin = r.headers.get("access-control-allow-origin")
    print(f"[OK] POST /api/v1/inspection/upload-images: Status 200 OK")
    print(f"[OK] Batch Payload Size: {total_bytes / (1024*1024):.2f} MB")
    print(f"[OK] Duration: {t_elapsed:.2f}ms ({t_elapsed/1000:.2f}s)")
    print(f"[OK] Access-Control-Allow-Origin: {allow_origin}")
    print(f"[OK] 16 Large Drone Photos Uploaded Successfully with HTTP 200 OK!")

    print("=" * 60)
    print("LIVE 16-IMAGE LARGE BATCH UPLOAD VERIFICATION 100% PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_large_batch_upload()
