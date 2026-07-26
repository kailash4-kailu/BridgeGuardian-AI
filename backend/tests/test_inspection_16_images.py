"""
BridgeGuardian AI — Multi-Image Upload Integration Test
Verifies that batch uploading 16 images to /api/v1/inspection/upload-images succeeds with HTTP 200 OK
and does not trigger HTTP 413 Payload Too Large error.
"""
import io
import pytest
from PIL import Image


def create_dummy_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (200, 200), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_upload_16_images_batch_success(client):
    """Verify batch upload of 16 drone images to /api/v1/inspection/upload-images."""
    dummy_bytes = create_dummy_jpeg_bytes()
    
    files = [
        ("files", (f"drone_img_{i:02d}.jpg", dummy_bytes, "image/jpeg"))
        for i in range(16)
    ]
    
    response = client.post("/api/v1/inspection/upload-images", files=files)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 16
    for item in data:
        assert "filename" in item
        assert "filepath" in item
