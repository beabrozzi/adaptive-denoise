import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from skimage import data, img_as_float

from api.main import app
from denoise.core import add_noise, denoise, psnr

# Creates a test client connected to your FastAPI application
# It allows you to call endpoints using client.post() or client.get() without deploying a server
client = TestClient(app)

# Reference image used to evaluate the denoising algorithms
CLEAN = img_as_float(data.camera())[:128, :128] # small crop = fast tests

# Test whether denoising improves PSNR
@pytest.mark.parametrize("method,param", [("gaussian", 1.0), ("tv", 0.1)])
def test_denoising_improves_psnr(method, param):
    noisy = add_noise(CLEAN, sigma=0.1, seed=0) # adds reproducible Gaussian noise to the reference image
    result = denoise(noisy, method, param) # applies the selected denoising method
    assert psnr(CLEAN, result) > psnr(CLEAN, noisy) # checks that PSNR after denoising is higher than before

# Test an invalid method
def test_invalid_method_raises():
    with pytest.raises(ValueError):
        denoise(CLEAN, "magic", 0.1)

# Test an invalid parameter
def test_invalid_param_raises():
    with pytest.raises(ValueError):
        denoise(CLEAN, "tv", -1)

# Test that the API returns an image
def test_api_returns_image(): # Defines an API integration test
    buf = io.BytesIO()
    Image.fromarray((CLEAN * 255).astype(np.uint8)).save(buf, format="PNG")
    r = client.post( # Simulates an HTTP POST request
        "/denoise",
        files={"file": ("x.png", buf.getvalue(), "image/png")},
        # Uploads the image under the required form field file
        data={"method": "tv", "param": "0.1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["shape"] == [128, 128]
    assert body["image_png_base64"]

# Test that the API rejects an invalid method
def test_api_rejects_bad_method():
    r = client.post( # Simulates another HTTP request.
        "/denoise",
        files={"file": ("x.png", b"not an image", "image/png")}, # Claims the uploaded data is a PNG
        data={"method": "magic"},
    )
    assert r.status_code == 400

def _png_bytes():
    buf = io.BytesIO()
    Image.fromarray((CLEAN * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_api_adaptive_predicts_param():
    r = client.post("/denoise", files={"file": ("x.png", _png_bytes(), "image/png")},
                    data={"method": "tv_adaptive"})
    assert r.status_code == 200
    body = r.json()
    assert body["param_source"] == "predicted"
    assert body["param"] > 0


def test_api_requires_param_for_fixed_methods():
    r = client.post("/denoise", files={"file": ("x.png", _png_bytes(), "image/png")},
                    data={"method": "tv"})
    assert r.status_code == 400

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Every test uses a fresh, temporary database."""
    monkeypatch.setattr("api.main.DB_PATH", tmp_path / "test.db")


def test_requests_are_logged_in_stats():
    client.post("/denoise", files={"file": ("x.png", _png_bytes(), "image/png")},
                data={"method": "tv_adaptive"})
    client.post("/denoise", files={"file": ("x.png", _png_bytes(), "image/png")},
                data={"method": "tv", "param": "0.1"})
    stats = client.get("/stats").json()
    assert stats["total_requests"] == 2
    assert set(stats["by_method"]) == {"tv", "tv_adaptive"}


def test_failed_requests_are_not_logged():
    client.post("/denoise", files={"file": ("x.png", _png_bytes(), "image/png")},
                data={"method": "magic"})
    assert client.get("/stats").json()["total_requests"] == 0

def test_recent_requests_endpoint():
    client.post("/denoise", files={"file": ("x.png", _png_bytes(), "image/png")},
                data={"method": "tv_adaptive"})
    rows = client.get("/requests").json()
    assert len(rows) == 1
    assert rows[0]["method"] == "tv_adaptive"
    assert rows[0]["param"] > 0