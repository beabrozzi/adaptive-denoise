
# REST API exposing the denoising methods
import base64
import io
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from denoise.core import METHODS, denoise
from denoise.model import LambdaPredictor  

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lambda_mlp.pt"
API_METHODS = METHODS + ("tv_adaptive",)
# Creating the FastAPI application, returns an application obj stored in the var app
app = FastAPI(title="Adaptive Denoise API")
# Load the model once
predictor = LambdaPredictor(str(MODEL_PATH))


def load_grayscale(data: bytes) -> np.ndarray:
    # Turn uploaded bytes into a grayscale float image in [0, 1],
    # which is the umeric format expected by the denoising algorithm
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0

# It does the opposite of the load_grayscale function
# It converts a processed NumPy array back into an image that can be transmitted to the client
def to_png_based64(image: np.ndarray) -> str:
    # Encode a [0, 1] image as a PNG, then as base64 text so it fits in JSON
    img = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# Useful for checking whether the API is running

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/denoise")
async def denoise_endpoint(
    file: UploadFile = File(...),
    method: str = Form("tv_adaptive"),
    param: float | None = Form(None),
):
    # Validate the method
    if method not in API_METHODS:
        raise HTTPException(400, f"method must be one of {API_METHODS}")
    if method != "tv_adaptive" and param is None:
        raise HTTPException(400, f"param is required for method '{method}'")
    # Read the uploaded image
    try:
        image = load_grayscale(await file.read())
    except Exception:
        raise HTTPException(400, "could not read image")

    # Execute the algorithm
    start = time.perf_counter()
    try:
        if method == "tv_adaptive":
            param = predictor.predict(image)          # the model chooses lambda
            result = denoise(image, "tv", param)
        else:
            result = denoise(image, method, param)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    # This measures the core denoising execution time, not the total API response time
    runtime_ms = (time.perf_counter() - start) * 1000

    # FastAPI automatically converts this dictionary into JSON
    return {
                "method": method,
        "param": round(param, 4),
        "param_source": "predicted" if method == "tv_adaptive" else "user",
        "shape": list(image.shape),
        "runtime_ms": round(runtime_ms, 2),
        "image_png_base64": to_png_based64(result),
    }