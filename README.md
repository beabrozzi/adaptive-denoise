
# Adaptive Denoise

An image-denoising service where a small neural network chooses the
**Total Variation (TV) regularisation strength λ for each image**, instead of
using one fixed value for every input.

![Fixed vs adaptive λ on an unseen image](results/comparison.png)

## Motivation

TV denoising removes noise while keeping edges sharp, but its result depends
heavily on λ: too small and noise remains, too large and the image becomes
flat and "cartoon-like". The best λ changes with the noise level and the
image content, so a single hand-picked value is a compromise.

During my research internship at UCL I worked on learning TV regularisation
parameters using Fenchel–Young losses. This is an independent personal
project exploring the same question with a simpler, fully supervised approach,
and turning it into a usable tool.

## How it works

1. **Training data**: random 128×128 patches from scikit-image sample images,
   corrupted with Gaussian noise (σ between 0.02 and 0.25). For each patch,
   25 log-spaced values of λ are tried and the one with the highest PSNR is
   kept as the label.
2. **Features**: the model sees only the noisy image, summarised by
   3 features: estimated noise level (wavelet-based), mean gradient magnitude,
   and intensity standard deviation.
3. **Model**: a small MLP (3 → 16 → 16 → 1) trained in PyTorch to predict
   **log(λ)**, so predictions are always positive after `exp()`.
   Features are standardised with training statistics, and early stopping
   on a validation split prevents overfitting.
4. **Serving**: a FastAPI service exposes `/denoise`. With
   `method=tv_adaptive` the model predicts λ, then TV denoising runs with it.
   A small web page calls the API and shows the methods side by side.

## Results

Evaluated on **80 patches from 2 images never used in training**
(`camera`, `coins`). The fixed baseline is the single λ with the best
average PSNR on the *training* set (λ = 0.136).

| Method | Mean PSNR (dB) |
|---|---|
| Noisy input | 18.95 |
| TV, fixed λ | 25.87 |
| **TV, adaptive λ (MLP)** | **26.85** |
| TV, oracle λ (best of grid, needs clean image) | 27.14 |

- Adaptive λ closes **77%** of the gap between the fixed baseline and the oracle.
- It beats the fixed λ on **74%** of test patches.
- The gain is largest at low noise, where a fixed λ over-smooths:

![PSNR gain over fixed λ by noise level](results/eval.png)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn api.main:app --reload
```

- Web page: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs

To reproduce the model and results:

```bash
python -m scripts.make_dataset   # build training/test data
python -m scripts.train          # train the MLP
python -m scripts.evaluate       # metrics + results/eval.png
pytest -v                        # run the tests
```

## Using the API from another application

```bash
curl -X POST http://127.0.0.1:8000/denoise \
  -F "file=@noisy.png" \
  -F "method=tv_adaptive"
```

Response (JSON):

```json
{
  "method": "tv_adaptive",
  "param": 0.1046,
  "param_source": "predicted",
  "shape": [512, 512],
  "runtime_ms": 26.8,
  "image_png_base64": "..."
}
```

Methods: `gaussian` (param = blur σ), `tv` (param = λ), `tv_adaptive` (λ predicted).

## Project structure

```
denoise/    core algorithms, feature extraction, model definition
api/        FastAPI service
static/     web page (HTML + JavaScript client of the API)
scripts/    dataset generation, training, evaluation
tests/      unit and API tests
models/     trained model weights
results/    evaluation charts
```

## Limitations and future work

- **High noise**: gains are small at high noise levels; the gradient
  feature is dominated by noise there. Noise-robust features (e.g. gradients
  of a pre-smoothed image) or a small CNN on the pixels could help.
- **Metric**: PSNR does not fully match perceived quality. Two values of λ
  can have the same PSNR but keep different amounts of texture. Evaluating
  (or training) with SSIM is a natural next step.
- **TV itself** loses fine texture even at the best λ; stronger denoisers
  (non-local means, BM3D, learned denoisers) would raise the ceiling.
- Grayscale only, Gaussian noise only, trained on 128×128 patches.
- Small test set (2 images); more held-out images would give tighter estimates.