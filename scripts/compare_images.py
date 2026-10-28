
"""Visual comparison: fixed vs adaptive TV weight on an unseen image."""
import matplotlib.pyplot as plt
from skimage import data, img_as_float

from denoise.core import add_noise, denoise, psnr
from denoise.model import LambdaPredictor

FIXED_LAM = 0.136                      # best fixed weight from training (Step 8)
clean = img_as_float(data.camera())    # test image: never seen in training
predictor = LambdaPredictor("models/lambda_mlp.pt")

fig, axes = plt.subplots(2, 4, figsize=(14, 8.5), constrained_layout=True)
for row, sigma in enumerate([0.05, 0.15]):
    noisy = add_noise(clean, sigma, seed=0)
    lam = predictor.predict(noisy)
    fixed = denoise(noisy, "tv", FIXED_LAM)
    adaptive = denoise(noisy, "tv", lam)

    panels = [
        (clean, "clean"),
        (noisy, f"noisy (σ={sigma})\n{psnr(clean, noisy):.2f} dB"),
        (fixed, f"fixed λ={FIXED_LAM}\n{psnr(clean, fixed):.2f} dB"),
        (adaptive, f"adaptive λ={lam:.3f}\n{psnr(clean, adaptive):.2f} dB"),
    ]
    for ax, (img, title) in zip(axes[row], panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

fig.savefig("results/comparison.png", dpi=120)
print("saved results/comparison.png")