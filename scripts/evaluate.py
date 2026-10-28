# Compare fixed, adaptive (MLP) and oracle TV weights on unseen test images
import matplotlib.pyplot as plt
import numpy as np

from denoise.core import denoise, psnr
from denoise.model import LambdaPredictor

d = np.load("data/dataset.npz")
lambdas = d["lambdas"]
scores_train, scores_test = d["scores_train"], d["scores_test"]
clean = d["clean_test"].astype(np.float64)
noisy = d["noisy_test"].astype(np.float64)
sigma_hat = d["X_test"][:, 0]

# 1. Fixed baseline: the one weight with the best average PSNR on TRAINING patches
fixed_idx = int(np.argmax(scores_train.mean(axis=0)))
fixed_lam = lambdas[fixed_idx]
psnr_fixed = scores_test[:, fixed_idx]

# 2. Oracle: best grid weight for each test patch (needs the clean image)
psnr_oracle = scores_test.max(axis=1)

# 3. Adaptive: weight predicted by the MLP from the noisy patch alone
predictor = LambdaPredictor("models/lambda_mlp.pt")
pred_lams = np.array([predictor.predict(n) for n in noisy])
psnr_adaptive = np.array([psnr(c, denoise(n, "tv", lam))
                          for c, n, lam in zip(clean, noisy, pred_lams)])

psnr_noisy = np.array([psnr(c, n) for c, n in zip(clean, noisy)])

print(f"Fixed weight (chosen on training data): {fixed_lam:.3f}\n")
print(f"{'method':<20}{'mean PSNR (dB)':>16}")
for name, v in [("noisy input", psnr_noisy), ("fixed weight", psnr_fixed),
                ("adaptive (MLP)", psnr_adaptive), ("oracle", psnr_oracle)]:
    print(f"{name:<20}{v.mean():>16.2f}")

gap = (psnr_adaptive.mean() - psnr_fixed.mean()) / (psnr_oracle.mean() - psnr_fixed.mean())
print(f"\nAdaptive closes {gap:.0%} of the gap between fixed and oracle")
print(f"Adaptive beats fixed on {(psnr_adaptive > psnr_fixed).mean():.0%} of test patches")

print("\nPer test image (mean gain over fixed, dB):")
for i, name in enumerate(["camera", "coins"]):
    s = slice(40 * i, 40 * (i + 1))
    print(f"  {name:<8} adaptive {np.mean(psnr_adaptive[s] - psnr_fixed[s]):+.2f}"
          f"   oracle {np.mean(psnr_oracle[s] - psnr_fixed[s]):+.2f}")

# Chart: PSNR gain over the fixed weight, grouped by estimated noise level
order = np.argsort(sigma_hat)
groups = np.array_split(order, 5)
x = [sigma_hat[g].mean() for g in groups]

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color="grey", lw=1, label=f"fixed weight ({fixed_lam:.3f})")
ax.plot(x, [np.mean(psnr_adaptive[g] - psnr_fixed[g]) for g in groups],
        marker="o", label="adaptive (MLP)")
ax.plot(x, [np.mean(psnr_oracle[g] - psnr_fixed[g]) for g in groups],
        marker="o", linestyle="--", label="oracle (best grid weight)")
ax.set_xlabel("estimated noise level σ̂")
ax.set_ylabel("PSNR gain over fixed weight (dB)")
ax.set_title("Adaptive TV weight on unseen images")
ax.legend()
fig.tight_layout()
fig.savefig("results/eval.png", dpi=150)
print("\nsaved results/eval.png")