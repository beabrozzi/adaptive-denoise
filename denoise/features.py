
# Features describing a noisy image, used to predict the TV weight.

import numpy as np
from skimage.restoration import estimate_sigma

FEATURE_NAMES = ("sigma_hat", "grad_mean", "intensity_std")

def extract_features(noisy: np.ndarray) -> np.ndarray:
    sigma_hat = estimate_sigma(noisy)            # estimated noise std
    gy, gx = np.gradient(noisy)
    grad_mean = np.mean(np.hypot(gx, gy))        # how much edge/texture
    intensity_std = np.std(noisy)                # contrast of the image
    return np.array([sigma_hat, grad_mean, intensity_std])