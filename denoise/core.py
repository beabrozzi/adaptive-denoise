# Core image denoising functions
import numpy as np
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio

METHODS = ("gaussian", "tv")

def add_noise(image: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    # Add Gaussian noise with standard deviation 'sigma' to an image in [0, 1]
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 1)

def denoise(image : np.ndarray, method: str, param: float) -> np.ndarray:
    # Denoise a grayscale image with values in [0, 1]
    # gaussian: param = sigma of the blur kernel
    # tv:       param = regularisation weight (lambda)
    if param <= 0:
        raise ValueError("param must be positive")
    if method == "gaussian":
        return gaussian(image, sigma=param)
    if method == "tv":
        return denoise_tv_chambolle(image, weight=param)
    raise ValueError(f"unknown method '{method}', choose from {METHODS}")

def psnr(clean: np.ndarray, test: np.ndarray) -> float:
    # Peak signal-to-noise ratio in dB (higher = closer to the clean image)
    return float(peak_signal_noise_ratio(clean, test, data_range=1.0))