"""Build the dataset: for each noisy patch, try every TV weight and record the PSNR."""
import numpy as np
from skimage import data, img_as_float
from skimage.color import rgb2gray

from denoise.core import add_noise, denoise, psnr
from denoise.features import extract_features

LAMBDAS = np.geomspace(0.01, 0.5, 25)   # candidate TV weights, log-spaced
PATCH = 128
PATCHES_PER_IMAGE = 40
SIGMA_RANGE = (0.02, 0.25)

TRAIN_IMAGES = ["astronaut", "moon", "page", "text", "brick",
                "grass", "gravel", "coffee", "chelsea", "rocket"]
TEST_IMAGES = ["camera", "coins"]        # never seen during training


def load_gray(name):
    img = getattr(data, name)()
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])
    return img_as_float(img)


def psnr_per_lambda(clean, noisy):
    """PSNR of TV denoising for every candidate weight."""
    return np.array([psnr(clean, denoise(noisy, "tv", lam)) for lam in LAMBDAS])


def build(names, rng, keep_patches=False):
    X, y, scores, cleans, noisies = [], [], [], [], []
    for name in names:
        try:
            img = load_gray(name)
        except Exception as e:
            print(f"skipping {name}: {e}")
            continue
        h, w = img.shape
        for _ in range(PATCHES_PER_IMAGE):
            r = rng.integers(0, h - PATCH + 1)
            c = rng.integers(0, w - PATCH + 1)
            clean = img[r:r + PATCH, c:c + PATCH]
            sigma = rng.uniform(*SIGMA_RANGE)
            noisy = add_noise(clean, sigma, seed=int(rng.integers(0, 2**31)))

            s = psnr_per_lambda(clean, noisy)
            X.append(extract_features(noisy))
            y.append(np.log(LAMBDAS[np.argmax(s)]))   # label: log of best weight
            scores.append(s)
            if keep_patches:
                cleans.append(clean.astype(np.float32))
                noisies.append(noisy.astype(np.float32))
        print(f"{name}: done")
    out = dict(X=np.array(X), y=np.array(y), scores=np.array(scores))
    if keep_patches:
        out["clean"] = np.array(cleans)
        out["noisy"] = np.array(noisies)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    train = build(TRAIN_IMAGES, rng)
    test = build(TEST_IMAGES, rng, keep_patches=True)
    np.savez(
        "data/dataset.npz",
        lambdas=LAMBDAS,
        X_train=train["X"], y_train=train["y"], scores_train=train["scores"],
        X_test=test["X"], y_test=test["y"], scores_test=test["scores"],
        clean_test=test["clean"], noisy_test=test["noisy"],
    )
    print(f"train: {train['X'].shape}, test: {test['X'].shape}")