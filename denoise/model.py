# Small MLP that predicts log(TV weight) from image features
import numpy as np
import torch
from torch import nn

from denoise.features import extract_features


class LambdaMLP(nn.Module):
    def __init__(self, n_features: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # one number per image: log(lambda)


class LambdaPredictor:
    # Loads a trained model and predicts the TV weight for a noisy image

    def __init__(self, path: str):
        ckpt = torch.load(path)
        self.model = LambdaMLP()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mean, self.std = ckpt["mean"], ckpt["std"]

    def predict(self, noisy: np.ndarray) -> float:
        x = torch.tensor(extract_features(noisy), dtype=torch.float32)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            log_lam = self.model(x.unsqueeze(0)).item()
        return float(np.exp(log_lam))    # exp() => always positive