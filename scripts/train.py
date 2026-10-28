
"""Train the MLP to predict log(best TV weight) from image features."""
import numpy as np
import torch
from torch import nn
import copy

from denoise.model import LambdaMLP

torch.manual_seed(0)
d = np.load("data/dataset.npz")
X_tr = torch.tensor(d["X_train"], dtype=torch.float32)
y_tr = torch.tensor(d["y_train"], dtype=torch.float32)
X_te = torch.tensor(d["X_test"], dtype=torch.float32)
y_te = torch.tensor(d["y_test"], dtype=torch.float32)

# Standardise features using TRAINING statistics only
mean, std = X_tr.mean(0), X_tr.std(0)
norm = lambda X: (X - mean) / std

# Hold out 20% of training patches to watch for overfitting
perm = torch.randperm(len(X_tr))
n_val = len(X_tr) // 5
val_idx, fit_idx = perm[:n_val], perm[n_val:]

model = LambdaMLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

best_val, best_state = float("inf"), None

for epoch in range(2000):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(norm(X_tr[fit_idx])), y_tr[fit_idx])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val = loss_fn(model(norm(X_tr[val_idx])), y_tr[val_idx]).item()
    if val < best_val:                       # remember the best epoch so far
        best_val, best_state = val, copy.deepcopy(model.state_dict())
        best_epoch = epoch
    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  train {loss.item():.4f}  val {val:.4f}")

model.load_state_dict(best_state)            # roll back to the best weights
print(f"best epoch: {best_epoch} (val {best_val:.4f})")

model.eval()
with torch.no_grad():
    test_mse = loss_fn(model(norm(X_te)), y_te).item()
baseline_mse = ((y_te - y_tr.mean()) ** 2).mean().item()

print(f"\nTest MSE (log space): model {test_mse:.4f} | always-guess-average {baseline_mse:.4f}")
print(f"Typical error: predicted lambda off by a factor of x{np.exp(np.sqrt(test_mse)):.2f}")

torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std},
           "models/lambda_mlp.pt")
print("saved models/lambda_mlp.pt")