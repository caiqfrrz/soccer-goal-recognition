"""Models that classify a (dim, 40) window as goal / not goal."""
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class GoalCNN(nn.Module):
    """1D-CNN over time. Input (B, in_dim, T) -> one logit per window."""

    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_logreg():
    """Baseline: logistic regression on the time-averaged window."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))


# --- checkpoints: weights + the normalization they were trained with ---------
def save_checkpoint(path, model, mean, std, **extra):
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, **extra}, path)


def load_checkpoint(path, device):
    """Returns (model in eval mode, mean, std)."""
    # weights_only=False because the file also stores numpy mean/std (our own file)
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model = GoalCNN(in_dim=len(ckpt["mean"])).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]
