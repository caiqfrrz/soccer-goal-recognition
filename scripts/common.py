"""
Things every script needs. Read this file once and the scripts become easy.

  1. SETTINGS        - folders and numbers used everywhere
  2. READING DATA    - goals, events, features, training windows
  3. THE MODEL       - the 1D-CNN, how to train it, how to use it
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 1. SETTINGS
# =============================================================================

PROJECT = Path(__file__).resolve().parent.parent      # the "projeto" folder
SOCCERNET = PROJECT / "data" / "soccernet"            # downloaded games
GOALS_CSV = PROJECT / "data" / "goals.csv"            # list of all goals (script 01)
OUTPUTS = PROJECT / "outputs"

# Which features to use. SoccerNet's own are stored as "<half>_<name>.npy"
# inside each game folder. Later, our own features will get a new name and
# live in data/features/<name>/ (see feature_file below).
FEATURES = "ResNET_TF2_PCA512"
WINDOWS = PROJECT / "data" / "windows" / FEATURES      # training windows (script 03)
MODEL_FILE = OUTPUTS / "models" / f"goal_cnn_hard_{FEATURES}.pt"

# The features have 2 rows per second of video: row 0 = 0.0 s, row 1 = 0.5 s ...
FPS = 2
# A "window" is 20 rows before + 20 rows after a moment = 40 rows = 20 seconds.
HALF_WINDOW = 20

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seconds_to_row(seconds):
    return int(seconds * FPS)


# =============================================================================
# 2. READING DATA
# =============================================================================

def feature_file(game, half):
    """Path of the feature file of one half of one game."""
    if FEATURES.startswith(("ResNET", "C3D", "I3D")):                # SoccerNet's
        return SOCCERNET / game / f"{half}_{FEATURES}.npy"
    return PROJECT / "data" / "features" / FEATURES / game / f"{half}.npy"   # ours


def load_features(game, half):
    """Array of shape (rows, 512): one row per half second of the match."""
    return np.load(feature_file(game, half))


def load_goals(split):
    """All goals of one split ("train", "valid" or "test").
    Returns a list of dicts: {game, half, seconds, visibility}."""
    goals = []
    with open(GOALS_CSV) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                goals.append({"game": row["game"],
                              "half": int(row["half"]),
                              "seconds": int(row["position_ms"]) / 1000,
                              "visibility": row["visibility"]})
    return goals


def load_events(game):
    """Every annotated event of a game (goals, fouls, corners ...).
    Returns a list of dicts: {half, seconds, label}."""
    labels = json.loads((SOCCERNET / game / "Labels-v2.json").read_text())
    events = []
    for a in labels["annotations"]:
        events.append({"half": int(a["gameTime"][0]),       # gameTime looks like "1 - 13:10"
                       "seconds": int(a["position"]) / 1000,
                       "label": a["label"]})
    return events


def load_windows(split):
    """The windows built by script 03.
    X:    (N, 40, 512) features of each window
    y:    (N,)         1 = goal, 0 = not a goal
    info: list of N dicts saying where each window comes from and its kind
          ("goal", "random" or "hard")."""
    X = np.load(WINDOWS / f"X_{split}.npy")
    y = np.load(WINDOWS / f"y_{split}.npy")
    with open(WINDOWS / f"meta_{split}.csv") as f:
        info = list(csv.DictReader(f))
    return X, y, info


# =============================================================================
# 3. THE MODEL
# =============================================================================

class GoalCNN(nn.Module):
    """Looks at a 20 s window and outputs one number: how much it looks like a goal.

    Input shape (batch, 512, 40): 512 features, 40 time steps.
      Conv1d layers slide along TIME, so they can see patterns that unfold over
      a few seconds (shot -> net -> celebration).
      MaxPool keeps the strongest response anywhere in the window.
      Linear turns it into a single score (a "logit"; sigmoid(logit) = probability).
    """

    def __init__(self, n_features=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def normalize(X, mean, std):
    """Give every feature mean 0 / std 1 (using TRAIN statistics) and reorder
    (N, 40, 512) -> (N, 512, 40), the shape Conv1d wants."""
    X = (X - mean) / std
    return torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)


def train_cnn(X_train, y_train, seed, epochs=10):
    """Train a GoalCNN from zero. X_train must already be normalized."""
    torch.manual_seed(seed)          # same seed -> same result
    model = GoalCNN(n_features=X_train.shape[1]).to(DEVICE)

    # There are ~13x more non-goals than goals. pos_weight makes a missed goal
    # cost as much as 13 false alarms, so the model does not just say "no" always.
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos],
                                                           dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(X_train, torch.tensor(y_train, dtype=torch.float32))
    batches = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in batches:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict(model, X):
    """Probability of 'goal' (0..1) for each window of X (already normalized)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            logits = model(X[start:start + 256].to(DEVICE))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def save_model(model, mean, std):
    """Save the weights AND the normalization numbers (needed to use it later)."""
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std}, MODEL_FILE)


def load_model():
    """Returns (model, mean, std) saved by save_model."""
    saved = torch.load(MODEL_FILE, weights_only=False, map_location=DEVICE)
    model = GoalCNN(n_features=len(saved["mean"])).to(DEVICE)
    model.load_state_dict(saved["state"])
    model.eval()
    return model, saved["mean"], saved["std"]
