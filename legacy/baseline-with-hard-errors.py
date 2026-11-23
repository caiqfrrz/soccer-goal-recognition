"""
Step 4e: does training with hard negatives help?
Trains on ./dataset (random negatives only) and on ./dataset (+ hard negatives),
and evaluates BOTH on the same, harder validation set: ./dataset valid.
"""
import csv

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEEDS, EPOCHS, BATCH_SIZE, LR = [1, 2, 3], 10, 64, 1e-3

# --- Part 1: the common, harder validation set ---
X_valid = np.load("dataset/X_valid.npy")
y_valid = np.load("dataset/y_valid.npy")
with open("dataset/meta_valid.csv") as f:
    kind_valid = np.array([m["kind"] for m in csv.DictReader(f)])   # goal / random / hard


def scores(pred):
    """precision/recall/F1 on goals + false-alarm rate on each negative type."""
    return {
        "P": precision_score(y_valid, pred), "R": recall_score(y_valid, pred),
        "F1": f1_score(y_valid, pred),
        "FA_random": pred[kind_valid == "random"].mean(),
        "FA_hard": pred[kind_valid == "hard"].mean(),
    }


class GoalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def run_cnn(X_tr, y_tr, X_va, seed):
    torch.manual_seed(seed)
    model = GoalCNN().to(device)
    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, torch.tensor(y_tr, dtype=torch.float32)),
                        batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xb.to(device)).cpu()
                            for xb in DataLoader(X_va, batch_size=256)])
    return (torch.sigmoid(logits) > 0.5).numpy().astype(int)


def summarize(name, runs):
    keys = runs[0].keys()
    text = "  ".join(f"{k}={np.mean([r[k] for r in runs]):.3f}"
                     + (f"±{np.std([r[k] for r in runs]):.3f}" if len(runs) > 1 else "")
                     for k in keys)
    print(f"{name:32} {text}")


# --- Part 2: train with and without hard negatives, evaluate on the harder valid set ---
# Both training sets come from dataset: "random only" just drops the hard rows,
# which gives exactly the old ./dataset train set without storing it on disk.
X_full = np.load("dataset/X_train.npy", mmap_mode="r")   # read rows only when needed
y_full = np.load("dataset/y_train.npy")
with open("dataset/meta_train.csv") as f:
    kind_train = np.array([m["kind"] for m in csv.DictReader(f)])

for name, mask in [("random negatives only", kind_train != "hard"),
                   ("random + hard negatives", np.ones(len(y_full), dtype=bool))]:
    X_tr = np.asarray(X_full[mask])
    y_tr = y_full[mask]
    print(f"\nTraining on {name}: {X_tr.shape[0]} windows "
          f"({int(y_tr.sum())} goals, {int((y_tr == 0).sum())} negatives)")

    # logistic regression on time-averaged windows
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced"))
    lr.fit(X_tr.mean(axis=1), y_tr)
    summarize("  logistic regression", [scores(lr.predict(X_valid.mean(axis=1)))])

    # 1D-CNN, standardized with THIS training set's statistics
    mean, std = X_tr.mean(axis=(0, 1)), X_tr.std(axis=(0, 1)) + 1e-6
    X_tr_t = torch.tensor((X_tr - mean) / std, dtype=torch.float32).permute(0, 2, 1)
    X_va_t = torch.tensor((X_valid - mean) / std, dtype=torch.float32).permute(0, 2, 1)
    del X_tr                                   # free RAM before training
    runs = [scores(run_cnn(X_tr_t, y_tr, X_va_t, s)) for s in SEEDS]
    summarize(f"  1D-CNN ({len(SEEDS)} seeds)", runs)