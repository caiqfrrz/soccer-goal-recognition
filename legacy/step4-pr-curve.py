"""
Step 4f: threshold-free evaluation (precision-recall curve + average precision)
for the models trained with hard negatives, evaluated on the hard valid set.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEEDS, EPOCHS, BATCH_SIZE, LR = [1, 2, 3], 10, 64, 1e-3
TARGET_RECALLS = [0.70, 0.80, 0.90]

# --- Part 1: data (train with hard negatives, hard valid) ---
X_tr = np.load("dataset/X_train.npy")
y_tr = np.load("dataset/y_train.npy")
X_va = np.load("dataset/X_valid.npy")
y_va = np.load("dataset/y_valid.npy")
print(f"valid: {int(y_va.sum())} goals, {int((y_va == 0).sum())} negatives "
      f"(a random guess would have AP ≈ {y_va.mean():.3f})")


# --- Part 2: helpers to read the precision-recall curve ---
def operating_points(prob):
    """Best precision reachable at each target recall, plus the max-F1 point."""
    precision, recall, thresholds = precision_recall_curve(y_va, prob)
    precision, recall = precision[:-1], recall[:-1]        # align with thresholds
    out = {}
    for target in TARGET_RECALLS:
        ok = recall >= target
        i = np.argmax(np.where(ok, precision, -1))
        out[f"R>={target:.2f}"] = (precision[i], recall[i], thresholds[i])
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    i = np.argmax(f1)
    out["max F1"] = (precision[i], recall[i], thresholds[i], f1[i])
    return out


def report(name, probs):
    aps = [average_precision_score(y_va, p) for p in probs]
    std = f" ± {np.std(aps):.3f}" if len(aps) > 1 else ""
    print(f"\n{name}:  AP = {np.mean(aps):.3f}{std}")
    points = [operating_points(p) for p in probs]
    for key in points[0]:
        P = np.mean([pt[key][0] for pt in points])
        R = np.mean([pt[key][1] for pt in points])
        T = np.mean([pt[key][2] for pt in points])
        extra = f"  F1={np.mean([pt[key][3] for pt in points]):.3f}" if key == "max F1" else ""
        print(f"  {key:8}  precision={P:.3f}  recall={R:.3f}  threshold={T:.3f}{extra}")


# --- Part 3: logistic regression ---
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(max_iter=1000, class_weight="balanced"))
lr.fit(X_tr.mean(axis=1), y_tr)
lr_prob = lr.predict_proba(X_va.mean(axis=1))[:, 1]
report("Logistic regression", [lr_prob])


# --- Part 4: 1D-CNN, 3 seeds, keep the probabilities ---
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


mean, std = X_tr.mean(axis=(0, 1)), X_tr.std(axis=(0, 1)) + 1e-6
X_tr_t = torch.tensor((X_tr - mean) / std, dtype=torch.float32).permute(0, 2, 1)
X_va_t = torch.tensor((X_va - mean) / std, dtype=torch.float32).permute(0, 2, 1)
del X_tr
pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()],
                          dtype=torch.float32, device=device)

cnn_probs = []
for seed in SEEDS:
    torch.manual_seed(seed)
    model = GoalCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr_t, torch.tensor(y_tr, dtype=torch.float32)),
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
                            for xb in DataLoader(X_va_t, batch_size=256)])
    cnn_probs.append(torch.sigmoid(logits).numpy())
    if seed == SEEDS[0]:   # keep one model (+ its normalization) for the spotting step
        torch.save({"state": model.state_dict(), "mean": mean, "std": std}, "goal_cnn_hard.pt")

report(f"1D-CNN ({len(SEEDS)} seeds)", cnn_probs)

# --- Part 5: plot the curves (optional, needs matplotlib) ---
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    p, r, _ = precision_recall_curve(y_va, lr_prob)
    plt.plot(r, p, label=f"Logistic regression (AP {average_precision_score(y_va, lr_prob):.2f})")
    p, r, _ = precision_recall_curve(y_va, cnn_probs[0])
    plt.plot(r, p, label=f"1D-CNN, seed 1 (AP {average_precision_score(y_va, cnn_probs[0]):.2f})")
    plt.axhline(y_va.mean(), linestyle="--", color="gray", label="random guess")
    plt.xlabel("Recall (goals found)")
    plt.ylabel("Precision (alarms that are goals)")
    plt.title("Goal detection, hard validation set")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("pr_curve.png", dpi=150, bbox_inches="tight")
    print("\nSaved pr_curve.png")
except ImportError:
    print("\n(matplotlib not installed: pip install matplotlib to get pr_curve.png)")