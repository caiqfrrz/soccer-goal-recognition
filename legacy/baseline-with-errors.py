import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("./SoccerNet")
FPS = 2
SEED, EPOCHS, BATCH_SIZE, LR = 1, 10, 64, 1e-3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Part 1: train the same 1D-CNN as step 4c (one seed) ---
X_train, y_train = np.load("dataset/X_train.npy"), np.load("dataset/y_train.npy")
X_valid, y_valid = np.load("dataset/X_valid.npy"), np.load("dataset/y_valid.npy")
mean, std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)) + 1e-6
X_train_t = torch.tensor((X_train - mean) / std, dtype=torch.float32).permute(0, 2, 1)
X_valid_t = torch.tensor((X_valid - mean) / std, dtype=torch.float32).permute(0, 2, 1)
y_train_t = torch.tensor(y_train, dtype=torch.float32)


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


torch.manual_seed(SEED)
model = GoalCNN().to(device)
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()],
                          dtype=torch.float32, device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
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
                        for xb in DataLoader(X_valid_t, batch_size=256)])
prob = torch.sigmoid(logits).numpy()      # model's confidence that each window is a goal
pred = (prob > 0.5).astype(int)

# --- Part 2: attach metadata to every valid window ---
with open("dataset/meta_valid.csv") as f:
    meta = list(csv.DictReader(f))        # same order as X_valid / y_valid

# visibility of each goal, looked up by (game, half, row)
visibility = {}
with open("goals.csv") as f:
    for g in csv.DictReader(f):
        if g["split"] == "valid":
            row = int(int(g["position_ms"]) / 1000 * FPS)
            visibility[(g["game"], int(g["half"]), row)] = g["visibility"]

# every annotated event of each valid half, to see what happens inside a window
events = {}
def events_in_window(game, half, center_row, radius_s=10):
    if game not in events:
        anns = json.loads((ROOT / game / "Labels-v2.json").read_text())["annotations"]
        events[game] = [(int(a["gameTime"][0]), int(a["position"]), a["label"]) for a in anns]
    center_ms = center_row / FPS * 1000
    return sorted({lab for h, ms, lab in events[game]
                   if h == half and abs(ms - center_ms) <= radius_s * 1000})


def clock(center_row):
    s = center_row / FPS
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


rows = []
for m, y, p, pr in zip(meta, y_valid, pred, prob):
    game, half, c = m["game"], int(m["half"]), int(m["center_row"])
    kind = {(1, 1): "TP", (1, 0): "missed goal", (0, 1): "false alarm", (0, 0): "TN"}[(y, p)]
    rows.append({
        "kind": kind, "game": game, "half": half, "time": clock(c),
        "prob": round(float(pr), 3),
        "visibility": visibility.get((game, half, c), "") if y == 1 else "",
        "events_in_window": "; ".join(events_in_window(game, half, c)),
    })

# --- Part 3: missed goals — is visibility the reason? ---
goals = [r for r in rows if r["kind"] in ("TP", "missed goal")]
print("Recall by goal visibility:")
for vis, n in Counter(r["visibility"] for r in goals).items():
    found = sum(r["kind"] == "TP" for r in goals if r["visibility"] == vis)
    print(f"  {vis:10}  {found}/{n} found  (recall {found / n:.2f})")

# --- Part 4: false alarms — which events fool the model? ---
negatives = [r for r in rows if r["kind"] in ("TN", "false alarm")]
false_alarms = [r for r in negatives if r["kind"] == "false alarm"]
print(f"\nFalse alarms: {len(false_alarms)} of {len(negatives)} negatives")
count_all = Counter(e for r in negatives for e in r["events_in_window"].split("; ") if e)
count_fa = Counter(e for r in false_alarms for e in r["events_in_window"].split("; ") if e)
print(f"  {'event within ±10 s':25} {'in false alarms':>16} {'in all negatives':>17} {'FA rate':>8}")
for event, n_all in count_all.most_common():
    n_fa = count_fa[event]
    print(f"  {event:25} {n_fa:>16} {n_all:>17} {n_fa / n_all:>8.2f}")
no_event_fa = sum(1 for r in false_alarms if not r["events_in_window"])
no_event_all = sum(1 for r in negatives if not r["events_in_window"])
print(f"  {'(no annotated event)':25} {no_event_fa:>16} {no_event_all:>17} "
      f"{no_event_fa / max(no_event_all, 1):>8.2f}")

# --- Part 5: save every error for manual inspection ---
errors = sorted([r for r in rows if r["kind"] in ("missed goal", "false alarm")],
                key=lambda r: (r["kind"], -r["prob"]))
with open("errors_valid.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(errors)
print(f"\nSaved {len(errors)} errors to errors_valid.csv")