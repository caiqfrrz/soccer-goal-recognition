"""
Step 10: goal SPOTTING over full halves (valid split).

1. score every second of every half with the trained CNN (cached to a file)
2. peak picking (non-maximum suppression) -> predicted goal times
3. match predictions to real goals within a tolerance
4. metrics: AP, recall, precision, false alarms per match, localization offset
"""
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from SoccerNet.utils import getListGames

ROOT = Path("./SoccerNet")
FEAT = "ResNET_TF2_PCA512.npy"
FPS = 2
HALF_WINDOW = 20           # same window as training: 40 rows = 20 s
STRIDE = 2                 # move the window 2 rows = 1 s at a time
NMS_S = 60                 # after a peak, ignore other peaks within ±60 s
TOLERANCES_S = [10, 30]    # a prediction is correct if within ± this of a real goal
THRESHOLDS = [0.3, 0.5, 0.557, 0.7, 0.9]
CACHE = Path("spotting_scores_valid.pkl")
SPLIT = "valid"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --- Part 1: load the trained model (same architecture as training) ---
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


# weights_only=False because the file also stores the numpy mean/std (our own file, safe)
ckpt = torch.load("goal_cnn_hard.pt", weights_only=False)
model = GoalCNN().to(device)
model.load_state_dict(ckpt["state"])
model.eval()
MEAN, STD = ckpt["mean"], ckpt["std"]


# --- Part 2: score timeline for one half ---
def score_half(features):
    """Returns (times_s, scores): one score per second, for the window centered there."""
    x = ((features - MEAN) / STD).astype(np.float32)
    # every 40-row window, already shaped (n_windows, 512, 40) like in training
    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * HALF_WINDOW, axis=0)[::STRIDE]
    scores = []
    with torch.no_grad():
        for i in range(0, len(windows), 512):
            xb = torch.from_numpy(np.ascontiguousarray(windows[i:i + 512])).to(device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    centers = np.arange(len(windows)) * STRIDE + HALF_WINDOW     # row at the window center
    return centers / FPS, np.concatenate(scores)


if CACHE.exists():
    timelines = pickle.loads(CACHE.read_bytes())
    print(f"Loaded cached scores from {CACHE}")
else:
    timelines = {}
    games = getListGames(split=SPLIT)
    for k, game in enumerate(games, 1):
        for half in (1, 2):
            timelines[(game, half)] = score_half(np.load(ROOT / game / f"{half}_{FEAT}"))
        print(f"\rscored {k}/{len(games)} games", end="")
    CACHE.write_bytes(pickle.dumps(timelines))
    print(f"\nSaved scores to {CACHE}")

# real goals per half, in seconds
goals = defaultdict(list)
with open("goals.csv") as f:
    for g in csv.DictReader(f):
        if g["split"] == SPLIT:
            goals[(g["game"], int(g["half"]))].append(int(g["position_ms"]) / 1000)
n_goals = sum(len(v) for v in goals.values())
n_matches = len(timelines) // 2
print(f"{n_matches} matches, {n_goals} goals")


# --- Part 3: peak picking (non-maximum suppression) ---
def nms(times, scores, min_score=0.01):
    """Repeatedly take the highest score and suppress everything within ±NMS_S."""
    picked = []
    for i in np.argsort(-scores):
        if scores[i] < min_score:
            break
        if all(abs(times[i] - t) > NMS_S for t, _ in picked):
            picked.append((times[i], scores[i]))
    return picked                                # sorted by score, highest first


predictions = {key: nms(*tl) for key, tl in timelines.items()}


# --- Part 4: match predictions to real goals ---
def match(tol):
    """Label every prediction TP/FP (best scores first). Returns rows (score, is_tp, offset_s)."""
    rows = []
    for key, preds in predictions.items():
        unmatched = list(goals.get(key, []))
        for t, s in preds:
            dists = [abs(t - g) for g in unmatched]
            if dists and min(dists) <= tol:
                g = unmatched.pop(int(np.argmin(dists)))
                rows.append((s, True, t - g))     # offset > 0 = predicted after the goal
            else:
                rows.append((s, False, None))
    return rows


def average_precision(rows):
    """Detection AP: precision averaged at each correct prediction, over all real goals."""
    rows = sorted(rows, key=lambda r: -r[0])
    tp_cum = np.cumsum([r[1] for r in rows])
    precision = tp_cum / np.arange(1, len(rows) + 1)
    return sum(p for p, r in zip(precision, rows) if r[1]) / n_goals


# --- Part 5: metrics ---
for tol in TOLERANCES_S:
    rows = match(tol)
    print(f"\n=== tolerance ±{tol} s ===   AP = {average_precision(rows):.3f}")
    print(f"  {'threshold':>9} {'recall':>7} {'precision':>9} {'FA/match':>9} {'predictions':>11}")
    for thr in THRESHOLDS:
        kept = [r for r in rows if r[0] >= thr]
        tp = sum(r[1] for r in kept)
        fp = len(kept) - tp
        print(f"  {thr:>9.3f} {tp / n_goals:>7.3f} {tp / max(len(kept), 1):>9.3f} "
              f"{fp / n_matches:>9.2f} {len(kept):>11}")

# where are the peaks relative to the real goal? (correct predictions, ±30 s)
offsets = np.array([r[2] for r in match(30) if r[1] and r[0] >= 0.557])
print(f"\nLocalization (threshold 0.557, ±30 s): median |offset| = {np.median(np.abs(offsets)):.1f} s,"
      f" mean signed offset = {offsets.mean():+.1f} s (positive = after the goal)")

# are false alarms mostly right after real goals (celebrations / replays)?
fa_after_goal, fa_total = 0, 0
for key, preds in predictions.items():
    for t, s in preds:
        if s < 0.557 or any(abs(t - g) <= 10 for g in goals.get(key, [])):
            continue                                 # not a false alarm at ±10 s
        fa_total += 1
        if any(10 < t - g <= 120 for g in goals.get(key, [])):
            fa_after_goal += 1
print(f"False alarms (threshold 0.557, ±10 s): {fa_total}, of which "
      f"{fa_after_goal} fall 10–120 s after a real goal")