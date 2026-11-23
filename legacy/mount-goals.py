"""
Step 3b: same dataset as step 3, plus HARD negatives (windows centered on
shots and corners). Saved to ./dataset_hard/ so the old ./dataset/ stays
available for comparison.
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from SoccerNet.utils import getListGames

ROOT = Path("./SoccerNet")
OUT = Path("./dataset_hard")
FEAT = "ResNET_TF2_PCA512.npy"
FPS = 2

HALF_WINDOW = 20
EXCLUDE_BEFORE = 10 * FPS
EXCLUDE_AFTER = 60 * FPS
NEG_PER_HALF = 8
HARD_LABELS = {"Shots on target", "Shots off target", "Corner"}   # from the error analysis
SPLITS = ["train", "valid", "test"]

random.seed(42)   # same seed and same order as step 3 -> identical random negatives

# --- Part 1: goals per half (same as step 3) ---
goals_by_half = defaultdict(list)
with open("goals.csv") as f:
    for g in csv.DictReader(f):
        row = int(int(g["position_ms"]) / 1000 * FPS)
        goals_by_half[(g["split"], g["game"], int(g["half"]))].append(row)


# --- Part 2: hard-negative events per half, read from Labels-v2.json ---
def hard_events(game, half):
    anns = json.loads((ROOT / game / "Labels-v2.json").read_text())["annotations"]
    return [(int(int(a["position"]) / 1000 * FPS), a["label"]) for a in anns
            if a["label"] in HARD_LABELS and int(a["gameTime"][0]) == half]


def build_split(split):
    X, y, meta = [], [], []
    skipped_goals = 0
    hard_count, hard_rejected = Counter(), 0

    for game in getListGames(split=split):
        for half in (1, 2):
            features = np.load(ROOT / game / f"{half}_{FEAT}")
            n = len(features)
            goal_rows = goals_by_half.get((split, game, half), [])

            # positives
            for r in goal_rows:
                if r - HALF_WINDOW < 0 or r + HALF_WINDOW > n:
                    skipped_goals += 1
                    continue
                X.append(features[r - HALF_WINDOW:r + HALF_WINDOW].copy())
                y.append(1)
                meta.append([game, half, r, 1, "goal", "Goal"])

            # forbidden zone around goals
            forbidden = np.zeros(n, dtype=bool)
            for r in goal_rows:
                forbidden[max(0, r - EXCLUDE_BEFORE - HALF_WINDOW):
                          min(n, r + EXCLUDE_AFTER + HALF_WINDOW)] = True

            # random negatives (unchanged -> same windows as ./dataset/)
            candidates = [c for c in range(HALF_WINDOW, n - HALF_WINDOW) if not forbidden[c]]
            for c in random.sample(candidates, min(NEG_PER_HALF, len(candidates))):
                X.append(features[c - HALF_WINDOW:c + HALF_WINDOW].copy())
                y.append(0)
                meta.append([game, half, c, 0, "random", ""])

            # NEW: hard negatives, one window centered on each shot / corner
            for c, label in hard_events(game, half):
                if c - HALF_WINDOW < 0 or c + HALF_WINDOW > n or forbidden[c]:
                    hard_rejected += 1      # too close to a goal, or past the file end
                    continue
                X.append(features[c - HALF_WINDOW:c + HALF_WINDOW].copy())
                y.append(0)
                meta.append([game, half, c, 0, "hard", label])
                hard_count[label] += 1

    return np.stack(X), np.array(y), meta, skipped_goals, hard_count, hard_rejected


# --- Part 3: build and save ---
OUT.mkdir(exist_ok=True)
for split in SPLITS:
    X, y, meta, skipped, hard_count, rejected = build_split(split)
    np.save(OUT / f"X_{split}.npy", X)
    np.save(OUT / f"y_{split}.npy", y)
    with open(OUT / f"meta_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "half", "center_row", "label", "kind", "event"])
        w.writerows(meta)
    print(f"{split:5}: X={X.shape}  goals={int(y.sum())}  "
          f"random_neg={sum(m[4] == 'random' for m in meta)}  "
          f"hard_neg={sum(hard_count.values())} {dict(hard_count)}  "
          f"hard_rejected={rejected}  skipped_goals={skipped}")