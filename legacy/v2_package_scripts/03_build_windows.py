"""Step 3: cut 40-row (20 s) windows from every half -> data/windows/<FEATURE_SET>/

Window kinds (column "kind" in meta_<split>.csv):
  goal   - centered on a goal                       (label 1)
  random - random position far from any goal        (label 0)
  hard   - centered on a shot or corner, which the  (label 0)
           error analysis showed fool the model
Training on "random only" = drop the rows with kind == "hard".
"""
import csv
import random
from collections import Counter

import numpy as np
from SoccerNet.utils import getListGames

from goalspot import config as C
from goalspot.data import goal_rows_by_half, load_annotations, load_half_features, windows_dir

HW = C.HALF_WINDOW
EXCLUDE_BEFORE = 10 * C.FPS        # no negatives from 10 s before a goal ...
EXCLUDE_AFTER = 60 * C.FPS         # ... to 60 s after it (celebration, replays)
NEG_PER_HALF = 8
HARD_LABELS = {"Shots on target", "Shots off target", "Corner"}

random.seed(42)   # same seed and order as the original build -> identical random negatives
goals_by_half = goal_rows_by_half()


def hard_events(game, half):
    return [(int(ms / 1000 * C.FPS), label) for h, ms, label in load_annotations(game)
            if label in HARD_LABELS and h == half]


def build_split(split):
    X, y, meta = [], [], []
    skipped_goals, hard_rejected, hard_count = 0, 0, Counter()

    for game in getListGames(split=split):
        for half in (1, 2):
            features = load_half_features(game, half)
            n = len(features)
            goal_rows = goals_by_half.get((split, game, half), [])

            for r in goal_rows:                                          # positives
                if r - HW < 0 or r + HW > n:
                    skipped_goals += 1
                    continue
                X.append(features[r - HW:r + HW].copy())
                y.append(1)
                meta.append([game, half, r, 1, "goal", "Goal"])

            forbidden = np.zeros(n, dtype=bool)                          # zone around goals
            for r in goal_rows:
                forbidden[max(0, r - EXCLUDE_BEFORE - HW):min(n, r + EXCLUDE_AFTER + HW)] = True

            candidates = [c for c in range(HW, n - HW) if not forbidden[c]]   # random negatives
            for c in random.sample(candidates, min(NEG_PER_HALF, len(candidates))):
                X.append(features[c - HW:c + HW].copy())
                y.append(0)
                meta.append([game, half, c, 0, "random", ""])

            for c, label in hard_events(game, half):                     # hard negatives
                if c - HW < 0 or c + HW > n or forbidden[c]:
                    hard_rejected += 1
                    continue
                X.append(features[c - HW:c + HW].copy())
                y.append(0)
                meta.append([game, half, c, 0, "hard", label])
                hard_count[label] += 1

    return np.stack(X), np.array(y), meta, skipped_goals, hard_count, hard_rejected


out = windows_dir()
out.mkdir(parents=True, exist_ok=True)
for split in C.SPLITS:
    X, y, meta, skipped, hard_count, rejected = build_split(split)
    np.save(out / f"X_{split}.npy", X)
    np.save(out / f"y_{split}.npy", y)
    with open(out / f"meta_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "half", "center_row", "label", "kind", "event"])
        w.writerows(meta)
    print(f"{split:5}: X={X.shape}  goals={int(y.sum())}  "
          f"random_neg={sum(m[4] == 'random' for m in meta)}  "
          f"hard_neg={sum(hard_count.values())} {dict(hard_count)}  "
          f"hard_rejected={rejected}  skipped_goals={skipped}")
print("Saved to", out)
