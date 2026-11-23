"""
STEP 3 - Cut the matches into 20-second windows to train on.

Question:  what examples do we show the model?
Reads:     data/goals.csv, features, labels
Writes:    data/windows/<FEATURES>/X_<split>.npy  (N, 40, 512) the windows
                                   y_<split>.npy  (N,)  1 = goal, 0 = no goal
                                   meta_<split>.csv     where each window comes from

Three kinds of windows:
  "goal"   centered on a goal                                   -> label 1
  "random" a random moment, far from any goal (8 per half)      -> label 0
  "hard"   centered on a shot or a corner. These look a lot     -> label 0
           like goals, so they teach the model the difference.
"""
import csv
import random

import numpy as np
from SoccerNet.utils import getListGames

from common import FPS, HALF_WINDOW, WINDOWS, load_events, load_features, load_goals, seconds_to_row

RANDOM_PER_HALF = 8
HARD_EVENTS = ["Shots on target", "Shots off target", "Corner"]
# Random/hard windows may not come from 10 s before a goal to 60 s after it
# (that time is full of celebrations and replays of the goal).
NO_NEGATIVES_BEFORE = 10 * FPS
NO_NEGATIVES_AFTER = 60 * FPS

random.seed(42)    # always pick the same random windows


def cut_window(features, center):
    """The 40 rows around `center`, or None if it would go past the start/end."""
    if center - HALF_WINDOW < 0 or center + HALF_WINDOW > len(features):
        return None
    return features[center - HALF_WINDOW:center + HALF_WINDOW].copy()


def make_split(split):
    X, y, meta = [], [], []
    goals = load_goals(split)

    for game in getListGames(split=split):
        events = load_events(game)

        for half in (1, 2):
            features = load_features(game, half)
            n = len(features)
            goal_rows = [seconds_to_row(g["seconds"]) for g in goals
                         if g["game"] == game and g["half"] == half]

            # 1) goal windows
            for row in goal_rows:
                window = cut_window(features, row)
                if window is not None:
                    X.append(window); y.append(1); meta.append([game, half, row, 1, "goal", "Goal"])

            # mark the rows where negative windows are not allowed
            forbidden = np.zeros(n, dtype=bool)
            for row in goal_rows:
                start = max(0, row - NO_NEGATIVES_BEFORE - HALF_WINDOW)
                end = min(n, row + NO_NEGATIVES_AFTER + HALF_WINDOW)
                forbidden[start:end] = True

            # 2) random windows
            allowed = [c for c in range(HALF_WINDOW, n - HALF_WINDOW) if not forbidden[c]]
            for row in random.sample(allowed, min(RANDOM_PER_HALF, len(allowed))):
                X.append(cut_window(features, row)); y.append(0); meta.append([game, half, row, 0, "random", ""])

            # 3) hard windows: shots and corners
            for event in events:
                if event["half"] != half or event["label"] not in HARD_EVENTS:
                    continue
                row = seconds_to_row(event["seconds"])
                window = cut_window(features, row)
                if window is not None and not forbidden[row]:
                    X.append(window); y.append(0); meta.append([game, half, row, 0, "hard", event["label"]])

    return np.stack(X), np.array(y), meta


WINDOWS.mkdir(parents=True, exist_ok=True)
for split in ["train", "valid", "test"]:
    X, y, meta = make_split(split)
    np.save(WINDOWS / f"X_{split}.npy", X)
    np.save(WINDOWS / f"y_{split}.npy", y)
    with open(WINDOWS / f"meta_{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "half", "center_row", "label", "kind", "event"])
        writer.writerows(meta)

    kinds = [m[4] for m in meta]
    print(f"{split}: {len(X)} windows = {kinds.count('goal')} goal + "
          f"{kinds.count('random')} random + {kinds.count('hard')} hard")
print("Saved to", WINDOWS)
