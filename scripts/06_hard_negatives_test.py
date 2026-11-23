"""
STEP 6 - Do the "hard" windows (shots, corners) actually help?

Question:  is the model better if we also train it on shots and corners?
Reads:     data/windows/
Writes:    nothing

Experiment: train the CNN twice and test both on the same valid set.
  1) WITHOUT hard windows (goal + random only)
  2) WITH hard windows    (goal + random + hard)
"false alarms on hard" = how often a shot/corner window is called a goal.
"""
import numpy as np
from sklearn.metrics import average_precision_score

from common import load_windows, normalize, predict, train_cnn

X_all, y_all, info_all = load_windows("train")
X_valid, y_valid, info_valid = load_windows("valid")
valid_kind = np.array([w["kind"] for w in info_valid])
train_kind = np.array([w["kind"] for w in info_all])

for name, keep in [("WITHOUT hard windows", train_kind != "hard"),
                   ("WITH hard windows", np.ones(len(y_all), dtype=bool))]:
    X_train, y_train = X_all[keep], y_all[keep]
    mean, std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)) + 1e-6
    X_train_n, X_valid_n = normalize(X_train, mean, std), normalize(X_valid, mean, std)

    print(f"\nTraining {name}: {len(X_train)} windows")
    for seed in [1, 2, 3]:
        prob = predict(train_cnn(X_train_n, y_train, seed), X_valid_n)
        said_goal = prob > 0.5
        print(f"  seed {seed}:  AP={average_precision_score(y_valid, prob):.3f}  "
              f"goals found={said_goal[y_valid == 1].mean():.3f}  "
              f"false alarms on random={said_goal[valid_kind == 'random'].mean():.3f}  "
              f"on hard={said_goal[valid_kind == 'hard'].mean():.3f}")
