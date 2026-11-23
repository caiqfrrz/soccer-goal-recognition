"""Step 5: 1D-CNN over time, several seeds, precision/recall/F1 at threshold 0.5."""
import numpy as np

from goalspot import config as C
from goalspot.data import fit_standardizer, load_windows, standardize
from goalspot.evaluation import prf, summarize_runs
from goalspot.training import predict_proba, to_pred, train_cnn

SEEDS = [1, 2, 3, 4, 5]
device = C.get_device()
print("Device:", device)

X_train, y_train, _ = load_windows("train")
X_valid, y_valid, _ = load_windows("valid")
mean, std = fit_standardizer(X_train)
X_train_t, X_valid_t = standardize(X_train, mean, std), standardize(X_valid, mean, std)

runs = []
for seed in SEEDS:
    model = train_cnn(X_train_t, y_train, seed, device)
    runs.append(prf(y_valid, to_pred(predict_proba(model, X_valid_t, device))))
    print(f"seed {seed}:  " + summarize_runs(runs[-1:]))

print(f"\nOver {len(SEEDS)} seeds (mean ± std):  {summarize_runs(runs)}")
