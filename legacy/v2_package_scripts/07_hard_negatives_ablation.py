"""Step 7: does training with hard negatives help?
Trains on train WITHOUT the hard rows and on the full train set, and evaluates
both on the same valid set (which contains hard negatives).
"""
import numpy as np

from goalspot import config as C
from goalspot.data import fit_standardizer, load_windows, standardize
from goalspot.evaluation import prf, summarize_runs
from goalspot.models import make_logreg
from goalspot.training import predict_proba, to_pred, train_cnn

SEEDS = [1, 2, 3]
device = C.get_device()

X_valid, y_valid, meta_valid = load_windows("valid")
kind_valid = np.array([m["kind"] for m in meta_valid])          # goal / random / hard


def scores(pred):
    """P/R/F1 on goals + false-alarm rate on each negative type."""
    return {**prf(y_valid, pred),
            "FA_random": pred[kind_valid == "random"].mean(),
            "FA_hard": pred[kind_valid == "hard"].mean()}


X_full, y_full, meta_train = load_windows("train", mmap=True)   # read rows only when needed
kind_train = np.array([m["kind"] for m in meta_train])

for name, mask in [("random negatives only", kind_train != "hard"),
                   ("random + hard negatives", np.ones(len(y_full), dtype=bool))]:
    X_tr, y_tr = np.asarray(X_full[mask]), y_full[mask]
    print(f"\nTraining on {name}: {len(X_tr)} windows "
          f"({int(y_tr.sum())} goals, {int((y_tr == 0).sum())} negatives)")

    lr = make_logreg().fit(X_tr.mean(axis=1), y_tr)
    print(f"  {'logistic regression':30} {summarize_runs([scores(lr.predict(X_valid.mean(axis=1)))])}")

    mean, std = fit_standardizer(X_tr)              # standardize with THIS training set
    X_tr_t, X_va_t = standardize(X_tr, mean, std), standardize(X_valid, mean, std)
    del X_tr
    runs = [scores(to_pred(predict_proba(train_cnn(X_tr_t, y_tr, s, device), X_va_t, device)))
            for s in SEEDS]
    print(f"  {f'1D-CNN ({len(SEEDS)} seeds)':30} {summarize_runs(runs)}")
