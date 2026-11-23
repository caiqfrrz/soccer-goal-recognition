"""Step 8: threshold-free evaluation (PR curve + average precision) on valid,
for models trained with hard negatives. Saves the seed-1 CNN for spotting:
outputs/models/goal_cnn_hard_<FEATURE_SET>.pt
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from goalspot import config as C
from goalspot.data import fit_standardizer, load_windows, standardize
from goalspot.evaluation import operating_points
from goalspot.models import make_logreg, save_checkpoint
from goalspot.training import predict_proba, train_cnn

SEEDS = [1, 2, 3]
device = C.get_device()
C.ensure_output_dirs()

X_tr, y_tr, _ = load_windows("train")
X_va, y_va, _ = load_windows("valid")
print(f"valid: {int(y_va.sum())} goals, {int((y_va == 0).sum())} negatives "
      f"(a random guess would have AP ≈ {y_va.mean():.3f})")


def report(name, probs):
    aps = [average_precision_score(y_va, p) for p in probs]
    std = f" ± {np.std(aps):.3f}" if len(aps) > 1 else ""
    print(f"\n{name}:  AP = {np.mean(aps):.3f}{std}")
    points = [operating_points(y_va, p) for p in probs]
    for key in points[0]:
        P, R, T = (np.mean([pt[key][j] for pt in points]) for j in range(3))
        extra = f"  F1={np.mean([pt[key][3] for pt in points]):.3f}" if key == "max F1" else ""
        print(f"  {key:8}  precision={P:.3f}  recall={R:.3f}  threshold={T:.3f}{extra}")


# --- logistic regression ---
lr_prob = make_logreg().fit(X_tr.mean(axis=1), y_tr).predict_proba(X_va.mean(axis=1))[:, 1]
report("Logistic regression", [lr_prob])

# --- 1D-CNN ---
mean, std = fit_standardizer(X_tr)
X_tr_t, X_va_t = standardize(X_tr, mean, std), standardize(X_va, mean, std)
del X_tr
cnn_probs = []
for seed in SEEDS:
    model = train_cnn(X_tr_t, y_tr, seed, device)
    cnn_probs.append(predict_proba(model, X_va_t, device))
    if seed == SEEDS[0]:
        path = C.MODELS_DIR / f"goal_cnn_hard_{C.FEATURE_SET}.pt"
        save_checkpoint(path, model, mean, std, feature_set=C.FEATURE_SET, seed=seed)
        print("Saved", path)
report(f"1D-CNN ({len(SEEDS)} seeds)", cnn_probs)

# --- plot ---
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
for label, prob in [("Logistic regression", lr_prob), ("1D-CNN, seed 1", cnn_probs[0])]:
    p, r, _ = precision_recall_curve(y_va, prob)
    plt.plot(r, p, label=f"{label} (AP {average_precision_score(y_va, prob):.2f})")
plt.axhline(y_va.mean(), linestyle="--", color="gray", label="random guess")
plt.xlabel("Recall (goals found)")
plt.ylabel("Precision (alarms that are goals)")
plt.title(f"Goal detection, valid ({C.FEATURE_SET})")
plt.legend()
plt.grid(alpha=0.3)
out = C.FIGURES_DIR / f"pr_curve_{C.FEATURE_SET}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("\nSaved", out)
