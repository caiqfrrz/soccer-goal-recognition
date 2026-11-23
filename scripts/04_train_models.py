"""
STEP 4 - Train two models on the windows and compare them.

Question:  given a 20 s window, can a model tell if it contains a goal?
Reads:     data/windows/
Writes:    outputs/models/goal_cnn_hard_<FEATURES>.pt   (the CNN, used by step 7)
           outputs/figures/pr_curve_<FEATURES>.png

Two models:
  A) Logistic regression - averages the 40 rows into one row, so it ignores
     WHEN things happen inside the window. Simple baseline.
  B) 1D-CNN - looks at the 40 rows in order. Trained 3 times (seeds 1, 2, 3)
     because results change a bit with the random start.

How to read the numbers (evaluated on "valid", data never used for training):
  precision = of the windows the model called "goal", how many really were
  recall    = of the real goals, how many the model found
  F1        = one number combining both
  AP        = average precision over ALL thresholds (not just 0.5).
              1.0 = perfect, ~0.08 = random guessing on this data.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, OUTPUTS, load_windows, normalize, predict, save_model, train_cnn

X_train, y_train, _ = load_windows("train")
X_valid, y_valid, _ = load_windows("valid")
print(f"train: {len(X_train)} windows ({y_train.sum()} goals)")
print(f"valid: {len(X_valid)} windows ({y_valid.sum()} goals)\n")


def show_scores(name, prob):
    """Print precision/recall/F1 at threshold 0.5 and AP."""
    pred = (prob > 0.5).astype(int)
    print(f"{name:22}  precision={precision_score(y_valid, pred):.3f}  "
          f"recall={recall_score(y_valid, pred):.3f}  F1={f1_score(y_valid, pred):.3f}  "
          f"AP={average_precision_score(y_valid, prob):.3f}")


# --- A) logistic regression on the time-averaged window ---
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
logreg.fit(X_train.mean(axis=1), y_train)
logreg_prob = logreg.predict_proba(X_valid.mean(axis=1))[:, 1]
show_scores("Logistic regression", logreg_prob)

# --- B) 1D-CNN ---
mean = X_train.mean(axis=(0, 1))                 # normalization numbers come from TRAIN only
std = X_train.std(axis=(0, 1)) + 1e-6
X_train_n = normalize(X_train, mean, std)
X_valid_n = normalize(X_valid, mean, std)

cnn_probs = []
for seed in [1, 2, 3]:
    model = train_cnn(X_train_n, y_train, seed)
    prob = predict(model, X_valid_n)
    cnn_probs.append(prob)
    show_scores(f"1D-CNN seed {seed}", prob)
    if seed == 1:
        save_model(model, mean, std)              # this one is used for spotting (step 7)

aps = [average_precision_score(y_valid, p) for p in cnn_probs]
print(f"\n1D-CNN AP over 3 seeds: {np.mean(aps):.3f} ± {np.std(aps):.3f}")

# --- precision-recall curve: what you get at every threshold ---
plt.figure(figsize=(6, 5))
for name, prob in [("Logistic regression", logreg_prob), ("1D-CNN seed 1", cnn_probs[0])]:
    precision, recall, _ = precision_recall_curve(y_valid, prob)
    plt.plot(recall, precision, label=f"{name} (AP {average_precision_score(y_valid, prob):.2f})")
plt.axhline(y_valid.mean(), linestyle="--", color="gray", label="random guess")
plt.xlabel("Recall (goals found)")
plt.ylabel("Precision (alarms that are goals)")
plt.title(f"Goal windows, valid set ({FEATURES})")
plt.legend()
plt.grid(alpha=0.3)
out = OUTPUTS / "figures" / f"pr_curve_{FEATURES}.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
