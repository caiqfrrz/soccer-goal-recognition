"""Metrics for window classification and for spotting over full halves."""
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score


# --- window classification ---------------------------------------------------
def prf(y_true, pred):
    return {"P": precision_score(y_true, pred), "R": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred)}


def summarize_runs(runs):
    """[{metric: value}, ...] over seeds -> 'P=0.812±0.010  R=...' string."""
    keys = runs[0].keys()
    return "  ".join(f"{k}={np.mean([r[k] for r in runs]):.3f}"
                     + (f"±{np.std([r[k] for r in runs]):.3f}" if len(runs) > 1 else "")
                     for k in keys)


def operating_points(y_true, prob, target_recalls=(0.70, 0.80, 0.90)):
    """Best precision reachable at each target recall, plus the max-F1 point.
    Values are (precision, recall, threshold[, f1])."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    precision, recall = precision[:-1], recall[:-1]          # align with thresholds
    out = {}
    for target in target_recalls:
        i = np.argmax(np.where(recall >= target, precision, -1))
        out[f"R>={target:.2f}"] = (precision[i], recall[i], thresholds[i])
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    i = np.argmax(f1)
    out["max F1"] = (precision[i], recall[i], thresholds[i], f1[i])
    return out


# --- spotting ----------------------------------------------------------------
def nms(times, scores, radius_s, min_score=0.01):
    """Repeatedly take the highest score and suppress everything within ±radius_s.
    Returns [(time_s, score)] sorted by score, highest first."""
    picked = []
    for i in np.argsort(-scores):
        if scores[i] < min_score:
            break
        if all(abs(times[i] - t) > radius_s for t, _ in picked):
            picked.append((times[i], scores[i]))
    return picked


def match_predictions(predictions, goals, tol_s):
    """predictions: {half_key: [(t, score)]}, goals: {half_key: [t]}.
    Greedy matching, best scores first. Returns rows (score, is_tp, offset_s)."""
    rows = []
    for key, preds in predictions.items():
        unmatched = list(goals.get(key, []))
        for t, s in preds:
            dists = [abs(t - g) for g in unmatched]
            if dists and min(dists) <= tol_s:
                g = unmatched.pop(int(np.argmin(dists)))
                rows.append((s, True, t - g))     # offset > 0 = predicted after the goal
            else:
                rows.append((s, False, None))
    return rows


def detection_ap(rows, n_goals):
    """Precision averaged at each correct prediction, over all real goals."""
    rows = sorted(rows, key=lambda r: -r[0])
    tp_cum = np.cumsum([r[1] for r in rows])
    precision = tp_cum / np.arange(1, len(rows) + 1)
    return sum(p for p, r in zip(precision, rows) if r[1]) / n_goals
