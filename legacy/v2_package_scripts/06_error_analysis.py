"""Step 6: what does the CNN get wrong?
  - missed goals, by visibility (goal shown on camera or not)
  - false alarms, by which annotated events are inside the window
Every error is saved to outputs/reports/errors_valid_<FEATURE_SET>.csv
"""
import csv
from collections import Counter

from goalspot import config as C
from goalspot.data import (fit_standardizer, load_annotations, load_goals, load_windows,
                           ms_to_row, row_to_clock, standardize)
from goalspot.training import predict_proba, to_pred, train_cnn

SEED = 1
RADIUS_S = 10                      # "inside the window" = within ±10 s of its center
device = C.get_device()
C.ensure_output_dirs()

# --- train one CNN, score valid ---
X_train, y_train, _ = load_windows("train")
X_valid, y_valid, meta = load_windows("valid")
mean, std = fit_standardizer(X_train)
model = train_cnn(standardize(X_train, mean, std), y_train, SEED, device)
prob = predict_proba(model, standardize(X_valid, mean, std), device)
pred = to_pred(prob)

# --- attach metadata to every valid window ---
visibility = {(g["game"], int(g["half"]), ms_to_row(g["position_ms"])): g["visibility"]
              for g in load_goals("valid")}


def events_in_window(game, half, center_row):
    center_ms = center_row / C.FPS * 1000
    return sorted({lab for h, ms, lab in load_annotations(game)
                   if h == half and abs(ms - center_ms) <= RADIUS_S * 1000})


rows = []
for m, y, p, pr in zip(meta, y_valid, pred, prob):
    game, half, c = m["game"], int(m["half"]), int(m["center_row"])
    rows.append({
        "kind": {(1, 1): "TP", (1, 0): "missed goal", (0, 1): "false alarm", (0, 0): "TN"}[(y, p)],
        "window_kind": m["kind"], "game": game, "half": half, "time": row_to_clock(c),
        "prob": round(float(pr), 3),
        "visibility": visibility.get((game, half, c), "") if y == 1 else "",
        "events_in_window": "; ".join(events_in_window(game, half, c)),
    })

# --- missed goals vs visibility ---
goals = [r for r in rows if r["kind"] in ("TP", "missed goal")]
print("Recall by goal visibility:")
for vis, n in Counter(r["visibility"] for r in goals).items():
    found = sum(r["kind"] == "TP" for r in goals if r["visibility"] == vis)
    print(f"  {vis:10}  {found}/{n} found  (recall {found / n:.2f})")

# --- false alarms vs events in the window ---
negatives = [r for r in rows if r["kind"] in ("TN", "false alarm")]
false_alarms = [r for r in negatives if r["kind"] == "false alarm"]
print(f"\nFalse alarms: {len(false_alarms)} of {len(negatives)} negatives")
for wk in ("random", "hard"):
    sub = [r for r in negatives if r["window_kind"] == wk]
    if sub:
        print(f"  {wk:6} negatives: FA rate {sum(r['kind'] == 'false alarm' for r in sub) / len(sub):.3f}")
count_all = Counter(e for r in negatives for e in r["events_in_window"].split("; ") if e)
count_fa = Counter(e for r in false_alarms for e in r["events_in_window"].split("; ") if e)
print(f"\n  {'event within ±10 s':25} {'in false alarms':>16} {'in all negatives':>17} {'FA rate':>8}")
for event, n_all in count_all.most_common():
    print(f"  {event:25} {count_fa[event]:>16} {n_all:>17} {count_fa[event] / n_all:>8.2f}")
no_ev_fa = sum(1 for r in false_alarms if not r["events_in_window"])
no_ev_all = sum(1 for r in negatives if not r["events_in_window"])
print(f"  {'(no annotated event)':25} {no_ev_fa:>16} {no_ev_all:>17} {no_ev_fa / max(no_ev_all, 1):>8.2f}")

# --- save every error ---
errors = sorted([r for r in rows if r["kind"] in ("missed goal", "false alarm")],
                key=lambda r: (r["kind"], -r["prob"]))
out = C.REPORTS_DIR / f"errors_valid_{C.FEATURE_SET}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(errors)
print(f"\nSaved {len(errors)} errors to {out}")
