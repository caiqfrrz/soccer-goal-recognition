"""
STEP 5 - Look at the mistakes of the CNN.

Question:  which goals does it miss, and what fools it into false alarms?
Reads:     data/windows/, labels
Writes:    outputs/reports/errors_valid_<FEATURES>.csv (every mistake, to inspect by hand)

Two kinds of mistakes:
  missed goal  - a goal window the model scored below 0.5
  false alarm  - a non-goal window the model scored above 0.5
For each false alarm we list which events (foul, corner ...) happen within
±10 s, to see what the model confuses with a goal.
"""
import csv
from collections import Counter

from common import FEATURES, FPS, OUTPUTS, load_events, load_windows, normalize, predict, train_cnn

# --- train one CNN and score every valid window ---
X_train, y_train, _ = load_windows("train")
X_valid, y_valid, info = load_windows("valid")
mean, std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)) + 1e-6
model = train_cnn(normalize(X_train, mean, std), y_train, seed=1)
probs = predict(model, normalize(X_valid, mean, std))

# --- describe every window ---
events_cache = {}
rows = []
for window, label, prob in zip(info, y_valid, probs):
    game, half = window["game"], int(window["half"])
    center_s = int(window["center_row"]) / FPS
    said_goal = prob > 0.5

    if label == 1:
        result = "found goal" if said_goal else "missed goal"
    else:
        result = "false alarm" if said_goal else "correct no-goal"

    if game not in events_cache:
        events_cache[game] = load_events(game)
    nearby = sorted({e["label"] for e in events_cache[game]
                     if e["half"] == half and abs(e["seconds"] - center_s) <= 10})

    rows.append({"result": result, "window_kind": window["kind"], "game": game, "half": half,
                 "time": f"{int(center_s // 60):02d}:{int(center_s % 60):02d}",
                 "prob": round(float(prob), 3), "events_nearby": "; ".join(nearby)})

# --- summary ---
print("Results on valid:")
for result, count in Counter(r["result"] for r in rows).most_common():
    print(f"  {result:16} {count}")

for kind in ["random", "hard"]:
    windows = [r for r in rows if r["window_kind"] == kind]
    alarms = [r for r in windows if r["result"] == "false alarm"]
    print(f"\n'{kind}' windows: {len(alarms)} of {len(windows)} were false alarms "
          f"({len(alarms) / len(windows):.1%})")

print("\nWhich events are near false alarms (and how often each event fools the model):")
negatives = [r for r in rows if r["window_kind"] != "goal"]
for event, total in Counter(e for r in negatives for e in r["events_nearby"].split("; ") if e).most_common():
    fooled = sum(1 for r in negatives if r["result"] == "false alarm" and event in r["events_nearby"])
    print(f"  {event:22} near {total:5} non-goal windows, {fooled:4} false alarms ({fooled / total:.0%})")

# --- save mistakes ---
mistakes = [r for r in rows if r["result"] in ("missed goal", "false alarm")]
mistakes.sort(key=lambda r: (r["result"], -r["prob"]))
out = OUTPUTS / "reports" / f"errors_valid_{FEATURES}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(mistakes)
print(f"\nSaved {len(mistakes)} mistakes to {out}")
