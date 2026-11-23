"""Step 9: goal SPOTTING over full halves.

1. score every second of every half with the trained CNN (cached)
2. peak picking (non-maximum suppression) -> predicted goal times
3. match predictions to real goals within a tolerance
4. metrics: AP, recall, precision, false alarms per match, localization offset

Delete the cache file (or pass --rescore) after retraining the model.
"""
import argparse
import pickle
from collections import defaultdict

import numpy as np
from SoccerNet.utils import getListGames

from goalspot import config as C
from goalspot.data import load_goals, load_half_features
from goalspot.evaluation import detection_ap, match_predictions, nms
from goalspot.models import load_checkpoint
from goalspot.spotting import score_half

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="valid")
parser.add_argument("--rescore", action="store_true", help="ignore the cached scores")
args = parser.parse_args()

STRIDE = 2                 # window moves 2 rows = 1 s at a time
NMS_S = 60                 # after a peak, ignore other peaks within ±60 s
TOLERANCES_S = [10, 30]    # a prediction is correct if within ± this of a real goal
THRESHOLDS = [0.3, 0.5, 0.557, 0.7, 0.9]
OP_THRESHOLD = 0.557       # max-F1 threshold of the CNN from step 8
MODEL = C.MODELS_DIR / f"goal_cnn_hard_{C.FEATURE_SET}.pt"
CACHE = C.CACHE_DIR / f"spotting_scores_{args.split}_{C.FEATURE_SET}.pkl"
device = C.get_device()
C.ensure_output_dirs()

# --- 1. score timelines ---
if CACHE.exists() and not args.rescore:
    timelines = pickle.loads(CACHE.read_bytes())
    print(f"Loaded cached scores from {CACHE.relative_to(C.PROJECT_ROOT)}")
else:
    model, mean, std = load_checkpoint(MODEL, device)
    timelines = {}
    games = getListGames(split=args.split)
    for k, game in enumerate(games, 1):
        for half in (1, 2):
            timelines[(game, half)] = score_half(model, load_half_features(game, half),
                                                 mean, std, device, stride=STRIDE)
        print(f"\rscored {k}/{len(games)} games", end="")
    CACHE.write_bytes(pickle.dumps(timelines))
    print(f"\nSaved scores to {CACHE}")

goals = defaultdict(list)                      # real goal times (s) per half
for g in load_goals(args.split):
    goals[(g["game"], int(g["half"]))].append(int(g["position_ms"]) / 1000)
n_goals = sum(len(v) for v in goals.values())
n_matches = len(timelines) // 2
print(f"{n_matches} matches, {n_goals} goals")

# --- 2. peak picking ---
predictions = {key: nms(t, s, NMS_S) for key, (t, s) in timelines.items()}

# --- 3/4. matching + metrics ---
for tol in TOLERANCES_S:
    rows = match_predictions(predictions, goals, tol)
    print(f"\n=== tolerance ±{tol} s ===   AP = {detection_ap(rows, n_goals):.3f}")
    print(f"  {'threshold':>9} {'recall':>7} {'precision':>9} {'FA/match':>9} {'predictions':>11}")
    for thr in THRESHOLDS:
        kept = [r for r in rows if r[0] >= thr]
        tp = sum(r[1] for r in kept)
        fp = len(kept) - tp
        print(f"  {thr:>9.3f} {tp / n_goals:>7.3f} {tp / max(len(kept), 1):>9.3f} "
              f"{fp / n_matches:>9.2f} {len(kept):>11}")

# where are the peaks relative to the real goal? (correct predictions, ±30 s)
offsets = np.array([r[2] for r in match_predictions(predictions, goals, 30)
                    if r[1] and r[0] >= OP_THRESHOLD])
print(f"\nLocalization (threshold {OP_THRESHOLD}, ±30 s): median |offset| = "
      f"{np.median(np.abs(offsets)):.1f} s, mean signed offset = {offsets.mean():+.1f} s "
      f"(positive = after the goal)")

# are false alarms mostly right after real goals (celebrations / replays)?
fa_after_goal, fa_total = 0, 0
for key, preds in predictions.items():
    for t, s in preds:
        if s < OP_THRESHOLD or any(abs(t - g) <= 10 for g in goals.get(key, [])):
            continue                                 # not a false alarm at ±10 s
        fa_total += 1
        fa_after_goal += any(10 < t - g <= 120 for g in goals.get(key, []))
print(f"False alarms (threshold {OP_THRESHOLD}, ±10 s): {fa_total}, of which "
      f"{fa_after_goal} fall 10–120 s after a real goal")
