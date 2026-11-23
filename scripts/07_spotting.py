"""
STEP 7 - Spotting: find the goals in FULL matches.

Question:  if we give the model a whole match, does it say when the goals are?
Reads:     the CNN from step 4, features of every valid game
Writes:    outputs/cache/spotting_scores_valid_<FEATURES>.pkl  (delete it after retraining!)

Until now the model only saw windows we cut for it. Real use is different:
  1. SCORE   - slide the window over the whole half, 1 second at a time,
               and get a goal probability for every second.
  2. PEAKS   - around a goal many seconds in a row score high. Keep only the
               highest point and ignore everything within 60 s of it.
               Each remaining peak is a "prediction": a time + a score.
  3. COMPARE - a prediction is correct if a real goal is within ±10 s (strict)
               or ±30 s (loose). Each real goal can be matched only once.

How to read the output:
  recall     = fraction of real goals found
  precision  = fraction of predictions that are real goals
  FA/match   = false alarms per match
  AP         = summary over all thresholds (1.0 = perfect)
  offset     = how many seconds after the real goal the peak is
"""
import pickle

import numpy as np
import torch
from SoccerNet.utils import getListGames

from common import DEVICE, FEATURES, FPS, HALF_WINDOW, OUTPUTS, load_features, load_goals, load_model

SPLIT = "valid"
PEAK_DISTANCE = 60          # seconds: keep only one peak per 60 s
THRESHOLD = 0.557           # best-F1 threshold found in step 4
CACHE = OUTPUTS / "cache" / f"spotting_scores_{SPLIT}_{FEATURES}.pkl"


# ---------------------------------------------------------------------------
# 1. SCORE every second of every half
# ---------------------------------------------------------------------------
def score_half(model, mean, std, features):
    """Returns (times, scores): a goal probability for each second of the half."""
    x =((features - mean) / std).astype(np.float32)
    centers = list(range(HALF_WINDOW, len(x) - HALF_WINDOW + 1, 2))   # every 2 rows = 1 s
    scores = []
    with torch.no_grad():
        for start in range(0, len(centers), 512):                     # 512 windows at a time
            batch = [x[c - HALF_WINDOW:c + HALF_WINDOW].T for c in centers[start:start + 512]]
            batch = torch.from_numpy(np.stack(batch)).to(DEVICE)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.array(centers) / FPS, np.concatenate(scores)


if CACHE.exists():
    print("Using saved scores from", CACHE.name, "(delete the file to score again)")
    scores = pickle.loads(CACHE.read_bytes())
else:
    model, mean, std = load_model()
    scores = {}                                   # (game, half) -> (times, scores)
    games = getListGames(split=SPLIT)
    for i, game in enumerate(games, 1):
        for half in (1, 2):
            scores[(game, half)] = score_half(model, mean, std, load_features(game, half))
        print(f"\rscored {i}/{len(games)} games", end="")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(scores))
    print()


# ---------------------------------------------------------------------------
# 2. PEAKS: turn the score curve into a few predictions
# ---------------------------------------------------------------------------
def find_peaks(times, probs):
    """Take the highest score, drop everything within PEAK_DISTANCE, repeat."""
    peaks = []
    for i in np.argsort(-probs):                  # from highest to lowest score
        if probs[i] < 0.01:
            break
        too_close = any(abs(times[i] - t) <= PEAK_DISTANCE for t, _ in peaks)
        if not too_close:
            peaks.append((times[i], probs[i]))
    return peaks


predictions = {half: find_peaks(times, probs) for half, (times, probs) in scores.items()}

real_goals = {half: [] for half in scores}         # (game, half) -> [goal times in s]
for g in load_goals(SPLIT):
    real_goals[(g["game"], g["half"])].append(g["seconds"])
n_goals = sum(len(v) for v in real_goals.values())
n_matches = len(scores) // 2
print(f"{n_matches} matches, {n_goals} goals")


# ---------------------------------------------------------------------------
# 3. COMPARE predictions with the real goals
# ---------------------------------------------------------------------------
def compare(tolerance):
    """For every prediction: (score, was it correct?, seconds after the real goal)."""
    results = []
    for half, peaks in predictions.items():
        goals_left = list(real_goals[half])        # each goal can be found only once
        for time, score in peaks:                  # best predictions first
            distances = [abs(time - g) for g in goals_left]
            if distances and min(distances) <= tolerance:
                goal = goals_left.pop(int(np.argmin(distances)))
                results.append((score, True, time - goal))
            else:
                results.append((score, False, None))
    return results


def average_precision(results):
    """Walk the predictions from best to worst score; every time we hit a real
    goal, note the precision so far. AP = sum of those / number of goals."""
    results = sorted(results, key=lambda r: -r[0])
    correct_so_far, total = 0, 0.0
    for k, (score, correct, _) in enumerate(results, 1):
        if correct:
            correct_so_far += 1
            total += correct_so_far / k
    return total / n_goals


for tolerance in [10, 30]:
    results = compare(tolerance)
    print(f"\n=== a prediction counts if within ±{tolerance} s ===   AP = {average_precision(results):.3f}")
    print(f"  {'threshold':>9} {'recall':>7} {'precision':>9} {'FA/match':>9} {'predictions':>11}")
    for thr in [0.3, 0.5, THRESHOLD, 0.7, 0.9]:
        kept = [r for r in results if r[0] >= thr]
        correct = sum(1 for r in kept if r[1])
        wrong = len(kept) - correct
        print(f"  {thr:>9.3f} {correct / n_goals:>7.3f} {correct / max(len(kept), 1):>9.3f} "
              f"{wrong / n_matches:>9.2f} {len(kept):>11}")

# --- how far from the real goal are the correct peaks? ---
offsets = np.array([r[2] for r in compare(30) if r[1] and r[0] >= THRESHOLD])
print(f"\nCorrect peaks are typically {np.median(np.abs(offsets)):.1f} s away from the goal; "
      f"on average {offsets.mean():+.1f} s (+ = after the goal)")

# --- are false alarms just replays/celebrations of a real goal? ---
false_alarms, right_after_goal = 0, 0
for half, peaks in predictions.items():
    for time, score in peaks:
        if score < THRESHOLD or any(abs(time - g) <= 10 for g in real_goals[half]):
            continue                               # not a false alarm
        false_alarms += 1
        if any(10 < time - g <= 120 for g in real_goals[half]):
            right_after_goal += 1
print(f"False alarms: {false_alarms}, of which {right_after_goal} are 10-120 s after a real goal")
