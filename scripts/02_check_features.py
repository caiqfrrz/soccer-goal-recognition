"""
STEP 2 - Download the video features and check they make sense.

Question:  what do the features look like, and do the goal times fit in them?
Reads:     data/goals.csv
Writes:    nothing (downloads the feature files into data/soccernet/)

What a feature file is: for one half of one game, a table with one row every
0.5 s of video. Each row is 512 numbers that describe that frame (computed by
a ResNet image network, then compressed with PCA). The model never sees pixels,
only these numbers.
"""
import numpy as np
from SoccerNet.Downloader import SoccerNetDownloader

from common import FEATURES, FPS, SOCCERNET, feature_file, load_features, load_goals, seconds_to_row

downloader = SoccerNetDownloader(LocalDirectory=str(SOCCERNET))
downloader.downloadGames(files=[f"1_{FEATURES}.npy", f"2_{FEATURES}.npy"],
                         split=["train", "valid", "test"])

# --- look at one file ---
goal = load_goals("valid")[0]
features = load_features(goal["game"], goal["half"])
print("Example:", goal["game"], "half", goal["half"])
print("  shape:", features.shape, "-> rows x numbers per row")
print(f"  duration: {features.shape[0] / FPS / 60:.1f} minutes")

# --- every goal must happen before the end of its feature file ---
for split in ["train", "valid", "test"]:
    goals = load_goals(split)
    problems = 0
    for goal in goals:
        n_rows = np.load(feature_file(goal["game"], goal["half"]), mmap_mode="r").shape[0]
        if seconds_to_row(goal["seconds"]) >= n_rows:
            problems += 1
            print("  goal after end of file:", goal)
    print(f"{split}: {len(goals) - problems}/{len(goals)} goals fit in their feature file")
