import csv
from pathlib import Path

import numpy as np
from SoccerNet.Downloader import SoccerNetDownloader

ROOT = Path("./SoccerNet")
FPS = 2                              # features are sampled at 2 per second
FEAT = "ResNET_TF2_PCA512.npy"       # file name after the "1_" / "2_" prefix

# --- download features (valid split only, to start) ---
downloader = SoccerNetDownloader(LocalDirectory=str(ROOT))
downloader.downloadGames(files=[f"1_{FEAT}", f"2_{FEAT}"], split=["valid"])

# --- load the goals of that split from step 1b ---
with open("goals.csv") as f:
    goals = [row for row in csv.DictReader(f) if row["split"] == "valid"]
print("Goals in valid:", len(goals))

# --- Part 3: open the feature file of one goal and inspect it ---
goal = goals[0]
features = np.load(ROOT / goal["game"] / f"{goal['half']}_{FEAT}")
print("\nGame:", goal["game"], "| half", goal["half"])
print("Shape:", features.shape, "| dtype:", features.dtype)
print(f"Duration: {features.shape[0] / FPS / 60:.1f} min")

# --- Part 4: goal time -> row -> window of features around it ---
row = int(int(goal["position_ms"]) / 1000 * FPS)
HALF_WINDOW = 20                     # 20 rows = 10 s on each side
start = max(0, row - HALF_WINDOW)
end = min(len(features), row + HALF_WINDOW)
window = features[start:end]
print(f"\nGoal at {int(goal['position_ms']) / 1000:.1f}s -> row {row}")
print("Window shape:", window.shape)

# --- Part 5: check that every goal falls inside its feature file ---
outside = 0
for g in goals:
    path = ROOT / g["game"] / f"{g['half']}_{FEAT}"
    n_rows = np.load(path, mmap_mode="r").shape[0]   # reads only the header
    r = int(int(g["position_ms"]) / 1000 * FPS)
    if r >= n_rows:
        outside += 1
        print("  outside:", g["game"], "half", g["half"], "row", r, ">=", n_rows)
print(f"\n{len(goals) - outside}/{len(goals)} goals fall inside their feature file")