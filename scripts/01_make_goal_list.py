"""
STEP 1 - Make a list of every goal.

Question:  where are the goals?
Reads:     SoccerNet labels (downloads them if missing)
Writes:    data/goals.csv  ->  split, game, half, position_ms, visibility
"""
import csv
import json

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

from common import GOALS_CSV, SOCCERNET

# Download the label file of every game (skips files already downloaded).
downloader = SoccerNetDownloader(LocalDirectory=str(SOCCERNET))
downloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])

rows = []
for split in ["train", "valid", "test"]:
    games = getListGames(split=split)
    goals_in_split = 0

    for game in games:
        labels = json.loads((SOCCERNET / game / "Labels-v2.json").read_text())
        for event in labels["annotations"]:
            if event["label"] == "Goal":
                half = int(event["gameTime"][0])               # "2 - 13:10" -> 2
                rows.append([split, game, half, int(event["position"]), event["visibility"]])
                goals_in_split += 1

    print(f"{split}: {len(games)} games, {goals_in_split} goals")

with open(GOALS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "game", "half", "position_ms", "visibility"])
    writer.writerows(rows)
print("Saved", GOALS_CSV)
