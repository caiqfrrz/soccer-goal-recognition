"""Step 1: download SoccerNet labels and list every goal -> data/goals.csv"""
import csv
import json

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

from goalspot import config as C

downloader = SoccerNetDownloader(LocalDirectory=str(C.SOCCERNET_DIR))
downloader.downloadGames(files=["Labels-v2.json"], split=C.SPLITS)

rows = []
for split in C.SPLITS:
    games = getListGames(split=split)
    for game in games:
        labels = json.loads((C.SOCCERNET_DIR / game / "Labels-v2.json").read_text())
        for a in labels["annotations"]:
            if a["label"] == "Goal":
                half = int(a["gameTime"].split(" - ")[0])
                rows.append([split, game, half, int(a["position"]), a["visibility"]])
    n = sum(1 for r in rows if r[0] == split)
    print(f"{split}: {len(games)} games, {n} goals")

C.GOALS_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(C.GOALS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "game", "half", "position_ms", "visibility"])
    w.writerows(rows)
print("Saved", C.GOALS_CSV)
