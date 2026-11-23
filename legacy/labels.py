import csv
import json
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from pathlib import Path

ROOT = Path("./SoccerNet")
SPLITS = ["train", "valid", "test"]

d = SoccerNetDownloader(LocalDirectory=str(ROOT))
d.downloadGames(files=["Labels-v2.json"], split=SPLITS)

rows = []
for split in SPLITS:
    games = getListGames(split=split)
    for game in games:
        labels = json.loads((ROOT / game / "Labels-v2.json").read_text())
        for a in labels["annotations"]:
            if a["label"] == "Goal":
                half = int(a["gameTime"].split(" - ")[0])
                rows.append([split, game, half, int(a["position"]), a["visibility"]])

    n = sum(1 for r in rows if r[0] == split)
    print(f"{split}: {len(games)} games, {n} goals")

with open("goals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "game", "half", "position_ms", "visibility"])
    w.writerows(rows)