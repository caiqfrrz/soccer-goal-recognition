"""Step 2: download SoccerNet features and sanity-check them against the goals.

Only needed for SoccerNet feature sets (config.FEATURE_SET). For our own
features, the extraction script writes the files instead.
"""
import argparse

from SoccerNet.Downloader import SoccerNetDownloader

from goalspot import config as C
from goalspot.data import feature_path, load_goals, load_half_features, ms_to_row

parser = argparse.ArgumentParser()
parser.add_argument("--splits", nargs="+", default=C.SPLITS)
parser.add_argument("--no-download", action="store_true")
args = parser.parse_args()

if not args.no_download:
    assert C.FEATURE_SET in C.SOCCERNET_FEATURE_SETS, "not a SoccerNet feature set"
    downloader = SoccerNetDownloader(LocalDirectory=str(C.SOCCERNET_DIR))
    downloader.downloadGames(files=[f"1_{C.FEATURE_SET}.npy", f"2_{C.FEATURE_SET}.npy"],
                             split=args.splits)

for split in args.splits:
    goals = load_goals(split)

    # inspect one file
    g = goals[0]
    features = load_half_features(g["game"], g["half"])
    print(f"\n[{split}] {len(goals)} goals | example: {g['game']} half {g['half']}")
    print(f"  shape {features.shape} {features.dtype}, "
          f"duration {features.shape[0] / C.FPS / 60:.1f} min")

    # every goal must fall inside its feature file
    outside = 0
    for g in goals:
        n_rows = load_half_features(g["game"], g["half"], mmap=True).shape[0]
        r = ms_to_row(g["position_ms"])
        if r >= n_rows:
            outside += 1
            print("  outside:", feature_path(g["game"], g["half"]), "row", r, ">=", n_rows)
    print(f"  {len(goals) - outside}/{len(goals)} goals fall inside their feature file")
