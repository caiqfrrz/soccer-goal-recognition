"""Loading labels, per-half features and the pre-built training windows."""
import csv
import json
from collections import defaultdict
from functools import lru_cache

import numpy as np

from . import config as C


# --- time <-> feature row ----------------------------------------------------
def ms_to_row(ms):
    return int(int(ms) / 1000 * C.FPS)


def row_to_clock(row):
    s = row / C.FPS
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


# --- features ----------------------------------------------------------------
def feature_path(game, half, feature_set=None):
    """Where the (n_rows, dim) feature array of one half lives.

    This is the ONE place that knows the file layout, so switching to our own
    extracted features only means saving them under data/features/<name>/ and
    setting config.FEATURE_SET = "<name>".
    """
    feature_set = feature_set or C.FEATURE_SET
    if feature_set in C.SOCCERNET_FEATURE_SETS:
        return C.SOCCERNET_DIR / game / f"{half}_{feature_set}.npy"
    return C.OWN_FEATURES_DIR / feature_set / game / f"{half}.npy"


def load_half_features(game, half, feature_set=None, mmap=False):
    return np.load(feature_path(game, half, feature_set), mmap_mode="r" if mmap else None)


# --- labels ------------------------------------------------------------------
def load_goals(split=None):
    """Rows of data/goals.csv (dicts), optionally filtered to one split."""
    with open(C.GOALS_CSV) as f:
        return [g for g in csv.DictReader(f) if split is None or g["split"] == split]


def goal_rows_by_half(split=None):
    """{(split, game, half): [feature row of each goal]}"""
    out = defaultdict(list)
    for g in load_goals(split):
        out[(g["split"], g["game"], int(g["half"]))].append(ms_to_row(g["position_ms"]))
    return out


@lru_cache(maxsize=None)
def load_annotations(game):
    """All annotated events of a game as (half, position_ms, label)."""
    anns = json.loads((C.SOCCERNET_DIR / game / "Labels-v2.json").read_text())["annotations"]
    return [(int(a["gameTime"].split(" - ")[0]), int(a["position"]), a["label"]) for a in anns]


# --- pre-built windows (scripts/03_build_windows.py) -------------------------
def windows_dir(feature_set=None):
    return C.WINDOWS_DIR / (feature_set or C.FEATURE_SET)


def load_windows(split, feature_set=None, mmap=False):
    """X: (N, 40, dim), y: (N,), meta: list of dicts (game, half, center_row, label, kind, event)."""
    d = windows_dir(feature_set)
    X = np.load(d / f"X_{split}.npy", mmap_mode="r" if mmap else None)
    y = np.load(d / f"y_{split}.npy")
    with open(d / f"meta_{split}.csv") as f:
        meta = list(csv.DictReader(f))
    return X, y, meta


def standardize(X, mean, std):
    """(N, T, D) numpy -> (N, D, T) float32 tensor, the layout Conv1d expects."""
    import torch
    return torch.tensor((X - mean) / std, dtype=torch.float32).permute(0, 2, 1)


def fit_standardizer(X_train):
    return X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)) + 1e-6
