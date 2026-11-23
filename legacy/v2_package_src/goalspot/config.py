"""Project-wide paths and constants. Change things here, not in the scripts."""
from pathlib import Path

import torch

# --- paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SOCCERNET_DIR = DATA_DIR / "soccernet"          # downloaded games: labels + SoccerNet features
OWN_FEATURES_DIR = DATA_DIR / "features"        # features we extract ourselves (future)
WINDOWS_DIR = DATA_DIR / "windows"              # training windows, one subfolder per feature set
GOALS_CSV = DATA_DIR / "goals.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CACHE_DIR = OUTPUTS_DIR / "cache"

# --- data --------------------------------------------------------------------
SPLITS = ["train", "valid", "test"]
FPS = 2                    # feature rows per second
HALF_WINDOW = 20           # rows on each side of the center -> 40 rows = 20 s window

# Which feature set to use. SoccerNet ones live next to the labels as
# "<half>_<name>.npy"; anything else is read from data/features/<name>/.
FEATURE_SET = "ResNET_TF2_PCA512"
SOCCERNET_FEATURE_SETS = {"ResNET_TF2", "ResNET_TF2_PCA512", "C3D", "C3D_PCA512", "I3D", "I3D_PCA512"}

# --- training ----------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def ensure_output_dirs():
    for d in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
