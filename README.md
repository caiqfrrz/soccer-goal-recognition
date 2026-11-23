# Goal spotting on SoccerNet

## The idea in 5 lines

1. Every match half is turned into a table: **one row every 0.5 s**, each row = 512 numbers describing the frame ("features").
2. We cut **20-second windows** (40 rows): some centered on goals, some not.
3. A model learns to say **"goal / not goal"** for a window.
4. Then we **slide** the model over a whole match, second by second, and the peaks of its score are the predicted goals.
5. Today the features come from SoccerNet (ResNet). **The end goal is to compute our own.**

## Folders

```
scripts/            ← everything you run, in order
  common.py         settings + shared functions (read this first)
  01_make_goal_list.py      where are the goals?            -> data/goals.csv
  02_check_features.py      download + inspect features
  03_make_windows.py        cut 20 s training windows       -> data/windows/
  04_train_models.py        train logistic regression + CNN -> outputs/models/, PR curve
  05_error_analysis.py      what does the CNN get wrong?    -> outputs/reports/
  06_hard_negatives_test.py do shot/corner windows help?
  07_spotting.py            find goals in full matches

data/               the data (large, never edited by hand)
  soccernet/        downloaded games: labels + SoccerNet features
  windows/          output of step 03
  goals.csv         output of step 01

outputs/            results: models, figures, reports, cache
legacy/             old versions of the scripts, not used anymore
```

## How to run

```bash
source .venv/bin/activate
python scripts/04_train_models.py      # any script, from the projeto folder
```

Steps 01–03 only need to run once (their outputs already exist).
After retraining in step 04, delete `outputs/cache/` before running step 07.

## Results so far (valid set)

| step | what | result |
|---|---|---|
| 04 | AP on windows, logistic regression | 0.66 |
| 04 | AP on windows, 1D-CNN (3 seeds) | 0.77 ± 0.02 |
| 07 | goals found in full matches (±30 s) | 81% |
| 07 | of the predictions, how many are real goals | 27% (~8 false alarms per match) |
| 07 | correct peaks are late by | ~5.5 s |

Known issue: `goals.csv` contains duplicated goals (same goal twice, <2 s apart):
32 in train, 15 in valid, 11 in test.

## Using our own features later

1. Save one `(rows, n_features)` array per half, 2 rows per second, at
   `data/features/<name>/<league>/<season>/<game>/1.npy` and `2.npy`.
2. In `scripts/common.py` set `FEATURES = "<name>"`.
3. Run steps 03 → 07. Results are saved under the new name, so the ResNet
   results stay for comparison.
