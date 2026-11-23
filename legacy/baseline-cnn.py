import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 10          # valid F1 stopped improving around epoch 4 last time
BATCH_SIZE = 64
LR = 1e-3


# --- Part 1: load and standardize once (same for every seed) ---
def load(split):
    return np.load(f"dataset/X_{split}.npy"), np.load(f"dataset/y_{split}.npy")


X_train, y_train = load("train")
X_valid, y_valid = load("valid")
mean = X_train.mean(axis=(0, 1))
std = X_train.std(axis=(0, 1)) + 1e-6
X_train_t = torch.tensor((X_train - mean) / std, dtype=torch.float32).permute(0, 2, 1)
X_valid_t = torch.tensor((X_valid - mean) / std, dtype=torch.float32).permute(0, 2, 1)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()],
                          dtype=torch.float32, device=device)


# --- Part 2: same model as step 4b ---
class GoalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict(model):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xb.to(device)).cpu()
                            for xb in DataLoader(X_valid_t, batch_size=256)])
    return (torch.sigmoid(logits) > 0.5).numpy().astype(int)


# --- Part 3: one full training run for a given seed ---
def run(seed):
    torch.manual_seed(seed)                  # controls weight init, dropout and shuffling
    model = GoalCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                        batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    # Evaluate ONLY the final model: no picking the best epoch by looking at valid
    pred = predict(model)
    return (precision_score(y_valid, pred), recall_score(y_valid, pred), f1_score(y_valid, pred))


# --- Part 4: repeat for every seed and summarize ---
results = []
for seed in SEEDS:
    p, r, f = run(seed)
    results.append((p, r, f))
    print(f"seed {seed}:  precision={p:.3f}  recall={r:.3f}  F1={f:.3f}")

results = np.array(results)
print(f"\nOver {len(SEEDS)} seeds (mean ± std):")
for name, col in zip(["precision", "recall", "F1"], results.T):
    print(f"  {name:9} {col.mean():.3f} ± {col.std():.3f}")