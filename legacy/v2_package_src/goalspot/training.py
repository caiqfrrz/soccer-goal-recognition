"""Training / inference loops shared by every experiment."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .models import GoalCNN


def train_cnn(X_train_t, y_train, seed, device, epochs=C.EPOCHS):
    """Train a GoalCNN from scratch. X_train_t: (N, dim, T) tensor, y_train: numpy 0/1.

    Only the final model is returned: no picking the best epoch on valid.
    """
    torch.manual_seed(seed)                      # weight init, dropout and shuffling
    model = GoalCNN(in_dim=X_train_t.shape[1]).to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(X_train_t, torch.tensor(y_train, dtype=torch.float32)),
                        batch_size=C.BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    return model


def predict_proba(model, X_t, device, batch_size=256):
    """Goal probability for every window of X_t (N, dim, T)."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xb.to(device)).cpu()
                            for xb in DataLoader(X_t, batch_size=batch_size)])
    return torch.sigmoid(logits).numpy()


def to_pred(prob, threshold=0.5):
    return (np.asarray(prob) > threshold).astype(int)
