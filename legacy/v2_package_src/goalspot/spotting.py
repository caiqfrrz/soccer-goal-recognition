"""Slide a trained window classifier over a full half -> one score per step."""
import numpy as np
import torch

from . import config as C


def score_half(model, features, mean, std, device, stride=2, batch_size=512):
    """features: (n_rows, dim). Returns (times_s, scores) for each window center."""
    x = ((features - mean) / std).astype(np.float32)
    # every 40-row window, already shaped (n_windows, dim, 40) like in training
    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * C.HALF_WINDOW, axis=0)[::stride]
    scores = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(windows[i:i + batch_size])).to(device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    centers = np.arange(len(windows)) * stride + C.HALF_WINDOW    # row at the window center
    return centers / C.FPS, np.concatenate(scores)
