from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == targets).mean())


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))
