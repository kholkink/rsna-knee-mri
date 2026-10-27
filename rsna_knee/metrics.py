"""Competition metric: macro-averaged ROC-AUC over the 12 labels."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .constants import LABELS


def per_label_auc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """AUC per label; labels without both classes are NaN. ``mask`` (N,L) marks valid targets."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    out: dict[str, float] = {}
    for j, name in enumerate(LABELS[: y_true.shape[1]]):
        m = np.ones(len(y_true), dtype=bool) if mask is None else np.asarray(mask[:, j], dtype=bool)
        t = (y_true[m, j] >= 0.5).astype(int)
        p = y_pred[m, j]
        if len(t) == 0 or t.min() == t.max():
            out[name] = float("nan")
        else:
            out[name] = float(roc_auc_score(t, p))
    return out


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    vals = [v for v in per_label_auc(y_true, y_pred, mask).values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")
