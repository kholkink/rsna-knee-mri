"""Pseudo-labels: blend report-derived soft targets with out-of-fold image-model predictions.

On the 58 expert studies the report labels score 0.893 macro AUC, a single held-out image model 0.904 and their
probability blend 0.917-0.921: the images carry what the report text leaves out. Only out-of-fold predictions are
used (a model never relabels a study it was trained on); expert-labelled studies keep their expert targets.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .constants import LABELS


def collect_oof(oofs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Mean out-of-fold probability per study over all runs that held the study out. Index: study_uid, columns: p_<label>."""
    pcols = [f"p_{k}" for k in LABELS]
    frames = []
    for o in oofs:
        o = o[~o["is_gold"].astype(bool)] if "is_gold" in o.columns else o
        frames.append(o[["study_uid", *pcols]].assign(study_uid=lambda d: d["study_uid"].astype(str)))
    if not frames:
        return pd.DataFrame(columns=pcols + ["n_oof"])
    allp = pd.concat(frames, ignore_index=True)
    out = allp.groupby("study_uid")[pcols].mean()
    out["n_oof"] = allp.groupby("study_uid").size()
    return out


def blend_pseudo_labels(table: pd.DataFrame, oof: pd.DataFrame, alpha: float | dict[str, float] = 0.5) -> pd.DataFrame:
    """y_new = (1 - alpha) * y_report + alpha * p_oof for non-gold studies that have an out-of-fold prediction."""
    out = table.copy()
    uid = out["study_uid"].astype(str)
    has = uid.isin(oof.index).values & ~out["is_gold"].astype(bool).values
    for k in LABELS:
        a = float(alpha.get(k, alpha.get("default", 0.5)) if isinstance(alpha, dict) else alpha)
        p = oof[f"p_{k}"].reindex(uid[has]).values.astype(np.float32)
        y = out.loc[has, f"y_{k}"].values.astype(np.float32)
        out.loc[has, f"y_{k}"] = np.clip((1.0 - a) * y + a * p, 0.0, 1.0)
    if "src" in out.columns:
        out.loc[has, "src"] = out.loc[has, "src"].astype(str) + "+oof"
    return out
