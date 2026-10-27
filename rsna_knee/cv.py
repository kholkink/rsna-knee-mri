"""Site-grouped, label-stratified cross-validation.

Random folds leak scanner signatures (reported ~+0.05 macro-AUC optimism); the hidden test set
comes from the same 16-19 sites but with per-site shift, so folds are grouped by the scanner
fingerprint (manufacturer | model | software | frequency | coil) and stratified by the rarest
positive label so every fold sees fractures / contusions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import LABELS


def study_groups(meta_df: pd.DataFrame) -> pd.Series:
    """study_uid -> most frequent scanner fingerprint among its series."""
    return meta_df.groupby("study_uid")["fingerprint"].agg(lambda s: s.value_counts().index[0])


def strat_key(label_table: pd.DataFrame) -> np.ndarray:
    y = label_table[[f"y_{k}" for k in LABELS]].values.astype(float)
    pos = np.nan_to_num(y, nan=0.0) >= 0.5
    prev = pos.mean(0)
    key = np.full(len(y), -1)
    for j in np.argsort(prev)[::-1]:  # decreasing prevalence; rarest assigned last -> wins
        key[pos[:, j]] = j
    if "is_gold" in label_table:
        key = np.where(label_table["is_gold"].values.astype(bool), key + 100, key)
    return key


def merge_rare_classes(key: np.ndarray, n_folds: int) -> np.ndarray:
    """StratifiedGroupKFold needs >= n_folds members per class: pool tiny classes, then fold into the majority."""
    key = key.copy()
    vals, counts = np.unique(key, return_counts=True)
    small = vals[counts < n_folds]
    if len(small):
        key[np.isin(key, small)] = -2
    vals, counts = np.unique(key, return_counts=True)
    small = vals[counts < n_folds]
    if len(small):
        key[np.isin(key, small)] = vals[np.argmax(counts)]
    return key


def make_folds(label_table: pd.DataFrame, groups: pd.Series | None = None, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    uids = label_table["study_uid"].astype(str).values
    g = groups.reindex(uids).fillna("unknown").values if groups is not None else uids.copy()
    if len(np.unique(g)) < n_folds:  # degenerate grouping (single scanner): fall back to per-study groups
        g = uids.copy()
    key = merge_rare_classes(strat_key(label_table), n_folds)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(uids), -1)
    for f, (_, val_idx) in enumerate(skf.split(uids, key, g)):
        fold[val_idx] = f
    return pd.DataFrame({"study_uid": uids, "fold": fold, "group": g, "strat_key": key})
