#!/usr/bin/env python
"""Scanner-grouped stratified folds.

    python scripts/make_folds.py --label-table labels/label_table.parquet --meta data/meta_train.parquet --out labels/folds.parquet
"""
from __future__ import annotations

import argparse

import pandas as pd

from rsna_knee.cv import make_folds, study_groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-table", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    lt = pd.read_parquet(a.label_table)
    groups = study_groups(pd.read_parquet(a.meta)) if a.meta else None
    folds = make_folds(lt, groups, a.n_folds, a.seed)
    folds.to_parquet(a.out, index=False)
    print(folds["fold"].value_counts().sort_index())
    print("gold per fold:", folds.merge(lt[["study_uid", "is_gold"]], on="study_uid").groupby("fold")["is_gold"].sum().to_dict())
    print(f"folds -> {a.out}")


if __name__ == "__main__":
    main()
