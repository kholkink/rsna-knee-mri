#!/usr/bin/env python
"""Report-derived label table + out-of-fold model predictions -> pseudo-label table.

    python scripts/make_pseudo_labels.py --labels configs/labels/label_table_blend_v3.parquet \
        --oof 'data/oof/oof-nano/fold*/oof.parquet' --alpha 0.5 --out configs/labels/label_table_pseudo_v1.parquet

Use the last-epoch `oof.parquet` (not `oof_best.parquet`: that epoch was picked on the very studies it predicts).
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import LABELS
from rsna_knee.pseudo import blend_pseudo_labels, collect_oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--oof", nargs="+", required=True, help="oof parquet files or globs (any number of runs / folds)")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the model prediction in the new target")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    table = pd.read_parquet(a.labels)
    files = sorted({f for pat in a.oof for f in glob.glob(pat)})
    if not files:
        raise SystemExit(f"no OOF files match {a.oof}")
    oof = collect_oof(pd.read_parquet(f) for f in files)
    n_train = int((~table["is_gold"].astype(bool)).sum())
    print(f"{len(files)} OOF files, {len(oof)} studies covered of {n_train} non-gold ({len(oof) / max(n_train, 1):.1%}); "
          f"runs per study: {oof['n_oof'].value_counts().sort_index().to_dict()}")

    new = blend_pseudo_labels(table, oof, a.alpha)
    t = table.assign(study_uid=table["study_uid"].astype(str)).set_index("study_uid").loc[oof.index]
    rows = []
    for k in LABELS:
        y, p = t[f"y_{k}"].values, oof[f"p_{k}"].values
        conf = (y <= 0.15) | (y >= 0.85)
        yb = (y[conf] >= 0.5).astype(int)
        auc = roc_auc_score(yb, p[conf]) if 0 < yb.sum() < len(yb) else np.nan
        rows.append({"label": k, "oof_auc_vs_confident": round(auc, 4), "mean_y": round(float(y.mean()), 3), "mean_p": round(float(p.mean()), 3),
                     "mean_abs_shift": round(float(a.alpha * np.abs(p - y).mean()), 3),
                     "flips_0.5": int((((1 - a.alpha) * y + a.alpha * p >= 0.5) != (y >= 0.5)).sum())})
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    print(f"macro OOF AUC vs confident report labels: {rep['oof_auc_vs_confident'].mean():.4f}")
    new.to_parquet(a.out, index=False)
    print(f"alpha={a.alpha} -> {a.out} ({len(new)} rows)")


if __name__ == "__main__":
    main()
