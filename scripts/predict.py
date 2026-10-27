#!/usr/bin/env python
"""Local / Kaggle inference entry point.

    python scripts/predict.py --comp-root /kaggle/input/rsna-knee-abnormality-detection --runs runs/exp/fold0 runs/exp/fold1
"""
from __future__ import annotations

import argparse
import logging
import os

from rsna_knee.infer import InferenceConfig, predict, write_submission
from rsna_knee.series_select import SelectionPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument("--max-per-plane", type=int, default=1)
    ap.add_argument("--max-total", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    icfg = InferenceConfig(comp_root=a.comp_root, run_dirs=a.runs, out_path=a.out, tta_hflip=not a.no_tta,
                           batch_size=a.batch_size, num_workers=a.workers, time_budget_s=a.budget_hours * 3600,
                           policy=SelectionPolicy(max_per_plane=a.max_per_plane, max_total=a.max_total), limit=a.limit)
    pred = predict(icfg)
    sub = write_submission(pred, os.path.join(a.comp_root, "sample_submission.csv"), a.out)
    print(sub.head())
    print(f"submission -> {a.out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
