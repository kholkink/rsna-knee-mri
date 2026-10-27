#!/usr/bin/env python
"""Select series per study and build the JPEG-npz cache.

    python scripts/build_cache.py --meta data/meta_train.parquet --out cache --size 384 --max-per-plane 1 --max-total 3
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from rsna_knee.dicom import SeriesMeta
from rsna_knee.preprocess import CacheJob, build_cache
from rsna_knee.series_select import SelectionPolicy, select_series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--resize-mode", default="pad", choices=["pad", "stretch"])
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--max-slices", type=int, default=64)
    ap.add_argument("--max-per-plane", type=int, default=1)
    ap.add_argument("--max-total", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="studies (debug)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th study starting at i (split across notebooks)")
    a = ap.parse_args()

    meta = pd.read_parquet(a.meta)
    meta = meta[meta["error"].fillna("") == ""] if "error" in meta else meta
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    policy = SelectionPolicy(max_per_plane=a.max_per_plane, max_total=a.max_total)
    jobs: list[CacheJob] = []
    n_studies = 0
    for k, (uid, g) in enumerate(meta.groupby("study_uid", sort=True)):
        if k % shard_n != shard_i:
            continue
        n_studies += 1
        if a.limit and n_studies > a.limit:
            break
        metas = [SeriesMeta.from_dict(r) for r in g.to_dict("records")]
        for m in select_series(metas, policy):
            jobs.append(CacheJob(m, os.path.join(a.out, str(uid), f"{m.series_uid}.npz"), a.size, a.resize_mode, a.quality,
                                 a.max_slices))
    print(f"shard {a.shard}: {len(jobs)} series selected from {n_studies} studies")
    df = build_cache(jobs, workers=a.workers or None)
    idx_path = os.path.join(a.out, "cache_index.parquet")
    if os.path.exists(idx_path):
        old = pd.read_parquet(idx_path)
        df = pd.concat([old[~old["series_uid"].isin(df["series_uid"])], df], ignore_index=True)
    df.to_parquet(idx_path, index=False)
    print(df["status"].value_counts())
    print(f"index -> {idx_path}")


if __name__ == "__main__":
    main()
