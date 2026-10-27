#!/usr/bin/env python
"""Scan one DICOM header per series -> parquet (plane, sequence, geometry, scanner fingerprint).

    python scripts/scan_metadata.py --image-root /kaggle/input/.../train_images --out data/meta_train.parquet
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from rsna_knee.dicom import scan_series_header


def _job(args: tuple[str, str, str]) -> dict:
    sdir, study, series = args
    return scan_series_header(sdir, study, series).to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--series-csv", default=None, help="train_series.csv / test_series.csv (optional, else walk dirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n over sorted study uids")
    a = ap.parse_args()
    if not os.path.isdir(a.image_root):
        raise SystemExit(f"image root not found: {a.image_root!r}. The DICOM files live only on Kaggle "
                         "(/kaggle/input/rsna-knee-abnormality-detection/train_images) - run this step in a Kaggle "
                         "notebook (kaggle/prepare_cache_notebook.py) or on a cloud box after downloading the data.")

    jobs: list[tuple[str, str, str]] = []
    if a.series_csv:
        df = pd.read_csv(a.series_csv)
        for r in df.itertuples():
            jobs.append((os.path.join(a.image_root, str(r.StudyInstanceUID), str(r.SeriesInstanceUID)),
                         str(r.StudyInstanceUID), str(r.SeriesInstanceUID)))
    else:
        for study in sorted(os.listdir(a.image_root)):
            sdir = os.path.join(a.image_root, study)
            if not os.path.isdir(sdir):
                continue
            for series in sorted(os.listdir(sdir)):
                if os.path.isdir(os.path.join(sdir, series)):
                    jobs.append((os.path.join(sdir, series), study, series))
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    if shard_n > 1:
        order = {u: k for k, u in enumerate(sorted({j[1] for j in jobs}))}
        jobs = [j for j in jobs if order[j[1]] % shard_n == shard_i]
    if a.limit:
        jobs = jobs[: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(tqdm(ex.map(_job, jobs, chunksize=16), total=len(jobs), desc="scan"))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_parquet(a.out, index=False)
    n_err = int((df["error"] != "").sum())
    if n_err:
        print(f"WARNING: {n_err} series with errors, e.g. {df.loc[df['error'] != '', 'error'].iloc[0]!r}")
    print(df[["plane", "weighting", "fat_sat"]].value_counts().head(20))
    print(f"{len(df)} series, {df['study_uid'].nunique()} studies, {df['fingerprint'].nunique()} fingerprints -> {a.out}")


if __name__ == "__main__":
    main()
