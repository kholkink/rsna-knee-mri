"""Cache builder: DICOM series -> compact per-series ``.npz`` of JPEG-encoded uint8 slices.

Why JPEG-in-npz: 13k series x 32 slices x 384^2 raw uint8 = ~60 GB, too big for a Kaggle
dataset; JPEG q=92 grayscale is ~1 MB / series (~13 GB total) and decodes at >1000 slices/s
with cv2. Random slice access is cheap: only the selected slices are decoded.
"""
from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dicom import SeriesMeta, preprocess_series

log = logging.getLogger(__name__)


def encode_jpeg_stack(vol: np.ndarray, quality: int = 92) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    offsets = [0]
    for img in vol:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        b = np.frombuffer(buf.tobytes(), dtype=np.uint8)
        chunks.append(b)
        offsets.append(offsets[-1] + b.size)
    data = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)
    return data, np.asarray(offsets, dtype=np.int64)


def save_series_npz(path: str, vol: np.ndarray, positions: np.ndarray, quality: int = 92) -> None:
    data, offsets = encode_jpeg_stack(vol, quality)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, jpeg=data, offsets=offsets, positions=positions.astype(np.float32),
             shape=np.asarray(vol.shape, dtype=np.int64))
    os.replace(tmp, path)


def load_series_npz(path: str, indices: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode selected slices (all if ``indices`` is None). Returns (uint8 (K,H,W), positions (N,))."""
    with np.load(path) as z:
        data, offsets, positions, shape = z["jpeg"], z["offsets"], z["positions"], z["shape"]
    n = int(shape[0])
    idx = list(range(n)) if indices is None else list(indices)
    out = np.empty((len(idx), int(shape[1]), int(shape[2])), dtype=np.uint8)
    for k, i in enumerate(idx):
        buf = data[offsets[i]:offsets[i + 1]]
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"jpeg decode failed: {path}[{i}]")
        out[k] = img
    return out, positions


@dataclass
class CacheJob:
    meta: SeriesMeta
    out_path: str
    size: int = 384
    resize_mode: str = "pad"
    quality: int = 92
    max_slices: int | None = 64
    lo_pct: float = 0.5
    hi_pct: float = 99.5
    crop_mm: float | None = None
    canonical: bool = False
    mirror: bool = False


def _run_job(job: CacheJob) -> dict:
    row = {"study_uid": job.meta.study_uid, "series_uid": job.meta.series_uid, "path": job.out_path,
           "plane": job.meta.plane, "weighting": job.meta.weighting, "fat_sat": job.meta.fat_sat,
           "plane_id": job.meta.plane_id, "seq_id": job.meta.seq_id, "n_slices": 0, "status": "ok", "error": ""}
    try:
        if os.path.exists(job.out_path):
            with np.load(job.out_path) as z:
                row["n_slices"] = int(z["shape"][0])
            row["status"] = "cached"
            return row
        vol, pos, _ = preprocess_series(job.meta.series_dir, job.size, job.resize_mode, job.lo_pct, job.hi_pct,
                                        meta=job.meta, max_slices=job.max_slices, crop_mm=job.crop_mm,
                                        canonical=job.canonical, mirror=job.mirror)
        save_series_npz(job.out_path, vol, pos, job.quality)
        row["n_slices"] = int(vol.shape[0])
    except Exception as e:  # never kill the pool
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        log.warning("cache job failed %s: %s\n%s", job.meta.series_dir, e, traceback.format_exc(limit=2))
    return row


def build_cache(jobs: Iterable[CacheJob], workers: int | None = None, progress: bool = True,
                on_done: "Callable[[dict], None] | None" = None) -> pd.DataFrame:
    """Run cache jobs in a process pool. ``on_done(row)`` is called in the parent as each series finishes."""
    jobs = list(jobs)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []
    if workers <= 1:
        for j in tqdm(jobs, disable=not progress, desc="cache"):
            row = _run_job(j)
            if on_done:
                on_done(row)
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_job, j) for j in jobs]
            for f in tqdm(as_completed(futs), total=len(futs), disable=not progress, desc="cache", mininterval=30):
                row = f.result()
                if on_done:
                    on_done(row)
                rows.append(row)
    return pd.DataFrame(rows)
