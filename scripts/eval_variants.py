#!/usr/bin/env python
"""Inference-time variants on trained runs, no training: window budget, slice-offset passes, per-finding pooling of passes.

    python scripts/eval_variants.py --run /content/drive/MyDrive/rsna-knee/work-v8a-nano-pseudo/runs/v8a-nano-pseudo/fold3 \
        --windows 48 72 96 --offsets 0 0.5 0.25 0.75 --out variants_v8a

The validation split is rebuilt from the run's own config.yaml, so a gold_holdout run is scored on its held-out fold
(report labels, `confident`) and on gold; a train_on_all run only has gold as clean data. Several --run arguments are
rank-averaged into an ensemble before scoring.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.constants import LABELS
from rsna_knee.data.dataset import StudyDataset, collate_studies, view_kwargs
from rsna_knee.infer import load_run
from rsna_knee.metrics import macro_auc, per_label_auc
from rsna_knee.train import amp_dtype, build_records, pick_device, to_device

FOCAL = ("medial_meniscus", "lateral_meniscus", "bakers_cyst", "contusion", "fracture")  # small lesions: one window may carry the evidence


@torch.no_grad()
def predict(model, recs, cfg, k: int, offset: float, device, dtype, workers: int) -> np.ndarray:
    d = cfg.data
    ds = StudyDataset(recs, d.img_size, d.n_slices_eval or d.n_slices, train=False, max_series=d.max_series, seed=cfg.seed,
                      slice_band=tuple(d.slice_band), windows_per_study=k, slice_offset=offset, view=view_kwargs(d))
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=workers, collate_fn=collate_studies, pin_memory=device.type == "cuda")
    out = []
    for batch in dl:
        b = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(b["images"], b["slice_pos"], b["plane_ids"], b["seq_ids"], b["series_batch_idx"], b["batch_size"],
                           slice_valid=b.get("slice_valid"))["logits"]
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def score(P: np.ndarray, T: np.ndarray, G: np.ndarray) -> dict[str, float]:
    res: dict[str, float] = {}
    R = ~G
    if R.any():
        conf = (T <= 0.15) | (T >= 0.85)
        res["confident"] = macro_auc(T[R], P[R], conf[R])
    if G.any():
        m = np.ones_like(T[G], dtype=bool)
        res["gold"] = macro_auc(T[G], P[G], m)
        res.update({f"gold_{k}": v for k, v in per_label_auc(T[G], P[G], m).items()})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="run dir with config.yaml + best.pt (repeat for an ensemble)")
    ap.add_argument("--windows", nargs="+", type=int, default=[48, 72, 96])
    ap.add_argument("--offsets", nargs="+", type=float, default=[0.0, 0.5, 0.25, 0.75])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="variants")
    a = ap.parse_args()

    device = pick_device()
    passes: dict[tuple[int, float], list[np.ndarray]] = {}
    T = G = None
    for run in a.run:
        model, cfg, _ = load_run(run, device)
        dtype = amp_dtype(cfg.optim.amp, device)
        recs = build_records(cfg, "val", cfg.fold)
        t = np.stack([r.targets for r in recs]); g = np.array([r.is_gold for r in recs])
        if T is None:
            T, G, uids = t, g, [r.study_uid for r in recs]
            print(f"{len(recs)} validation studies ({int(g.sum())} gold)")
        assert [r.study_uid for r in recs] == uids, "runs disagree on the validation split: ensemble only runs of one protocol"
        for k, off in itertools.product(a.windows, a.offsets):
            p = predict(model, recs, cfg, k, off, device, dtype, a.workers)
            passes.setdefault((k, off), []).append(pd.DataFrame(p).rank().values / len(p) if len(a.run) > 1 else p)
            print(f"  {run.rstrip('/').split('/')[-2]}: windows={k} offset={off} done", flush=True)
        del model
    P = {key: np.mean(v, axis=0) for key, v in passes.items()}  # ensemble over runs

    focal = [LABELS.index(k) for k in FOCAL]
    rows = []
    for k in a.windows:
        for n_off in sorted({1, 2, len(a.offsets)}):
            stack = np.stack([P[(k, o)] for o in a.offsets[:n_off]])
            rows.append({"windows": k, "passes": n_off, "pool": "mean", **score(stack.mean(0), T, G)})
            if n_off > 1:
                mx = stack.mean(0)
                mx[:, focal] = stack.max(0)[:, focal]
                rows.append({"windows": k, "passes": n_off, "pool": "max for focal", **score(mx, T, G)})
    rep = pd.DataFrame(rows)
    main_cols = [c for c in ("windows", "passes", "pool", "confident", "gold") if c in rep.columns]
    print(rep[main_cols].round(4).to_string(index=False))
    rep.to_csv(f"{a.out}.csv", index=False)
    np.savez_compressed(f"{a.out}.npz", uids=np.array(uids), targets=T, is_gold=G,
                        **{f"p_w{k}_o{o}": v for (k, o), v in P.items()})
    print(f"saved {a.out}.csv / {a.out}.npz")


if __name__ == "__main__":
    main()
