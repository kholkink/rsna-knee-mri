"""Offline inference for the Kaggle notebook: DICOM -> ensemble of fold models -> submission.csv.

Constraints handled here: no internet, <= 9 h wall clock, unknown test size (~1.3k studies),
arbitrary protocols per site. Every study that fails to load gets the training prior instead of
crashing the run.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig, load_config
from .constants import ID_COL, LABELS, N_LABELS, SERIES_COL, resolve_label_columns
from .data.dataset import SeriesEntry, StudyDataset, StudyRecord, collate_studies, view_kwargs
from .dicom import SeriesMeta, knee_side, scan_series_header
from .paths import find_comp_root, find_image_root
from .series_select import SelectionPolicy, select_series
from .train import amp_dtype, build_model, pick_device, to_device

log = logging.getLogger("rsna_knee.infer")

@dataclass
class InferenceConfig:
    comp_root: str = "/kaggle/input/rsna-knee-abnormality-detection"
    run_dirs: list[str] = field(default_factory=list)   # each holds config.yaml + best.pt (+ priors.json)
    out_path: str = "submission.csv"
    tta_hflip: bool = True
    batch_size: int = 2
    num_workers: int = 4
    time_budget_s: float = 8.0 * 3600
    policy: SelectionPolicy = field(default_factory=SelectionPolicy)
    n_slices_eval: int = 0   # 0 -> from run config
    max_series: int = 0      # 0 -> from run config
    limit: int = 0           # debug
    ensemble: str = "rank"   # "rank": average per-label ranks across models (AUC only reads order) | "mean": probabilities
    slice_offsets: tuple[float, ...] = (0.0,)  # slice-offset TTA: extra passes with the sampling grid shifted (e.g. 0.5 step)
    label_mix: dict[str, dict[str, float]] = field(default_factory=dict)  # {"synovitis": {"effusion": 0.5}}: see mix_labels
    split: str = "test"      # "train": run the SAME live-DICOM path on training studies (train/serve skew check against the cache path)
    study_uids: tuple[str, ...] = ()  # restrict to these studies (with split="train": e.g. the expert-labelled ones)


def load_run(run_dir: str, device: torch.device) -> tuple[torch.nn.Module, TrainConfig, np.ndarray]:
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    cfg.model.pretrained = False
    cfg.model.checkpoint_path = ""
    model = build_model(cfg)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    pri_path = os.path.join(run_dir, "priors.json")
    priors = np.full(N_LABELS, 0.5, dtype=np.float32)
    if os.path.exists(pri_path):
        d = json.loads(Path(pri_path).read_text())
        priors = np.asarray([d.get(k, 0.5) for k in LABELS], dtype=np.float32)
    return model, cfg, priors


def scan_study(image_root: str, study_uid: str, series_uids: list[str] | None) -> list[SeriesMeta]:
    sdir = os.path.join(image_root, study_uid)
    if series_uids is None or not series_uids:
        series_uids = sorted(d for d in os.listdir(sdir)) if os.path.isdir(sdir) else []
    metas = []
    for suid in series_uids:
        p = os.path.join(sdir, suid)
        if os.path.isdir(p):
            metas.append(scan_series_header(p, study_uid, suid))
    return metas


def build_test_records(image_root: str, test_df: pd.DataFrame, test_series_df: pd.DataFrame | None,
                       policy: SelectionPolicy, workers: int = 8, lat_normalize: bool = False) -> list[StudyRecord]:
    uids = test_df[ID_COL].astype(str).tolist()
    by_study: dict[str, list[str]] = {}
    if test_series_df is not None and SERIES_COL in test_series_df:
        for s, g in test_series_df.groupby(ID_COL):
            by_study[str(s)] = g[SERIES_COL].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        metas_list = list(ex.map(lambda u: scan_study(image_root, u, by_study.get(u)), uids))
    recs = []
    for uid, metas in zip(uids, metas_list):
        chosen = select_series(metas, policy)
        if not chosen and metas:  # nothing passed the filters - relax and take what we have
            chosen = select_series(metas, SelectionPolicy(min_slices=1, max_slices=10_000, allow_unknown_plane=True,
                                                          max_per_plane=policy.max_per_plane, max_total=policy.max_total))
        mirror = lat_normalize and knee_side([m.center_x for m in metas]) == "right"
        entries = [SeriesEntry(source=m.series_dir, kind="dicom", plane_id=m.plane_id, seq_id=m.seq_id,
                               n_slices=m.n_slices, meta=m, mirror=mirror) for m in chosen]
        recs.append(StudyRecord(uid, entries))
    return recs


@torch.no_grad()
def predict(icfg: InferenceConfig) -> pd.DataFrame:
    t_start = time.time()
    device = pick_device()
    models = [load_run(r, device) for r in icfg.run_dirs]
    if not models:
        raise ValueError("no run_dirs")
    cfg0 = models[0][1]
    priors = np.mean([m[2] for m in models], axis=0)
    dtype = amp_dtype(cfg0.optim.amp, device)
    icfg.comp_root = find_comp_root(icfg.comp_root)
    test_df = pd.read_csv(os.path.join(icfg.comp_root, f"{icfg.split}.csv"))
    if icfg.study_uids:
        test_df = test_df[test_df[ID_COL].astype(str).isin(set(icfg.study_uids))].reset_index(drop=True)
    ts_path = os.path.join(icfg.comp_root, f"{icfg.split}_series.csv")
    test_series = pd.read_csv(ts_path) if os.path.exists(ts_path) else None
    if icfg.limit:
        test_df = test_df.head(icfg.limit)
    image_root = find_image_root(icfg.comp_root, icfg.split)
    d0 = cfg0.data
    recs = build_test_records(image_root, test_df, test_series, icfg.policy, workers=max(4, icfg.num_workers),
                              lat_normalize=d0.lat_normalize)
    n_slices = icfg.n_slices_eval or cfg0.data.n_slices_eval
    max_series = icfg.max_series or cfg0.data.max_series
    per_model: list[dict[str, np.ndarray]] = [{} for _ in models]
    failed: set[str] = set()
    tta = icfg.tta_hflip and not d0.lat_normalize  # a horizontal flip would undo the left/right normalisation
    active = list(range(len(models)))
    offsets = list(icfg.slice_offsets) or [0.0]
    dls = [DataLoader(StudyDataset(recs, d0.img_size, n_slices, train=False, max_series=max_series, dicom_max_slices=d0.cache_max_slices,
                                   slice_band=tuple(d0.slice_band), crop_mm=d0.crop_mm, canonical=d0.canonical,
                                   windows_per_study=d0.windows_per_study_eval or d0.windows_per_study, slice_offset=off, view=view_kwargs(d0)),
                      batch_size=icfg.batch_size, shuffle=False, num_workers=icfg.num_workers, collate_fn=collate_studies,
                      pin_memory=device.type == "cuda") for off in offsets]
    n_batches = len(dls[0]) * len(dls)
    acc_views: list[dict[str, list[np.ndarray]]] = [{} for _ in models]
    for bi, batch in enumerate(b for dl in dls for b in dl):
        # time-budget guard: shed TTA, then ensemble members, before we risk the 9 h limit
        elapsed = time.time() - t_start
        if bi > 3:
            projected = elapsed / bi * n_batches
            if projected > icfg.time_budget_s and tta:
                log.warning("projected %.0fs > budget: disabling TTA", projected)
                tta = False
            elif projected > icfg.time_budget_s and len(active) > 1:
                log.warning("projected %.0fs > budget: dropping ensemble member", projected)
                active = active[:-1]
        b = to_device(batch, device)
        counts = torch.bincount(b["series_batch_idx"], minlength=b["batch_size"]).cpu().numpy()
        for mi in active:
            model = models[mi][0]
            acc = np.zeros((b["batch_size"], N_LABELS), dtype=np.float32)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                views = [b["images"]] + ([b["images"].flip(-1)] if (tta and b["images"].ndim == 5) else [])
                for v in views:
                    logits = model(v, b["slice_pos"], b["plane_ids"], b["seq_ids"], b["series_batch_idx"], b["batch_size"], slice_valid=b.get("slice_valid"))["logits"]
                    acc += torch.sigmoid(logits).float().cpu().numpy()
            acc /= len(views)
            for k, uid in enumerate(batch["study_uids"]):
                acc_views[mi].setdefault(uid, []).append(acc[k])
        for k, uid in enumerate(batch["study_uids"]):
            if counts[k] == 0:
                failed.add(uid)
        if bi % 20 == 0:
            log.info("batch %d/%d elapsed %.0fs", bi, n_batches, time.time() - t_start)
    uids = [r.study_uid for r in recs]
    for mi in range(len(models)):  # average the slice-offset passes that completed for each study
        per_model[mi] = {u: np.mean(v, axis=0) for u, v in acc_views[mi].items()}
    full = [mi for mi in range(len(models)) if len(per_model[mi]) == len(uids)]  # members dropped mid-run are excluded
    mats = [np.stack([per_model[mi][u] for u in uids]) for mi in full]
    if icfg.ensemble == "rank" and len(mats) > 1 and len(uids) > 1:
        ranks = [pd.DataFrame(m).rank(method="average").values / len(uids) for m in mats]
        final = np.mean(ranks, axis=0)
    else:
        final = np.mean(mats, axis=0)
    final = mix_labels(final, icfg.label_mix)
    if failed:  # studies with no readable series: neutral rank / training prior, never a crash
        fill = np.full(N_LABELS, 0.5) if (icfg.ensemble == "rank" and len(mats) > 1) else priors
        for i, u in enumerate(uids):
            if u in failed:
                final[i] = fill
        log.warning("%d studies had no readable series", len(failed))
    out = pd.DataFrame({ID_COL: uids})
    for j, k in enumerate(LABELS):
        out[k] = final[:, j]
    out.attrs["members"] = {icfg.run_dirs[mi]: mats[j] for j, mi in enumerate(full)}  # raw per-model probabilities (diagnostics)
    out.attrs["failed"] = sorted(failed)
    return out


def mix_labels(final: np.ndarray, label_mix: dict[str, dict[str, float]]) -> np.ndarray:
    """Cross-finding rank mixing: target <- (1 - sum w) * rank(target) + sum w * rank(source).

    Experts call synovitis mostly where there is effusion (22 of 27 gold positives) while reports rarely name it, so the
    effusion output ranks expert synovitis better than the synovitis head itself (gold AUC 0.773 vs 0.735, 9 of 9 models).
    """
    if not label_mix or len(final) < 2:
        return final
    ranks = pd.DataFrame(final).rank(method="average").values / len(final)
    out = final.copy()
    for tgt, srcs in label_mix.items():
        j = LABELS.index(tgt)
        out[:, j] = (1.0 - sum(srcs.values())) * ranks[:, j] + sum(w * ranks[:, LABELS.index(s)] for s, w in srcs.items())
    return out


def write_submission(pred: pd.DataFrame, sample_submission_path: str, out_path: str) -> pd.DataFrame:
    sub = pd.read_csv(sample_submission_path)
    cols = resolve_label_columns(sub.columns)
    missing = [k for k in LABELS if k not in cols]
    if missing:
        raise ValueError(f"cannot map labels to sample_submission columns: {missing}; columns={list(sub.columns)}")
    id_col = ID_COL if ID_COL in sub.columns else sub.columns[0]
    pred = pred.set_index(ID_COL)
    for k in LABELS:
        sub[cols[k]] = sub[id_col].astype(str).map(pred[k]).fillna(sub[cols[k]]).astype(float).clip(0, 1)
    sub.to_csv(out_path, index=False)
    return sub
