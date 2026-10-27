"""Training loop: AMP, EMA weights, rare-label-aware sampling, grouped-CV OOF, resumable checkpoints."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import TrainConfig, load_config, save_config, to_dict
from .constants import LABELS, PLANE_INDEX
from .data.dataset import SeriesEntry, StudyDataset, StudyRecord, collate_studies, compute_sample_weights, view_kwargs
from .data.transforms import AugConfig, gpu_intensity_aug
from .metrics import macro_auc, per_label_auc
from .models import QueuePairwiseLoss, SoftTargetBCE, StudyModel

log = logging.getLogger("rsna_knee.train")


# ------------------------------------------------------------------------------------ records
def build_records(cfg: TrainConfig, split: str = "all", fold: int | None = None) -> list[StudyRecord]:
    idx = pd.read_parquet(cfg.data.cache_index)
    idx = idx[idx["status"].isin(["ok", "cached"]) & (idx["n_slices"] > 0)]
    plane_ids = {PLANE_INDEX[p] for p in cfg.data.planes}
    idx = idx[idx["plane_id"].isin(plane_ids)]
    table = cfg.data.eval_label_table if (split == "val" and cfg.data.eval_label_table) else cfg.data.label_table
    labels = pd.read_parquet(table)
    labels["study_uid"] = labels["study_uid"].astype(str)
    labels = labels.set_index("study_uid")
    folds = None
    if Path(cfg.data.folds).exists():
        fdf = pd.read_parquet(cfg.data.folds)
        folds = dict(zip(fdf["study_uid"].astype(str), fdf["fold"].astype(int)))
    gparts = gold_parts(labels, cfg.gold_parts) if cfg.gold_train else {}
    kind = "feat" if cfg.model.feature_dim > 0 else "npz"
    ycols, wcols = [f"y_{k}" for k in LABELS], [f"w_{k}" for k in LABELS]
    recs: list[StudyRecord] = []
    for uid, g in idx.groupby("study_uid", sort=False):
        uid = str(uid)
        if uid not in labels.index:
            continue
        f = folds.get(uid, -1) if folds is not None else -1
        row = labels.loc[uid]
        is_gold = bool(row.get("is_gold", False))
        if cfg.train_on_all and cfg.gold_train and is_gold:
            # gold joins training. gold_cv: one part is held out and is the whole validation set; otherwise (final models)
            # gold is in both splits and the validation numbers are in-sample (sanity only).
            if cfg.gold_cv and (split == "val") != (gparts[uid] == cfg.fold % cfg.gold_parts):
                continue
        elif cfg.gold_holdout and cfg.gold_train and is_gold:
            if split == "val":  # teacher / OOF runs: the expert studies always train, the held-out fold alone validates
                continue
        elif cfg.train_on_all:  # all non-gold studies train; the gold studies are the only validation
            if (split == "train") == is_gold:
                continue
        elif cfg.gold_holdout:  # val = held-out fold (report labels, n~900) + all gold (expert labels); gold never trains
            in_val = is_gold or f == fold
            if (split == "train" and in_val) or (split == "val" and not in_val):
                continue
        else:
            if split == "train" and f == fold:
                continue
            if split == "val" and f != fold:
                continue
        t = row[ycols].values.astype(np.float32)
        w = row[wcols].values.astype(np.float32)
        series = [SeriesEntry(source=_resolve(str(r.path), cfg.data.cache_root), kind=kind, plane_id=int(r.plane_id), seq_id=int(r.seq_id),
                              n_slices=int(r.n_slices)) for r in g.itertuples()]
        recs.append(StudyRecord(uid, series, t, w, bool(row.get("is_gold", False)), group=str(f)))
    return recs


def gold_parts(labels: pd.DataFrame, n: int) -> dict[str, int]:
    """Deterministic split of the expert-labelled studies: sorted by number of positive findings, dealt round-robin."""
    g = labels[labels["is_gold"].astype(bool)]
    npos = g[[f"y_{k}" for k in LABELS]].sum(axis=1)
    order = sorted(g.index, key=lambda u: (-float(npos[u]), str(u)))
    return {str(u): i % n for i, u in enumerate(order)}


def _resolve(path: str, root: str) -> str:
    return path if (not root or os.path.isabs(path)) else os.path.join(root, path)


# ------------------------------------------------------------------------------------ helpers
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(mode: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or mode == "off":
        return None
    # is_bf16_supported() is True on Turing (T4) through slow emulation: require Ampere+ for the auto choice
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8
    if mode == "bf16" or (mode == "auto" and native_bf16):
        return torch.bfloat16
    return torch.float16


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def build_model(cfg: TrainConfig) -> StudyModel:
    m = cfg.model
    return StudyModel(backbone=m.backbone, pretrained=m.pretrained, checkpoint_path=m.checkpoint_path,
                      feature_dim=m.feature_dim, d_model=m.d_model, n_heads=m.n_heads, series_layers=m.series_layers,
                      study_layers=m.study_layers, dropout=m.dropout, drop_path=m.drop_path,
                      grad_checkpoint=m.grad_checkpoint, channels_last=m.channels_last and m.feature_dim == 0,
                      spatial_head=m.spatial_head, spatial_dim=m.spatial_dim, pool=m.pool)


def make_aug(cfg: TrainConfig) -> AugConfig:
    d = cfg.data
    return AugConfig(rotate=d.aug_rotate, scale=tuple(d.aug_scale), translate=d.aug_translate, hflip=d.aug_hflip,
                     brightness=d.aug_brightness, contrast=d.aug_contrast, gamma=tuple(d.aug_gamma), noise=d.aug_noise,
                     erase_p=d.aug_erase_p)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype | None,
             tta_hflip: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    probs, targets, weights, golds, uids = [], [], [], [], []
    for batch in loader:
        b = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(b["images"], b["slice_pos"], b["plane_ids"], b["seq_ids"], b["series_batch_idx"], b["batch_size"], slice_valid=b.get("slice_valid"))["logits"]
            p = torch.sigmoid(logits)
            if tta_hflip and b["images"].ndim == 5:
                logits2 = model(b["images"].flip(-1), b["slice_pos"], b["plane_ids"], b["seq_ids"], b["series_batch_idx"], b["batch_size"], slice_valid=b.get("slice_valid"))["logits"]
                p = 0.5 * (p + torch.sigmoid(logits2))
        probs.append(p.float().cpu().numpy())
        targets.append(batch["targets"].numpy())
        weights.append(batch["weights"].numpy())
        golds.append(batch["is_gold"].numpy())
        uids.extend(batch["study_uids"])
    P, T, W, G = (np.concatenate(x) for x in (probs, targets, weights, golds))
    G = G.astype(bool)
    mask = np.isfinite(T) & (W > 0)
    R = ~G if (~G).any() else np.ones_like(G)  # report-labelled rows (all rows when the val set is gold only)
    metrics: dict[str, Any] = {"macro_auc": macro_auc(T[R], P[R], mask[R]), "per_label": per_label_auc(T[R], P[R], mask[R]), "n_val": int(R.sum())}
    if G.any():
        metrics["gold_macro_auc"] = macro_auc(T[G], P[G], mask[G])
        metrics["gold_n"] = int(G.sum())
        if (~G).any():
            metrics["gold_per_label"] = per_label_auc(T[G], P[G], mask[G])
    # confident-only AUC (report states present / absent) - far less noisy than the soft targets
    conf = mask & ((T <= 0.15) | (T >= 0.85))
    metrics["confident_macro_auc"] = macro_auc(T[R], P[R], conf[R])
    metrics["confident_per_label"] = per_label_auc(T[R], P[R], conf[R])
    oof = pd.DataFrame({"study_uid": uids, "is_gold": G})
    for j, k in enumerate(LABELS):
        oof[f"p_{k}"] = P[:, j]
        oof[f"y_{k}"] = T[:, j]
    return metrics, oof


# ------------------------------------------------------------------------------------ train
def train_fold(cfg: TrainConfig) -> dict[str, Any]:
    seed_everything(cfg.seed + cfg.fold)
    device = pick_device()
    dtype = amp_dtype(cfg.optim.amp, device)
    torch.backends.cudnn.benchmark = True
    out_dir = Path(cfg.output_dir) / f"fold{cfg.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("device=%s amp=%s out=%s", device, dtype, out_dir)

    train_recs = build_records(cfg, "train", cfg.fold)
    val_recs = build_records(cfg, "val", cfg.fold)
    if cfg.max_train_studies:
        train_recs = train_recs[: cfg.max_train_studies]
        val_recs = val_recs[: max(8, cfg.max_train_studies // 4)]
    log.info("train studies=%d val studies=%d", len(train_recs), len(val_recs))
    if not train_recs or not val_recs:
        raise SystemExit(f"empty split (train={len(train_recs)}, val={len(val_recs)}): check cache_index/label_table/folds ids"
                         f" and train_on_all={cfg.train_on_all}")
    T_train = np.stack([r.targets for r in train_recs])
    priors = np.nanmean(np.where(np.isfinite(T_train), T_train, np.nan), axis=0)
    (out_dir / "priors.json").write_text(json.dumps({k: float(v) for k, v in zip(LABELS, priors)}))

    d = cfg.data
    aug_cfg = make_aug(cfg)
    ds_tr = StudyDataset(train_recs, d.img_size, d.n_slices, train=True, aug=aug_cfg, slice_jitter=d.slice_jitter,
                         max_series=d.max_series, seed=cfg.seed, slice_band=tuple(d.slice_band), windows_per_study=d.windows_per_study, view=view_kwargs(d))
    ds_va = StudyDataset(val_recs, d.img_size, d.n_slices_eval, train=False, max_series=d.max_series, seed=cfg.seed,
                         slice_band=tuple(d.slice_band), windows_per_study=d.windows_per_study_eval or d.windows_per_study, view=view_kwargs(d))
    sampler = None
    gold_tr = np.array([r.is_gold for r in train_recs])
    if d.rare_boost_power > 0 or (gold_tr.any() and cfg.gold_repeat != 1.0):
        sw = compute_sample_weights(T_train, d.rare_boost_power) if d.rare_boost_power > 0 else np.ones(len(train_recs))
        sw[gold_tr] *= cfg.gold_repeat
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), num_samples=len(ds_tr), replacement=True)
    if gold_tr.any():
        log.info("gold in training: %d studies, sampling x%.1f%s", int(gold_tr.sum()), cfg.gold_repeat,
                 "" if cfg.gold_cv else " (validation is IN-SAMPLE: all gold trains)")
    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=cfg.optim.batch_size, sampler=sampler, shuffle=sampler is None,
                       num_workers=d.num_workers, collate_fn=collate_studies, drop_last=True, pin_memory=pin,
                       persistent_workers=d.num_workers > 0)
    dl_va = DataLoader(ds_va, batch_size=max(1, cfg.optim.batch_size), shuffle=False, num_workers=d.num_workers,
                       collate_fn=collate_studies, pin_memory=pin)

    model = build_model(cfg).to(device)
    if cfg.init_from and not (cfg.resume and (out_dir / "last.pt").exists()):
        n = warm_start(model, cfg.init_from, cfg.init_backbone_only)
        log.info("warm start from %s: %d tensors (%s)", cfg.init_from, n, "backbone only" if cfg.init_backbone_only else "whole model")
    if cfg.compile and model.backbone is not None:
        model.backbone = torch.compile(model.backbone)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.optim.ema_decay), use_buffers=True)
    opt = torch.optim.AdamW(model.param_groups(cfg.optim.lr_backbone, cfg.optim.lr_head, cfg.optim.weight_decay))
    steps_per_epoch = max(1, len(dl_tr) // cfg.optim.accum)
    total_steps = cfg.optim.epochs * steps_per_epoch
    warmup = max(1, int(total_steps * cfg.optim.warmup_frac))

    def lr_lambda(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    criterion = SoftTargetBCE(cfg.optim.label_smoothing)
    o = cfg.optim
    pair = QueuePairwiseLoss(len(LABELS), o.pair_queue, lo=o.pair_lo, hi=o.pair_hi, tau=o.pair_tau).to(device) if o.pair_weight > 0 else None

    start_epoch, best_auc, step = 0, -1.0, 0
    last_path = out_dir / "last.pt"
    if cfg.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, step = ck["epoch"] + 1, ck["best_auc"], ck["step"]
        log.info("resumed from epoch %d (best %.4f)", ck["epoch"], best_auc)

    history: list[dict[str, Any]] = []
    topk: list[tuple[float, dict[str, torch.Tensor]]] = []
    t_run, epoch_times = time.time(), []
    for epoch in range(start_epoch, cfg.optim.epochs):
        if cfg.max_hours and epoch_times and (time.time() - t_run + 1.15 * max(epoch_times)) > cfg.max_hours * 3600:
            log.warning("time budget %.1fh: stopping before epoch %d", cfg.max_hours, epoch)
            break
        model.train()
        ds_tr.set_epoch(epoch)
        t0, run_loss, n_loss = time.time(), 0.0, 0
        t_data, t_mark = 0.0, time.time()
        log_every = 10 if epoch == start_epoch else 50
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl_tr):
            t_data += time.time() - t_mark  # time spent waiting for the loader (CPU-bound if this dominates)
            b = to_device(batch, device)
            if b["images"].ndim == 5 and b["images"].shape[0] > 0:
                b["images"] = gpu_intensity_aug(b["images"], aug_cfg)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                out = model(b["images"], b["slice_pos"], b["plane_ids"], b["seq_ids"], b["series_batch_idx"], b["batch_size"], slice_valid=b.get("slice_valid"))
                loss = criterion(out["logits"], b["targets"], b["weights"])
                if pair is not None:  # always run it so the queue is warm; it joins the loss only in the last epochs
                    pl = pair(out["logits"], b["targets"])
                    if epoch >= o.epochs - o.pair_last_epochs:
                        loss = loss + o.pair_weight * pl
                loss = loss / cfg.optim.accum
            scaler.scale(loss).backward()
            run_loss += float(loss.detach()) * cfg.optim.accum
            n_loss += 1
            if (i + 1) % cfg.optim.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update_parameters(model)
                step += 1
                if step % log_every == 0:
                    el = time.time() - t0
                    log.info("ep %d step %d/%d loss %.4f lr %.2e | %.2f studies/s, loader wait %.0f%%, eta %.0f min", epoch, step,
                             (epoch + 1) * steps_per_epoch, run_loss / n_loss, sched.get_last_lr()[-1],
                             (i + 1) * cfg.optim.batch_size / el, 100 * t_data / el, el / (i + 1) * (len(dl_tr) - i - 1) / 60)
            t_mark = time.time()
        train_loss = run_loss / max(1, n_loss)
        rec: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "time_s": time.time() - t0}
        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.optim.epochs:
            metrics, oof = evaluate(ema.module, dl_va, device, dtype)
            rec.update(metrics)
            oof.to_parquet(out_dir / "oof.parquet", index=False)
            sel = selection_score(metrics, cfg.select_by, epoch)
            if cfg.optim.topk_avg > 1:
                topk.append((sel, {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}))
                topk.sort(key=lambda t: -t[0]); del topk[cfg.optim.topk_avg:]
            if (math.isfinite(sel) and sel > best_auc) or not (out_dir / "best.pt").exists():  # NaN metric (tiny val) still saves once
                best_auc = sel if math.isfinite(sel) else best_auc
                torch.save({"model": ema.module.state_dict(), "config": to_dict(cfg), "epoch": epoch,
                            "metrics": {k: v for k, v in metrics.items()}}, out_dir / "best.pt")
                oof.to_parquet(out_dir / "oof_best.parquet", index=False)
            log.info("epoch %d: loss %.4f | val(n=%d) labels %.4f confident %.4f | gold(n=%d) %.4f | best[%s] %.4f", epoch, train_loss,
                     metrics["n_val"], metrics["macro_auc"], metrics["confident_macro_auc"], metrics.get("gold_n", 0),
                     metrics.get("gold_macro_auc", float("nan")), cfg.select_by, best_auc)
            log.info("confident per-label: %s", {k: round(v, 3) for k, v in metrics["confident_per_label"].items()})
            if "gold_per_label" in metrics:
                log.info("gold per-label: %s", {k: round(v, 3) for k, v in metrics["gold_per_label"].items()})
        history.append(rec)
        epoch_times.append(time.time() - t0)
        with open(out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_auc": best_auc, "step": step}, last_path)
    if cfg.optim.topk_avg > 1 and len(topk) > 1:  # weight-space average of the best epochs (Raptor recipe)
        avg = average_state_dicts([sd for _, sd in topk])
        ema.module.load_state_dict(avg)
        metrics, oof = evaluate(ema.module, dl_va, device, dtype)
        sel = selection_score(metrics, cfg.select_by, cfg.optim.epochs)
        log.info("top-%d weight average: val macro AUC %.4f gold %.4f (selection %.4f vs best single %.4f)", len(topk),
                 metrics["macro_auc"], metrics.get("gold_macro_auc", float("nan")), sel, best_auc)
        torch.save({"model": avg, "config": to_dict(cfg), "epoch": -1, "metrics": {k: v for k, v in metrics.items()}}, out_dir / f"avg_top{len(topk)}.pt")
        if sel >= best_auc or cfg.select_by == "last":
            torch.save({"model": avg, "config": to_dict(cfg), "epoch": -1, "metrics": {k: v for k, v in metrics.items()}}, out_dir / "best.pt")
            oof.to_parquet(out_dir / "oof_best.parquet", index=False)
            best_auc = sel
    return {"best_auc": best_auc, "history": history, "out_dir": str(out_dir)}


def warm_start(model: torch.nn.Module, path: str, backbone_only: bool = True) -> int:
    """Initialise from a checkpoint of one of our own runs. Never use a run that saw the current validation studies."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    if backbone_only:
        sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    own = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    if not sd:
        raise ValueError(f"nothing to load from {path}: no matching tensors")
    model.load_state_dict(sd, strict=False)
    return len(sd)


def selection_score(metrics: dict[str, Any], select_by: str, epoch: int) -> float:
    if select_by == "gold" and math.isfinite(metrics.get("gold_macro_auc", float("nan"))):
        return float(metrics["gold_macro_auc"])
    if select_by == "confident" and math.isfinite(metrics.get("confident_macro_auc", float("nan"))):
        return float(metrics["confident_macro_auc"])
    if select_by == "last":
        return float(epoch)  # later epochs always win -> top-k = last k epochs
    return float(metrics["macro_auc"])


def average_state_dicts(sds: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sds[0].items():
        if v.dtype.is_floating_point:
            out[k] = torch.stack([sd[k].float() for sd in sds]).mean(0).to(v.dtype)
        else:
            out[k] = v.clone()  # num_batches_tracked etc.
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train one CV fold of the study transformer")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    if a.fold is not None:
        cfg.fold = a.fold
    res = train_fold(cfg)
    print(json.dumps({"best_auc": res["best_auc"], "out_dir": res["out_dir"]}))


if __name__ == "__main__":
    main()
