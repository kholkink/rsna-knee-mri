#!/usr/bin/env python
"""Frozen-backbone slice features for the fast iteration path (train the study transformer in minutes).

    python scripts/extract_features.py --cache-index cache/cache_index.parquet --out features/dinov2b \
        --backbone vit_base_patch14_dinov2.lvd142m --img-size 518 --n-slices 32
Then train with model.feature_dim=<F> and data.cache_index=features/dinov2b/feature_index.parquet
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rsna_knee.data.dataset import NpzSeries, sample_centers
from rsna_knee.models.backbone import create_backbone
from rsna_knee.models.study_transformer import IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--img-size", type=int, default=0, help="0 -> cache size")
    ap.add_argument("--n-slices", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dim = create_backbone(a.backbone, pretrained=True, checkpoint_path=a.checkpoint, drop_path=0.0)
    model.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    idx = pd.read_parquet(a.cache_index)
    idx = idx[idx["status"].isin(["ok", "cached"])].copy()
    rng = np.random.default_rng(0)
    paths = []
    for r in tqdm(idx.itertuples(), total=len(idx), desc="features"):
        out_path = os.path.join(a.out, str(r.study_uid), f"{r.series_uid}.npz")
        paths.append(out_path)
        if os.path.exists(out_path):
            continue
        src = NpzSeries(str(r.path))
        n = src.n
        centers = sample_centers(n, min(a.n_slices, n), False, rng)
        nb = np.stack([centers - 1, centers, centers + 1], 1).clip(0, n - 1)
        needed = np.unique(nb)
        slices = src.decode(needed.tolist())
        lookup = {int(i): k for k, i in enumerate(needed)}
        imgs = slices[np.vectorize(lookup.get)(nb)]  # (N,3,H,W)
        x = torch.from_numpy(imgs).to(device).float().div_(255.0)
        if a.img_size and x.shape[-1] != a.img_size:
            x = torch.nn.functional.interpolate(x, size=(a.img_size, a.img_size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feats = torch.cat([model(x[i:i + a.batch]) for i in range(0, x.shape[0], a.batch)]).half().cpu().numpy()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, feats=feats, positions=src.positions[centers].astype(np.float32))
    idx["path"] = paths
    idx["n_slices"] = idx["n_slices"].clip(upper=a.n_slices)
    idx.to_parquet(os.path.join(a.out, "feature_index.parquet"), index=False)
    print(f"feature dim = {dim}; index -> {os.path.join(a.out, 'feature_index.parquet')}")


if __name__ == "__main__":
    main()
