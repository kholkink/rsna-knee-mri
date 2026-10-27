"""Locate competition files on Kaggle (mount layout differs between images / over time)."""
from __future__ import annotations

import glob
import os

COMP_SLUG = "rsna-knee-abnormality-detection"
IMAGE_ROOT_CANDIDATES = {"train": ["train_images", "train_series", "train", "images/train"],
                         "test": ["test_images", "test_series", "test", "images/test"]}


def find_comp_root(hint: str | None = None) -> str:
    """Directory holding train.csv / sample_submission.csv."""
    cands = [hint, os.environ.get("RSNA_COMP_ROOT"), f"/kaggle/input/{COMP_SLUG}",
             f"/kaggle/input/competitions/{COMP_SLUG}"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "sample_submission.csv")):
            return c
    for depth in ("*", "*/*", "*/*/*"):
        for p in sorted(glob.glob(f"/kaggle/input/{depth}/sample_submission.csv")):
            if "knee" in p.lower():
                return os.path.dirname(p)
    raise FileNotFoundError(f"competition root not found (tried {[c for c in cands if c]})")


def find_image_root(comp_root: str, split: str) -> str:
    for c in IMAGE_ROOT_CANDIDATES[split]:
        p = os.path.join(comp_root, c)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"no {split} image dir under {comp_root}: {sorted(os.listdir(comp_root))}")
