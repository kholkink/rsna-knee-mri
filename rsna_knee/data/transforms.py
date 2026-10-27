"""Series-consistent augmentation (same geometric params for every slice of a series)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugConfig:
    rotate: float = 10.0
    scale: tuple[float, float] = (0.9, 1.1)
    translate: float = 0.05
    hflip: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    gamma: tuple[float, float] = (0.8, 1.25)
    noise: float = 0.02
    erase_p: float = 0.2
    erase_n: int = 2
    erase_size: tuple[float, float] = (0.05, 0.2)


def sample_affine(rng: np.random.Generator, size: int, cfg: AugConfig) -> np.ndarray | None:
    """2x3 affine matrix (cv2 convention) or None when identity."""
    ang = rng.uniform(-cfg.rotate, cfg.rotate) if cfg.rotate > 0 else 0.0
    sc = rng.uniform(*cfg.scale) if cfg.scale != (1.0, 1.0) else 1.0
    tx = rng.uniform(-cfg.translate, cfg.translate) * size if cfg.translate > 0 else 0.0
    ty = rng.uniform(-cfg.translate, cfg.translate) * size if cfg.translate > 0 else 0.0
    flip = rng.random() < cfg.hflip
    if ang == 0.0 and sc == 1.0 and tx == 0.0 and ty == 0.0 and not flip:
        return None
    c = size / 2.0
    M = cv2.getRotationMatrix2D((c, c), ang, sc)  # 2x3
    M[0, 2] += tx
    M[1, 2] += ty
    if flip:
        F = np.array([[-1.0, 0.0, size - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        M3 = np.vstack([M, [0.0, 0.0, 1.0]]) @ F
        M = M3[:2]
    return M


def apply_affine(imgs: np.ndarray, M: np.ndarray | None) -> np.ndarray:
    """imgs: (K,H,W) uint8 -> same, warped with border 0."""
    if M is None:
        return imgs
    h, w = imgs.shape[1:]
    out = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        out[i] = cv2.warpAffine(imgs[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def intensity_aug(x: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """x: float32 in [0,1], any shape. Same params across the whole series."""
    if cfg.contrast > 0 or cfg.brightness > 0:
        c = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        b = rng.uniform(-cfg.brightness, cfg.brightness)
        x = (x - 0.5) * c + 0.5 + b
    if cfg.gamma != (1.0, 1.0):
        g = math.exp(rng.uniform(math.log(cfg.gamma[0]), math.log(cfg.gamma[1])))
        x = np.clip(x, 0.0, 1.0) ** g
    if cfg.noise > 0:
        x = x + rng.normal(0.0, cfg.noise, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0, out=x)


def random_erase(x: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """x: (..., H, W); boxes are shared across leading dims (i.e. across the slices of a series)."""
    if cfg.erase_p <= 0 or rng.random() >= cfg.erase_p:
        return x
    h, w = x.shape[-2:]
    for _ in range(int(rng.integers(1, cfg.erase_n + 1))):
        fh = rng.uniform(*cfg.erase_size)
        fw = rng.uniform(*cfg.erase_size)
        bh, bw = max(1, int(fh * h)), max(1, int(fw * w))
        y0 = int(rng.integers(0, max(1, h - bh)))
        x0 = int(rng.integers(0, max(1, w - bw)))
        x[..., y0:y0 + bh, x0:x0 + bw] = 0.0
    return x


def gpu_intensity_aug(images, cfg: AugConfig):
    """(M,N,3,H,W) uint8/float tensor on the GPU -> float [0,1]; one parameter set per series.

    The CPU version costs ~1 s / study on Kaggle's 4 cores (gamma + gaussian noise over 10M floats per series).
    """
    import torch

    x = images.float().div_(255.0) if images.dtype == torch.uint8 else images
    shape = (x.shape[0], 1, 1, 1, 1)
    dev = x.device

    def u(lo: float, hi: float):
        return torch.empty(shape, device=dev).uniform_(lo, hi)

    if cfg.contrast > 0 or cfg.brightness > 0:
        x = (x - 0.5) * (1.0 + u(-cfg.contrast, cfg.contrast)) + 0.5 + u(-cfg.brightness, cfg.brightness)
    if tuple(cfg.gamma) != (1.0, 1.0):
        g = torch.exp(u(math.log(cfg.gamma[0]), math.log(cfg.gamma[1])))
        x = x.clamp_(0.0, 1.0).pow(g)
    if cfg.noise > 0:
        x = x + torch.randn_like(x) * cfg.noise
    return x.clamp_(0.0, 1.0)
