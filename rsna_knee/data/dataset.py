"""Study-level dataset: variable number of series per study, N sampled 2.5D slices per series."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import N_LABELS
from ..dicom import SeriesMeta, late_crop, preprocess_series, smart_shift_mm
from .transforms import AugConfig, apply_affine, intensity_aug, random_erase, sample_affine

log = logging.getLogger(__name__)


@dataclass
class SeriesEntry:
    source: str          # npz path or DICOM directory
    kind: str            # "npz" | "dicom"
    plane_id: int
    seq_id: int
    n_slices: int = 0
    meta: SeriesMeta | None = None
    mirror: bool = False


@dataclass
class StudyRecord:
    study_uid: str
    series: list[SeriesEntry]
    targets: np.ndarray | None = None   # (L,) soft targets
    weights: np.ndarray | None = None   # (L,) per-label loss weights
    is_gold: bool = False
    group: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class NpzSeries:
    """Lazy JPEG-in-npz reader: decode only the requested slices."""

    def __init__(self, path: str):
        with np.load(path) as z:
            self.data, self.offsets, self.positions = z["jpeg"], z["offsets"], z["positions"]
            self.shape = tuple(int(v) for v in z["shape"])

    @property
    def n(self) -> int:
        return self.shape[0]

    def decode(self, indices: Sequence[int]) -> np.ndarray:
        out = np.empty((len(indices), self.shape[1], self.shape[2]), dtype=np.uint8)
        for k, i in enumerate(indices):
            img = cv2.imdecode(self.data[self.offsets[i]:self.offsets[i + 1]], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("jpeg decode failed")
            out[k] = img
        return out


def sample_centers(n: int, k: int, train: bool, rng: np.random.Generator, jitter: float = 0.15,
                   band: tuple[float, float] = (0.0, 1.0), offset: float = 0.0) -> np.ndarray:
    """k slice indices spread over the ``band`` of the series; random sub-range + jitter at train time.

    ``offset`` (eval only, in units of the inter-sample step) shifts the deterministic grid: slice-offset TTA.
    """
    if n <= 0:
        return np.zeros(k, dtype=int)
    lo, hi = band[0] * (n - 1), band[1] * (n - 1)
    if hi - lo < 2:
        lo, hi = 0.0, float(n - 1)
    if not train and offset and k > 1:
        step = (hi - lo) / (k - 1)
        lo, hi = lo + offset * step, hi + offset * step
        hi = min(hi, float(n - 1)); lo = max(0.0, min(lo, hi - 1.0))
    if train and jitter > 0 and n > 2:
        frac = rng.uniform(1.0 - jitter, 1.0)
        span = max(1.0, (hi - lo) * frac)
        start = rng.uniform(lo, hi - span) if hi - lo - span > 0 else lo
        lo, hi = start, start + span
    centers = np.linspace(lo, hi, k)
    if train and k > 1:
        centers = centers + rng.uniform(-0.5, 0.5, size=k) * ((hi - lo) / (k - 1))
    return np.clip(np.round(centers), 0, n - 1).astype(int)


class StudyDataset(Dataset):
    def __init__(
        self,
        records: Sequence[StudyRecord],
        img_size: int = 384,
        n_slices: int = 24,
        train: bool = False,
        aug: AugConfig | None = None,
        slice_jitter: float = 0.15,
        slice_step: int = 1,
        max_series: int = 3,
        dicom_resize_mode: str = "pad",
        dicom_max_slices: int | None = 64,
        seed: int = 0,
        cpu_intensity_aug: bool = False,
        slice_band: tuple[float, float] = (0.0, 1.0),
        crop_mm: float = 0.0,
        canonical: bool = False,
        windows_per_study: int = 0,
        slice_offset: float = 0.0,
        view: dict[str, Any] | None = None,
    ):
        v = view or {}
        self.cache_mm, self.view_mm = float(v.get("cache_mm", 0.0)), float(v.get("view_mm", 0.0))
        self.cache_size, self.smart_crop = int(v.get("cache_size", 384)), bool(v.get("smart_crop", False))
        self.crop_jitter_mm = float(v.get("crop_jitter_mm", 0.0))
        self.late = self.cache_mm > self.view_mm > 0
        self.records = list(records)
        self.img_size = img_size
        self.n_slices = n_slices
        self.train = train
        self.aug = aug or AugConfig()
        self.slice_jitter = slice_jitter
        self.slice_step = slice_step
        self.max_series = max_series
        self.dicom_resize_mode = dicom_resize_mode
        self.dicom_max_slices = dicom_max_slices
        self.seed = seed
        self.cpu_intensity_aug = cpu_intensity_aug
        self.slice_band = tuple(slice_band)
        self.crop_mm = crop_mm
        self.canonical = canonical
        self.slice_offset = slice_offset
        self.windows_per_study = windows_per_study  # >0: split this budget over the study's series (proportional to length)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ per-series
    def _load_series(self, entry: SeriesEntry, rng: np.random.Generator, n_slices: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """-> (images (N,3,H,W) uint8, slice_pos (N,) float32)."""
        k = n_slices or self.n_slices
        if entry.kind == "feat":
            with np.load(entry.source) as z:
                feats, positions = z["feats"], z["positions"]
            n = feats.shape[0]
            centers = sample_centers(n, k, self.train, rng, self.slice_jitter, self.slice_band, self.slice_offset)
            pos = positions[centers].astype(np.float32)
            return feats[centers].astype(np.float32), pos
        if entry.kind == "npz":
            src = NpzSeries(entry.source)
            n, positions = src.n, src.positions
            centers = sample_centers(n, k, self.train, rng, self.slice_jitter, self.slice_band, self.slice_offset)
            nb = np.stack([centers - self.slice_step, centers, centers + self.slice_step], 1).clip(0, n - 1)
            needed = np.unique(nb)
            slices = src.decode(needed.tolist())
        else:
            vol, positions, _ = preprocess_series(entry.source, self.cache_size if self.late else self.img_size, self.dicom_resize_mode,
                                                  meta=entry.meta, max_slices=self.dicom_max_slices,
                                                  crop_mm=(self.cache_mm if self.late else self.crop_mm) or None,
                                                  canonical=self.canonical, mirror=entry.mirror)
            n = vol.shape[0]
            centers = sample_centers(n, k, self.train, rng, self.slice_jitter, self.slice_band, self.slice_offset)
            nb = np.stack([centers - self.slice_step, centers, centers + self.slice_step], 1).clip(0, n - 1)
            needed = np.unique(nb)
            slices = vol[needed]
        lookup = {int(i): k for k, i in enumerate(needed)}
        if self.late:  # wide cache: every window cuts its own view; the three channels of a window share one shift
            jit = rng.uniform(-self.crop_jitter_mm, self.crop_jitter_mm, 2) if (self.train and self.crop_jitter_mm > 0) else np.zeros(2)
            margin = (self.cache_mm - self.view_mm) / 2.0
            imgs = np.empty((len(centers), 3, self.img_size, self.img_size), dtype=np.uint8)
            for wi, c in enumerate(centers):
                sh = smart_shift_mm(slices[lookup[int(c)]], c / max(n - 1, 1), self.cache_mm, self.view_mm) if self.smart_crop else (0.0, 0.0)
                sh = tuple(float(np.clip(sh[a] + jit[a], -margin, margin)) for a in (0, 1))
                for ch in range(3):
                    crop = late_crop(slices[lookup[int(nb[wi, ch])]], self.cache_mm, self.view_mm, sh)
                    imgs[wi, ch] = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            if self.train:
                imgs = apply_affine(imgs.reshape(-1, self.img_size, self.img_size), sample_affine(rng, self.img_size, self.aug)).reshape(imgs.shape)
        else:
            if slices.shape[1] != self.img_size or slices.shape[2] != self.img_size:
                slices = np.stack([cv2.resize(s, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) for s in slices])
            if self.train:
                slices = apply_affine(slices, sample_affine(rng, self.img_size, self.aug))
            imgs = slices[np.vectorize(lookup.get)(nb)]  # (N,3,H,W)
        if self.train and self.cpu_intensity_aug:
            x = imgs.astype(np.float32) / 255.0
            x = intensity_aug(x, rng, self.aug)
            x = random_erase(x, rng, self.aug)
            imgs = (x * 255.0 + 0.5).astype(np.uint8)
        elif self.train:  # intensity is applied on the GPU in the training loop (gpu_intensity_aug)
            imgs = random_erase(np.ascontiguousarray(imgs), rng, self.aug)
        pos = positions[centers].astype(np.float32) if len(positions) == n else (centers / max(n - 1, 1)).astype(np.float32)
        return imgs, pos

    def _window_budget(self, entries: list[SeriesEntry]) -> list[int]:
        """Windows per series: fixed n_slices, or a study budget split proportionally to series length (min 2)."""
        if not entries:
            return []
        if self.windows_per_study <= 0:
            return [self.n_slices] * len(entries)
        lens = np.array([max(1, e.n_slices) for e in entries], dtype=float)
        raw = self.windows_per_study * lens / lens.sum()
        k = np.maximum(2, np.floor(raw)).astype(int)
        for _ in range(int(self.windows_per_study - k.sum())):  # hand out the remainder to the longest series
            k[int(np.argmax(raw - k))] += 1
        return k.tolist()

    # ------------------------------------------------------------------ per-study
    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        rng = np.random.default_rng(None if self.train else (self.seed * 1_000_003 + i))
        entries = list(rec.series)
        if len(entries) > self.max_series:
            # keep plane diversity: the k-th series of each plane sorts after the (k-1)-th of every plane
            if self.train:
                rng.shuffle(entries)
            counts: dict[int, int] = {}
            keyed = []
            for e in entries:
                k = counts.get(e.plane_id, 0)
                counts[e.plane_id] = k + 1
                keyed.append((k, e))
            entries = [e for _, e in sorted(keyed, key=lambda t: t[0])]  # stable sort
        entries = entries[: self.max_series]
        images, pos, planes, seqs = [], [], [], []
        per_series = self._window_budget(entries)
        for e, k in zip(entries, per_series):
            try:
                im, p = self._load_series(e, rng, k)
            except Exception as ex:
                log.warning("study %s series %s failed: %s", rec.study_uid, e.source, ex)
                continue
            images.append(im)
            pos.append(p)
            planes.append(e.plane_id)
            seqs.append(e.seq_id)
        if self.windows_per_study > 0:
            # FLAT BAG: every window of the study in one pseudo-series of constant length K, with per-window plane /
            # sequence ids. Constant batch shape (no padding waste, cudnn autotunes once, predictable memory).
            K = self.windows_per_study
            if images:
                img = np.concatenate(images)[:K]
                ps = np.concatenate(pos)[:K]
                pl = np.concatenate([np.full(len(im), v, dtype=np.int64) for im, v in zip(images, planes)])[:K]
                sq = np.concatenate([np.full(len(im), v, dtype=np.int64) for im, v in zip(images, seqs)])[:K]
                n = len(img)
                valid = np.zeros((1, K), dtype=bool); valid[0, :n] = True
                if n < K:  # a series failed to load: repeat the last window, masked out
                    r = K - n
                    img = np.concatenate([img, np.repeat(img[-1:], r, 0)]); ps = np.concatenate([ps, np.repeat(ps[-1:], r)])
                    pl = np.concatenate([pl, np.repeat(pl[-1:], r)]); sq = np.concatenate([sq, np.repeat(sq[-1:], r)])
                images, pos, S = [img], [ps], 1
                plane_t, seq_t = torch.from_numpy(pl[None]), torch.from_numpy(sq[None])
            else:
                S, valid = 0, np.zeros((0, K), dtype=bool)
                plane_t = seq_t = torch.zeros((0, K), dtype=torch.long)
            n_max = K
        else:
            S = len(images)
            n_max = max([im.shape[0] for im in images] + [self.n_slices])
            valid = np.zeros((S, n_max), dtype=bool)
            for j in range(S):  # series may carry different window counts: pad by repeating the last window, mask it out
                valid[j, : images[j].shape[0]] = True
                if images[j].shape[0] < n_max:
                    r = n_max - images[j].shape[0]
                    images[j] = np.concatenate([images[j], np.repeat(images[j][-1:], r, 0)])
                    pos[j] = np.concatenate([pos[j], np.repeat(pos[j][-1:], r)])
            plane_t, seq_t = torch.tensor(planes, dtype=torch.long), torch.tensor(seqs, dtype=torch.long)
        out: dict[str, Any] = {
            "study_uid": rec.study_uid,
            "images": torch.from_numpy(np.stack(images)) if S else torch.zeros((0, n_max, 3, self.img_size, self.img_size), dtype=torch.uint8),
            "slice_pos": torch.from_numpy(np.stack(pos)) if S else torch.zeros((0, n_max), dtype=torch.float32),
            "slice_valid": torch.from_numpy(valid),
            "plane_ids": plane_t,
            "seq_ids": seq_t,
            "n_series": S,
            "targets": torch.from_numpy(rec.targets.astype(np.float32)) if rec.targets is not None else torch.full((N_LABELS,), float("nan")),
            "weights": torch.from_numpy(rec.weights.astype(np.float32)) if rec.weights is not None else torch.ones(N_LABELS),
            "is_gold": bool(rec.is_gold),
        }
        return out


def _pad_slices(x: torch.Tensor, n: int) -> torch.Tensor:
    """(S, N_i, ...) -> (S, n, ...) by repeating the last slice."""
    if x.shape[0] == 0 or x.shape[1] == n:
        return x
    reps = x[:, -1:].expand(x.shape[0], n - x.shape[1], *x.shape[2:])
    return torch.cat([x, reps], 1)


def collate_studies(items: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(items)
    sbi = torch.cat([torch.full((it["n_series"],), b, dtype=torch.long) for b, it in enumerate(items)])
    n_max = max([it["images"].shape[1] for it in items if it["n_series"] > 0] + [1])

    def pad_valid(v: torch.Tensor) -> torch.Tensor:
        return v if v.shape[1] == n_max or v.shape[0] == 0 else torch.cat([v, torch.zeros((v.shape[0], n_max - v.shape[1]), dtype=torch.bool)], 1)

    return {
        "images": torch.cat([_pad_slices(it["images"], n_max) for it in items], 0),
        "slice_pos": torch.cat([_pad_slices(it["slice_pos"], n_max) for it in items], 0),
        "slice_valid": torch.cat([pad_valid(it["slice_valid"]) for it in items], 0),
        "plane_ids": torch.cat([_pad_slices(it["plane_ids"], n_max) if it["plane_ids"].ndim == 2 else it["plane_ids"] for it in items], 0),
        "seq_ids": torch.cat([_pad_slices(it["seq_ids"], n_max) if it["seq_ids"].ndim == 2 else it["seq_ids"] for it in items], 0),
        "series_batch_idx": sbi,
        "batch_size": B,
        "targets": torch.stack([it["targets"] for it in items]),
        "weights": torch.stack([it["weights"] for it in items]),
        "is_gold": torch.tensor([it["is_gold"] for it in items]),
        "study_uids": [it["study_uid"] for it in items],
    }


def view_kwargs(d: Any) -> dict[str, Any]:
    """DataConfig -> the late-crop settings of StudyDataset (one place for train, eval and inference)."""
    return {"cache_mm": d.cache_mm, "view_mm": d.view_mm, "cache_size": d.cache_size, "smart_crop": d.smart_crop, "crop_jitter_mm": d.crop_jitter_mm}


def compute_sample_weights(targets: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Study sampling weight: boost studies positive for rare labels (macro-AUC weights labels equally)."""
    t = np.nan_to_num(np.asarray(targets, dtype=np.float32), nan=0.0)
    pos = t >= 0.5
    prev = pos.mean(0).clip(1e-3, 1.0)
    rarity = (1.0 / prev) ** power
    rarity = rarity / rarity.min()
    w = np.where(pos, rarity[None, :], 1.0).max(1)
    return (w / np.median(w)).astype(np.float64)
