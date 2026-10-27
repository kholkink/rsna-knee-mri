"""Typed configuration (dataclasses) loadable from YAML with ``key.sub=value`` CLI overrides."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    cache_index: str = "cache/cache_index.parquet"
    crop_mm: float = 0.0          # physical centre crop applied when the cache was built (0 = full field of view)
    canonical: bool = False       # canonical display orientation from ImageOrientationPatient
    cache_mm: float = 0.0         # >0: the cache holds a WIDER physical window than the model sees ("cache wide, crop late")
    view_mm: float = 0.0          # physical window cut out of the cache at load time (needs cache_mm > view_mm)
    cache_size: int = 384         # pixel size of the wide cache (the live DICOM path renders the same before the late crop)
    smart_crop: bool = False      # late crop follows the anatomy towards the ends of a stack (dicom.smart_shift_mm)
    crop_jitter_mm: float = 0.0   # train: random shift of the view window per series (translation aug in millimetres)
    lat_normalize: bool = False   # right knees mirrored onto left (needs aug_hflip=0 and no flip TTA)
    slice_band: tuple[float, float] = (0.0, 1.0)  # fraction of the stack that slices are sampled from
    windows_per_study: int = 0       # >0: total 2.5D windows per study split across its series (Raptor-style bag)
    windows_per_study_eval: int = 0
    cache_max_slices: int = 48  # series were uniformly subsampled to this many slices when cached
    cache_root: str = ""        # prefix for relative paths in the index (Kaggle: where the cache tars were unpacked)
    label_table: str = "labels/label_table.parquet"
    eval_label_table: str = ""  # fixed reference targets for the VALIDATION split (compare label variants on equal terms)
    folds: str = "labels/folds.parquet"
    img_size: int = 384
    n_slices: int = 24          # slices sampled per series at train time
    n_slices_eval: int = 32     # at eval / inference
    max_series: int = 3         # per study
    planes: tuple[str, ...] = ("sagittal", "coronal", "axial")
    slice_jitter: float = 0.15  # random crop of the slice range (fraction) at train time
    num_workers: int = 4
    aug_rotate: float = 10.0
    aug_scale: tuple[float, float] = (0.9, 1.1)
    aug_translate: float = 0.05
    aug_hflip: float = 0.5
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    aug_gamma: tuple[float, float] = (0.8, 1.25)
    aug_noise: float = 0.02
    aug_erase_p: float = 0.2
    rare_boost_power: float = 0.5  # sampler: weight ~ (1/prevalence)^power for positives of rare labels


@dataclass
class ModelConfig:
    backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    pretrained: bool = True
    checkpoint_path: str = ""     # local timm weights for offline Kaggle
    feature_dim: int = 0           # >0 => precomputed-feature mode (no backbone)
    d_model: int = 512
    n_heads: int = 8
    series_layers: int = 2
    study_layers: int = 2
    dropout: float = 0.1
    drop_path: float = 0.1
    grad_checkpoint: bool = False
    channels_last: bool = True
    spatial_head: bool = False     # per-finding attention over the slice feature map (residual branch, zero-init)
    spatial_dim: int = 256
    pool: str = "attn"             # attn | meanmax (mean + max over windows -> linear; what the top teams describe)


@dataclass
class OptimConfig:
    epochs: int = 8
    batch_size: int = 2            # studies per step
    accum: int = 4
    lr_backbone: float = 1e-4
    lr_head: float = 3e-4
    weight_decay: float = 0.02
    warmup_frac: float = 0.05
    grad_clip: float = 2.0
    ema_decay: float = 0.998
    amp: str = "auto"              # auto | fp16 | bf16 | off
    label_smoothing: float = 0.0
    gold_weight: float = 5.0
    topk_avg: int = 0              # >1: average the weights of the k best epochs at the end (free at inference)
    pair_weight: float = 0.0       # >0: add the queue-based pairwise (AUC surrogate) loss with this weight; 0 = exactly the old training
    pair_last_epochs: int = 3      # the pairwise term is active only in the last N epochs (the queue is filled from the start)
    pair_queue: int = 512
    pair_tau: float = 1.0
    pair_lo: float = 0.15          # a cell is a negative for pairing when target <= pair_lo ...
    pair_hi: float = 0.85          # ... and a positive when target >= pair_hi (public CoAtNet recipe: 0.4 / 0.6, weight 0.05, all epochs)


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fold: int = 0
    n_folds: int = 5
    seed: int = 42
    output_dir: str = "runs/exp"
    resume: bool = True
    wandb_project: str = ""
    compile: bool = False
    eval_every: int = 1
    max_train_studies: int = 0     # debug: subsample
    max_hours: float = 0.0         # stop before an epoch that would overrun this wall-clock budget (0 = off)
    train_on_all: bool = False     # train on every non-gold study; validate on the 58 gold studies only
    gold_holdout: bool = False     # ablation protocol: val = held-out fold + ALL gold studies; gold never trains
    select_by: str = "macro_auc"   # macro_auc | confident | gold | last  (checkpoint / top-k selection metric)
    init_from: str = ""            # warm start: best.pt of an earlier run of OURS (domain-adapted weights instead of ImageNet)
    init_backbone_only: bool = True  # take only the backbone from init_from; the finding heads start fresh
    gold_train: bool = False       # train_on_all only: the expert-labelled studies join the training set (the test set is labelled the same way)
    gold_cv: bool = False          # with gold_train: hold out gold part `fold % gold_parts` as the only validation; pooled over folds = honest gold OOF
    gold_parts: int = 5
    gold_repeat: float = 4.0       # sampling multiplier of gold studies (their per-cell loss weight comes from the label table, w=5)


def _coerce(value: Any, target_type: Any) -> Any:
    origin = getattr(target_type, "__origin__", None)
    if origin is tuple and isinstance(value, (list, tuple)):
        args = target_type.__args__
        elem_t = args[0] if args else str
        return tuple(_coerce(v, elem_t) for v in value)
    if target_type is bool and isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "y")
    if target_type in (int, float) and isinstance(value, str):
        return target_type(value)
    if target_type is str and value is None:
        return ""
    return value


def _from_dict(cls: Any, d: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        t = hints.get(f.name, f.type)
        v = d[f.name]
        if isinstance(t, type) and is_dataclass(t):
            kwargs[f.name] = _from_dict(t, v or {})
        else:
            kwargs[f.name] = _coerce(v, t)
    return cls(**kwargs)


def _apply_override(d: dict[str, Any], key: str, value: str) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    v: Any = value
    try:
        v = yaml.safe_load(value)
    except Exception:
        pass
    cur[parts[-1]] = v


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    d: dict[str, Any] = {}
    if path:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _apply_override(d, k.strip(), v.strip())
    return _from_dict(TrainConfig, d)


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
