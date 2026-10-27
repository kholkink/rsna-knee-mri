"""Canonical label keys, plane / sequence vocabularies and CSV column resolution.

The competition CSV column names are resolved at runtime (``resolve_label_columns``) so the
code never hard-codes a spelling such as ``Bakers_Cyst`` vs ``Baker's Cyst``.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"

# Canonical internal keys - order defines the model output order.
LABELS: list[str] = [
    "acl",
    "mcl",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_oa",
    "lateral_oa",
    "patellofemoral_oa",
    "effusion",
    "synovitis",
    "bakers_cyst",
    "contusion",
    "fracture",
]
LABEL_INDEX: dict[str, int] = {k: i for i, k in enumerate(LABELS)}
N_LABELS = len(LABELS)

# Normalised (lower-case, alnum only) spellings accepted for each label.
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "acl": ("acl", "anteriorcruciateligament", "acltear"),
    "mcl": ("mcl", "medialcollateralligament", "mcltear"),
    "medial_meniscus": ("medialmeniscus", "medialmeniscustear", "medialmeniscaltear"),
    "lateral_meniscus": ("lateralmeniscus", "lateralmeniscustear", "lateralmeniscaltear"),
    "medial_oa": ("medialoa", "medialosteoarthritis", "medialcompartmentoa", "medialtibiofemoraloa"),
    "lateral_oa": ("lateraloa", "lateralosteoarthritis", "lateralcompartmentoa", "lateraltibiofemoraloa"),
    "patellofemoral_oa": ("patellofemoraloa", "pfoa", "patellofemoralosteoarthritis", "patellofemoral"),
    "effusion": ("effusion", "jointeffusion"),
    "synovitis": ("synovitis",),
    "bakers_cyst": ("bakerscyst", "bakercyst", "poplitealcyst", "bakers", "baker"),
    "contusion": ("contusion", "bonecontusion", "bonemarrowedema", "bonebruise"),
    "fracture": ("fracture",),
}

PLANES = ["sagittal", "coronal", "axial", "unknown"]
PLANE_INDEX = {p: i for i, p in enumerate(PLANES)}
N_PLANES = len(PLANES)

WEIGHTINGS = ["pd", "t2", "stir", "t1", "other"]
FLUID_SENSITIVE = {"pd", "t2", "stir"}
N_SEQ_IDS = len(WEIGHTINGS) * 2


def seq_id(weighting: str, fat_sat: bool) -> int:
    """Embedding id for (weighting, fat-sat) - 10 classes."""
    w = weighting if weighting in WEIGHTINGS else "other"
    return WEIGHTINGS.index(w) * 2 + int(bool(fat_sat))


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_label_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map canonical label key -> actual CSV column name.

    Exact normalised match first, then substring match on long aliases only
    (short aliases such as ``acl`` would otherwise collide with unrelated columns).
    """
    cols = list(columns)
    norm = {c: normalize_name(c) for c in cols}
    out: dict[str, str] = {}
    for key, aliases in LABEL_ALIASES.items():
        hit = next((c for c in cols if norm[c] in aliases), None)
        if hit is None:
            hit = next(
                (c for c in cols if any(a in norm[c] for a in aliases if len(a) >= 6)),
                None,
            )
        if hit is not None:
            out[key] = hit
    return out


def missing_labels(columns: Iterable[str]) -> list[str]:
    found = resolve_label_columns(columns)
    return [k for k in LABELS if k not in found]
