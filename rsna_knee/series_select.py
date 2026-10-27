"""Choose which series of a study feed the model.

Knee protocols differ across the 16-19 sites; the CSV ``Fluid_Sensitive`` / ``Fat_Suppression``
flags are reported to be degenerate, so the choice is driven by DICOM headers
(:class:`~rsna_knee.dicom.SeriesMeta`).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .dicom import SeriesMeta


@dataclass
class SelectionPolicy:
    planes: tuple[str, ...] = ("sagittal", "coronal", "axial")
    max_per_plane: int = 1
    max_total: int = 3
    min_slices: int = 8
    max_slices: int = 400
    prefer_slices: tuple[int, int] = (14, 64)
    allow_unknown_plane: bool = False

    @classmethod
    def core3(cls) -> SelectionPolicy:
        """One best fluid-sensitive series per plane (fast baseline)."""
        return cls()

    @classmethod
    def rich(cls) -> SelectionPolicy:
        """Up to 2 series per plane, 6 total (fluid-sensitive + T1/PD non-FS complement)."""
        return cls(max_per_plane=2, max_total=6)


def series_score(m: SeriesMeta, policy: SelectionPolicy) -> float | None:
    """Higher is better; ``None`` means excluded."""
    if m.error or m.n_slices < policy.min_slices or m.n_slices > policy.max_slices:
        return None
    if m.looks_like_localizer:
        return None
    if m.plane not in policy.planes and not (policy.allow_unknown_plane and m.plane == "unknown"):
        return None
    s = 0.0
    if m.is_fluid_sensitive:
        s += 3.0
        if m.fat_sat:
            s += 2.0
        if m.weighting == "stir":
            s -= 0.5  # STIR is fluid sensitive but lower SNR than PD-FS
    elif m.weighting == "t1":
        s += 1.0
    else:
        s -= 0.5
    lo, hi = policy.prefer_slices
    if lo <= m.n_slices <= hi:
        s += 1.0
    elif m.n_slices > 100:
        s -= 1.5
    else:
        s -= 0.5
    if (m.rows or 0) >= 256 and (m.cols or 0) >= 256:
        s += 0.3
    if (m.mr_acq_type or "").upper() == "3D":
        s -= 1.0
    if m.plane == "unknown":
        s -= 2.0
    return s


def select_series(metas: Iterable[SeriesMeta], policy: SelectionPolicy | None = None) -> list[SeriesMeta]:
    policy = policy or SelectionPolicy()
    scored: list[tuple[float, SeriesMeta]] = []
    for m in metas:
        sc = series_score(m, policy)
        if sc is not None:
            scored.append((sc, m))

    # de-duplicate identical (plane, weighting, fat_sat): keep best score, tie -> n_slices closest to 32
    best: dict[tuple[str, str, bool], tuple[float, SeriesMeta]] = {}
    for sc, m in scored:
        key = (m.plane, m.weighting, m.fat_sat)
        cur = best.get(key)
        if cur is None or (sc, -abs(m.n_slices - 32)) > (cur[0], -abs(cur[1].n_slices - 32)):
            best[key] = (sc, m)

    per_plane: dict[str, list[tuple[float, SeriesMeta]]] = {}
    for sc, m in best.values():
        per_plane.setdefault(m.plane, []).append((sc, m))
    chosen: list[tuple[float, SeriesMeta]] = []
    for plane in list(policy.planes) + (["unknown"] if policy.allow_unknown_plane else []):
        cand = sorted(per_plane.get(plane, []), key=lambda t: (-t[0], abs(t[1].n_slices - 32)))
        chosen.extend(cand[: policy.max_per_plane])
    chosen.sort(key=lambda t: -t[0])
    chosen = chosen[: policy.max_total]
    order = {p: i for i, p in enumerate(list(policy.planes) + ["unknown"])}
    chosen.sort(key=lambda t: (order.get(t[1].plane, 99), -t[0]))
    return [m for _, m in chosen]
