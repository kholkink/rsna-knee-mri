"""Map structured extractions (+ gold labels) to soft training targets, and calibrate on the gold set."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ..constants import ID_COL, LABELS, N_LABELS, resolve_label_columns
from .schema import Finding, ReportExtraction

log = logging.getLogger(__name__)
SEVERITY_RANK = {"unknown": 0, "mild": 1, "moderate": 2, "severe": 3}


@dataclass
class LabelMapConfig:
    present: float = 1.0
    absent: float = 0.0
    equivocal: float = 0.5
    equivocal_by_label: dict[str, float] = field(default_factory=dict)
    not_mentioned: float = 0.05
    not_mentioned_by_label: dict[str, float] = field(default_factory=lambda: {
        "medial_oa": 0.12, "lateral_oa": 0.08, "patellofemoral_oa": 0.12, "synovitis": 0.08, "effusion": 0.08})
    min_severity: dict[str, str] = field(default_factory=dict)  # label -> "moderate": mild positives get `below_value`
    # label not mentioned but another label present -> raised prior. Gold shows expert "synovitis" is an
    # image-level call that co-occurs with effusion (22/27) while reports rarely name it.
    conditional_prior: dict[str, list] = field(default_factory=lambda: {"synovitis": ["effusion", 0.45]})
    below_value: float = 0.35
    w_present: float = 1.0
    w_absent: float = 1.0
    w_equivocal: float = 0.5
    w_not_mentioned: float = 0.4
    w_gold: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelMapConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


def map_finding(label: str, f: Finding, cfg: LabelMapConfig) -> tuple[float, float]:
    if f.status == "present":
        need = cfg.min_severity.get(label)
        if need and SEVERITY_RANK.get(f.severity, 0) < SEVERITY_RANK[need] and f.severity != "unknown":
            return cfg.below_value, cfg.w_equivocal
        return cfg.present, cfg.w_present
    if f.status == "absent":
        return cfg.absent, cfg.w_absent
    if f.status == "equivocal":
        return cfg.equivocal_by_label.get(label, cfg.equivocal), cfg.w_equivocal
    return cfg.not_mentioned_by_label.get(label, cfg.not_mentioned), cfg.w_not_mentioned


def map_extraction(ext: ReportExtraction, cfg: LabelMapConfig) -> tuple[np.ndarray, np.ndarray]:
    t = np.zeros(N_LABELS, dtype=np.float32)
    w = np.zeros(N_LABELS, dtype=np.float32)
    for j, label in enumerate(LABELS):
        t[j], w[j] = map_finding(label, getattr(ext, label), cfg)
    for label, (other, value) in cfg.conditional_prior.items():
        j = LABELS.index(label)
        if getattr(ext, label).status == "not_mentioned" and getattr(ext, other).status == "present":
            t[j], w[j] = float(value), cfg.w_not_mentioned
    return t, w


def gold_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """Rows with at least one non-null gold label -> columns study_uid + canonical label keys (NaN if missing)."""
    cols = resolve_label_columns(train_df.columns)
    out = pd.DataFrame({"study_uid": train_df[ID_COL].astype(str)})
    for k in LABELS:
        out[k] = pd.to_numeric(train_df[cols[k]], errors="coerce").values if k in cols else np.nan
    has = out[LABELS].notna().any(axis=1)
    return out[has].reset_index(drop=True)


def build_label_table(
    train_df: pd.DataFrame,
    extractions: dict[str, ReportExtraction],
    cfg: LabelMapConfig,
    rules: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """One row per study: y_<label>, w_<label>, src (gold|llm|rules|none), is_gold."""
    gold = gold_table(train_df).set_index("study_uid") if len(train_df) else pd.DataFrame()
    rows = []
    for uid in train_df[ID_COL].astype(str):
        t = np.full(N_LABELS, np.nan, dtype=np.float32)
        w = np.zeros(N_LABELS, dtype=np.float32)
        src = "none"
        if uid in extractions:
            t, w = map_extraction(extractions[uid], cfg)
            src = "llm"
        elif rules and uid in rules:
            r = rules[uid]
            for j, label in enumerate(LABELS):
                s = r.get(label, "unknown")
                t[j], w[j] = {"positive": (cfg.present, cfg.w_present * 0.7), "negative": (cfg.absent, cfg.w_absent * 0.7),
                              "equivocal": (cfg.equivocal, cfg.w_equivocal)}.get(
                    s, (cfg.not_mentioned_by_label.get(label, cfg.not_mentioned), cfg.w_not_mentioned * 0.5))
            src = "rules"
        is_gold = False
        if len(gold) and uid in gold.index:
            g = gold.loc[uid, LABELS].values.astype(np.float32)
            m = np.isfinite(g)
            t[m] = g[m]
            w[m] = cfg.w_gold
            is_gold = bool(m.any())
            src = "gold" if src == "none" else src + "+gold"
        row = {"study_uid": uid, "src": src, "is_gold": is_gold}
        row.update({f"y_{k}": float(t[j]) for j, k in enumerate(LABELS)})
        row.update({f"w_{k}": float(w[j]) for j, k in enumerate(LABELS)})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_mapping(extractions: dict[str, ReportExtraction], gold: pd.DataFrame, cfg: LabelMapConfig) -> pd.DataFrame:
    """Per-label agreement between mapped soft labels and gold on the gold studies."""
    recs = []
    for j, label in enumerate(LABELS):
        ys, ps = [], []
        for _, r in gold.iterrows():
            uid = r["study_uid"]
            if uid not in extractions or not np.isfinite(r[label]):
                continue
            ys.append(int(r[label] >= 0.5))
            ps.append(float(map_extraction(extractions[uid], cfg)[0][j]))
        y, p = np.asarray(ys), np.asarray(ps)
        auc = float(roc_auc_score(y, p)) if len(y) and y.min() != y.max() else float("nan")
        acc = float(((p >= 0.5) == (y == 1)).mean()) if len(y) else float("nan")
        ll = float(log_loss(y, np.clip(p, 1e-3, 1 - 1e-3), labels=[0, 1])) if len(y) else float("nan")
        recs.append({"label": label, "n": len(y), "n_pos": int(y.sum()) if len(y) else 0, "auc": auc, "acc": acc, "logloss": ll})
    return pd.DataFrame(recs)


def calibrate(extractions: dict[str, ReportExtraction], gold: pd.DataFrame, base: LabelMapConfig | None = None,
              min_gain: float = 0.02) -> tuple[LabelMapConfig, pd.DataFrame]:
    """Tiny per-label grid search on the gold set (guarded by ``min_gain`` - 58 studies overfit easily)."""
    base = base or LabelMapConfig()
    cfg = LabelMapConfig.from_dict(base.to_dict())
    cfg.not_mentioned_by_label = dict(base.not_mentioned_by_label)
    cfg.equivocal_by_label = dict(base.equivocal_by_label)
    cfg.min_severity = dict(base.min_severity)
    cfg.conditional_prior = {k: list(v) for k, v in base.conditional_prior.items()}
    base_eval = evaluate_mapping(extractions, gold, base).set_index("label")
    report = []
    for label in LABELS:
        best_auc, best_ll, best = base_eval.loc[label, "auc"], base_eval.loc[label, "logloss"], None
        for nm, eq, sev in itertools.product([0.02, 0.05, 0.12, 0.25], [0.3, 0.5, 0.7], [None, "moderate"]):
            trial = LabelMapConfig.from_dict(cfg.to_dict())
            trial.not_mentioned_by_label = dict(cfg.not_mentioned_by_label, **{label: nm})
            trial.equivocal_by_label = dict(cfg.equivocal_by_label, **{label: eq})
            trial.min_severity = dict(cfg.min_severity)
            if sev:
                trial.min_severity[label] = sev
            else:
                trial.min_severity.pop(label, None)
            ev = evaluate_mapping(extractions, gold, trial).set_index("label").loc[label]
            if np.isfinite(ev["auc"]) and (ev["auc"] > (best_auc if np.isfinite(best_auc) else -1) + 1e-9
                                            or (abs(ev["auc"] - best_auc) < 1e-9 and ev["logloss"] < best_ll)):
                best_auc, best_ll, best = ev["auc"], ev["logloss"], (nm, eq, sev)
        gain = best_auc - base_eval.loc[label, "auc"] if np.isfinite(base_eval.loc[label, "auc"]) else 0.0
        applied = best is not None and gain >= min_gain
        if applied:
            nm, eq, sev = best
            cfg.not_mentioned_by_label[label] = nm
            cfg.equivocal_by_label[label] = eq
            if sev:
                cfg.min_severity[label] = sev
        report.append({"label": label, "base_auc": base_eval.loc[label, "auc"], "best_auc": best_auc,
                       "gain": gain, "applied": applied, "choice": best})
    return cfg, pd.DataFrame(report)


def label_table_from_graded_csv(train_df: pd.DataFrame, graded: pd.DataFrame, cfg: LabelMapConfig,
                                rules: dict[str, dict[str, str]] | None = None, min_weight: float = 0.15) -> pd.DataFrame:
    """Label table from a graded per-finding CSV (columns <Label>, <Label>__conf), e.g. the public CC0 table
    ``pilkwang/rsna-knee-llm-labels``. Score -> soft target, confidence -> loss weight; rules fill missing studies,
    gold overrides everything."""
    cols = resolve_label_columns([c for c in graded.columns if "__" not in c])
    g = graded.copy()
    g[ID_COL] = g[ID_COL].astype(str)
    g = g.drop_duplicates(ID_COL).set_index(ID_COL)
    base = build_label_table(train_df, {}, cfg, rules)  # rules + gold skeleton
    base = base.set_index("study_uid")
    gold = gold_table(train_df).set_index("study_uid")
    hit = base.index.intersection(g.index)
    for k in LABELS:
        if k not in cols:
            continue
        y = g.loc[hit, cols[k]].astype(float).clip(0, 1)
        conf_col = cols[k] + "__conf"
        w = g.loc[hit, conf_col].astype(float).clip(min_weight, 1.0) if conf_col in g.columns else pd.Series(1.0, index=hit)
        is_g = pd.Series(hit.isin(gold.index), index=hit) & gold[k].reindex(hit).notna()
        base.loc[hit[~is_g.values], f"y_{k}"] = y[~is_g.values].values
        base.loc[hit[~is_g.values], f"w_{k}"] = w[~is_g.values].values
    base.loc[hit, "src"] = np.where(base.loc[hit, "is_gold"], "graded+gold", "graded")
    return base.reset_index()
