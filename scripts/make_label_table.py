#!/usr/bin/env python
"""Extractions (+rules fallback, +gold) -> soft label table; report agreement with gold; optional calibration.

    python scripts/make_label_table.py --train-csv data/train.csv --extractions labels/extractions.jsonl \
        --rules labels/rules.parquet --out labels/label_table.parquet --calibrate --map-config labels/label_map.yaml
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

from rsna_knee.constants import ID_COL, LABELS
from rsna_knee.reports.extract_llm import load_extractions
from rsna_knee.reports.labels import (
    LabelMapConfig,
    build_label_table,
    calibrate,
    evaluate_mapping,
    gold_table,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-config", default=None, help="YAML LabelMapConfig to load (and save when --calibrate)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--gold-weight", type=float, default=5.0)
    a = ap.parse_args()

    train = pd.read_csv(a.train_csv)
    train[ID_COL] = train[ID_COL].astype(str)
    ext = load_extractions(a.extractions)
    rules = None
    if a.rules and os.path.exists(a.rules):
        rdf = pd.read_parquet(a.rules)
        rules = {str(r["study_uid"]): {k: r.get(k, "unknown") for k in LABELS} for _, r in rdf.iterrows()}
    cfg = LabelMapConfig()
    if a.map_config and os.path.exists(a.map_config):
        cfg = LabelMapConfig.from_dict(yaml.safe_load(open(a.map_config)))
    cfg.w_gold = a.gold_weight
    gold = gold_table(train)
    print(f"{len(ext)} LLM extractions, {len(gold)} gold studies, {len(train)} train rows")
    if len(gold):
        print("agreement with gold (base mapping):")
        print(evaluate_mapping(ext, gold, cfg).to_string(index=False))
        if a.calibrate:
            cfg, rep = calibrate(ext, gold, cfg)
            print("calibration:")
            print(rep.to_string(index=False))
            if a.map_config:
                os.makedirs(os.path.dirname(a.map_config) or ".", exist_ok=True)
                yaml.safe_dump(cfg.to_dict(), open(a.map_config, "w"), sort_keys=False)
                print(f"map config -> {a.map_config}")
    table = build_label_table(train, ext, cfg, rules)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    table.to_parquet(a.out, index=False)
    print(table["src"].value_counts())
    pos = (table[[f"y_{k}" for k in LABELS]] >= 0.5).mean().round(3)
    print("positive rate per label:\n", pos.to_string())
    print(f"label table -> {a.out}")


if __name__ == "__main__":
    main()
