#!/usr/bin/env python
"""Prepare report batches for in-session agent labelling and merge/validate their outputs.

    python scripts/relabel_batches.py make --train-csv data/train.csv --out labels/agent --set gold --size 30
    python scripts/relabel_batches.py make --train-csv data/train.csv --out labels/agent --set disagree --size 40
    python scripts/relabel_batches.py merge --dir labels/agent --out labels/extractions_agent.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
from pydantic import ValidationError

from rsna_knee.constants import ID_COL, LABELS, resolve_label_columns
from rsna_knee.reports.schema import ReportExtraction, empty_extraction


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--train-csv", required=True); m.add_argument("--out", required=True)
    m.add_argument("--set", choices=["gold", "disagree", "all", "rest"], default="gold"); m.add_argument("--size", type=int, default=30)
    m.add_argument("--exclude", default="", help="jsonl of already labelled studies to skip")
    g = sub.add_parser("merge"); g.add_argument("--dir", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "make":
        tr = pd.read_csv(a.train_csv); tr[ID_COL] = tr[ID_COL].astype(str)
        cols = resolve_label_columns(tr.columns)
        is_gold = tr[[cols[k] for k in LABELS]].notna().any(axis=1)
        if a.set == "gold":
            sel = tr[is_gold]
        elif a.set == "all":
            sel = tr
        else:
            own = pd.read_parquet("labels/label_table_qwen_cal.parquet").set_index("study_uid")
            pub = pd.read_parquet("labels/label_table_public_llm.parquet").set_index("study_uid")
            dis = pd.Series(False, index=own.index)
            for k in LABELS:
                dis |= (own[f"y_{k}"] >= .5) != (pub.reindex(own.index)[f"y_{k}"] >= .5)
            ids = set(dis[dis].index)
            sel = tr[tr[ID_COL].isin(ids) & ~is_gold] if a.set == "disagree" else tr[~tr[ID_COL].isin(ids) & ~is_gold]
        if a.exclude and os.path.exists(a.exclude):
            done = {json.loads(l)["study_uid"] for l in open(a.exclude) if l.strip()}
            sel = sel[~sel[ID_COL].isin(done)]
        os.makedirs(a.out, exist_ok=True)
        rows = sel[[ID_COL, "Report"]].reset_index(drop=True)
        for i in range(0, len(rows), a.size):
            path = os.path.join(a.out, f"batch_{a.set}_{i // a.size:03d}.jsonl")
            with open(path, "w") as f:
                for _, r in rows.iloc[i:i + a.size].iterrows():
                    f.write(json.dumps({"study_uid": r[ID_COL], "report": str(r["Report"])}, ensure_ascii=False) + "\n")
        print(f"{len(rows)} reports -> {(len(rows) + a.size - 1) // a.size} batches in {a.out} (set={a.set})")
    else:
        n_ok = n_bad = 0
        seen = {}
        for p in sorted(glob.glob(os.path.join(a.dir, "out_*.jsonl"))):
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    uid = str(rec.pop("study_uid"))
                    ext = ReportExtraction.model_validate(rec.get("extraction", rec))
                    seen[uid] = ext
                    n_ok += 1
                except (ValidationError, ValueError, KeyError) as e:
                    n_bad += 1
                    print("bad line in", os.path.basename(p), ":", str(e)[:120])
        with open(a.out, "w") as f:
            for uid, ext in seen.items():
                f.write(json.dumps({"study_uid": uid, "backend": "claude-agent", "model": "claude-agent", "extraction": ext.model_dump(), "error": ""}, ensure_ascii=False) + "\n")
        print(f"merged {len(seen)} studies ({n_ok} ok lines, {n_bad} bad) -> {a.out}")


if __name__ == "__main__":
    main()
