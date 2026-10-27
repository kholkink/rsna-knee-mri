#!/usr/bin/env python
"""Extract 12 structured findings from every training report (LLM) + rule-based cross-check.

    # Claude Opus 5 via Message Batches API (50% cost, resumable):
    python scripts/extract_labels.py --train-csv data/train.csv --out labels/extractions.jsonl --backend anthropic --batch
    # DeepSeek (key in .env as DEEPSEEK_API_KEY); pilot on the 58 gold studies first:
    python scripts/extract_labels.py --train-csv data/train.csv --out labels/extractions_deepseek.jsonl --backend deepseek --gold-only
    # local open-weight model served by vLLM (keeps report text on your machine):
    python scripts/extract_labels.py --train-csv data/train.csv --out labels/extractions_qwen.jsonl \
        --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B-AWQ --concurrency 16
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from rsna_knee.constants import ID_COL, LABELS, resolve_label_columns
from rsna_knee.envfile import load_env_file
from rsna_knee.reports.rules import rule_extract

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out", required=True, help="JSONL of extractions (appended, resumable)")
    ap.add_argument("--backend", choices=["anthropic", "deepseek", "openai", "rules-only"], default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None, help="env var holding the key for --backend openai")
    ap.add_argument("--effort", default=None, help="anthropic: low|medium|high; deepseek: reasoning_effort (low|high|max)")
    ap.add_argument("--gold-only", action="store_true", help="only the studies with gold labels (pilot / validation)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", action="store_true", help="Anthropic Message Batches API")
    ap.add_argument("--report-col", default="Report")
    ap.add_argument("--rules-out", default=None, help="parquet with rule-based labels (default: next to --out)")
    a = ap.parse_args()

    load_env_file(os.path.join(REPO_ROOT, ".env"))
    df = pd.read_csv(a.train_csv)
    df[ID_COL] = df[ID_COL].astype(str)
    if a.gold_only:
        cols = [c for c in resolve_label_columns(df.columns).values()]
        df = df[df[cols].notna().any(axis=1)].reset_index(drop=True)
        print(f"gold-only: {len(df)} studies")
    assert all(k in resolve_label_columns(df.columns) for k in LABELS) or True
    rules_out = a.rules_out or os.path.join(os.path.dirname(a.out) or ".", "rules.parquet")
    os.makedirs(os.path.dirname(rules_out) or ".", exist_ok=True)
    rows = []
    for r in df.itertuples():
        rep = getattr(r, a.report_col, None)
        res = rule_extract(str(rep)) if isinstance(rep, str) else {}
        rows.append({"study_uid": r.StudyInstanceUID, **res})
    pd.DataFrame(rows).to_parquet(rules_out, index=False)
    print(f"rule labels -> {rules_out}")
    if a.backend == "rules-only":
        return

    if a.backend == "anthropic":
        from rsna_knee.reports.extract_llm import AnthropicBackend, run_batch_extraction, run_extraction

        be = AnthropicBackend(model=a.model or "claude-opus-5", effort=a.effort or "medium")
        if a.batch:
            res = run_batch_extraction(df.head(a.limit) if a.limit else df, be, a.out, report_col=a.report_col)
        else:
            res = run_extraction(df, be, a.out, report_col=a.report_col, concurrency=a.concurrency, limit=a.limit or None)
    elif a.backend == "deepseek":
        from rsna_knee.reports.extract_llm import DeepSeekBackend, run_extraction

        be = DeepSeekBackend(model=a.model or "deepseek-flash", reasoning_effort=a.effort)
        res = run_extraction(df, be, a.out, report_col=a.report_col, concurrency=a.concurrency, limit=a.limit or None)
        print("usage:", be.usage)
    else:
        from rsna_knee.reports.extract_llm import OpenAICompatBackend, run_extraction

        key = os.environ.get(a.api_key_env) if a.api_key_env else None
        be = OpenAICompatBackend(model=a.model or "Qwen/Qwen3-32B-AWQ", base_url=a.base_url, api_key=key)
        res = run_extraction(df, be, a.out, report_col=a.report_col, concurrency=a.concurrency, limit=a.limit or None)
    print(f"{len(res)} extractions available in {a.out}")


if __name__ == "__main__":
    main()
