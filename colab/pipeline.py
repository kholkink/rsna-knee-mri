#!/usr/bin/env python
"""One command for the whole production chain on Colab: teacher OOF -> pseudo-labels -> production seeds -> Kaggle dataset.

    python colab/pipeline.py --name v10 --stage all --warm                # confirmed switches only
    python colab/pipeline.py --name v10 --stage all --warm --gold         # + expert studies in training
    python colab/pipeline.py --name v10 --stage prod --warm --labels configs/labels/label_table_pseudo_v1.parquet

Stages (each is resumable: finished folds are skipped, an interrupted fold continues from its last epoch on Drive):
  oof     5 teacher models, each predicts the fold it never saw            -> work-<name>-oof/runs/<name>-oof/fold*/oof.parquet
  labels  report labels * (1 - alpha) + OOF predictions * alpha            -> configs/labels/label_table_pseudo_<name>.parquet
  prod    N seeds on all data with the new table, published as a dataset   -> kirillkholkin/rsna-knee-runs-<name>-full

Switches (one per idea; keep OFF whatever has not beaten the baseline by 0.005 on two seeds):
  --warm      start the backbone from one of OUR earlier models instead of ImageNet (holdout +0.006 / gold +0.006)
  --gold      the 58 expert studies join training (teachers: always; production: all 58, validation becomes in-sample)
  --pair W    queue-based pairwise AUC loss in the last 3 epochs with weight W
  --cover224  224 px with 48 training / 96 evaluation windows instead of 288 px with 30 / 48

Warm-start parents never saw what they are validated on: teacher fold k starts from the round-1 OOF model of fold k,
production seed s from the round-1 production seed s. Parents are pulled from our Kaggle datasets when not on Drive.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_TABLE = "configs/labels/label_table_ref_v1.parquet"
A100 = ["optim.batch_size=4", "optim.accum=2", "model.grad_checkpoint=false", "data.num_workers=10"]


def sh(cmd: list[str], dry: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=REPO)


def fetch_parent(dataset: str, fold: int, root: str, dry: bool) -> str:
    """best.pt of fold `fold` from one of our weight datasets -> <root>/parents/<dataset slug>/fold<k>/best.pt."""
    out_dir = os.path.join(root, "parents", dataset.split("/")[-1], f"fold{fold}")
    path = os.path.join(out_dir, "best.pt")
    if os.path.exists(path) or dry:
        return path
    os.makedirs(out_dir, exist_ok=True)
    kaggle = shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")
    subprocess.run([kaggle, "datasets", "download", dataset, "-f", f"fold{fold}/best.pt", "-p", out_dir], check=True)
    for z in glob.glob(os.path.join(out_dir, "*.zip")):
        subprocess.run(["unzip", "-o", "-q", z, "-d", out_dir], check=True)
        os.remove(z)
    if not os.path.exists(path):
        raise SystemExit(f"parent weights not found after download: {path}")
    return path


def fold_done(work: str, name: str, fold: int, epochs: int) -> bool:
    hist = os.path.join(work, "runs", name, f"fold{fold}", "history.jsonl")
    if not os.path.exists(hist) or not os.path.exists(os.path.join(os.path.dirname(hist), "best.pt")):
        return False
    with open(hist) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    return bool(lines) and int(lines[-1]["epoch"]) >= epochs - 1


def train_call(a, name: str, fold: int, config: str, labels: str, extra: list[str], eval_table: str = "", upload: bool = False,
               first: bool = False) -> list[str]:
    cmd = [sys.executable, "colab/colab_train.py", "--name", name, "--folds", str(fold), "--config", config, "--label-table", labels,
           "--drive-root", a.drive_root, "--max-hours", "11"]
    if eval_table:
        cmd += ["--eval-label-table", eval_table]
    for o in [f"model.backbone={a.backbone}", *A100, *extra]:
        cmd += ["--override", o]
    if not first:
        cmd.append("--skip-pip")
    if upload:
        cmd.append("--upload")
    return cmd


def feature_overrides(a) -> list[str]:
    out = []
    if a.pair > 0:
        out += [f"optim.pair_weight={a.pair}", "optim.pair_last_epochs=3"]
    if a.cover224:
        out += ["data.img_size=224", "data.windows_per_study=48", "data.windows_per_study_eval=96"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="prefix of this generation, e.g. v10")
    ap.add_argument("--stage", choices=["oof", "labels", "prod", "all"], default="all")
    ap.add_argument("--backbone", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--report-labels", default="configs/labels/label_table_blend_v3.parquet", help="report-derived table the OOF predictions are mixed INTO")
    ap.add_argument("--teacher-labels", default="configs/labels/label_table_pseudo_v1.parquet", help="table the teachers train on")
    ap.add_argument("--labels", default="", help="prod only: train on this table instead of the one built by the labels stage")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--pair", type=float, default=0.0)
    ap.add_argument("--cover224", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--oof-folds", nargs="+", type=int, default=[2, 3, 4, 0, 1])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--teacher-parents", default="kirillkholkin/rsna-knee-runs-oof-nano")
    ap.add_argument("--prod-parents", default="kirillkholkin/rsna-knee-runs-v8-nano-pseudo-full")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/rsna-knee")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args()

    t0, first = time.time(), True
    feats = feature_overrides(a)
    new_table = f"configs/labels/label_table_pseudo_{a.name}.parquet"
    saved_table = os.path.join(a.drive_root, os.path.basename(new_table))
    print(f"pipeline {a.name}: stage={a.stage} warm={a.warm} gold={a.gold} pair={a.pair} cover224={a.cover224} alpha={a.alpha}")

    if a.stage in ("oof", "all"):
        name, work = f"{a.name}-oof", os.path.join(a.drive_root, f"work-{a.name}-oof")
        for k in a.oof_folds:
            if fold_done(work, name, k, a.epochs) and not a.dry_run:
                print(f"teacher fold {k}: already finished, skipped")
                continue
            extra = [f"optim.epochs={a.epochs}", *feats]
            if a.gold:
                extra.append("gold_train=true")
            if a.warm:
                extra.append(f"init_from={fetch_parent(a.teacher_parents, k, a.drive_root, a.dry_run)}")
            sh(train_call(a, name, k, "configs/v4_base.yaml", a.teacher_labels, extra, eval_table=REF_TABLE, first=first), a.dry_run)
            first = False

    if a.stage in ("labels", "all"):
        oof_glob = os.path.join(a.drive_root, f"work-{a.name}-oof", "runs", f"{a.name}-oof", "fold*", "oof.parquet")
        n = len(glob.glob(oof_glob))
        if n < 5 and not a.dry_run:
            raise SystemExit(f"only {n} of 5 teacher folds have oof.parquet under {oof_glob}: finish the oof stage first")
        env_cmd = [sys.executable, "scripts/make_pseudo_labels.py", "--labels", a.report_labels, "--oof", oof_glob, "--alpha", str(a.alpha), "--out", new_table]
        print("$ PYTHONPATH=. " + " ".join(env_cmd), flush=True)
        if not a.dry_run:
            subprocess.run(env_cmd, check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": REPO})
            shutil.copy(os.path.join(REPO, new_table), saved_table)  # /content/code is wiped by the setup cell: keep a copy on Drive

    if a.stage in ("prod", "all"):
        labels = a.labels or new_table
        if not a.labels and not os.path.exists(os.path.join(REPO, new_table)) and os.path.exists(saved_table):
            shutil.copy(saved_table, os.path.join(REPO, new_table))
        if not os.path.exists(os.path.join(REPO, labels)) and not a.dry_run:
            raise SystemExit(f"label table {labels} is missing: run the labels stage or pass --labels")
        name, work = f"{a.name}-full", os.path.join(a.drive_root, f"work-{a.name}-full")
        todo = [s for s in a.seeds if a.dry_run or not fold_done(work, name, s, a.epochs)]
        for s in a.seeds:
            if s not in todo:
                print(f"production seed {s}: already finished, skipped")
        for i, s in enumerate(todo):
            # with gold in training the gold numbers are in-sample: keep the last epochs instead of selecting on them
            extra = [f"optim.epochs={a.epochs}", f"select_by={'last' if a.gold else 'gold'}", *feats]
            if a.gold:
                extra.append("gold_train=true")
            if a.warm:
                extra.append(f"init_from={fetch_parent(a.prod_parents, s % 3, a.drive_root, a.dry_run)}")
            last = i == len(todo) - 1
            sh(train_call(a, name, s, "configs/v3c_simplehead_effv2s.yaml", labels, extra, upload=last and not a.no_upload, first=first), a.dry_run)
            first = False
        if not a.dry_run:
            for h in sorted(glob.glob(os.path.join(work, "runs", name, "fold*", "history.jsonl"))):
                with open(h) as f:
                    rec = [json.loads(x) for x in f if x.strip()][-1]
                note = " (IN-SAMPLE: gold trained)" if a.gold else ""
                print(f"{h.split(os.sep)[-2]}: last epoch gold {rec.get('gold_macro_auc', float('nan')):.4f}{note}")
    print(f"pipeline {a.name} done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
