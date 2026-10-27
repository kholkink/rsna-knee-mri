# ---- kernel body: unpack cache tars, build labels + grouped folds, train fold(s) (one process per GPU) ----
import glob, json, os, subprocess, sys, tarfile, time

T0 = time.time()
P = {"config": "configs/base.yaml", "overrides": [], "folds": [0, 1], "n_folds": 5, "max_hours": 10.5,
     "calibrate": True, "run_name": "exp", "label_source": "auto", "folds_file": "", "label_table_file": "", "eval_label_table_file": "", "cache_tag": "",
     "seeds": [], "init_parents": "", **KERNEL_PARAMS}  # seeds: optional, parallel to folds -> several seeds of ONE validation fold in one kernel (runs/<name>-s<seed>/fold<k>)
ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = os.environ.get("RSNA_WORK_DIR", "/kaggle/working" if ON_KAGGLE else "work")
INPUT = os.environ.get("RSNA_INPUT_DIR", "/kaggle/input")
CACHE = os.environ.get("RSNA_TMP_DIR", "/tmp/rsna") + "/cache"
CODE = os.environ.get("RSNA_CODE_DIR", "/tmp/rsna_code")
os.makedirs(WORK, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

import pandas as pd
import yaml
from rsna_knee.constants import ID_COL, LABELS
from rsna_knee.cv import make_folds, study_groups
from rsna_knee.paths import find_comp_root
from rsna_knee.reports.extract_llm import load_extractions
from rsna_knee.reports.labels import (LabelMapConfig, build_label_table, calibrate, evaluate_mapping, gold_table,
                                      label_table_from_graded_csv)
from rsna_knee.reports.rules import rule_extract


def find(pattern):
    return sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))


# 1. cache: tars -> local disk, one merged index
tars = find(f"cache_train{P['cache_tag']}_shard*.tar")
assert tars, f"no cache tars under {INPUT}: attach the cache kernels as kernel_sources"
for t in tars:
    marker = os.path.join(CACHE, f".done_{os.path.basename(t)}")
    if os.path.exists(marker):
        print(f"already unpacked {os.path.basename(t)}", flush=True)
        continue
    with tarfile.open(t) as tf:
        tf.extractall(CACHE)
    open(marker, "w").close()
    print(f"unpacked {os.path.basename(t)} ({time.time() - T0:.0f}s)", flush=True)
idx = pd.concat([pd.read_parquet(p) for p in find(f"cache_index_train{P['cache_tag']}_shard*.parquet")], ignore_index=True)
meta = pd.concat([pd.read_parquet(p) for p in find(f"meta_train{P['cache_tag']}_shard*.parquet")], ignore_index=True)
idx.to_parquet(os.path.join(WORK, "cache_index.parquet"), index=False)
print(f"cache: {len(idx)} series / {idx['study_uid'].nunique()} studies; status={idx['status'].value_counts().to_dict()}")

# 2. labels: LLM extractions if a kernel output provides them, rules as fallback, gold overrides
comp = find_comp_root()
train = pd.read_csv(os.path.join(comp, "train.csv")); train[ID_COL] = train[ID_COL].astype(str)
ext_files = find("extractions*.jsonl")
ext = load_extractions(ext_files[0]) if ext_files else {}
rules = {r[ID_COL]: rule_extract(str(r["Report"])) for _, r in train.iterrows()}
cfg_map, gold = LabelMapConfig(), gold_table(train)
if ext and len(gold):
    print(evaluate_mapping(ext, gold, cfg_map).to_string(index=False))
    if P["calibrate"]:
        cfg_map, rep = calibrate(ext, gold, cfg_map)
        print(rep.to_string(index=False))
graded_files = find("report_labels*.csv")
src = P["label_source"]
if src == "auto":
    src = "graded" if graded_files else ("llm" if ext else "rules")
print(f"label source: {src} (graded files: {graded_files}, own extractions: {len(ext)})")
if P["label_table_file"]:  # ready-made table shipped inside the code snapshot (e.g. blended readers)
    src = "table:" + P["label_table_file"]
    table = pd.read_parquet(os.path.join(CODE, P["label_table_file"]))
elif src == "graded":
    table = label_table_from_graded_csv(train, pd.read_csv(graded_files[0]), cfg_map, rules)
elif src == "llm":
    table = build_label_table(train, ext, cfg_map, rules)
else:
    table = build_label_table(train, {}, cfg_map, rules)
table.to_parquet(os.path.join(WORK, "label_table.parquet"), index=False)
yaml.safe_dump(cfg_map.to_dict(), open(os.path.join(WORK, "label_map.yaml"), "w"))
print("label sources:", table["src"].value_counts().to_dict())

# 3. scanner-grouped folds
if P["folds_file"] and os.path.exists(os.path.join(CODE, P["folds_file"])):  # frozen folds: comparable runs across label sources
    folds = pd.read_parquet(os.path.join(CODE, P["folds_file"]))
    print("using frozen folds", P["folds_file"])
else:
    folds = make_folds(table, study_groups(meta), n_folds=P["n_folds"], seed=42)
folds.to_parquet(os.path.join(WORK, "folds.parquet"), index=False)
print("fold sizes:", folds["fold"].value_counts().sort_index().to_dict())

# 4. train: one subprocess per fold, pinned to its own GPU
import torch
n_gpu = max(1, torch.cuda.device_count())
base = [f"data.cache_index={WORK}/cache_index.parquet", f"data.cache_root={CACHE}", f"data.label_table={WORK}/label_table.parquet",
        f"data.folds={WORK}/folds.parquet", f"output_dir={WORK}/runs/{P['run_name']}", f"n_folds={P['n_folds']}",
        f"max_hours={max(0.5, P['max_hours'] - (time.time() - T0) / 3600)}", f"data.num_workers={max(1, (os.cpu_count() or 4) // n_gpu)}"]
if P["eval_label_table_file"]:
    base.append(f"data.eval_label_table={os.path.join(CODE, P['eval_label_table_file'])}")
procs, failed = [], []
for k, fold in enumerate(P["folds"]):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k % n_gpu), "PYTHONPATH": CODE}
    cmd = [sys.executable, "-m", "rsna_knee.train", "--config", os.path.join(CODE, P["config"]), "--fold", str(fold)] + base + list(P["overrides"])
    if P["seeds"]:
        cmd += [f"seed={P['seeds'][k]}", f"output_dir={WORK}/runs/{P['run_name']}-s{P['seeds'][k]}"]
    if P["init_parents"]:  # warm start: fold k starts from <attached dataset or kernel>/.../<init_parents>/fold<k>/best.pt
        cand = [c for c in find("best.pt") if P["init_parents"] in c and os.path.basename(os.path.dirname(c)) == f"fold{fold % 3}"]
        assert cand, f"no parent weights for fold {fold}: attach the dataset holding {P['init_parents']}/fold{fold % 3}/best.pt"
        cmd.append(f"init_from={cand[0]}")
    print("$", " ".join(cmd), flush=True)
    procs.append((fold, subprocess.Popen(cmd, env=env, cwd=CODE)))
    if len(procs) == n_gpu or k == len(P["folds"]) - 1:
        for f, pr in procs:
            rc = pr.wait()
            print(f"fold {f} exit code {rc}", flush=True)
            if rc != 0:
                failed.append(f)
        procs = []
for d in glob.glob(f"{WORK}/runs/{P['run_name']}*/fold*"):  # keep outputs small: drop resumable state
    lp = os.path.join(d, "last.pt")
    if os.path.exists(lp) and os.path.exists(os.path.join(d, "best.pt")):
        os.remove(lp)
    hist = os.path.join(d, "history.jsonl")
    if os.path.exists(hist):
        print(d, open(hist).read().strip().splitlines()[-1][:600])
print(f"done in {(time.time() - T0) / 3600:.2f}h")
if failed:
    raise SystemExit(f"folds failed: {failed}")
