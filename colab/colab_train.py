#!/usr/bin/env python
"""Train on Google Colab with the Kaggle-built cache v2. Run from the code checkout (/content/code).

    python colab/colab_train.py --name colab-r6 --folds 0 3 --param optim.epochs=10 \
        --override optim.batch_size=4 --override model.grad_checkpoint=false --upload

Steps: Kaggle auth -> pip deps -> competition CSVs -> cache v2 shard outputs (~19 GB, cached on local disk) ->
training via kaggle/launch.py run-local (checkpoints + resume on Google Drive) -> optional upload of the run dirs
as a private Kaggle dataset (rsna-knee-runs-<name>) that the inference kernels pick up via --dataset-source.
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

CACHE_KERNELS = ["kirillkholkin/rsna-knee-cache-0of3", "kirillkholkin/rsna-knee-cache-1of3", "kirillkholkin/rsna-knee-cache-2of3"]
COMP = "rsna-knee-abnormality-detection"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=kw.pop("check", True), **kw)


def ensure_kaggle_auth() -> None:
    tok = os.environ.get("KAGGLE_API_TOKEN")
    path = os.path.expanduser("~/.kaggle/access_token")
    if tok and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(tok.strip()); os.chmod(path, 0o600)
    if not (tok or os.path.exists(path) or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))):
        raise SystemExit("no Kaggle credentials: set KAGGLE_API_TOKEN (Colab secret) or upload ~/.kaggle/access_token")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 3])
    ap.add_argument("--config", default="configs/v2_effv2s.yaml")
    ap.add_argument("--param", action="append", default=[], help="kernel param k=v (json value)")
    ap.add_argument("--override", action="append", default=[], help="training config override key=value")
    ap.add_argument("--label-table", default="configs/labels/label_table_blend_v1.parquet")
    ap.add_argument("--eval-label-table", default="", help="fixed reference table for validation targets")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/rsna-knee")
    ap.add_argument("--data-root", default="/content/rsna_data")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--wide-cache", action="store_true", help="use the 180 mm cache (kernels rsna-knee-cache-w180-*, tag _v3): crop late via data.cache_mm / data.view_mm")
    ap.add_argument("--no-mirror-cache", action="store_true", help="wide cache WITHOUT right->left mirroring (kernels rsna-knee-cache-w180nomir-*, tag _v4); sets data.lat_normalize=false")
    ap.add_argument("--upload", action="store_true", help="publish run dirs as a private Kaggle dataset when done")
    ap.add_argument("--skip-pip", action="store_true")
    ap.add_argument("--no-download", action="store_true", help="use whatever is already under --data-root (dry runs)")
    a = ap.parse_args()

    ensure_kaggle_auth()
    kaggle = shutil.which("kaggle") or f"{sys.executable} -m kaggle"
    if not a.skip_pip:
        # Colab already ships torch / cv2 / pandas: install only what is missing, never reinstall opencv or torch
        sh([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle>=2.0"], check=False)
        sh([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.15", "pydicom>=3.0", "pyarrow", "scikit-learn", "pyyaml", "tqdm"], check=False)
    variant = "w180nomir" if a.no_mirror_cache else "w180"
    a.wide_cache = a.wide_cache or a.no_mirror_cache
    if a.no_mirror_cache and not any(o.startswith("data.lat_normalize=") for o in a.override):
        a.override.append("data.lat_normalize=false")  # inference must not mirror either
    if a.wide_cache:  # a separate tree: the caches must never mix under one RSNA_INPUT_DIR
        a.data_root = a.data_root.rstrip("/") + "_" + variant
        if not any(o.startswith("data.cache_mm=") for o in a.override):
            a.override += ["data.cache_mm=180", "data.cache_size=384"]
        if not any(o.startswith("data.view_mm=") for o in a.override):
            a.override.append("data.view_mm=140")
    tag = ("_v4" if a.no_mirror_cache else "_v3") if a.wide_cache else "_v2"
    kernels = [k.replace("rsna-knee-cache-", f"rsna-knee-cache-{variant}-") for k in CACHE_KERNELS] if a.wide_cache else CACHE_KERNELS
    comp = os.path.join(a.data_root, "comp"); kin = os.path.join(a.data_root, "kin"); os.makedirs(comp, exist_ok=True); os.makedirs(kin, exist_ok=True)
    for f in ("train.csv", "train_series.csv", "sample_submission.csv"):
        if not os.path.exists(os.path.join(comp, f)) and not a.no_download:
            sh([kaggle, "competitions", "download", "-c", COMP, "-f", f, "-p", comp], check=False)
            z = os.path.join(comp, f + ".zip")
            if os.path.exists(z):
                sh(["unzip", "-o", "-q", z, "-d", comp]); os.remove(z)
    if not a.no_download:
        sys.path.insert(0, os.getcwd())
        from rsna_knee.kaggle_io import download_kernel_outputs  # streaming + resume (the CLI restarts multi-GB files from zero)
        for i, k in enumerate(kernels):
            download_kernel_outputs(k, os.path.join(kin, f"cache-{i}"), tag + r"_shard.*\.(tar|parquet|json)$")
    print("cache files:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(kin, "*", "*"))))

    work = os.path.join(a.drive_root, f"work-{a.name}")
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ, RSNA_INPUT_DIR=kin, RSNA_TMP_DIR=os.path.join(a.data_root, "tmp"), PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "kaggle/launch.py", "run-local", "train", "--name", a.name, "--comp-root", comp, "--work", work,
           "--embed", "configs/folds_v1.parquet", "--embed", a.label_table,
           "--param", f"folds={json.dumps(a.folds)}", "--param", f"max_hours={a.max_hours}", "--param", f'cache_tag="{tag}"',
           "--param", f"config={a.config}", "--param", "folds_file=configs/folds_v1.parquet", "--param", f"label_table_file={a.label_table}",
           "--param", f"overrides={json.dumps(a.override)}"] + [x for p in a.param for x in ("--param", p)]
    if a.eval_label_table:
        cmd += ["--embed", a.eval_label_table, "--param", f"eval_label_table_file={a.eval_label_table}"]
    t0 = time.time()
    log_path = os.path.join(work, "train.log")  # on Drive: readable from the Files panel while the cell is running
    print("$", " ".join(cmd), f"\nlive log: {log_path}", flush=True)
    noise = ("DeprecationWarning", "tarfile.open", "HTTP Request", "HF_TOKEN", "Safe alternative")
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc, \
            open(log_path, "a", buffering=1) as logf:
        logf.write(f"\n===== {time.strftime('%F %T')} {a.name} folds={a.folds} =====\n")
        for line in proc.stdout:
            if len(line) > 2000 or any(n in line for n in noise):
                continue
            sys.stdout.write(line); sys.stdout.flush()
            logf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"training failed with exit code {rc}; see {log_path}")
    print(f"training finished in {(time.time() - t0) / 3600:.2f} h; runs under {work}/runs")

    if a.upload:
        runs = os.path.join(work, "runs")
        slug = f"rsna-knee-runs-{a.name}".lower()
        user = subprocess.run([sys.executable, "kaggle/launch.py", "whoami"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        meta = {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
        json.dump(meta, open(os.path.join(runs, "dataset-metadata.json"), "w"))
        r = sh([kaggle, "datasets", "status", meta["id"]], check=False, capture_output=True, text=True)
        if r.returncode == 0 and "not found" not in (r.stdout + r.stderr).lower():
            sh([kaggle, "datasets", "version", "-p", runs, "-m", f"update {time.strftime('%F %H:%M')}", "--dir-mode", "zip"])
        else:
            sh([kaggle, "datasets", "create", "-p", runs, "--dir-mode", "zip"])
        print(f"weights published: https://www.kaggle.com/datasets/{meta['id']}  -> use --dataset-source {meta['id']} for inference kernels")


if __name__ == "__main__":
    main()
