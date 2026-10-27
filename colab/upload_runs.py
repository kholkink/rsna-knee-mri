#!/usr/bin/env python
"""Publish finished run dirs from Google Drive as a private Kaggle dataset (use after interrupting a training cell).

    python colab/upload_runs.py --name v3-effv2s
"""
from __future__ import annotations

import argparse, glob, json, os, shutil, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--drive-root", default="/content/drive/MyDrive/rsna-knee")
a = ap.parse_args()
runs = os.path.join(a.drive_root, f"work-{a.name}", "runs")
folds = [d for d in sorted(glob.glob(os.path.join(runs, "*", "fold*"))) if os.path.exists(os.path.join(d, "best.pt"))]
if not folds:
    raise SystemExit(f"no finished fold (best.pt) under {runs}")
stage = tempfile.mkdtemp()
for d in folds:  # only what inference needs: weights, config, priors, history
    out = os.path.join(stage, os.path.relpath(d, runs)); os.makedirs(out, exist_ok=True)
    for f in ("best.pt", "config.yaml", "priors.json", "history.jsonl"):
        if os.path.exists(os.path.join(d, f)):
            shutil.copy(os.path.join(d, f), out)
    print("staged", os.path.relpath(d, runs), [json.loads(l).get("gold_macro_auc") for l in open(os.path.join(d, "history.jsonl"))][-3:])
user = subprocess.run([sys.executable, "kaggle/launch.py", "whoami"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
slug = f"rsna-knee-runs-{a.name}".lower()
json.dump({"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(stage, "dataset-metadata.json"), "w"))
kaggle = shutil.which("kaggle") or "kaggle"
st = subprocess.run([kaggle, "datasets", "status", f"{user}/{slug}"], capture_output=True, text=True)
cmd = ([kaggle, "datasets", "version", "-p", stage, "-m", f"update {time.strftime('%F %H:%M')}", "--dir-mode", "zip"]
       if st.returncode == 0 and "not found" not in (st.stdout + st.stderr).lower() else [kaggle, "datasets", "create", "-p", stage, "--dir-mode", "zip"])
print("$", " ".join(cmd)); subprocess.run(cmd, check=True)
print(f"published https://www.kaggle.com/datasets/{user}/{slug} with {len(folds)} fold(s)")
