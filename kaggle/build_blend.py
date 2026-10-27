#!/usr/bin/env python
"""Append OUR inference as an extra stage to the forked public-ensemble notebook and rank-blend the two.

    python kaggle/build_blend.py --runs-filter v2-effv2s-r4 --weight 0.15 \
        --kernel-source kirillkholkin/rsna-knee-train-v2-effv2s-r4 --slug rsna-knee-blend-a
    kaggle kernels push -p build/blend/rsna-knee-blend-a

The public stages run untouched and write submission.csv; our stage runs in a SUBPROCESS (clean CUDA context, a crash
cannot take the public result down), writes ours.csv, and the final cell replaces submission.csv by the per-finding
rank blend. If our stage fails or runs out of time, the public submission is kept as is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kaggle"))
import launch  # noqa: E402

STAGE_RUN = '''# === OWN STAGE: study-transformer models trained on LLM-blend labels (runs in a subprocess) ===
import base64, gc, os, subprocess, sys, time
_t_own = time.time()
try:
    import torch
    gc.collect(); torch.cuda.empty_cache()
except Exception as _e:
    print("cache release skipped:", _e)
os.makedirs("/kaggle/working/_own", exist_ok=True)
with open("/kaggle/working/_own/own_infer.py", "wb") as _f:
    _f.write(base64.b64decode("{script_b64}"))
_env = dict(os.environ, RSNA_WORK_DIR="/kaggle/working/_own", RSNA_CODE_DIR="/tmp/rsna_code_own")
_budget_s = {own_budget_s}
try:
    _r = subprocess.run([sys.executable, "/kaggle/working/_own/own_infer.py"], env=_env, timeout=_budget_s, capture_output=True, text=True)
    print(_r.stdout[-3000:]); print(_r.stderr[-1500:])
    print(f"own stage exit code {{_r.returncode}} in {{time.time() - _t_own:.0f}}s")
except subprocess.TimeoutExpired:
    print(f"own stage exceeded {{_budget_s}}s - keeping the public submission")
'''

STAGE_BLEND = '''# === FINAL: per-finding rank blend of the public ensemble and the own stage ===
import os
import pandas as pd
W_DEFAULT, W_BY_LABEL = {weight}, {weights_by_label}
pub_path, own_path = "/kaggle/working/submission.csv", "/kaggle/working/_own/ours.csv"
pub = pd.read_csv(pub_path)
if os.path.exists(own_path):
    own = pd.read_csv(own_path).set_index(pub.columns[0]).reindex(pub[pub.columns[0]])
    out = pub.copy()
    for c in pub.columns[1:]:
        w = W_BY_LABEL.get(c, W_DEFAULT)
        if c in own.columns and own[c].notna().all() and w > 0:
            out[c] = (1 - w) * pub[c].rank(method="average").values / len(pub) + w * own[c].rank(method="average").values / len(pub)
    out.to_csv(pub_path, index=False)
    print("blended", len(out), "rows; weights:", {{c: W_BY_LABEL.get(c, W_DEFAULT) for c in pub.columns[1:]}})
    try:
        print("spearman public vs own:", {{c: round(float(pub[c].corr(pd.Series(own[c].values), method="spearman")), 3) for c in pub.columns[1:]}})
    except Exception as _e:
        print("correlation skipped:", _e)
else:
    print("own predictions missing - public submission kept unchanged")
for _p in ("/kaggle/working/_own/own_infer.py",):
    if os.path.exists(_p):
        os.remove(_p)
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="rsna-knee-blend-a")
    ap.add_argument("--base", default=str(REPO / "build" / "fork941"))
    ap.add_argument("--kernel-source", action="append", default=[], help="own train kernels holding runs/*/fold*/best.pt")
    ap.add_argument("--dataset-source", action="append", default=[], help="own weight datasets (user/slug) holding fold*/best.pt")
    ap.add_argument("--runs-filter", action="append", default=[], help="substring(s) of run dirs to use")
    ap.add_argument("--weight", type=float, default=0.15)
    ap.add_argument("--weights-by-label", default="{}", help='JSON, e.g. {"Synovitis": 0.3}')
    ap.add_argument("--own-budget-s", type=int, default=5400)
    ap.add_argument("--n-slices-eval", type=int, default=32)
    ap.add_argument("--max-total", type=int, default=3)
    ap.add_argument("--label-mix", default="{}", help='JSON for the own stage, e.g. {"synovitis": {"effusion": 0.5}}')
    a = ap.parse_args()

    import base64

    params = {"out_name": "ours.csv", "runs_filter": a.runs_filter, "budget_hours": a.own_budget_s / 3600 * 0.9, "tta": False,
              "n_slices_eval": a.n_slices_eval, "max_total": a.max_total, "max_per_plane": 2 if a.max_total > 3 else 1, "label_mix": json.loads(a.label_mix)}
    script_dir = launch.build("infer", a.slug + "-ownstage", params, [], [], None)
    script_b64 = base64.b64encode((script_dir / "script.py").read_bytes()).decode()

    base = Path(a.base)
    nb = json.loads((base / "notebook.ipynb").read_text())
    nb["cells"].append(code_cell(STAGE_RUN.format(script_b64=script_b64, own_budget_s=a.own_budget_s)))
    nb["cells"].append(code_cell(STAGE_BLEND.format(weight=a.weight, weights_by_label=json.dumps(json.loads(a.weights_by_label)))))
    meta = json.loads((base / "kernel-metadata.json").read_text())
    user = launch.username()
    meta.update({"id": f"{user}/{a.slug}", "title": a.slug})
    meta["kernel_sources"] = list(dict.fromkeys(meta.get("kernel_sources", []) + a.kernel_source))
    meta["dataset_sources"] = list(dict.fromkeys(meta.get("dataset_sources", []) + [f"{user}/rsna-knee-wheels"] + a.dataset_source))
    out = REPO / "build" / "blend" / a.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {out}: {len(nb['cells'])} cells, own stage {len(script_b64) // 1024} KB, weight {a.weight}, sources +{a.kernel_source}")


if __name__ == "__main__":
    main()
