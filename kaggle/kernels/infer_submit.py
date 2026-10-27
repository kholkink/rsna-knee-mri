# ---- kernel body: OFFLINE inference -> submission.csv (internet off, <= 9 h). Inputs: train kernel outputs + wheels dataset ----
import glob, logging, os, subprocess, sys, time

T0 = time.time()
P = {"tta": True, "batch_size": 2, "budget_hours": 8.0, "max_per_plane": 1, "max_total": 3, "limit": 0,
     "out_name": "submission.csv", "n_slices_eval": 0, "ensemble": "rank", "slice_offsets": [0.0], "label_mix": {}, "debug_gold": False, "debug_extra": 150, **KERNEL_PARAMS}
INPUT = os.environ.get("RSNA_INPUT_DIR", "/kaggle/input")
WORK = os.environ.get("RSNA_WORK_DIR", "/kaggle/working" if os.path.isdir("/kaggle/input") else "work")
os.makedirs(WORK, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

wheel_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "**", "pylibjpeg*.whl"), recursive=True)})
if wheel_dirs:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"]
    for d in wheel_dirs:
        cmd += ["--find-links", d]
    r = subprocess.run(cmd + ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], capture_output=True, text=True)
    print("wheels:", wheel_dirs, "rc", r.returncode, (r.stderr or "")[-300:])
try:
    import libjpeg, openjpeg, pylibjpeg  # noqa: F401
    print("pylibjpeg decoders available")
except Exception as e:
    print(f"WARNING: compressed DICOM decoders missing ({e}) - JPEG2000/JPEG-Lossless series will fall back to priors")

from rsna_knee.infer import InferenceConfig, predict, write_submission
from rsna_knee.paths import find_comp_root
from rsna_knee.series_select import SelectionPolicy

comp = find_comp_root()
runs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "**", "best.pt"), recursive=True)})
only = P.get("runs_filter")
if only:
    runs = [r for r in runs if any(o in r for o in only)]
print(f"comp={comp}\nruns={runs}")
assert runs, "no best.pt found: attach the train kernel(s) as kernel_sources"
icfg = InferenceConfig(comp_root=comp, run_dirs=runs, tta_hflip=bool(P["tta"]), batch_size=P["batch_size"],
                       num_workers=os.cpu_count() or 4, time_budget_s=P["budget_hours"] * 3600,
                       policy=SelectionPolicy(max_per_plane=P["max_per_plane"], max_total=P["max_total"]), limit=P["limit"],
                       n_slices_eval=int(P["n_slices_eval"]), ensemble=P["ensemble"], slice_offsets=tuple(P["slice_offsets"]), label_mix=dict(P["label_mix"]))
if P["debug_gold"]:
    # TRAIN/SERVE SKEW CHECK: the expert-labelled training studies go through the live-DICOM path used on the hidden test;
    # the same models scored them through the JPEG cache during training (oof_best.parquet next to best.pt).
    import dataclasses
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from rsna_knee.constants import ID_COL, LABELS, resolve_label_columns
    tr = pd.read_csv(os.path.join(comp, "train.csv")); tr[ID_COL] = tr[ID_COL].astype(str)
    cols = resolve_label_columns(tr.columns)
    gold = tr[tr[[cols[k] for k in LABELS]].notna().all(axis=1)]
    extra = tr[~tr[ID_COL].isin(gold[ID_COL])].sample(n=min(int(P["debug_extra"]), len(tr) - len(gold)), random_state=0)
    uids = tuple(gold[ID_COL]) + tuple(extra[ID_COL])
    dcfg = dataclasses.replace(icfg, split="train", study_uids=uids, limit=0, label_mix={})
    dp = predict(dcfg)
    print(f"DEBUG live path: {len(dp)} train studies ({len(gold)} gold), unreadable: {len(dp.attrs['failed'])} -> {dp.attrs['failed'][:5]}")
    dp.to_csv(os.path.join(WORK, "debug_live_predictions.csv"), index=False)
    g = dp.set_index(ID_COL).loc[gold[ID_COL]]
    Y = gold[[cols[k] for k in LABELS]].values.astype(int)
    auc = lambda M: float(np.mean([roc_auc_score(Y[:, j], M[:, j]) for j in range(len(LABELS)) if 0 < Y[:, j].sum() < len(Y)]))
    print(f"DEBUG gold AUC, live path, ensemble: {auc(g[LABELS].values):.4f}")
    pos = {u: i for i, u in enumerate(dp[ID_COL])}
    gi = [pos[u] for u in gold[ID_COL]]
    for run, M in dp.attrs["members"].items():
        line = f"DEBUG {run.split('/')[-2]}/{run.split('/')[-1]}: live {auc(M[gi]):.4f}"
        oof_path = os.path.join(run, "oof_best.parquet")
        if os.path.exists(oof_path):
            o = pd.read_parquet(oof_path); o = o[o["is_gold"].astype(bool)].set_index("study_uid").reindex(gold[ID_COL].values)
            C = o[[f"p_{k}" for k in LABELS]].values
            if np.isfinite(C).all():
                corr = float(np.mean([np.corrcoef(M[gi][:, j], C[:, j])[0, 1] for j in range(len(LABELS))]))
                line += f" | cache {auc(C):.4f} | mean corr live-vs-cache {corr:.4f} | mean abs diff {float(np.abs(M[gi] - C).mean()):.4f}"
        print(line, flush=True)
pred = predict(icfg)
sub = write_submission(pred, os.path.join(comp, "sample_submission.csv"), os.path.join(WORK, P["out_name"]))
print(sub.head().to_string())
print(f"{len(sub)} rows, {time.time() - T0:.0f}s")
