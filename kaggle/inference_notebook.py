# %% [markdown]
# # RSNA Knee - offline inference (<= 9 h, no internet)
# Attach three Kaggle datasets: `rsna-knee-code` (this repo), `rsna-knee-runs` (fold dirs with config.yaml,
# best.pt, priors.json) and `rsna-knee-wheels` (pip wheels built with `bash kaggle/make_wheels.sh`).
# %%
import os, subprocess, sys, time
t0 = time.time()
WHEELS = "/kaggle/input/rsna-knee-wheels"
if os.path.isdir(WHEELS):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", WHEELS, "-q",
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "timm", "pydicom"], check=False)
sys.path.insert(0, "/kaggle/input/rsna-knee-code")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from rsna_knee.infer import InferenceConfig, predict, write_submission
from rsna_knee.series_select import SelectionPolicy

COMP = "/kaggle/input/rsna-knee-abnormality-detection"
RUNS = sorted(os.path.join("/kaggle/input/rsna-knee-runs", d) for d in os.listdir("/kaggle/input/rsna-knee-runs")
              if os.path.exists(os.path.join("/kaggle/input/rsna-knee-runs", d, "best.pt")))
print("runs:", RUNS)
icfg = InferenceConfig(comp_root=COMP, run_dirs=RUNS, tta_hflip=True, batch_size=2, num_workers=os.cpu_count() or 4,
                       time_budget_s=8.0 * 3600, policy=SelectionPolicy(max_per_plane=1, max_total=3))
pred = predict(icfg)
sub = write_submission(pred, os.path.join(COMP, "sample_submission.csv"), "submission.csv")
print(sub.head(), f"\n{len(sub)} rows in {time.time() - t0:.0f}s")
