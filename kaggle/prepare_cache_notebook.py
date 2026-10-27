# %% [markdown]
# # RSNA Knee - metadata scan + JPEG cache (run on Kaggle, where the DICOMs are)
# Attach the dataset `rsna-knee-code` (built with `bash kaggle/push_code.sh`). Set SHARD to "0/3", "1/3", "2/3"
# in three copies of this notebook: each shard writes ~7-9 GB, under the ~20 GB output limit. Save & Run All
# (commit) so the output persists; then add each output as a data source ("+ Add input" -> Your Work).
# %%
import os, subprocess, sys, time
t0 = time.time()
sys.path.insert(0, "/kaggle/input/rsna-knee-code")
os.chdir("/kaggle/working")
COMP = "/kaggle/input/rsna-knee-abnormality-detection"
SHARD = os.environ.get("SHARD", "0/3")        # <- change per notebook copy
SIZE, MAX_SLICES, QUALITY = 384, 48, 90

# metadata scan is cheap (one header per series, ~10 min); each shard recomputes it so notebooks are independent
subprocess.run([sys.executable, "/kaggle/input/rsna-knee-code/scripts/scan_metadata.py",
                "--image-root", f"{COMP}/train_images", "--series-csv", f"{COMP}/train_series.csv",
                "--out", "meta_train.parquet"], check=True)
subprocess.run([sys.executable, "/kaggle/input/rsna-knee-code/scripts/build_cache.py",
                "--meta", "meta_train.parquet", "--out", "cache", "--size", str(SIZE), "--max-slices", str(MAX_SLICES),
                "--quality", str(QUALITY), "--max-per-plane", "1", "--max-total", "3", "--shard", SHARD], check=True)
print(subprocess.run(["du", "-sh", "cache"], capture_output=True, text=True).stdout, f"{time.time() - t0:.0f}s")
