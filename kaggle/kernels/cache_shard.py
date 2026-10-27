# ---- kernel body: metadata scan + JPEG cache for one shard of studies (CPU kernel, internet on) ----
# KERNEL_PARAMS / code snapshot are injected above this line by kaggle/launch.py
import json, os, subprocess, sys, tarfile, time

T0 = time.time()
P = {"shard": "0/1", "size": 384, "max_slices": 48, "quality": 90, "max_per_plane": 1, "max_total": 3,
     "limit_studies": 0, "split": "train", "crop_mm": 0.0, "canonical": False, "lat_normalize": False, "tag": "",
     **KERNEL_PARAMS}
ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = os.environ.get("RSNA_WORK_DIR", "/kaggle/working" if ON_KAGGLE else "work")
TMP = os.environ.get("RSNA_TMP_DIR", "/tmp/rsna_cache")
os.makedirs(WORK, exist_ok=True); os.makedirs(TMP, exist_ok=True)

if ON_KAGGLE:  # what does the offline image already ship? (matters for the no-internet submission notebook)
    out = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True).stdout
    print("preinstalled:", [l for l in out.splitlines() if any(k in l.lower() for k in ("pydicom", "pylibjpeg", "gdcm", "dicomsdl", "timm", "openjpeg"))])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom>=3.0", "pylibjpeg>=2.0", "pylibjpeg-libjpeg",
                    "pylibjpeg-openjpeg"], check=False)

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from rsna_knee.dicom import SeriesMeta, knee_side, scan_series_header
from rsna_knee.paths import find_comp_root, find_image_root
from rsna_knee.preprocess import CacheJob, build_cache
from rsna_knee.series_select import SelectionPolicy, select_series

comp = find_comp_root()
img_root = find_image_root(comp, P["split"])
shard_i, shard_n = (int(x) for x in P["shard"].split("/"))
series_csv = os.path.join(comp, f"{P['split']}_series.csv")
print(f"comp={comp} images={img_root} shard={P['shard']} cpu={os.cpu_count()}")

if os.path.exists(series_csv):
    sdf = pd.read_csv(series_csv, dtype=str)
    pairs = list(zip(sdf["StudyInstanceUID"], sdf["SeriesInstanceUID"]))
else:
    pairs = [(st, se) for st in sorted(os.listdir(img_root)) for se in sorted(os.listdir(os.path.join(img_root, st)))]
order = {u: k for k, u in enumerate(sorted({st for st, _ in pairs}))}
pairs = [(st, se) for st, se in pairs if order[st] % shard_n == shard_i]
if P["limit_studies"]:
    keep = set(sorted({st for st, _ in pairs})[: P["limit_studies"]])
    pairs = [p for p in pairs if p[0] in keep]
print(f"{len(pairs)} series in {len({st for st, _ in pairs})} studies")


def _scan(p):
    return scan_series_header(os.path.join(img_root, p[0], p[1]), p[0], p[1]).to_dict()


workers = max(2, os.cpu_count() or 4)
with ProcessPoolExecutor(workers) as ex:
    meta = pd.DataFrame(list(ex.map(_scan, pairs, chunksize=32)))
tag = f"{P['split']}{P['tag']}_shard{shard_i}of{shard_n}"
meta.to_parquet(os.path.join(WORK, f"meta_{tag}.parquet"), index=False)
print(f"scan done in {time.time() - T0:.0f}s; errors={int((meta['error'] != '').sum())}")
print(meta[["plane", "weighting", "fat_sat"]].value_counts().head(12).to_string())

policy = SelectionPolicy(max_per_plane=P["max_per_plane"], max_total=P["max_total"])
jobs = []
sides = {}
for uid, g in meta[meta["error"] == ""].groupby("study_uid", sort=True):
    metas = [SeriesMeta.from_dict(r) for r in g.to_dict("records")]
    sides[uid] = knee_side([m.center_x for m in metas])
    mirror = bool(P["lat_normalize"]) and sides[uid] == "right"
    for m in select_series(metas, policy):
        jobs.append(CacheJob(m, os.path.join(TMP, str(uid), f"{m.series_uid}.npz"), P["size"], "pad", P["quality"], P["max_slices"],
                             crop_mm=P["crop_mm"] or None, canonical=bool(P["canonical"]), mirror=mirror))
meta["knee_side"] = meta["study_uid"].map(sides)
meta.to_parquet(os.path.join(WORK, f"meta_{tag}.parquet"), index=False)
print("knee side:", pd.Series(sides).value_counts().to_dict(),
      "| agreement with Laterality tag:", pd.crosstab(meta.drop_duplicates("study_uid")["laterality"].str.upper().str[:1], meta.drop_duplicates("study_uid")["knee_side"]).to_dict())
print(f"{len(jobs)} series selected")

tar_path = os.path.join(WORK, f"cache_{tag}.tar")
tar = tarfile.open(tar_path, "w")
state = {"bytes": 0, "n": 0}


def on_done(row):  # stream each finished npz into the tar and free the temp file (output dir has a file-count cap)
    if row["status"] in ("ok", "cached") and os.path.exists(row["path"]):
        arc = os.path.relpath(row["path"], TMP)
        state["bytes"] += os.path.getsize(row["path"]); state["n"] += 1
        tar.add(row["path"], arcname=arc)
        os.remove(row["path"])
        row["path"] = arc  # store paths relative to the cache root
    if state["n"] and state["n"] % 500 == 0:
        print(f"  {state['n']} series, {state['bytes'] / 1e9:.2f} GB, {time.time() - T0:.0f}s", flush=True)


idx = build_cache(jobs, workers=workers, progress=False, on_done=on_done)
tar.close()
idx.to_parquet(os.path.join(WORK, f"cache_index_{tag}.parquet"), index=False)
summary = {"tag": tag, "series": int(len(idx)), "ok": int(idx["status"].isin(["ok", "cached"]).sum()),
           "errors": idx.loc[idx["status"] == "error", "error"].head(5).tolist(), "tar_gb": round(os.path.getsize(tar_path) / 1e9, 2),
           "seconds": round(time.time() - T0)}
print(json.dumps(summary))
json.dump(summary, open(os.path.join(WORK, f"summary_{tag}.json"), "w"))
