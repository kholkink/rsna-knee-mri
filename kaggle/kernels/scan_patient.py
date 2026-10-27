# ---- kernel body: one DICOM header per study -> patient-level tags (sex, age, weight, size) (CPU kernel) ----
# KERNEL_PARAMS / code snapshot are injected above this line by kaggle/launch.py
import os, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pydicom

T0 = time.time()
P = {"split": "train", "limit_studies": 0, **KERNEL_PARAMS}
ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = os.environ.get("RSNA_WORK_DIR", "/kaggle/working" if ON_KAGGLE else "work")
os.makedirs(WORK, exist_ok=True)

from rsna_knee.paths import find_comp_root, find_image_root

root = find_image_root(find_comp_root(), P["split"])
studies = sorted(os.listdir(root))
if P["limit_studies"]:
    studies = studies[: P["limit_studies"]]
TAGS = ["PatientSex", "PatientAge", "PatientBirthDate", "StudyDate", "PatientWeight", "PatientSize", "BodyPartExamined",
        "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "InstitutionName", "StationName"]


def first_header(study: str) -> dict:
    rec = {"study_uid": study}
    sdir = os.path.join(root, study)
    try:
        for series in sorted(os.listdir(sdir)):
            files = sorted(f for f in os.listdir(os.path.join(sdir, series)) if f.endswith(".dcm"))
            if not files:
                continue
            ds = pydicom.dcmread(os.path.join(sdir, series, files[0]), stop_before_pixels=True, specific_tags=TAGS)
            for t in TAGS:
                v = getattr(ds, t, None)
                rec[t] = "" if v is None else str(v)
            break
    except Exception as e:  # unreadable study: keep the row, empty tags
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    return rec


with ThreadPoolExecutor(max_workers=16) as ex:
    rows = list(ex.map(first_header, studies))
df = pd.DataFrame(rows)
df.to_parquet(os.path.join(WORK, f"patient_meta_{P['split']}.parquet"), index=False)
print(f"{len(df)} studies in {time.time() - T0:.0f}s")
for t in ("PatientSex", "PatientAge", "PatientBirthDate", "PatientWeight", "PatientSize", "InstitutionName", "StationName"):
    if t in df.columns:
        s = df[t].fillna("")
        print(f"{t}: filled {(s != '').mean():.1%}; top values {s[s != ''].value_counts().head(6).to_dict()}")
