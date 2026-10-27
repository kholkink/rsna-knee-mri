"""Robust, deterministic DICOM series loading for knee MRI.

Design goals
------------
* never crash on odd files (JPEG-Lossless / JPEG2000 transfer syntaxes, multi-frame,
  MONOCHROME1, missing geometry tags, mixed matrix sizes, dual-echo series)
* slice order defined in patient (LPS) coordinates, never by file name
* one preprocessing function (:func:`preprocess_series`) shared by the cache builder and by
  the offline Kaggle inference notebook, so train / test pixels are produced identically
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import pydicom

from .constants import FLUID_SENSITIVE, PLANE_INDEX, seq_id

log = logging.getLogger(__name__)

HEADER_TAGS = [
    "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "SeriesDescription", "SeriesNumber",
    "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
    "SliceThickness", "SpacingBetweenSlices", "Rows", "Columns", "NumberOfFrames",
    "PhotometricInterpretation", "BitsStored", "PixelRepresentation", "RescaleSlope",
    "RescaleIntercept", "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
    "MagneticFieldStrength", "ImagingFrequency", "ReceiveCoilName", "RepetitionTime", "EchoTime",
    "InversionTime", "FlipAngle", "ScanOptions", "ScanningSequence", "SequenceVariant",
    "MRAcquisitionType", "Laterality", "BodyPartExamined", "PatientPosition", "Modality",
]

_LOCALIZER_RE = re.compile(r"loc|scout|survey|calib|smartprep|screen|posdisp|\bmip\b|report|secondary", re.IGNORECASE)

_FS_PATTERNS = [
    r"(?<![a-z])fs(?![a-z])", r"(?:pd|t2|t1|dp)[\s_\-]?fs(?![a-z])", r"fat[\s_\-]?sat", r"fatsat",
    r"fat[\s_\-]?sup", r"spair", r"(?<![a-z])spir(?![a-z])", r"stir", r"tirm", r"sat[\s_\-]?gras",
    r"fett", r"жиропод", r"fatsup", r"(?<![a-z])fsat", r"(?<![a-z])fse?[\s_\-]?fs",
]
_FS_RE = re.compile("|".join(_FS_PATTERNS), re.IGNORECASE)
_STIR_RE = re.compile(r"stir|tirm", re.IGNORECASE)
_T1_RE = re.compile(r"(?<![a-z0-9])t1(?![0-9])", re.IGNORECASE)
_T2_RE = re.compile(r"(?<![a-z0-9])t2(?![0-9])", re.IGNORECASE)
_PD_RE = re.compile(r"(?<![a-z])(pd|dp|proton|protonen|densit)", re.IGNORECASE)


# --------------------------------------------------------------------------------------- helpers
def _f(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
            v = v[0]
        return float(v)
    except (TypeError, ValueError):
        return default


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return "\\".join(str(x) for x in v)
    return str(v)


def _floats(v: Any, n: int) -> list[float] | None:
    try:
        vals = [float(x) for x in v]
        return vals if len(vals) == n else None
    except Exception:
        return None


def list_dicom_files(series_dir: str) -> list[str]:
    if not os.path.isdir(series_dir):
        return []
    names = sorted(os.listdir(series_dir))
    dcm = [n for n in names if n.lower().endswith(".dcm")]
    if not dcm:  # files without extension
        dcm = [n for n in names if not n.startswith(".") and n.upper() != "DICOMDIR"]
    return [os.path.join(series_dir, n) for n in dcm]


def read_header(path: str) -> pydicom.Dataset:
    return pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=HEADER_TAGS)


def plane_from_iop(iop: Sequence[float] | None) -> str:
    """Dominant axis of the slice normal: x -> sagittal, y -> coronal, z -> axial (LPS)."""
    vals = _floats(iop, 6) if iop is not None else None
    if vals is None:
        return "unknown"
    r, c = np.asarray(vals[:3]), np.asarray(vals[3:])
    n = np.cross(r, c)
    if not np.isfinite(n).all() or np.abs(n).max() < 1e-6:
        return "unknown"
    return ["sagittal", "coronal", "axial"][int(np.argmax(np.abs(n)))]


def dominant_axis(iop: Sequence[float] | None) -> int | None:
    vals = _floats(iop, 6) if iop is not None else None
    if vals is None:
        return None
    n = np.cross(np.asarray(vals[:3]), np.asarray(vals[3:]))
    if not np.isfinite(n).all() or np.abs(n).max() < 1e-6:
        return None
    return int(np.argmax(np.abs(n)))


def infer_sequence(
    description: str,
    scan_options: str = "",
    scanning_sequence: str = "",
    te: float | None = None,
    tr: float | None = None,
    ti: float | None = None,
) -> tuple[str, bool]:
    """Return (weighting, fat_sat). Weighting in {pd, t2, stir, t1, other}."""
    desc = (description or "").lower()
    opts = (scan_options or "").upper()
    fat_sat = bool(_FS_RE.search(desc)) or "FS" in opts.split("\\") or "SFS" in opts.split("\\")
    is_ir = "IR" in scanning_sequence.upper().split("\\") if scanning_sequence else False
    if _STIR_RE.search(desc) or (is_ir and ti is not None and 80.0 <= ti <= 260.0):
        return "stir", True
    if _T1_RE.search(desc):
        return "t1", fat_sat
    if _PD_RE.search(desc):
        return "pd", fat_sat
    if _T2_RE.search(desc):
        return "t2", fat_sat
    if te is not None and tr is not None:
        if te < 30.0 and tr < 900.0:
            return "t1", fat_sat
        if te < 45.0:
            return "pd", fat_sat
        return "t2", fat_sat
    return "other", fat_sat


# ---------------------------------------------------------------------------------- data classes
@dataclass
class SeriesMeta:
    study_uid: str
    series_uid: str
    series_dir: str
    n_files: int
    n_slices: int
    plane: str
    weighting: str
    fat_sat: bool
    series_description: str = ""
    series_number: int | None = None
    rows: int | None = None
    cols: int | None = None
    pixel_spacing: float | None = None
    slice_thickness: float | None = None
    spacing_between: float | None = None
    manufacturer: str = ""
    model_name: str = ""
    software: str = ""
    field_strength: float | None = None
    imaging_frequency: float | None = None
    coil: str = ""
    tr: float | None = None
    te: float | None = None
    ti: float | None = None
    flip_angle: float | None = None
    scan_options: str = ""
    scanning_sequence: str = ""
    sequence_variant: str = ""
    mr_acq_type: str = ""
    photometric: str = ""
    is_multiframe: bool = False
    laterality: str = ""
    error: str = ""
    row_spacing: float | None = None   # mm between rows (PixelSpacing[0])
    col_spacing: float | None = None   # mm between columns (PixelSpacing[1])
    iop: str = ""                      # "rx|ry|rz|dx|dy|dz"
    center_x: float | None = None      # patient-x (LPS, +x = patient left) of the image centre, mm

    @property
    def fingerprint(self) -> str:
        freq = f"{self.imaging_frequency:.1f}" if self.imaging_frequency is not None else "na"
        parts = [self.manufacturer, self.model_name, self.software, freq, self.coil]
        key = "|".join(p.strip() for p in parts)
        return key if key.strip("|") else "unknown"

    @property
    def is_fluid_sensitive(self) -> bool:
        return self.weighting in FLUID_SENSITIVE

    @property
    def plane_id(self) -> int:
        return PLANE_INDEX.get(self.plane, PLANE_INDEX["unknown"])

    @property
    def seq_id(self) -> int:
        return seq_id(self.weighting, self.fat_sat)

    @property
    def looks_like_localizer(self) -> bool:
        return bool(_LOCALIZER_RE.search(self.series_description or ""))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        d["plane_id"] = self.plane_id
        d["seq_id"] = self.seq_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeriesMeta:
        fields = {k: d.get(k) for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        for k in ("fat_sat", "is_multiframe"):
            fields[k] = bool(fields.get(k))
        for k in ("n_files", "n_slices"):
            fields[k] = int(fields.get(k) or 0)
        for k in ("study_uid", "series_uid", "series_dir", "plane", "weighting"):
            fields[k] = "" if fields.get(k) is None else str(fields[k])
        fields["iop"] = "" if fields.get("iop") is None else str(fields["iop"])
        for k in ("row_spacing", "col_spacing", "center_x"):
            v = fields.get(k)
            fields[k] = None if v is None or (isinstance(v, float) and v != v) else float(v)
        for k in ("series_description", "manufacturer", "model_name", "software", "coil", "scan_options",
                  "scanning_sequence", "sequence_variant", "mr_acq_type", "photometric", "laterality", "error"):
            fields[k] = "" if fields.get(k) is None else str(fields[k])
        return cls(**fields)


@dataclass
class LoadedSeries:
    volume: np.ndarray  # (N, H, W) float32, rescaled, MONOCHROME1 inverted
    positions: np.ndarray  # (N,) float32 coordinate along the slice normal (LPS), or index
    meta: SeriesMeta


# ------------------------------------------------------------------------------- header scanning
def scan_series_header(series_dir: str, study_uid: str | None = None, series_uid: str | None = None) -> SeriesMeta:
    """Read ONE header per series (cheap) and derive plane / sequence / scanner fingerprint."""
    files = list_dicom_files(series_dir)
    if not files:
        err = "missing dir" if not os.path.isdir(series_dir) else "no files"
        return SeriesMeta(study_uid or "", series_uid or os.path.basename(series_dir), series_dir, 0, 0,
                          "unknown", "other", False, error=err)
    try:
        ds = read_header(files[len(files) // 2])
    except Exception as e:  # pragma: no cover - defensive
        return SeriesMeta(study_uid or "", series_uid or os.path.basename(series_dir), series_dir, len(files),
                          len(files), "unknown", "other", False, error=f"header: {e}")
    n_frames = int(_f(ds.get("NumberOfFrames"), 1) or 1)
    is_mf = n_frames > 1
    n_slices = n_frames if (len(files) == 1 and is_mf) else len(files)
    desc = _s(ds.get("SeriesDescription"))
    te, tr, ti = _f(ds.get("EchoTime")), _f(ds.get("RepetitionTime")), _f(ds.get("InversionTime"))
    weighting, fat_sat = infer_sequence(desc, _s(ds.get("ScanOptions")), _s(ds.get("ScanningSequence")), te, tr, ti)
    ps = _floats(ds.get("PixelSpacing"), 2) if ds.get("PixelSpacing") is not None else None
    iop = _floats(ds.get("ImageOrientationPatient"), 6) if ds.get("ImageOrientationPatient") is not None else None
    ipp = _floats(ds.get("ImagePositionPatient"), 3) if ds.get("ImagePositionPatient") is not None else None
    rows_n, cols_n = int(_f(ds.get("Rows"), 0) or 0), int(_f(ds.get("Columns"), 0) or 0)
    center_x = None
    if iop and ipp and ps and rows_n and cols_n:
        center_x = float(ipp[0] + iop[0] * ps[1] * cols_n / 2.0 + iop[3] * ps[0] * rows_n / 2.0)
    return SeriesMeta(
        study_uid=study_uid or _s(ds.get("StudyInstanceUID")),
        series_uid=series_uid or _s(ds.get("SeriesInstanceUID")) or os.path.basename(series_dir),
        series_dir=series_dir,
        n_files=len(files),
        n_slices=n_slices,
        plane=plane_from_iop(ds.get("ImageOrientationPatient")),
        weighting=weighting,
        fat_sat=fat_sat,
        series_description=desc,
        series_number=int(_f(ds.get("SeriesNumber"), 0) or 0),
        rows=int(_f(ds.get("Rows"), 0) or 0),
        cols=int(_f(ds.get("Columns"), 0) or 0),
        pixel_spacing=float(np.mean(ps)) if ps else None,
        slice_thickness=_f(ds.get("SliceThickness")),
        spacing_between=_f(ds.get("SpacingBetweenSlices")),
        manufacturer=_s(ds.get("Manufacturer")),
        model_name=_s(ds.get("ManufacturerModelName")),
        software=_s(ds.get("SoftwareVersions")),
        field_strength=_f(ds.get("MagneticFieldStrength")),
        imaging_frequency=_f(ds.get("ImagingFrequency")),
        coil=_s(ds.get("ReceiveCoilName")),
        tr=tr, te=te, ti=ti,
        flip_angle=_f(ds.get("FlipAngle")),
        scan_options=_s(ds.get("ScanOptions")),
        scanning_sequence=_s(ds.get("ScanningSequence")),
        sequence_variant=_s(ds.get("SequenceVariant")),
        mr_acq_type=_s(ds.get("MRAcquisitionType")),
        photometric=_s(ds.get("PhotometricInterpretation")),
        is_multiframe=is_mf,
        laterality=_s(ds.get("Laterality")),
        row_spacing=ps[0] if ps else None,
        col_spacing=ps[1] if ps else None,
        iop="|".join(f"{v:.5f}" for v in iop) if iop else "",
        center_x=center_x,
    )


# ----------------------------------------------------------------------------------- pixel loading
def _frame_positions_multiframe(ds: pydicom.Dataset, n: int, axis: int | None) -> list[float | None]:
    out: list[float | None] = [None] * n
    seq = ds.get("PerFrameFunctionalGroupsSequence")
    if seq is None or axis is None:
        return out
    for k in range(min(n, len(seq))):
        try:
            ipp = seq[k].PlanePositionSequence[0].ImagePositionPatient
            out[k] = float(ipp[axis])
        except Exception:
            pass
    return out


def load_series(series_dir: str, meta: SeriesMeta | None = None, files: list[str] | None = None) -> LoadedSeries:
    """Decode every slice of a series, rescale, invert MONOCHROME1, sort in patient coordinates."""
    files = files or list_dicom_files(series_dir)
    if meta is None:
        meta = scan_series_header(series_dir)
    frames: list[dict[str, Any]] = []
    axis: int | None = None
    for idx, path in enumerate(files):
        try:
            ds = pydicom.dcmread(path, force=True)
            arr = ds.pixel_array
        except Exception as e:
            log.warning("skip %s: %s", path, e)
            continue
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4) and int(_f(ds.get("NumberOfFrames"), 1) or 1) == 1:
            arr = arr[..., :3].mean(axis=-1)  # RGB -> gray (never expected for MR)
        arr = arr.astype(np.float32)
        slope, intercept = _f(ds.get("RescaleSlope"), 1.0), _f(ds.get("RescaleIntercept"), 0.0)
        if slope != 1.0 or intercept != 0.0:
            arr = arr * slope + intercept
        if _s(ds.get("PhotometricInterpretation")).upper() == "MONOCHROME1":
            arr = -arr
        if axis is None:
            axis = dominant_axis(ds.get("ImageOrientationPatient"))
        inst = _f(ds.get("InstanceNumber"))
        te = _f(ds.get("EchoTime"))
        if arr.ndim == 3:  # multi-frame
            pos = _frame_positions_multiframe(ds, arr.shape[0], axis)
            for k in range(arr.shape[0]):
                frames.append({"img": arr[k], "pos": pos[k], "inst": (inst or 0) * 10000 + k, "idx": idx * 10000 + k, "te": te})
        else:
            ipp = _floats(ds.get("ImagePositionPatient"), 3) if ds.get("ImagePositionPatient") is not None else None
            pos = float(ipp[axis]) if (ipp is not None and axis is not None) else None
            frames.append({"img": arr, "pos": pos, "inst": inst, "idx": idx, "te": te})
    if not frames:
        raise RuntimeError(f"no decodable slices in {series_dir}")

    # dual-echo series (PD + T2 in one series): keep the echo with most frames, tie -> shorter TE
    tes = [fr["te"] for fr in frames if fr["te"] is not None]
    if tes and len(set(tes)) > 1:
        cnt = Counter(tes)
        keep_te = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        frames = [fr for fr in frames if fr["te"] == keep_te]

    if all(fr["pos"] is not None for fr in frames):
        frames.sort(key=lambda fr: (fr["pos"], fr["inst"] if fr["inst"] is not None else 0, fr["idx"]))
        positions = np.asarray([fr["pos"] for fr in frames], dtype=np.float32)
    elif all(fr["inst"] is not None for fr in frames):
        frames.sort(key=lambda fr: (fr["inst"], fr["idx"]))
        positions = np.arange(len(frames), dtype=np.float32)
    else:
        frames.sort(key=lambda fr: fr["idx"])
        positions = np.arange(len(frames), dtype=np.float32)

    shapes = Counter(fr["img"].shape for fr in frames)
    target_shape = shapes.most_common(1)[0][0]
    imgs = []
    for fr in frames:
        img = fr["img"]
        if img.shape != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        imgs.append(img)
    volume = np.stack(imgs, axis=0).astype(np.float32)
    meta.n_slices = int(volume.shape[0])
    return LoadedSeries(volume=volume, positions=positions, meta=meta)


# ------------------------------------------------------------------------------------ normalisation
def normalize_volume(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5, max_samples: int = 2_000_000) -> np.ndarray:
    """Per-series percentile window -> uint8. MRI intensities have no absolute meaning."""
    flat = vol.reshape(-1)
    if flat.size > max_samples:
        rng = np.random.default_rng(0)
        flat = flat[rng.integers(0, flat.size, max_samples)]
    lo, hi = np.percentile(flat, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = float(np.nanmin(vol)), float(np.nanmax(vol))
        if hi - lo < 1e-6:
            return np.zeros(vol.shape, dtype=np.uint8)
    out = (vol - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0 + 0.5).astype(np.uint8)


def resize_volume(vol: np.ndarray, size: int, mode: str = "pad") -> np.ndarray:
    """Resize (N,H,W) uint8 -> (N,size,size). ``pad`` keeps aspect ratio (letterbox), ``stretch`` does not."""
    n, h, w = vol.shape
    if h == size and w == size:
        return vol
    if mode == "stretch":
        nh, nw, top, left = size, size, 0, 0
    else:
        scale = size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        top, left = (size - nh) // 2, (size - nw) // 2
    interp = cv2.INTER_AREA if (nh < h or nw < w) else cv2.INTER_LINEAR
    out = np.zeros((n, size, size), dtype=vol.dtype)
    for i in range(n):
        out[i, top:top + nh, left:left + nw] = cv2.resize(vol[i], (nw, nh), interpolation=interp)
    return out


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    p = positions.astype(np.float32)
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 1e-6:
        return np.linspace(0.0, 1.0, len(p), dtype=np.float32) if len(p) > 1 else np.zeros_like(p)
    return (p - lo) / (hi - lo)


_WANT = {"sagittal": (1, 1.0, 2, -1.0), "coronal": (0, 1.0, 2, -1.0), "axial": (0, 1.0, 1, 1.0)}


def canonical_ops(iop: Sequence[float] | None, plane: str) -> tuple[bool, bool, bool]:
    """(transpose, flip_rows, flip_cols) that bring an image to the canonical display of its plane.

    Canonical: columns run to patient-left (+x) on coronal/axial and to posterior (+y) on sagittal;
    rows run to inferior (-z) on sagittal/coronal and to posterior (+y) on axial. Oblique or unknown -> no-op.
    """
    vals = _floats(iop, 6) if iop is not None else None
    if vals is None or plane not in _WANT:
        return False, False, False
    r, d = np.asarray(vals[:3]), np.asarray(vals[3:])
    col_axis, col_sign, row_axis, row_sign = _WANT[plane]
    r_ax, d_ax = int(np.argmax(np.abs(r))), int(np.argmax(np.abs(d)))
    transpose = r_ax == row_axis and d_ax == col_axis
    if transpose:
        r, d = d, r
    elif not (r_ax == col_axis and d_ax == row_axis):
        return False, False, False
    return transpose, bool(np.sign(d[row_axis]) != row_sign), bool(np.sign(r[col_axis]) != col_sign)


def knee_side(center_xs: Sequence[float | None], min_offset_mm: float = 20.0) -> str:
    """'right' / 'left' / 'unknown' from the patient-x of image centres (+x points to the patient's left)."""
    xs = [float(x) for x in center_xs if x is not None and np.isfinite(x)]
    if not xs:
        return "unknown"
    m = float(np.median(xs))
    return "right" if m < -min_offset_mm else ("left" if m > min_offset_mm else "unknown")


def crop_center_mm(vol: np.ndarray, row_spacing: float | None, col_spacing: float | None, crop_mm: float | None) -> np.ndarray:
    """Centre crop (or zero-pad) every slice to a crop_mm x crop_mm physical window."""
    if not crop_mm or not row_spacing or not col_spacing or row_spacing <= 0 or col_spacing <= 0:
        return vol
    n, h, w = vol.shape
    th, tw = max(8, int(round(crop_mm / row_spacing))), max(8, int(round(crop_mm / col_spacing)))
    out = np.zeros((n, th, tw), dtype=vol.dtype)
    sh, sw = min(h, th), min(w, tw)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    Y0, X0 = (th - sh) // 2, (tw - sw) // 2
    out[:, Y0:Y0 + sh, X0:X0 + sw] = vol[:, y0:y0 + sh, x0:x0 + sw]
    return out


def smart_shift_mm(img: np.ndarray, depth: float, cache_mm: float, view_mm: float, central: float = 0.12, ramp: float = 0.38,
                   deadband_mm: float = 2.0, thr: int = 24) -> tuple[float, float]:
    """(dy, dx) in mm by which the view window should follow the anatomy on this slice.

    The middle of a knee stack sits at the frame centre, so a central band of the stack is never shifted. Towards the
    ends the cross-section shrinks and drifts (wide-FOV sagittal / coronal, long axial stacks, per-site positioning
    habits): there the window follows the foreground centroid, ramped in with the distance from mid-stack and bounded
    by the margin the wide cache leaves around the view. Idea from the public smart-crop recipe; own implementation.
    """
    margin = (cache_mm - view_mm) / 2.0
    a = abs(float(depth) - 0.5)
    if margin <= 0 or a <= central:
        return 0.0, 0.0
    mask = img > thr
    if mask.sum() < 0.02 * mask.size:
        return 0.0, 0.0
    ys, xs = np.nonzero(mask)
    h, w = img.shape
    gain = min(1.0, (a - central) / ramp)
    out = []
    for c, n in ((ys.mean(), h), (xs.mean(), w)):
        mm = gain * (c - (n - 1) / 2.0) * cache_mm / n
        out.append(0.0 if abs(mm) < deadband_mm else float(np.clip(mm, -margin, margin)))
    return out[0], out[1]


def late_crop(img: np.ndarray, cache_mm: float, view_mm: float, shift_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """view_mm x view_mm window out of a cache_mm x cache_mm slice (H, W), optionally shifted; never leaves the frame."""
    h, w = img.shape
    th, tw = max(8, int(round(h * view_mm / cache_mm))), max(8, int(round(w * view_mm / cache_mm)))
    y0 = int(round((h - th) / 2.0 + shift_mm[0] * h / cache_mm))
    x0 = int(round((w - tw) / 2.0 + shift_mm[1] * w / cache_mm))
    y0, x0 = min(max(y0, 0), h - th), min(max(x0, 0), w - tw)
    return img[y0:y0 + th, x0:x0 + tw]


def preprocess_series(
    series_dir: str,
    size: int = 384,
    resize_mode: str = "pad",
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
    meta: SeriesMeta | None = None,
    max_slices: int | None = None,
    crop_mm: float | None = None,
    canonical: bool = False,
    mirror: bool = False,
) -> tuple[np.ndarray, np.ndarray, SeriesMeta]:
    """DICOM dir -> (uint8 volume (N,size,size), normalised positions (N,), meta).

    The single source of truth for pixel preprocessing: used by the cache builder AND inference.
    ``canonical`` fixes display orientation from ImageOrientationPatient, ``crop_mm`` keeps a physical window
    around the image centre (the joint), ``mirror`` maps a right knee onto a left one (horizontal mirror on
    coronal/axial, reversed slice order on sagittal) so medial/lateral findings share one image side.
    """
    loaded = load_series(series_dir, meta=meta)
    m = loaded.meta
    vol = normalize_volume(loaded.volume, lo_pct, hi_pct)
    pos = normalize_positions(loaded.positions)
    rs, cs = m.row_spacing, m.col_spacing
    if canonical and m.iop:
        tr, fr, fc = canonical_ops([float(v) for v in m.iop.split("|")], m.plane)
        if tr:
            vol, rs, cs = vol.transpose(0, 2, 1), cs, rs
        if fr:
            vol = vol[:, ::-1, :]
        if fc:
            vol = vol[:, :, ::-1]
    if mirror:
        if m.plane == "sagittal":
            vol, pos = vol[::-1], (1.0 - pos)[::-1]
        else:
            vol = vol[:, :, ::-1]
    vol = np.ascontiguousarray(vol)
    if max_slices is not None and vol.shape[0] > max_slices:
        idx = np.linspace(0, vol.shape[0] - 1, max_slices).round().astype(int)
        vol, pos = vol[idx], pos[idx]
    vol = crop_center_mm(vol, rs, cs, crop_mm)
    vol = resize_volume(vol, size, resize_mode)
    return vol, np.ascontiguousarray(pos), m
