from __future__ import annotations

import os

import numpy as np
import pytest

from rsna_knee.dicom import (
    infer_sequence,
    load_series,
    normalize_volume,
    plane_from_iop,
    preprocess_series,
    resize_volume,
    scan_series_header,
)
from tests.conftest import write_series


def test_plane_from_iop():
    assert plane_from_iop([0, 1, 0, 0, 0, -1]) == "sagittal"
    assert plane_from_iop([1, 0, 0, 0, 0, -1]) == "coronal"
    assert plane_from_iop([1, 0, 0, 0, 1, 0]) == "axial"
    assert plane_from_iop(None) == "unknown"


@pytest.mark.parametrize("desc,opts,te,tr,ti,exp", [
    ("SAG PD FS", "", 30, 3000, None, ("pd", True)),
    ("COR T1", "", 12, 600, None, ("t1", False)),
    ("AX T2 FS", "", 80, 4000, None, ("t2", True)),
    ("sag STIR", "", 40, 4000, 150, ("stir", True)),
    ("Sag DP FatSat", "", 35, 2500, None, ("pd", True)),
    ("t2_tse_tra_fs", "", 90, 4000, None, ("t2", True)),
    ("", "FS", 35, 3000, None, ("pd", True)),
    ("", "", 12, 500, None, ("t1", False)),
    ("", "", 100, 4000, None, ("t2", False)),
    ("ax pd tse", "", 35, 3000, None, ("pd", False)),
])
def test_infer_sequence(desc, opts, te, tr, ti, exp):
    assert infer_sequence(desc, opts, "SE", te, tr, ti) == exp


def test_load_series_sorted_in_patient_coords(tmp_path):
    d = str(tmp_path / "s1")
    write_series(d, "1.2.3", "1.2.4", n=7, plane="sagittal")
    ls = load_series(d)
    assert ls.volume.shape == (7, 64, 48)
    rows = ls.volume[:, 0, 0]
    assert np.allclose(rows, np.arange(7) * 10)  # sorted by ImagePositionPatient x, not by name / InstanceNumber
    assert np.all(np.diff(ls.positions) > 0)


def test_monochrome1_and_rescale(tmp_path):
    d = str(tmp_path / "s2")
    write_series(d, "1.2.3", "1.2.4", n=4, plane="coronal", monochrome1=True, rescale=(2.0, -5.0))
    ls = load_series(d)
    # inverted (4000 - img) then negated -> increasing with i
    assert ls.volume[3, 0, 0] > ls.volume[0, 0, 0]
    u8 = normalize_volume(ls.volume)
    assert u8.dtype == np.uint8 and u8.max() == 255 and u8.min() == 0


def test_scan_header_and_fingerprint(tmp_path):
    d = str(tmp_path / "s3")
    write_series(d, "1.2.3", "1.2.4", n=5, plane="axial", desc="AX T2 FS")
    m = scan_series_header(d, "1.2.3", "1.2.4")
    assert (m.plane, m.weighting, m.fat_sat, m.n_slices) == ("axial", "t2", True, 5)
    assert m.fingerprint.startswith("SIEMENS|Skyra|")
    assert m.seq_id == 3  # t2 index 1 * 2 + fat_sat


def test_preprocess_resize_pad(tmp_path):
    d = str(tmp_path / "s4")
    write_series(d, "1.2.3", "1.2.4", n=6, plane="sagittal")
    vol, pos, meta = preprocess_series(d, size=32, resize_mode="pad", max_slices=4)
    assert vol.shape == (4, 32, 32) and vol.dtype == np.uint8
    assert pos[0] == 0.0 and pos[-1] == 1.0
    # letterbox: 64x48 -> 32x24 centred -> columns outside [4,28) are zero
    assert vol[:, :, :4].max() == 0 and vol[:, :, 28:].max() == 0
    assert resize_volume(vol, 32).shape == (4, 32, 32)


def test_jpeg2000_roundtrip(tmp_path):
    """Kaggle files use JPEG-Lossless / JPEG2000; make sure this environment can decode them."""
    import pydicom
    from pydicom.uid import JPEG2000Lossless

    d = str(tmp_path / "s5")
    write_series(d, "1.2.3", "1.2.4", n=2, plane="sagittal", shuffle_names=False)
    src = os.path.join(d, "0000.dcm")
    ds = pydicom.dcmread(src)
    try:
        ds.compress(JPEG2000Lossless)
    except Exception as e:  # encoder plugin missing - decoding is what matters, but we cannot test it here
        pytest.skip(f"no JPEG2000 encoder available: {e}")
    ds.save_as(src)
    ls = load_series(d)
    assert ls.volume.shape[0] == 2 and ls.volume[0, 0, 0] == 0


def test_canonical_ops_and_knee_side():
    from rsna_knee.dicom import canonical_ops, crop_center_mm, knee_side

    assert canonical_ops([1, 0, 0, 0, 0, -1], "coronal") == (False, False, False)      # already canonical
    assert canonical_ops([-1, 0, 0, 0, 0, -1], "coronal") == (False, False, True)      # mirrored display
    assert canonical_ops([1, 0, 0, 0, 0, 1], "coronal") == (False, True, False)        # upside down
    assert canonical_ops([0, 0, -1, 1, 0, 0], "coronal") == (True, False, False)       # transposed
    assert canonical_ops([0, 1, 0, 0, 0, -1], "sagittal") == (False, False, False)
    assert canonical_ops([0.7, 0.7, 0, 0, 0, -1], "unknown") == (False, False, False)
    assert knee_side([-95.0, -90.0, -100.0]) == "right" and knee_side([80.0, 85.0]) == "left"
    assert knee_side([5.0, -3.0]) == "unknown" and knee_side([None]) == "unknown"
    vol = np.arange(2 * 100 * 80, dtype=np.float32).reshape(2, 100, 80).astype(np.uint8)
    out = crop_center_mm(vol, 0.5, 0.5, 30.0)   # 60 x 60 px window
    assert out.shape == (2, 60, 60) and (out == vol[:, 20:80, 10:70]).all()
    assert crop_center_mm(vol, 0.5, 0.5, 60.0).shape == (2, 120, 120)  # larger than the image -> zero padded


def test_preprocess_mirror_and_crop(tmp_path):
    d = str(tmp_path / "s6")
    write_series(d, "1.2.3", "1.2.4", n=6, plane="sagittal", shape=(64, 64))
    v0, p0, m = preprocess_series(d, size=32, canonical=True)
    v1, p1, _ = preprocess_series(d, size=32, canonical=True, mirror=True)
    assert m.row_spacing == 0.4 and m.center_x is not None and m.iop
    assert (v1 == v0[::-1]).all() and np.allclose(p1, p0)          # sagittal: slice order reversed, positions re-normalised
    d2 = str(tmp_path / "s7")
    write_series(d2, "1.2.3", "1.2.5", n=4, plane="coronal", shape=(64, 64))
    c0, _, _ = preprocess_series(d2, size=32, canonical=True)
    c1, _, _ = preprocess_series(d2, size=32, canonical=True, mirror=True)
    assert (c1 == c0[:, :, ::-1]).all()                              # coronal: horizontal mirror
    z, _, _ = preprocess_series(d2, size=32, crop_mm=12.8)           # 12.8 mm / 0.4 mm = 32 px centre window, no resize
    full = preprocess_series(d2, size=64)[0]
    assert (z == full[:, 16:48, 16:48]).all()


def test_slice_offset_tta_grid():
    from rsna_knee.data.dataset import sample_centers
    rng = np.random.default_rng(0)
    base = sample_centers(40, 8, False, rng)
    shifted = sample_centers(40, 8, False, rng, offset=0.5)
    assert len(shifted) == 8 and np.all(np.diff(shifted) >= 0) and shifted.max() <= 39
    assert not np.array_equal(base, shifted) and np.abs(shifted - base).max() <= 3
    assert np.array_equal(sample_centers(40, 8, True, rng, offset=0.5).shape, base.shape)


def test_late_crop_and_smart_shift():
    import numpy as np

    from rsna_knee.dicom import late_crop, smart_shift_mm

    img = np.zeros((360, 360), np.uint8)  # 180 mm frame, 0.5 mm per pixel
    img[40:140, 200:300] = 200            # anatomy up and to the right of the centre
    c = late_crop(img, 180.0, 140.0)
    assert c.shape == (280, 280) and c[0, 0] == img[40, 40]
    assert late_crop(img, 180.0, 140.0, (-500.0, 500.0)).shape == (280, 280)  # shifts are clamped to the frame
    assert smart_shift_mm(img, 0.5, 180.0, 140.0) == (0.0, 0.0)                # protected central band of the stack
    dy, dx = smart_shift_mm(img, 0.0, 180.0, 140.0)
    assert -20.0 <= dy < 0 < dx <= 20.0                                        # follows the anatomy, bounded by the margin
    assert smart_shift_mm(np.zeros((360, 360), np.uint8), 0.0, 180.0, 140.0) == (0.0, 0.0)
    assert smart_shift_mm(img, 0.0, 140.0, 140.0) == (0.0, 0.0)               # no margin, no shift
