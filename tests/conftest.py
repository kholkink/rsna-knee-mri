"""Synthetic DICOM knee studies for unit / integration tests (no real data needed)."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

IOP = {"sagittal": [0, 1, 0, 0, 0, -1], "coronal": [1, 0, 0, 0, 0, -1], "axial": [1, 0, 0, 0, 1, 0]}
AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def make_slice(i: int, shape=(64, 48)) -> np.ndarray:
    """Pixel content encodes the slice index (row 0 = i*10) so ordering can be asserted."""
    h, w = shape
    img = np.random.default_rng(i).integers(0, 200, size=(h, w)).astype(np.int16)
    img[0, :] = i * 10
    return img


def write_series(series_dir: str, study_uid: str, series_uid: str, n: int = 10, plane: str = "sagittal",
                 desc: str = "SAG PD FS", te: float = 30.0, tr: float = 3000.0, shape=(64, 48),
                 monochrome1: bool = False, rescale: tuple[float, float] | None = None, manufacturer: str = "SIEMENS",
                 model: str = "Skyra", shuffle_names: bool = True) -> None:
    os.makedirs(series_dir, exist_ok=True)
    order = list(range(n))
    names = [f"{random.getrandbits(32):08x}.dcm" for _ in order] if shuffle_names else [f"{i:04d}.dcm" for i in order]
    for i, name in zip(order, names):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        ds.SeriesDescription = desc
        ds.SeriesNumber = 3
        ds.InstanceNumber = n - i  # deliberately opposite to spatial order
        ds.ImageOrientationPatient = [float(v) for v in IOP[plane]]
        ipp = [10.0, 20.0, 30.0]
        ipp[AXIS[plane]] = float(i) * 3.5 - 20.0
        ds.ImagePositionPatient = ipp
        ds.PixelSpacing = [0.4, 0.4]
        ds.SliceThickness = 3.0
        ds.SpacingBetweenSlices = 3.5
        ds.Manufacturer = manufacturer
        ds.ManufacturerModelName = model
        ds.SoftwareVersions = "syngo MR E11"
        ds.ImagingFrequency = 123.2
        ds.ReceiveCoilName = "Knee15"
        ds.MagneticFieldStrength = 3.0
        ds.EchoTime = te
        ds.RepetitionTime = tr
        ds.FlipAngle = 150
        ds.ScanningSequence = "SE"
        ds.SequenceVariant = "SK"
        ds.ScanOptions = "FS" if "FS" in desc else ""
        ds.MRAcquisitionType = "2D"
        img = make_slice(i, shape)
        if monochrome1:
            img = 4000 - img
        ds.Rows, ds.Columns = img.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1
        if rescale:
            ds.RescaleSlope, ds.RescaleIntercept = rescale
        ds.PixelData = img.astype(np.int16).tobytes()
        ds.save_as(os.path.join(series_dir, name), enforce_file_format=True)


def write_study(root: str, study_uid: str, planes=("sagittal", "coronal", "axial"), n: int = 10, **kw) -> list[str]:
    descs = {"sagittal": "SAG PD FS", "coronal": "COR T1", "axial": "AX T2 FS"}
    series = []
    for p in planes:
        suid = generate_uid()
        write_series(os.path.join(root, study_uid, suid), study_uid, suid, n=n, plane=p, desc=descs[p], **kw)
        series.append(suid)
    # one localizer that must be ignored
    suid = generate_uid()
    write_series(os.path.join(root, study_uid, suid), study_uid, suid, n=3, plane="axial", desc="localizer", **kw)
    series.append(suid)
    return series


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("dicom")
    random.seed(0)
    studies = {}
    manufacturers = [("SIEMENS", "Skyra"), ("GE", "Signa"), ("Philips", "Ingenia")]
    for k in range(6):
        uid = f"1.2.826.0.1.{k}"
        man, mod = manufacturers[k % 3]
        studies[uid] = write_study(str(root), uid, n=8 + k, manufacturer=man, model=mod)
    return str(root), studies
