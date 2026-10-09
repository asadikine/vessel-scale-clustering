import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pydicom = pytest.importorskip("pydicom")

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "ircadb" / "ircadb_to_nifti.py"
spec = importlib.util.spec_from_file_location("ircadb_to_nifti", SCRIPT)
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

SPACING = (0.7, 0.7, 1.6)        # column spacing, row spacing, slice step (mm)
ORIGIN = np.array([-180.0, -170.0, 40.0])   # LPS position of the first voxel


def _write_series(folder: Path, volume: np.ndarray, rescale=None):
    """Write volume[x, y, z] as one DICOM file per slice, in reverse file order."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder.mkdir(parents=True)
    nz = volume.shape[2]
    for k in range(nz):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.Rows, ds.Columns = volume.shape[1], volume.shape[0]
        ds.PixelSpacing = [SPACING[1], SPACING[0]]
        ds.SliceThickness = SPACING[2]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = list(ORIGIN + [0, 0, k * SPACING[2]])
        ds.InstanceNumber = k + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if rescale else 0
        plane = volume[:, :, k].T                     # rows = y, columns = x
        if rescale:
            ds.RescaleSlope, ds.RescaleIntercept = 1, rescale
            ds.PixelData = (plane - rescale).astype(np.int16).tobytes()
        else:
            ds.PixelData = plane.astype(np.uint16).tobytes()
        # file names in reverse order of position, to check the sorting
        target = folder / f"image_{nz - 1 - k}"
        try:                                           # pydicom >= 3
            ds.save_as(target, enforce_file_format=True)
        except TypeError:                              # pydicom 2.x (Python 3.9)
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(target, write_like_original=False)


@pytest.fixture
def patient(tmp_path):
    from vessel_scales.synthetic import make_synthetic_tree
    tree, radius = make_synthetic_tree(shape=(64, 72, 40),
                                       segments=[((5, 36, 20), (30, 36, 20), 4.0),
                                                 ((30, 36, 20), (58, 15, 10), 2.5),
                                                 ((30, 36, 20), (58, 60, 30), 2.0)])
    portal = tree & (radius >= 2.5)
    venous = tree & (radius < 2.5)
    root = tmp_path / "3Dircadb1.99"
    masks = root / "MASKS_DICOM"
    _write_series(masks / "portalvein", portal * 255)
    _write_series(masks / "venoussystem", venous * 255)
    _write_series(masks / "liver", np.ones_like(tree) * 255)
    ct = np.where(tree, 200, -100).astype(np.int16)
    _write_series(root / "PATIENT_DICOM", ct, rescale=-1024)
    return root, portal, venous, ct


def _check_geometry(img):
    expected = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])
    expected[:3, 3] = [-ORIGIN[0], -ORIGIN[1], ORIGIN[2]]
    np.testing.assert_allclose(img.affine, expected, atol=1e-4)
    assert nib.aff2axcodes(img.affine) == ("L", "P", "S")


def test_vessels_are_merged_with_dicom_geometry(patient, tmp_path):
    root, portal, venous, ct = patient
    out = tmp_path / "vessels.nii.gz"
    conv.main([str(root), str(out), "--ct", str(tmp_path / "ct.nii.gz")])
    img = nib.load(str(out))
    data = np.asanyarray(img.dataobj)
    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, (portal | venous).astype(np.uint8))
    _check_geometry(img)
    ct_img = nib.load(str(tmp_path / "ct.nii.gz"))
    np.testing.assert_array_equal(np.asanyarray(ct_img.dataobj), ct)
    _check_geometry(ct_img)


def test_zipped_masks_and_structure_choice(patient, tmp_path):
    root, portal, _, _ = patient
    shutil.make_archive(str(root / "MASKS_DICOM"), "zip", root, "MASKS_DICOM")
    shutil.rmtree(root / "MASKS_DICOM")
    out = tmp_path / "portal.nii.gz"
    conv.main([str(root), str(out), "--structures", "PortalVein"])
    np.testing.assert_array_equal(np.asanyarray(nib.load(str(out)).dataobj), portal)


def test_unknown_structure(patient, tmp_path):
    root, *_ = patient
    with pytest.raises(SystemExit):
        conv.main([str(root), str(tmp_path / "x.nii.gz"), "--structures", "aorta"])
