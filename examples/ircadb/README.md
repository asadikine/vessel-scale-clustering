# Real example: hepatic vessels from 3D-IRCADb-01

[3D-IRCADb-01](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/)
contains 20 contrast-enhanced abdominal CT scans with manual segmentations of
the liver and its vessels, stored as DICOM series. Its licence
(CC BY-NC-ND 4.0) does not allow redistributing modified data, so no IRCADb
file is included in this repository: download the patients you need from the
IRCAD page.

## 1. Download a patient

On the dataset page, download one patient (for example `3Dircadb1.12`) or the
whole dataset, and unzip it. Each patient folder contains `PATIENT_DICOM` (the
CT), `MASKS_DICOM` (one DICOM series per segmented structure), `LABELLED_DICOM`
and `MESHES_VTK`. `MASKS_DICOM` may itself be a zip file; the converter reads
both.

## 2. Convert the vessel masks to NIfTI

```bash
pip install -e ".[nifti,ircadb]"          # adds pydicom

python examples/ircadb/ircadb_to_nifti.py 3Dircadb1.12 --list
python examples/ircadb/ircadb_to_nifti.py 3Dircadb1.12 ircadb12_vessels.nii.gz \
    --ct ircadb12_ct.nii.gz
```

The first command lists the structures of the patient. The second merges the
hepatic vessels, portal vein and venous system (hepatic veins), into one
binary mask, and saves the CT next to it for viewing. Choose other structures
with `--structures`, for example `--structures portalvein` for the portal vein
only. Voxel size, orientation and position are taken from the DICOM headers.

## 3. Run the clustering

```bash
vessel-scales ircadb12_vessels.nii.gz -o results/ --figure
```

`results/` then contains the scale map, one mask per scale, the branch labels,
a CSV of branch radii, a JSON summary and a 3D render. To check the result on
the CT in ITK-SNAP:

```bash
itksnap -g ircadb12_ct.nii.gz -s results/ircadb12_vessels_scales.nii.gz \
    -l results/ircadb12_vessels_itksnap_labels.txt
```
