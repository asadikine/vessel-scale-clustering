# Real example: hepatic vessels from a public CT dataset

This example runs the clustering on a liver vessel mask from the
**Medical Segmentation Decathlon, Task08 Hepatic Vessel** (portal-venous CT,
443 cases, vessel and tumour labels).

## 1. Get a mask

Two options, both public:

- **Original MSD labels** (`Task08_HepaticVessel/labelsTr/hepaticvessel_XXX.nii.gz`),
  licence CC BY-SA 4.0. Download Task08 from
  [medicaldecathlon.com](http://medicaldecathlon.com) or the AWS bucket
  (`aws s3 ls --no-sign-request s3://msd-for-monai/`). In these files
  label 1 is vessel and label 2 is tumour.
- **Refined masks** from the *Refined 3D Hepatic Vessel Skeleton Dataset*
  (Nodar-Corral et al., Universidade de Santiago de Compostela, Zenodo 2025,
  CC BY 4.0): 11 MSD Task08 masks cleaned in 3D Slicer (vessels outside the
  liver, inferior vena cava and aorta removed). Binary masks: 1 is vessel.

## 2. Run the clustering

```bash
# original MSD label file: keep label 1 (vessels), ignore label 2 (tumour)
vessel-scales hepaticvessel_001.nii.gz --labels 1 -o results/ --figure

# refined binary mask
vessel-scales hepaticvessel_001.nii.gz -o results/ --figure
```

The voxel size is read from the NIfTI header, so radii and thresholds are in mm.
`results/` then contains the scale map, one mask per scale (same geometry and
header as the input), a CSV of branch radii, a JSON summary and a figure.
Open `hepaticvessel_001.nii.gz` and `results/hepaticvessel_001_scales.nii.gz`
together in 3D Slicer or ITK-SNAP to check the overlay.

## Citation of the data

- A. L. Simpson et al., *A large annotated medical image dataset for the
  development and evaluation of segmentation algorithms*, arXiv:1902.09063, 2019.
- M. Antonelli et al., *The Medical Segmentation Decathlon*, Nature
  Communications 13, 4128, 2022.
- For the refined masks: L. Nodar-Corral et al., *Refined 3D Hepatic Vessel
  Skeleton Dataset from the Medical Segmentation Decathlon (Task08_HepaticVessel)*,
  Zenodo, 2025, doi:10.5281/zenodo.15729285.
