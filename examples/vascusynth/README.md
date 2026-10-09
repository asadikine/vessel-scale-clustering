# VascuSynth example: simulated vascular trees

[VascuSynth](https://github.com/sfu-mial/VascuSynth) (Hamarneh & Jassi, 2010)
grows vascular trees from an oxygen demand map and draws them as 3D volumes.
`examples/data/vascusynth_tree.nii.gz` is one such tree (101³ voxels,
56 bifurcations), binarised with `vascusynth_to_nifti.py`, and
`examples/data/vascusynth_tree_structure.mat` is its tree, with the true
radius of every segment.

```bash
vessel-scales examples/data/vascusynth_tree.nii.gz -o results/ --figure
python examples/vascusynth/compare_ground_truth.py \
    examples/data/vascusynth_tree.nii.gz examples/data/vascusynth_tree_structure.mat
```

## Make your own trees

### 1. Build VascuSynth on macOS

VascuSynth is C++ and needs ITK and CMake. These steps follow the upstream
Ubuntu instructions, adapted to Homebrew; they have not been tested on every
macOS version.

```bash
xcode-select --install                 # compilers, once
brew install cmake itk

git clone https://github.com/sfu-mial/VascuSynth.git
cd VascuSynth
sed -i '' '/SET (ITK_DIR/d' CMakeLists.txt     # remove the hard-coded ITK path

ITK_DIR=$(ls -d "$(brew --prefix)"/lib/cmake/ITK-* | head -1)
cmake -B build -DITK_DIR="$ITK_DIR" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j
cp build/VascuSynth .                  # replaces the Linux binary shipped in the repo
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is needed with CMake 4 because the
project declares `cmake_minimum_required(VERSION 3.0)`. If Homebrew's ITK 5
does not compile, build ITK 4.13.3 from source as in the upstream README and
pass its build folder as `ITK_DIR`.

### 2. Fix the example parameter files

The parameter files shipped with VascuSynth (`p1.txt` … `p5.txt`) write
`ROW:` and `MEW:`, but the parser reads `RHO:` and `MU:`; with the shipped
files VascuSynth stops with *not all parameters have been defined*.

```bash
sed -i '' 's/^ROW:/RHO:/; s/^MEW:/MU:/' p*.txt
```

### 3. Generate two or three trees

Run from the VascuSynth folder. Each tree differs by its random seed and its
number of terminal nodes.

```bash
for i in 1 2 3; do
  sed -e "s/^RANDOM_SEED:.*/RANDOM_SEED: $i/" \
      -e "s/^NUM_NODES:.*/NUM_NODES: $((100 * i + 100))/" p1.txt > tree$i.txt
done
printf "tree1.txt\ntree2.txt\ntree3.txt\n" > myParams.txt
printf "tree1\ntree2\ntree3\n" > myNames.txt

./VascuSynth myParams.txt myNames.txt 0.02
```

The last argument is the voxel width of the drawn image. The parameter files
set `VOXEL_WIDTH: 0.04` for a 100-unit map, so:

| Last argument | Volume | Vessel radii |
|---|---|---|
| `0.04` | 101³ | as drawn (thinnest vessels ≈ 1 voxel) |
| `0.02` | 201³ | twice as many voxels per radius; drawing takes longer |

Each `treeN/` folder then holds `original_image/image000.jpg …` (one slice
per file) and `tree_structure.xml` (the tree, with the radius of every
segment).

### 4. Convert to NIfTI and run

```bash
for i in 1 2 3; do
  python examples/vascusynth/vascusynth_to_nifti.py \
      /path/to/VascuSynth/tree$i/original_image tree$i.nii.gz --threshold 15
done
vessel-scales tree1.nii.gz tree2.nii.gz tree3.nii.gz -o results/ --figure
```

VascuSynth draws with partial-volume anti-aliasing: each voxel holds
0, 31, 63, … or 223 depending on how many of its 7 sub-samples lie inside a vessel.
`vascusynth_to_nifti.py` keeps the voxels above `--threshold` as vessel.
JPEG slices carry compression noise in the background, hence
`--threshold 15`; a lossless NIfTI volume can use the default `0`.

### 5. Compare with the ground truth

```bash
python examples/vascusynth/compare_ground_truth.py tree1.nii.gz \
    /path/to/VascuSynth/tree1/tree_structure.xml --image-voxel-width 0.02
```

`compare_ground_truth.py` reads either the `tree_structure.xml` written by
VascuSynth or a `.mat` tree structure. `--image-voxel-width` is the last
argument given to VascuSynth, when it differs from `VOXEL_WIDTH` (0.04). It
prints, for both surface methods, how well the estimated branch radii follow
the true radii and how many branches get the scale they would get with the
true radii.

## Licence and citation

VascuSynth is distributed under the Apache 2.0 licence. The volumes it
generates are yours to share. Please cite:

- G. Hamarneh and P. Jassi, *VascuSynth: Simulating vascular trees for
  generating volumetric image data with ground truth segmentation and tree
  analysis*, Computerized Medical Imaging and Graphics 34(8), 605–616, 2010.
- P. Jassi and G. Hamarneh, *VascuSynth: Vascular tree synthesis software*,
  Insight Journal, 2011.
