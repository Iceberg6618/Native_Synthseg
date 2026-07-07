# Native SynthSeg Module

Native SynthSeg Module runs SynthSeg on a NIfTI image and registers the SynthSeg
segmentation back to the original native image space.

This package is designed for repeated inference. When multiple input images are
provided, the SynthSeg model is loaded once and reused.

## What This Package Does

1. Loads a NIfTI image.
2. Runs SynthSeg in the preprocessing space used by the original SynthSeg code.
3. Converts the SynthSeg segmentation into a NIfTI image.
4. Registers the SynthSeg label map back to the input image's native space.
5. Saves the native-space SynthSeg output as `.nii.gz`.

The package uses the original SynthSeg source code placed under:

```text
Native_Synthseg_Module/SynthSeg
```

## Attribution

This module uses the original SynthSeg source code from
[BBillot/SynthSeg](https://github.com/BBillot/SynthSeg). The wrapper in this
repository adds native-space registration utilities and a simplified Python API
for single-image and batch NIfTI processing.

## Installation

Create a clean CPU-only conda environment:

```bash
conda create -n Native_Synthseg_Module python=3.10 pip -y
conda activate Native_Synthseg_Module
```

Install this package from the repository root:

```bash
cd Native_Synthseg_Module
pip install -e .
```

Alternatively, install the runtime requirements first:

```bash
pip install -r requirements.txt
pip install -e .
```

The tested CPU-only core dependency set is:

```text
python 3.10
tensorflow-cpu 2.10.1
keras 2.10.0
h5py 3.8.0
numpy 1.23.5
scipy 1.10.1
nibabel 5.3.3
antspyx 0.6.3
protobuf 3.19.6
```

## Required Files

The package expects the SynthSeg model and labels to be present at:

```text
Native_Synthseg_Module/data/models/synthseg_2.0.h5
Native_Synthseg_Module/data/labels_classes_priors/synthseg_segmentation_labels_2.0.npy
Native_Synthseg_Module/SynthSeg/data/labels_classes_priors/*.npy
```

## Usage

Import the public API from the package root:

```python
from Native_Synthseg_Module import run_native_synthseg
```

### 1. path2path

Use one input file path and one output file path.

```python
run_native_synthseg(
    input_paths="subj01.nii.gz",
    output_paths="outputs/subj01_seg.nii.gz",
)
```

Saved output:

```text
outputs/subj01_seg.nii.gz
```

### 2. path2dir

Use one input file path and one output directory.

```python
run_native_synthseg(
    input_paths="subj01.nii.gz",
    output_paths="outputs",
)
```

Saved output:

```text
outputs/subj01_native_synthseg.nii.gz
```

### 3. pathlist2pathlist

Use a list of input file paths and a matching list of output file paths.

```python
run_native_synthseg(
    input_paths=[
        "subj01.nii.gz",
        "subj02.nii.gz",
    ],
    output_paths=[
        "outputs/subj01_seg.nii.gz",
        "outputs/subj02_seg.nii.gz",
    ],
)
```

Saved outputs:

```text
outputs/subj01_seg.nii.gz
outputs/subj02_seg.nii.gz
```

### 4. pathlist2dir

Use a list of input file paths and one output directory.

```python
run_native_synthseg(
    input_paths=[
        "subj01.nii.gz",
        "subj02.nii.gz",
    ],
    output_paths="outputs",
)
```

Saved outputs:

```text
outputs/subj01_native_synthseg.nii.gz
outputs/subj02_native_synthseg.nii.gz
```

## Overwrite Existing Outputs

Existing outputs are reused by default. To regenerate outputs:

```python
run_native_synthseg(
    input_paths="subj01.nii.gz",
    output_paths="outputs",
    overwrite=True,
)
```

## Low-Level API

You can also use the lower-level API directly:

```python
from Native_Synthseg_Module import NativeSynthSegRunner, synthseg_single_nifti

runner = NativeSynthSegRunner.get()
synthseg_single_nifti(
    "subj01.nii.gz",
    output_path="outputs/subj01_seg.nii.gz",
    runner=runner,
)
```

## Notes

- TensorFlow is configured for CPU-only execution by default.
- The native-space registration uses ANTsPyX rigid registration.
- Label maps are resampled with label interpolation.
- The package is intended to be run on 3D NIfTI images.
