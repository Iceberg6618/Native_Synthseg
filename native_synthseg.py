"""Run SynthSeg and register the segmentation back to native image space.

Main entry points:
    synthseg_single_nifti: run one NIfTI image.
    synthseg_multi_nifti: run multiple NIfTI images with one cached model.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import ClassVar, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ants
import nibabel as nib
import numpy as np


MODULE_ROOT = os.path.abspath(os.path.dirname(__file__))
SYNTHSEG_ROOT = os.path.join(MODULE_ROOT, "SynthSeg")

if SYNTHSEG_ROOT not in sys.path:
    sys.path.insert(0, SYNTHSEG_ROOT)

from SynthSeg.predict import get_flip_indices
from SynthSeg.predict_synthseg import build_model, postprocess, preprocess
from ext.lab2im import utils as synthseg_utils

try:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
except Exception:
    tf = None


LOCAL_DATA_ROOT = os.path.join(MODULE_ROOT, "data")
LOCAL_MODEL_PATH = os.path.join(LOCAL_DATA_ROOT, "models", "synthseg_2.0.h5")
LOCAL_LABELS_PATH = os.path.join(
    LOCAL_DATA_ROOT,
    "labels_classes_priors",
    "synthseg_segmentation_labels_2.0.npy",
)
SYNTHSEG_LABELS_ROOT = os.path.join(
    SYNTHSEG_ROOT,
    "data",
    "labels_classes_priors",
)
SYNTHSEG_LABELS_PATH = os.path.join(SYNTHSEG_LABELS_ROOT, "synthseg_segmentation_labels_2.0.npy")
SYNTHSEG_PARC_LABELS_PATH = os.path.join(SYNTHSEG_LABELS_ROOT, "synthseg_parcellation_labels.npy")
SYNTHSEG_QC_LABELS_PATH = os.path.join(SYNTHSEG_LABELS_ROOT, "synthseg_qc_labels_2.0.npy")


def _prefer_local(local_path: str, fallback_path: str) -> str:
    return local_path if os.path.exists(local_path) else fallback_path


def _nifti_from_volume(volume, aff, header, dtype=None) -> nib.Nifti1Image:
    if header is None:
        header = nib.Nifti1Header()
    if aff is None:
        aff = np.eye(4)
    if dtype is not None:
        if "int" in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)
    nifty = nib.Nifti1Image(volume, aff, header)
    if dtype is not None:
        nifty.set_data_dtype(dtype)
    return nifty


def _ants_image_from_nifti(nifti_image: nib.Nifti1Image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    try:
        nib.save(nifti_image, temp_path)
        return ants.image_read(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _nifti_from_ants_image(ants_image: ants.core.ants_image.ANTsImage) -> nib.Nifti1Image:
    temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    try:
        ants.image_write(ants_image, temp_path)
        loaded = nib.load(temp_path)
        return nib.Nifti1Image(loaded.get_fdata(), loaded.affine, loaded.header)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@contextmanager
def _temporary_cuda_setting(value: str):
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous

@dataclass(frozen=True)
class NativeSynthSegConfig:
    """Settings that define a reusable SynthSeg model instance."""

    path_model_segmentation: str = LOCAL_MODEL_PATH
    labels_segmentation: str = _prefer_local(LOCAL_LABELS_PATH, SYNTHSEG_LABELS_PATH)
    labels_denoiser: str = _prefer_local(LOCAL_LABELS_PATH, SYNTHSEG_LABELS_PATH)
    path_model_parcellation: str = os.path.join(LOCAL_DATA_ROOT, "models", "synthseg_parc_2.0.h5")
    labels_parcellation: str = SYNTHSEG_PARC_LABELS_PATH
    path_model_qc: str = os.path.join(LOCAL_DATA_ROOT, "models", "synthseg_qc_2.0.h5")
    labels_qc: str = SYNTHSEG_QC_LABELS_PATH
    robust: bool = False
    fast: bool = False
    v1: bool = False
    n_neutral_labels: int = 19
    do_parcellation: bool = False
    sigma_smoothing: float = 0.5
    input_shape_qc: int = 224
    cropping: int | None = 192
    ct: bool = False
    force_cpu: bool = True

class NativeSynthSegRunner:
    """Cached SynthSeg inference plus native-space registration."""

    _cache: ClassVar[dict[NativeSynthSegConfig, "NativeSynthSegRunner"]] = {}

    @classmethod
    def get(cls, config: NativeSynthSegConfig | None = None) -> "NativeSynthSegRunner":
        config = config or NativeSynthSegConfig()
        if config not in cls._cache:
            cls._cache[config] = cls(config)
        return cls._cache[config]

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    def __init__(self, config: NativeSynthSegConfig):
        self.config = config

        labels_segmentation, _ = synthseg_utils.get_list_labels(config.labels_segmentation)
        self.labels_segmentation, self.flip_indices, _ = get_flip_indices(
            labels_segmentation, config.n_neutral_labels
        )
        self.labels_denoiser = np.unique(synthseg_utils.get_list_labels(config.labels_denoiser)[0])

        if config.cropping is not None:
            self.cropping = synthseg_utils.reformat_to_list(config.cropping, length=3, dtype="int")
            self.min_pad = self.cropping
        else:
            self.cropping = None
            self.min_pad = 128

        device_context = _temporary_cuda_setting("-1") if config.force_cpu else nullcontext()
        with device_context:
            self.net = build_model(
                path_model_segmentation=config.path_model_segmentation,
                path_model_parcellation=config.path_model_parcellation,
                path_model_qc=config.path_model_qc,
                input_shape_qc=config.input_shape_qc,
                labels_segmentation=self.labels_segmentation,
                labels_denoiser=self.labels_denoiser,
                labels_parcellation=config.labels_parcellation,
                labels_qc=config.labels_qc,
                sigma_smoothing=config.sigma_smoothing,
                flip_indices=self.flip_indices,
                robust=config.robust,
                do_parcellation=config.do_parcellation,
                do_qc=False,
            )

    def run_native(
        self,
        image: str | nib.Nifti1Image,
        output_path: str | None = None,
        overwrite: bool = False,
    ) -> nib.Nifti1Image:
        """Run SynthSeg on one image and return/save segmentation in native space."""
        if output_path is not None and os.path.exists(output_path) and not overwrite:
            return nib.load(output_path)

        image_path = os.fspath(image) if isinstance(image, (str, os.PathLike)) else None
        native_t2 = nib.load(image_path) if image_path is not None else image
        native_t2.header["qform_code"] = 1

        temp_path = None
        try:
            if image_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                temp_path = temp_file.name
                temp_file.close()
                nib.save(native_t2, temp_path)
                image_path = temp_path

            synthseg_iso, t2_iso = self._predict_isotropic(image_path)
            native_seg = self._register_to_native(
                synthseg_iso=synthseg_iso,
                t2_iso=t2_iso,
                native_t2=native_t2,
            )
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            nib.save(native_seg, output_path)

        return native_seg

    def _predict_isotropic(self, image_path: str) -> tuple[nib.Nifti1Image, nib.Nifti1Image | None]:
        device_context = _temporary_cuda_setting("-1") if self.config.force_cpu else nullcontext()
        resample_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        resample_path = resample_file.name
        resample_file.close()
        t2_iso = None

        with device_context:
            try:
                preprocess_outputs = preprocess(
                    path_image=image_path,
                    ct=self.config.ct,
                    crop=self.cropping,
                    min_pad=self.min_pad,
                    path_resample=resample_path,
                )
                if len(preprocess_outputs) == 8:
                    image_array, aff, header, im_res, shape, pad_idx, crop_idx, t2_iso = preprocess_outputs
                else:
                    image_array, aff, header, im_res, shape, pad_idx, crop_idx = preprocess_outputs
                    if os.path.exists(resample_path) and os.path.getsize(resample_path) > 0:
                        resampled_nib = nib.load(resample_path)
                        t2_iso = nib.Nifti1Image(
                            resampled_nib.get_fdata(),
                            resampled_nib.affine,
                            resampled_nib.header,
                        )

                post_patch_segmentation = self.net.predict(image_array)
                seg, _, _ = postprocess(
                    post_patch_seg=post_patch_segmentation,
                    post_patch_parc=None,
                    shape=shape,
                    pad_idx=pad_idx,
                    crop_idx=crop_idx,
                    labels_segmentation=self.labels_segmentation,
                    labels_parcellation=self.config.labels_parcellation,
                    aff=aff,
                    im_res=im_res,
                    fast=self.config.fast,
                    topology_classes=None,
                    v1=self.config.v1,
                )
            finally:
                if os.path.exists(resample_path):
                    os.remove(resample_path)

        synthseg_iso = _nifti_from_volume(seg, aff, header, dtype="int32")
        return synthseg_iso, t2_iso

    @staticmethod
    def _register_to_native(
        synthseg_iso: nib.Nifti1Image,
        t2_iso: nib.Nifti1Image | None,
        native_t2: nib.Nifti1Image,
    ) -> nib.Nifti1Image:
        if t2_iso is None:
            t2_iso = native_t2

        synthseg_iso.header["qform_code"] = 1
        t2_iso.header["qform_code"] = 1

        native_t2_ants = _ants_image_from_nifti(native_t2)
        t2_iso_ants = _ants_image_from_nifti(t2_iso)
        synthseg_iso_ants = _ants_image_from_nifti(synthseg_iso)

        iso_to_native = ants.registration(
            fixed=t2_iso_ants,
            moving=native_t2_ants,
            type_of_transform="Rigid",
        )
        native_seg_ants = ants.apply_transforms(
            fixed=native_t2_ants,
            moving=synthseg_iso_ants,
            transformlist=iso_to_native["invtransforms"],
            interpolator="genericLabel",
        )

        native_seg = _nifti_from_ants_image(native_seg_ants)
        native_seg.header["qform_code"] = 1
        return native_seg


def _default_output_path(input_path: str, output_dir: str) -> str:
    name = os.path.basename(input_path)
    
    if name.endswith(".nii.gz"):
        stem = name[:-7]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        stem = os.path.splitext(name)[0]
        
    return os.path.join(output_dir, f"{stem}_native_synthseg.nii.gz")

def synthseg_single_nifti(
    input_path: str,
    output_path: str | None = None,
    *,
    config: NativeSynthSegConfig | None = None,
    runner: NativeSynthSegRunner | None = None,
    overwrite: bool = False,
) -> nib.Nifti1Image:
    """Run SynthSeg for a single NIfTI and register the result to native space."""
    active_runner = runner or NativeSynthSegRunner.get(config)
    return active_runner.run_native(input_path, output_path=output_path, overwrite=overwrite)


def synthseg_multi_nifti(
    input_paths: Iterable[str],
    output_paths: Iterable[str],
    *,
    config: NativeSynthSegConfig | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Run SynthSeg for multiple NIfTI path pairs, reusing one loaded model."""
    input_path_list = list(input_paths)
    output_path_list = list(output_paths)

    if len(input_path_list) != len(output_path_list):
        raise ValueError(
            f"input_paths and output_paths must have the same length "
            f"({len(input_path_list)} != {len(output_path_list)})."
        )

    runner = NativeSynthSegRunner.get(config)

    for input_path, output_path in zip(input_path_list, output_path_list):
        runner.run_native(input_path, output_path=output_path, overwrite=overwrite)

    return output_path_list
