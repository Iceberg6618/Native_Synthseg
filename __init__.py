"""Native-space SynthSeg package."""

from __future__ import annotations

import os
from typing import Iterable

from .native_synthseg import (
    NativeSynthSegConfig,
    NativeSynthSegRunner,
    _default_output_path,
    synthseg_multi_nifti,
    synthseg_single_nifti,
)


def _is_path_sequence(value) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, os.PathLike))


def _normalise_input_paths(input_paths) -> tuple[list[str], bool]:
    if _is_path_sequence(input_paths):
        return [os.fspath(path) for path in input_paths], False
    return [os.fspath(input_paths)], True


def _normalise_output_paths(output_paths, input_path_list: list[str]) -> list[str]:
    if output_paths is None:
        return [_default_output_path(path, os.getcwd()) for path in input_path_list]

    if _is_path_sequence(output_paths):
        return [os.fspath(path) for path in output_paths]

    output_path = os.fspath(output_paths)
    if len(input_path_list) == 1:
        if output_path.endswith(".nii") or output_path.endswith(".nii.gz"):
            return [output_path]
        return [_default_output_path(input_path_list[0], output_path)]

    if output_path.endswith(".nii") or output_path.endswith(".nii.gz"):
        raise ValueError("A single output file path can only be used with one input image.")

    return [_default_output_path(path, output_path) for path in input_path_list]


def run_native_synthseg(
    input_paths,
    output_paths=None,
    *,
    config: NativeSynthSegConfig | None = None,
    overwrite: bool = False,
):
    """
    Run SynthSeg and register the segmentation back to each input's native space.

    This high-level function accepts a single NIfTI path or a sequence of NIfTI
    paths. For multiple inputs, the SynthSeg model is loaded once and reused.

    Args:
        input_paths:
            Single input path or list/tuple of input paths. Inputs should be
            ``.nii`` or ``.nii.gz`` files.
        output_paths:
            Output file path, output directory, list/tuple of output file paths,
            or ``None``. The accepted forms are:

            path2path:
                ``run_native_synthseg("subj01.nii.gz", "out/subj01_seg.nii.gz")``
                saves exactly ``out/subj01_seg.nii.gz``.

            path2dir:
                ``run_native_synthseg("subj01.nii.gz", "out")``
                saves ``out/subj01_native_synthseg.nii.gz``.

            pathlist2pathlist:
                ``run_native_synthseg(["subj01.nii.gz", "subj02.nii.gz"],
                ["out/subj01_seg.nii.gz", "out/subj02_seg.nii.gz"])``
                saves exactly ``out/subj01_seg.nii.gz`` and
                ``out/subj02_seg.nii.gz``.

            pathlist2dir:
                ``run_native_synthseg(["subj01.nii.gz", "subj02.nii.gz"], "out")``
                saves ``out/subj01_native_synthseg.nii.gz`` and
                ``out/subj02_native_synthseg.nii.gz``.

            If ``output_paths`` is ``None``, outputs are written to the current
            working directory using ``{input_stem}_native_synthseg.nii.gz``.
        config:
            Optional ``NativeSynthSegConfig`` for model paths and SynthSeg
            settings.
        overwrite:
            If ``False``, existing output files are reused. If ``True``, outputs
            are regenerated.

    Returns:
        A ``nibabel.Nifti1Image`` for a single input, or a list of output paths
        for multiple inputs.
    """
    input_path_list, single_input = _normalise_input_paths(input_paths)
    output_path_list = _normalise_output_paths(output_paths, input_path_list)

    if len(input_path_list) != len(output_path_list):
        raise ValueError(
            f"input_paths and output_paths must have the same length "
            f"({len(input_path_list)} != {len(output_path_list)})."
        )

    if single_input:
        return synthseg_single_nifti(
            input_path_list[0],
            output_path=output_path_list[0],
            config=config,
            overwrite=overwrite,
        )

    return synthseg_multi_nifti(
        input_path_list,
        output_path_list,
        config=config,
        overwrite=overwrite,
    )


__all__ = [
    "NativeSynthSegConfig",
    "NativeSynthSegRunner",
    "run_native_synthseg",
    "synthseg_multi_nifti",
    "synthseg_single_nifti",
]
