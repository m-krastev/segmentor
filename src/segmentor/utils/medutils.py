from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import numpy.typing as npt


def load_nifti(img_path: str | Path) -> npt.NDArray:
    return np.asarray(nib.load(img_path).dataobj)


def save_nifti(data: npt.NDArray, filename: str | Path, other: str | Path = None):
    if other is None:
        other = nib.load(filename)
    else:
        other = nib.load(other)
    new_image = nib.Nifti1Image(data, other.affine, other.header)
    nib.save(new_image, filename)


def normalize_ct(
    nii,
    quantiles: tuple[float, float] = (0.0005, 0.9995),
    window: tuple[float, float] = (50, 400),
):
    nii = np.where(
        (nii > np.quantile(nii, quantiles[0])) & (nii < (np.quantile(nii, quantiles[1]))),
        nii,
        0,
    )
    window_c, window_w = window
    nii = np.where((nii >= window_c - window_w / 2) & (nii <= window_c + window_w / 2), nii, 0)
    # nii = nii / np.linalg.norm(nii, axis=-1, keepdims=True)
    return nii


def load_and_normalize_nifti(filename, quantiles=(0.0005, 0.9995), window=(50, 400)):
    nii = nib.load(filename)
    nii = np.asarray(nii.dataobj)
    nii = normalize_ct(nii, quantiles, window)
    return nii


def load_and_resample_nifti(filename: str, factor=0.5, order=0, normalize=False):
    # Load the image
    image = nib.load(filename)
    if normalize:
        image.dataobj = normalize_ct(image.dataobj)
    target_shape = np.array(image.affine) * factor
    # Create a new affine matrix by copying the original and scaling the rotation/scaling part.
    # This assumes the original affine encodes isotropic voxel sizes.
    new_affine = image.affine.copy()
    new_affine[:3, :3] *= 1 / factor
    return resample_from_to(image, (target_shape, new_affine), order=order)
