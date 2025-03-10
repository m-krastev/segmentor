from pathlib import Path
from typing import Optional

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import numpy.typing as npt


def load_nifti(img_path: str | Path) -> npt.NDArray:
    """Loads a NIfTI image from the given path and returns it as a NumPy array."""
    return np.asarray(nib.load(img_path).dataobj)


def save_nifti(data: npt.NDArray, filename: str | Path, other: str | Path = None):
    """
    Saves a NumPy array as a NIfTI image.

    Args:
        data (npt.NDArray): The NumPy array to save.
        filename (str | Path): The filename to save the image to.
        other (str | Path, optional): Another NIfTI image to copy the header from. Defaults to None.
                                        If None, the header from the image to be saved is used.
    """
    if other is None:
        other = nib.load(filename)
    else:
        other = nib.load(other)
    new_image = nib.Nifti1Image(data, other.affine, other.header)
    nib.save(new_image, filename)


def normalize_ct(
    nii: npt.NDArray,
    quantiles: Optional[tuple[float, float]] = (0.0005, 0.9995),
    window: Optional[tuple[float, float]] = (50, 400),
) -> npt.NDArray:
    """
    Normalizes a CT scan image.

    Args:
        nii (npt.NDArray): The CT scan image as a NumPy array.
        quantiles (tuple[float, float], optional): The quantiles to use for clipping. Defaults to (0.0005, 0.9995).
        window (tuple[float, float], optional): The window to use for normalization. Defaults to (50, 400).

    Returns:
        npt.NDArray: The normalized CT scan image.
    """
    # Clip values outside the specified quantiles to 0
    if quantiles:
        nii = np.clip(nii, *np.quantile(nii, quantiles))
    if window:
        window_c, window_w = window
        low = window_c - window_w / 2
        high = window_c + window_w / 2
        # Clip to window
        nii = np.clip(nii, low, high)
        # Normalize to [0, 1]
        nii = (nii - low) / (high - low)
    else:
        nii = (nii - np.min(nii)) / (np.max(nii) - np.min(nii))
    return nii

def load_and_normalize_nifti(
    filename: str | Path,
    quantiles: Optional[tuple[float, float]] = (0.0005, 0.9995),
    window: Optional[tuple[float, float]] = (50, 400),
) -> npt.NDArray:
    """Loads a NIfTI image from the given path, normalizes it, and returns it as a NumPy array."""
    nii = nib.load(filename)
    nii = np.asarray(nii.dataobj)
    nii = normalize_ct(nii, quantiles, window)
    return nii


def load_and_resample_nifti(
    filename: str, factor: float = 0.5, order: int = 0, normalize: bool = False, **kwargs
) -> nib.Nifti1Image:
    """
    Loads a NIfTI image from the given path, resamples it, and returns it as a NIfTI image.

    Args:
        filename (str): The path to the NIfTI image.
        factor (float, optional): The resampling factor. Defaults to 0.5.
        order (int, optional): The order of interpolation. Defaults to 0.
        normalize (bool, optional): Whether to normalize the image before resampling. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the normalization function.
    Returns:
        nib.Nifti1Image: The resampled NIfTI image.
    """
    # Load the image
    image = nib.load(filename)
    image_data = np.asarray(image.dataobj)

    if normalize:
        image_data = normalize_ct(image_data, **kwargs)

    # Resample the image data
    # Original image affine
    affine = image.affine

    # Original image shape
    shape = np.array(image_data.shape)

    # Calculate the new shape based on the resampling factor
    new_shape = shape * factor

    # Create a new affine matrix
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * (1 / factor)

    # Create a new NIfTI image
    new_image = nib.Nifti1Image(image_data, affine)

    resampled_image = resample_from_to(new_image, (new_shape, new_affine), order=order)
    return resampled_image
