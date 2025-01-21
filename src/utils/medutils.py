from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt


def load_nifti(img_path: str | Path) -> npt.NDArray:
    return np.asarray(nib.load(img_path).dataobj)
