import pyvista as pv
import numpy as np
import numpy.typing as npt


def wrap_numpy_object(obj: npt.NDArray):
    """
    Wrap any given VTK data object to its appropriate PyVista data object.

    Other formats that are supported include:

    2D numpy.ndarray of XYZ vertices
    3D numpy.ndarray representing a volume. Values will be scalars.
    3D trimesh.Trimesh mesh.
    3D meshio.Mesh mesh.
    """
    return pv.wrap(obj)

