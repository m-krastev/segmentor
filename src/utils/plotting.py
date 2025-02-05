import pyvista as pv
import numpy as np
import numpy.typing as npt
from typing import Any


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


def plot_skeleton_3d(skeleton: Any, volume: npt.NDArray = None):
    """Plot a 3D skeleton with pyvista

    Parameters
    ----------
    skeleton : cloudvolume.Skeleton
        The skeleton to plot. Should contain the vertices, edges and radii
    volume : np.ndarray, optional
        The volume segmentation mask to plot. Default is None.
    """
    plotter = pv.Plotter()

    skeleton_3d = pv.PolyData(
        skeleton.vertices,
        lines=np.concatenate((np.full(skeleton.edges.shape[0], 2).reshape(-1, 1), skeleton.edges), axis=1),
    )

    if hasattr(skeleton, "radii"):
        # Not sure why we need to remove one element, but for some reason the polydata is one element less.
        skeleton_3d.cell_data["width"] = skeleton.radii[1:]

    plotter.add_mesh(skeleton_3d, show_edges=True, line_width=5, scalars="width")

    if volume is not None:
        plotter.add_volume(volume * 20, cmap="viridis", specular=0.5, specular_power=15)

    plotter.view_xz()
    plotter.show()
