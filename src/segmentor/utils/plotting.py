import pyvista as pv
import numpy as np
import numpy.typing as npt
from typing import Any, Union
from scipy.ndimage import binary_dilation


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


def plotLine3d(x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> list[tuple[int, int, int]]:
    """Based on the Bresenham's line algorithm in 3D.
    
    Source: http://members.chello.at/easyfilter/bresenham.html

    Parameters
    ----------
    (x0, y0, z0) : tuple of int
        The start point of the line.
    (x1, y1, z1) : tuple of int
        The end point of the line.

    Returns
    -------
    list of tuple of int
        The coordinates of the line
    """
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    dz = abs(z1 - z0)
    sz = 1 if z0 < z1 else -1
    dm = max(dx, dy, dz)
    i = dm
    ox1 = oy1 = oz1 = dm // 2

    while True:
        points.append((x0, y0, z0))
        if i == 0:
            break
        i -= 1
        ox1 -= dx
        if ox1 < 0:
            ox1 += dm
            x0 += sx
        oy1 -= dy
        if oy1 < 0:
            oy1 += dm
            y0 += sy
        oz1 -= dz
        if oz1 < 0:
            oz1 += dm
            z0 += sz
    return points


def path3d(reference: np.ndarray, path: list[tuple], dilate: int = 0, return_coords = True) -> np.ndarray | Union[np.ndarray, list]:
    """Draws a path on a 3D volume."""
    base = np.zeros_like(reference)
    base[tuple(np.asarray(path).T)] = 1

    if return_coords:
        coord_list = []

    for start, end in zip(path[:-1], path[1:]):
        coords = plotLine3d(*start, *end)
        base[tuple(np.asarray(coords).T)] = 1

        if return_coords:
            coord_list.extend(coords)

    if dilate:
        base = binary_dilation(base, iterations=dilate)

    return (base, coord_list) if return_coords else base