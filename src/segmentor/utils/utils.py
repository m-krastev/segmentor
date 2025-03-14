from dataclasses import _DataclassT
from typing import Any, Dict
import numpy as np
import vtk

def create_base(model_class: _DataclassT, data: Dict[str, Any]) -> _DataclassT:
    valid_keys = set(_DataclassT.__dataclass_fields__.keys())
    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
    return model_class(**filtered_data)


def nii_2_mesh(filename_nii, filename_stl, label: int=1, num_iterations:int =30):
    """
    Read a NIFTI file containing a binary segmentation mask, convert the specified label to a mesh, and save it as an STL file.

    Args:
        filename_nii (str): Path to the input NIFTI file.
        filename_stl (str): Path to the output STL file.
        label (int): The label value in the segmentation mask to convert to a mesh.
        num_iterations (int, optional): Number of smoothing iterations. Higher values result in a smoother mesh. Defaults to 30.

    Returns:
        None: The function saves an STL file to the specified path.

    Courtesy of: https://github.com/MahsaShk/MeshProcessing/blob/master/nii_2_mesh_conversion.py
    """

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()

    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, label)  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf.GetOutput())
    else:
        smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(num_iterations)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    # save the output
    writer = vtk.vtkOBJWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    # writer.SetFileTypeToASCII()
    writer.SetFileName(filename_stl)
    writer.Write()


def Bresenham3D(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    if x2 > x1:
        xs = 1
    else:
        xs = -1
    if y2 > y1:
        ys = 1
    else:
        ys = -1
    if z2 > z1:
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints


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


def create_volumetric_image(
    nodes: list[tuple[float, float, float]],
    connections: list[tuple[Any, Any]],
    voxel_size=(1, 1, 1),
    image_size=None,
):
    # Convert node list to a dictionary for easier access
    node_dict = {index: (x, y, z) for index, (x, y, z) in enumerate(nodes)}

    if not image_size:
        max_extent = np.max(nodes, axis=0)
        image_size = np.ceil(max_extent / np.array(voxel_size)).astype(int)

    volume = np.zeros(image_size, dtype=np.uint8)
    voxel_size = np.array(voxel_size)

    # Draw lines between connected nodes
    for index, parent_index in connections:
        if parent_index == -1:
            continue  # Skip the root node which has no parent
        p0 = np.array(node_dict[index])
        p1 = np.array(node_dict[parent_index])
        line_points = Bresenham3D(*np.floor(p0), *np.ceil(p1))
        line_points.extend(Bresenham3D(*np.ceil(p0), *np.floor(p1)))

        # for x, y, z in set(line_points):
        #     i, j, k = np.round(np.array([x, y, z]) / np.array(voxel_size)).astype(int)
        #     volume[i, j, k] = 1

        volume[np.round(np.array(line_points) / voxel_size).astype(int)] = 1

    return volume



# ########################################################
# EVALUATION METRICS
# ########################################################

def line_length(points: list[tuple]):
    """
    Calculate the length of a line in ND space.

    Args:
        list[tuple]: List of tuples containing the coordinates of the connected points.

    Returns:
        float: The length of the line.
    """
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=-1))

def dice_overlap(segmentation: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate the Dice coefficient between two binary masks.

    Args:
        segmentation (np.ndarray): The predicted binary mask.
        ground_truth (np.ndarray): The ground truth binary mask.

    Returns:
        float: The Dice coefficient.
    """
    intersection = np.sum(segmentation * ground_truth)
    return 2 * intersection / (np.sum(segmentation) + np.sum(ground_truth))

def hausdorff_distance(segmentation: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate the Hausdorff distance between two binary masks.

    Args:
        segmentation (np.ndarray): The predicted binary mask.
        ground_truth (np.ndarray): The ground truth binary mask.

    Returns:
        float: The Hausdorff distance.

    NOTE: This algo is trash and takes 10 million years, don't use it.
    """
    seg_to_gt = np.max(np.array([np.min(np.linalg.norm(segmentation - gt, axis=-1)) for gt in np.argwhere(ground_truth)]))
    gt_to_seg = np.max(np.array([np.min(np.linalg.norm(gt - segmentation, axis=-1)) for gt in np.argwhere(ground_truth)]))
    return max(seg_to_gt, gt_to_seg)


def average_gradient(path: list[tuple[int, int, int]], image: np.ndarray) -> float:
    """
    Calculate the average of the gradient along a path in an image.

    Args:
        path (list[tuple[int, int, int]]): List of tuples containing the coordinates of the connected points.
        image (np.ndarray): The image to use as index.

    Returns:
        float: The average of the gradient along the path.
    """
    return np.mean(image[coordinate_to_index(np.asarray(path))])

def coordinate_to_index(coord):
    """
    Convert a list of coordinates to a list of indices, e.g.
     [[1, 2], [3, 4], [5, 6]] -> [[1, 3, 5], [2, 4, 6]]
    """
    return tuple(coord.astype(int).T)