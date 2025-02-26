from dataclasses import _DataclassT
from typing import Any, Dict
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
