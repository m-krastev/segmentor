# import cv2
# import numpy as np
# import SimpleITK as sitk
# from scipy.ndimage import morphology
# from tqdm import tqdm
#
# from archive.utils.utils import torch_from_nii, save_nii
#
#
# def read_swc(file_path):
#     """
#     Reads an SWC file and returns a list of nodes and their connections.
#     Each node is represented as a tuple: (index, x, y, z, radius, parent_index).
#     """
#     nodes = []
#     connections = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             if line.startswith('#') or len(line) <= 1:  # Skip comments
#                 continue
#
#             parts = line.strip().split()
#             index, _, x, y, z, radius, parent_index = map(float, parts)
#             nodes.append((index, x, y, z, radius))
#             connections.append((index, parent_index))
#     return nodes, connections
#
#
#
#
# def interpolate_points(p0, p1):
#     """
#     Bresenham's line algorithm for 3D to interpolate points between two points p0 and p1.
#     Returns a list of points (as tuples) on the line between p0 and p1.
#     """
#     points = []
#     x0, y0, z0 = p0
#     x1, y1, z1 = p1
#     dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
#     xs, ys, zs = np.sign(x1 - x0) / 10, np.sign(y1 - y0) / 10, np.sign(z1 - z0) / 10
#     ax, ay, az = 2 * dx, 2 * dy, 2 * dz
#     if dx >= dy and dx >= dz:
#         yd, zd = ay - dx, az - dx
#         current_dist = abs(x1-x0)
#         while abs(x1-x0) <= current_dist:
#             points.append((x0, y0, z0))
#             if yd >= 0:
#                 y0 += ys
#                 yd -= ax
#             if zd >= 0:
#                 z0 += zs
#                 zd -= ax
#             x0 += xs
#             yd += ay
#             zd += az
#     elif dy >= dx and dy >= dz:
#         xd, zd = ax - dy, az - dy
#         current_dist = abs(y1-y0)
#         while abs(y1-y0) <= current_dist:
#             points.append((x0, y0, z0))
#             if xd >= 0:
#                 x0 += xs
#                 xd -= ay
#             if zd >= 0:
#                 z0 += zs
#                 zd -= ay
#             y0 += ys
#             xd += ax
#             zd += az
#     else:
#         xd, yd = ax - dz, ay - dz
#         current_dist = abs(z1-z0)
#         while abs(z1-z0) <= current_dist:
#             points.append((x0, y0, z0))
#             if xd >= 0:
#                 x0 += xs
#                 xd -= az
#             if yd >= 0:
#                 y0 += ys
#                 yd -= az
#             z0 += zs
#             xd += ax
#             yd += ay
#     points.append((x1, y1, z1))
#     return points
#
#
#
#
#
# def create_volumetric_image_no_conn(nodes, voxel_size=(1, 1, 1), image_size=None):
#     """
#     Creates a 3D matrix from the list of nodes. Each node's position is marked in the matrix.
#     Optionally uses voxel_size to scale node coordinates.
#     """
#     if not image_size:
#         # Determine the required size of the 3D matrix
#         max_extent = np.max(np.array(nodes)[:, :3], axis=0)
#         image_size = np.ceil(max_extent / np.array(voxel_size)).astype(int)
#
#     # Create an empty 3D matrix
#     volume = np.zeros(image_size, dtype=np.uint8)
#
#     for _, x, y, z, radius in nodes:
#         # Convert node coordinates to voxel indices
#         i, j, k = np.round(np.array([x, y, z]) / np.array(voxel_size)).astype(int)
#         volume[i, j, k] = 1  # Mark the voxel as occupied
#
#     return volume
#
#
# if __name__ == "__main__":
#     # Path to your SWC file
#     swc_file_path = 'graph.swc'
#
#     # Read SWC file and create a list of nodes
#     nodes, connections = read_swc(swc_file_path)
#
#     # Create a 3D matrix from the nodes
#     voxel_grid = create_volumetric_image(nodes, connections, voxel_size=(1.25, 1.25, 1.25), image_size=(100, 160, 160))
#
#
#     from scipy import ndimage
#     import matplotlib.pyplot as plt
#
#     struct1 = ndimage.generate_binary_structure(3, 1)
#     dilated = ndimage.binary_dilation(voxel_grid, structure=struct1, iterations=1).astype(np.uint8)
#
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(voxel_grid[50])
#     ax[1].imshow(dilated[50])
#     plt.show()
#
#     # Save image in correct coordinates
#     _, header = torch_from_nii("/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/annotations_v2/Bowel_06v2.nii")
#     save_nii(dilated, header, "centerline_connected_dilated.nii")
