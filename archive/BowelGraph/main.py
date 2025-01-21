from collections import defaultdict

import kimimaro
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from tqdm import tqdm
import os

from archive.utils.datasets import get_sdf
from archive.utils.utils import torch_from_nii, save_nii


def visualize(graph, plot_nodes=False, vis_deg=None, highlight_nodes=None, border_threshold=0.05):
    """
    Visualize graph in 3D
    :param graph: networkx graph
    :param plot_nodes: boolean to determine to plot all nodes
    :param vis_deg: list of degrees to color the corresponding nodes
    :param highlight_nodes: list of nodes to highlight in red
    """
    if vis_deg is None:
        vis_deg = list()
    if highlight_nodes is None:
        highlight_nodes = list()

    node_xyz = np.array([graph.nodes[i]['coords'] for i in sorted(graph)])
    edge_xyz = np.array([(graph.nodes[u]['coords'], graph.nodes[v]['coords']) for u, v in graph.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    if plot_nodes:
        ax.scatter(*node_xyz.T, s=10, ec="w")

    # Color degree nodes
    colors = list(mcolors.TABLEAU_COLORS.values())
    degs = get_degree_nodes(graph)
    for deg, nodes in degs.items():
        if deg in vis_deg:
            c = colors.pop()
            nodes = np.array([graph.nodes[node]['coords'] for node in nodes if node not in highlight_nodes])
            border_frac = nodes / np.array([100, 160, 160])
            at_border = (border_frac <= border_threshold).any(axis=1) | (border_frac >= (1 - border_threshold)).any(axis=1)

            # Plot nodes at border
            ax.scatter(*nodes[at_border].T, s=70, marker='*', c='red', edgecolors=c, label=f'Border node (deg {deg})')

            # Plot other nodes
            ax.scatter(*nodes[~at_border].T, s=70, ec="w", c=c, label=f'Deg {deg}')

    # Color highlighted nodes
    for node in highlight_nodes:
        ax.scatter(*graph.nodes[node]['coords'], color='red', s=50, ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", alpha=.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.legend()
    fig.tight_layout()
    plt.show()


def get_degree_nodes(graph):
    degree = defaultdict(list)
    for node in graph.nodes():
        order = len(graph.edges(node))
        degree[order].append(node)

    return degree


def seg_to_graph(segmentation):
    graph = kimimaro.skeletonize(
        segmentation.bool(),
        teasar_params={
            "scale": .5,
            "const": 10,
        },
        anisotropy=(1.25, 1.25, 1.25),
        progress=False,
        parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
    )

    # Check if there are multiple representations found
    assert len(graph.keys()) == 1, f"Found {len(graph.keys())} medial axis representations. Expected 1"
    graph = graph[1]

    # Correct vertices for anisotropy
    graph.vertices = graph.vertices / 1.25

    # Create mapping from node index to its voxel coordinates
    attrs = {i: {"coords": val} for i, val in enumerate(graph.vertices)}

    # Convert to networkx graph
    g = nx.Graph()
    g.add_nodes_from(attrs.keys())

    # Add coordinate (voxel space) of node as its attribute
    nx.set_node_attributes(g, attrs)

    # Add edges with length (norm) as attribute
    edges = []
    for (start, end) in graph.edges:
        length = np.linalg.norm(g.nodes[start]['coords'] - g.nodes[end]['coords'])
        length *= 1.25  # Correct the length for the physical distance a voxel represents (1.25mm)

        edges.append((start, end, {'length': length}))

    g.add_edges_from(edges)

    return g


def shortest_path(source, graph, ends):
    best_path = set()
    best_length = np.inf

    # Find the shortest path from source to any end (don't include path to source node)
    for end in set(ends) - {source}:
        if not nx.has_path(graph, source, end):
            continue

        # Find the shortest path based on edge's length attribute
        temp_path = nx.shortest_path(graph, source=source, target=end, weight='length')

        # Calculate length of shortest path
        temp_graph = nx.path_graph(temp_path)
        length = sum([graph.edges[start, end]['length'] for (start, end) in temp_graph.edges()])

        # Update best path
        if length < best_length:
            best_path = set(temp_path) - {source, end}  # Don't include source and end in path
            best_length = length

    return best_path, best_length


def post_process_graph(graph, end_threshold=15, distance_threshold=np.inf):
    for branch_node in get_degree_nodes(graph)[3]:
        # Check that branch node still has degree of 3
        if branch_node not in graph or len(graph.edges(branch_node)) != 3:
            continue

        degs = get_degree_nodes(graph)

        # Find the shortest paths between branch nodes and ends, and branch nodes and branch nodes
        shortest_end_path, end_length = shortest_path(branch_node, graph, degs[1])
        shortest_branch_path, branch_length = shortest_path(branch_node, graph, degs[3])

        # If no path to end node, remove branch path
        if not shortest_end_path:
            path_to_remove = shortest_branch_path
        # If no path to branch node, or end path is shorter than threshold, remove end path
        elif not shortest_branch_path or end_length <= end_threshold:
            path_to_remove = shortest_end_path
        # Prefer to remove branch path if it's shorter
        elif branch_length < end_length:
            path_to_remove = shortest_branch_path
        else:
            path_to_remove = shortest_end_path

        # There is a zero degree node left because of the path removal, this node should be deleted too
        zero_node = get_degree_nodes(graph.subgraph(graph.nodes - path_to_remove))[0]
        assert len(zero_node) <= 1, f"Found to many zero nodes: {len(zero_node)}"
        path_to_remove |= set(zero_node)

        if path_to_remove:
            # Calculate SDF as if path is removed
            img = torch.from_numpy(graph_to_image(graph.subgraph(graph.nodes - path_to_remove), dilate=True))
            sdf = get_sdf(img[None, ...])[0]  # Expect batch size dimension
            sdf *= 1.25  # SDF units to physical distance (mm)

            # Extract removed path region from sdf (include branch node in subgraph since removal path spans from this node)
            img = torch.from_numpy(graph_to_image(graph.subgraph({branch_node, *path_to_remove}), dilate=True))

            if sdf[img.bool()].max() <= distance_threshold:
                graph.remove_nodes_from(path_to_remove)

    # Remove 0 degree nodes
    for node in get_degree_nodes(graph)[0]:
        graph.remove_node(node)

    return graph


def Bresenham3D(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Y-axis
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints


def create_volumetric_image(nodes, connections, voxel_size=(1, 1, 1), image_size=None):
    # Convert node list to a dictionary for easier access
    node_dict = {index: (x, y, z) for index, x, y, z in nodes}

    if not image_size:
        max_extent = np.max(nodes, axis=0)
        image_size = np.ceil(max_extent / np.array(voxel_size)).astype(int)

    volume = np.zeros(image_size, dtype=np.uint8)

    # Draw lines between connected nodes
    for index, parent_index in connections:
        if parent_index == -1:
            continue  # Skip the root node which has no parent
        p0 = np.array(node_dict[index])
        p1 = np.array(node_dict[parent_index])
        line_points = Bresenham3D(*np.floor(p0), *np.ceil(p1))
        line_points.extend(Bresenham3D(*np.ceil(p0), *np.floor(p1)))

        for x, y, z in set(line_points):
            i, j, k = np.round(np.array([x, y, z]) / np.array(voxel_size)).astype(int)
            volume[i, j, k] = 1

    return volume


def graph_to_image(graph, dilate=False):
    # Create a 3D matrix from the nodes
    img = create_volumetric_image([(i, *graph.nodes[i]['coords']) for i in graph.nodes], graph.edges, voxel_size=(1., 1., 1.), image_size=(100, 160, 160))

    if dilate:
        struct1 = ndimage.generate_binary_structure(3, 1)
        img = ndimage.binary_dilation(img, structure=struct1, iterations=1).astype(np.uint8)

    return img


def score_graph(graph, gt_seg=None, gt_graph=None, voxel_spacing=1.25):
    # Intrinsic graph metrics
    n_cycles, connected_comps, degs = intrinsic_graph_stats(graph)
    length = sum([graph.edges[start, end]['length'] for (start, end) in graph.edges()])

    # Calculate number of end nodes NOT at the border margin of the volume
    border_margin = 5  # mm
    border_margin_voxels = border_margin / voxel_spacing
    inside_end_nodes = 0
    if degs[1]:
        for node in degs[1]:
            in_border_margin = any(graph.nodes[node]['coords'] <= border_margin_voxels)
            in_border_margin |= any(abs(graph.nodes[node]['coords'] - np.array([100, 160, 160])) <= border_margin_voxels)
            if not in_border_margin:
                inside_end_nodes += 1

    # Graph to seg metrics
    if gt_seg is not None:
        # calculate distance of every segmented voxel to the nearest point on the graph
        # Get SDF of graph
        img = torch.from_numpy(graph_to_image(graph, dilate=True))
        sdf = get_sdf(img[None, ...])[0]  # Expect batch size dimension
        sdf = sdf[gt_seg.bool()]  # Mask SDF with area of interest (gt segmentation)

    # Graph to graph metrics
    if gt_graph is not None:
        pass

    # Print metrics
    print("Graph info".center(30, "-"))
    print(f"components: {connected_comps}")
    print(f"cycles: {n_cycles}")
    print(f"length: {round(length,1)}mm")
    print(f"inside end nodes: {inside_end_nodes} ({round(inside_end_nodes/len(degs[1])*100)}%)")

    print(f"degree info".rjust(15, "~").ljust(30, "~"))
    for deg in sorted(degs.keys()):
        print(f"deg {deg}: {len(degs[deg])}", end="")
        # Add inside volume info
        if deg == 1:
            print(f" (inside: {inside_end_nodes}, {round(inside_end_nodes/len(degs[1])*100)}%)")
        else:
            print()

    print("-".center(30, "-"))

    if gt_seg is not None:
        return sdf, degs, connected_comps


def intrinsic_graph_stats(graph):
    n_cycles = len(list(nx.simple_cycles(graph)))
    connected_comps = nx.number_connected_components(graph)
    degs = get_degree_nodes(graph)

    return n_cycles, connected_comps, degs


def analyze_graph(graph):
    n_cycles, connected_comps, degs = intrinsic_graph_stats(graph)

    print("Graph info".center(30, "-"))
    print(f"nodes: {len(graph.nodes)}")
    print(f"edges: {len(graph.edges)}")
    print(f"components: {connected_comps}")
    print(f"cycles: {n_cycles}")

    print(f"degree info".rjust(15, "~").ljust(30, "~"))
    for deg in sorted(degs.keys()):
        print(f"deg {deg}: {len(degs[deg])}")
    print("-".center(30, "-"))


def print_sdf_stats(sdf, distance_threshold, voxel_spacing=1.25):
    sdf = sdf.clone() * voxel_spacing
    thr_frac = (sdf <= distance_threshold).float().mean()

    print(f"Mean distance: {round(sdf.mean().item(), 1)}mm")
    print(f"Std distance: {round(sdf.std().item(), 1)}mm")
    print(f"Within {distance_threshold}mm: {round(thr_frac.item() * 100)}%")
    return thr_frac


if __name__ == "__main__":
    # visualize(graph, plot_nodes=False, vis_deg=[1, 3, 4])#, highlight_nodes=[141])  # 483
    # save_nii(graph_to_image(graph, dilate=True), header, f"results/pt_{PATIENT}_MAT.nii")

    np.set_printoptions(suppress=True)

    gve_dist_thresh = 20  # mm
    end_thresh = 50  # mm
    dist_thresh = 25  # mm

    def calc_dist(graph_pred, gt_centerline, voxel_spacing=1.25):
        img = torch.from_numpy(graph_to_image(graph_pred, dilate=True))
        sdf = get_sdf(gt_centerline[None, ...])[0]  # Expect batch size dimension
        sdf *= voxel_spacing
        return sdf[img.bool()].mean()

    seg, header = torch_from_nii("/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/experiment/TwoStep/Test/seg/pt_012.nii")
    seg = seg.int()
    #
    gt_center, _ = torch_from_nii("/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/Centerlines/labelsTr/pt_012.nii")
    #
    graph = seg_to_graph(seg)
    score_graph(graph, gt_seg=seg)
    graph = post_process_graph(graph, end_threshold=end_thresh, distance_threshold=dist_thresh)
    sdf_post, gc_degs, gc_cc = score_graph(graph, gt_seg=seg)
    # expl_mat = print_sdf_stats(sdf_post, gve_dist_thresh)
    #
    # # save_nii(graph_to_image(graph, dilate=True), header, f"../../Data/experiment/TwoStep/graph/pt_012.nii")
    # print(calc_dist(graph, gt_center))
    # exit(1)

    mat = []
    graph_center = []

    seg_dir = "/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/Segmentation/labelsTr"
    center_dir = "/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/Centerlines/labelsTr"

    for PATIENT in tqdm(os.listdir(center_dir)):
        if not PATIENT.endswith(".nii") or "013" not in PATIENT:
            continue

        gt_center, _ = torch_from_nii(f"{center_dir}/{PATIENT}")
        seg, header = torch_from_nii(f"{seg_dir}/{PATIENT}")

        graph = seg_to_graph(seg)
        sdf_og, mat_degs, mat_cc = score_graph(graph, gt_seg=seg)
        mat.append(calc_dist(graph, gt_center))

        graph = post_process_graph(graph, end_threshold=end_thresh, distance_threshold=dist_thresh)
        sdf_post, gc_degs, gc_cc = score_graph(graph, gt_seg=seg)
        graph_center.append(calc_dist(graph, gt_center))
        save_nii(graph_to_image(graph, dilate=True), header, f"results/pt_{PATIENT}_post_processed_check.nii")

        # print("Explainability".center(30, "-"))
        # print(f"Original".rjust(15, "~").ljust(30, "~"))
        # expl_mat = print_sdf_stats(sdf_og, gve_dist_thresh)

        # print(f"Post-processed".rjust(15, "~").ljust(30, "~"))
        # expl_gc = print_sdf_stats(sdf_post, gve_dist_thresh)
        # print("-".center(30, "-"))

        # Not explained
        img = graph_to_image(graph, dilate=True)
        full_sdf = get_sdf(torch.from_numpy(img)[None, ...])[0]
        full_sdf *= 1.25

        # SDF outside of segmentation is not important
        full_sdf[~seg.bool()] = 0
        full_sdf = (full_sdf > gve_dist_thresh).int().numpy()
        save_nii(full_sdf, header, f"results/{PATIENT.replace('.nii', '')}_not_explained.nii")

    print()
    print(np.mean(np.array(mat), axis=0))
    print(np.mean(np.array(graph_center), axis=0))


