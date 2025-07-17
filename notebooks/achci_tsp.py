from pathlib import Path
import networkx as nx
import numpy as np

import itertools
import time
import json
import pyvista as pv
import argparse
import nibabel as nib
from math import dist
from skimage.draw import line_nd
from scipy.ndimage import binary_dilation

# --- Constants ---
SAMPLE_FILEPATH = "rag2_pruned.json"  # Default filepath

# --- Helper Functions ---


# def euclidean_distance(coord1, coord2):
#     """Calculates Euclidean distance between two 3D points."""
#     return np.linalg.norm(np.array(coord1) - np.array(coord2))

def euclidean_distance(coord1, coord2):
    return dist(coord1, coord2)


def calculate_path_cost(graph, path_nodes):
    """Calculates the total cost of an open path using the 'cost' attribute."""
    cost = 0
    n = len(path_nodes)
    if n < 2:
        return 0
    for i in range(n - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        try:
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v)
                edge_cost = edge_data.get("cost", float("inf"))
                if edge_cost == float("inf"):
                    print(f"Warning: Edge ({u}, {v}) missing 'cost'.")
                cost += edge_cost
            else:
                print(f"Warning: Edge ({u}, {v}) missing.")
                return float("inf")
        except Exception as e:
            print(f"Error accessing edge ({u}, {v}) cost: {e}")
            return float("inf")
    if np.isinf(cost):
        print("Warning: Path cost became infinite.")
    return cost


def load_graph_from_node_link(filepath, directed=False):
    """Loads UNDIRECTED graph, expects 'id', 'centroid', 'cost'."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        G = nx.node_link_graph(data, directed=directed, multigraph=False)
        if not G:
            print(f"Warning: Loaded graph empty: {filepath}")
            return G
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print(f"Loaded UNDIRECTED graph: {num_nodes} nodes, {num_edges} edges.")
        if num_nodes > 0:
            first_node_id = list(G.nodes())[0]
            if "centroid" not in G.nodes[first_node_id]:
                print(f"Warning: Nodes missing 'centroid'.")
        if num_edges > 0:
            u, v, first_edge_data = list(G.edges(data=True))[0]
            if "cost" not in first_edge_data:
                print(f"Warning: Links missing 'cost'.")
        return G
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON decode error: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None


# --- ACHCI Algorithm with KNN Start & Euclidean Distance Constraint ---


def achci_path_centroid(graph, start, end, k_nn=3, max_euclidean_dist=10.0):
    """
    Finds path: KNN start (frozen) + ACHCI insert + Euclidean distance constraint.
    Uses UNDIRECTED graph, node 'centroid', edge 'cost'.

    Args:
        graph (nx.Graph): Undirected NetworkX graph.
        start: Starting node ID.
        end: Ending node ID.
        k_nn (int): Nodes in the initial FROZEN KNN segment.
        max_euclidean_dist (float): Maximum allowed Euclidean distance between connected nodes.

    Returns:
        tuple: (path_nodes, path_cost, computation_time) or (None, inf, 0).
    """
    start_time_total = time.time()

    # --- Input Validation ---
    if not isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
        print("Warning: Graph not nx.Graph.")
    if start not in graph:
        print(f"Error: Start node '{start}' not found.")
        return None, float("inf"), 0
    if end not in graph:
        print(f"Error: End node '{end}' not found.")
        return None, float("inf"), 0
    if not isinstance(k_nn, int) or k_nn < 1:
        print(f"Warning: Invalid k_nn ({k_nn}). Setting k_nn=1.")
        k_nn = 1
    if max_euclidean_dist <= 0:
        print("Warning: max_euclidean_dist must be positive. Constraint inactive.")
        max_euclidean_dist = float("inf")

    nodes = list(graph.nodes())
    n = len(nodes)
    num_dims = 3
    if n == 0:
        print("Error: Graph empty.")
        return [], 0, 0
    if n == 1:
        return [start], 0, 0 if start == end else (None, float("inf"), 0)
    k_nn = min(k_nn, n)  # Clamp k_nn

    # --- 0. Setup ---
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    C = np.full((n, n), float("inf"))
    np.fill_diagonal(C, 0)
    X = np.zeros((n, num_dims))  # Matrix to store centroid coordinates
    print("Extracting centroids and building SYMMETRIC cost matrix...")
    try:
        for node_id, idx in node_to_idx.items():
            centroid = graph.nodes[node_id].get("centroid")
            if centroid is None or len(centroid) != num_dims:
                raise ValueError(f"Node '{node_id}' missing/invalid 'centroid'.")
            X[idx, :] = centroid  # Store centroid
        for u, v, data in graph.edges(data=True):
            cost_val = data.get("cost", float("inf"))
            idx_u, idx_v = node_to_idx[u], node_to_idx[v]
            C[idx_u, idx_v] = C[idx_v, idx_u] = cost_val
    except Exception as e:
        print(f"Error during setup: {e}")
        return None, float("inf"), 0

    # --- 1. Initialization (Conditional KNN vs. Single Closest with Distance Check) ---
    print(f"Initializing path from '{start}' to '{end}'. Max Euclidean dist: {max_euclidean_dist}")
    current_path = [start]
    visited_nodes = {start}
    start_time_knn_achci = time.time()
    intermediate_nodes = set(nodes) - {start, end}
    unvisited_intermediate = set(intermediate_nodes)

    if k_nn == 1:
        print("k_nn=1: Finding single closest intermediate node to start (with dist check).")
        if unvisited_intermediate:
            first_node_to_insert = None
            min_cost_from_start = float("inf")
            start_idx = node_to_idx[start]
            start_coord = X[start_idx]  # Get start coord

            for k_node in unvisited_intermediate:
                k_idx = node_to_idx[k_node]
                cost_val = C[start_idx, k_idx]

                # Check cost AND distance constraint
                k_coord = X[k_idx]
                dist_sk = euclidean_distance(start_coord, k_coord)

                if cost_val < min_cost_from_start and dist_sk <= max_euclidean_dist:
                    min_cost_from_start = cost_val
                    first_node_to_insert = k_node

            if first_node_to_insert is not None:  # min_cost check implies finite cost
                print(
                    f"Inserting first intermediate node '{first_node_to_insert}' (closest valid)."
                )
                current_path.append(first_node_to_insert)
                visited_nodes.add(first_node_to_insert)
            elif unvisited_intermediate:
                print(
                    f"Error: Cannot connect start node '{start}' to any intermediate node satisfying constraints."
                )
                if start == end:
                    return [start], 0, time.time() - start_time_knn_achci
                else:
                    return None, float("inf"), time.time() - start_time_knn_achci

    elif k_nn > 1:
        print(
            f"Initializing FROZEN path segment with {k_nn}-NN starting from '{start}' (with dist check)."
        )
        nodes_to_consider_for_knn = set(nodes) - visited_nodes
        for _ in range(k_nn - 1):
            if not nodes_to_consider_for_knn:
                break
            last_node, last_node_idx = current_path[-1], node_to_idx[current_path[-1]]
            last_coord = X[last_node_idx]  # Get last coord
            next_node, min_cost = None, float("inf")

            nodes_to_check_knn = list(nodes_to_consider_for_knn)
            for potential_next_node in nodes_to_check_knn:
                potential_next_idx = node_to_idx[potential_next_node]
                cost_val = C[last_node_idx, potential_next_idx]

                # Check cost AND distance constraint
                potential_coord = X[potential_next_idx]
                dist_ln = euclidean_distance(last_coord, potential_coord)

                if cost_val < min_cost and dist_ln <= max_euclidean_dist:
                    min_cost = cost_val
                    next_node = potential_next_node

            if next_node is not None:  # min_cost check implies finite cost
                current_path.append(next_node)
                visited_nodes.add(next_node)
                nodes_to_consider_for_knn.remove(next_node)
            else:
                print(
                    f"Warning: KNN phase stopped early at {len(current_path)} nodes (no valid neighbor found)."
                )
                break

    # --- Determine actual_knn_len and remaining unvisited nodes ---
    actual_knn_len = len(current_path)
    print(f"Initial path segment (length {actual_knn_len}): {current_path}")
    unvisited_nodes = set(nodes) - visited_nodes
    if end in unvisited_nodes:
        unvisited_nodes.remove(end)

    # --- 2. Main Cheapest Insertion Loop (Frozen Logic + Distance Check) ---
    print(f"Starting ACHCI insertion for {len(unvisited_nodes)} remaining nodes...")
    epsilon = 1e-9
    iteration_count = 0
    max_iterations = n * n

    while unvisited_nodes and iteration_count < max_iterations:
        iteration_count += 1
        best_k_node, best_insert_pos = None, -1
        min_insertion_metric = float("inf")
        current_unvisited_copy = list(unvisited_nodes)

        for k_node in current_unvisited_copy:
            k_idx = node_to_idx[k_node]
            k_coord = X[k_idx]  # Get coord of node to insert
            path_len = len(current_path)

            for i in range(path_len):  # Check insertion *after* node i
                vi_node = current_path[i]
                vi_idx = node_to_idx[vi_node]
                vi_coord = X[vi_idx]  # Get coord of predecessor

                # Freezing Check
                if (i + 1) < actual_knn_len:
                    continue

                # Determine vj
                if i < path_len - 1:
                    vj_node, vj_idx = current_path[i + 1], node_to_idx[current_path[i + 1]]
                    vj_coord = X[vj_idx]  # Get coord of successor
                else:
                    vj_node, vj_idx = end, node_to_idx[end]
                    vj_coord = X[vj_idx]  # Get coord of end node

                if vi_idx == vj_idx:
                    continue

                # --- Euclidean Distance Constraint Check ---
                dist_ik = euclidean_distance(vi_coord, k_coord)
                dist_kj = euclidean_distance(k_coord, vj_coord)
                if dist_ik > max_euclidean_dist or dist_kj > max_euclidean_dist:
                    # print(f"Skipping insert {k_node} between {vi_node},{vj_node} due to distance: {dist_ik:.1f}, {dist_kj:.1f}") # Debug
                    continue  # Skip if either new segment violates distance

                # --- Cost Calculation & Ratio ---
                Cik = C[vi_idx, k_idx]
                Ckj = C[k_idx, vj_idx]
                Cij = C[vi_idx, vj_idx]
                if np.isinf(Cik) or np.isinf(Ckj) or np.isinf(Cij):
                    continue

                current_metric = float("inf")
                if Cij >= epsilon:
                    current_metric = (Cik + Ckj) / Cij

                # Update Best Insertion
                if current_metric < min_insertion_metric:
                    min_insertion_metric = current_metric
                    best_k_node = k_node
                    best_insert_pos = i + 1

        # Perform Insertion
        if best_k_node is not None and best_insert_pos != -1:
            current_path.insert(best_insert_pos, best_k_node)
            visited_nodes.add(best_k_node)
            unvisited_nodes.remove(best_k_node)
        elif unvisited_nodes:
            print(
                f"Warning: ACHCI could not insert remaining node(s) satisfying constraints: {unvisited_nodes}. Path incomplete."
            )
            break

    if iteration_count >= max_iterations:
        print("Warning: Reached maximum iterations.")

    # --- 3. Finalize Path and Calculate Cost (with Distance Check) ---
    if current_path[-1] != end:
        last_node_in_path = current_path[-1]
        last_node_idx = node_to_idx[last_node_in_path]
        end_idx = node_to_idx[end]

        # Check cost AND distance constraint for final connection
        cost_to_end = C[last_node_idx, end_idx]
        dist_to_end = euclidean_distance(X[last_node_idx], X[end_idx])

        if np.isinf(cost_to_end) or dist_to_end > max_euclidean_dist:
            print(
                f"Error: Cannot connect last node '{last_node_in_path}' to end node '{end}' satisfying constraints (Cost:{cost_to_end}, Dist:{dist_to_end:.2f}). Path incomplete."
            )
            return current_path, float("inf"), time.time() - start_time_knn_achci  # Return partial
        else:
            current_path.append(end)
            visited_nodes.add(end)

    final_path = current_path
    computation_time = time.time() - start_time_knn_achci
    print(f"KNN + ACHCI (Frozen, Dist Constrained) completed in {computation_time:.4f} seconds.")

    # Final check for completeness
    expected_node_count = n
    final_visited_count = len(set(final_path))
    if final_visited_count != expected_node_count:
        missing_nodes = set(nodes) - set(final_path)
        print(
            f"ERROR: Final path is incomplete! Visited {final_visited_count}/{expected_node_count} unique nodes."
        )
        if missing_nodes:
            print(f"   Missing nodes: {missing_nodes}")
        print(f"   Path Found: {final_path}")
    elif len(final_path) != expected_node_count and not (start == end and n > 1):
        print(
            f"Warning: Path length ({len(final_path)}) != node count ({expected_node_count}). Check logic."
        )

    final_path_cost = calculate_path_cost(graph, final_path)
    if np.isinf(final_path_cost):
        print("Warning: Final path cost is infinite.")

    return final_path, final_path_cost, computation_time


# --- PyVista Visualization (No changes needed) ---
def visualize_path_pyvista(path_nodes, graph, volume_path: str = None, title="Path Visualization"):
    """Visualizes the open path in 3D using PyVista."""
    # (Keep this function exactly as it was)
    if not path_nodes:
        print("Cannot visualize: Path is empty.")
        return
    if not graph:
        print("Cannot visualize: Graph is invalid.")
        return
    print("Preparing PyVista visualization for the path...")
    points_xyz_list = []
    node_ids_with_coords = []
    node_id_to_coord_idx = {}
    for i, nid in enumerate(graph.nodes()):
        centroid = graph.nodes[nid].get("centroid")
        if centroid is not None and len(centroid) == 3:
            points_xyz_list.append(centroid)
            node_ids_with_coords.append(nid)
            node_id_to_coord_idx[nid] = len(points_xyz_list) - 1
        else:
            print(f"Warning: Node '{nid}' skipped in plot (missing/invalid centroid).")
    if not points_xyz_list:
        print("Cannot visualize: No valid centroids found.")
        return
    points_xyz = np.array(points_xyz_list)
    path_point_indices = []
    for node_id in path_nodes:
        if node_id in node_id_to_coord_idx:
            path_point_indices.append(node_id_to_coord_idx[node_id])
        else:
            print(f"Warning: Node '{node_id}' from path not plotted (no valid centroid).")
    path_polydata = None
    if len(path_point_indices) >= 2:
        path_points = points_xyz[path_point_indices]
        num_path_points = len(path_points)
        lines = np.insert(np.arange(num_path_points, dtype=int), 0, num_path_points)
        path_polydata = pv.PolyData(path_points, lines=lines)
    else:
        print("Warning: Not enough valid points in the path to draw lines.")

    volume_data = None  # Initialize volume_data
    if volume_path:
        try:
            volume = nib.load(volume_path)
            volume_data = volume.get_fdata().astype(np.uint8)  # Load volume data
        except Exception as e:
            print(f"Error loading volume data from {volume_path}: {e}")
            # Decide if you want to proceed without volume or stop
            # volume_path = None # Option: Proceed without volume

    try:
        plotter = pv.Plotter(window_size=[800, 600])
        plotter.background_color = "white"
        if volume_data is not None:  # Check if volume data was loaded successfully
            plotter.add_volume(volume_data * 20, cmap="viridis", opacity="linear")

        plotter.add_points(
            points_xyz,
            render_points_as_spheres=True,
            point_size=10,
            color="cornflowerblue",
            label="All Nodes",
        )
        plotter.add_point_labels(
            points_xyz,
            [str(nid) for nid in node_ids_with_coords],
            point_size=20,
            font_size=10,
            text_color="black",
            shape=None,
            show_points=False,
        )
        start_idx = node_id_to_coord_idx.get(path_nodes[0])
        end_idx = node_id_to_coord_idx.get(path_nodes[-1])
        if start_idx is not None:
            plotter.add_points(
                points_xyz[start_idx],
                color="green",
                point_size=15,
                render_points_as_spheres=True,
                label="Start Node",
            )
        if end_idx is not None and end_idx != start_idx:
            plotter.add_points(
                points_xyz[end_idx],
                color="orange",
                point_size=15,
                render_points_as_spheres=True,
                label="End Node",
            )
        if path_polydata:
            plotter.add_mesh(path_polydata, color="red", line_width=4, label="Calculated Path")
        plotter.add_legend(bcolor=None, face=None)
        plotter.title = title
        plotter.show_axes()
        plotter.camera_position = "xz"
        print("Displaying PyVista plot...")
        plotter.export_html("achci_path_visualization.html")
        plotter.show()
        print("Plot window closed.")
    except Exception as e:
        print(f"Error during PyVista plotting: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find path: Frozen KNN start + ACHCI insert + Dist Constraint."
    )

    parser.add_argument(
        "--graph",
        default=SAMPLE_FILEPATH,
        type=Path,
        help=f"Path to node-link JSON (default: {SAMPLE_FILEPATH})",
    )
    parser.add_argument(
        "--reference_volume", default=None, help="Path to reference_volume volume (NIfTI format)."
    )
    parser.add_argument(
        "--save_dir", type=Path, help="Directory to save the TSPLIB file and solution."
    )
    parser.add_argument("--start", help="ID of the starting node.", type=int)
    parser.add_argument("--end", help="ID of the ending node.", type=int)
    parser.add_argument(
        "-k", "--knn", type=int, default=3, help="Nodes in initial FROZEN KNN segment (default: 3)."
    )
    parser.add_argument(
        "-d",
        "--max_dist",
        type=float,
        default=100.0,
        help="Max allowed Euclidean distance for path segments (default: 10.0).",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or args.graph.parent / "achci_tsp"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading UNDIRECTED graph from: {args.graph}")
    G_loaded = load_graph_from_node_link(args.graph, directed=False)

    if G_loaded:
        print(
            f"\nRunning Algorithm (Frozen K={args.knn} NN + ACHCI + MaxDist={args.max_dist}) from '{args.start}' to '{args.end}'..."
        )
        achci_path, achci_cost, computation_t = achci_path_centroid(
            G_loaded,
            args.start,
            args.end,
            k_nn=args.knn,
            max_euclidean_dist=args.max_dist,
        )
        
        np.savetxt(save_dir / "tour.txt", achci_path, fmt="%d")
        
        tour = [G_loaded.nodes[node]["centroid"] for node in achci_path]
        tour = np.vstack(tour)
        np.savetxt(save_dir / "tracking_history.npy", tour, fmt="%d")
        
        ref = nib.load(args.reference_volume)
        ref_data = ref.get_fdata()
        save_vol = np.zeros(ref_data.shape, dtype=np.uint8)
        all_lines = []
        for start, end in zip(tour[:-1], tour[1:]):
            all_lines.append(line_nd(start, end))
            
        all_lines = np.concatenate(all_lines, axis=1).T
        save_vol[tuple(all_lines.T)] = 1
        save_vol = binary_dilation(save_vol, iterations=3).astype(np.uint8)
        nib.save(nib.Nifti1Image(save_vol, ref.affine), save_dir / "cumulative_path_mask.nii.gz")

        if achci_path is not None:
            print("\n--- Path Results ---")
            print(f"Path: {achci_path}")
            print(f"Cost: {achci_cost}")
            print(f"Computation Time (KNN+ACHCI): {computation_t:.4f} s")
            expected_nodes = set(G_loaded.nodes())
            found_nodes = set(achci_path)
            if found_nodes != expected_nodes:
                print(f"\nWARNING: Path incomplete. Missing: {expected_nodes - found_nodes}")
            else:
                print("\nPath includes all nodes.")
            # visualize_path_pyvista(
            #     achci_path,
            #     G_loaded,
            #     volume_path=args.reference_volume,
            #     title=f"Path (K={args.knn}, MaxDist={args.max_dist}): {args.start} to {args.end}",
            # )
        else:
            print("\nAlgorithm failed to produce a valid path satisfying constraints.")
    else:
        print(f"\nCould not run algorithm: graph loading failed for {args.filepath}.")
    print("\nScript finished.")
