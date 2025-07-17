from pathlib import Path
import subprocess
import numpy as np
from io import StringIO
import networkx as nx
import argparse
import json
from scipy.ndimage import binary_dilation
from skimage.draw import line_nd
import nibabel as nib


def graph_to_tsplib(
    graph,
    weight="cost",
    rounding_factor=1e4,
    graph_name="Untitled",
    comment="No comment",
):
    tsplib_str = "NAME : {}\nTYPE : TSP\nCOMMENT : {}\nDIMENSION: {}\nEDGE_WEIGHT_TYPE :  EXPLICIT\nEDGE_WEIGHT_FORMAT :  FULL_MATRIX\nEDGE_WEIGHT_SECTION\n".format(
        graph_name,
        comment,
        len(graph.nodes),
    )
    buffer = StringIO()
    matrix = (
        (nx.adjacency_matrix(graph, weight=weight).toarray() * rounding_factor).round().astype(int)
    )
    np.savetxt(buffer, matrix, fmt="%d")
    tsplib_str += buffer.getvalue()
    tsplib_str += "EOF"
    return tsplib_str


def solve_tsp(graph, save_dir: Path, start=None, end=None, rounding_factor=1e4, **kwargs):
    """
    Solve the Traveling Salesman Problem (TSP) using the Concorde TSP solver.
    Args:
        graph (networkx.Graph): The input graph representing the TSP.
        save_dir (Path): Directory to save the TSPLIB file and solution.
        **kwargs: Additional arguments for `graph_to_tsplib`.
    Returns:
        np.ndarray: The tour as a numpy array of shape (N, 3), where N is the number of nodes in the tour.
    Raises:
        RuntimeError: If Concorde fails to solve the TSP.

    NOTE: The concorde solver must be installed and available in the system PATH.
    """

    if start is not None and end is not None:
        # Add dummy node
        graph.add_node("dummy", centroid=(100, 100, 100))
        graph.add_edge("dummy", start, cost=0)
        graph.add_edge("dummy", end, cost=0)
        for node in graph.nodes:
            if node not in (start, end, "dummy"):
                graph.add_edge("dummy", node, cost=2**30 / rounding_factor)

    tsplib_str = graph_to_tsplib(graph, rounding_factor=rounding_factor, **kwargs)
    with open(save_dir / "matrix_representation.tsp", "w") as f:
        f.write(tsplib_str)
    result = subprocess.run(
        [
            "concorde",
            "-o",
            str(save_dir / "small_bowel_tsp.tour"),
            str(save_dir / "matrix_representation.tsp"),
        ],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Concorde failed with error: {result.stderr}")

    with open(save_dir / "small_bowel_tsp.tour") as f:
        solution = f.read().strip().split("\n")[1:]
        solution = [int(num) for row in solution for num in row.split() if num != "dummy"]

    if start is not None and end is not None:
        graph.remove_node("dummy")
    idx_to_node = {i: node for i, node in enumerate(graph.nodes)}
    node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
    if start is not None and end is not None:
        # Remove dummy node from the solution
        dummy_idx = solution.index(len(graph.nodes))
        # If the index of the next node is the index of the start node, then this will be the start and the indexes before it will be the end (and will be moved back)
        if solution[dummy_idx + 1] == node_to_idx[start]:
            solution = solution[dummy_idx + 1 :][::-1] + solution[: dummy_idx - 1][::-1]
        elif solution[dummy_idx + 1] == node_to_idx[end]:
            # Need to move the end node to the end of the solution
            solution = solution[:dummy_idx][::-1] + solution[dummy_idx + 1 :][::-1]

    tour = [idx_to_node[i] for i in solution]
    tour = np.row_stack([graph.nodes[node]["centroid"] for node in tour])
    return tour


def main():
    parser = argparse.ArgumentParser(description="Solve TSP using Concorde.")
    parser.add_argument(
        "--graph", type=Path, required=True, help="Path to the graph file in TSPLIB format."
    )
    parser.add_argument(
        "--reference_volume", type=Path, required=True, help="Path to the reference volume file."
    )
    parser.add_argument(
        "--save_dir", type=Path, help="Directory to save the TSPLIB file and solution."
    )
    parser.add_argument("--weight", type=str, default="cost", help="Edge weight attribute to use.")
    parser.add_argument(
        "--rounding_factor", type=float, default=1e6, help="Rounding factor for edge weights."
    )

    parser.add_argument(
        "--comment", type=str, default="No comment", help="Comment for the TSPLIB file."
    )
    parser.add_argument("--start", type=int, help="Starting node index for the TSP tour.")
    parser.add_argument("--end", type=int, help="Ending node index for the TSP tour.")

    args = parser.parse_args()

    save_dir = args.save_dir or args.graph.parent / "concorde_tsp"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(args.graph, "r") as f:
        graph = nx.node_link_graph(json.load(f))
    tour = solve_tsp(
        graph,
        save_dir=save_dir,
        weight=args.weight,
        rounding_factor=args.rounding_factor,
        comment=args.comment,
        graph_name=str(args.graph),
        start=args.start,
        end=args.end,
    )

    np.savetxt(save_dir / "tracking_history.npy", tour, fmt="%d")

    ref = nib.load(args.reference_volume)
    ref_data = ref.get_fdata()
    save_vol = np.zeros_like(ref_data, dtype=np.uint8)
    save_vol[tuple(tour.T)] = 1
    all_lines = []
    for start, end in zip(tour[:-1], tour[1:]):
        all_lines.append(line_nd(start, end))
    all_lines = np.concatenate(all_lines, axis=1).T
    save_vol[tuple(all_lines.T)] = 1
    save_vol = binary_dilation(save_vol, iterations=3).astype(np.uint8)
    nib.save(nib.Nifti1Image(save_vol, ref.affine), save_dir / "cumulative_path_mask.nii.gz")


if __name__ == "__main__":
    main()
