"""Deterministic geometry controls for Navigator experiments."""

from itertools import product

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, binary_propagation
from scipy.spatial import cKDTree
from skimage.draw import line_nd
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize


NEIGHBOR_OFFSETS = tuple(offset for offset in product((-1, 0, 1), repeat=3) if any(offset))


def _mask_path(
    mask: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> np.ndarray:
    if start == end:
        return np.asarray([start], dtype=np.int64)
    costs = np.where(mask, 1.0, np.inf)
    solver = MCP_Geometric(costs, fully_connected=True)
    distances, _ = solver.find_costs([start])
    if not np.isfinite(distances[end]):
        raise ValueError(f"No mask-constrained path between {start} and {end}.")
    return np.asarray(solver.traceback(end), dtype=np.int64)


def _skeleton_graph(skeleton: np.ndarray) -> tuple[nx.Graph, np.ndarray]:
    coordinates = np.argwhere(skeleton)
    if not len(coordinates):
        raise ValueError("The segmentation produced an empty medial skeleton.")
    node_by_coordinate = {tuple(coordinate): index for index, coordinate in enumerate(coordinates)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(coordinates)))
    for node, coordinate in enumerate(coordinates):
        for offset in NEIGHBOR_OFFSETS:
            neighbor_coordinate = tuple(coordinate + np.asarray(offset))
            neighbor = node_by_coordinate.get(neighbor_coordinate)
            if neighbor is not None and neighbor > node:
                graph.add_edge(node, neighbor, weight=float(np.linalg.norm(offset)))
    return graph, coordinates


def _append_closed_subtree_walk(
    tree: nx.Graph,
    branch: int,
    parent: int,
    main_path_nodes: set[int],
    route: list[int],
) -> None:
    """Visit an off-path subtree and return to its attachment node."""

    route.append(branch)
    branch_children = tuple(
        neighbor
        for neighbor in tree.neighbors(branch)
        if neighbor != parent and neighbor not in main_path_nodes
    )
    stack = [
        (
            branch,
            parent,
            iter(branch_children),
        )
    ]
    while stack:
        node, node_parent, children = stack[-1]
        try:
            child = next(children)
        except StopIteration:
            stack.pop()
            route.append(node_parent)
            continue
        route.append(child)
        child_neighbors = tuple(
            neighbor
            for neighbor in tree.neighbors(child)
            if neighbor != node and neighbor not in main_path_nodes
        )
        stack.append(
            (
                child,
                node,
                iter(child_neighbors),
            )
        )


def skeleton_covering_route(
    segmentation_mask: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> np.ndarray:
    """Return a continuous start-to-end walk covering the medial-skeleton tree.

    A minimum spanning tree removes cycles caused by thick or locally ambiguous
    skeleton junctions. Every off-path branch is visited and backtracked before
    the walk advances along the unique tree path to the anatomical endpoint.
    """

    mask = np.asarray(segmentation_mask, dtype=bool)
    start = tuple(int(value) for value in start)
    end = tuple(int(value) for value in end)
    if not mask[start] or not mask[end]:
        raise ValueError("Skeleton route endpoints must be inside the segmentation.")

    component_seed = np.zeros_like(mask, dtype=bool)
    component_seed[start] = True
    component_mask = binary_propagation(
        component_seed,
        structure=np.ones((3, 3, 3), dtype=bool),
        mask=mask,
    )
    if not component_mask[end]:
        raise ValueError(f"No mask-constrained path between {start} and {end}.")

    # Nearby disconnected fragments can be closer to an endpoint than the
    # medial skeleton of its actual (thick) bowel component. Skeletonizing only
    # the start/end component prevents the expert from attaching to a spatially
    # close but topologically unreachable fragment.
    skeleton = skeletonize(component_mask, method="lee")
    if not np.any(skeleton):
        return _mask_path(component_mask, start, end)
    graph, coordinates = _skeleton_graph(skeleton)
    coordinate_tree = cKDTree(coordinates)
    start_node = int(coordinate_tree.query(start)[1])
    start_component = nx.node_connected_component(graph, start_node)
    component_nodes = np.fromiter(start_component, dtype=np.int64)
    component_tree = cKDTree(coordinates[component_nodes])
    end_node = int(component_nodes[int(component_tree.query(end)[1])])

    component_graph = graph.subgraph(start_component)
    tree = nx.minimum_spanning_tree(component_graph, weight="weight")
    main_path = nx.shortest_path(tree, start_node, end_node, weight="weight")
    main_path_nodes = set(main_path)
    skeleton_route = [start_node]
    for path_index, node in enumerate(main_path):
        previous_node = main_path[path_index - 1] if path_index else None
        next_node = main_path[path_index + 1] if path_index + 1 < len(main_path) else None
        for neighbor in tree.neighbors(node):
            if neighbor in (previous_node, next_node) or neighbor in main_path_nodes:
                continue
            _append_closed_subtree_walk(
                tree,
                neighbor,
                node,
                main_path_nodes,
                skeleton_route,
            )
        if next_node is not None:
            skeleton_route.append(next_node)

    skeleton_coordinates = coordinates[np.asarray(skeleton_route)]
    start_connector = _mask_path(
        component_mask,
        start,
        tuple(skeleton_coordinates[0]),
    )
    end_connector = _mask_path(
        component_mask,
        tuple(skeleton_coordinates[-1]),
        end,
    )
    route = np.concatenate(
        (
            start_connector,
            skeleton_coordinates[1:],
            end_connector[1:],
        ),
        axis=0,
    )
    keep = np.concatenate(([True], np.any(route[1:] != route[:-1], axis=1)))
    return route[keep]


def compress_route_with_action_support(
    segmentation_mask: np.ndarray,
    route: np.ndarray,
    action_displacements: tuple[tuple[int, int, int], ...],
    max_route_lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily compress a dense route into exact supported, on-mask actions.

    Lookahead is bounded in route order so a spatially nearby folded loop
    cannot be selected as a shortcut. The returned indices map every
    compressed waypoint back to the original dense route.
    """

    if max_route_lookahead < 1:
        raise ValueError("max_route_lookahead must be positive")
    mask = np.asarray(segmentation_mask, dtype=bool)
    dense_route = np.asarray(route, dtype=np.int64).reshape(-1, 3)
    if not len(dense_route):
        raise ValueError("Cannot compress an empty route")
    shape = np.asarray(mask.shape, dtype=np.int64)
    if np.any(dense_route < 0) or np.any(dense_route >= shape):
        raise ValueError("Route contains a point outside the segmentation volume")
    if not np.asarray(mask[tuple(dense_route.T)]).all():
        raise ValueError("Route contains a point outside the segmentation mask")

    support = {
        tuple(int(component) for component in displacement)
        for displacement in action_displacements
        if any(displacement)
    }
    if not support and len(dense_route) > 1:
        raise ValueError("Action support contains no nonzero displacement")

    selected_indices = [0]
    current_index = 0
    while current_index < len(dense_route) - 1:
        final_candidate = min(
            len(dense_route) - 1,
            current_index + max_route_lookahead,
        )
        selected_index = None
        for candidate_index in range(final_candidate, current_index, -1):
            displacement = tuple(
                int(value)
                for value in (
                    dense_route[candidate_index] - dense_route[current_index]
                )
            )
            if displacement not in support:
                continue
            segment = line_nd(
                tuple(dense_route[current_index]),
                tuple(dense_route[candidate_index]),
                endpoint=True,
            )
            if np.asarray(mask[segment]).all():
                selected_index = candidate_index
                break
        if selected_index is None:
            required = tuple(
                int(value)
                for value in (
                    dense_route[current_index + 1] - dense_route[current_index]
                )
            )
            raise ValueError(
                "Action support cannot execute the next dense-route step "
                f"{required} at index {current_index}"
            )
        selected_indices.append(selected_index)
        current_index = selected_index

    indices = np.asarray(selected_indices, dtype=np.int64)
    return dense_route[indices], indices


def path_dice(
    segmentation_mask: np.ndarray,
    route: np.ndarray,
    radius_vox: int,
) -> float:
    """Return exact Dice for an L1-dilated route and segmentation."""

    target = np.asarray(segmentation_mask, dtype=bool)
    path = np.zeros_like(target, dtype=bool)
    route = np.asarray(route, dtype=np.int64).reshape(-1, 3)
    path[tuple(route.T)] = True
    if radius_vox:
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = True
        for axis in range(3):
            lower = [1, 1, 1]
            upper = [1, 1, 1]
            lower[axis] = 0
            upper[axis] = 2
            structure[tuple(lower)] = True
            structure[tuple(upper)] = True
        path = binary_dilation(path, structure=structure, iterations=radius_vox)
    intersection = np.logical_and(path, target).sum()
    denominator = path.sum() + target.sum()
    return float(2 * intersection / denominator) if denominator else 0.0
