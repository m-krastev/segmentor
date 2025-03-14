# ## A Graph-theoretic Algorithm for Small Bowel Path Tracking in CT Scans
# Seung Yeon Shin, Sungwon Lee, Ronald M. Summers;
# Reproduction by Matey Krastev
#
# ----
#
# ![image.png](attachment:image.png)
# We are given the following algorithm. It assumes inputs of a 3D CT scan and its corresponding 3D binary segmentation mask (of the small bowel), and outputs a path through the small bowel.
#
# It works as follows:
# 1. Compute edge map of the CT scan using a Meijering filter. Here, low values correspond to flat (no-gradient) regions, and high values correspond to (high-gradient) edge regions. The edge map is computed on 3D space and assumes a grayscale image.
# 2. Compute supervoxels of the edge map using a SLIC algorithm.
# 3. Compute a graph from the supervoxels. The graph is undirected and weighted. The weight of an edge is the mean edge/gradient magnitude of the supervoxels it connects, more concretely, along their border. We assume that edges with low or zero weights will be a part of the small bowel since there is no gradient change.
# 4. Filter the graph in several ways:
#     - Remove nodes lying outside the segmentation mask. This is done by checking if any portion of the supervoxel lies inside the mask.
#
# 5. As the graph will likely be big, it will be computationally infeasible to solve the Travelling Salesman Problem (TSP) on it. We sample must-pass nodes:
#    1. Compute the Eucliean distance transform of the segmentation mask. This will give us a distance map where each pixel value is the distance to the nearest boundary pixel.
#    2. There are two hyperparameters here: $\theta_d$ and $\theta_v$ which are the minimum voxel distance between voxels and the minimum gradient magnitude, respectively. We sample nodes that are at least $\theta_d$ apart and have a gradient magnitude of at least $\theta_v$.
#
# 6. For the TSP problem:
#    1.  Select start and end nodes (we can take, for example, the duodenum's end and the ileum's end, that is, the start of the colon). Add a dummy node which connects to all nodes, but all its edges, except for those connecting to the start and end nodes, have infinite cost.
#    2.  Solve the TSP problem using the Co.
#

import logging
import os
import pickle
from pathlib import Path
from itertools import combinations
import argparse
import json  # ADDED
from dataclasses import dataclass, field, replace  # MODIFIED
from typing import List, Optional, Tuple, Dict, Any  # MODIFIED

import networkx as nx
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pyvista as pv
import skimage as ski
from skimage.filters import meijering
from skimage.segmentation import slic
from skimage.feature import peak_local_max
from scipy.ndimage import binary_dilation, distance_transform_edt
from sklearn.metrics import euclidean_distances as edist

from segmentor.utils.medutils import (
    load_and_normalize_nifti,
    load_and_resample_nifti,
    save_nifti,
    load_nifti,
)
import kimimaro
import edt
import rustworkx as rx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("trame_server").setLevel(logging.ERROR)
logging.getLogger("trame_server.controller").setLevel(logging.ERROR)
logging.getLogger("trame_client").setLevel(logging.ERROR)
logging.getLogger("trame_client.widgets.core").setLevel(logging.ERROR)


class NumpyEncoder(json.JSONEncoder):
    """Special class to serialize numpy types such as int32, int16, float32, etc."""

    def default(self, obj):
        if isinstance(obj, (np.int32, np.int64, np.int16)):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, rx.AllPairsPathLengthMapping):
            return {k: dict(v) for k, v in obj.items()}
        return json.JSONEncoder.default(self, obj)


@dataclass
class Config:
    supervoxel_size: int = 216
    sigmas: List[float] = field(default_factory=lambda: [1.0])
    edge_threshold: float = 0.15
    black_ridges: bool = False
    dilation_iterations: int = 3
    thetav: int = 3
    thetad: int = 6
    delta: int = 500
    start_node: Optional[int] = None
    end_node: Optional[int] = None
    start_volume: Optional[str] = None
    end_volume: Optional[str] = None
    precompute: bool = False
    use_rustworkx: bool = False

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "Config":
        """Read hyperparameters from a JSON configuration file."""
        with open(path, "r") as f:
            config = json.load(f)
        return Config(**config)

    @classmethod
    def from_dict(cls, config: dict) -> "Config":
        """Read hyperparameters from a dictionary configuration."""
        return Config(**config)

    def to_json(self, path: str):
        """Save hyperparameters to a JSON configuration file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        """Return hyperparameters as a dictionary."""
        return self.__dict__


class SmallBowelSegmentor:
    def __init__(
        self,
        filename_ct: str,
        filename_gt: str,
        config: Config,
        label: Optional[int] = None,
    ):
        self.filename_ct: str = filename_ct
        self.filename_gt: str = filename_gt
        self.config: Config = config
        self.supervoxel_size: int = config.supervoxel_size
        self.cache_path: Path = Path("cache")
        self.cache_path.mkdir(exist_ok=True)
        self.edges_cache: Path = self.cache_path / "edges.npy"
        self.segments_cache: Path = self.cache_path / "segments.npy"
        self.rag_cache: Path = self.cache_path / "rag.json"
        self.label: Optional[int] = (
            label if label is not None else (1 if "nnUNet" in filename_gt else 18)
        )
        self.ground_truth: np.ndarray = (
            np.asarray(nib.load(self.filename_gt).dataobj) == self.label
        ).astype(np.uint8)
        self.nii: np.ndarray = load_and_normalize_nifti(self.filename_ct)
        self.edges: Optional[np.ndarray] = None
        self.segments: Optional[np.ndarray] = None
        self.rag: Optional[nx.Graph] = None
        self.peak_idxs: Optional[np.ndarray] = None
        self.pcoordinates: Optional[np.ndarray] = None
        self.idx_to_peak: Optional[Dict[int, int]] = None
        self.peak_to_idx: Optional[Dict[int, int]] = None

    def compute_supervoxels(self) -> int:
        """Compute number of supervoxels needed for desired supervoxel size"""
        voxel_size: int = 1 * 1 * 1  # mm3
        num_voxels: int = np.prod(self.ground_truth.shape)
        num_supervoxels: int = int(num_voxels * voxel_size / self.supervoxel_size)
        logging.info(
            f"Desired supervoxel size: {self.supervoxel_size:<4} mm3; Assumed voxel size: {voxel_size:<4} mm3; Number of supervoxels required:{num_supervoxels:<4}"
        )
        return num_supervoxels

    def compute_edges(self) -> np.ndarray:
        """Compute edge map using Meijering filter."""
        if self.edges_cache.exists():
            self.edges = np.load(self.edges_cache)
            logging.info("Edges loaded")
        else:
            logging.info("Edge map not found. Generating...")
            self.edges = meijering(
                1 - self.nii, sigmas=self.config.sigmas, black_ridges=self.config.black_ridges
            ).astype(np.float32)
            self.edges = ski.filters.median(self.edges)
            self.edges = ski.filters.gaussian(self.edges, sigma=3)
            np.save(self.edges_cache, self.edges)
            logging.info("Edges generated")
        return self.edges

    def compute_segments(self, num_supervoxels: int) -> np.ndarray:
        """Compute supervoxels using SLIC algorithm."""
        if self.segments_cache.exists():
            self.segments = np.load(self.segments_cache)
            logging.info(f"{np.max(self.segments)} segments loaded")
        else:
            self.edges = np.where(self.edges > self.config.edge_threshold, self.edges, 0)
            self.segments = slic(
                self.edges,
                n_segments=num_supervoxels,
                compactness=0.01,
                start_label=1,
                channel_axis=None,
                sigma=0,
            ).astype(np.uint16 if num_supervoxels < 2**16 else np.uint32)
            np.save(self.segments_cache, self.segments)
            logging.info(f"{np.max(self.segments)} segments generated")
        return self.segments

    def generate_rag(self) -> nx.Graph:
        """Generate the Region Adjacency Graph (RAG)."""
        if self.rag_cache.exists():
            with open(self.rag_cache, "r") as f:
                self.rag = nx.node_link_graph(json.load(f))
                logging.info("RAG loaded")
        else:
            self.rag = ski.graph.rag_boundary(self.segments, self.edges)
            with open(self.rag_cache, "w") as f:
                json.dump(nx.node_link_data(self.rag), f, cls=NumpyEncoder)
            logging.info("RAG saved")
        return self.rag

    def compute_distance_map(self) -> np.ndarray:
        """Compute the distance map from the inverted small bowel segmentation."""
        return edt.edt(
            # binary_dilation(
            #     self.edges > self.config.edge_threshold, iterations=self.config.dilation_iterations, mask=self.ground_truth
            # )
            self.ground_truth.astype(bool)
        )

    def compute_peaks(self, distance_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the must-pass nodes as local peaks on the distance map."""
        self.peak_idxs = peak_local_max(
            distance_map,
            min_distance=self.config.thetad,
            threshold_abs=self.config.thetav,
            labels=self.segments,
        )  # labels = segments is probably needed here, otherwise the points are too sparse
        peaks = np.unique(
            self.segments[self.peak_idxs[:, 0], self.peak_idxs[:, 1], self.peak_idxs[:, 2]]
        )
        logging.info(f"Found {len(peaks)} peaks")
        return self.peak_idxs, peaks

    def find_start_end(self, start_volume, end_volume):
        peaks = list(self.peak_to_idx.keys())
        start_segments = np.intersect1d(peaks, self.segments[start_volume])
        end_segments = np.intersect1d(peaks, self.segments[end_volume])
        # Let start be the first peak
        start = start_segments[0]
        end = end_segments
        # Let end be the lowest peak
        # end = np.isin(segments, end)
        # end = end[np.max(end[:, 2])]
        # end = segments[end[0], end[1], end[2]]

        def get_ileocecal_end(colon: np.ndarray) -> np.ndarray:
            colon_skeleton = kimimaro.skeletonize(
                colon,
                teasar_params={
                    "scale": 3,
                    "const": 5,
                    "pdrf_scale": 10000,
                    "pdrf_exponent": 4,
                    "soma_acceptance_threshold": 3500,  # physical units
                    # "soma_detection_threshold": 750,  # physical units
                    # "soma_invalidation_const": 300,  # physical units
                    # "soma_invalidation_scale": 2,
                },
                anisotropy=(1, 1, 1),
                dust_threshold=5,
                fix_branching=True,
                progress=False,
                parallel_chunk_size=100,  # for the progress bar
            )[1]
            colon_graph = nx.Graph()
            colon_graph.add_nodes_from(
                (idx, {"location": loc}) for idx, loc in enumerate(colon_skeleton.vertices)
            )
            colon_graph.add_edges_from(colon_skeleton.edges)

            ends = list(node[0] for node in colon_graph.degree() if node[1] == 1)
            ends
            rectal_end = max(ends, key=lambda x: colon_graph.nodes[x]["location"][2])
            ends.remove(rectal_end)
            if len(ends) > 1:
                # The ileocecal end is the one farthest away from the rectal end
                # This can be done by finding the end that is the farthest away in terms of graph distance
                ileocecal_end = max(
                    ends, key=lambda x: nx.shortest_path_length(colon_graph, rectal_end, x)
                )
                # Alternatively, find the end that is the farthest away in terms of Euclidean distance
                # ileocecal_end = max(
                #     ends,
                #     key=lambda x: np.linalg.norm(
                #         colon_graph.nodes[x]["location"] - colon_graph.nodes[rectal_end]["location"]
                #     ),
                # )
            else:
                ileocecal_end = ends[0]

            return colon_graph.nodes[ileocecal_end]["location"].astype(int), list(
                map(lambda end: colon_graph.nodes[end]["location"].astype(int), ends)
            )

        ileocecal_end_coords, _ = get_ileocecal_end(end_volume)
        end = self.segments[
            ileocecal_end_coords[0], ileocecal_end_coords[1], ileocecal_end_coords[2]
        ]
        if end not in peaks:
            # Find the closest peak to the ileocecal end
            end_idx = np.argmin(
                np.linalg.norm(
                    self.peak_idxs - ileocecal_end_coords,
                    axis=1,
                )
            )
            end = self.segments[
                self.peak_idxs[end_idx][0], self.peak_idxs[end_idx][1], self.peak_idxs[end_idx][2]
            ]
        return start, end

    def simplify_graph(self, peaks: np.ndarray) -> nx.Graph:
        """Simplify the graph."""
        rag2: nx.Graph = nx.Graph()
        rag2.add_nodes_from(peaks)
        nodes: List[int] = list(rag2.nodes)

        self.idx_to_peak = {
            idx: peak
            for idx, peak in enumerate(
                self.segments[self.peak_idxs[:, 0], self.peak_idxs[:, 1], self.peak_idxs[:, 2]]
            )
        }
        self.peak_to_idx = {
            peak: idx for idx, peak in self.idx_to_peak.items()
        }  # NOTE: Not bijective

        self.pcoordinates: np.ndarray = np.asarray(self.peak_idxs)
        euclid: np.ndarray = edist(self.pcoordinates)

        allpairs = combinations(nodes, 2)

        # Check for cache of precomputed distances
        if (self.cache_path / "lengths.json").exists():
            logging.info("Cache for Dijkstra distances found. Loading precomputed distances...")
            with open(self.cache_path / "lengths.pkl", "rb") as f:
                lengths = pickle.load(f)
            self.config.precompute = True

        else:
            if self.config.precompute:
                if self.config.use_rustworkx:
                    import rustworkx as rx

                    ragx = rx.networkx_converter(self.rag)
                    lengths = rx.all_pairs_dijkstra_path_lengths(
                        ragx, edge_cost_fn=lambda edge: edge["weight"]
                    )
                else:
                    lengths = dict(nx.all_pairs_dijkstra_path_length(self.rag, weight="cost"))
                    # Save lengths to cache
                    logging.info("Precomputed distances saved to cache")
                    with open(self.cache_path / "lengths.pkl", "wb") as f:
                        pickle.dump(lengths, f)

        dists: List[float] = []
        for node1, node2 in tqdm(
            allpairs, desc="Computing distances", total=len(nodes) * (len(nodes) - 1) // 2
        ):
            dist: float = euclid[self.peak_to_idx[node1], self.peak_to_idx[node2]]
            if dist <= self.config.delta:
                if self.config.precompute:
                    dijkstra_length = lengths[node1][node2]
                else:
                    dijkstra_length = nx.dijkstra_path_length(self.rag, node1, node2, weight="cost")
                rag2.add_edge(node1, node2, cost=dist, normalized=False)
                dists.append(dijkstra_length)
            else:
                rag2.add_edge(node1, node2, cost=dist / self.config.delta, normalized=True)

        maxdist: float = max(dists)
        # Second pass to normalize by the max
        for edge in rag2.edges:
            if not rag2.edges[edge]["normalized"]:
                rag2.edges[edge]["cost"] /= maxdist
                rag2.edges[edge]["normalized"] = True
        return rag2

    def solve_tsp(self, rag2: nx.Graph) -> Tuple[List[int], int, int, float]:
        """Solve the Traveling Salesman Problem (TSP)."""
        start_volume = load_nifti(self.config.start_volume) == 1
        end_volume = load_nifti(self.config.end_volume) == 1
        start_node, end_node = self.find_start_end(start_volume, end_volume)
        logging.info(f"Start node: {start_node}, End node: {end_node}")

        # Add a final node with infinite cost to all other nodes
        dummy_node: float = np.inf
        rag2.add_edge(dummy_node, start_node, cost=0)
        rag2.add_edge(end_node, dummy_node, cost=0)
        for node in rag2.nodes:
            if node not in [start_node, end_node, dummy_node]:
                rag2.add_edge(node, dummy_node, cost=np.inf, normalized=True)

        # path = nx.algorithms.approximation.traveling_salesman_problem(rag2, weight="cost", cycle=False)
        path: List[int] = nx.algorithms.approximation.simulated_annealing_tsp(
            rag2, "greedy", source=start_node, weight="cost"
        )
        # path = nx.algorithms.approximation.asadpour_atsp(rag2.to_directed(), weight="cost", source=start_node)
        return path, start_node, end_node, dummy_node

    def filter_path(
        self, path: List[int], start_node: int, end_node: int, dummy_node: float
    ) -> List[int]:
        """Filter the TSP path."""
        start_idx: int = path.index(start_node)
        end_idx: int = path.index(end_node)
        dummy_idx: int = path.index(dummy_node)
        print(len(path), path)
        if start_idx > end_idx:
            # TODO: Check.
            filtered_path: List[int] = path[start_idx:-1][::-1] + path[: end_idx + 1][::-1]
            print(filtered_path)
        else:
            # start_idx < end_idx
            filtered_path = path[: start_idx + 1][::-1] + path[end_idx:-1][::-1]
            print(filtered_path)
        filtered_path = [self.peak_to_idx[node] for node in filtered_path]
        return filtered_path

    def visualize_path(self, filtered_path: List[int]) -> None:
        """Plot the path in 3D."""
        plotter: pv.Plotter = pv.Plotter()
        plotter.add_volume(self.ground_truth * 20, cmap="viridis", opacity="linear")
        lines: pv.PolyData = pv.lines_from_points(self.pcoordinates[filtered_path])
        plotter.add_mesh(lines, line_width=10, cmap="viridis")
        plotter.add_points(self.pcoordinates[filtered_path], color="blue", point_size=10)
        # Add labels at each point
        # nodes = list(rag2.nodes)
        # for i in range(len(filtered_path)):
        #     plotter.add_point_labels(pcoordinates[filtered_path[i]], [nodes[filtered_path[i]]],
        #                             point_size=0,
        #                             font_size=12)

        # Visualize segments

        # active_segments = np.zeros_like(segments)
        # for i, node in enumerate(path, start=1):
        #     active_segments[segments == node] = i

        # plotter.add_volume(active_segments, shade=True, cmap="hot")
        np.savetxt("path.txt", self.pcoordinates[filtered_path])
        plotter.export_html("path.html")
        # TODO: Figure out how to export the path to NIFTI

    def run(self) -> None:
        """Run the entire segmentation pipeline."""
        num_supervoxels: int = self.compute_supervoxels()
        self.compute_edges()
        self.compute_segments(num_supervoxels)
        self.generate_rag()
        distance_map: np.ndarray = self.compute_distance_map()
        self.peak_idxs, peaks = self.compute_peaks(distance_map)
        rag2: nx.Graph = self.simplify_graph(peaks)
        path, start_node, end_node, dummy_node = self.solve_tsp(rag2)
        filtered_path: List[int] = self.filter_path(path, start_node, end_node, dummy_node)
        self.visualize_path(filtered_path)


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small Bowel Segmentation")
    parser.add_argument("--filename_ct", type=str, required=True, help="Path to CT scan")
    parser.add_argument("--filename_gt", type=str, required=True, help="Path to ground truth")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")

    # Add arguments to overwrite config properties
    parser.add_argument("--supervoxel_size", type=int, help="Supervoxel size")
    parser.add_argument("--sigmas", type=float, nargs="+", help="Sigmas for Meijering filter")
    parser.add_argument("--edge_threshold", type=float, help="Edge threshold")
    parser.add_argument(
        "--black_ridges", action="store_true", help="Use black ridges in Meijering filter"
    )
    parser.add_argument(
        "--dilation_iterations", type=int, help="Number of iterations for binary dilation"
    )
    parser.add_argument("--thetav", type=int, help="Theta v")
    parser.add_argument("--thetad", type=int, help="Theta d")
    parser.add_argument("--delta", type=int, help="Delta")
    parser.add_argument("--start_node", type=int, help="Start node")
    parser.add_argument("--end_node", type=int, help="End node")
    parser.add_argument("--start_volume", type=str, help="Path to start volume")
    parser.add_argument("--end_volume", type=str, help="Path to end volume")
    parser.add_argument("--precompute", action="store_true", help="Precompute distances")
    parser.add_argument(
        "--use_rustworkx", action="store_true", help="Use rustworkx for precomputation"
    )
    args = parser.parse_args()

    if args.config:
        config = Config.from_json(args.config)
    else:
        config = Config()

    # Overwrite config properties with arguments
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config = replace(
        config, **{k: arg_overrides[k] for k in config.__dict__.keys() if k in arg_overrides}
    )

    logging.info(config)
    segmentor = SmallBowelSegmentor(args.filename_ct, args.filename_gt, config, 1)
    segmentor.run()
