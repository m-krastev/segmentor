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
# 1. Compute edge map of the CT scan using a Meijering filter. Here, low values correspond to flat (no-gradient) regions, and high values correspond to (high-gradient) ridges. The edge map is computed on 3D space and assumes a grayscale intensity image.
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
#    2.  Solve the TSP problem using the Concorde solver
#

import argparse
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import kimimaro
import networkx as nx
import nibabel as nib
import numpy as np
import pyvista as pv
import skimage as ski
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter as gaussian,
    median_filter as median,
)
from skimage.feature import peak_local_max
from skimage.filters import meijering
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.color import label2rgb
from sklearn.metrics import euclidean_distances as edist
from tqdm import tqdm

from segmentor.utils.medutils import (
    load_and_normalize_nifti,
    load_and_resample_nifti,
    load_nifti,
    save_nifti,
)
from segmentor.utils.plotting import path3d
from segmentor.utils.utils import calculate_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("trame_server").setLevel(logging.ERROR)
logging.getLogger("trame_server.controller").setLevel(logging.ERROR)
logging.getLogger("trame_client").setLevel(logging.ERROR)
logging.getLogger("trame_client.widgets.core").setLevel(logging.ERROR)


try:
    import cupy
    from cucim.core.operations.morphology import (
        distance_transform_edt as _distance_transform_edt,
    )
    from cucim.skimage.feature import peak_local_max as _peak_local_max
    from cucim.skimage.filters import (
        meijering as _meijering,
        gaussian as _gaussian,
        median as _median,
    )
    from cucim.skimage.morphology import binary_dilation as _binary_dilation
    from cucim.skimage.color import label2rgb as _label2rgb

    def distance_transform_edt(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _distance_transform_edt(image, **kwargs).get()

    def binary_dilation(image, iterations=None, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        if iterations is None:
            image = _binary_dilation(image, **kwargs)
        else:
            for _ in range(iterations):
                image = _binary_dilation(image, **kwargs)
        return image.get()

    def meijering(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _meijering(image, **kwargs).get()

    def gaussian(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _gaussian(image, **kwargs).get()

    def median(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _median(image, **kwargs).get()

    def label2rgb(labels, image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(labels, cupy.ndarray):
            labels = cupy.array(labels, dtype=cupy.int32)
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _label2rgb(labels, image, **kwargs).get()

    # Significantly slower than skimage's implementation
    # def peak_local_max(image, **kwargs):
    #     # Check if image is already a CuPy array
    #     if not isinstance(image, cupy.ndarray):
    #         image = cupy.array(image, dtype=cupy.float32)

    #     labels = kwargs.pop("labels", None)
    #     if labels is not None:
    #         labels = cupy.array(labels, dtype=cupy.int32)
    #     return _peak_local_max(image, labels=labels, **kwargs).get()

    logging.info("CuCIM/CuPy installed. Using GPU for graphics heavy operations.")

except ImportError:
    logging.error("CuCIM/CuPy not installed. Please install it to enable GPU acceleration.")
    try:
        from edt import edt as distance_transform_edt

        logging.info("EDT installed. Using CPU for distance transform.")
    except ImportError:
        logging.error(
            "EDT not installed. Please install it to enable fast distance transform: `pip install edt`."
        )


# try:
#     from cuda_slic import slic as _slic

#     def slic(
#         image,
#         n_segments=100,
#         compactness=1.0,
#         spacing=(1, 1, 1),
#         slic_zero=None,
#         multichannel=False,
#         channel_axis=None,
#         start_label=None,
#         mask=None,
#         sigma=0,
#     ):
#         return _slic(
#             image,
#             n_segments,
#             spacing=spacing,
#             compactness=compactness,
#             multichannel=multichannel,
#             # start_label=start_label,
#             # mask=mask,
#             # sigma=sigma,
#         )

# except ImportError:
#     logging.error(
#         "CUDA SLIC not installed. Please install it to enable GPU acceleration for SLIC superpixels: `pip install gpu-slic --no-deps`."
#     )

# Speed up imports
try:
    import rustworkx as rx
except ImportError:
    logging.error(
        "Rustworkx not installed. Please install it to enable fast precomputation of Dijkstra lengths."
    )


class NumpyEncoder(json.JSONEncoder):
    """Special class to serialize numpy types such as int32, int16, float32, etc."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
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
    start_end: Optional[str] = None
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
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self) -> dict:
        """Return hyperparameters as a dictionary."""
        return self.__dict__


class SmallBowelSegmentor:
    image: np.ndarray
    ground_truth: np.ndarray
    edges: Optional[np.ndarray]
    segments: Optional[np.ndarray]
    rag: Optional[nx.Graph]
    pcoordinates: Optional[np.ndarray]
    idx_to_peak: Optional[Dict[int, int]]
    peak_to_idx: Optional[Dict[int, int]]
    voxel_size: Tuple[float, float, float]

    def __init__(
        self,
        filename_ct: str,
        filename_gt: str,
        output_dir: str,
        config: Config,
        label: Optional[int] = None,
    ):
        self.filename_ct: str = filename_ct
        self.filename_gt: str = filename_gt
        self.config: Config = config
        self.supervoxel_size: int = config.supervoxel_size

        self.output_dir: Path = Path(output_dir)
        self.cache_dir = self.output_dir / "cache"

        # Config check for cache invalidation
        config_path = self.cache_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                old_config = Config.from_dict(json.load(f))
            if old_config != config:
                logging.warning("Config has changed. Clearing cache to avoid conflicts...")
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True, parents=True)

        else:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            config_path.write_text(json.dumps(config.to_dict(), indent=4))

        self.label: Optional[int] = (
            label if label is not None else (1 if "nnUNet" in filename_gt else 18)
        )
        self.ground_truth: np.ndarray = (np.asarray(nib.load(self.filename_gt).dataobj) > 0).astype(
            np.uint8
        )
        self.ground_truth = binary_dilation(self.ground_truth, self.config.dilation_iterations)
        self.image = load_and_normalize_nifti(self.filename_ct).astype(np.float32)
        self.voxel_size = nib.load(self.filename_ct).header.get_zooms()
        self.affine = nib.load(self.filename_ct).affine

    @staticmethod
    def compute_supervoxels(image, supervoxel_size, voxel_size: tuple = (1, 1, 1)) -> int:
        """Calculate the number of supervoxels needed based on desired physical size.

        This function determines how many supervoxels to create based on:
        - The desired physical supervoxel size (in mm³)
        - The actual voxel size from the CT scan
        - The total volume of the image

        Returns:
            int: The number of supervoxels needed to achieve the desired physical size

        Example:
            If desired_size = 216mm³, voxel_size = 1mm³, image_size = 512x512x512
            Then num_supervoxels = (512*512*512 * 1) / 216 ≈ 625,777
        """
        voxel_size = np.prod(voxel_size)
        num_voxels: int = np.sum(image > 0)
        num_supervoxels: int = int(num_voxels * voxel_size / supervoxel_size)
        logging.info(
            f"Desired supervoxel size: {supervoxel_size:>6} mm3; "
            f"Assumed voxel size: {voxel_size:>6.2f} mm3; "
            f"Number of supervoxels required: {num_supervoxels:>6}"
        )
        return num_supervoxels

    def compute_edges(self) -> np.ndarray:
        """Compute edge map of the CT scan using Meijering filter.

        Algorithm steps:
        1. Check for cached results to avoid recomputation
        2. Apply Meijering filter to detect tubular structures:
            - Uses multiple sigma values for multi-scale detection
            - Inverts image if black_ridges=True
        3. Remove false positives from air bubbles
        4. Apply median filter to reduce noise

        Returns:
            np.ndarray: Edge map where higher values indicate stronger edges

        Note:
            The Meijering filter is particularly good at detecting tubular structures
            like blood vessels and intestines, making it ideal for bowel tracking.
        """
        edges_cache = self.cache_dir / "wall_map.nii.gz"
        if edges_cache.exists():
            self.edges = nib.load(edges_cache).get_fdata()
            logging.info("Edges loaded")
            return self.edges
        logging.info("Edge map not found. Generating...")
        self.edges = meijering(
            self.image,
            sigmas=self.config.sigmas,
            # Set to always true for now.
            black_ridges=True,
        ).astype(np.float32)

        # remove potential FP edges caused by air bubbles
        # Likely un-needed as the Black ridges parameter should take care of this
        self.edges[self.image < 0.01] = 0
        # The median filter might be too aggressive
        self.edges = np.clip(self.edges, 0, 0.1) / 0.1
        # self.edges = median(self.edges)
        self.edges = gaussian(self.edges, sigma=1)
        save_nifti(self.edges, edges_cache, self.filename_gt)
        logging.info("Edges generated")
        return self.edges

    def compute_segments(self) -> np.ndarray:
        """Compute supervoxels using SLIC algorithm."""
        # DON'T save this as naive NIFTI, it messes up with loading later.
        segments_cache = self.cache_dir / "segments.npy"
        if segments_cache.exists():
            self.segments = np.load(segments_cache)
            logging.info(f"{np.max(self.segments)} segments loaded")
            return self.segments

        num_supervoxels = self.compute_supervoxels(
            self.ground_truth, self.supervoxel_size, self.voxel_size
        )

        self.segments = slic(
            self.edges,
            n_segments=num_supervoxels,
            compactness=1,
            slic_zero=True,
            start_label=1,
            channel_axis=None,
            sigma=0,
            spacing=self.voxel_size,
            max_size_factor=10000,
            mask=self.ground_truth,
        ).astype(np.uint32)

        # TODO: Remove edges which cross the segmentation mask
        # save_nifti(self.segments, self.cache_dir / "segments.nii.gz", self.filename_gt)
        np.save(segments_cache, self.segments)
        # Save another copy of the segments for visualization
        # segments_viz = label2rgb(self.segments, self.edges, bg_label=0, kind="avg")
        # save_nifti(segments_viz, self.cache_dir / "segments_viz.nii.gz", self.filename_gt)
        logging.info(f"{np.max(self.segments)} segments generated")
        return self.segments

    def generate_rag(self) -> nx.Graph:
        """Generate the Region Adjacency Graph (RAG)."""
        rag_cache = self.cache_dir / "rag.json"
        if rag_cache.exists():
            with open(rag_cache) as f:
                self.rag = nx.node_link_graph(json.load(f))
            logging.info(f"RAG loaded n={len(self.rag.nodes)}")
            return self.rag
        self.rag = ski.graph.rag_boundary(self.segments, self.edges, 3)

        # Compute region properties for each supervoxel
        regions = regionprops(self.segments, intensity_image=self.edges)
        for node, region in zip(self.rag.nodes, regions):
            # Add region properties to each node
            self.rag.nodes[node].update({
                "centroid": tuple(int(x) for x in region.centroid),
                "area": region.area,
                "intensity_mean": region.intensity_mean,
                "intensity_min": region.min_intensity,
                "intensity_max": region.max_intensity,
            })

        with open(rag_cache, "w") as f:
            json.dump(nx.node_link_data(self.rag), f, cls=NumpyEncoder)
        logging.info("RAG saved")
        return self.rag

    def compute_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Identify must-pass nodes as local maxima in the distance map.

        Algorithm:
        1. Find local maxima in the EDT that are:
            - At least thetad voxels apart (minimum distance)
            - Above thetav threshold (minimum peak height)
            - Within valid supervoxel regions
        2. Convert peak coordinates to supervoxel indices

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Array of peak coordinates (N x 3)
                - Array of unique supervoxel indices containing peaks

        Note:
            These peaks will serve as waypoints that the final path must visit
        """

        distance_map = distance_transform_edt(
            binary_dilation(
                self.ground_truth,
                iterations=self.config.dilation_iterations,
            )
        )
        save_nifti(distance_map, self.cache_dir / "distance_map.nii.gz", self.filename_ct)

        peak_idxs = peak_local_max(
            distance_map,
            min_distance=self.config.thetad,
            threshold_abs=self.config.thetav,
            # Uncomment for much denser centers
            labels=self.segments,
            num_peaks=512,
        )

        peaks, indices = np.unique(self.segments[tuple(peak_idxs.T)], return_index=True)
        self.pcoordinates = peak_idxs[indices]
        peaks_map = np.zeros_like(self.ground_truth)
        peaks_map[tuple(self.pcoordinates.T)] = 1
        peaks_map = binary_dilation(peaks_map, 3)
        save_nifti(peaks_map, self.cache_dir / "peaks.nii.gz", self.filename_gt)
        logging.info(f"Found {len(peaks)} peaks")
        return self.pcoordinates, peaks

    def simplify_graph(self) -> nx.Graph:
        """Create a simplified graph connecting must-pass points.

        Parameters:
            peaks (np.ndarray): Array of supervoxel indices containing peaks

        Algorithm:
        1. Create new graph with peaks as nodes
        2. Create mappings between peak indices and supervoxel indices
        3. Compute Euclidean distances between all peak pairs
        4. For each pair of peaks:
            - If distance <= delta: Add edge with normalized cost based on Dijkstra path
            - If distance > delta: Add edge with cost proportional to Euclidean distance
        5. Normalize all costs to [0,1] range

        Returns:
            nx.Graph: Simplified graph where:
                - Nodes are supervoxels containing peaks
                - Edges represent possible paths between peaks
                - Edge weights represent normalized path costs

        Note:
            This simplification is crucial for making the TSP problem tractable
        """
        cache_path = self.cache_dir / "rag2.json"
        if cache_path.exists():
            with open(cache_path) as f:
                rag2 = nx.node_link_graph(json.load(f))

            self.idx_to_peak = {k: v for k, v in enumerate(rag2.nodes)}
            self.peak_to_idx = {v: k for k, v in enumerate(rag2.nodes)}
            self.pcoordinates = np.asarray([_["centroid"] for node, _ in rag2.nodes(data=True)])
            logging.info(f"Simplified RAG loaded (n={len(rag2.nodes)})")
            return rag2

        self.pcoordinates, peaks = self.compute_peaks()

        # Create bidirectional mappings for peak indices
        self.idx_to_peak = {idx: peak for idx, peak in enumerate(peaks)}
        self.peak_to_idx = {peak: idx for idx, peak in enumerate(peaks)}
        # self.rag = nx.subgraph(self.rag, peaks)

        rag2 = nx.Graph()
        rag2.add_nodes_from(self.rag.subgraph(peaks).nodes(data=True))
        for node in rag2.nodes:
            rag2.nodes[node]["centroid"] = tuple(self.pcoordinates[self.peak_to_idx[node]])

        if self.config.use_rustworkx:
            graphx = rx.networkx_converter(self.rag)
            nx_to_rx_map = {
                node_nx: node_rx
                for node_rx, node_nx in zip(graphx.node_indices(), self.rag.nodes())
            }

        # Load or compute path lengths
        if (self.cache_dir / "lengths.pkl").exists():
            logging.info("Cache for Dijkstra distances found. Loading precomputed distances...")
            with open(self.cache_dir / "lengths.pkl", "rb") as f:
                lengths = pickle.load(f)
            self.config.precompute = True
        else:
            if self.config.precompute:
                if self.config.use_rustworkx:
                    lengths = rx.all_pairs_dijkstra_path_lengths(
                        graphx, edge_cost_fn=lambda edge: edge["weight"] + 0.5
                    )
                else:
                    lengths = dict(
                        nx.all_pairs_dijkstra_path_length(self.rag, weight="cost", cutoff=20)
                    )
                    # Save lengths to cache
                    logging.info("Precomputed distances saved to cache")
                    with open(self.cache_dir / "lengths.pkl", "wb") as f:
                        pickle.dump(lengths, f)

        # Build edges with appropriate weights
        dists = []
        euclid = edist(self.pcoordinates)

        # Process all possible peak pairs
        allpairs = combinations(range(len(rag2.nodes())), 2)

        for idx1, idx2 in allpairs:
            node1, node2 = self.idx_to_peak[idx1], self.idx_to_peak[idx2]
            dist = euclid[idx1, idx2]
            if dist <= self.config.delta:
                if self.config.precompute:
                    # Paranoid copy to avoid memory leak from rx
                    # import pdb; pdb.set_trace()
                    try:
                        dijkstra_length = (
                            float(lengths[nx_to_rx_map[node1]][nx_to_rx_map[node2], 1e6])
                            if self.config.use_rustworkx
                            else lengths[node1][node2]
                        )
                    # Due to the earlier graph constraint
                    except Exception:
                        dijkstra_length = 1e6
                else:
                    dijkstra_length = (
                        rx.graph_dijkstra_shortest_path_lengths(
                            graphx,
                            node=nx_to_rx_map[node1],
                            edge_cost_fn=lambda edge: edge["weight"],
                            goal=nx_to_rx_map[node2],
                        )[nx_to_rx_map[node2]]
                        if self.config.use_rustworkx
                        else nx.dijkstra_path_length(self.rag, node1, node2, weight="cost")
                    )
                rag2.add_edge(node1, node2, cost=dist, normalized=False)
                dists.append(dijkstra_length)
            else:
                rag2.add_edge(
                    node1, node2, cost=dist / self.config.delta, normalized=True, euclid=dist
                )

        maxdist: float = max(dists)
        for edge in rag2.edges:
            if not rag2.edges[edge]["normalized"]:
                rag2.edges[edge]["cost"] = rag2.edges[edge]["cost"] / maxdist
                rag2.edges[edge]["normalized"] = True

        # Cache the simplified graph
        with open(self.cache_dir / "rag2.json", "w") as f:
            json.dump(nx.node_link_data(rag2), f, cls=NumpyEncoder)
        return rag2

    def solve_tsp(
        self, rag2: nx.Graph, start: tuple[int, ...] = None, end: tuple[int, ...] = None
    ) -> Tuple[List[int], int, int]:
        """Solve the Traveling Salesman Problem (TSP)."""
        if start is None or end is None:
            start_volume = load_nifti(self.config.start_volume) > 0.5
            end_volume = load_nifti(self.config.end_volume) > 0.5
            _start, _end = find_start_end(start_volume, end_volume, self.ground_truth, self.affine)
            np.savetxt(self.cache_dir / "start_end.npy", [_start, _end], fmt="%d")
        start_node = self.segments[start or _start]
        end_node = self.segments[end or _end]
        if start_node not in rag2.nodes:
            start_node = np.argmin(np.linalg.norm(self.pcoordinates - _start, axis=1))
            start_node = self.idx_to_peak[start_node]
        if end_node not in rag2.nodes:
            end_node = np.argmin(np.linalg.norm(self.pcoordinates - _end, axis=1))
            end_node = self.idx_to_peak[end_node]

        logging.info(f"Start node: {start_node}, End node: {end_node}")

        # Add a final node with infinite cost to all other nodes
        dummy_node: float = np.inf
        rag2.add_edge(dummy_node, start_node, cost=0)
        rag2.add_edge(end_node, dummy_node, cost=0)
        for node in rag2.nodes:
            if node not in [start_node, end_node, dummy_node]:
                rag2.add_edge(node, dummy_node, cost=np.inf, normalized=True)

        path = nx.algorithms.approximation.simulated_annealing_tsp(
            rag2, "greedy", source=start_node, weight="cost"
        )
        # path = nx.algorithms.approximation.asadpour_atsp(rag2.to_directed(), weight="cost", source=start_node)

        logging.info(f"Path: ({len(path)}) {path}")
        end_idx: int = path.index(end_node)
        filtered_path = path[1 : end_idx - 1][::-1] + path[end_idx:][::-1]
        assert (
            ((len(path) - 2) == len(filtered_path))
            and (filtered_path[0] == start_node)
            and (filtered_path[-1] == end_node)
        ), (
            f"Expected paths to be of the same lengths but got {len(path) - 2} and {len(filtered_path)}, {filtered_path}"
        )
        logging.info(f"Filtered path: ({len(filtered_path)}) {filtered_path}")
        path = self.pcoordinates[[self.peak_to_idx[node] for node in filtered_path]]

        return path, start_node, end_node

    def visualize_path(self, path: np.ndarray) -> None:
        """Plot the path in 3D."""
        plotter: pv.Plotter = pv.Plotter(off_screen=True)
        plotter.add_volume(
            self.ground_truth.astype(np.uint8) * 10,
            cmap=["blue"],
            clim=[0, 255],
            opacity="linear",
            show_scalar_bar=False,
        )
        lines = pv.lines_from_points(path)
        plotter.add_mesh(lines, line_width=10, color="yellow")
        plotter.add_points(path, color="blue", point_size=10)
        plotter.add_points(np.vstack((path[:1], path[-1:])), color="red", point_size=30)
        plotter.view_xz()
        plotter.export_html(self.output_dir / "path.html")

        np.savetxt(self.output_dir / "tracking_history.npy", path, fmt="%d")
        path_map, coords = path3d(self.image, path, dilate=2)
        np.savetxt(self.output_dir / "path.npy", coords, fmt="%d")
        path_map[tuple(path[0])] = 10
        path_map[tuple(path[-1])] = 10
        save_nifti(
            path_map.astype(np.uint8),
            self.output_dir / "cumulative_path_mask.nii.gz",
            self.filename_gt,
        )

    def run(self) -> None:
        logging.info("Running segmentation pipeline...")
        # Edges
        self.compute_edges()

        # Segments
        self.compute_segments()

        # RAG
        self.generate_rag()
        rag2: nx.Graph = self.simplify_graph()

        # Solve TSP
        start, end = (
            np.loadtxt(self.config.start_end, dtype=int) if self.config.start_end else (None, None)
        )
        path, start_node, end_node = self.solve_tsp(rag2, start, end)
        self.visualize_path(path)
        metrics = calculate_metrics(
            path_coords=path,
            ground_truth=self.ground_truth,
            edges=self.edges,
            voxel_size=self.voxel_size[0],
        )
        metadata = {
            "start": start_node,
            "end": end_node,
            "config": self.config.to_dict(),
            "num_peaks": len(rag2.nodes),
            "num_segments": np.max(self.segments),
        }
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics | {"metadata": metadata}, f, indent=4, cls=NumpyEncoder)
        logging.info("Segmentation pipeline completed with the following metrics: %s", metrics)


def find_start_end(
    duodenum_volume: np.ndarray,
    colon_volume: np.ndarray,
    small_bowel_volume: np.ndarray,
    affine: np.ndarray = None,
):
    """
    Find start and end points for small bowel navigation based on anatomical structures.

    NOTE: Built with the assumption that the duodenum is on the left side of the body (affine: -1 x -1 x 1) and that the input will be XYZ.

    Args:
        duodenum_volume: Duodenum segmentation
        colon_volume: Colon segmentation
        small_bowel_volume: Small bowel segmentation

    Returns:
        Tuple containing start and end coordinates.
    """
    import kimimaro
    import networkx as nx

    def find_duodenojejunal_flexure(
        duodenum_volume: np.ndarray, affine: np.ndarray = None
    ) -> np.ndarray:
        """Find the duodenojejunal flexure as a start point."""
        duodenum_skeleton = kimimaro.skeletonize(
            binary_dilation(duodenum_volume, iterations=5),
            teasar_params={
                "scale": 3,
                "const": 5,
                "pdrf_scale": 10000,
                "pdrf_exponent": 4,
                "soma_acceptance_threshold": 3500,
            },
            anisotropy=(1, 1, 1),
            dust_threshold=5,
            fix_branching=True,
            progress=False,
            parallel_chunk_size=100,
        )
        if 1 not in duodenum_skeleton:
            raise RuntimeError("Kimimaro failed on duodenum.")
        duodenum_skeleton = duodenum_skeleton[1]
        duodenum_graph = nx.Graph()
        vertices_xyz = {idx: loc for idx, loc in enumerate(duodenum_skeleton.vertices)}
        duodenum_graph.add_nodes_from((idx, {"location": loc}) for idx, loc in vertices_xyz.items())
        duodenum_graph.add_edges_from(duodenum_skeleton.edges)
        if duodenum_graph.number_of_nodes() == 0:
            raise RuntimeError("Duodenum skeleton graph empty.")
        ends = [node for node, degree in duodenum_graph.degree() if degree == 1]
        # NOTE: The DJF is the leftmost point (aka min in X axis). This is dependent on the affine transform used in the segmentation. So, sometimes it might correspond to the max in the X axis.
        # TODO: Fix this to be more robust to affine transforms
        nonzero = np.nonzero(duodenum_volume)
        marker = (
            nonzero[0].max(),
            nonzero[1].mean(),
            nonzero[2].mean(),
        )
        start_node = min(
            ends,
            key=lambda n: math.dist(duodenum_graph.nodes[n]["location"], marker),
        )
        return duodenum_graph.nodes[start_node]["location"]

    def find_ileocecal_junction(colon_volume: np.ndarray, affine: np.ndarray = None) -> np.ndarray:
        """Find the ileocecal junction as an end point."""
        colon_skeleton = kimimaro.skeletonize(
            binary_dilation(colon_volume, iterations=5),
            teasar_params={
                "scale": 3,
                "const": 5,
                "pdrf_scale": 10000,
                "pdrf_exponent": 4,
                "soma_acceptance_threshold": 3500,
            },
            # NOTE: Assuming isotropic spacing
            anisotropy=(1, 1, 1),
            dust_threshold=5,
            fix_branching=True,
            progress=False,
            parallel_chunk_size=100,
        )
        if 1 not in colon_skeleton:
            raise RuntimeError("Kimimaro failed on colon.")
        colon_skeleton = colon_skeleton[1]
        colon_graph = nx.Graph()
        vertices_xyz = {idx: loc for idx, loc in enumerate(colon_skeleton.vertices)}
        colon_graph.add_nodes_from((idx, {"location": loc}) for idx, loc in vertices_xyz.items())
        colon_graph.add_edges_from(colon_skeleton.edges)
        if colon_graph.number_of_nodes() == 0:
            raise RuntimeError("Colon skeleton graph empty.")
        ends = [node for node, degree in colon_graph.degree() if degree == 1]

        # NOTE: The ICJ is the rightmost point (aka max in X axis) and positioned relatively to the middle. This is dependent on the affine transform used in the segmentation. So, sometimes it might correspond to the min in the X axis (e.g. when the affine for the X axis is negative).
        # TODO: Fix this to be more robust to affine transforms
        colon_bounding_box = colon_volume.nonzero()
        marker = (
            colon_bounding_box[0].min(),
            colon_bounding_box[1].mean(),
            np.percentile(colon_bounding_box[2], 25),
        )
        ileocecal_end = min(ends, key=lambda n: math.dist(colon_graph.nodes[n]["location"], marker))
        return colon_graph.nodes[ileocecal_end]["location"]

    # Find approximate landmark points
    from scipy.spatial.distance import cdist

    raw_start_coord = find_duodenojejunal_flexure(duodenum_volume, affine=affine)
    raw_end_coord = find_ileocecal_junction(colon_volume, affine=affine)
    sb_coords_xyz = np.argwhere(small_bowel_volume > 0)
    if sb_coords_xyz.shape[0] == 0:
        raise ValueError("Small bowel segmentation empty.")

    # Map to nearest small bowel voxels
    start_distances = cdist(raw_start_coord.reshape(1, 3), sb_coords_xyz)
    nearest_start_idx = np.argmin(start_distances)
    final_start_coord = tuple(sb_coords_xyz[nearest_start_idx].astype(int))

    end_distances = cdist(raw_end_coord.reshape(1, 3), sb_coords_xyz)
    nearest_end_idx = np.argmin(end_distances)
    final_end_coord = tuple(sb_coords_xyz[nearest_end_idx].astype(int))

    return final_start_coord, final_end_coord


# Example Usage
def main():
    parser = argparse.ArgumentParser(description="Small Bowel Segmentation")
    parser.add_argument("--filename_ct", type=str, required=True, help="Path to CT scan")
    parser.add_argument("--filename_gt", type=str, required=True, help="Path to ground truth")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--output", "-o", type=str, default="output", help="Output directory")

    # Add arguments to overwrite config properties
    parser.add_argument("--supervoxel_size", type=int, help="Supervoxel size")
    parser.add_argument("--sigmas", type=float, nargs="+", help="Sigmas for Meijering filter")
    parser.add_argument("--edge_threshold", type=float, help="Edge threshold")
    parser.add_argument(
        "--black_ridges",
        action="store_true",
        help="Use black ridges in Meijering filter",
    )
    parser.add_argument(
        "--dilation_iterations",
        type=int,
        default=3,
        help="Number of iterations for binary dilation",
    )
    parser.add_argument("--thetav", type=int, help="Theta v")
    parser.add_argument("--thetad", type=int, help="Theta d")
    parser.add_argument("--delta", type=int, help="Delta")
    parser.add_argument("--start_end", type=str, help="Start node")
    parser.add_argument("--start_volume", type=str, help="Path to start volume")
    parser.add_argument("--end_volume", type=str, help="Path to end volume")
    parser.add_argument("--precompute", action="store_true", help="Precompute distances")
    parser.add_argument(
        "--use_rustworkx", action="store_true", help="Use rustworkx for precomputation"
    )
    parser.add_argument("--label", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        config = Config.from_json(args.config)
    else:
        config = Config()

    # Overwrite config properties with arguments
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config = replace(
        config,
        **{k: arg_overrides[k] for k in config.__dict__.keys() if k in arg_overrides},
    )

    logging.info(config)
    segmentor = SmallBowelSegmentor(
        args.filename_ct, args.filename_gt, args.output, config, args.label
    )
    segmentor.run()


if __name__ == "__main__":
    main()
