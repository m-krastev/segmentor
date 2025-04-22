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

import kimimaro
import networkx as nx
import nibabel as nib
import numpy as np
import pyvista as pv
import skimage as ski
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.filters import meijering
from skimage.segmentation import slic
from sklearn.metrics import euclidean_distances as edist
from tqdm import tqdm

from segmentor.utils.medutils import (
    load_and_normalize_nifti,
    load_and_resample_nifti,
    load_nifti,
    save_nifti,
)
from segmentor.utils.plotting import path3d

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("trame_server").setLevel(logging.ERROR)
logging.getLogger("trame_server.controller").setLevel(logging.ERROR)
logging.getLogger("trame_client").setLevel(logging.ERROR)
logging.getLogger("trame_client.widgets.core").setLevel(logging.ERROR)


try:
    import cupy
    from cucim.core.operations.morphology import distance_transform_edt as _distance_transform_edt
    from cucim.skimage.feature import peak_local_max as _peak_local_max
    from cucim.skimage.filters import meijering as _meijering
    from cucim.skimage.morphology import binary_dilation as _binary_dilation

    def distance_transform_edt(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return cupy.asnumpy(_distance_transform_edt(image, **kwargs).get())

    def binary_dilation(image, iterations=None, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        if iterations is None:
            image = _binary_dilation(image, **kwargs)
        else:
            for _ in range(iterations):
                image = _binary_dilation(image, **kwargs)
        return cupy.asnumpy(image.get())

    def meijering(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        image = _meijering(cupy.array(image, dtype=cupy.float32), **kwargs).get()
        return cupy.asnumpy(image)

    def peak_local_max(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)

        labels = kwargs.pop("labels", None)
        if labels is not None:
            labels = cupy.array(labels, dtype=cupy.int32)
        return cupy.asnumpy(_peak_local_max(image, labels=labels, **kwargs).get())

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
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self) -> dict:
        """Return hyperparameters as a dictionary."""
        return self.__dict__


class SmallBowelSegmentor:
    nii: np.ndarray
    ground_truth: np.ndarray
    edges: Optional[np.ndarray]
    segments: Optional[np.ndarray]
    rag: Optional[nx.Graph]
    peak_idxs: Optional[np.ndarray]
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
                if (self.output_dir / "metrics.json").exists():
                    logging.info("Metrics found. Skipping computation...")
                    exit(0)
        else:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            config_path.write_text(json.dumps(config.to_dict(), indent=4))

        self.label: Optional[int] = (
            label if label is not None else (1 if "nnUNet" in filename_gt else 18)
        )
        self.ground_truth: np.ndarray = (
            np.asarray(nib.load(self.filename_gt).dataobj) == self.label
        ).astype(np.uint8)
        self.nii: np.ndarray = load_and_normalize_nifti(self.filename_ct).astype(np.float32)
        self.voxel_size: Tuple[float, float, float] = nib.load(self.filename_ct).header.get_zooms()

    def compute_supervoxels(self) -> int:
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
        voxel_size = np.prod(self.voxel_size)
        num_voxels: int = np.prod(self.ground_truth.shape)
        num_supervoxels: int = int((num_voxels * voxel_size) / self.supervoxel_size)
        logging.info(
            f"Desired supervoxel size: {self.supervoxel_size:>6} mm3; "
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
        5. Cache results for future use

        Returns:
            np.ndarray: Edge map where higher values indicate stronger edges

        Note:
            The Meijering filter is particularly good at detecting tubular structures
            like blood vessels and intestines, making it ideal for bowel tracking.
        """
        edges_cache = self.cache_dir / "edges.npy"
        if edges_cache.exists():
            self.edges = np.load(edges_cache)
            logging.info("Edges loaded")
        else:
            logging.info("Edge map not found. Generating...")
            self.edges = meijering(
                1 - self.nii,
                sigmas=self.config.sigmas,
                black_ridges=self.config.black_ridges,
            ).astype(np.float32)

            # remove potential FP edges caused by air bubbles
            self.edges[self.nii < np.quantile(self.nii, 0.01)] = 0
            # The median filter might be too aggressive
            self.edges = ski.filters.median(self.edges)
            # self.edges = ski.filters.gaussian(self.edges, sigma=3)
            np.save(edges_cache, self.edges)
            save_nifti(self.edges, self.cache_dir / "edges.nii.gz", self.filename_gt)
            logging.info("Edges generated")
        return self.edges

    def compute_segments(self, num_supervoxels: int) -> np.ndarray:
        """Compute supervoxels using SLIC algorithm."""
        segments_cache = self.cache_dir / "segments.npy"
        if segments_cache.exists():
            self.segments = np.load(segments_cache)
            logging.info(f"{np.max(self.segments)} segments loaded")
        else:
            # self.edges = np.where(
            #     self.edges > self.config.edge_threshold, self.edges, 0
            # )
            self.segments = slic(
                self.edges,
                n_segments=num_supervoxels,
                compactness=0.01,
                slic_zero=True,
                start_label=1,
                channel_axis=None,
                sigma=0,
            ).astype(np.uint16 if num_supervoxels < 2**16 else np.uint32)

            # TODO: Remove edges which cross the segmentation mask
            np.save(segments_cache, self.segments)

            save_nifti(self.segments, self.cache_dir / "segments.nii.gz", self.filename_gt)
            # Save another copy of the segments for visualization
            # Values inside any given supervoxel equal the average value in the supervoxel
            segments_viz = ski.color.label2rgb(self.segments, self.edges, bg_label=0, kind="avg")
            save_nifti(segments_viz, self.cache_dir / "segments_viz.nii.gz", self.filename_gt)
            logging.info(f"{np.max(self.segments)} segments generated")
        return self.segments

    def generate_rag(self) -> nx.Graph:
        """Generate the Region Adjacency Graph (RAG)."""
        rag_cache = self.cache_dir / "rag.json"
        if rag_cache.exists():
            with open(rag_cache, "r") as f:
                self.rag = nx.node_link_graph(json.load(f))
                logging.info("RAG loaded")
        else:
            self.rag = ski.graph.rag_boundary(self.segments, self.edges)
            with open(rag_cache, "w") as f:
                json.dump(nx.node_link_data(self.rag), f, cls=NumpyEncoder)
            logging.info("RAG saved")
        return self.rag

    def compute_distance_map(self) -> np.ndarray:
        """Compute Euclidean distance transform of the segmentation mask.

        The distance transform assigns to each voxel the distance to the nearest
        boundary of the segmentation mask. This is useful for:
        1. Finding the centerline of the bowel
        2. Identifying potential must-pass points
        3. Ensuring the path stays within the segmentation

        Returns:
            np.ndarray: Distance map where each value is the distance to the nearest boundary
        """
        return distance_transform_edt(self.ground_truth.astype(bool))
        # return distance_transform_edt(
        #     # binary_dilation(
        #     #     self.edges > self.config.edge_threshold, iterations=self.config.dilation_iterations, mask=self.
        #     ground_truth
        #     # )
        #     self.ground_truth.astype(bool)
        # )

    def compute_peaks(self, distance_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Identify must-pass nodes as local maxima in the distance map.

        Parameters:
            distance_map (np.ndarray): Distance transform of the segmentation mask

        Algorithm:
        1. Find local maxima in the distance map that are:
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
        self.peak_idxs = peak_local_max(
            distance_map,
            min_distance=self.config.thetad,
            threshold_abs=self.config.thetav,
            labels=self.segments,
        )
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
        if len(start_segments) > 0:
            start = start_segments[0]
        else:
            # Generate the skeleton of the duodenum
            duodenum_skeleton = kimimaro.skeletonize(
                # Paranoid dilation to make sure the duodenum is connected
                binary_dilation(start_volume, iterations=5),
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
            duodenum_graph = nx.Graph()
            duodenum_graph.add_nodes_from(
                (idx, {"location": loc}) for idx, loc in enumerate(duodenum_skeleton.vertices)
            )
            duodenum_graph.add_edges_from(duodenum_skeleton.edges)

            # Get the lowest end of the duodenum
            ends = list(node[0] for node in duodenum_graph.degree() if node[1] == 1)
            start = max(ends, key=lambda x: duodenum_graph.nodes[x]["location"][2])
            # Get the closest peak to the duodenum end
            start_idx = np.argmin(
                np.linalg.norm(
                    self.peak_idxs - duodenum_graph.nodes[start]["location"],
                    axis=1,
                )
            )
            start = self.segments[
                self.peak_idxs[start_idx][0],
                self.peak_idxs[start_idx][1],
                self.peak_idxs[start_idx][2],
            ]

        end = end_segments
        # Let end be the lowest peak
        # end = np.isin(segments, end)
        # end = end[np.max(end[:, 2])]
        # end = segments[end[0], end[1], end[2]]

        def get_ileocecal_end(colon: np.ndarray) -> np.ndarray:
            colon_skeleton = kimimaro.skeletonize(
                # Paranoid dilation to make sure the colon is connected
                binary_dilation(colon, iterations=5),
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

            # Secondary graph that will only use the largest connected component (copy probably unneeded but its not that big anyway)
            colon_graph = colon_graph.subgraph(
                max(nx.connected_components(colon_graph), key=len)
            ).copy()

            ends = list(node[0] for node in colon_graph.degree() if node[1] == 1)
            rectal_end = max(ends, key=lambda x: colon_graph.nodes[x]["location"][2])
            ends.remove(rectal_end)
            if len(ends) > 1:
                # The ileocecal end is the one farthest away from the rectal end
                # This can be done by finding the end that is the farthest away in terms of graph distance
                ileocecal_end = max(
                    ends,
                    key=lambda x: nx.shortest_path_length(colon_graph, rectal_end, x),
                    # # Alternatively, find the end that is the farthest away in terms of Euclidean distance
                    # key=lambda x: np.linalg.norm(
                    #     colon_graph.nodes[x]["location"] - colon_graph.nodes[rectal_end]["location"]
                    # ),
                )
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
                self.peak_idxs[end_idx][0],
                self.peak_idxs[end_idx][1],
                self.peak_idxs[end_idx][2],
            ]
        return start, end

    def simplify_graph(self, peaks: np.ndarray) -> nx.Graph:
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
        rag2: nx.Graph = nx.Graph()
        rag2.add_nodes_from(peaks)
        nodes: List[int] = list(rag2.nodes)

        # Create bidirectional mappings for peak indices
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

        # Process all possible peak pairs
        allpairs = combinations(nodes, 2)

        # Load or compute path lengths
        if (self.cache_dir / "lengths.pkl").exists():
            logging.info("Cache for Dijkstra distances found. Loading precomputed distances...")
            with open(self.cache_dir / "lengths.pkl", "rb") as f:
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
                    with open(self.cache_dir / "lengths.pkl", "wb") as f:
                        pickle.dump(lengths, f)

        # Build edges with appropriate weights
        dists: List[float] = []
        for node1, node2 in tqdm(
            allpairs,
            desc="Computing distances",
            total=len(nodes) * (len(nodes) - 1) // 2,
        ):
            dist: float = euclid[self.peak_to_idx[node1], self.peak_to_idx[node2]]
            if dist <= self.config.delta:
                if self.config.precompute:
                    dijkstra_length = (
                        # Paranoid copy to avoid memory leak from rustworkx
                        float(lengths[self.peak_to_idx[node1]][self.peak_to_idx[node2]])
                        if self.config.use_rustworkx
                        else lengths[node1][node2]
                    )
                else:
                    dijkstra_length = nx.dijkstra_path_length(self.rag, node1, node2, weight="cost")
                rag2.add_edge(node1, node2, cost=dist, normalized=False)
                dists.append(dijkstra_length)
            else:
                rag2.add_edge(node1, node2, cost=dist / self.config.delta, normalized=True)

        # Normalize costs
        maxdist: float = max(dists)
        for edge in rag2.edges:
            if not rag2.edges[edge]["normalized"]:
                rag2.edges[edge]["cost"] /= maxdist
                rag2.edges[edge]["normalized"] = True

        # Cache the simplified graph
        with open(self.cache_dir / "rag2.json", "w") as f:
            json.dump(nx.node_link_data(rag2), f, cls=NumpyEncoder)
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

        path_map, coords = path3d(self.nii, self.pcoordinates[filtered_path], dilate=2)

        np.savetxt(self.output_dir / "nodes.txt", self.pcoordinates[filtered_path])
        np.save(self.output_dir / "path.npy", coords)
        save_nifti(path_map.astype(np.uint8), self.output_dir / "path.nii.gz", self.filename_gt)
        plotter.export_html(self.output_dir / "path.html")

    def evaluate_metrics(self, path: list[list] = None):
        if path is None:
            path = np.load(self.output_dir / "path.npy")

        length = path.shape[0]
        average_gradient = np.mean(self.nii[tuple(np.asarray(path).T)])

        dice_overlap = []
        path_map = np.zeros_like(self.ground_truth)
        path_map[path[:, 0], path[:, 1], path[:, 2]] = 1
        for i in range(1, 6):
            # Dilate at each iteration
            path_map = binary_dilation(path_map, iterations=1)
            dice_overlap.append(
                2
                * np.sum(self.ground_truth * path_map)
                / (np.sum(self.ground_truth) + np.sum(path_map))
            )

        dice_overlap = np.mean(dice_overlap)
        metrics = {
            "curve_length": length,
            "average_gradient": average_gradient,
            "dice_overlap": dice_overlap,
        }

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)

        return metrics

    def run(self) -> None:
        logging.info("Running segmentation pipeline...")
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
        metrics = self.evaluate_metrics()
        logging.info("Segmentation pipeline completed with the following metrics: %s", metrics)


# Example Usage
if __name__ == "__main__":
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
        help="Number of iterations for binary dilation",
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
        config,
        **{k: arg_overrides[k] for k in config.__dict__.keys() if k in arg_overrides},
    )

    logging.info(config)
    segmentor = SmallBowelSegmentor(args.filename_ct, args.filename_gt, args.output, config, 1)
    segmentor.run()
