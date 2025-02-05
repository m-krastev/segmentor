import math
from collections.abc import Iterable
from warnings import warn

import jax
import jax.numpy as jnp
import jax.random as random
from jax.ops import segment_sum
from jax.scipy.ndimage import gaussian_filter

# These imports assume that your existing helper functions have been ported
# to work with JAX arrays.

###############################################################################
# A simple k-means clustering implementation using JAX.
###############################################################################
def jax_kmeans2(data, init_centroids, num_iters=5):
    """
    Run a simple k-means clustering on data using JAX.

    Parameters
    ----------
    data : array, shape (N, D)
        Data to be clustered.
    init_centroids : array, shape (K, D)
        Initial centroid positions.
    num_iters : int
        Number of iterations.
    
    Returns
    -------
    centroids : array, shape (K, D)
        Final centroid positions.
    labels : array, shape (N,)
        Label assignment for each data point.
    """
    centroids = init_centroids
    for _ in range(num_iters):
        # Compute squared Euclidean distance between each data point and each centroid
        distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        labels = jnp.argmin(distances, axis=1)
        num_segments = centroids.shape[0]
        # Use segment_sum to compute the sum of points in each cluster.
        sum_points = segment_sum(data, labels, num_segments)
        counts = segment_sum(jnp.ones((data.shape[0],), dtype=data.dtype), labels, num_segments)
        # Avoid division by zero
        centroids = sum_points / jnp.maximum(counts[:, None], 1)
    return centroids, labels

###############################################################################
# Mask-based centroid initialization using k-means.
###############################################################################
def _get_mask_centroids(mask, n_centroids, multichannel, key):
    """
    Find regularly spaced centroids on a mask.
    
    Parameters
    ----------
    mask : array, 3D
        The mask within which the centroids must be positioned.
    n_centroids : int
        The number of centroids to be returned.
    multichannel : bool
        Whether the image is multichannel.
    key : jax.random.PRNGKey
        Random key for reproducibility.
    
    Returns
    -------
    centroids : array, shape (n_centroids, 3)
        The coordinates of the centroids.
    steps : array, shape (3,)
        The approximate distance between two seeds in each dimension.
    """
    # Get the coordinates of nonzero mask elements.
    # jnp.nonzero returns a tuple of 1D arrays; stack them as columns.
    coords = jnp.stack(jnp.nonzero(mask), axis=1).astype(jnp.float32)
    n_coords = coords.shape[0]
    n_choice = jnp.minimum(n_centroids, n_coords)
    idx = random.choice(key, n_coords, shape=(n_choice,), replace=False)
    idx = jnp.sort(idx)

    dense_factor = 10
    ndim_spatial = mask.ndim - 1 if multichannel else mask.ndim
    n_dense = int((dense_factor ** ndim_spatial) * n_centroids)
    if n_coords > n_dense:
        idx_dense = random.choice(key, n_coords, shape=(n_dense,), replace=False)
        idx_dense = jnp.sort(idx_dense)
        data_for_kmeans = coords[idx_dense]
    else:
        data_for_kmeans = coords

    initial_centroids = coords[idx]
    centroids, _ = jax_kmeans2(data_for_kmeans, initial_centroids, num_iters=5)

    # Compute pairwise distances between centroids.
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    # Set the diagonal to infinity so that the minimum does not pick the same point.
    dist = dist.at[jnp.diag_indices(dist.shape[0])].set(jnp.inf)
    closest_pts = jnp.argmin(dist, axis=-1)
    steps = jnp.mean(jnp.abs(centroids - centroids[closest_pts]), axis=0)
    return centroids, steps

###############################################################################
# Grid-based centroid initialization.
###############################################################################
def _get_grid_centroids(image, n_centroids):
    """
    Find regularly spaced centroids on the image.
    
    Parameters
    ----------
    image : array
        Input image.
    n_centroids : int
        Approximate number of centroids.
    
    Returns
    -------
    centroids : array, shape (~n_centroids, 3)
        The coordinates of the centroids.
    steps : array, shape (3,)
        The approximate distance between two seeds in each dimension.
    """
    d, h, w = image.shape[:3]
    grid_z, grid_y, grid_x = jnp.meshgrid(
        jnp.arange(d), jnp.arange(h), jnp.arange(w), indexing='ij'
    )
    slices = regular_grid(image.shape[:3], n_centroids)
    centroids_z = grid_z[slices].ravel()[..., None]
    centroids_y = grid_y[slices].ravel()[..., None]
    centroids_x = grid_x[slices].ravel()[..., None]
    centroids = jnp.concatenate([centroids_z, centroids_y, centroids_x], axis=-1)
    steps = jnp.array([float(s.step) if s.step is not None else 1.0 for s in slices])
    return centroids, steps

###############################################################################
# Placeholders for the SLIC algorithm and connectivity enforcement.
###############################################################################
def _slic_jax(
    image,
    mask,
    segments,
    step,
    max_num_iter,
    spacing,
    slic_zero,
    ignore_color,
    start_label,
):
    """
    Placeholder for a JAX implementation of the SLIC algorithm.
    In a full rewrite, you would implement the iterative clustering
    in JAX here.
    """
    # TODO: Implement the iterative clustering in JAX.
    # For now, return a dummy label image.
    return jnp.zeros(image.shape[:3], dtype=jnp.int32)

def _enforce_label_connectivity_jax(labels, min_size, max_size, start_label):
    """
    Placeholder for a JAX implementation of label connectivity enforcement.
    In a full rewrite, you would implement connected component analysis in JAX.
    """
    # TODO: Implement connectivity enforcement in JAX.
    return labels

###############################################################################
# The main SLIC function (rewritten for JAX)
###############################################################################
def slic(
    image,
    n_segments=100,
    compactness=10.0,
    max_num_iter=10,
    sigma=0,
    spacing=None,
    enforce_connectivity=True,
    min_size_factor=0.5,
    max_size_factor=3,
    slic_zero=False,
    start_label=1,
    mask=None,
    *,
    channel_axis=-1,
    key=random.PRNGKey(123)
):
    """
    Segment image using k-means clustering in Color-(x,y,z) space.
    This JAX version follows the structure of the original code.
    """
    if image.ndim == 2 and channel_axis is not None:
        raise ValueError(
            f"channel_axis={channel_axis} indicates multichannel, which is not "
            "supported for a two-dimensional image; use channel_axis=None if "
            "the image is grayscale"
        )

    # Convert the image to float and the supported float type.
    float_dtype = jnp.float32 # HARDCODED
    image = image.astype(float_dtype)

    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)
        if channel_axis is not None:
            mask_ = jnp.expand_dims(mask, axis=channel_axis)
            mask_ = jnp.broadcast_to(mask_, image.shape)
        else:
            mask_ = mask
        image_values = image[mask_]
    else:
        image_values = image

    # Rescale the image to [0, 1]
    imin = jnp.min(image_values)
    imax = jnp.max(image_values)
    if jnp.isnan(imin):
        raise ValueError("unmasked NaN values in image are not supported")
    if jnp.isinf(imin) or jnp.isinf(imax):
        raise ValueError("unmasked infinite values in image are not supported")
    image = image - imin
    if imax != imin:
        image = image / (imax - imin)

    use_mask = mask is not None
    dtype = image.dtype

    is_2d = False
    multichannel = channel_axis is not None
    if image.ndim == 2:
        # 2D grayscale image: add depth and channel dimensions.
        image = image[jnp.newaxis, ..., jnp.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # 2D multichannel image: add a singleton depth dimension.
        image = image[jnp.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # 3D image with no explicit channel dimension: add a channel.
        image = image[..., jnp.newaxis]

    if start_label not in [0, 1]:
        raise ValueError("start_label should be 0 or 1.")

    # Initialize centroids.
    update_centroids = False
    if use_mask:
        mask = mask.astype(jnp.uint8)
        if mask.ndim == 2:
            mask = mask[jnp.newaxis, ...]
        if mask.shape != image.shape[:3]:
            raise ValueError("image and mask should have the same shape.")
        centroids, steps = _get_mask_centroids(mask, n_segments, multichannel, key)
        update_centroids = True
    else:
        centroids, steps = _get_grid_centroids(image, n_segments)

    # Process spacing.
    if spacing is None:
        spacing = jnp.ones(3, dtype=dtype)
    elif isinstance(spacing, Iterable):
        spacing = jnp.asarray(spacing, dtype=dtype)
        if is_2d:
            if spacing.size != 2:
                if spacing.size == 3:
                    warn(
                        "Input image is 2D: spacing number of "
                        "elements must be 2. In the future, a ValueError "
                        "will be raised.",
                        FutureWarning,
                        stacklevel=2,
                    )
                else:
                    raise ValueError(
                        f"Input image is 2D, but spacing has "
                        f"{spacing.size} elements (expected 2)."
                    )
            else:
                spacing = jnp.insert(spacing, 0, 1)
        elif spacing.size != 3:
            raise ValueError(
                f"Input image is 3D, but spacing has "
                f"{spacing.size} elements (expected 3)."
            )
        spacing = jnp.ascontiguousarray(spacing)
    else:
        raise TypeError("spacing must be None or iterable.")

    # Process sigma.
    if jnp.isscalar(sigma):
        sigma = jnp.array([sigma, sigma, sigma], dtype=dtype)
        sigma = sigma / spacing
    elif isinstance(sigma, Iterable):
        sigma = jnp.asarray(sigma, dtype=dtype)
        if is_2d:
            if sigma.size != 2:
                if spacing.size == 3:
                    warn(
                        "Input image is 2D: sigma number of "
                        "elements must be 2. In the future, a ValueError "
                        "will be raised.",
                        FutureWarning,
                        stacklevel=2,
                    )
                else:
                    raise ValueError(
                        f"Input image is 2D, but sigma has "
                        f"{sigma.size} elements (expected 2)."
                    )
            else:
                sigma = jnp.insert(sigma, 0, 0)
        elif sigma.size != 3:
            raise ValueError(
                f"Input image is 3D, but sigma has "
                f"{sigma.size} elements (expected 3)."
            )

    if jnp.any(sigma > 0):
        # Add zero smoothing for the channel dimension.
        sigma = list(sigma) + [0]
        image = gaussian_filter(image, sigma=sigma, mode='reflect')

    n_centroids = centroids.shape[0]
    segments = jnp.concatenate(
        [centroids, jnp.zeros((n_centroids, image.shape[3]), dtype=dtype)], axis=-1
    )
    # Scale the step and adjust the image intensity.
    step = jnp.max(steps)
    ratio = 1.0 / compactness
    image = jnp.ascontiguousarray(image * ratio, dtype=dtype)

    # Run SLIC iterations.
    if update_centroids:
        _slic_jax(
            image,
            mask,
            segments,
            step,
            max_num_iter,
            spacing,
            slic_zero,
            ignore_color=True,
            start_label=start_label,
        )

    labels = _slic_jax(
        image,
        mask,
        segments,
        step,
        max_num_iter,
        spacing,
        slic_zero,
        ignore_color=False,
        start_label=start_label,
    )

    if enforce_connectivity:
        if use_mask:
            segment_size = jnp.sum(mask) / n_centroids
        else:
            segment_size = math.prod(image.shape[:3]) / n_centroids
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_jax(
            labels, min_size, max_size, start_label=start_label
        )

    if is_2d:
        labels = labels[0]

    return labels
