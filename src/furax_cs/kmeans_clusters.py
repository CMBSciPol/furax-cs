"""K-means clustering utilities for CMB component separation.

This module provides a unified interface for generating K-means cluster
assignments used in adaptive component separation. The clustering allows
different spectral parameters in different sky regions.
"""


import jax
import jax.numpy as jnp
from jax_healpy.clustering import (
    find_kmeans_clusters,
    get_cutout_from_mask,
    normalize_by_first_occurrence,
)
from jaxtyping import Array, Float, Int, PRNGKeyArray


def kmeans_clusters(
    prngkey: PRNGKeyArray,
    mask: Float[Array, " npix"],
    indices: Int[Array, " n_valid"],
    regions: dict[str, int],
    max_patches: dict[str, int] | None = None,
    initial_sample_size: int = 1,
) -> dict[str, Int[Array, " n_valid"]]:
    """Generate K-means cluster assignments for spectral parameter optimization.

    This function performs spherical K-means clustering on HEALPix sky pixels
    to partition the sky into regions with potentially different spectral
    parameters. The clustering is applied separately for each parameter type
    (e.g., dust temperature, dust spectral index, synchrotron index).

    Args:
        prngkey: JAX PRNG key for reproducible clustering initialization.
        mask: Full-sky HEALPix mask array (1 for valid pixels, 0 for masked).
        indices: Indices of unmasked pixels, typically from ``jnp.where(mask == 1)``.
        regions: Dictionary mapping patch names to target number of clusters.
            Expected keys: ``"temp_dust_patches"``, ``"beta_dust_patches"``,
            ``"beta_pl_patches"``.
        max_patches: Maximum number of clusters per parameter. If None, uses regions
            values. This controls the size of the output arrays for JIT compatibility.
        initial_sample_size: Number of initial samples for K-means initialization.
            Defaults to 1.

    Returns:
        Dictionary of cluster assignments (cutout arrays, not full-sky).
        Keys match ``regions``. Values are int64 arrays of shape (n_unmasked,).

    Example:
        >>> from furax_cs.data import get_mask
        >>> mask = get_mask("GAL020")
        >>> (indices,) = jnp.where(mask == 1)
        >>> regions = {
        ...     "temp_dust_patches": 50,
        ...     "beta_dust_patches": 500,
        ...     "beta_pl_patches": 50,
        ... }
        >>> clusters = kmeans_clusters(
        ...     jax.random.key(0), mask, indices, regions
        ... )
        >>> clusters["beta_dust_patches"].shape
        (n_unmasked_pixels,)

    Notes:
        The function uses ``normalize_by_first_occurrence`` to ensure cluster indices
        are contiguous starting from 0, which is required for parameter indexing.
    """
    if max_patches is None:
        max_patches = regions

    # Generate full-sky cluster indices using spherical K-means
    patch_indices = jax.tree.map(
        lambda c, mp: find_kmeans_clusters(
            mask,
            indices,
            c,
            prngkey,
            max_centroids=mp,
            initial_sample_size=initial_sample_size,
        ),
        regions,
        max_patches,
    )

    # Extract cutout (only unmasked pixels)
    guess_clusters = get_cutout_from_mask(patch_indices, indices)

    # Normalize cluster indices for logical indexing (0, 1, 2, ...)
    guess_clusters = jax.tree.map(
        lambda g, c, mp: normalize_by_first_occurrence(g, c, mp).astype(jnp.int64),
        guess_clusters,
        regions,
        max_patches,
    )

    # Ensure int64 dtype for all cluster arrays
    guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), guess_clusters)

    return guess_clusters
