"""Multi-resolution clustering utilities using HEALPix ud_grade.

This module provides utilities for generating resolution-based patches
where pixels sharing the same low-resolution parent pixel are grouped
together. This is the approach used in the LiteBIRD PTEP methodology.
"""


import jax
import jax.numpy as jnp
import jax_healpy as jhp
from jax_healpy.clustering import get_cutout_from_mask
from jaxtyping import Array, Float, Int


def _ud_grade_patches(
    ipix: Int[Array, " npix"], nside_in: int, nside_out: int
) -> Int[Array, " npix"]:
    """Create resolution-based patch indices using ud_grade.

    Downgrade pixel indices to target resolution then upgrade back,
    creating uniform patches at the target resolution scale.

    Args:
        ipix: Full-sky pixel indices (0 to npix-1).
        nside_in: Input map resolution (nside).
        nside_out: Target resolution for patches. If 0, returns all zeros (single patch).

    Returns:
        Patch indices at original resolution with values grouped by
        target resolution pixels.
    """
    if nside_out == 0:
        return jnp.zeros_like(ipix)
    else:
        # Downgrade to target resolution
        lowered = jhp.ud_grade(ipix.astype(jnp.float64), nside_out=nside_out)
        # Upgrade back to original resolution
        return jhp.ud_grade(lowered, nside_out=nside_in).astype(jnp.int64)


def _normalize_array(arr: Int[Array, " n"]) -> Int[Array, " n"]:
    """Normalize patch indices to be contiguous from 0."""
    unique_vals, indices_norm = jnp.unique(arr, return_inverse=True, size=arr.size)
    return indices_norm.astype(jnp.int64)


def multires_clusters(
    mask: Float[Array, " npix"],
    indices: Int[Array, " n_valid"],
    target_ud_grade: dict[str, int],
    nside: int | None = None,
) -> dict[str, Int[Array, " n_valid"]]:
    """Generate multi-resolution cluster assignments using HEALPix ud_grade.

    This function creates resolution-based patches where pixels sharing
    the same low-resolution parent pixel are grouped together. This is
    the approach used in the LiteBIRD PTEP methodology.

    Args:
        mask: Full-sky HEALPix mask array (1 for valid pixels, 0 for masked).
        indices: Indices of unmasked pixels, typically from ``jnp.where(mask == 1)``.
        target_ud_grade: Dictionary mapping parameter names to target nside values.
            Expected keys: ``"beta_dust"``, ``"temp_dust"``, ``"beta_pl"``.
            Values are nside parameters (must be powers of 2).
            Use 0 for a single global patch.
        nside: Input map resolution. If None, inferred from mask size.

    Returns:
        Dictionary of normalized cluster assignments (cutout arrays).
        Keys are ``"{param}_patches"`` format. Values are int64 arrays.

    Example:
        >>> from furax_cs.data import get_mask
        >>> mask = get_mask("GAL020")
        >>> (indices,) = jnp.where(mask == 1)
        >>> target_resolutions = {
        ...     "beta_dust": 64,   # Full resolution
        ...     "temp_dust": 32,   # 4x fewer patches
        ...     "beta_pl": 16,     # 16x fewer patches
        ... }
        >>> clusters = multires_clusters(mask, indices, target_resolutions)
        >>> clusters["temp_dust_patches"].shape
        (n_unmasked_pixels,)
    """
    # Infer nside from mask if not provided
    if nside is None:
        npix = mask.shape[0]
        nside = int(jnp.sqrt(npix / 12))

    # Create full-sky pixel indices
    npix = nside**2 * 12
    ipix = jnp.arange(npix)

    # Generate patch indices for each parameter
    patch_indices = {}
    for param_name, target_nside in target_ud_grade.items():
        patch_key = f"{param_name}_patches"
        patch_indices[patch_key] = _ud_grade_patches(ipix, nside, int(target_nside))

    # Extract cutout (only unmasked pixels)
    masked_patches = get_cutout_from_mask(patch_indices, indices)

    # Normalize indices for consistent indexing (0, 1, 2, ...)
    masked_patches = jax.tree.map(_normalize_array, masked_patches)

    return masked_patches
