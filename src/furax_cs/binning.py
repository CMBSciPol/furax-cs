"""Equal-width binning for spectral-parameter maps."""

from __future__ import annotations

import numpy as np


def bin_parameter_map(
    pixel_map: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a valid-pixels-only parameter map into equal-width bins.

    Parameters
    ----------
    pixel_map : np.ndarray
        1-D array of parameter values for **valid pixels only** (no UNSEEN).
    nbins : int
        Number of equal-width bins.

    Returns
    -------
    patch_indices : np.ndarray
        0-based bin indices, shape ``(n_valid,)``, values in ``[0, nbins-1]``.
    bin_centers : np.ndarray
        Bin center values, shape ``(nbins,)``.
    bin_edges : np.ndarray
        Bin edges, shape ``(nbins+1,)``.

    Example
    -------
    Reconstruct a full-sky binned map from valid-pixel results::

        import numpy as np
        import healpy as hp
        from furax_cs import bin_parameter_map

        nside = 64
        npix = hp.nside2npix(nside)
        mask = np.load("mask.npy")
        (valid,) = np.where(mask == 1)

        pixel_values = ...  # parameter values for valid pixels only
        patch_indices, centers, edges = bin_parameter_map(pixel_values, nbins=10)

        # Write back to a full-sky map (masked pixels = UNSEEN)
        out_map = np.full(npix, hp.UNSEEN)
        out_map[valid] = patch_indices.astype(float)
        np.save("patches_beta_dust.npy", out_map)
    """
    vmin = pixel_map.min()
    vmax = pixel_map.max()
    bin_edges = np.linspace(vmin, vmax + 1e-10, nbins + 1)
    bin_indices = np.digitize(pixel_map, bin_edges)
    bin_indices = np.clip(bin_indices, 1, nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_indices - 1, bin_centers, bin_edges  # 0-based
