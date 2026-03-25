"""Post-clustering parameter binning: produce patch index .npy files.

Reads original results.npz / mask.npy from one or more result folders,
bins spectral-parameter maps into equal-width bins, and writes full-sky
``patches_{param}.npy`` files that can be fed to ``kmeans-model -c``.
"""

from __future__ import annotations

import os
import time

import healpy as hp
import jax.numpy as jnp
import numpy as np
from jax_healpy.clustering import combine_masks

from ..binning import bin_parameter_map
from ..logging_utils import error, info, success, warning

# Mapping from bin_config param name -> (patch_key, param_key) in results.npz
_BIN_PARAM_KEYS: dict[str, tuple[str, str]] = {
    "beta_dust": ("beta_dust_patches", "beta_dust"),
    "temp_dust": ("temp_dust_patches", "temp_dust"),
    "beta_pl": ("beta_pl_patches", "beta_pl"),
}

_ALL_PARAM_NAMES = ["beta_dust", "temp_dust", "beta_pl"]


def _squeeze_patches(arr: np.ndarray) -> np.ndarray:
    """Squeeze n_gridpts=1 leading dim from patch arrays if present."""
    if arr.ndim > 1:
        return arr[0]
    return arr


def run_bin(
    folders: list[str],
    nside: int,
    output_dir: str,
    bin_config: dict[str, int],
    noise_selection: str = "min-value",
) -> int:
    """Bin spectral parameters and write ``patches_{param}.npy`` files.

    All *folders* are combined (disjoint masks stitched via
    ``combine_masks``) before binning.  The output is written directly to
    *output_dir*.

    Parameters
    ----------
    folders : list[str]
        Paths to result folders (each containing results.npz + mask.npy).
    nside : int
        HEALPix resolution.
    output_dir : str
        Directory for output ``.npy`` files.
    bin_config : dict[str, int]
        Parameter name -> number of bins.
    noise_selection : str
        Strategy to pick reference realization: ``'min-value'``, ``'min-nll'``,
        or an integer index.

    Returns
    -------
    int
        0 on success.
    """
    os.makedirs(output_dir, exist_ok=True)
    npix = hp.nside2npix(nside)
    t0 = time.perf_counter()

    info(f"Bin config: {bin_config}")
    info(f"Processing {len(folders)} folder(s)")

    # ------------------------------------------------------------------
    # 1. Load all folders, build per-folder cutout parameter maps
    # ------------------------------------------------------------------
    all_cutouts: dict[str, list[np.ndarray]] = {p: [] for p in _ALL_PARAM_NAMES}
    all_indices: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    for folder in folders:
        try:
            results = dict(np.load(f"{folder}/results.npz"))
            mask = np.load(f"{folder}/mask.npy")
        except (FileNotFoundError, OSError) as e:
            warning(f"Skipping {folder}: {e}")
            continue

        all_masks.append(mask)
        (indices,) = jnp.where(mask == 1)
        all_indices.append(indices)

        run_index = 0
        # Select reference realization
        if noise_selection == "min-nll":
            ref_idx = int(np.argmin(results["NLL"][run_index]))
        elif noise_selection == "min-value":
            ref_idx = int(np.argmin(results["value"][run_index]))
        else:
            ref_idx = int(noise_selection)

        for param_name in _ALL_PARAM_NAMES:
            patch_key, value_key = _BIN_PARAM_KEYS[param_name]
            patches = _squeeze_patches(results[patch_key])  # (npix_masked,)
            params = results[value_key][run_index, ref_idx]  # (n_clusters,)
            pixel_values = params[patches]  # (npix_masked,)
            all_cutouts[param_name].append(jnp.asarray(pixel_values))

    if not all_indices:
        error("No folders loaded successfully.")
        return 1

    # ------------------------------------------------------------------
    # 2. Combine cutouts → full-sky parameter maps via combine_masks
    # ------------------------------------------------------------------
    combined_mask = np.logical_or.reduce(all_masks).astype(float)
    (combined_valid,) = np.where(combined_mask == 1)

    param_fullsky: dict[str, np.ndarray] = {}
    for param_name in _ALL_PARAM_NAMES:
        full = np.asarray(combine_masks(all_cutouts[param_name], all_indices, nside))
        param_fullsky[param_name] = full[combined_valid]

    # ------------------------------------------------------------------
    # 3. Bin each parameter & write patches .npy
    # ------------------------------------------------------------------
    for param_name in _ALL_PARAM_NAMES:
        valid_values = param_fullsky[param_name]

        if param_name in bin_config:
            nbins = bin_config[param_name]
            bin_indices, _, _ = bin_parameter_map(valid_values, nbins)
            n_unique = len(np.unique(bin_indices))
            info(f"  {param_name}: {nbins} bins requested, {n_unique} unique")
        else:
            # No binning: renumber original values to contiguous 0-based
            _, bin_indices = np.unique(valid_values, return_inverse=True)
            info(f"  {param_name}: unbinned, {len(np.unique(bin_indices))} unique clusters")

        out_map = np.full(npix, hp.UNSEEN)
        out_map[combined_valid] = bin_indices.astype(float)
        np.save(os.path.join(output_dir, f"patches_{param_name}.npy"), out_map)

    # Save combined mask for reference
    np.save(os.path.join(output_dir, "mask.npy"), combined_mask)

    elapsed = time.perf_counter() - t0
    success(f"Wrote patches to {output_dir} in {elapsed:.2f}s")
    return 0
