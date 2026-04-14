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


def smooth_param_maps(
    param_fullsky: dict[str, np.ndarray],
    combined_mask: np.ndarray,
    combined_valid: np.ndarray,
    nside: int,
    fwhm_deg: float,
) -> dict[str, np.ndarray]:
    """Smooth each parameter map with a Gaussian beam and return smoothed valid-pixel values.

    Parameters
    ----------
    param_fullsky : dict[str, np.ndarray]
        Mapping param_name -> 1-D array of valid-pixel values (no UNSEEN).
    combined_mask : np.ndarray
        Full-sky binary mask, shape ``(npix,)``, 1 = valid.
    combined_valid : np.ndarray
        Indices of valid pixels, i.e. ``np.where(combined_mask == 1)[0]``.
    nside : int
        HEALPix resolution. Maps are assumed to be in RING ordering.
    fwhm_deg : float
        FWHM in degrees.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping param_name -> smoothed valid-pixel values (1-D, same length as input).
    """
    npix = hp.nside2npix(nside)
    fwhm_rad = np.radians(fwhm_deg)
    smoothed_valid: dict[str, np.ndarray] = {}
    for param_name in _ALL_PARAM_NAMES:
        valid_values = param_fullsky[param_name]
        # Fill masked pixels with the mean of valid pixels to minimize boundary ringing
        fill_value = float(np.mean(valid_values))
        full_map = np.full(npix, fill_value, dtype=np.float64)
        full_map[combined_valid] = valid_values.astype(np.float64)
        # pol=False: each spectral-param map is an independent scalar spin-0 field
        # Maps are RING-ordered (healpy default); hp.smoothing operates in RING
        smoothed = hp.smoothing(full_map, fwhm=fwhm_rad, pol=False)
        smoothed_valid[param_name] = smoothed[combined_valid]
    return smoothed_valid


def _plot_param_maps(
    fullsky: dict[str, np.ndarray],
    output_dir: str,
    filename: str,
    titles: list[str],
) -> None:
    """Plot a 1×3 mollview grid of parameter maps and save as *filename*."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    keys = ["beta_dust", "temp_dust", "beta_pl"]
    fig = plt.figure(figsize=(16, 8))
    for i, (key, title) in enumerate(zip(keys, titles)):
        hp.mollview(
            fullsky[key],
            title=title,
            sub=(1, 3, i + 1),
            bgcolor=(0.0,) * 4,
            cbar=True,
            format="%.4f",
            hold=False,
        )
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    success(f"Saved map plot: {out_path}")


def plot_original_maps(fullsky: dict[str, np.ndarray], output_dir: str) -> None:
    """Plot pre-smoothing parameter maps → original_params.png."""
    _plot_param_maps(
        fullsky,
        output_dir,
        "original_params.png",
        [r"$\beta_d$ (original)", r"$T_d$ [K] (original)", r"$\beta_s$ (original)"],
    )


def plot_smoothed_maps(fullsky: dict[str, np.ndarray], output_dir: str, fwhm_deg: float) -> None:
    """Plot post-smoothing parameter maps → smoothed_params.png."""
    _plot_param_maps(
        fullsky,
        output_dir,
        "smoothed_params.png",
        [
            rf"$\beta_d$ (FWHM={fwhm_deg}$^\circ$)",
            rf"$T_d$ [K] (FWHM={fwhm_deg}$^\circ$)",
            rf"$\beta_s$ (FWHM={fwhm_deg}$^\circ$)",
        ],
    )


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
    fwhm_deg: float | None = None,
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
    fwhm_deg : float | None
        If not None, smooth each parameter's full-sky map with a Gaussian
        beam of this FWHM (degrees) before binning.  Smoothed full-sky arrays
        (with ``hp.UNSEEN`` outside the mask) are saved as
        ``smoothed_{param}.npy`` and ``smoothed_maps.npz``, and a mollview
        plot is saved as ``smoothed_params.png``.

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
    # 2b. Optional HEALPix smoothing
    # ------------------------------------------------------------------
    if fwhm_deg is not None:
        # Save & plot original maps before smoothing
        orig_fullsky: dict[str, np.ndarray] = {}
        orig_npz: dict[str, np.ndarray] = {}
        for param_name in _ALL_PARAM_NAMES:
            om = np.full(npix, hp.UNSEEN)
            om[combined_valid] = param_fullsky[param_name]
            orig_fullsky[param_name] = om
            orig_npz[param_name] = om
            np.save(os.path.join(output_dir, f"original_{param_name}.npy"), om)
        np.savez(os.path.join(output_dir, "original_maps.npz"), **orig_npz)
        info("Saved original_maps.npz")
        plot_original_maps(orig_fullsky, output_dir)

        info(f"Smoothing parameter maps with FWHM = {fwhm_deg} deg")
        param_fullsky = smooth_param_maps(
            param_fullsky, combined_mask, combined_valid, nside, fwhm_deg
        )
        # Reconstruct full-sky smoothed maps (UNSEEN outside mask) for saving/plotting
        smoothed_fullsky: dict[str, np.ndarray] = {}
        npz_data: dict[str, np.ndarray] = {}
        for param_name in _ALL_PARAM_NAMES:
            sm = np.full(npix, hp.UNSEEN)
            sm[combined_valid] = param_fullsky[param_name]
            smoothed_fullsky[param_name] = sm
            npz_data[param_name] = sm
            np.save(os.path.join(output_dir, f"smoothed_{param_name}.npy"), sm)
        np.savez(os.path.join(output_dir, "smoothed_maps.npz"), **npz_data)
        info("Saved smoothed_maps.npz")
        plot_smoothed_maps(smoothed_fullsky, output_dir, fwhm_deg)

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
