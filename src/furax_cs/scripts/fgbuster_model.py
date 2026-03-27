#!/usr/bin/env python3
"""
FGBuster Component Separation with Multi-Resolution Clustering

This script implements multi-resolution clustering for optimizing parametric
component separation in CMB polarization analysis using FGBuster.
The method uses HEALPix ud_grade to partition the sky at different resolutions
for each spectral parameter, then performs component separation using
FGBuster's adaptive_comp_sep function.

This is the NumPy/FGBuster equivalent of ptep_model.py which uses JAX/Furax.

Usage:
    python fgbuster_model.py -n 64 -ud 64 32 16 -tag c1d1s1 -m GAL020 -i LiteBIRD

Parameters:
    -ud: Target nside values for [beta_dust, temp_dust, beta_pl] ud_grade clustering
    -tag: Sky simulation configuration tag
    -m: Galactic mask (GAL020, GAL040, GAL060)
    -i: Instrument specification

Output:
    Results saved to results/fgbuster_{config}_{instrument}_{mask}_{samples}/
    - best_params.npz: Optimized spectral parameters per cluster
    - results.npz: Full clustering and component separation results
    - mask.npy: Sky mask used for analysis

Note:
    FGBuster operates with NumPy, not JAX. Multiple noise simulations are not
    supported in this implementation (use ptep_model.py for Monte Carlo analysis).

Author: FURAX Team
"""

import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
from importlib.resources import files as pkg_files
from time import perf_counter
from typing import Any

import healpy as hp
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from jaxtyping import Array

try:
    from fgbuster import (
        CMB,
        Dust,
        Synchrotron,
        adaptive_comp_sep,
    )
    from fgbuster import (
        get_instrument as get_fgbuster_instrument,
    )
except ImportError:
    raise ImportError(
        "FGBuster is required for this baseline comparison script. Install with:\n"
        "  pip install fgbuster\n"
        "or\n"
        "  pip install git+https://github.com/fgbuster/fgbuster.git"
    )

from furax.obs.stokes import Stokes
from furax_cs import (
    MASK_CHOICES,
    generate_noise_operator,
    get_instrument,
    get_mask,
    kmeans_clusters,
    load_cmb_map,
    load_fg_map,
    load_from_cache,
    multires_clusters,
    sanitize_mask_name,
)
from furax_cs.logging_utils import info, success
from jax_healpy.clustering import (
    get_cutout_from_mask,
    get_fullmap_from_cutout,
    normalize_by_first_occurrence,
)

jax.config.update("jax_enable_x64", True)

# Mapping from parameter name to precomputed pixel subset filename
_PIXEL_SUBSET_FILES = {
    "beta_dust": "pixel_subsets_from_true_d1_s1_Bd_nbins100.npy",
    "temp_dust": "pixel_subsets_from_true_d1_s1_Td_nbins100.npy",
    "beta_pl": "pixel_subsets_from_true_d1_s1_Bs_nbins100.npy",
}


def load_precomputed_clusters(param_name: str, indices: Array) -> tuple[Array, int]:
    """Load precomputed pixel subset clusters for a parameter.

    Returns the masked+normalized cluster array and the number of unique clusters.
    """
    filename = _PIXEL_SUBSET_FILES[param_name]
    path = pkg_files("furax_cs").joinpath("data", "pixelsubset", filename)
    full_sky = jnp.array(np.load(str(path)))
    masked = full_sky[indices]
    n_clusters = int(jnp.unique(masked).size)
    normalized = normalize_by_first_occurrence(masked, n_clusters, n_clusters).astype(jnp.int64)
    return normalized, n_clusters


def load_patches_from_file(path: str, indices: Array) -> tuple[Array, int]:
    """Load a full-sky patches .npy file and extract valid pixels.

    The file should be a float64 array of shape ``(npix,)`` where valid pixels
    have bin indices (0.0, 1.0, ...) and masked pixels have ``hp.UNSEEN``.

    Returns the masked+normalized cluster array and the number of unique clusters.
    """
    full_sky = jnp.array(np.load(path))
    masked = full_sky[indices].astype(jnp.int64)
    n_clusters = int(jnp.unique(masked).size)
    normalized = normalize_by_first_occurrence(masked, n_clusters, n_clusters).astype(jnp.int64)
    normalized = masked.astype(jnp.int64)
    return normalized, n_clusters


def parse_cluster_specs(cluster_args: list[str]) -> dict[str, str | int]:
    """Parse the -c argument values into a dict keyed by parameter name.

    Returns dict mapping param name -> 'true', int count, or .npy file path.
    """
    param_names = ["beta_dust", "temp_dust", "beta_pl"]
    specs = {}
    for name, val in zip(param_names, cluster_args):
        if val.lower() == "true":
            specs[name] = "true"
        elif val.endswith(".npy"):
            specs[name] = val
        else:
            specs[name] = int(val)
    return specs


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for FGBuster multi-resolution component separation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - nside: HEALPix resolution parameter
        - noise_sim: Number of noise simulations (must be 1 for FGBuster)
        - noise_ratio: Noise level relative to signal
        - tag: Sky simulation configuration identifier
        - mask: Galactic mask choice (GAL020, GAL040, GAL060)
        - instrument: Instrument configuration (LiteBIRD, Planck)
        - target_ud_grade: Target nside values for [beta_dust, temp_dust, beta_pl]
        - best_only: Flag to only compute optimal configuration
    """
    parser = argparse.ArgumentParser(
        description="FGBuster Component Separation with Multi-Resolution Clustering"
    )

    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=64,
        help="HEALPix nside parameter determining map resolution (nside=64 → ~55 arcmin pixels)",
    )
    parser.add_argument(
        "-ns",
        "--noise-sim",
        type=int,
        default=1,
        help="Number of noise simulations (must be 1 for FGBuster implementation)",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.1,
        help="Noise level as fraction of signal RMS (0.2 = 20%% noise)",
    )
    parser.add_argument(
        "-ss",
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed for noise simulations",
    )
    parser.add_argument(
        "-tag",
        "--tag",
        type=str,
        default="c1d1s1",
        help="Sky simulation tag: c(CMB)d(dust)s(synchrotron) with 0/1 for off/on",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="GAL020_U",
        help=f"Galactic mask: GAL020/040/060 (20%%/40%%/60%% sky coverage), _U/_L for upper/lower. "
        f"Available masks: {MASK_CHOICES}. "
        "Combine with + (union) or - (subtract), e.g., GAL020+GAL040 or ALL-GALACTIC",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
        help="Instrument configuration with frequency bands and noise characteristics",
    )
    parser.add_argument(
        "-ud",
        "--target-ud-grade",
        type=float,
        nargs=3,
        default=[64, 32, 16],
        help=(
            "List of three target nside values (for ud_grade downgrading) corresponding to "
            "beta_dust, temp_dust, beta_pl respectively. Used when -pc and -c are not provided."
        ),
    )
    parser.add_argument(
        "-pc",
        "--patch-count",
        type=int,
        nargs=3,
        default=None,
        help=(
            "K-means patch counts for [beta_dust, temp_dust, beta_pl]. "
            "When provided, uses K-means clustering instead of ud_grade. "
            "Example: --patch-count 100 50 50"
        ),
    )
    parser.add_argument(
        "-c",
        "--clusters",
        type=str,
        nargs=3,
        default=None,
        help=(
            "Cluster source for [beta_dust, temp_dust, beta_pl]. "
            "Each value is 'true' (precomputed pixel subsets from true params), "
            "an integer (K-means with N clusters), "
            "or a path to a .npy file (full-sky patches). "
            "Overrides -pc when provided. "
            "Precomputed 'true' subsets only available for tag c1d1s1."
        ),
    )
    parser.add_argument(
        "-b",
        "--best-only",
        action="store_true",
        help="Only generate best results",
    )
    parser.add_argument(
        "-mi",
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations for TNC solver",
    )
    parser.add_argument(
        "-sp",
        "--starting-params",
        type=float,
        nargs=3,
        default=[1.54, 20.0, -3.0],
        help="Starting parameters for [beta_dust, temp_dust, beta_pl]",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override the default output folder name.",
    )
    return parser.parse_args()


def run_fgbuster_comp_sep(
    freq_maps: Array,
    patch_ids_fg: list[Any],
    components: list[Any],
    instrument: Any,
    max_iter: int = 1000,
    tol: float = 1e-18,
) -> Any:
    """
    Run FGBuster adaptive component separation.

    Parameters
    ----------
    freq_maps : ndarray
        Frequency maps with shape (n_freq, n_stokes, n_pix) or (n_freq, 2, n_pix) for QU
    patch_ids_fg : list of ndarray
        List of patch indices for [temp_dust, beta_dust, beta_pl]
    instrument : dict
        FGBuster instrument specification
    max_iter : int
        Maximum optimization iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    result : object
        FGBuster result object containing:
        - x: optimized parameters
        - s: separated components (CMB, dust, synchrotron)
        - fun: final objective function value
    """

    # FGBuster bounds: [beta_dust, temp_dust, beta_pl]
    bounds = [(0.0, 5.0), (10.0, 40.0), (-6.0, 0.0)]

    options = {
        "disp": False,
        "gtol": tol,
        "eps": tol,
        "maxfun": max_iter * 10,
        "ftol": tol,
        "xtol": tol,
    }
    method = "TNC"

    # Convert to numpy arrays for FGBuster
    freq_maps_np = np.asarray(freq_maps)
    patch_ids_np = [np.asarray(p) for p in patch_ids_fg]

    result = adaptive_comp_sep(
        components,
        instrument,
        freq_maps_np,
        patch_ids_np,
        bounds=bounds,
        options=options,
        method=method,
        tol=tol,
    )

    return result


def main():
    """
    Main execution function for FGBuster multi-resolution component separation.

    Implements the multi-resolution clustering algorithm that partitions the sky into
    regions with spatially-varying spectral parameters using FGBuster's
    adaptive_comp_sep function.

    Algorithm Steps:
    1. Initialize sky masks and clustering parameters
    2. Load CMB and foreground simulations
    3. Perform multi-resolution clustering via HEALPix ud_grade
    4. Run FGBuster adaptive component separation
    5. Evaluate CMB reconstruction variance
    6. Save clustering configuration and parameters
    """
    # Step 1: Parse arguments and validate
    args = parse_args()

    # Parse cluster specs (-c flag)
    cluster_specs = None
    if args.clusters is not None:
        cluster_specs = parse_cluster_specs(args.clusters)
        if any(v == "true" for v in cluster_specs.values()) and args.tag != "c1d1s1":
            raise ValueError(
                f"Precomputed pixel subsets are only available for tag 'c1d1s1', got '{args.tag}'"
            )

    # Determine clustering mode: cluster_specs (-c) > kmeans (-pc) > multires (-ud)
    use_kmeans = cluster_specs is not None or args.patch_count is not None

    # Setup output directory
    if args.name is not None:
        out_folder = f"{args.output}/{args.name}"
    elif use_kmeans:

        def _spec_label(v):
            if v == "true":
                return "true"
            if isinstance(v, str) and v.endswith(".npy"):
                return "file"
            return str(v)

        if cluster_specs is not None:
            bd_label = _spec_label(cluster_specs["beta_dust"])
            td_label = _spec_label(cluster_specs["temp_dust"])
            bs_label = _spec_label(cluster_specs["beta_pl"])
        else:
            bd_label = str(args.patch_count[0])
            td_label = str(args.patch_count[1])
            bs_label = str(args.patch_count[2])
        patches = f"BD{bd_label}_TD{td_label}_BS{bs_label}"
        out_folder = f"{args.output}/fgbuster_kmeans_{args.tag}_{patches}_{args.instrument}_{sanitize_mask_name(args.mask)}_{int(args.noise_ratio * 100)}"
    else:
        ud_grades = f"BD{int(args.target_ud_grade[0])}_TD{int(args.target_ud_grade[1])}_BS{int(args.target_ud_grade[2])}"
        out_folder = f"{args.output}/fgbuster_multires_{args.tag}_{ud_grades}_{args.instrument}_{sanitize_mask_name(args.mask)}_{int(args.noise_ratio * 100)}"

    # Step 2: Initialize physical and computational parameters
    nside = args.nside
    dust_nu0 = 160.0  # Dust reference frequency (GHz) - Match kmeans-model
    synchrotron_nu0 = 20.0  # Synchrotron reference frequency (GHz)

    # Get instrument for noise generation
    furax_instrument = get_instrument(args.instrument)

    # Step 3: Load galactic mask and extract valid pixel indices
    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)
    (unseen_indices,) = jnp.where(mask != 1)

    # Step 4: Load frequency maps and component maps
    info(f"Loading data for nside={nside}, tag={args.tag}, instrument={args.instrument}")
    _, freqmaps = load_from_cache(nside, instrument_name=args.instrument, sky=args.tag)
    _, fg_maps = load_fg_map(nside, instrument_name=args.instrument, sky=args.tag)
    cmb_map = load_cmb_map(nside, sky=args.tag)

    # Create Stokes objects for data handling
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    fg_stokes = Stokes.from_stokes(fg_maps[:, 1], fg_maps[:, 2])
    cmb_map_stokes = Stokes.from_stokes(cmb_map[1], cmb_map[2])

    # FGBuster PROCESSING: Uses full HEALPix maps with hp.UNSEEN for masked pixels
    # This is required by FGBuster's adaptive_comp_sep function
    # Noise is added to full maps before masking
    masked_d = jax.tree.map(lambda x: x.at[..., unseen_indices].set(hp.UNSEEN), d)
    masked_fg = jax.tree.map(lambda x: x.at[..., unseen_indices].set(hp.UNSEEN), fg_stokes)
    masked_cmb = jax.tree.map(lambda x: x.at[..., unseen_indices].set(hp.UNSEEN), cmb_map_stokes)

    # to numpy for FGBuster
    masked_d = jax.tree.map(np.asarray, masked_d)
    masked_fg = jax.tree.map(np.asarray, masked_fg)
    masked_cmb = jax.tree.map(np.asarray, masked_cmb)

    d_cutout = get_cutout_from_mask(d, indices, axis=1)

    # Step 5: Perform clustering for patch indices (mode-dependent)
    if use_kmeans:
        # --- K-means / precomputed mode ---
        if cluster_specs is not None:
            patch_counts = {
                "beta_dust": (
                    cluster_specs["beta_dust"]
                    if isinstance(cluster_specs["beta_dust"], int)
                    else None
                ),
                "temp_dust": (
                    cluster_specs["temp_dust"]
                    if isinstance(cluster_specs["temp_dust"], int)
                    else None
                ),
                "beta_pl": (
                    cluster_specs["beta_pl"] if isinstance(cluster_specs["beta_pl"], int) else None
                ),
            }
        else:
            patch_counts = {
                "beta_dust": args.patch_count[0],
                "temp_dust": args.patch_count[1],
                "beta_pl": args.patch_count[2],
            }

        # Load precomputed clusters where requested
        precomputed_clusters: dict[str, tuple] = {}
        if cluster_specs is not None:
            for param_name, spec in cluster_specs.items():
                if spec == "true":
                    precomputed_clusters[param_name] = load_precomputed_clusters(
                        param_name, indices
                    )
                elif isinstance(spec, str) and spec.endswith(".npy"):
                    precomputed_clusters[param_name] = load_patches_from_file(spec, indices)

        # Determine n_clusters per param (for max_patches)
        _param_to_patch_key = {
            "beta_dust": "beta_dust_patches",
            "temp_dust": "temp_dust_patches",
            "beta_pl": "beta_pl_patches",
        }
        max_count = {}
        for param_name, patch_key in _param_to_patch_key.items():
            if param_name in precomputed_clusters:
                max_count[patch_key] = precomputed_clusters[param_name][1]
            else:
                max_count[patch_key] = min(patch_counts[param_name], indices.size)

        # Build K-means clusters for non-precomputed params
        kmeans_regions = {
            k: v
            for k, v in max_count.items()
            if k not in {_param_to_patch_key[p] for p in precomputed_clusters}
        }
        if kmeans_regions:
            info(
                "Computing K-means clusters: "
                + ", ".join(f"{k}={v}" for k, v in kmeans_regions.items())
            )
            cutout_clusters = kmeans_clusters(
                jax.random.key(0), mask, indices, kmeans_regions, max_count
            )
        else:
            cutout_clusters = {}

        # Insert precomputed clusters
        for param_name, (cluster_arr, _) in precomputed_clusters.items():
            cutout_clusters[_param_to_patch_key[param_name]] = cluster_arr

    else:
        # --- Multi-resolution (ud_grade) mode ---
        target_ud_grade = {
            "beta_dust": int(args.target_ud_grade[0]),
            "temp_dust": int(args.target_ud_grade[1]),
            "beta_pl": int(args.target_ud_grade[2]),
        }
        info(
            f"Computing multi-resolution clusters: beta_dust@NS{target_ud_grade['beta_dust']}, "
            f"temp_dust@NS{target_ud_grade['temp_dust']}, beta_pl@NS{target_ud_grade['beta_pl']}"
        )
        cutout_clusters = multires_clusters(mask, indices, target_ud_grade, nside)

    # Convert cutout clusters to full-sky maps for FGBuster
    # FGBuster needs full-sky maps where masked pixels have max cluster index
    guess_clusters = jax.tree.map(
        lambda x: get_fullmap_from_cutout(x, indices, nside),
        cutout_clusters,
    )
    # Ensure masked pixels have the max cluster index
    guess_clusters = jax.tree.map(
        lambda x: jnp.where(mask == 1, x, x.max()).astype(jnp.int64), guess_clusters
    )
    patch_ids_fg = [
        np.asarray(guess_clusters["beta_dust_patches"]),
        np.asarray(guess_clusters["temp_dust_patches"]),
        np.asarray(guess_clusters["beta_pl_patches"]),
    ]

    # Get FGBuster instrument
    instrument = get_fgbuster_instrument(args.instrument)

    # Storage for results across simulations
    results_storage = {
        "value": [],
        "CMB_O": [],
        "NLL": [],
        "beta_dust": [],
        "temp_dust": [],
        "beta_pl": [],
        "NOISED_D": [],
        "small_n": [],
    }

    from tqdm import tqdm

    # Step 6 & 7: Loop over noise simulations and Run FGBuster
    info(f"Running {args.noise_sim} noise simulations with FGBuster")

    for sim_idx in tqdm(
        range(args.seed_start, args.seed_start + args.noise_sim), desc="FGBuster Simulations"
    ):
        # Generate noise on cutout using standard Furax generator
        # This ensures exact match with ptep_model.py
        key = jax.random.PRNGKey(sim_idx)
        noised_d_cutout, _, small_n_cutout = generate_noise_operator(
            key, args.noise_ratio, indices, nside, d_cutout, furax_instrument
        )
        # Expand to full map for FGBuster (with UNSEEN in masked pixels)
        noised_d_full = get_fullmap_from_cutout(noised_d_cutout, indices, nside, axis=1)
        noised_d_masked = jax.tree.map(
            lambda x: x.at[..., unseen_indices].set(hp.UNSEEN), noised_d_full
        )
        noised_d_np = jax.tree.map(np.asarray, noised_d_masked)

        # Prepare frequency maps for FGBuster
        # FGBuster expects shape (n_freq, n_stokes, n_pix) with Q, U stokes
        freq_maps_fg = jnp.stack([noised_d_np.q, noised_d_np.u], axis=1)
        freq_maps_cutout_fg = jnp.stack([noised_d_cutout.q, noised_d_cutout.u], axis=0)
        small_n_np = jnp.stack([small_n_cutout.q, small_n_cutout.u], axis=0)

        # Run FGBuster component separation
        components = [CMB(), Dust(dust_nu0), Synchrotron(synchrotron_nu0)]
        components[1]._set_default_of_free_symbols(
            beta_d=args.starting_params[0], temp=args.starting_params[1]
        )
        components[2]._set_default_of_free_symbols(beta_pl=args.starting_params[2])

        start_time = perf_counter()
        result = run_fgbuster_comp_sep(
            freq_maps_fg,
            patch_ids_fg,
            components,
            instrument,
            max_iter=args.max_iter,
            tol=1e-18,
        )
        end_time = perf_counter()
        info(f"Run {sim_idx} component separation took {end_time - start_time:.2f} seconds")

        # Step 8: Extract results
        # OUTPUT CONVERSION: Convert FGBuster outputs from full maps to cutouts
        cmb_q_full, cmb_u_full = result.s[0]  # CMB is the first component (full map)

        # Convert to Stokes and extract cutout (only unmasked pixels)
        cmb_result_stokes = Stokes.from_stokes(cmb_q_full, cmb_u_full)
        cmb_cutout = get_cutout_from_mask(cmb_result_stokes, indices)
        cmb_q, cmb_u = cmb_cutout.q, cmb_cutout.u

        # Compute variance on cutout (not full map)
        cmb_var = np.var(cmb_q) + np.var(cmb_u)
        cmb_np = np.stack([cmb_q, cmb_u])  # Already cutout

        # Store results
        results_storage["value"].append(cmb_var)
        results_storage["CMB_O"].append(cmb_np)
        results_storage["NLL"].append(result.fun)
        results_storage["beta_dust"].append(result.x[0])
        results_storage["temp_dust"].append(result.x[1])
        results_storage["beta_pl"].append(result.x[2])
        results_storage["NOISED_D"].append(freq_maps_cutout_fg)
        results_storage["small_n"].append(small_n_np)

    # Convert lists to arrays and add axes to match kmeans_model format
    # Target shape: (1, ns, ...)
    results = {
        "update_history": np.zeros((1, args.noise_sim, args.max_iter, 2)),  # Placeholder
        "value": np.array(results_storage["value"])[np.newaxis, ...],
        "CMB_O": np.array(results_storage["CMB_O"])[np.newaxis, ...],
        "NLL": np.array(results_storage["NLL"])[np.newaxis, ...],
        "beta_dust": np.array(results_storage["beta_dust"])[np.newaxis, ...],
        "temp_dust": np.array(results_storage["temp_dust"])[np.newaxis, ...],
        "beta_pl": np.array(results_storage["beta_pl"])[np.newaxis, ...],
        "beta_dust_patches": np.asarray(cutout_clusters["beta_dust_patches"])[np.newaxis, ...],
        "temp_dust_patches": np.asarray(cutout_clusters["temp_dust_patches"])[np.newaxis, ...],
        "beta_pl_patches": np.asarray(cutout_clusters["beta_pl_patches"])[np.newaxis, ...],
        "NOISED_D": np.array(results_storage["NOISED_D"])[np.newaxis, ...],
        "small_n": np.array(results_storage["small_n"])[np.newaxis, ...],
    }

    info("Component separation complete.")
    info(f"Average CMB variance: {results['value'].mean():.6e}")

    os.makedirs(out_folder, exist_ok=True)
    if not args.best_only:
        np.savez(f"{out_folder}/results.npz", **results)

    # Save best parameters and auxiliary data
    # Convert masked maps (full HEALPix with UNSEEN) to cutouts for r_analysis
    masked_cmb_cutout = get_cutout_from_mask(masked_cmb, indices)
    masked_fg_cutout = get_cutout_from_mask(masked_fg, indices, axis=1)
    masked_d_cutout = get_cutout_from_mask(masked_d, indices, axis=1)

    best_params = {}
    best_params["I_CMB"] = np.stack(
        [np.asarray(masked_cmb_cutout.q), np.asarray(masked_cmb_cutout.u)], axis=0
    )
    best_params["I_D"] = np.stack(
        [np.asarray(masked_d_cutout.q), np.asarray(masked_d_cutout.u)], axis=1
    )
    best_params["I_D_NOCMB"] = np.stack(
        [np.asarray(masked_fg_cutout.q), np.asarray(masked_fg_cutout.u)], axis=1
    )

    np.savez(f"{out_folder}/best_params.npz", **best_params)
    np.save(f"{out_folder}/mask.npy", np.asarray(mask))

    success(f"Run complete. Results saved to {out_folder}")


if __name__ == "__main__":
    main()
