#!/usr/bin/env python3
"""
K-Means Clustering for Adaptive CMB Component Separation

This script implements the main contribution of the FURAX framework: adaptive K-means
clustering for optimizing parametric component separation in CMB polarization analysis.
The method uses spherical K-means to partition the sky based on spatial coordinates,
then performs variance-based model selection to determine optimal cluster configurations.

Key Innovation:
    - Spherical K-means clustering using RA/Dec coordinates with 3D Cartesian averaging
    - Variance-based selection to minimize CMB reconstruction contamination
    - Distributed grid search across clustering configurations
    - Adaptive sky partitioning for spatially-varying spectral parameters

Usage:
    python 08-KMeans-model.py -n 64 -pc 100 5 1 -tag c1d1s1 -m GAL020 -i LiteBIRD

Parameters:
    -pc: Number of clusters for [dust_temp, dust_beta, sync_beta] parameters
    -tag: Sky simulation configuration tag
    -m: Galactic mask (GAL020, GAL040, GAL060)
    -i: Instrument specification

Output:
    Results saved to results/kmeans_{config}_{instrument}_{mask}_{samples}/
    - best_params.npz: Optimized spectral parameters per cluster
    - results.npz: Full clustering and component separation results
    - mask.npy: Sky mask used for analysis

Author: FURAX Team
"""

import os

os.environ["EQX_ON_ERROR"] = "nan"

import argparse
from functools import partial
from importlib.resources import files as pkg_files
from typing import Any

import jax
from jaxtyping import Array
from tqdm import tqdm

# =============================================================================
# 1. If running on a distributed system, initialize JAX distributed
# =============================================================================
if (
    int(os.environ.get("SLURM_NTASKS", 0)) > 1
    or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
):
    os.environ["VSCODE_PROXY_URI"] = ""
    os.environ["no_proxy"] = ""
    os.environ["NO_PROXY"] = ""
    del os.environ["VSCODE_PROXY_URI"]
    del os.environ["no_proxy"]
    del os.environ["NO_PROXY"]
    jax.distributed.initialize()
# =============================================================================
import operator
from time import perf_counter

import jax.numpy as jnp
import jax.random
import lineax as lx
import numpy as np
from furax import Config
from furax.obs import (
    negative_log_likelihood,
    sky_signal,
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
    minimize,
    sanitize_mask_name,
)
from furax_cs.logging_utils import info, success
from jax_healpy.clustering import get_cutout_from_mask, normalize_by_first_occurrence

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
    Parse command-line arguments for K-means clustering component separation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - nside: HEALPix resolution parameter
        - noise_sim: Number of noise simulations for MC analysis
        - noise_ratio: Noise level relative to signal
        - tag: Sky simulation configuration identifier
        - mask: Galactic mask choice (GAL020, GAL040, GAL060)
        - instrument: Instrument configuration (LiteBIRD, Planck)
        - patch_count: Target cluster counts for [dust_beta, dust_temp, sync_beta]
        - best_only: Flag to only compute optimal configuration
    """
    parser = argparse.ArgumentParser(
        description="K-Means Clustering for Adaptive CMB Component Separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  1. Basic adaptive clustering (100 beta_d, 50 temp_d, 50 beta_s clusters):
     kmeans-model -n 64 -pc 100 50 50 -m GAL020

  2. High-precision run with noise simulations:
     kmeans-model -n 128 -pc 500 100 100 -ns 10 -nr 0.1 -m GAL040

  3. Use specific instrument and sky model:
     kmeans-model -n 64 -i Planck -tag c1d0s1 -pc 50 10 10
""",
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
        help="Number of Monte Carlo noise realizations for statistical analysis",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.0,
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
        "-pc",
        "--patch-count",
        type=int,
        nargs=3,  # Expecting exactly three values
        default=[10000, 500, 500],  # Example target patch counts for beta_dust, temp_dust, beta_pl
        help=(
            "List of three target patch counts for beta_dust, temp_dust, and beta_pl. "
            "Example: --patch-count 10000 500 500"
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
            "Each value is 'true' (use precomputed pixel subsets from true params), "
            "an integer (K-means with N clusters), "
            "or a path to a .npy file (full-sky patches from 'r_analysis bin'). "
            "Example: -c true 10 5, or -c bd.npy td.npy bs.npy. "
            "Precomputed 'true' subsets only available for tag c1d1s1. "
            "Overrides -pc when provided."
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
        help="Maximum number of optimization iterations for L-BFGS solver",
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
        "-s",
        "--solver",
        type=str,
        default="optax_lbfgs",
        help="Solver for optimization. Options: optax_lbfgs, optax_lbfgs, "
        "optimistix_bfgs_wolfe, optimistix_lbfgs_wolfe, optimistix_ncg_hs_wolfe, "
        "scipy_tnc, zoom (alias), backtrack (alias), adam",
    )
    parser.add_argument(
        "-cond",
        "--cond",
        action="store_true",
        help="Enable conditioning (pre and post) for L-BFGS solver",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--vmap-batch",
        type=int,
        default=1,
        help="Number of noise simulations per vmap batch. "
        "Use 1 to disable vmap and run a serial for-loop of JIT'd single runs.",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=20,
        help="Steps after a constraint release before termination is allowed (active set solvers).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=10,
        help="Minimum iterations before termination is considered (active set solvers).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug printing for active set solvers.",
    )
    parser.add_argument(
        "-ls",
        "--linesearch",
        type=str,
        default="backtracking",
        choices=["backtracking", "zoom"],
        help="Linesearch strategy for active_set and optax_lbfgs solvers.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override the default output folder name.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-18,
        help="Absolute tolerance for optimizer convergence",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-16,
        help="Relative tolerance for optimizer convergence",
    )
    return parser.parse_args()


def main():
    """
    Main execution function for K-means clustering component separation.

    Implements the adaptive clustering algorithm that partitions the sky into
    regions with spatially-varying spectral parameters. Uses variance-based
    model selection to determine optimal cluster configurations.

    Algorithm Steps:
    1. Initialize sky masks and clustering parameters
    2. Load CMB and foreground simulations
    3. Perform spherical K-means clustering on sky coordinates
    4. Optimize spectral parameters within each cluster
    5. Evaluate CMB reconstruction variance for model selection
    6. Save optimal clustering configuration and parameters
    """
    # Step 1: Parse arguments and setup output directory
    args = parse_args()

    # Parse cluster specs (-c flag)
    cluster_specs = None
    if args.clusters is not None:
        cluster_specs = parse_cluster_specs(args.clusters)
        if any(v == "true" for v in cluster_specs.values()) and args.tag != "c1d1s1":
            raise ValueError(
                f"Precomputed pixel subsets are only available for tag 'c1d1s1', got '{args.tag}'"
            )

    if args.name is not None:
        out_folder = f"{args.output}/{args.name}"
    else:
        if cluster_specs is not None:

            def _spec_label(v):
                if v == "true":
                    return "true"
                if isinstance(v, str) and v.endswith(".npy"):
                    return "file"
                return str(v)

            bd_label = _spec_label(cluster_specs["beta_dust"])
            td_label = _spec_label(cluster_specs["temp_dust"])
            bs_label = _spec_label(cluster_specs["beta_pl"])
        else:
            bd_label = str(args.patch_count[0])
            td_label = str(args.patch_count[1])
            bs_label = str(args.patch_count[2])
        patches = f"BD{bd_label}_TD{td_label}_BS{bs_label}_SP{args.starting_params[0]}_{args.starting_params[1]}_{args.starting_params[2]}"
        config = (
            f"{args.solver}_cond{args.cond}_ls{args.linesearch}_noise{int(args.noise_ratio * 100)}"
            f"_cd{args.cooldown}_ms{args.min_steps}"
        )

        out_folder = f"{args.output}/kmeans_{args.tag}_{patches}_{args.instrument}_{sanitize_mask_name(args.mask)}_{config}"

    # Step 2: Initialize physical and computational parameters
    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0  # Dust reference frequency (GHz)
    synchrotron_nu0 = 20.0  # Synchrotron reference frequency (GHz)

    # Step 3: Load galactic mask and extract valid pixel indices
    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)  # Get indices of unmasked pixels

    # Step 4: Determine maximum cluster counts (limited by available pixels)
    # Load precomputed clusters if requested, otherwise use patch counts
    precomputed_clusters = {}  # param_name -> (cluster_array, n_clusters)
    if cluster_specs is not None:
        for param_name, spec in cluster_specs.items():
            if spec == "true":
                precomputed_clusters[param_name] = load_precomputed_clusters(param_name, indices)
            elif isinstance(spec, str) and spec.endswith(".npy"):
                precomputed_clusters[param_name] = load_patches_from_file(spec, indices)

    if cluster_specs is not None:
        B_dust_patches = (
            precomputed_clusters["beta_dust"][1]
            if "beta_dust" in precomputed_clusters
            else min(cluster_specs["beta_dust"], indices.size)
        )
        T_dust_patches = (
            precomputed_clusters["temp_dust"][1]
            if "temp_dust" in precomputed_clusters
            else min(cluster_specs["temp_dust"], indices.size)
        )
        B_synchrotron_patches = (
            precomputed_clusters["beta_pl"][1]
            if "beta_pl" in precomputed_clusters
            else min(cluster_specs["beta_pl"], indices.size)
        )
    else:
        B_dust_patches = min(args.patch_count[0], indices.size)
        T_dust_patches = min(args.patch_count[1], indices.size)
        B_synchrotron_patches = min(args.patch_count[2], indices.size)

    base_params = {
        "beta_dust": args.starting_params[0],
        "temp_dust": args.starting_params[1],
        "beta_pl": args.starting_params[2],
    }
    lower_bound = {
        "beta_dust": 0.5,
        "temp_dust": 10.0,
        "beta_pl": -7.0,
    }
    upper_bound = {
        "beta_dust": 3.0,
        "temp_dust": 40.0,
        "beta_pl": -0.5,
    }

    instrument = get_instrument(args.instrument)
    nu = instrument.frequency

    sky_signal_fn = partial(sky_signal, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)
    negative_log_likelihood_fn = partial(
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
    )

    _, freqmaps = load_from_cache(nside, instrument_name=args.instrument, sky=args.tag)
    _, fg_maps = load_fg_map(nside, instrument_name=args.instrument, sky=args.tag)
    cmb_map = load_cmb_map(nside, sky=args.tag)
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    fg_stokes = Stokes.from_stokes(fg_maps[:, 1], fg_maps[:, 2])
    cmb_map_stokes = Stokes.from_stokes(cmb_map[1], cmb_map[2])
    masked_d = get_cutout_from_mask(d, indices, axis=1)
    masked_fg = get_cutout_from_mask(fg_stokes, indices, axis=1)
    masked_cmb = get_cutout_from_mask(cmb_map_stokes, indices)

    max_count = {
        "beta_dust": B_dust_patches,
        "temp_dust": T_dust_patches,
        "beta_pl": B_synchrotron_patches,
    }
    max_patches = {
        "temp_dust_patches": max_count["temp_dust"],
        "beta_dust_patches": max_count["beta_dust"],
        "beta_pl_patches": max_count["beta_pl"],
    }

    options = {
        "linesearch": args.linesearch,
        "verbose_print": args.verbose,
        "cooldown": args.cooldown,
        "min_steps": args.min_steps,
    }

    def compute_minimum_variance(
        T_d_patches: int,
        B_d_patches: int,
        B_s_patches: int,
        planck_mask: Array,
        indices: Array,
        vmap_batch: int = 10,
    ) -> dict[str, Any]:
        n_regions = {
            "temp_dust_patches": T_d_patches,
            "beta_dust_patches": B_d_patches,
            "beta_pl_patches": B_s_patches,
        }

        # Build clusters: use precomputed where available, K-means for the rest
        _param_to_patch_key = {
            "beta_dust": "beta_dust_patches",
            "temp_dust": "temp_dust_patches",
            "beta_pl": "beta_pl_patches",
        }
        if precomputed_clusters:
            # Start with K-means for non-precomputed params
            kmeans_regions = {
                k: v
                for k, v in n_regions.items()
                if k not in {_param_to_patch_key[p] for p in precomputed_clusters}
            }
            if kmeans_regions:
                kmeans_max = {k: max_patches[k] for k in kmeans_regions}
                guess_clusters = kmeans_clusters(
                    jax.random.key(0), mask, indices, kmeans_regions, kmeans_max
                )
            else:
                guess_clusters = {}
            # Insert precomputed clusters
            for param_name, (cluster_arr, _) in precomputed_clusters.items():
                guess_clusters[_param_to_patch_key[param_name]] = cluster_arr
        else:
            guess_clusters = kmeans_clusters(
                jax.random.key(0), mask, indices, n_regions, max_patches
            )

        guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)
        lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        def single_run(noise_id):
            key = jax.random.PRNGKey(noise_id)
            noised_d, N, small_n = generate_noise_operator(
                key, noise_ratio, indices, nside, masked_d, instrument
            )

            final_params, final_state = minimize(
                fn=negative_log_likelihood_fn,
                init_params=guess_params,
                solver_name=args.solver,
                max_iter=args.max_iter,
                atol=args.atol,
                rtol=args.rtol,
                lower_bound=lower_bound_tree,
                upper_bound=upper_bound_tree,
                precondition=args.cond,
                options=options,
                nu=nu,
                N=N,
                d=noised_d,
                patch_indices=guess_clusters,
            )
            s = sky_signal_fn(final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters)
            cmb = s["cmb"]
            cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))

            cmb_np = jnp.stack([cmb.q, cmb.u])
            noise_d_np = jnp.stack([noised_d.q, noised_d.u])
            small_n_np = jnp.stack([small_n.q, small_n.u])

            nll = negative_log_likelihood_fn(
                final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters
            )

            return {
                "value": cmb_var,
                "CMB_O": cmb_np,
                "NLL": nll,
                "beta_dust": final_params["beta_dust"],
                "temp_dust": final_params["temp_dust"],
                "beta_pl": final_params["beta_pl"],
                "iter_num": final_state.iter_num,
                "NOISED_D": noise_d_np,
                "small_n": small_n_np,
            }

        if vmap_batch == 1:
            # Serial for-loop: JIT a single run at a time
            single_run_jit = jax.jit(single_run)
            results_list = list(
                tqdm(
                    [
                        single_run_jit(i)
                        for i in range(args.seed_start, args.seed_start + nb_noise_sim)
                    ],
                    desc="Running noise simulations",
                )
            )
            results = jax.tree.map(lambda *xs: jnp.stack(xs), *results_list)
        else:
            # Batched vmap: for-loop of JIT'd vmaps, each processing vmap_batch sims
            vmapped_run_jit = jax.jit(jax.vmap(single_run))
            seeds = list(range(args.seed_start, args.seed_start + nb_noise_sim))
            batches = []
            for i in tqdm(range(0, nb_noise_sim, vmap_batch), desc="Running noise simulations"):
                batch_seeds = jnp.array(seeds[i : i + vmap_batch])
                batches.append(vmapped_run_jit(batch_seeds))
            results = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batches)
        results["beta_dust_patches"] = guess_clusters["beta_dust_patches"]
        results["temp_dust_patches"] = guess_clusters["temp_dust_patches"]
        results["beta_pl_patches"] = guess_clusters["beta_pl_patches"]
        return results

    with Config(solver=lx.CG(atol=1e-10, rtol=1e-6, max_steps=1000)), jax.disable_jit(False):

        def objective_function(T_d_patches, B_d_patches, B_s_patches):
            return compute_minimum_variance(
                T_d_patches,
                B_d_patches,
                B_s_patches,
                mask,
                indices,
                vmap_batch=args.vmap_batch,
            )

        if not args.best_only:
            start_time = perf_counter()
            results = objective_function(T_dust_patches, B_dust_patches, B_synchrotron_patches)
            jax.tree.map(lambda x: x.block_until_ready(), results)
            end_time = perf_counter()
            min_bd, max_bd = results["beta_dust"].min(), results["beta_dust"].max()
            min_td, max_td = results["temp_dust"].min(), results["temp_dust"].max()
            min_bs, max_bs = results["beta_pl"].min(), results["beta_pl"].max()
            info(f"min beta_dust: {min_bd} max beta_dust: {max_bd}")
            info(f"min temp_dust: {min_td} max temp_dust: {max_td}")
            info(f"min beta_pl: {min_bs} max beta_pl: {max_bs}")
            info(f"Number of iterations: {results['iter_num'].mean()}")
            info(f"Objective function evaluation took {end_time - start_time:.2f} seconds")
            # Add a new axis to the results so it matches the shape of grid search results
            os.makedirs(out_folder, exist_ok=True)
            results = jax.tree.map(lambda x: x[np.newaxis, ...], results)
            np.savez(f"{out_folder}/results.npz", **results)

    os.makedirs(out_folder, exist_ok=True)
    # Save results and mask
    best_params = {}
    cmb_map = np.stack([masked_cmb.q, masked_cmb.u], axis=0)
    fg_map = np.stack([masked_fg.q, masked_fg.u], axis=1)
    d_map = np.stack([masked_d.q, masked_d.u], axis=1)
    best_params["I_CMB"] = cmb_map
    best_params["I_D"] = d_map
    best_params["I_D_NOCMB"] = fg_map

    np.savez(f"{out_folder}/best_params.npz", **best_params)
    np.save(f"{out_folder}/mask.npy", mask)
    success(f"Run complete. Results saved to {out_folder}")


if __name__ == "__main__":
    main()
