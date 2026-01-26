#!/usr/bin/env python3
"""
Distributed Grid Search for CMB Component Separation Parameter Optimization

WARNING: This script performs an exhaustive grid search across different sky regions
and can take SEVERAL HOURS to complete, especially when running on multiple GPUs.
Designed for HPC environments with SLURM scheduling.

This script implements distributed grid search optimization to find optimal spectral
parameters for CMB component separation across different galactic mask zones. It uses
JAX for GPU acceleration and distributed computing to efficiently explore parameter
space for dust and synchrotron foreground components.

Usage:
    python 04-distributed-gridding.py -n 64 -ns 100 -nr 1.0 -tag c1d1s1 -m GAL020 -i LiteBIRD

    # Dump default search space configuration to customize:
    python 04-distributed-gridding.py --dump-search-space my_search_space.yaml

    # Use custom search space:
    python 04-distributed-gridding.py -n 64 -ss my_search_space.yaml

Key Features:
    - Distributed execution across multiple GPUs using JAX
    - Grid search over dust temperature, dust spectral index, and synchrotron index
    - Automatic sky region partitioning based on galactic masks
    - Configurable search space via YAML files
    - Results saved in structured format for analysis

Author: FURAX Team
"""

import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
from functools import partial
from typing import Any

import jax
from jaxtyping import Array, Int

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

import jax.numpy as jnp
import jax.random
import numpy as np
from furax._instruments.sky import (
    get_noise_sigma_from_instrument,
)
from furax.obs import (
    negative_log_likelihood,
    sky_signal,
)
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from furax_cs.data.generate_maps import (
    MASK_CHOICES,
    get_mask,
    load_cmb_map,
    load_fg_map,
    load_from_cache,
    sanitize_mask_name,
)
from furax_cs.data.instruments import get_instrument
from furax_cs.data.search_space import dump_default_search_space, load_search_space
from furax_cs.logging_utils import info, success
from furax_cs.optim import minimize
from jax_grid_search import DistributedGridSearch
from jax_healpy.clustering import (
    find_kmeans_clusters,
    get_cutout_from_mask,
    normalize_by_first_occurrence,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods"
    )

    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=64,
        help="The nside of the map",
    )
    parser.add_argument(
        "-ns",
        "--noise-sim",
        type=int,
        default=1,
        help="Number of noise simulations",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.0,
        help="Noise ratio",
    )
    parser.add_argument(
        "-tag",
        "--tag",
        type=str,
        default="c1d1s1",
        help="Tag for the observation",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="GAL020_U",
        help=f"Mask to use. Available masks: {MASK_CHOICES}. "
        "Combine with + (union) or - (subtract), e.g., GAL020+GAL040 or ALL-GALACTIC",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
    )
    parser.add_argument(
        "-b",
        "--best-only",
        action="store_true",
        help="Only generate best results",
    )
    parser.add_argument(
        "-c",
        "--clean-up",
        type=str,
        nargs="?",
        default=None,
        help="Clean up the output folder",
    )
    parser.add_argument(
        "-mi",
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations for L-BFGS solver",
    )
    parser.add_argument(
        "-ss",
        "--search-space",
        type=str,
        default=None,
        help="Path to custom search space YAML file. If not provided, uses default configuration.",
    )
    parser.add_argument(
        "-d",
        "--dump-search-space",
        type=str,
        default=None,
        help="Dump the default search space configuration to specified YAML file and exit.",
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
    return parser.parse_args()


def clean_up(folder: str) -> None:
    batch_size = 50

    sorted_results = DistributedGridSearch.batched_stack_results(
        result_folder=folder, batch_size=batch_size
    )

    output_folder = os.path.join("final", folder)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "results.npz")

    np.savez(output_path, **sorted_results)
    success(f"Saved stacked results to {output_path}")


def main():
    args = parse_args()

    # Handle dump mode: save default search space and exit
    if args.dump_search_space is not None:
        dump_default_search_space(args.dump_search_space)
        success(f"\nSearch space template saved to: {args.dump_search_space}")
        info("You can now customize this file and use it with --search-space option.")
        return

    config = f"{args.solver}_cond{args.cond}_noise{int(args.noise_ratio * 100)}"
    out_folder = f"{args.output}/compsep_{args.tag}_SP{args.starting_params[0]}_{args.starting_params[1]}_{args.starting_params[2]}_{args.instrument}_{sanitize_mask_name(args.mask)}_{config}"

    if args.clean_up is not None:
        clean_up(args.clean_up)
        return

    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    base_params = {
        "beta_dust": args.starting_params[0],
        "temp_dust": args.starting_params[1],
        "beta_pl": args.starting_params[2],
    }
    lower_bound = {
        "beta_dust": 0.5,
        "temp_dust": 6.0,
        "beta_pl": -7.0,
    }
    upper_bound = {
        "beta_dust": 5.0,
        "temp_dust": 40.0,
        "beta_pl": -0.5,
    }

    instrument = get_instrument(args.instrument)
    nu = instrument.frequency
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, "QU")

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

    # Load search space configuration from YAML
    if args.search_space is not None:
        info(f"Loading custom search space from: {args.search_space}")
        search_space = load_search_space(args.search_space)
    else:
        info("Using default search space configuration")
        search_space = load_search_space()

    # Ensure we do not have more patches than pixels
    search_space = jax.tree.map(lambda x: jnp.clip(x, 1, indices.size), search_space)

    max_count = {
        "beta_dust": np.max(np.array(search_space["B_d_patches"])),
        "temp_dust": np.max(np.array(search_space["T_d_patches"])),
        "beta_pl": np.max(np.array(search_space["B_s_patches"])),
    }
    max_patches = {
        "temp_dust_patches": max_count["temp_dust"],
        "beta_dust_patches": max_count["beta_dust"],
        "beta_pl_patches": max_count["beta_pl"],
    }

    @partial(jax.jit, static_argnums=())
    def compute_minimum_variance(
        T_d_patches: Int[Array, " batch"],
        B_d_patches: Int[Array, " batch"],
        B_s_patches: Int[Array, " batch"],
        indices: Int[Array, " indices"],
    ) -> dict[str, Any]:
        T_d_patches = T_d_patches.squeeze()
        B_d_patches = B_d_patches.squeeze()
        B_s_patches = B_s_patches.squeeze()

        n_regions = {
            "temp_dust_patches": T_d_patches,
            "beta_dust_patches": B_d_patches,
            "beta_pl_patches": B_s_patches,
        }

        patch_indices = jax.tree.map(
            lambda c, mp: find_kmeans_clusters(
                mask, indices, c, jax.random.key(0), max_centroids=mp, initial_sample_size=1
            ),
            n_regions,
            max_patches,
        )
        guess_clusters = get_cutout_from_mask(patch_indices, indices)
        # Normalize the cluster to make indexing more logical
        guess_clusters = jax.tree.map(
            lambda g, c, mp: normalize_by_first_occurrence(g, c, mp).astype(jnp.int64),
            guess_clusters,
            n_regions,
            max_patches,
        )
        guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), guess_clusters)

        guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)
        lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        def single_run(noise_id):
            key = jax.random.PRNGKey(noise_id)
            white_noise = f_landscapes.normal(key) * noise_ratio
            white_noise = get_cutout_from_mask(white_noise, indices, axis=1)
            instrument = get_instrument(args.instrument)
            sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
            noise = white_noise * sigma
            noised_d = masked_d + noise

            small_n = (sigma * noise_ratio) ** 2
            small_n = 1.0 if noise_ratio == 0 else small_n

            N = NoiseDiagonalOperator(small_n, _in_structure=masked_d.structure)

            final_params, final_state = minimize(
                fn=negative_log_likelihood_fn,
                init_params=guess_params,
                solver_name=args.solver,
                max_iter=args.max_iter,
                atol=1e-15,
                rtol=1e-10,
                lower_bound=lower_bound_tree,
                upper_bound=upper_bound_tree,
                precondition=args.cond,
                nu=nu,
                N=N,
                d=noised_d,
                patch_indices=guess_clusters,
            )

            s = sky_signal_fn(final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters)
            cmb = s["cmb"]
            # Variance of the CMB map
            cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))
            # This is equivalent to jnp.var(cmb.q) + jnp.var(cmb.u)

            cmb_np = jnp.stack([cmb.q, cmb.u])

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
            }

        results = jax.vmap(single_run)(jnp.arange(nb_noise_sim))
        results["beta_dust_patches"] = guess_clusters["beta_dust_patches"]
        results["temp_dust_patches"] = guess_clusters["temp_dust_patches"]
        results["beta_pl_patches"] = guess_clusters["beta_pl_patches"]
        return results

    # Put the good values for the grid
    if os.path.exists(out_folder) and not args.best_only:
        old_results = DistributedGridSearch.batched_stack_results(
            result_folder=out_folder, batch_size=100
        )
    else:
        old_results = None

    @jax.jit
    def objective_function(T_d_patches, B_d_patches, B_s_patches):
        return compute_minimum_variance(
            T_d_patches,
            B_d_patches,
            B_s_patches,
            indices,
        )

    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=1,
        progress_bar=True,
        result_dir=out_folder,
        old_results=old_results,
    )
    info(f"Number of combinations: {grid_search.n_combinations}")
    if not args.best_only:
        grid_search.run()

    if not args.best_only:
        results = grid_search.stack_results(result_folder=out_folder)
        np.savez(f"{out_folder}/results.npz", **results)

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
