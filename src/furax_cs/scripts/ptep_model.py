import os

os.environ["EQX_ON_ERROR"] = "nan"

import argparse
import operator
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from furax.obs import negative_log_likelihood, sky_signal
from furax.obs.stokes import Stokes
from furax_cs import generate_noise_operator, multires_clusters
from furax_cs.data.generate_maps import (
    MASK_CHOICES,
    get_mask,
    load_cmb_map,
    load_fg_map,
    load_from_cache,
    sanitize_mask_name,
)
from furax_cs.data.instruments import get_instrument
from furax_cs.logging_utils import info, success
from furax_cs.optim import minimize
from jax_healpy.clustering import get_cutout_from_mask
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Benchmark FGBuster and Furax Component Separation Methods (single run with ud_grade)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  1. Multi-resolution separation (Beta_d @ NS64, Temp_d @ NS32, Beta_s @ NS16):
     ptep-model -n 64 -ud 64 32 16 -m GAL020

  2. Uniform resolution (equivalent to standard parametric separation):
     ptep-model -n 64 -ud 64 64 64

  3. Low-resolution synchrotron model with Planck instrument:
     ptep-model -n 128 -i Planck -ud 128 64 16
""",
    )

    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=64,
        help="The nside of the input map",
    )
    parser.add_argument(
        "-ns",
        "--noise-sim",
        type=int,
        default=1,
        help="Number of noise simulations (single run uses 1)",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.0,
        help="Noise ratio",
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
        help="Instrument to use",
    )
    parser.add_argument(
        "-ud",
        "--target-ud-grade",
        type=float,
        nargs=3,  # Expecting exactly three values
        default=[64, 32, 16],  # Example target nsides for beta_dust, temp_dust, beta_pl
        help=(
            "List of three target nside values (for ud_grade downgrading) corresponding to "
            "beta_dust, temp_dust, beta_pl respectively"
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
        "-v",
        "--use-vmap",
        action="store_true",
        help="Use jax.vmap instead of for-loop for noise simulations. "
        "Only activate this option when using JAX JIT cache (persistent compilation cache) "
        "to avoid recompilation overhead on each run.",
    )
    parser.add_argument(
        "-top_k",
        "--top-k-release",
        type=float,
        default=None,
        help="Fraction of constraints to release in active set solver (e.g., 0.1 for 10%%).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override the default output folder name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Define the output folder and create it if necessary
    if args.name is not None:
        out_folder = f"{args.output}/{args.name}"
    else:
        ud_grades = f"BD{int(args.target_ud_grade[0])}_TD{int(args.target_ud_grade[1])}_BS{int(args.target_ud_grade[2])}_SP{args.starting_params[0]}_{args.starting_params[1]}_{args.starting_params[2]}"
        config = f"{args.solver}_cond{args.cond}_noise{int(args.noise_ratio * 100)}"
        if args.top_k_release is not None:
            config += f"_topk{args.top_k_release}"
        out_folder = f"{args.output}/ptep_{args.tag}_{ud_grades}_{args.instrument}_{sanitize_mask_name(args.mask)}_{config}"

    # Set up parameters
    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    # Get the mask and its indices
    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    # Load frequency maps and extract the Stokes Q/U maps (for example)
    _, freqmaps = load_from_cache(nside, instrument_name=args.instrument, sky=args.tag)
    _, fg_maps = load_fg_map(nside, instrument_name=args.instrument, sky=args.tag)
    cmb_map = load_cmb_map(nside, sky=args.tag)
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    fg_stokes = Stokes.from_stokes(fg_maps[:, 1], fg_maps[:, 2])
    cmb_map_stokes = Stokes.from_stokes(cmb_map[1], cmb_map[2])
    masked_d = get_cutout_from_mask(d, indices, axis=1)
    masked_fg = get_cutout_from_mask(fg_stokes, indices, axis=1)
    masked_cmb = get_cutout_from_mask(cmb_map_stokes, indices)

    # Use multi-resolution clustering with ud_grade
    target_ud_grade = {
        "beta_dust": int(args.target_ud_grade[0]),
        "temp_dust": int(args.target_ud_grade[1]),
        "beta_pl": int(args.target_ud_grade[2]),
    }
    patch_indices = multires_clusters(mask, indices, target_ud_grade, nside)
    max_count = {
        key.replace("_patches", ""): int(jnp.unique(val).size) for key, val in patch_indices.items()
    }

    # Define the base parameters and bounds
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

    sky_signal_fn = partial(sky_signal, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)
    negative_log_likelihood_fn = partial(
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
    )

    guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)
    lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
    upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

    solver_options = {}
    if args.top_k_release is not None:
        solver_options["max_constraints_to_release"] = args.top_k_release

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
            atol=1e-18,
            rtol=1e-18,
            lower_bound=lower_bound_tree,
            upper_bound=upper_bound_tree,
            precondition=args.cond,
            solver_options=solver_options,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

        s = sky_signal_fn(final_params, nu=nu, d=noised_d, N=N, patch_indices=patch_indices)
        cmb = s["cmb"]
        cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))

        cmb_np = jnp.stack([cmb.q, cmb.u])
        noise_d_np = jnp.stack([noised_d.q, noised_d.u])
        small_n_np = jnp.stack([small_n.q, small_n.u])

        nll = negative_log_likelihood_fn(
            final_params, nu=nu, d=noised_d, N=N, patch_indices=patch_indices
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

    # Save results and mask
    if not args.best_only:
        start_time = perf_counter()
        if args.use_vmap:
            # Vmap approach - JIT the entire vmapped computation
            results = jax.jit(jax.vmap(single_run))(
                jnp.arange(args.seed_start, args.seed_start + nb_noise_sim)
            )
        else:
            # For-loop approach - JIT single_run only
            single_run_jit = jax.jit(single_run)
            results_list = tqdm(
                [single_run_jit(i) for i in range(args.seed_start, args.seed_start + nb_noise_sim)],
                desc="Running noise simulations",
            )
            results = jax.tree.map(lambda *xs: jnp.stack(xs), *results_list)
        jax.tree.map(lambda x: x.block_until_ready(), results)
        end_time = perf_counter()
        min_bd, max_bd = results["beta_dust"].min(), results["beta_dust"].max()
        min_td, max_td = results["temp_dust"].min(), results["temp_dust"].max()
        min_bs, max_bs = results["beta_pl"].min(), results["beta_pl"].max()
        info(f"min beta_dust: {min_bd} max beta_dust: {max_bd}")
        info(f"min temp_dust: {min_td} max temp_dust: {max_td}")
        info(f"min beta_pl: {min_bs} max beta_pl: {max_bs}")
        info(f"Number of iterations: {results['iter_num'].mean()}")
        info(f"Component separation took {end_time - start_time:.2f} seconds")

        results["beta_dust_patches"] = patch_indices["beta_dust_patches"]
        results["temp_dust_patches"] = patch_indices["temp_dust_patches"]
        results["beta_pl_patches"] = patch_indices["beta_pl_patches"]
        # Add a new axis to the results so it matches the shape of grid search results
        os.makedirs(out_folder, exist_ok=True)
        results = jax.tree.map(lambda x: x[np.newaxis, ...], results)
        np.savez(f"{out_folder}/results.npz", **results)

    os.makedirs(out_folder, exist_ok=True)
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
