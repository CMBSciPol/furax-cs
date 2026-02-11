# Necessary imports
import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import operator
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jaxtyping import Array

try:
    from fgbuster import (
        CMB,
        Dust,
        Synchrotron,
        adaptive_comp_sep,
        get_instrument,
    )
except ImportError:
    raise ImportError(
        "FGBuster is required for benchmark comparisons. Install with:\n"
        "  pip install fgbuster\n"
        "or\n"
        "  pip install git+https://github.com/fgbuster/fgbuster.git"
    )

from furax.obs import negative_log_likelihood, spectral_cmb_variance
from furax.obs.stokes import Stokes
from furax_cs import generate_noise_operator
from furax_cs.data.generate_maps import load_from_cache, save_to_cache
from furax_cs.logging_utils import info
from furax_cs.optim import minimize
from jax_healpy.clustering import find_kmeans_clusters
from jax_hpc_profiler import JaxTimer, NumpyTimer
from jax_hpc_profiler.plotting import plot_weak_scaling

jax.config.update("jax_enable_x64", True)


def run_fg_buster(
    nside: int,
    cluster_count: int,
    freq_maps: Array,
    dust_nu0: float,
    synchrotron_nu0: float,
    numpy_timer: Any,
    max_iter: int,
    tol: float,
    fgbuster_solver: str,
    instrument: Any,
    noise_ratio: float,
    n_sims: int,
) -> tuple[Any, Any, Any]:
    info(
        f"Running FGBuster {fgbuster_solver} Comp sep nside={nside} cluster_count={cluster_count}..."
    )

    d_clean = Stokes.from_stokes(Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])

    mask = jnp.ones_like(d_clean.q[0]).astype(jnp.int64)

    (indices,) = jnp.where(mask == 1)

    # Split cluster_count into 50% beta_dust, 30% temp_dust, 20% beta_pl
    n_beta_dust = int(cluster_count * 0.5)
    n_temp_dust = int(cluster_count * 0.3)
    n_beta_pl = cluster_count - n_beta_dust - n_temp_dust

    # Ensure at least 1 cluster if possible/needed, though inputs are large
    n_beta_dust = max(1, n_beta_dust)
    n_temp_dust = max(1, n_temp_dust)
    n_beta_pl = max(1, n_beta_pl)

    temp_dust_patch_indices = find_kmeans_clusters(
        mask,
        indices,
        n_temp_dust,
        jax.random.PRNGKey(0),
        max_centroids=n_temp_dust,
        initial_sample_size=1,
    )
    beta_dust_patch_indices = find_kmeans_clusters(
        mask,
        indices,
        n_beta_dust,
        jax.random.PRNGKey(1),
        max_centroids=n_beta_dust,
        initial_sample_size=1,
    )
    beta_pl_patch_indices = find_kmeans_clusters(
        mask,
        indices,
        n_beta_pl,
        jax.random.PRNGKey(2),
        max_centroids=n_beta_pl,
        initial_sample_size=1,
    )
    patch_ids_fg = [
        temp_dust_patch_indices.astype(jnp.int64),
        beta_dust_patch_indices.astype(jnp.int64),
        beta_pl_patch_indices.astype(jnp.int64),
    ]

    components = [CMB(), Dust(dust_nu0), Synchrotron(synchrotron_nu0)]

    bounds = [(0.5, 5.0), (10.0, 40), (-6.0, -1.0)]
    options = {"disp": False, "gtol": tol, "eps": tol, "maxiter": max_iter, "tol": tol}
    method = fgbuster_solver
    # instrument = get_instrument("LiteBIRD") # Provided as arg now

    patch_ids_fg = [np.asarray(p) for p in patch_ids_fg]

    comp_sep = partial(adaptive_comp_sep, bounds=bounds, options=options, method=method, tol=tol)

    def single_run_fgbuster(seed_key):
        noised_d, _, _ = generate_noise_operator(
            seed_key, noise_ratio, indices, nside, d_clean, instrument
        )
        freq_maps_fg = jnp.stack([noised_d.q, noised_d.u], axis=1)
        freq_maps_fg = np.asarray(freq_maps_fg)

        result = comp_sep(components, instrument, freq_maps_fg, patch_ids_fg)

        cmb_q, cmb_u = result.s[0]
        cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, (cmb_q, cmb_u)))

        return result.params, cmb_var, result.fun

    variances = []
    likelihoods = []
    final_params_raw: Any = None

    for i in range(n_sims):
        seed = jax.random.PRNGKey(i)
        params, var, nll = numpy_timer.chrono_fun(single_run_fgbuster, seed)
        variances.append(var)
        likelihoods.append(nll)
        final_params_raw = params

    avg_var = np.mean(variances)
    avg_nll = np.mean(likelihoods)

    final_params = {
        "beta_dust": final_params_raw[0],
        "temp_dust": final_params_raw[1],
        "beta_pl": final_params_raw[2],
    }

    return final_params, avg_var, avg_nll


def run_jax_minimize(
    nside: int,
    cluster_count: int,
    freq_maps: Array,
    nu: Array,
    dust_nu0: float,
    synchrotron_nu0: float,
    jax_timer: Any,
    max_iter: int,
    tol: float,
    solver_name: str,
    precondition: bool,
    instrument: Any,
    noise_ratio: float,
    n_sims: int,
) -> tuple[Any, Any, Any]:
    """Run JAX-based negative log-likelihood with configurable solver."""

    info(f"Running Furax {solver_name} Comp sep nside={nside} cluster_count={cluster_count}...")

    # Split cluster_count into 50% beta_dust, 30% temp_dust, 20% beta_pl
    n_beta_dust = int(cluster_count * 0.5)
    n_temp_dust = int(cluster_count * 0.3)
    n_beta_pl = cluster_count - n_beta_dust - n_temp_dust

    # Ensure at least 1 cluster
    n_beta_dust = max(1, n_beta_dust)
    n_temp_dust = max(1, n_temp_dust)
    n_beta_pl = max(1, n_beta_pl)

    best_params = {
        "beta_pl": jnp.full((n_beta_pl,), (-3.0)),
        "beta_dust": jnp.full((n_beta_dust,), 1.54),
        "temp_dust": jnp.full((n_temp_dust,), 20.0),
    }

    lower_params = {
        "beta_pl": jnp.full((n_beta_pl,), -6.0),
        "beta_dust": jnp.full((n_beta_dust,), 0.5),
        "temp_dust": jnp.full((n_temp_dust,), 10.0),
    }

    upper_params = {
        "beta_pl": jnp.full((n_beta_pl,), -1.0),
        "beta_dust": jnp.full((n_beta_dust,), 5.0),
        "temp_dust": jnp.full((n_temp_dust,), 40.0),
    }

    guess_params = jax.tree.map_with_path(
        lambda path, x: x + jax.random.normal(jax.random.key(path[0].__hash__()), x.shape),
        best_params,
    )

    d_clean = Stokes.from_stokes(Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    # invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure) # Replaced by generated noise
    mask = jnp.ones_like(d_clean.q[0]).astype(jnp.int64)

    (indices,) = jnp.where(mask == 1)

    temp_dust_patch_indices = find_kmeans_clusters(
        mask,
        indices,
        n_temp_dust,
        jax.random.PRNGKey(0),
        max_centroids=n_temp_dust,
        initial_sample_size=1,
    )
    beta_dust_patch_indices = find_kmeans_clusters(
        mask,
        indices,
        n_beta_dust,
        jax.random.PRNGKey(1),
        max_centroids=n_beta_dust,
        initial_sample_size=1,
    )
    beta_pl_patch_indices = find_kmeans_clusters(
        mask,
        indices,
        n_beta_pl,
        jax.random.PRNGKey(2),
        max_centroids=n_beta_pl,
        initial_sample_size=1,
    )

    patch_indices = {
        "temp_dust_patches": temp_dust_patch_indices.astype(jnp.int64),
        "beta_dust_patches": beta_dust_patch_indices.astype(jnp.int64),
        "beta_pl_patches": beta_pl_patch_indices.astype(jnp.int64),
    }

    def furax_adaptative_comp_sep(guess_params, seed_key):
        noised_d, N, _ = generate_noise_operator(
            seed_key, noise_ratio, indices, nside, d_clean, instrument
        )

        nll = partial(
            negative_log_likelihood,
            nu=nu,
            N=N,
            d=noised_d,
            dust_nu0=dust_nu0,
            synchrotron_nu0=synchrotron_nu0,
            patch_indices=patch_indices,
            analytical_gradient=True,
        )

        final_params, _ = minimize(
            fn=nll,
            init_params=guess_params,
            solver_name=solver_name,
            max_iter=max_iter,
            rtol=tol * 1e5,
            atol=tol,
            precondition=precondition,
            lower_bound=lower_params,
            upper_bound=upper_params,
        )

        last_L = nll(final_params)
        cmb_variance = spectral_cmb_variance(
            final_params, nu, N, noised_d, dust_nu0, synchrotron_nu0, patch_indices
        )

        # Return beta_pl only as the "result" for compatibility with original code structure?
        # The original code returned (final_params["beta_pl"], final_params)
        # But Timer calls this. We need to extract variance and L later.
        # Let's return full tuple.
        return final_params, cmb_variance, last_L

    variances = []
    likelihoods = []
    final_params = None

    for i in range(n_sims):
        seed = jax.random.PRNGKey(i)
        if i == 0:
            params, var, nll = jax_timer.chrono_jit(furax_adaptative_comp_sep, guess_params, seed)
        else:
            params, var, nll = jax_timer.chrono_fun(furax_adaptative_comp_sep, guess_params, seed)

        variances.append(var)
        likelihoods.append(nll)
        final_params = params

    avg_var = np.mean(variances)
    avg_nll = np.mean(likelihoods)

    return final_params, avg_var, avg_nll


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmarking FGBuster and Furax")
    parser.add_argument(
        "-n",
        "--nsides",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="List of nsides to benchmark",
    )
    parser.add_argument(
        "-cl",
        "--clusters",
        type=int,
        nargs="+",
        default=[1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="List of cluster counts to benchmark",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="comparison",
        help="Output filename prefix for the plots",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10, 6),
        help="Figure size for the plots (width, height)",
    )
    parser.add_argument(
        "-p",
        "--plot-only",
        action="store_true",
        help="Benchmark solvers: FGBuster, JAX LBFGS, and JAX TNC",
    )
    parser.add_argument(
        "-c",
        "--cache-run",
        action="store_true",
        help="Run the cache generation step",
    )
    parser.add_argument(
        "-mi",
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations for L-BFGS solver",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-15,
        help="Tolerance for optimization convergence",
    )
    parser.add_argument(
        "--jax-solver",
        type=str,
        default="SKIP",
        help="JAX solver name (e.g., optax_lbfgs, optimistix_bfgs, scipy_tnc)",
    )
    parser.add_argument(
        "--fgbuster-solver",
        type=str,
        default="SKIP",
        help="FGBuster scipy solver method (e.g., TNC, L-BFGS-B, SLSQP)",
    )
    parser.add_argument(
        "--precondition",
        action="store_true",
        help="Enable preconditioning for JAX optimization",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise ratio (0.0 = no noise, 1.0 = 100%% noise)",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=3,
        help="Number of noise simulations for benchmarking",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    instrument = get_instrument("LiteBIRD")
    nu = instrument["frequency"].values
    # stokes_type = 'IQU'
    dust_nu0, synchrotron_nu0 = 150.0, 20.0

    jax_timer = JaxTimer(save_jaxpr=True)
    np_timer = NumpyTimer()

    if not args.plot_only:
        for nside in args.nsides:
            sky = "c1d0s0"
            save_to_cache(nside, sky=sky, noise_ratio=0.0)

            if args.cache_run:
                continue

            nu, freq_maps = load_from_cache(nside, sky=sky, noise_ratio=0.0)

            for cluster_count in args.clusters:
                # Solver mode benchmarking
                info(f"Running solver benchmarking for nside={nside}...")

                # Run FGBuster with configurable solver
                if args.fgbuster_solver.lower() != "skip":
                    final_params, cmb_variance, last_L = run_fg_buster(
                        nside,
                        cluster_count,
                        freq_maps,
                        dust_nu0,
                        synchrotron_nu0,
                        np_timer,
                        args.max_iter,
                        args.tol,
                        args.fgbuster_solver,
                        instrument,
                        args.noise,
                        args.n_sims,
                    )
                    data = {
                        "cmb_variance": cmb_variance,
                        "last_L": last_L,
                    }
                    data.update(**final_params)
                    kwargs = {
                        "function": f"FGBuster-{args.fgbuster_solver} n={nside}",
                        "precision": "float64",
                        "x": cluster_count,
                        "y": 1,
                        "z": 1,
                        "npz_data": data,
                    }
                    np_timer.report("runs/CLUSTERS_FGBUSTER.csv", **kwargs)

                # Run JAX with configurable solver
                if args.jax_solver.lower() != "skip":
                    final_params, cmb_variance, last_L = run_jax_minimize(
                        nside,
                        cluster_count,
                        freq_maps,
                        nu,
                        dust_nu0,
                        synchrotron_nu0,
                        jax_timer,
                        args.max_iter,
                        args.tol,
                        args.jax_solver,
                        args.precondition,
                        instrument,
                        args.noise,
                        args.n_sims,
                    )
                    data = {
                        "cmb_variance": cmb_variance,
                        "last_L": last_L,
                    }
                    data.update(**final_params)
                    kwargs = {
                        "function": f"Furax-{args.jax_solver} n={nside}",
                        "precision": "float64",
                        "x": cluster_count,
                        "npz_data": data,
                    }
                    jax_timer.report("runs/CLUSTERS_FURAX.csv", **kwargs)

    # Plot solver results
    if not args.cache_run and args.plot_only:
        plt.rcParams.update({"font.size": 15})
        sns.set_context("paper")

        csv_file = ["runs/CLUSTERS_FGBUSTER.csv", "runs/CLUSTERS_FURAX.csv"]
        # Note: Update solvers list to match actual benchmarked configurations
        solvers = [
            f"Furax-{args.jax_solver} n=32",
            f"FGBuster-{args.fgbuster_solver} n=32",
        ]

        plot_weak_scaling(
            csv_files=csv_file,
            functions=solvers,
            figure_size=(12, 8),
            label_text="%f%",
            output="runs/CLUSTERS_FGBUSTER_FURAX.png",
        )


if __name__ == "__main__":
    main()
