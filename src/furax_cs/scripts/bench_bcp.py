#!/usr/bin/env python3
"""
Benchmark Component Separation Methods: FGBuster vs FURAX

This script benchmarks the performance and accuracy of different component separation
frameworks and optimization methods for CMB analysis. It compares FGBuster (baseline)
against FURAX (JAX-native) implementations across different HEALPix resolutions.

Benchmarking Categories:
    1. Log-likelihood evaluation performance
    2. Component separation solver performance (TNC, L-BFGS)
    3. Weak scaling analysis across different map resolutions

Usage:
    python 01-bench_bcp.py -n 64 128 256 -s -l    # Benchmark solvers and likelihood
    python 01-bench_bcp.py -p                     # Plot existing results only

Output:
    - Performance timing data saved to runs/*.csv
    - Scaling plots generated as PNG files
    - Comparative analysis between frameworks

Author: FURAX Team
"""

import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

# Furax imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Healpy and PySM3 imports
# FGBuster imports
try:
    from fgbuster import (
        CMB,
        Dust,
        MixingMatrix,
        Synchrotron,
        basic_comp_sep,
        get_instrument,
    )
    from fgbuster.algebra import _build_bound_inv_logL_and_logL_dB
except ImportError:
    raise ImportError(
        "FGBuster is required for benchmark comparisons. Install with:\n"
        "  pip install fgbuster\n"
        "or\n"
        "  pip install git+https://github.com/fgbuster/fgbuster.git"
    )

from furax import HomothetyOperator
from furax.obs import negative_log_likelihood
from furax.obs.landscapes import Stokes
from furax_cs.data.generate_maps import load_from_cache, save_to_cache
from furax_cs.logging_utils import info
from furax_cs.optim import minimize
from jax_hpc_profiler import Timer
from jax_hpc_profiler.plotting import plot_weak_scaling

jax.config.update("jax_enable_x64", True)


def run_fgbuster_logL(
    nside: int, freq_maps: Any, components: list[Any], nu: Any, numpy_timer: Any
) -> None:
    """
    Benchmark FGBuster log-likelihood evaluation performance.

    This function evaluates the log-likelihood computation using FGBuster's
    traditional numpy-based implementation. It builds the mixing matrix and
    differential operators, then times the likelihood evaluation.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter (nside)
    freq_maps : array_like, shape (n_freq, 3, n_pix)
        Frequency maps with Stokes I, Q, U components
    components : list
        List of FGBuster component objects (CMB, Dust, Synchrotron)
    nu : array_like
        Frequency array in GHz
    numpy_timer : Timer
        Performance timing object for numpy-based functions

    Notes
    -----
    Times both JIT compilation and execution phases to measure
    realistic performance characteristics.
    """
    info(f"Running FGBuster Log Likelihood with nside={nside} ...")

    # Step 1: Build mixing matrix and derivative evaluators
    A = MixingMatrix(*components)
    A_ev = A.evaluator(nu)
    A_dB_ev = A.diff_evaluator(nu)
    data = freq_maps.T

    # Step 2: Construct bounded log-likelihood function
    logL, _, _ = _build_bound_inv_logL_and_logL_dB(A_ev, data, None, A_dB_ev, A.comp_of_dB)
    x0 = np.array([x for c in components for x in c.defaults])

    # Step 3: Time compilation and execution
    numpy_timer.chrono_jit(logL, x0)
    for _ in range(2):
        numpy_timer.chrono_fun(logL, x0)


def run_jax_negative_log_prob(
    nside: int,
    freq_maps: Any,
    best_params: Any,
    nu: Any,
    dust_nu0: float,
    synchrotron_nu0: float,
    jax_timer: Timer,
) -> None:
    """
    Benchmark FURAX JAX-based negative log-likelihood evaluation.

    This function evaluates the negative log-likelihood using FURAX's
    JAX-native implementation, which enables GPU acceleration and
    automatic differentiation.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter
    freq_maps : array_like, shape (n_freq, 3, n_pix)
        Multi-frequency sky maps (I, Q, U Stokes parameters)
    best_params : dict
        Dictionary containing spectral parameters:
        - temp_dust: Dust temperature (K)
        - beta_dust: Dust spectral index
        - beta_pl: Synchrotron spectral index
    nu : array_like
        Frequency array in GHz
    dust_nu0 : float
        Dust reference frequency in GHz
    synchrotron_nu0 : float
        Synchrotron reference frequency in GHz
    jax_timer : Timer
        Performance timing object for JAX functions

    Notes
    -----
    Uses JAX's JIT compilation for optimal performance. The negative
    log-likelihood is the objective function minimized during component
    separation parameter estimation.
    """
    info(f"Running Furax Log Likelihood nside={nside} ...")

    # Step 1: Convert frequency maps to Stokes data structure
    d = Stokes.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])

    # Step 2: Create identity noise operator (simplified for benchmarking)
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    # Step 3: Construct negative log-likelihood function with fixed parameters
    nll = partial(
        negative_log_likelihood,
        nu=nu,
        N=invN,
        d=d,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
    )

    # Step 4: Time JIT compilation and execution
    jax_timer.chrono_jit(nll, best_params)

    for _ in range(2):
        jax_timer.chrono_fun(nll, best_params)


def run_jax_minimize(
    nside: int,
    freq_maps: Any,
    best_params: Any,
    nu: Any,
    dust_nu0: float,
    synchrotron_nu0: float,
    jax_timer: Any,
    max_iter: int,
    solver_name: str,
    precondition: bool,
) -> None:
    """Run JAX-based negative log-likelihood with configurable solver."""

    info(f"Running Furax {solver_name} Comp sep nside={nside} ...")

    d = Stokes.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    nll = partial(
        negative_log_likelihood,
        nu=nu,
        N=invN,
        d=d,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
    )

    best_params = jax.tree.map(lambda x: jnp.array(x), best_params)
    guess_params = jax.tree.map_with_path(
        lambda path, x: x + jax.random.normal(jax.random.key(path[0].__hash__()), x.shape),
        best_params,
    )

    def basic_comp(guess_params):
        final_params, _ = minimize(
            fn=nll,
            init_params=guess_params,
            solver_name=solver_name,
            max_iter=max_iter,
            rtol=1e-5,
            atol=1e-5,
            precondition=precondition,
        )
        return final_params["beta_pl"], final_params

    jax_timer.chrono_jit(basic_comp, guess_params)
    for _ in range(2):
        jax_timer.chrono_fun(basic_comp, guess_params)


def run_fgbuster_comp_sep(
    nside, instrument, best_params, freq_maps, components, numpy_timer, fgbuster_solver
):
    """Run FGBuster component separation with configurable solver."""
    info(f"Running FGBuster {fgbuster_solver} Comp sep nside={nside} ...")

    best_params = jax.tree.map(lambda x: jnp.array(x), best_params)
    guess_params = jax.tree.map_with_path(
        lambda path, x: x + jax.random.normal(jax.random.key(path[0].__hash__()), x.shape),
        best_params,
    )
    guess_params = jax.tree.map(lambda x: jnp.array(x), guess_params)

    components[1]._set_default_of_free_symbols(
        beta_d=guess_params["beta_dust"], temp=guess_params["temp_dust"]
    )
    components[2]._set_default_of_free_symbols(beta_pl=guess_params["beta_pl"])

    numpy_timer.chrono_jit(basic_comp_sep, components, instrument, freq_maps)

    for _ in range(2):
        numpy_timer.chrono_fun(basic_comp_sep, components, instrument, freq_maps)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods"
    )
    parser.add_argument(
        "-n",
        "--nsides",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="List of nsides to benchmark",
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
        "-l",
        "--likelihood",
        action="store_true",
        help="Benchmark FGBuster and Furax log-likelihood methods",
    )
    parser.add_argument(
        "-s",
        "--solvers",
        action="store_true",
        help="Benchmark solvers: FGBuster, JAX LBFGS, and JAX TNC",
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
        default=100,
        help="Maximum number of optimization iterations for L-BFGS solver",
    )
    parser.add_argument(
        "--jax-solver",
        type=str,
        default="optax_lbfgs",
        help="JAX solver name (e.g., optax_lbfgs, optimistix_bfgs, scipy_tnc)",
    )
    parser.add_argument(
        "--fgbuster-solver",
        type=str,
        default="TNC",
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
    return parser.parse_args()


def main():
    """
    Main execution function for component separation benchmarking.

    Orchestrates the benchmarking workflow across different modes:
    likelihood evaluation, solver performance, and result plotting.
    """
    # Step 1: Parse command line arguments and setup parameters
    args = parse_args()
    instrument = get_instrument("LiteBIRD")
    nu = instrument["frequency"].values

    # Step 2: Initialize physical parameters for component separation
    dust_nu0, synchrotron_nu0 = 150.0, 20.0  # Reference frequencies in GHz
    components = [CMB(), Dust(dust_nu0), Synchrotron(synchrotron_nu0)]
    best_params = {"temp_dust": 20.0, "beta_dust": 1.54, "beta_pl": -3.0}

    # Step 3: Initialize performance timers for different frameworks
    jax_timer = Timer(save_jaxpr=False, jax_fn=True)  # For JAX/FURAX functions
    numpy_timer = Timer(save_jaxpr=False, jax_fn=False)  # For NumPy/FGBuster functions

    if args.likelihood and not args.plot_only:
        for nside in args.nsides:
            save_to_cache(nside, sky="c1d0s0", noise_ratio=args.noise)

            if args.cache_run:
                continue

            nu, freq_maps = load_from_cache(nside, sky="c1d0s0", noise_ratio=args.noise)

            # Likelihood mode benchmarking
            print(f"Running likelihood benchmarking for nside={nside}...")
            run_fgbuster_logL(nside, freq_maps, components, nu, numpy_timer)
            run_jax_negative_log_prob(
                nside, freq_maps, best_params, nu, dust_nu0, synchrotron_nu0, jax_timer
            )

            kwargs = {"function": "Furax LL", "precision": "float64", "x": nside}
            jax_timer.report("runs/LL_FURAX.csv", **kwargs)
            kwargs = {"function": "FGBuster LL", "precision": "float64", "x": nside}
            numpy_timer.report("runs/LL_FGBUSTER.csv", **kwargs)

    if args.solvers and not args.plot_only:
        for nside in args.nsides:
            save_to_cache(nside, sky="c1d1s1", noise_ratio=args.noise)

            if args.cache_run:
                continue

            nu, freq_maps = load_from_cache(nside, sky="c1d1s1", noise_ratio=args.noise)

            # Solver mode benchmarking
            print(f"Running solver benchmarking for nside={nside}...")
            run_fgbuster_comp_sep(
                nside,
                instrument,
                best_params,
                freq_maps,
                components,
                numpy_timer,
                args.fgbuster_solver,
            )
            kwargs = {
                "function": f"FGBuster-{args.fgbuster_solver}",
                "precision": "float64",
                "x": nside,
            }
            numpy_timer.report("runs/BCP_FGBUSTER.csv", **kwargs)

            # Run JAX with configurable solver
            run_jax_minimize(
                nside,
                freq_maps,
                best_params,
                nu,
                dust_nu0,
                synchrotron_nu0,
                jax_timer,
                args.max_iter,
                args.jax_solver,
                args.precondition,
            )
            kwargs = {
                "function": f"Furax-{args.jax_solver}",
                "precision": "float64",
                "x": nside,
            }
            jax_timer.report("runs/BCP_FURAX.csv", **kwargs)

    # Plot log-likelihood results
    if args.likelihood and not args.cache_run and args.plot_only:
        plt.rcParams.update({"font.size": 15})
        sns.set_context("paper")

        csv_file = ["runs/LL_FURAX.csv", "runs/LL_FGBUSTER.csv"]
        FWs = ["Furax LL", "FGBuster LL"]

        plot_weak_scaling(
            csv_files=csv_file,
            functions=FWs,
            figure_size=(12, 8),
            label_text="%f%",
            output="runs/nll.png",
        )

    # Plot solver results
    if args.solvers and not args.cache_run and args.plot_only:
        plt.rcParams.update({"font.size": 15})
        sns.set_context("paper")

        csv_file = ["runs/BCP_FGBUSTER.csv", "runs/BCP_FURAX.csv"]
        # Note: Update solvers list to match actual benchmarked configurations
        solvers = [
            f"Furax-{args.jax_solver}",
            f"FGBuster-{args.fgbuster_solver}",
        ]

        plot_weak_scaling(
            csv_files=csv_file,
            functions=solvers,
            figure_size=(12, 8),
            label_text="%f%",
            output="runs/solvers.png",
        )


if __name__ == "__main__":
    main()
