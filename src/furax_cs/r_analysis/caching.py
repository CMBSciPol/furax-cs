import os
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from furax import HomothetyOperator
from furax.obs import negative_log_likelihood, sky_signal
from furax.obs.stokes import Stokes
from furax_cs.optim import minimize
from jaxtyping import Array, Float, Int


def compute_w(
    nu: Float[Array, " n_freq"],
    d: Stokes,
    patches: dict[str, Int[Array, " n_valid"]],
    max_iter: int = 100,
    solver_name: str = "optax_lbfgs",
) -> Stokes:
    """Compute the foreground-only CMB reconstruction (W·d_fg).

    This is a pure computation function with no File I/O.

    Parameters
    ----------
    nu : array_like
        Frequency array in GHz.
    d : Stokes
        Foreground-only frequency maps.
    patches : dict
        Dictionary containing patch indices (beta_dust, temp_dust, beta_pl).
    max_iter : int, optional
        Maximum optimization iterations (default: 100).
    solver_name : str, optional
        Solver name for optimization (default: "optax_lbfgs").

    Returns
    -------
    Stokes
        Foreground-only CMB reconstruction (W·d_fg).
    """
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    max_count = {
        "beta_dust": patches["beta_dust_patches"].size,
        "temp_dust": patches["temp_dust_patches"].size,
        "beta_pl": patches["beta_pl_patches"].size,
    }

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }

    guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)

    N = HomothetyOperator(1.0, _in_structure=d.structure)

    negative_log_likelihood_fn = partial(
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
    )

    final_params, final_state = minimize(
        fn=negative_log_likelihood_fn,
        init_params=guess_params,
        solver_name=solver_name,
        max_iter=max_iter,
        atol=1e-15,
        rtol=1e-15,
        lower_bound={
            "beta_dust": jnp.full((max_count["beta_dust"],), 0.5),
            "temp_dust": jnp.full((max_count["temp_dust"],), 5.0),
            "beta_pl": jnp.full((max_count["beta_pl"],), -6.0),
        },
        upper_bound={
            "beta_dust": jnp.full((max_count["beta_dust"],), 3.0),
            "temp_dust": jnp.full((max_count["temp_dust"],), 50.0),
            "beta_pl": jnp.full((max_count["beta_pl"],), -1.0),
        },
        precondition=True,
        nu=nu,
        N=N,
        d=d,
        patch_indices=patches,
    )

    def W_op(p):
        return sky_signal(
            p,
            nu,
            N,
            d,
            dust_nu0=dust_nu0,
            synchrotron_nu0=synchrotron_nu0,
            patch_indices=patches,
        )["cmb"]

    return W_op(final_params)


def atomic_save_results(result_file: str, results_dict: dict[str, Any]) -> None:
    """Atomically write updated results to disk with backup protection."""
    if os.path.exists(result_file):
        backup_file = result_file.replace(".npz", ".bk.npz")
        os.rename(result_file, backup_file)

    temp_file = result_file.replace(".npz", ".tmp.npz")
    np.savez(temp_file, **results_dict)

    os.rename(temp_file, result_file)
