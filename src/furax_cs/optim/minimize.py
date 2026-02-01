from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxopt import ScipyBoundedMinimize
from jaxtyping import (
    Array,
    Float,
    PyTree,  # pyright: ignore
    Scalar,
)

from ..logging_utils import warning
from .solvers import SELFCONDITIONED_SOLVERS, SOLVER_NAMES, get_solver
from .utils import condition

# =============================================================================
# SCIPY MINIMIZE WITH VMAP SUPPORT
# =============================================================================


class ScipyMinimizeState(eqx.Module):
    """State returned by scipy minimize via pure_callback.

    This equinox module holds the optimization result in a JAX-compatible format
    that can be used with vmap/lax.map.

    Attributes
    ----------
    params : PyTree
        Optimized parameters.
    fun_val : Scalar
        Final objective function value (scalar).
    success : Scalar
        Whether optimization converged successfully (bool scalar).
    iter_num : Scalar
        Number of iterations performed (int32 scalar).
    """

    params: PyTree[Float[Array, " P"]]
    fun_val: Scalar
    success: Scalar
    iter_num: Scalar


def scipy_minimize(
    fn: Callable[..., Scalar],
    init_params: PyTree[Float[Array, " P"]],
    lower_bound: Optional[PyTree[Float[Array, " P"]]] = None,
    upper_bound: Optional[PyTree[Float[Array, " P"]]] = None,
    method: str = "tnc",
    maxiter: int = 1000,
    **fn_kwargs: Any,
) -> ScipyMinimizeState:
    """Scipy minimize wrapper that supports vmap via jax.pure_callback.

    This function wraps scipy optimization in a way that is compatible with
    JAX transformations like vmap and lax.map. It uses jax.pure_callback to
    call the host-side scipy solver.

    Parameters
    ----------
    fn : Callable
        Objective function to minimize. Should accept (params, **fn_kwargs).
    init_params : PyTree
        Initial parameter values.
    lower_bound : PyTree, optional
        Lower bounds for parameters. Same shape as init_params.
    upper_bound : PyTree, optional
        Upper bounds for parameters. Same shape as init_params.
    method : str
        Scipy optimization method (default "tnc").
    maxiter : int
        Maximum number of iterations.
    **fn_kwargs
        Additional arguments passed to fn.

    Returns
    -------
    ScipyMinimizeState
        Optimization result containing params, fun_val, success, and iter_num.
    """

    def host_solver_callback(x_init, lower, upper, fn_kwargs):
        """Host-side scipy solver callback."""
        # Handle bounds
        if lower is None and upper is None:
            bounds = None
        else:
            bounds = (lower, upper)

        # Define wrapped objective
        def scipy_fn(params, fn_kwargs):
            return fn(params, **fn_kwargs)

        # Scipy method handling
        solver_options = {"disp": False}
        if method == "cobyqa":
            try:
                import cobyqa  # noqa: F401
            except ImportError:
                raise ImportError(
                    "cobyqa not installed. Please install it with `pip install cobyqa`."
                )

        solver = ScipyBoundedMinimize(
            fun=scipy_fn,
            method=method,
            jit=False,
            maxiter=maxiter,
            options=solver_options,
        )

        res = solver.run(x_init, bounds=bounds, fn_kwargs=fn_kwargs)

        # Return numpy arrays for pure_callback
        return {
            "params": jax.tree.map(lambda x: np.array(x), res.params),
            "fun_val": np.array(res.state.fun_val, dtype=np.float32),
            "success": np.array(res.state.success, dtype=bool),
            "iter_num": np.array(res.state.iter_num, dtype=np.int32),
        }

    # Define result shape for pure_callback
    result_shape = {
        "params": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), init_params),
        "fun_val": jax.ShapeDtypeStruct((), jnp.float32),
        "success": jax.ShapeDtypeStruct((), jnp.bool_),
        "iter_num": jax.ShapeDtypeStruct((), jnp.int32),
    }

    result_dict = jax.pure_callback(
        host_solver_callback,
        result_shape,
        init_params,
        lower_bound,
        upper_bound,
        fn_kwargs,
        vmap_method="sequential",
    )

    return ScipyMinimizeState(
        params=result_dict["params"],
        fun_val=result_dict["fun_val"],
        success=result_dict["success"],
        iter_num=result_dict["iter_num"],
    )


# =============================================================================
# UNIFIED STATE
# =============================================================================


class UnifiedState(eqx.Module):
    """Unified optimization state.

    Attributes
    ----------
    best_loss : Scalar
        Best objective function value found.
    best_y : PyTree
        Best parameters found (in original space).
    iter_num : Scalar
        Number of iterations performed.
    solver_state : Any
        Internal solver state (Optimistix state or ScipyMinimizeState).
    """

    best_loss: Scalar
    best_y: PyTree[Float[Array, " P"]]
    iter_num: Scalar
    solver_state: Any


# =============================================================================
# UNIFIED OPTIMIZATION INTERFACE
# =============================================================================


def minimize(
    fn: Callable[..., Scalar],
    init_params: PyTree[Float[Array, " P"]],
    solver_name: SOLVER_NAMES = "optax_lbfgs",
    max_iter: int = 1000,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    lower_bound: Optional[PyTree[Float[Array, " P"]]] = None,
    upper_bound: Optional[PyTree[Float[Array, " P"]]] = None,
    precondition: bool = False,
    solver_options: dict[str, Any] = {},
    **fn_kwargs: Any,
) -> tuple[PyTree[Float[Array, " P"]], UnifiedState]:
    """
    Unified optimization interface.

    Supports optax solvers, optimistix solvers (via optimistix.minimise),
    and scipy solvers (via jaxopt.ScipyMinimize).

    Parameters
    ----------
    fn : Callable
        Objective function to minimize. Should accept (params, **fn_kwargs).
    init_params : PyTree
        Initial parameter values.
    solver_name : str
        Solver identifier. See SOLVER_NAMES for available options.
    max_iter : int
        Maximum iterations.
    rtol, atol : float
        Relative/absolute tolerance for optimization convergence.
    lower_bound, upper_bound : PyTree, optional
        Box constraints.
    precondition : bool
        Whether to apply parameter transformation and output scaling.
    solver_options : dict, optional
        Additional arguments passed to the solver factory (get_solver).
    **fn_kwargs
        Additional arguments passed to fn.

    Returns
    -------
    final_params : PyTree
        Optimized parameters.
    final_state : UnifiedState
        Final optimizer state containing best loss, best parameters, iteration count, and solver state.
    """
    solver_name = cast(SOLVER_NAMES, solver_name)

    if solver_name in SELFCONDITIONED_SOLVERS and precondition:
        warning(f"Solver '{solver_name}' is self-conditioned; ignoring preconditioning request.")
        precondition = False

    if precondition:
        fn, to_opt, from_opt = condition(
            fn,
            lower=lower_bound,
            upper=upper_bound,
            scale_function=precondition,
            init_params=init_params,
            **fn_kwargs,
        )
        init_params = to_opt(init_params)
        lower_bound = to_opt(lower_bound) if lower_bound is not None else None
        upper_bound = to_opt(upper_bound) if upper_bound is not None else None
    else:
        from_opt = lambda x: x

    solver_opts = solver_options if solver_options is not None else {}
    solver, solver_type = get_solver(
        solver_name,
        rtol=rtol,
        atol=atol,
        lower=lower_bound,
        upper=upper_bound,
        **solver_opts,
    )

    if solver_type == "optimistix":
        # Optimistix uses (y, args) signature, wrap fn
        def optx_fn(y, fn_kwargs):
            return fn(y, **fn_kwargs)

        sol = optx.minimise(
            optx_fn,
            solver,
            init_params,
            max_steps=max_iter,
            progress_meter=optx.TqdmProgressMeter(refresh_steps=10),
            throw=False,
            args=fn_kwargs,
        )

        unified_state = UnifiedState(
            best_loss=sol.state.best_loss,
            best_y=from_opt(sol.state.best_y),
            iter_num=sol.stats["num_steps"],
            solver_state=sol.state,
        )
        return from_opt(sol.value), unified_state

    elif solver_type == "scipy":
        # Scipy via vmap-compatible scipy_minimize
        method = solver_name.split("_")[1]
        options = solver_options.get("options", {})
        if method == "tnc":
            options["ftol"] = atol
            options["gtol"] = rtol
            options["xtol"] = atol
        elif method == "l-bfgs-b":
            options["ftol"] = atol
            options["gtol"] = rtol
        elif method == "cobyqa":  # COBYQA
            options["final_tr_radius"] = atol
        state = scipy_minimize(
            fn=fn,
            init_params=init_params,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method=method,
            maxiter=max_iter,
            **fn_kwargs,
        )

        unified_state = UnifiedState(
            best_loss=state.fun_val,
            best_y=from_opt(state.params),
            iter_num=state.iter_num,
            solver_state=state,
        )

        return from_opt(state.params), unified_state

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
