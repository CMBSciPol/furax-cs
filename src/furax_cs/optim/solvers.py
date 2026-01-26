from __future__ import annotations

from typing import Any, Literal, Optional, TypeAlias, Union

import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Bool, Float, PyTree
from optax._src import combine, transform
from optax._src import linesearch as _linesearch

from .active_set import active_set

# =============================================================================
# OFF-THE-SHELF L-BFGS SOLVERS
# =============================================================================

Solver: TypeAlias = Union[optx.BestSoFarMinimiser, str]


class ActiveSetMinimiser(optx.OptaxMinimiser):
    def terminate(
        self,
        fn: Any,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: Any,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        del fn, args, options
        terminate = jnp.where(state.opt_state.constraints_released, False, state.terminate)
        return terminate, optx.RESULTS.successful


def lbfgs_zoom(
    learning_rate: Optional[optax.ScalarOrSchedule] = None,
    memory_size: int = 10,
    scale_init_precond: bool = False,
    max_linesearch_steps: int = 200,
    initial_guess_strategy: str = "one",
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    verbose: bool = False,
    lower: Optional[PyTree[Float[Array, " P"]]] = None,
    upper: Optional[PyTree[Float[Array, " P"]]] = None,
) -> optax.GradientTransformation:
    """L-BFGS with zoom linesearch (strong Wolfe conditions).

    This is the standard L-BFGS with zoom linesearch that enforces both:
    - Sufficient decrease (Armijo): f(x + η*d) ≤ f(x) + c1*η*∇f(x)ᵀd
    - Curvature condition: |∇f(x + η*d)ᵀd| ≤ c2*|∇f(x)ᵀd|

    Args:
        learning_rate: Optional global scaling factor.
        memory_size: Number of past updates for Hessian approximation.
        scale_init_precond: Whether to scale initial Hessian approximation.
            WARNING: Set to False for numerically sensitive problems.
        max_linesearch_steps: Maximum iterations for zoom linesearch.
        initial_guess_strategy: "one" (start at η=1) or "keep" (use previous).
        slope_rtol: c1 parameter for Armijo condition (default 1e-4).
        curv_rtol: c2 parameter for curvature condition (default 0.9).
        verbose: Print linesearch debugging info.
        lower: Optional lower bounds for box projection (pytree).
        upper: Optional upper bounds for box projection (pytree).

    Returns:
        An optax GradientTransformation.
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = optax.scale_by_learning_rate(learning_rate)

    linesearch = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
        slope_rtol=slope_rtol,
        curv_rtol=curv_rtol,
        verbose=verbose,
    )

    chain_components = [
        transform.scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        base_scaling,
        linesearch,
    ]

    # Add projection if bounds provided
    if lower is not None and upper is not None:
        chain_components.append(apply_projection(lower, upper))

    return combine.chain(*chain_components)


def lbfgs_backtrack(
    learning_rate: Optional[optax.ScalarOrSchedule] = None,
    memory_size: int = 10,
    scale_init_precond: bool = False,
    max_backtracking_steps: int = 200,
    slope_rtol: float = 1e-4,
    decrease_factor: float = 0.8,
    increase_factor: float = 1.5,
    max_learning_rate: float = 1.0,
    verbose: bool = False,
    lower: Optional[PyTree[Float[Array, " P"]]] = None,
    upper: Optional[PyTree[Float[Array, " P"]]] = None,
) -> optax.GradientTransformation:
    """L-BFGS with backtracking linesearch (Armijo condition only).

    Simpler than zoom linesearch, only enforces sufficient decrease:
    - Armijo: f(x + η*d) ≤ f(x) + c1*η*∇f(x)ᵀd

    Args:
        learning_rate: Optional global scaling factor.
        memory_size: Number of past updates for Hessian approximation.
        scale_init_precond: Whether to scale initial Hessian approximation.
            WARNING: Set to False for numerically sensitive problems.
        max_backtracking_steps: Maximum backtracking iterations.
        slope_rtol: c1 parameter for Armijo condition (default 1e-4).
        decrease_factor: Multiply stepsize by this when condition fails (default 0.8).
        increase_factor: Initial guess = previous * this factor (default 1.5).
        max_learning_rate: Upper bound on stepsize (default 1.0).
        verbose: Print linesearch debugging info.
        lower: Optional lower bounds for box projection (pytree).
        upper: Optional upper bounds for box projection (pytree).

    Returns:
        An optax GradientTransformation.
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = optax.scale_by_learning_rate(learning_rate)

    linesearch = _linesearch.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_backtracking_steps,
        slope_rtol=slope_rtol,
        decrease_factor=decrease_factor,
        increase_factor=increase_factor,
        max_learning_rate=max_learning_rate,
        verbose=verbose,
    )

    chain_components = [
        transform.scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        base_scaling,
        linesearch,
    ]

    # Add projection if bounds provided
    if lower is not None and upper is not None:
        chain_components.append(apply_projection(lower, upper))

    return combine.chain(*chain_components)


def backtracking_adam(
    max_backtracking_steps: int = 200,
    slope_rtol: float = 1e-4,
    decrease_factor: float = 0.8,
    increase_factor: float = 1.5,
    max_learning_rate: float = 1.0,
    verbose: bool = False,
    lower: Optional[PyTree[Float[Array, " P"]]] = None,
    upper: Optional[PyTree[Float[Array, " P"]]] = None,
) -> optax.GradientTransformation:
    """Adam with backtracking linesearch (Armijo condition only)."""
    linesearch = _linesearch.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_backtracking_steps,
        slope_rtol=slope_rtol,
        decrease_factor=decrease_factor,
        increase_factor=increase_factor,
        max_learning_rate=max_learning_rate,
        verbose=verbose,
    )

    chain_components = [
        optax.adam(learning_rate=1.0),  # Learning rate handled by linesearch
        linesearch,
    ]

    # Add projection if bounds provided
    if lower is not None and upper is not None:
        chain_components.append(apply_projection(lower, upper))

    return combine.chain(*chain_components)


# =============================================================================
# BOX PROJECTION TRANSFORMATION
# =============================================================================


def apply_projection(
    lower: Optional[PyTree[Float[Array, " P"]]] = None,
    upper: Optional[PyTree[Float[Array, " P"]]] = None,
) -> optax.GradientTransformation:
    """Wrap box projection into a GradientTransformation.

    After applying this transformation, params + updates will be within [lower, upper].
    The update rule: u_new = clip(p + u, lower, upper) - p

    This can be chained with optimizers like:
        optimizer = optax.chain(
            optax.adam(learning_rate=1e-3),
            apply_projection(lower={'w': 0.0}, upper={'w': 1.0})
        )

    Args:
        lower: Lower bounds (pytree matching params structure)
        upper: Upper bounds (pytree matching params structure)

    Returns:
        GradientTransformation that projects updates to keep params in bounds.
    """

    def init_fn(params: PyTree[Float[Array, " P"]]) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        updates: PyTree[Float[Array, " P"]],
        state: optax.EmptyState,
        params: Optional[PyTree[Float[Array, " P"]]] = None,
    ) -> tuple[PyTree[Float[Array, " P"]], optax.EmptyState]:
        if params is None:
            raise ValueError("NO_PARAMS_MSG")

        if lower is None or upper is None:
            return updates, state

        def process_leaf(
            p: Float[Array, " P"],
            u: Float[Array, " P"],
            lo: Float[Array, " P"],
            hi: Float[Array, " P"],
        ) -> Float[Array, " P"]:
            if p is None or u is None:
                return u
            tentative = p + u
            projected = jnp.clip(tentative, lo, hi)
            return projected - p

        new_updates = jax.tree.map(process_leaf, params, updates, lower, upper)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# =============================================================================
# SOLVER NAMES AND FACTORY
# =============================================================================

SOLVER_NAMES = Literal[
    # Optax L-BFGS (jax_grid_search compatible)
    "optax_lbfgs",
    "optax_lbfgs",
    "adam",
    "sgd",
    "adabelief",
    "adaw",
    "active_set",
    "active_set_sgd",
    "active_set_adabelief",
    "active_set_adaw",
    # Optimistix BFGS
    "optimistix_bfgs",
    # Optimistix L-BFGS
    "optimistix_lbfgs",
    # Optimistix NCG (Armijo)
    "optimistix_ncg_pr",
    "optimistix_ncg_hs",
    "optimistix_ncg_fr",
    "optimistix_ncg_dy",
    # Scipy
    "scipy_tnc",
    "scipy_cobyqa",
    # Legacy aliases
    "zoom",
    "backtrack",
]

SELFCONDITIONED_SOLVERS = {"active_set", "active_set_sgd", "scipy_tnc", "scipy_cobyqa"}


def get_solver(
    solver_name: SOLVER_NAMES,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    learning_rate: float = 1e-3,
    max_linesearch_steps: int = 200,
    lower: Optional[PyTree[Float[Array, " P"]]] = None,
    upper: Optional[PyTree[Float[Array, " P"]]] = None,
    **kwargs: Any,
) -> tuple[Solver, Literal["optimistix", "scipy"]]:
    """
    Create a solver instance from a name string.

    Parameters
    ----------
    solver_name : str
        Solver identifier. See SOLVER_NAMES for available options.
    rtol : float
        Relative tolerance for optimistix solvers.
    atol : float
        Absolute tolerance for optimistix solvers.
    learning_rate : float
        Learning rate for adam solver.
    max_linesearch_steps : int
        Maximum linesearch steps for L-BFGS solvers.
    lower : PyTree, optional
        Lower bounds for box projection (optax solvers only).
    upper : PyTree, optional
        Upper bounds for box projection (optax solvers only).

    Returns
    -------
    solver : Solver can be either a BestSoFar wrapped minimiser or a string for scipy.
        The solver instance.
    solver_type : str
        One of "optimistix", "scipy".
    """
    # Resolve aliases
    # Optax solvers (with optional box projection)
    if solver_name == "optax_lbfgs":
        linesearch_type = kwargs.pop("linesearch", "zoom")
        if linesearch_type == "zoom":
            return optx.BestSoFarMinimiser(
                optx.OptaxMinimiser(
                    lbfgs_zoom(
                        max_linesearch_steps=max_linesearch_steps,
                        lower=lower,
                        upper=upper,
                        **kwargs,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
            ), "optimistix"
        elif linesearch_type == "backtracking":
            return optx.BestSoFarMinimiser(
                optx.OptaxMinimiser(
                    lbfgs_backtrack(
                        max_backtracking_steps=max_linesearch_steps,
                        lower=lower,
                        upper=upper,
                        **kwargs,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
            ), "optimistix"
        else:
            raise ValueError(
                f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
            )
    elif solver_name == "adam":
        # Chain adam with projection if bounds provided
        # learning_rate from kwargs takes precedence over function parameter
        lr = kwargs.pop("learning_rate", learning_rate)
        adam_opt = optax.adam(learning_rate=lr, **kwargs)
        if lower is not None and upper is not None:
            adam_opt = combine.chain(adam_opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(
            optx.OptaxMinimiser(adam_opt, atol=atol, rtol=rtol)
        ), "optimistix"
    elif solver_name == "sgd":
        # Chain sgd with projection if bounds provided
        # learning_rate from kwargs takes precedence (default 1.0 for linesearch use)
        lr = kwargs.pop("learning_rate", 1.0)
        direction = optax.sgd(learning_rate=lr)
        # Keep your line search
        linesearch = _linesearch.scale_by_backtracking_linesearch(
            max_backtracking_steps=max_linesearch_steps
        )
        if lower is not None and upper is not None:
            sgd_opt = combine.chain(direction, linesearch, apply_projection(lower, upper))
        else:
            sgd_opt = combine.chain(direction, linesearch)
        return optx.BestSoFarMinimiser(
            optx.OptaxMinimiser(sgd_opt, atol=atol, rtol=rtol)
        ), "optimistix"
    elif solver_name == "adabelief":
        lr = kwargs.pop("learning_rate", learning_rate)
        opt = optax.adabelief(learning_rate=lr)
        if lower is not None and upper is not None:
            opt = combine.chain(opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(optx.OptaxMinimiser(opt, atol=atol, rtol=rtol)), "optimistix"
    elif solver_name == "adaw" or solver_name == "adamw":
        lr = kwargs.pop("learning_rate", learning_rate)
        opt = optax.adamw(learning_rate=lr, **kwargs)
        if lower is not None and upper is not None:
            opt = combine.chain(opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(optx.OptaxMinimiser(opt, atol=atol, rtol=rtol)), "optimistix"
    elif solver_name == "active_set":
        # Default configuration for active set: Adam + configurable linesearch
        # Extract learning_rate and linesearch options
        lr = kwargs.pop("learning_rate", 1.0)
        linesearch_type = kwargs.pop("linesearch", "backtracking")

        direction = optax.adam(learning_rate=lr)
        if linesearch_type == "backtracking":
            linesearch = _linesearch.scale_by_backtracking_linesearch(
                max_backtracking_steps=max_linesearch_steps
            )
        elif linesearch_type == "zoom":
            linesearch = _linesearch.scale_by_zoom_linesearch(
                max_linesearch_steps=max_linesearch_steps
            )
        else:
            raise ValueError(
                f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
            )

        return optx.BestSoFarMinimiser(
            ActiveSetMinimiser(
                active_set(direction, linesearch, lower=lower, upper=upper, **kwargs),
                atol=atol,
                rtol=rtol,
            )
        ), "optimistix"
    elif solver_name == "active_set_sgd":
        # Default configuration for active set SGD: SGD + configurable linesearch
        # Extract learning_rate and linesearch options
        lr = kwargs.pop("learning_rate", 1.0)
        linesearch_type = kwargs.pop("linesearch", "backtracking")

        direction = optax.sgd(learning_rate=lr)

        if linesearch_type == "backtracking":
            linesearch = _linesearch.scale_by_backtracking_linesearch(
                max_backtracking_steps=max_linesearch_steps
            )
        elif linesearch_type == "zoom":
            linesearch = _linesearch.scale_by_zoom_linesearch(
                max_linesearch_steps=max_linesearch_steps
            )
        else:
            raise ValueError(
                f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
            )

        return optx.BestSoFarMinimiser(
            ActiveSetMinimiser(
                active_set(direction, linesearch, lower=lower, upper=upper, **kwargs),
                atol=atol,
                rtol=rtol,
            )
        ), "optimistix"
    elif solver_name == "active_set_adabelief" or solver_name.startswith("ADABK"):
        lr = kwargs.pop("learning_rate", 1.0)
        linesearch_type = kwargs.pop("linesearch", "zoom")
        max_constraints_to_release = kwargs.pop("max_constraints_to_release", None)
        if max_constraints_to_release is None:
            # check int in ADABKN as in ADABK5 for example
            if solver_name.startswith("ADABK") and len(solver_name) > 6:
                try:
                    max_constraints_to_release = int(solver_name[6:]) * 0.1
                except ValueError:
                    raise ValueError(
                        f"Invalid solver name: {solver_name}. "
                        f"When using 'ADABK' prefix, it should be followed by an integer."
                    )

        direction = optax.adabelief(learning_rate=lr)

        if linesearch_type == "backtracking":
            linesearch = _linesearch.scale_by_backtracking_linesearch(
                max_backtracking_steps=max_linesearch_steps
            )
        elif linesearch_type == "zoom":
            linesearch = _linesearch.scale_by_zoom_linesearch(
                max_linesearch_steps=max_linesearch_steps
            )
        else:
            raise ValueError(
                f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
            )

        return optx.BestSoFarMinimiser(
            ActiveSetMinimiser(
                active_set(
                    direction,
                    linesearch,
                    lower=lower,
                    upper=upper,
                    max_constraints_to_release=max_constraints_to_release,
                    **kwargs,
                ),
                atol=atol,
                rtol=rtol,
            )
        ), "optimistix"
    elif solver_name == "active_set_adaw":
        lr = kwargs.pop("learning_rate", 1.0)
        linesearch_type = kwargs.pop("linesearch", "backtracking")

        direction = optax.adamw(learning_rate=lr)

        if linesearch_type == "backtracking":
            linesearch = _linesearch.scale_by_backtracking_linesearch(
                max_backtracking_steps=max_linesearch_steps
            )
        elif linesearch_type == "zoom":
            linesearch = _linesearch.scale_by_zoom_linesearch(
                max_linesearch_steps=max_linesearch_steps
            )
        else:
            raise ValueError(
                f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
            )

        return optx.BestSoFarMinimiser(
            ActiveSetMinimiser(
                active_set(direction, linesearch, lower=lower, upper=upper, **kwargs),
                atol=atol,
                rtol=rtol,
            )
        ), "optimistix"
    # Optimistix BFGS
    elif solver_name == "optimistix_bfgs":
        return optx.BestSoFarMinimiser(optx.BFGS(rtol=rtol, atol=atol, **kwargs)), "optimistix"
    # Optimistix L-BFGS
    elif solver_name == "optimistix_lbfgs":
        return optx.BestSoFarMinimiser(optx.LBFGS(rtol=rtol, atol=atol, **kwargs)), "optimistix"
    # Optimistix NCG (Armijo)
    elif solver_name == "optimistix_ncg_pr":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.polak_ribiere, **kwargs)
        ), "optimistix"
    elif solver_name == "optimistix_ncg_hs":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.hestenes_stiefel, **kwargs)
        ), "optimistix"
    elif solver_name == "optimistix_ncg_fr":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.fletcher_reeves, **kwargs)
        ), "optimistix"
    elif solver_name == "optimistix_ncg_dy":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.dai_yuan, **kwargs)
        ), "optimistix"
    # Scipy
    elif solver_name == "scipy_tnc":
        # Note: ScipyMinimize needs fn passed at creation, handled in optimize()
        return "scipy_tnc", "scipy"
    elif solver_name == "scipy_cobyqa":
        return "scipy_cobyqa", "scipy"

    else:
        raise ValueError(f"Unknown solver: {solver_name}. Available: {list(SOLVER_NAMES.__args__)}")
