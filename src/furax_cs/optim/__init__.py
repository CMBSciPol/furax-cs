"""
Optimization utilities for FURAX component separation.

This package provides:
- L-BFGS solvers with zoom and backtracking linesearch
- Box projection transformation for constrained optimization
- Unified optimization interface supporting optax, optimistix, and scipy
- Function conditioning (parameter transformation and gradient-based scaling)

Example usage:
    >>> from furax_cs.optim import minimize
    >>>
    >>> # Simple optimization
    >>> params, state = minimize(
    ...     fn=objective,
    ...     init_params={'beta': 1.5},
    ...     solver_name='optax_lbfgs',
    ... )
"""

from .minimize import ScipyMinimizeState, minimize, scipy_minimize
from .solvers import (
    SOLVER_NAMES,
    apply_projection,
    get_solver,
    lbfgs_backtrack,
    lbfgs_zoom,
)
from .utils import condition

__all__ = [
    "SOLVER_NAMES",
    "ScipyMinimizeState",
    "apply_projection",
    "condition",
    "get_solver",
    "lbfgs_backtrack",
    "lbfgs_zoom",
    "scipy_minimize",
    "minimize",
]
