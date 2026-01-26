from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from jax.flatten_util import ravel_pytree
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    PyTree,  # pyright: ignore
    Scalar,
)

from ..logging_utils import info

# --- Helper Logic ---


def _compute_initial_pivot(
    y: PyTree[Float[Array, " P"]],
    lower: PyTree[Float[Array, " P"]],
    upper: PyTree[Float[Array, " P"]],
    scale: PyTree[Float[Array, " P"]],
    offset: PyTree[Float[Array, " P"]],
) -> PyTree[Int[Array, " P"]]:
    """Compute initial pivot based on position relative to bounds."""

    def _leaf_pivot(
        y_leaf: Float[Array, " P"],
        lo: Float[Array, " P"],
        up: Float[Array, " P"],
        sc: Float[Array, " P"],
        off: Float[Array, " P"],
    ) -> Int[Array, " P"]:
        EPS = 1e-8  # Slightly relaxed tolerance for float32/64 stability
        # Calculate physical bounds tolerance
        # Handle inf: if lo is -inf, tol doesn't matter as we check bounds later
        tol_lower = EPS * 10.0 * (jnp.abs(lo) + 1.0)
        tol_upper = EPS * 10.0 * (jnp.abs(up) + 1.0)

        is_constant = (sc == 0.0) | (lo == up)

        # Calculate physical Y to check bounds
        y_phys = y_leaf * sc + off

        # Check bounds only if they are finite
        is_finite_lower = lo > -1e20
        is_finite_upper = up < 1e20

        at_lower = is_finite_lower & (y_phys - lo <= tol_lower) & ~is_constant
        at_upper = is_finite_upper & (up - y_phys <= tol_upper) & ~is_constant

        p = jnp.zeros_like(y_leaf, dtype=jnp.int32)
        p = jnp.where(at_lower, -1, p)
        p = jnp.where(at_upper, 1, p)
        p = jnp.where(is_constant, 2, p)
        return p

    return jtu.tree_map(_leaf_pivot, y, lower, upper, scale, offset)


def _compute_step_max(
    step_limit: Scalar,
    y_int: PyTree[Float[Array, " P"]],
    direction: PyTree[Float[Array, " P"]],
    pivot: PyTree[Int[Array, " P"]],
    lower: PyTree[Float[Array, " P"]],
    upper: PyTree[Float[Array, " P"]],
    scale: PyTree[Float[Array, " P"]],
    offset: PyTree[Float[Array, " P"]],
) -> Scalar:
    """Compute max step size alpha such that y + alpha * d stays in bounds."""

    def _leaf_step(
        y_leaf: Float[Array, " P"],
        d_leaf: Float[Array, " P"],
        p_leaf: Int[Array, " P"],
        lo: Float[Array, " P"],
        up: Float[Array, " P"],
        sc: Float[Array, " P"],
        off: Float[Array, " P"],
    ) -> Float[Array, " P"]:
        # Only check bounds if we are moving towards them

        # Internal bounds
        lo_int = (lo - off) / jnp.where(sc == 0, 1.0, sc)
        up_int = (up - off) / jnp.where(sc == 0, 1.0, sc)

        # If d < 0, we worry about lower bound
        # alpha * d >= lo_int - y  =>  alpha <= (lo_int - y) / d  (flip sign)
        t_lower = jnp.where(d_leaf < -1e-12, (lo_int - y_leaf) / d_leaf, jnp.inf)

        # If d > 0, we worry about upper bound
        # alpha * d <= up_int - y  => alpha <= (up_int - y) / d
        t_upper = jnp.where(d_leaf > 1e-12, (up_int - y_leaf) / d_leaf, jnp.inf)

        return jnp.minimum(t_lower, t_upper)

    max_steps = jtu.tree_map(_leaf_step, y_int, direction, pivot, lower, upper, scale, offset)

    # Global minimum step across all parameters
    flat_steps = jtu.tree_leaves(max_steps)
    if not flat_steps:
        return step_limit

    dist_to_bound = jnp.min(jnp.stack([jnp.min(s) for s in flat_steps]))

    # We take the smaller of the proposed step limit or the distance to bound
    return jnp.minimum(step_limit, dist_to_bound)


def _update_pivot_at_boundary(
    y_int: PyTree[Float[Array, " P"]],
    direction: PyTree[Float[Array, " P"]],
    pivot: PyTree[Int[Array, " P"]],
    lower: PyTree[Float[Array, " P"]],
    upper: PyTree[Float[Array, " P"]],
    scale: PyTree[Float[Array, " P"]],
    offset: PyTree[Float[Array, " P"]],
    step_size: Scalar,
) -> PyTree[Int[Array, " P"]]:
    """Update pivot if we landed exactly on a boundary."""

    def _leaf_add(
        y_leaf: Float[Array, " P"],
        d_leaf: Float[Array, " P"],
        p_leaf: Int[Array, " P"],
        lo: Float[Array, " P"],
        up: Float[Array, " P"],
        sc: Float[Array, " P"],
        off: Float[Array, " P"],
    ) -> Int[Array, " P"]:
        # Predict where we landed
        y_next = y_leaf + step_size * d_leaf
        y_next_phys = y_next * sc + off

        EPS = 1e-8
        tol_lower = EPS * 10.0 * (jnp.abs(lo) + 1.0)
        tol_upper = EPS * 10.0 * (jnp.abs(up) + 1.0)

        # If we were free (0) and now hit bound, update
        is_free = p_leaf == 0

        # Check if we hit lower
        hits_lower = is_free & (d_leaf < 0) & (y_next_phys - lo <= tol_lower)
        # Check if we hit upper
        hits_upper = is_free & (d_leaf > 0) & (up - y_next_phys <= tol_upper)

        new_p = p_leaf
        new_p = jnp.where(hits_lower, -1, new_p)
        new_p = jnp.where(hits_upper, 1, new_p)
        return new_p

    return jtu.tree_map(
        lambda y, d, p, l, u, s, o: _leaf_add(y, d, p, l, u, s, o),
        y_int,
        direction,
        pivot,
        lower,
        upper,
        scale,
        offset,
    )


def _tree_top_k(tree: PyTree, k: int) -> PyTree:
    """
    Finds the indices of the top K largest values across an entire PyTree.

    Returns:
        A PyTree of the same structure as input, containing boolean masks.
        True indicates the leaf value at that position is among the global top K.
    """
    # 1. Flatten the entire tree into a single 1D array
    flat_data, unravel_fn = ravel_pytree(tree)

    # 2. Handle edge case where k > total elements
    n_params = flat_data.shape[0]

    # 3. Find Top-K indices on the flat array
    # values: the top k values
    # indices: the flat indices of those values
    _, top_indices = jax.lax.top_k(flat_data, k)

    # 4. Create a flat boolean mask
    # Initialize all False
    flat_mask = jnp.zeros(n_params, dtype=bool)
    # Set the top-k positions to True
    flat_mask = flat_mask.at[top_indices].set(True)

    # 5. Unravel the flat mask back into the original PyTree structure
    return unravel_fn(flat_mask)


def _release_constraints(
    pivot: PyTree[Int[Array, " P"]],
    gradients_int: PyTree[Float[Array, " P"]],
    max_release_k: int,
) -> tuple[PyTree[Int[Array, " P"]], Bool[Array, ""]]:
    """
    Release constraints if the negative gradient points into the feasible region.
    TNC checks: if pivot=-1 (at lower) and -grad > 0 (descent direction is up), release.

    This version uses a "score" (pivot * grad) and only releases the top-k
    constraints with the highest positive score (strongest desire to release).
    """

    def _compute_score(p: Int[Array, " P"], g: Float[Array, " P"]) -> Float[Array, " P"]:
        # descent direction is -g
        # Score = pivot * gradient
        # If at lower bound (p=-1), we release if -g > 0 => g < 0. Score = (-1)*(-ve) = +ve
        # If at upper bound (p=1),  we release if -g < 0 => g > 0. Score = (1)*(+ve)  = +ve

        # We only care about active constraints (abs(p) == 1)
        # We mask out non-active constraints (0 or 2) with -inf
        score = p * g
        return jnp.where(jnp.abs(p) == 1, score, -jnp.inf)

    # 1. Calculate scores for all parameters
    scores = jtu.tree_map(_compute_score, pivot, gradients_int)

    # Check if any constraint wants to be released
    flat_scores, _ = ravel_pytree(scores)
    constraints_released = jnp.any(flat_scores > 0)

    # 2. Find the mask for the Top-K scores globally
    top_k_mask = _tree_top_k(scores, max_release_k)

    # 3. Determine actual release mask:
    #    - Must be in Top K
    #    - Must have a positive score (gradient actually points inward)
    def _apply_release(
        p: Int[Array, " P"], is_top_k: Bool[Array, " P"], s: Float[Array, " P"]
    ) -> Int[Array, " P"]:
        should_release = is_top_k & (s > 0)
        return jnp.where(should_release, 0, p)

    return jtu.tree_map(_apply_release, pivot, top_k_mask, scores), constraints_released


# --- Active Set Component ---
def _rescale_adam_state(state: optax.OptState, scale_factor: Scalar) -> optax.OptState:
    """
    Recursively searches for Adam/AdaBelief states and rescales moments.
    Robustly handles optax.chain (tuples) and leaf states (NamedTuples).
    """
    # 1. Identify Adam/AdaBelief State (Target)
    if hasattr(state, "mu") and hasattr(state, "nu"):
        return state._replace(
            mu=otu.tree_scale(scale_factor, state.mu), nu=otu.tree_scale(scale_factor**2, state.nu)
        )

    # 2. Recurse into Containers (optax.chain uses plain tuples)
    # CRITICAL FIX: We must exclude NamedTuples (like EmptyState) from this check.
    # Plain tuples do not have `_fields`; NamedTuples do.
    elif isinstance(state, tuple | list) and not hasattr(state, "_fields"):
        return type(state)(_rescale_adam_state(s, scale_factor) for s in state)

    # 3. Leave everything else alone (EmptyState, ScheduleState, etc.)
    return state


class ActiveSetState(NamedTuple):
    count: Scalar
    pivot: PyTree[Int[Array, " P"]]
    xscale: PyTree[Float[Array, " P"]]
    offset: PyTree[Float[Array, " P"]]
    lower: PyTree[Float[Array, " P"]]
    upper: PyTree[Float[Array, " P"]]
    fscale: Scalar
    stepmx: Scalar
    max_release_k: Scalar
    direction_state: optax.OptState
    linesearch_state: optax.OptState
    constraints_released: Bool[Array, ""]


def active_set(
    direction_solver: optax.GradientTransformation,
    linesearch_solver: optax.GradientTransformation,
    lower: Optional[PyTree[Float[Array, " P"]]] = None,
    upper: Optional[PyTree[Float[Array, " P"]]] = None,
    rescale_threshold: float = 1.3,
    stepmx_init: float = 10.0,
    max_constraints_to_release: Optional[Union[int, float]] = None,
    verbose: bool = False,
) -> optax.GradientTransformation:
    def init_fn(params: PyTree[Float[Array, " P"]]) -> ActiveSetState:
        lo = lower if lower is not None else otu.tree_full_like(params, -jnp.inf)
        up = upper if upper is not None else otu.tree_full_like(params, jnp.inf)

        leaves = jtu.tree_leaves(params)
        total_params = sum(leaf.size for leaf in leaves)

        if max_constraints_to_release is None:
            # Default: 10% of params
            k_val = max(1, total_params // 10)
        elif isinstance(max_constraints_to_release, float):
            k_val = max(1, int(total_params * max_constraints_to_release))
        else:
            k_val = min(max_constraints_to_release, total_params)
            k_val = max(1, k_val)

        info(f"key active_set: max_constraints_to_release={k_val} / {total_params} params")

        # Init Scale & Offset logic from TNC
        def _init_scale(
            p: Float[Array, " P"], l: Float[Array, " P"], u: Float[Array, " P"]
        ) -> Float[Array, " P"]:
            is_bounded = (l > -1e20) & (u < 1e20)
            s_b = u - l
            s_u = 1.0 + jnp.abs(p)
            return jnp.where(is_bounded, s_b, s_u)

        def _init_offset(
            p: Float[Array, " P"], l: Float[Array, " P"], u: Float[Array, " P"]
        ) -> Float[Array, " P"]:
            is_bounded = (l > -1e20) & (u < 1e20)
            o_b = (l + u) * 0.5
            o_u = p
            return jnp.where(is_bounded, o_b, o_u)

        xscale = jtu.tree_map(_init_scale, params, lo, up)
        offset = jtu.tree_map(_init_offset, params, lo, up)

        # Calculate internal Y: y = (x - offset) / scale
        y_int = otu.tree_div(otu.tree_sub(params, offset), xscale)
        pivot = _compute_initial_pivot(y_int, lo, up, xscale, offset)

        return ActiveSetState(
            count=jnp.array(0, dtype=jnp.int32),
            pivot=pivot,
            xscale=xscale,
            offset=offset,
            lower=lo,
            upper=up,
            fscale=jnp.array(1.0),
            stepmx=jnp.array(stepmx_init),
            max_release_k=k_val,
            direction_state=direction_solver.init(params),
            linesearch_state=linesearch_solver.init(params),
            constraints_released=jnp.array(False),
        )

    def update_fn(
        grads: PyTree[Float[Array, " P"]],
        state: ActiveSetState,
        params: Optional[PyTree[Float[Array, " P"]]] = None,
        value: Optional[Scalar] = None,
        grad: Optional[PyTree[Float[Array, " P"]]] = None,
        value_fn: Optional[Callable[[PyTree[Float[Array, " P"]]], Scalar]] = None,
        **kwargs: Any,
    ) -> tuple[PyTree[Float[Array, " P"]], ActiveSetState]:
        if params is None or value_fn is None:
            raise ValueError("active_set requires 'params' and 'value_fn' arguments.")
        if value is None:
            # We need value for line search, but optax doesn't always provide it.
            # However, in furax it should be provided.
            value = value_fn(params)

        # --- 1. Internal Representation ---
        # Current internal point y
        y_int = otu.tree_div(otu.tree_sub(params, state.offset), state.xscale)

        # Scale Gradients to Internal Space: g_int = g_phys * xscale * fscale
        grads_int = otu.tree_scale(state.fscale, otu.tree_mul(grads, state.xscale))

        # --- 2. Release Active Constraints ---
        # If gradient points inward from a bound, release it (set pivot to 0)
        # TNC does this *before* projection
        pivot, constraints_released = _release_constraints(
            state.pivot, grads_int, state.max_release_k
        )

        # --- 3. Project Gradients (Input Masking) ---
        # If pivot != 0, set gradient to 0 so the solver "thinks" we are optimal there
        grads_int_proj = jax.tree.map(lambda p, pk: jnp.where(p == 0, pk, 0.0), pivot, grads_int)

        # --- 4. Dynamic Rescaling (TNC Logic) ---
        gnorm = otu.tree_norm(grads_int_proj)
        safe_gnorm = gnorm + 1e-20
        should_rescale = (gnorm > 1e-20) & (jnp.abs(jnp.log10(safe_gnorm)) > rescale_threshold)

        # If rescaling, we scale the gradients AND fscale
        grads_int_proj = otu.tree_where(
            should_rescale, otu.tree_scale(1.0 / safe_gnorm, grads_int_proj), grads_int_proj
        )
        new_fscale = jnp.where(should_rescale, state.fscale / safe_gnorm, state.fscale)

        current_dir_state = otu.tree_where(
            should_rescale,
            _rescale_adam_state(state.direction_state, 1.0 / safe_gnorm),
            state.direction_state,
        )

        # --- 5. Compute Direction (pk) ---
        # Use inner solver (Adam/LBFGS). Note: We pass internal gradients.
        pk, new_dir_state = direction_solver.update(grads_int_proj, current_dir_state, params)

        # --- FIX 1: Project Direction (Output Masking) ---
        # CRITICAL: Adam has momentum. Even if input grad is 0, output `pk` might not be.
        # We must force pk to 0 on active constraints to stop pushing into the wall.
        pk = jax.tree.map(lambda p, pk: jnp.where(p == 0, pk, 0.0), state.pivot, pk)
        # pk = otu.tree_where(pivot_is_zero, pk, otu.tree_full_like(pk, 0))

        if verbose:
            jax.debug.print("grads_int: {}", grads_int)
            jax.debug.print("pk: {}", pk)
            jax.debug.print("Iter {i} | gnorm {g}", i=state.count, g=gnorm)

        # --- 6. Step Limit (spe) ---
        # Calculate maximum step `spe` along `pk` before hitting ANY bound.
        pk_norm = otu.tree_norm(pk)

        # TNC heuristic for max unconstrained step
        ustpmax = state.stepmx / (pk_norm + 1e-20)

        # Distance to nearest bound along pk
        spe = _compute_step_max(
            ustpmax, y_int, pk, pivot, state.lower, state.upper, state.xscale, state.offset
        )

        # --- 7. Line Search ---
        # We need to wrap value_fn so Line Search sees Internal Space
        def internal_value_fn(y_candidate: PyTree[Float[Array, " P"]]) -> Scalar:
            # Unscale y -> x
            x_candidate = otu.tree_add(otu.tree_mul(y_candidate, state.xscale), state.offset)
            # Clip x_candidate to ensure physical bounds aren't violated by float noise
            x_candidate = jtu.tree_map(jnp.clip, x_candidate, state.lower, state.upper)
            return value_fn(x_candidate) * new_fscale  # Line search sees scaled function value

        # Optax LS typically calculates: param + update
        # We pass y_int as params, pk as update.
        # LS returns `scaled_update` which is (alpha * pk)
        ls_update_int, new_ls_state = linesearch_solver.update(
            pk,
            state.linesearch_state,
            y_int,
            value=value * new_fscale,
            grad=grads_int_proj,
            value_fn=internal_value_fn,
        )

        # --- 8. Step Clamping & Pivot Update ---
        # ls_update_int represents the desired step.
        # We must clamp its magnitude to `spe` * norm(pk)

        ls_step_len = otu.tree_norm(ls_update_int)  # approx alpha * pk_norm

        # Max length allowed = spe * pk_norm
        max_len = spe * pk_norm

        # If ls_step_len > max_len, we hit the wall.
        hit_wall = ls_step_len > max_len + 1e-10

        # Clamp factor
        clamp_scale = jnp.where(hit_wall, max_len / (ls_step_len + 1e-20), 1.0)

        final_update_int = otu.tree_scale(clamp_scale, ls_update_int)

        # If we clamped (hit wall), we must update the pivot to lock that variable
        # We pass the step size 'spe' implicitly by calculating where we land
        final_pivot = lax.cond(
            hit_wall,
            lambda: _update_pivot_at_boundary(
                y_int, pk, pivot, state.lower, state.upper, state.xscale, state.offset, spe
            ),
            lambda: pivot,  # No change if we didn't hit wall
        )

        # --- 9. Unscale Updates ---
        # x_new = x_old + dx
        # y_new = y_old + dy
        # (x_new - off)/sc = (x_old - off)/sc + dy
        # x_new = x_old + dy * sc
        # So physical update = internal update * xscale
        updates_phys = otu.tree_mul(final_update_int, state.xscale)

        new_state = ActiveSetState(
            count=state.count + 1,
            pivot=final_pivot,
            xscale=state.xscale,
            offset=state.offset,
            lower=state.lower,
            upper=state.upper,
            fscale=new_fscale,
            stepmx=state.stepmx,
            max_release_k=state.max_release_k,
            direction_state=new_dir_state,
            linesearch_state=new_ls_state,
            constraints_released=constraints_released,
        )

        return updates_phys, new_state

    return optax.GradientTransformation(init_fn, update_fn)
