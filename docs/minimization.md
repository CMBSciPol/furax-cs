# Minimization Solvers

`furax-cs` exposes a unified interface for various minimization solvers via the
[**CADRE**](https://github.com/CMBSciPol/CADRE) package (Constraint-Aware Descent Routine Executor),
which provides the underlying implementations for **Optax**, **Optimistix**, and **SciPy** solvers.

## Available Solvers

The `solver_name` argument in the `minimize` function accepts the following:

### Recommended
*   **`ADABK0`**: **Best for noisy maps.** AdaTopK: projected AdaBelief with a Top-K active set, releasing at most one constraint per iteration. Very robust in low-SNR regions. See [How AdaTopK Works](#how-adatopk-works) below.
*   **`optax_lbfgs`**: **Best for noiseless runs (systematics).** L-BFGS with zoom linesearch (Strong Wolfe conditions). Very fast and accurate for smooth, noise-free landscapes.

### Other Options

**AdaTopK** (self-conditioned):

*   `ADABK{N}` — AdaTopK releasing up to $\max(1, \lfloor 0.1\,N\,n \rfloor)$ constraints per iteration, with $n$ the number of parameters. `ADABK0` releases at most one constraint per iteration (most stable). See [How AdaTopK Works](#how-adatopk-works) and the paper for more information.

**Optax L-BFGS:**

*   `optax_lbfgs` — L-BFGS with zoom linesearch (default) or backtracking (`options={"linesearch": "backtracking"}`). The legacy names `zoom` and `backtrack` are aliases of `optax_lbfgs` with that line search.

**Optax first-order:**

*   `adam` — Adam optimizer.
*   `sgd` — SGD with backtracking linesearch.
*   `adabelief` — AdaBelief optimizer.
*   `adaw` / `adamw` — AdamW optimizer.

**Optimistix:**

*   `optimistix_bfgs` — Full BFGS.
*   `optimistix_lbfgs` — Limited-memory BFGS.
*   `optimistix_ncg_pr` — Nonlinear Conjugate Gradient (Polak-Ribière).
*   `optimistix_ncg_hs` — Nonlinear Conjugate Gradient (Hestenes-Stiefel).
*   `optimistix_ncg_fr` — Nonlinear Conjugate Gradient (Fletcher-Reeves).
*   `optimistix_ncg_dy` — Nonlinear Conjugate Gradient (Dai-Yuan).

**SciPy** (self-conditioned):

*   `scipy_tnc` — Truncated Newton (TNC).
*   `scipy_cobyqa` — COBYQA (derivative-free constrained optimizer).

For `scipy_tnc`, `atol` sets the SciPy options `ftol` and `xtol` and `rtol` sets `gtol`; for `scipy_cobyqa`, `atol` sets `final_tr_radius`. Other keys of `options` are passed to the SciPy method.


## How AdaTopK Works

AdaTopK (`ADABK{N}` in the solver names; Kabalan et al. 2026, arXiv:2604.08463) is a JAX-native minimiser for box-constrained problems. It combines a TNC-style active set with the AdaBelief direction (Zhuang et al. 2020) and a projected Armijo line search (Bertsekas 1982). It is implemented as an Optimistix minimiser, `cadre.AdaTopK`, and runs inside `optimistix.minimise`.

### Internal coordinates

Each parameter $x_i$ with bounds $[l_i, u_i]$ is mapped to

$$
z_i = \frac{x_i - c_i}{s_i}, \qquad c_i = \frac{l_i + u_i}{2}, \qquad s_i = u_i - l_i ,
$$

so every bounded parameter lives in $[-1/2, 1/2]$. A parameter without two finite bounds (unbounded, or bounded on one side only) uses $c_i = x_{0,i}$ and $s_i = 1 + |x_{0,i}|$, and a parameter with $l_i = u_i$ stays fixed at its bound. All steps are taken in $z$, which makes AdaTopK self-conditioned (see [Conditioning](#conditioning)).

### Active set and pivot vector

Each parameter has a pivot value $p_i$:

*   $p_i = -1$: at the lower bound (active constraint)
*   $p_i = +1$: at the upper bound (active constraint)
*   $p_i = 0$: free
*   $p_i = 2$: fixed ($l_i = u_i$)

Only free parameters move.

### One iteration

1.  **Release.** For every active constraint, the score $p_i\, g_i$ is positive when the descent direction $-g_i$ points into the box ($g$ is the gradient in internal coordinates). The $k$ constraints with the largest positive scores are released: $k = 1$ for `ADABK0`, $k = \max(1, \lfloor 0.1\,N\,n \rfloor)$ for `ADABK{N}`.
2.  **Direction.** The direction $d$ is AdaBelief on the gradient of the free parameters (heavy momentum $\beta_1 = 0.99$, short second-moment memory $\beta_2 = 0.9$, $\epsilon = 10^{-16}$). If the AdaBelief direction is not a descent direction, steepest descent of the same length is used for that step.
3.  **Projected path.** $z(\alpha) = \mathrm{clip}(z + \alpha\, d,\; z_l,\; z_u)$, coordinate by coordinate, so a parameter close to its bound does not limit the step of the others.
4.  **Line search.** Armijo backtracking along the projected path (sufficient-decrease constant $10^{-4}$, step halved at each trial, at most `max_linesearch_steps` trials). The first trial is twice the last accepted step, capped at 1. Each trial evaluates the value and the gradient, so the accepted trial provides the gradient of the next iteration.
5.  **Pinning.** Parameters that land on a bound while moving outwards become active constraints.
6.  **Restart or stop.** When the accepted step falls below `restart_alpha` or the line search fails, AdaTopK checks whether the best value improved by more than $\mathrm{ftol} \cdot \max(1, |f_\mathrm{best}|)$ since the last restart. If it did, the AdaBelief moments and the step length are reset and the iteration continues; otherwise the minimisation stops.

### Termination

AdaTopK stops when one of the following holds:

1.  the KKT residual, the norm of the projected gradient in physical units, satisfies $\|P(\nabla f)\| \le \mathrm{atol} + \mathrm{rtol} \cdot \|P(\nabla f(x_0))\|$. Here $P$ keeps the gradient of free parameters, $\min(g_i, 0)$ at a lower bound, $\max(g_i, 0)$ at an upper bound, and $0$ for fixed parameters;
2.  the restart-or-stop rule (step 6) stops it;
3.  `max_iter` iterations are reached.

Conditions 1 and 2 are ignored for `cooldown` iterations after a constraint release. The returned parameters are the best point found.

## AdaTopK Options

`atol`, `rtol` and `max_iter` are arguments of `minimize()`. The other options go in its `options` dict:

```python
final_params, state = minimize(
    fn=my_loss_fn,
    init_params=params,
    solver_name="ADABK0",
    max_iter=10000,
    atol=1e-18,
    rtol=1e-18,
    lower_bound=lower,
    upper_bound=upper,
    options={
        "cooldown": 50,          # default: 20
        "verbose_print": True,   # default: False
    },
)
```

| Option | Default | Description |
|--------|---------|-------------|
| `cooldown` | `20` | Iterations after a constraint release during which termination is suppressed. |
| `max_linesearch_steps` | `30` | Maximum Armijo backtracking trials per iteration. |
| `ftol` | `1e-15` | Relative improvement of the best value since the last restart required to restart (step 6) instead of stopping. |
| `restart_alpha` | `1e-10` | Accepted step length below which the step counts as collapsed (step 6). |
| `initial_step` | `1e-2` | Stored step length at the start and after a restart; the next line search starts at twice this value. |
| `armijo` | `1e-4` | Sufficient-decrease constant of the line search. |
| `backtrack` | `0.5` | Step reduction factor of the line search. |
| `b1`, `b2`, `eps` | `0.99`, `0.9`, `1e-16` | AdaBelief parameters. |
| `max_constraints_to_release` | none | Overrides the `ADABK{N}` release count: an int is a count, a float a fraction of the parameters. |
| `verbose_print` | `False` | Print per-iteration diagnostics (objective, KKT residual, step length, line-search trials, number of free parameters, release, restart, stop) with `jax.debug.print`, so it is JIT-compatible. |

With `record_history=True`, `minimize()` also returns the per-iteration step norm (internal coordinates) and objective in `state.history`. The AdaTopK state (`state.solver_state`) holds the KKT residual (`kkt`), the pivot vector (`pivot`) and counters (`n_evals`, `n_restart`, `n_ls_fail`, `n_safeguard`).

Any other key in `options` raises a `TypeError` for `ADABK*`: AdaTopK accepts only its own fields (and `cooldown`), so options of other solvers, such as `linesearch` or `learning_rate`, must not be passed to it.

### CLI equivalents

`kmeans-model` and `ptep-model` expose the cooldown and the diagnostics as flags:

```bash
kmeans-model ... -s ADABK0 --cooldown 50 --verbose
```

The `--min-steps` flag is still accepted but has no effect on the solver (it only appears in the run name).

## Conditioning

Conditioning (preconditioning) transforms the optimization problem to improve convergence. It applies two transformations before optimization:

1.  **Parameter scaling**: min-max normalization to [0, 1] based on bounds.
2.  **Gradient scaling**: the objective is scaled by 1/‖∇f‖ at initialization (like SciPy TNC's `fscale`), so the initial gradient norm is ≈ 1.

### Self-conditioned solvers

These solvers handle conditioning internally and ignore the `precondition` flag:

*   **AdaTopK** (`ADABK{N}`) — works in its own internal coordinates (see [Internal coordinates](#internal-coordinates)); `precondition=True` is ignored with a warning.
*   **SciPy solvers** (`scipy_tnc`, `scipy_cobyqa`) — SciPy handles bounds and scaling internally.

### Externally conditioned solvers

All other solvers (`optax_lbfgs`, `adam`, `optimistix_*`, etc.) benefit from external conditioning when dealing with poorly scaled problems. Pass `precondition=True` to `minimize`.

## Minimizing Programmatically

### Single Run
```python
from cadre import minimize

final_params, state = minimize(
    fn=my_loss_fn,
    init_params=params,
    solver_name="ADABK0",  # or "optax_lbfgs"
    lower_bound=lower,
    upper_bound=upper,
    **kwargs
)
```

### Using AdaTopK directly with Optimistix

`cadre.AdaTopK` is an `optimistix` minimiser, so it can be passed to `optimistix.minimise` without `minimize()`. The objective then takes `(params, args)`:

```python
import optimistix as optx
from cadre import AdaTopK

solver = AdaTopK(rtol=0.0, atol=1e-10, lower=lower, upper=upper, cooldown=50)
sol = optx.minimise(my_loss_fn, solver, params, args, max_steps=10_000, throw=False)

sol.value                # best parameters found
sol.stats["num_steps"]   # iterations
sol.state.best_loss, sol.state.kkt, sol.state.n_restart
```

### Advanced: Stepping Interactively with Solvers

Since most solvers (except SciPy) are JAX-compatible, you can step through the optimization process manually. This is useful for custom logging or adaptive strategies.

Here is an example of running the same optimization problem with multiple solvers programmatically using `get_solver`.

```python
import jax
import jax.numpy as jnp
import optimistix as optx
from cadre.solvers import get_solver
from functools import partial

# Define a simple quadratic loss function
def simple_loss(params, target):
    return jnp.sum((params - target) ** 2) , None

target = jnp.array([1.0, 2.0, 3.0])
init_params = jnp.array([0.0, 0.0, 0.0])
f_struct = jax.ShapeDtypeStruct((), jnp.float32)

# List of solvers to test

print(f"{'Loss':<12} | {'Step':<5} | {'Status':<10} | {'Final Params':<30}")
print("-" * 55)

# optax_lbfgs here; get_solver("ADABK0") returns an AdaTopK minimiser, whose loss is state.f
solver, _ = get_solver("optax_lbfgs", rtol=1e-6, atol=1e-6)

state = solver.init(simple_loss, init_params, target ,{}  , f_struct , None , frozenset())
step = partial(solver.step, fn=simple_loss, args=target, options={}, tags=frozenset())
terminate = partial(solver.terminate, fn=simple_loss, args=target, options={}, tags=frozenset())

y = init_params

for i in range(100):
    y , state , aux = step(y=y, state=state)
    done, result = terminate(y=y, state=state)
    loss = state.state.f
    print(f"{loss:<12.6f} | {i+1:<5} | {'Done' if done else 'In Progress':<10} | {y}")
    if done:
        break


y, _, _ = solver.postprocess(simple_loss, y, None, target, {}, state, frozenset(), result)

```
