# Minimization Solvers

`furax-cs` exposes a unified interface for various minimization solvers via the
[**CADRE**](https://github.com/CMBSciPol/CADRE) package (Constraint-Aware Descent Routine Executor),
which provides the underlying implementations for **Optax**, **Optimistix**, and **SciPy** solvers.

## Available Solvers

The `solver_name` argument in the `minimize` function accepts the following:

### Recommended
*   **`ADABK0`**: **Best for noisy maps.** Active-set method with AdaBelief direction and Top-K constraint release (K=0, i.e. one constraint released per iteration). Very robust in low-SNR regions. See [How ADABK Works](#how-adabk-works) below.
*   **`optax_lbfgs`**: **Best for noiseless runs (systematics).** L-BFGS with zoom linesearch (Strong Wolfe conditions). Very fast and accurate for smooth, noise-free landscapes.

### Other Options

**Active set variants** (self-conditioned):

*   `ADABK{N}` — AdaBelief + Top-K active set. `N * 0.1` = fraction of constraints released per step. `ADABK0` releases 1 constraint/step (most stable), `ADABK5` releases up to 50%. (see [How ADABK Works](#how-adabk-works) and paper for more info)
*   `active_set` — Active set with Adam direction.
*   `active_set_sgd` — Active set with SGD direction.
*   `active_set_adabelief` — Active set with AdaBelief direction.
*   `active_set_adaw` — Active set with AdamW direction.

**Optax L-BFGS:**

*   `optax_lbfgs` — L-BFGS with zoom linesearch (default) or backtracking.

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


## How ADABK Works

ADABK (Adaptive AdaBelief with Top-K Active Set, also called **AdaTopK** in the paper) is a JAX-native optimizer that combines the TNC active-set constraint strategy with the AdaBelief adaptive gradient method.

### Internal parameter space

Physical parameters **x** (bounded by **l**, **u**) are mapped to a normalized [0, 1] representation via an affine transform:

**y** = (**x** − **l**) / (**u** − **l**)

This normalizes the optimization landscape and ensures consistent step sizes across parameters with different physical scales.

### Active set and pivot vector

Each parameter has a pivot value p_i:

*   p_i = −1: parameter is at the lower bound (active constraint)
*   p_i = +1: parameter is at the upper bound (active constraint)
*   p_i = 0: parameter is free

Only free parameters (p_i = 0) are optimized at each iteration.

### Top-K constraint release

At each iteration, a release score is computed for every active constraint:

score_i = p_i × (−g_i)

A positive score means the negative gradient points into the feasible region — releasing this constraint could decrease the objective. The Top-K fraction K controls how many constraints are released per iteration:

*   **K = 0** (`ADABK0`): releases 1 constraint at a time. Most stable, consistently reaches the lowest objective values.
*   **K = N** (`ADABK{N}`): releases up to `N × 0.1` fraction of active constraints.

### Projected gradient and AdaBelief direction

Gradients for active constraints are zeroed out: **g_proj** = **g** ⊙ (p = 0). The projected gradient is then fed to AdaBelief, which adapts step sizes based on gradient variance. This makes it better suited to noisy gradient landscapes (low-SNR regions) than classical quasi-Newton methods (L-BFGS, TNC) which tend to reset their curvature history when gradients are unreliable.

### Dynamic state rescaling

When the gradient norm falls outside [10⁻¹⁵, 10¹⁵], the cost function and AdaBelief moment estimates are rescaled:

**m** ← f_scale · **m** ,  **v** ← f_scale² · **v**

This prevents numerical under/overflow across the extreme dynamic range between the bright Galactic plane and faint high-latitude sky, without resetting the optimizer's momentum.

### Bounded line search

The step size α is capped at the distance to the nearest bound (α_max), then a line search finds the optimal α in [0, α_max]. If a parameter hits a bound, it becomes an active constraint.

## Termination Control (ADABK solvers)

Four parameters control when ADABK solvers decide to stop. Pass them via the `options` dict of `minimize()`:

```python
final_params, state = minimize(
    fn=my_loss_fn,
    init_params=params,
    solver_name="ADABK0",
    lower_bound=lower,
    upper_bound=upper,
    options={
        "cooldown": 50,              # default: 20
        "min_steps": 200,            # default: 10
        "verbose_print": True,       # default: False
        "max_linesearch_steps": 100, # default: 50
    },
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cooldown` | `20` | Steps to suppress termination after a constraint is released. Prevents premature convergence caused by a transient function spike when a bound constraint opens. |
| `min_steps` | `10` | Minimum iterations before termination is ever considered. Useful when the initial gradient is near zero but the landscape is not yet explored. |
| `verbose_print` | `False` | Print per-step diagnostics: current `f`, `f_diff`, `best_f`, cooldown status, and termination decision. Uses `jax.debug.print` so it is JIT-compatible. |
| `max_linesearch_steps` | `50` | Maximum bounded line-search steps per iteration. |

### How termination is decided

Termination requires **all** of the following to hold simultaneously:

1. `f_diff = |f_current − f_prev| < atol + rtol × max(1, |best_f|)` — spike-immune f-change check
2. Cauchy-convergence in y-space (base Optimistix check)
3. Step count ≥ `min_steps`
4. Not inside the cooldown window after the last constraint release

### CLI equivalents

When using `kmeans-model` or `ptep-model`, the same parameters are available as flags:

```bash
kmeans-model ... --cooldown 50 --min-steps 200 --verbose
```

## Conditioning

Conditioning (preconditioning) transforms the optimization problem to improve convergence. It applies two transformations before optimization:

1.  **Parameter scaling**: min-max normalization to [0, 1] based on bounds.
2.  **Gradient scaling**: the objective is scaled by 1/‖∇f‖ at initialization (like SciPy TNC's `fscale`), so the initial gradient norm is ≈ 1.

### Self-conditioned solvers

These solvers handle conditioning internally and ignore the `precondition` flag:

*   **Active set variants** (`active_set`, `active_set_sgd`, `active_set_adabelief`, `active_set_adaw`, `ADABK{N}`) — use internal affine transform + dynamic state rescaling.
*   **SciPy solvers** (`scipy_tnc`, `scipy_cobyqa`) — SciPy handles bounds and scaling internally.

### Externally conditioned solvers

All other solvers (`optax_lbfgs`, `adam`, `optimistix_*`, etc.) benefit from external conditioning when dealing with poorly scaled problems. Pass `precondition=True` (or a custom scaling function) to `minimize`.

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

# active_set or optax_lbfgs for example
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
