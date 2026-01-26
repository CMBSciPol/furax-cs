# Minimization Solvers

`furax-cs` provides a unified interface for various minimization solvers, including wrappers for **Optax**, **Optimistix**, and **SciPy**.

## Available Solvers

The `solver_name` argument in the `minimize` function accepts the following:

### Recommended
*   **`active_set`**: **Best for noisy maps.** Uses a projected gradient method with active set constraints. Robust against noise but might be slower on very clean data.
*   **`optax_lbfgs`**: **Best for noiseless runs.** L-BFGS with zoom linesearch (Strong Wolfe conditions). Very fast and accurate for smooth, noise-free landscapes.

### Other Options
*   `optax_lbfgs`: L-BFGS.
*   `adam`: Simple Adam optimizer (good for stochastic settings).
*   `scipy_tnc`: Wrapper for SciPy's Truncated Newton (TNC).
*   `optimistix_bfgs`: Standard BFGS from Optimistix.
*   `optimistix_lbfgs`: Standard L-BFGS from Optimistix.
*   `optimistix_ncg_*`: Nonlinear Conjugate Gradient variants (`pr`, `hs`, `fr`, `dy`).

## Minimizing Programmatically

### Single Run
```python
from furax_cs import minimize

final_params, state = minimize(
    fn=my_loss_fn,
    init_params=params,
    solver_name="active_set",  # or "optax_lbfgs"
    lower_bound=lower,
    upper_bound=upper,
    **kwargs
)
```

### Advanced: Steping interactively with Solvers

Since most solvers (except SciPy) are JAX-compatible, you can step through the optimization process manually. This is useful for custom logging or adaptive strategies.

Here is an example of running the same optimization problem with multiple solvers programmatically using `get_solver`.

```python
import jax
import jax.numpy as jnp
import optimistix as optx
from furax_cs.optim.solvers import get_solver
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
