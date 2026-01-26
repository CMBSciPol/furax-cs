import os
from functools import partial
from typing import Any, Optional, Union

import healpy as hp
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np
from furax import Config
from furax._instruments.sky import FGBusterInstrument
from furax.obs import negative_log_likelihood
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_healpy.clustering import get_cutout_from_mask, get_fullmap_from_cutout
from jaxtyping import Array, Float, Int, PyTree
from tqdm import tqdm

from ..logging_utils import debug, error, info, success
from .plotting import get_run_color, save_or_show, set_font_size
from .utils import index_run_data


def _parse_perturb_spec(spec: str, values: Array) -> Array | None:
    """Parse perturbation spec: 'all', '-1', '0,1,2,3', '0:30', 'max', or 'min'.

    Returns None if spec is '-1' (skip this param).
    """
    n_params = values.shape[0]
    if spec == "-1":
        return None
    if spec == "all":
        return np.arange(n_params)
    if spec == "max":
        return np.array([np.argmax(values)])
    if spec == "min":
        return np.array([np.argmin(values)])
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else n_params
        return np.arange(start, min(end, n_params))
    return np.array([int(x) for x in spec.split(",")])


def compute_gradient_validation(
    # Data & Parameters
    final_params: PyTree[Float[Array, " P"]],
    patch_indices: PyTree[Int[Array, " P"]],
    mask_indices: Int[Array, " indices"],
    # Instrument & Config
    instrument: FGBusterInstrument,
    nside: int,
    # Validation Settings
    scales: list[float],
    steps_range: int,
    noised_d: Stokes,
    small_n: Stokes,
    # Perturbation specs per parameter
    perturb_beta_dust: str = "all",
    perturb_beta_pl: str = "all",
    perturb_temp_dust: str = "all",
    # Execution mode
    use_vmap: bool = True,
) -> dict[str, Any]:
    """Computes NLL and gradient norms for perturbed parameters across multiple scales.

    Validates the stability of the optimization by perturbing the solution and checking
    if the Negative Log-Likelihood (NLL) increases and gradients behave as expected.

    Args:
        final_params: The optimized parameters (beta_dust, beta_pl, temp_dust).
        patch_indices: Dictionary containing patch assignments for each parameter.
        mask_indices: Array of indices where the mask is applied (value 1).
        instrument: The instrument configuration object.
        nside: HEALPix nside resolution.
        scales: List of scaling factors for perturbation (e.g., [1e-1, 1e-2]).
        steps_range: Number of steps to perturb in positive and negative directions.
        noised_d: The noised data used in the run.
        small_n: The noise variance map used in the run.
        perturb_beta_dust: Spec string for beta_dust perturbation ('all', '-1', '0:3').
            Defaults to "all".
        perturb_beta_pl: Spec string for beta_pl perturbation. Defaults to "all".
        perturb_temp_dust: Spec string for temp_dust perturbation. Defaults to "all".
        use_vmap: Whether to use jax.vmap for vectorized computation. Defaults to True.

    Returns:
        A dictionary containing keys 'scales', 'steps', and 'results' (mapping scale -> metrics).
    """

    # 1. Construct Noise Operator & Data
    N = NoiseDiagonalOperator(small_n, _in_structure=noised_d.structure)

    # 2. Define Likelihood Functions
    dust_nu0 = 150.0
    synchrotron_nu0 = 20.0
    nu = instrument.frequency

    negative_log_likelihood_fn = partial(
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
    )

    @jax.jit
    def grad_nll(params: dict[str, Array]) -> dict[str, Array]:
        return jax.grad(negative_log_likelihood_fn)(
            params,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

    @jax.jit
    def nll(params: dict[str, Array]) -> Array:
        return negative_log_likelihood_fn(
            params,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

    # 3. Parse perturbation specs and create masks
    n_bd = final_params["beta_dust"].shape[0]
    n_bp = final_params["beta_pl"].shape[0]
    n_td = final_params["temp_dust"].shape[0]

    idx_bd = _parse_perturb_spec(perturb_beta_dust, final_params["beta_dust"])
    idx_bp = _parse_perturb_spec(perturb_beta_pl, final_params["beta_pl"])
    idx_td = _parse_perturb_spec(perturb_temp_dust, final_params["temp_dust"])

    # Create masks (1 where perturb, 0 elsewhere)
    mask_bd = np.zeros(n_bd)
    mask_bp = np.zeros(n_bp)
    mask_td = np.zeros(n_td)
    if idx_bd is not None:
        mask_bd[idx_bd] = 1
        info(
            f"Perturbing beta_dust at indices: {idx_bd} with values {final_params['beta_dust'][idx_bd]}"
        )
    if idx_bp is not None:
        info(
            f"Perturbing beta_pl at indices: {idx_bp} with values {final_params['beta_pl'][idx_bp]}"
        )
        mask_bp[idx_bp] = 1
    if idx_td is not None:
        info(
            f"Perturbing temp_dust at indices: {idx_td} with values {final_params['temp_dust'][idx_td]}"
        )
        mask_td[idx_td] = 1

    masks = {
        "beta_dust": jnp.array(mask_bd),
        "beta_pl": jnp.array(mask_bp),
        "temp_dust": jnp.array(mask_td),
    }

    # 4. Compute Validation Metrics
    steps = jnp.arange(-steps_range, steps_range + 1)  # inclusive range
    results = {}

    info("Computing NLLs and Gradients for multiple scales...")

    for scale in scales:
        info(f"  Processing scale: {scale:.1e}")

        # Calculate perturbations for this scale
        # Shape: (n_steps, 1)
        perturbations = steps.reshape(-1, 1) * scale

        # Apply perturbation with masks
        final_params_perturbed = {
            "beta_dust": final_params["beta_dust"].reshape(1, -1)
            + perturbations * masks["beta_dust"],
            "beta_pl": final_params["beta_pl"].reshape(1, -1) + perturbations * masks["beta_pl"],
            "temp_dust": final_params["temp_dust"].reshape(1, -1)
            + perturbations * masks["temp_dust"],
        }

        # Compute NLLs and gradients
        with Config(solver=lx.CG(rtol=1e-6, atol=1e-10, max_steps=10000)):
            if use_vmap:
                # Vectorized computation (faster but more memory)
                nlls = jax.vmap(nll)(final_params_perturbed)
                grads = jax.vmap(grad_nll)(final_params_perturbed)
            else:
                # For-loop approach (slower but less memory)
                n_steps = len(steps)
                nlls_list = []
                grads_list = []
                for i in tqdm(range(n_steps), desc="  Validating", unit="step"):
                    params_i = jax.tree.map(lambda x: x[i], final_params_perturbed)
                    nlls_list.append(nll(params_i))
                    grads_list.append(grad_nll(params_i))
                nlls = jnp.stack(nlls_list)
                grads = jax.tree.map(lambda *xs: jnp.stack(xs), *grads_list)

        # Compute Norms of gradients
        grads_beta_dust_norm = jnp.linalg.norm(grads["beta_dust"], axis=1)
        grads_beta_pl_norm = jnp.linalg.norm(grads["beta_pl"], axis=1)
        grads_temp_dust_norm = jnp.linalg.norm(grads["temp_dust"], axis=1)

        results[scale] = {
            "NLL": nlls,
            "grads_beta_dust": grads_beta_dust_norm,
            "grads_beta_pl": grads_beta_pl_norm,
            "grads_temp_dust": grads_temp_dust_norm,
            "grads_raw": grads,  # Store raw gradients for grad-maps
        }

    return {"scales": scales, "steps": steps, "results": results}


def compute_2d_validation(
    tem_name: str,
    final_params: PyTree[Float[Array, " P"]],
    patch_indices: PyTree[Int[Array, " P"]],
    mask_indices: Int[Array, " indices"],
    instrument: FGBusterInstrument,
    nside: int,
    param_scales: dict[str, float],
    steps_range: int,
    active_params: list[str],
    perturb_specs: dict[str, str],
    noised_d: Stokes,
    small_n: Stokes,
    use_vmap: bool = True,
) -> dict[str, Any]:
    """Computes NLL on a 2D grid of perturbed parameters.

    Args:
        final_params: Optimized parameters.
        patch_indices: Patch assignments.
        mask_indices: Mask indices.
        instrument: Instrument configuration.
        nside: HEALPix nside.
        param_scales: Dictionary mapping parameter names to their perturbation scales.
        steps_range: Number of steps in each direction.
        active_params: List of the two parameters to perturb (e.g. ["beta_dust", "temp_dust"]).
        perturb_specs: Dictionary of perturbation specs for all parameters.
        noised_d: The noised data.
        small_n: The noise variance map.
        use_vmap: Whether to use vmap.

    Returns:
        Dictionary with NLL grid, steps, active params, and scales.
    """
    info(f"Computing 2D NLL surface for {active_params}...")

    if noised_d.shape[-1] != mask_indices.size:
        noised_d = get_cutout_from_mask(noised_d, mask_indices, axis=-1)

    # 1. Setup Noise & Likelihood (Same as 1D)
    N = NoiseDiagonalOperator(small_n, _in_structure=noised_d.structure)

    dust_nu0 = 150.0
    synchrotron_nu0 = 20.0
    nu = instrument.frequency

    negative_log_likelihood_fn = partial(
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
    )

    @jax.jit
    def nll(params: dict[str, Array]) -> Array:
        return negative_log_likelihood_fn(
            params,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

    # 2. Setup Grid and Masks
    p1_name, p2_name = active_params
    scale1 = param_scales[p1_name]
    scale2 = param_scales[p2_name]

    # Parse specs and create masks
    masks = {}
    base_values = {}
    for name in active_params:
        n_param = final_params[name].shape[0]
        idx = _parse_perturb_spec(perturb_specs[name], final_params[name])
        m = np.zeros(n_param)
        if idx is not None:
            m[idx] = 1.0
            if idx.size == 1:
                base_values[name] = float(final_params[name][idx[0]])
        masks[name] = jnp.array(m)

    # Create Grid
    steps = jnp.arange(-steps_range, steps_range + 1)
    # indexing='ij' -> G1 changes with row i, G2 with col j
    G1, G2 = jnp.meshgrid(steps, steps, indexing="ij")

    # Flatten for vectorized evaluation
    flat_G1 = G1.ravel()
    flat_G2 = G2.ravel()

    perturbations1 = flat_G1 * scale1  # (N_grid,)
    perturbations2 = flat_G2 * scale2  # (N_grid,)

    # reshape for broadcasting: (N_grid, 1)
    P1 = perturbations1[:, None]
    P2 = perturbations2[:, None]

    # 3. Construct Parameter PyTree
    # We need to broadcast final_params to (N_grid, N_dim) and add perturbations
    params_expanded = {}
    n_grid = len(flat_G1)

    for name, val in final_params.items():
        # val is (N_dim,) -> expand to (N_grid, N_dim)
        base = jnp.repeat(val[None, :], n_grid, axis=0)

        if name == p1_name:
            base = base + P1 * masks[p1_name]
        if name == p2_name:
            base = base + P2 * masks[p2_name]

        params_expanded[name] = base

    # 4. Compute
    if use_vmap:
        nlls_flat = jax.vmap(nll)(params_expanded)
    else:
        # Fallback loop
        res_list = []
        for i in range(n_grid):
            p_i = jax.tree.map(lambda x: x[i], params_expanded)
            res_list.append(nll(p_i))
        nlls_flat = jnp.stack(res_list)

    nll_grid = nlls_flat.reshape(G1.shape)

    return {
        "NLL_grid": nll_grid,
        "steps": steps,
        "active_params": active_params,
        "scales": param_scales,
        "base_values": base_values,
    }


def _plot_lines_on_ax(
    ax: plt.Axes,
    validation_results: list[dict[str, Any]],
    labels: list[str],
    metric_key: str,
    reference_mins: dict[float, float] | None = None,
    is_nll: bool = False,
    use_legend: bool = True,
) -> None:
    """Helper to plot lines on a given axis, handling aggregation and shared minimums."""
    # Common Setup
    first_res = validation_results[0]
    scales = first_res["scales"]
    steps = first_res["steps"]

    # Visuals: colors by run (using shared palette), all lines dashed with markers
    n_runs = len(validation_results)
    colors = [get_run_color(i) for i in range(n_runs)]
    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "h", "*"]
    use_markers = len(steps) <= 10

    # Global Minimum Logic for NLL

    # Plotting Loop: colors and markers by run, all dashed lines
    for run_idx, (val_res, label) in enumerate(zip(validation_results, labels)):
        color = colors[run_idx]
        marker_char = markers[run_idx % len(markers)] if use_markers else None

        for scale_idx, scale in enumerate(scales):
            data = val_res["results"][scale]
            y_values = data[metric_key]

            # Label Generation
            if is_nll:
                assert reference_mins is not None, "reference_mins required for NLL plots"
                # Calculate stats relative to global min
                idx_zero = jnp.argmin(jnp.abs(steps))
                nll_zero = y_values[idx_zero]
                reference_min = reference_mins[scale]

                abs_diff = nll_zero - reference_min
                rel_diff = abs(abs_diff / reference_min) if reference_min != 0 else 0.0

                # Construct Legend
                prefix = f"{label} " if label else ""

                final_label = (
                    f"{prefix}\n"
                    f" Sol:  {nll_zero:.7e}\n"
                    f" Diff: {abs_diff:.2e}\n"
                    f" Rel:  {rel_diff:.2e}"
                )
            else:
                prefix = f"{label} " if label else ""
                final_label = f"{prefix}Scale {scale:.1e}"

            ax.plot(
                steps,
                y_values,
                linestyle="--",
                marker=marker_char,
                linewidth=2,
                color=color,
                label=final_label,
                alpha=0.8,
            )

    # Post-plot decorations
    if is_nll:
        assert reference_mins is not None, "reference_mins required for NLL plots"
        # Plot global minimum lines
        global_min = min(reference_mins.values())
        ax.axhline(
            global_min,
            color="gray",
            linestyle="--",
            alpha=0.6,
            label=f"Global Min: {global_min:.7e}",
        )

    ax.axvline(0, color="red", linestyle="--", alpha=0.5)

    if use_legend:
        if is_nll:
            # Legend outside plot on the right for NLL plots
            ax.legend(
                fontsize="small",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                frameon=True,
            )
        else:
            ax.legend(fontsize="small", loc="best")


def _plot_nll(
    validation_results: Union[dict[str, Any], list[dict[str, Any]]],
    file_name: str,
    title: str,
    labels: list[str] | None = None,
    reference_min: dict[float, float] | None = None,
    subfolder: str | None = None,
    output_format: str = "png",
    output_dir: str = "plots",
) -> None:
    """Plot NLL only."""
    if isinstance(validation_results, dict):
        validation_results = [validation_results]
        labels = labels or [""]

    fig, ax = plt.subplots(figsize=(10, 4))

    _plot_lines_on_ax(
        ax,
        validation_results,
        labels,
        reference_mins=reference_min,
        metric_key="NLL",
        is_nll=True,
    )

    ax.set_xlabel("Perturbation Steps (x Scale)")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave room on right for legend

    base_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, output_format, output_dir=output_dir, subfolder=subfolder)
    success(f"NLL plot saved to {file_name}.{output_format}")


def _plot_grad_norms(
    validation_results: Union[dict[str, Any], list[dict[str, Any]]],
    file_name: str,
    title: str,
    labels: list[str] | None = None,
    subfolder: str | None = None,
    output_format: str = "png",
    output_dir: str = "plots",
) -> None:
    """Plot gradient norms (1x3)."""
    if isinstance(validation_results, dict):
        validation_results = [validation_results]
        labels = labels or [""]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Gradient Norms {title}", fontsize=16)

    param_configs = [
        ("grads_beta_dust", r"$\beta_d$"),
        ("grads_beta_pl", r"$\beta_s$"),
        ("grads_temp_dust", r"$T_d$"),
    ]

    for ax, (key, param_title) in zip(axes, param_configs):
        _plot_lines_on_ax(
            ax,
            validation_results,
            labels,
            metric_key=key,
            is_nll=False,
            use_legend=(ax == axes[0]),  # Only legend on first plot
        )
        ax.set_xlabel("Perturbation Steps (x Scale)")
        ax.set_ylabel("L2 Norm")
        ax.set_title(f"Gradient: {param_title}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, output_format, output_dir=output_dir, subfolder=subfolder)
    success(f"Gradient norms plot saved to {file_name}.{output_format}")


def _plot_grad_maps(
    validation_results: dict[str, Any],
    step_idx: int,
    patches: dict[str, Array],
    mask_indices: Array,
    nside: int,
    file_name: str,
    title: str,
    subfolder: str | None = None,
    output_format: str = "png",
    output_dir: str = "plots",
) -> None:
    """Plot gradient healpix maps at a specific step index."""
    scales = validation_results["scales"]
    steps = validation_results["steps"]
    results = validation_results["results"]

    # Convert step_idx (can be negative like -3) to array index
    step_array_idx = int(jnp.argmin(jnp.abs(steps - step_idx)))
    # actual_step = steps[step_array_idx]

    # Use first scale
    scale = scales[0]
    grads_raw = results[scale]["grads_raw"]

    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(f"Gradient Maps {title} at solution", fontsize=14)

    param_configs = [
        ("beta_dust", "beta_dust_patches", r"Grad: $\beta_d$"),
        ("beta_pl", "beta_pl_patches", r"Grad: $\beta_s$"),
        ("temp_dust", "temp_dust_patches", r"Grad: $T_d$"),
    ]
    saved_maps = {}

    for i, (param_key, patch_key, param_title) in enumerate(param_configs):
        grad_values = grads_raw[param_key][step_array_idx]
        patch_idx = patches[patch_key]
        grad_at_pixels = grad_values[patch_idx]
        full_map = get_fullmap_from_cutout(grad_at_pixels, mask_indices, nside)
        saved_maps[param_key] = full_map
        hp.mollview(
            full_map,
            title=param_title,
            sub=(1, 3, i + 1),
            cmap="RdBu_r",
            bgcolor=(0.0,) * 4,
        )

    plt.tight_layout()

    base_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, output_format, output_dir=output_dir, subfolder=subfolder)
    success(f"Gradient maps saved to {file_name}.{output_format}")

    npz_path = os.path.join(base_dir, f"{file_name}.npz")
    np.savez(npz_path, **saved_maps)
    success(f"Gradient maps data saved to {npz_path}")


def _plot_contour(
    val_res: dict[str, Any],
    file_name: str,
    title: str,
    subfolder: str | None = None,
    output_format: str = "png",
    output_dir: str = "plots",
) -> None:
    """Plot 2D NLL contour."""
    NLL = val_res["NLL_grid"]
    steps = val_res["steps"]
    p1, p2 = val_res["active_params"]
    s1, s2 = val_res["scales"][p1], val_res["scales"][p2]
    base_vals = val_res.get("base_values", {})

    latex_labels = {
        "beta_dust": r"$\beta_d$",
        "temp_dust": r"$T_d$",
        "beta_pl": r"$\beta_s$",
    }
    label_p1 = latex_labels.get(p1, p1)
    label_p2 = latex_labels.get(p2, p2)

    # Determine X/Y arrays and labels
    if p1 in base_vals:
        base1 = base_vals[p1]
        X_vals = base1 + steps * s1
        x_label = f"{label_p1} (Value)"
    else:
        base1 = 0.0
        X_vals = steps * s1
        x_label = f"{label_p1} perturbation"

    if p2 in base_vals:
        base2 = base_vals[p2]
        Y_vals = base2 + steps * s2
        y_label = f"{label_p2} (Value)"
    else:
        base2 = 0.0
        Y_vals = steps * s2
        y_label = f"{label_p2} perturbation"

    fig, ax = plt.subplots(figsize=(8, 7))
    # levels = np.logspace(np.log10(NLL.min()), np.log10(NLL.max()), 20)
    # Use simple levels for now
    cp = ax.contourf(X_vals, Y_vals, NLL.T, levels=20, cmap="viridis")
    fig.colorbar(cp, ax=ax, label="Negative log likelihood")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Mark minimum
    min_idx_raveled = np.argmin(NLL)
    min1_idx, min2_idx = np.unravel_index(min_idx_raveled, NLL.shape)
    # min_idx is (row_idx, col_idx) = (p1_idx, p2_idx)
    min_p1 = X_vals[min1_idx]
    min_p2 = Y_vals[min2_idx]
    info(f"minimum is {NLL.min()} at indices {min_p1}, {min_p2} -> values {min_p1}, {min_p2}")

    dist = np.sqrt((min_p1 - base1) ** 2 + (min_p2 - base2) ** 2)

    ax.plot(min_p1, min_p2, "r*", markersize=12, label=f"Min NLL\nDist to min: {dist:.4f}")
    ax.plot(base1, base2, "wo", markersize=6, label="Solution")

    ax.set_title(title)

    ax.legend(frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    base_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, output_format, output_dir=output_dir, subfolder=subfolder)
    success(f"Contour plot saved to {file_name}.{output_format}")


def _plot_nll_grad(
    validation_results: Union[dict[str, Any], list[dict[str, Any]]],
    file_name: str,
    title: str,
    labels: list[str] | None = None,
    reference_mins: dict[float, float] | None = None,
    subfolder: str | None = None,
    output_format: str = "png",
    output_dir: str = "plots",
) -> None:
    """
    Generate a 2x2 plot of NLL and Gradient Norms across scales.
    """
    if isinstance(validation_results, dict):
        validation_results = [validation_results]
        labels = labels or [""]

    # Setup Plotting Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(f"Optimization Verification {title}", fontsize=16)

    # Plot 1: Negative Log Likelihood
    _plot_lines_on_ax(
        axes[0, 0],
        validation_results,
        labels,
        reference_mins=reference_mins,
        metric_key="NLL",
        is_nll=True,
        use_legend=True,
    )

    # Plot 2: Gradient Norm - Beta Dust
    _plot_lines_on_ax(
        axes[0, 1],
        validation_results,
        labels,
        metric_key="grads_beta_dust",
        is_nll=False,
        use_legend=False,
    )

    # Plot 3: Gradient Norm - Beta PL
    _plot_lines_on_ax(
        axes[1, 0],
        validation_results,
        labels,
        metric_key="grads_beta_pl",
        is_nll=False,
        use_legend=False,
    )

    # Plot 4: Gradient Norm - Temp Dust
    _plot_lines_on_ax(
        axes[1, 1],
        validation_results,
        labels,
        metric_key="grads_temp_dust",
        is_nll=False,
        use_legend=False,
    )

    # Common Formatting
    plot_configs = [
        (axes[0, 0], "Negative Log-Likelihood", "NLL"),
        (axes[0, 1], r"Gradient Norm: $\beta_d$", "L2 Norm"),
        (axes[1, 0], r"Gradient Norm: $\beta_s$", "L2 Norm"),
        (axes[1, 1], r"Gradient Norm: $T_d$", "L2 Norm"),
    ]

    for ax, ax_title, ylabel in plot_configs:
        ax.set_title(ax_title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Perturbation Steps (x Scale)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])  # Leave room on right for NLL legend

    # Save
    base_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, output_format, output_dir=output_dir, subfolder=subfolder)
    success(f"Validation plot saved to {file_name}.{output_format}")


def plot_gradient_validation(
    validation_results: Union[dict[str, Any], list[dict[str, Any]]],
    file_name: str,
    title: str,
    plot_type: str = "nll-grad",
    labels: list[str] | None = None,
    patches: dict[str, Array] | None = None,
    mask_indices: Array | None = None,
    reference_min: dict[float, float] | None = None,
    nside: int | None = None,
    subfolder: str | None = None,
    output_format: str = "png",
    output_dir: str = "plots",
) -> None:
    """Dispatches to the appropriate plotting function based on plot_type.

    Args:
        validation_results: Validation results (or list thereof) from
            compute_gradient_validation.
        file_name: Output file name (without extension).
        title: Plot title.
        plot_type: Type of plot to generate. One of: 'nll-grad', 'nll', 'grad',
            'grad-maps-{idx}'. Defaults to "nll-grad".
        labels: Optional labels for each result when aggregating multiple runs.
        patches: Optional patch indices (required for grad-maps).
        mask_indices: Optional mask indices (required for grad-maps).
        nside: Optional HEALPix nside (required for grad-maps).
        subfolder: Optional subfolder for output files.
        output_format: Output format for plots ('png', 'pdf', 'show'). Defaults to "png".
        output_dir: Directory to save plots. Defaults to "plots".

    Example:
        >>> plot_gradient_validation(
        ...     results,
        ...     file_name="validation_plot",
        ...     title="Run 0 Gradient Check",
        ...     plot_type="nll"
        ... )
    """
    if plot_type == "nll-grad":
        _plot_nll_grad(
            validation_results,
            file_name,
            title,
            labels,
            reference_min,
            subfolder,
            output_format,
            output_dir,
        )
    elif plot_type == "nll":
        _plot_nll(
            validation_results,
            file_name,
            title,
            labels,
            reference_min,
            subfolder,
            output_format,
            output_dir,
        )
    elif plot_type == "grad":
        _plot_grad_norms(
            validation_results, file_name, title, labels, subfolder, output_format, output_dir
        )
    elif plot_type.startswith("grad-maps-"):
        step_idx = int(plot_type.split("-")[-1])
        if isinstance(validation_results, list):
            validation_results = validation_results[0]
        if patches is None or mask_indices is None or nside is None:
            error("grad-maps requires patches, mask_indices, and nside")
            return
        _plot_grad_maps(
            validation_results,
            step_idx,
            patches,
            mask_indices,
            nside,
            file_name,
            title,
            subfolder,
            output_format,
            output_dir,
        )
    elif plot_type == "contour":
        _plot_contour(
            validation_results,  # type: ignore (dict expected)
            file_name,
            title,
            subfolder,
            output_format,
            output_dir,
        )
    else:
        error(f"Unknown plot_type: {plot_type}")


def run_validate(
    matched_results: dict[str, tuple[list[str], Union[int, tuple[int, int]], str]],
    names: list[str],
    nside: int,
    instrument: FGBusterInstrument,
    steps: int,
    scales: list[float],
    plot_type: list[str] = ["nll-grad"],
    perturb_beta_dust: str = "all",
    perturb_beta_pl: str = "all",
    perturb_temp_dust: str = "all",
    noise_selection: str = "min-value",
    aggregate: bool = False,
    use_vmap: bool = True,
    output_format: str = "png",
    font_size: int = 14,
    output_dir: Optional[str] = None,
) -> None:
    """Entry point for 'validate' subcommand to run the full validation pipeline.

    Loads results, computes perturbations, and generates validation plots for one or
    multiple runs.

    Args:
        matched_results: Dictionary mapping names to folder lists and metadata.
        names: List of group names to validate.
        nside: HEALPix nside resolution.
        instrument: FGBusterInstrument object.
        steps: Number of steps to perturb in each direction.
        scales: List of scales for perturbation.
        plot_type: List of plot types to generate ('nll-grad', 'nll', etc.).
            Defaults to ["nll-grad"].
        perturb_beta_dust: Spec for beta_dust perturbation. Defaults to "all".
        perturb_beta_pl: Spec for beta_pl perturbation. Defaults to "all".
        perturb_temp_dust: Spec for temp_dust perturbation. Defaults to "all".
        noise_selection: Strategy to select noise realization: 'min-value', 'min-nll', 'idx'.
        aggregate: If True, combines all runs onto a single plot. Defaults to False.
        use_vmap: Whether to use JAX vmap for vectorized execution. Defaults to True.
        output_format: Output format for plots ('png', 'pdf', 'show'). Defaults to "png".
        font_size: Font size for plots. Defaults to 14.
        output_dir: Output directory for plots. Defaults to None (uses PLOT_OUTPUTS default).
    """
    if not output_dir:
        output_dir = "plots"

    set_font_size(font_size)

    # Standard parameter order for scales logic
    std_param_order = ["beta_dust", "beta_pl", "temp_dust"]
    perturb_specs = {
        "beta_dust": perturb_beta_dust,
        "beta_pl": perturb_beta_pl,
        "temp_dust": perturb_temp_dust,
    }

    # Determine mode: contour vs 1d
    if "contour" in plot_type:
        if len(plot_type) > 1:
            error("Plot type 'contour' cannot be combined with other plot types.")
            return
        mode = "contour"

        # Setup for Contour
        active_params = []
        for p in std_param_order:
            if perturb_specs[p] != "-1":
                active_params.append(p)

        if len(active_params) != 2:
            error(
                f"Contour plots require exactly 2 parameters to be perturbed. Found {len(active_params)}: {active_params}. "
                "Use -1 to disable parameters."
            )
            return

        # Parse scales for 2D contour
        if len(scales) == 1:
            contour_scales = {p: scales[0] for p in active_params}
        elif len(scales) == 2:
            contour_scales = {p: s for p, s in zip(active_params, scales)}
        elif len(scales) == 3:
            scale_map = {n: s for n, s in zip(std_param_order, scales)}
            contour_scales = {p: scale_map[p] for p in active_params}
        else:
            error(
                f"Invalid number of scales ({len(scales)}) for contour plot. Expected 1, 2, or 3."
            )
            return

    else:
        mode = "1d"
        contour_scales = None
        active_params = None

    # Global aggregation collectors (only used for 1D)
    all_val_res = []
    all_labels = []
    last_patches = None
    last_indices = None
    last_nside = nside
    reference_min = {scale: np.inf for scale in scales}

    for name, (kw, matched_folders) in zip(names, matched_results.items()):
        folders, run_indices, root_dir = matched_folders

        plot_subfolder = kw
        if root_dir:
            plot_subfolder = os.path.join(root_dir, kw)

        # Normalize run_indices
        if isinstance(run_indices, int):
            run_indices_list = [run_indices]
        elif isinstance(run_indices, tuple) and len(run_indices) == 2:
            run_indices_list = list(range(run_indices[0], run_indices[1] + 1))
        else:
            run_indices_list = list(run_indices)  # type: ignore

        for folder in folders:
            results_path = f"{folder}/results.npz"
            mask_path = f"{folder}/mask.npy"

            try:
                full_results = dict(np.load(results_path))
                mask_arr = np.load(mask_path)
                (indices,) = np.where(mask_arr)
            except (FileNotFoundError, OSError) as e:
                error(f"Failed to load data for {folder}: {e}")
                continue

            for run_idx in run_indices_list:
                info(f"Validating run index {run_idx} in folder '{folder}'")

                # 1. Prepare Data
                run_data_sliced = index_run_data(full_results, run_idx)

                # Determine which noise realization to pick
                if noise_selection == "min-nll":
                    nll = run_data_sliced["NLL"]
                    indx = np.argmin(nll)
                    info(f"Selected noise realization {indx} (min NLL: {nll[indx]:.4e})")
                elif noise_selection == "min-value":
                    var = run_data_sliced["value"]
                    indx = np.argmin(var)
                    info(f"Selected noise realization {indx} (min variance: {var[indx]:.4e})")
                else:
                    # Fallback to int if string parses (for convenience)
                    try:
                        indx = int(noise_selection)
                        if indx >= run_data_sliced["value"].shape[0]:
                            error(f"Noise index {indx} out of bounds")
                            continue
                    except ValueError:
                        error(
                            f"Unknown noise selection strategy: {noise_selection}. Use 'min-value', 'min-nll', 'idx' or an integer."
                        )
                        continue

                # Extract Noise
                noised_d = Stokes.from_stokes(
                    Q=run_data_sliced["NOISED_D"][indx][0], U=run_data_sliced["NOISED_D"][indx][1]
                )
                small_n = Stokes.from_stokes(
                    Q=run_data_sliced["small_n"][indx][0], U=run_data_sliced["small_n"][indx][1]
                )

                patches = {
                    "beta_dust_patches": run_data_sliced["beta_dust_patches"],
                    "beta_pl_patches": run_data_sliced["beta_pl_patches"],
                    "temp_dust_patches": run_data_sliced["temp_dust_patches"],
                }

                final_params = {
                    "beta_dust": run_data_sliced["beta_dust"][indx],
                    "beta_pl": run_data_sliced["beta_pl"][indx],
                    "temp_dust": run_data_sliced["temp_dust"][indx],
                }
                debug(
                    f"argmax of temp_dust: {final_params['temp_dust'].argmax()} and it is {final_params['temp_dust'].max()}"
                )

                base_name = os.path.basename(folder.rstrip("/"))

                if mode == "contour":
                    assert contour_scales is not None, "contour_scales required for contour plot"
                    assert active_params is not None, "active_params required for contour plot"
                    val_res_2d = compute_2d_validation(
                        tem_name=name,
                        final_params=final_params,
                        patch_indices=patches,
                        mask_indices=indices,
                        instrument=instrument,
                        nside=nside,
                        param_scales=contour_scales,
                        steps_range=steps,
                        active_params=active_params,
                        perturb_specs=perturb_specs,
                        noised_d=noised_d,
                        small_n=small_n,
                        use_vmap=use_vmap,
                    )
                    file_name = f"{base_name}_seed_{run_idx}_contour"
                    plot_gradient_validation(
                        val_res_2d,
                        file_name=file_name,
                        title=name,
                        plot_type="contour",
                        subfolder=plot_subfolder,
                        output_format=output_format,
                        output_dir=output_dir,
                    )

                else:  # 1D
                    val_res = compute_gradient_validation(
                        final_params=final_params,
                        patch_indices=patches,
                        mask_indices=indices,
                        instrument=instrument,
                        nside=nside,
                        scales=scales,
                        steps_range=steps,
                        noised_d=noised_d,
                        small_n=small_n,
                        perturb_beta_dust=perturb_beta_dust,
                        perturb_beta_pl=perturb_beta_pl,
                        perturb_temp_dust=perturb_temp_dust,
                        use_vmap=use_vmap,
                    )

                    for scale in scales:
                        nlls = val_res["results"][scale]["NLL"]
                        min_nll = np.min(nlls)
                        if min_nll < reference_min[scale]:
                            reference_min[scale] = min_nll

                    if aggregate:
                        all_val_res.append(val_res)
                        all_labels.append(name)
                        last_patches = patches
                        last_indices = indices
                    else:
                        for pt in plot_type:
                            file_name = f"{base_name}_seed_{run_idx}_{pt}"
                            plot_gradient_validation(
                                val_res,
                                file_name=file_name,
                                title=name,
                                plot_type=pt,
                                patches=patches,
                                mask_indices=indices,
                                nside=nside,
                                reference_min=reference_min,
                                subfolder=plot_subfolder,
                                output_format=output_format,
                                output_dir=output_dir,
                            )

    # Aggregated plot (1D only)
    if mode == "1d" and aggregate and all_val_res:
        combined_title = ", ".join(names)
        for pt in plot_type:
            if pt.startswith("grad-maps-"):
                error(f"Plot type '{pt}' is not compatible with --aggregate, skipping")
                continue
            file_name = f"all_aggregated_{pt}"
            plot_gradient_validation(
                all_val_res,
                file_name=file_name,
                title=combined_title,
                plot_type=pt,
                labels=all_labels,
                patches=last_patches,
                mask_indices=last_indices,
                nside=last_nside,
                reference_min=reference_min,
                subfolder=None,
                output_format=output_format,
                output_dir=output_dir,
            )
