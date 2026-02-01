import os
import re
from typing import Any

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 # pyright: ignore[reportUnusedImport]
from furax._instruments.sky import FGBusterInstrument
from furax.obs.stokes import Stokes
from jaxtyping import Array, Float, Int
from matplotlib.colors import Normalize
from tqdm.auto import tqdm

from ..logging_utils import error, hint, info, success, warning
from .compute import compute_all
from .snapshot import load_and_filter_snapshot, serialize_snapshot_payload

plt.style.use("science")

# Shared color palette for consistent run colors across all plots
# These are the default matplotlib tab10 colors
RUN_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

font_size = 18
plt.rcParams.update(
    {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
    }
)


def get_run_color(index: int) -> str:
    """Get color for run by index, cycling through RUN_COLORS.

    Parameters
    ----------
    index : int
        Zero-based index of the run

    Returns
    -------
    str
        Hex color string
    """
    return RUN_COLORS[index % len(RUN_COLORS)]


def get_symmetric_percentile_limits(
    data_list: list[Float[Array, " n"]], percentile: float = 99
) -> tuple[float, float]:
    """Compute symmetric vmin/vmax from percentile across multiple maps.

    Parameters
    ----------
    data_list : list of arrays
        List of map arrays (UNSEEN values will be filtered out)
    percentile : float
        Percentile to use (e.g., 99 means 1st and 99th percentiles)

    Returns
    -------
    vmin, vmax : float
        Symmetric limits centered at 0
    """
    all_values = []
    for data in data_list:
        valid = data[~np.isnan(data) & (data != hp.UNSEEN)]
        all_values.append(valid)
    combined = np.concatenate(all_values)

    low = np.percentile(combined, 100 - percentile)
    high = np.percentile(combined, percentile)

    # Make symmetric around 0
    abs_max = max(abs(low), abs(high))
    return -abs_max, abs_max


def _truncate_name_if_too_long(name: str, max_length: int = 250) -> str:
    """Truncate long names for plot titles and filenames."""
    if len(name) > max_length:
        return name[: max_length - 3] + "..."
    return name


def set_font_size(size: int) -> None:
    """Set global font size for all plotting functions.

    Parameters
    ----------
    size : int
        Font size to use for all plot elements.
    """
    return
    font_size = size
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.titlesize": font_size,
        }
    )


def save_or_show(
    filename: str, output_format: str, output_dir: str = "plots", subfolder: str | None = None
) -> None:
    """Save figure to file or show inline based on output format.

    Parameters
    ----------
    filename : str
        Base filename (without extension).
    output_format : str
        'png', 'pdf', or 'show'.
    output_dir : str
        Directory to save plots to. Defaults to "plots".
    subfolder : str, optional
        Subdirectory under output_dir for grouped results.
    """
    if output_format == "show":
        plt.show()
    else:
        ext = "pdf" if output_format == "pdf" else "png"
        dpi = 300 if ext == "png" else None
        filename = _truncate_name_if_too_long(filename)

        base_dir = output_dir
        if subfolder:
            base_dir = os.path.join(output_dir, subfolder)

        os.makedirs(base_dir, exist_ok=True)

        filepath = os.path.join(base_dir, f"{filename}.{ext}")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close()
        success(f"Saved: {filepath}")


def get_min_variance(cmb_map: Stokes) -> Stokes:
    """Select the realization with minimum variance across Q/U components."""
    seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
    cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
    variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
    variance = sum(jax.tree.leaves(variance))
    argmin = jnp.argmin(variance)
    return jax.tree.map(lambda x: x[argmin], cmb_map)


def plot_params(
    name: str,
    params: dict[str, Float[Array, " npix"]],
    output_format: str,
    output_dir: str = "plots",
    plot_vertical: bool = False,
    subfolder: str | None = None,
) -> None:
    """Plot recovered spectral parameter maps for a single configuration."""
    if plot_vertical:
        fig_size = (8, 16)
        subplot_args = (3, 1, lambda i: i + 1)
    else:
        fig_size = (16, 8)
        subplot_args = (1, 3, lambda i: i + 1)

    _ = plt.figure(figsize=fig_size)

    keys = ["beta_dust", "temp_dust", "beta_pl"]
    names = [r"$\beta_d$", r"$T_d$", r"$\beta_s$"]

    for i, (key, param_name) in enumerate(zip(keys, names)):
        param_map = params[key]
        hp.mollview(
            param_map,
            title=f"{name} {param_name}",
            sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
            bgcolor=(0.0,) * 4,
            cbar=True,
        )

    save_or_show(f"params_{name}", output_format, output_dir=output_dir, subfolder=subfolder)
    base_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(base_dir, exist_ok=True)
    params_dict = {
        "beta_dust": params["beta_dust"],
        "temp_dust": params["temp_dust"],
        "beta_pl": params["beta_pl"],
    }
    np.savez(os.path.join(base_dir, f"params_{name}.npz"), **params_dict)


def plot_patches(
    name: str,
    patches: dict[str, Int[Array, " npix"]],
    output_format: str,
    output_dir: str = "plots",
    plot_vertical: bool = False,
    subfolder: str | None = None,
) -> None:
    """Visualise patch assignments (cluster labels) for each spectral parameter."""
    if plot_vertical:
        fig_size = (8, 16)
        subplot_args = (3, 1, lambda i: i + 1)
    else:
        fig_size = (16, 8)
        subplot_args = (1, 3, lambda i: i + 1)

    _ = plt.figure(figsize=fig_size)

    np.random.seed(0)

    def shuffle_labels(arr):
        unique_vals = np.unique(arr[arr != hp.UNSEEN])
        shuffled_vals = np.random.permutation(unique_vals)

        mapping = dict(zip(unique_vals, shuffled_vals))

        shuffled_arr = np.vectorize(lambda x: mapping.get(x, hp.UNSEEN))(arr)
        return shuffled_arr.astype(np.float64)

    base_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(base_dir, exist_ok=True)
    patches_dict = {
        "beta_dust_patches": patches["beta_dust_patches"],
        "temp_dust_patches": patches["temp_dust_patches"],
        "beta_pl_patches": patches["beta_pl_patches"],
    }
    np.savez(os.path.join(base_dir, f"patches_{name}.npz"), **patches_dict)
    patches_shuffled = jax.tree.map(shuffle_labels, patches)

    keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
    names = [r"$\beta_d$ Patches", r"$T_d$ Patches", r"$\beta_s$ Patches"]

    for i, (key, patch_name) in enumerate(zip(keys, names)):
        patch_map = patches_shuffled[key]
        hp.mollview(
            patch_map,
            title=f"{name} {patch_name}",
            sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
            bgcolor=(0.0,) * 4,
            cbar=True,
        )
    save_or_show(f"patches_{name}", output_format, output_dir=output_dir, subfolder=subfolder)


def get_masked_residual(true_map, model_map):
    return np.where(true_map == hp.UNSEEN, hp.UNSEEN, true_map - model_map)


def plot_all_params_residuals(
    names: list[str],
    params_map_list: list[dict[str, Float[Array, " npix"]]],
    true_params: dict[str, Float[Array, " npix"]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Generate the maps and residuals plot for all parameters across all runs."""

    param_configs = [
        {"key": "beta_dust", "label": r"$\beta_{dust}$"},
        {"key": "temp_dust", "label": r"$T_{dust}$"},
        {"key": "beta_pl", "label": r"$\beta_{s}$"},
    ]

    nb_runs = len(names)

    for config in param_configs:
        key = config["key"]
        label = config["label"]

        if key not in true_params:
            warning(f"Missing data for {key} in true params. Skipping.")
            continue

        fig = plt.figure(figsize=(12, 4 + 4 * nb_runs))
        gs = plt.GridSpec(nb_runs + 1, 2, hspace=0.3, wspace=0.1, height_ratios=[1] + [1] * nb_runs)

        # 1. TRUTH (Row 0)
        ax_true = fig.add_subplot(gs[0, :])
        plt.sca(ax_true)
        hp.mollview(true_params[key], title=f"True Parameters {label}", hold=True, bgcolor=(0,) * 4)

        for i, (name, params_map) in enumerate(zip(names, params_map_list)):
            if params_map is None or key not in params_map:
                continue

            # Calculate residual
            res = get_masked_residual(true_params[key], params_map[key])

            # Plot Parameter Map (Left Column)
            ax_map = fig.add_subplot(gs[i + 1, 0])
            plt.sca(ax_map)
            hp.mollview(
                params_map[key],
                title=f"{name} {label}",
                hold=True,
                bgcolor=(0,) * 4,
            )

            # Plot Residual (Right Column)
            ax_res = fig.add_subplot(gs[i + 1, 1])
            plt.sca(ax_res)
            hp.mollview(
                res,
                title=f"Residual (True - {name})",
                cmap="RdBu_r",
                hold=True,
                bgcolor=(0,) * 4,
            )

        name_str = "_".join(names)
        save_or_show(
            f"minimize_maps_residuals_{key}_{name_str}",
            output_format,
            output_dir=output_dir,
        )


def plot_all_histograms(
    names: list[str],
    all_params_list: list[list[dict[str, Float[Array, " npix"]]]],
    true_params: dict[str, Float[Array, " npix"]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Generate histograms of parameters comparing Truth vs Recovered across runs."""

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plot_configs = [
        {
            "key": "beta_dust",
            "label": r"$\beta_{dust}$",
            "title": "Dust Spectral Index",
            "ylim": (0, 10),
        },
        {
            "key": "temp_dust",
            "label": r"$T_{dust}$ [K]",
            "title": "Dust Temperature",
            "ylim": (0, 1),
        },
        {
            "key": "beta_pl",
            "label": r"$\beta_{s}$",
            "title": "Synchrotron Spectral Index",
            "ylim": (0, 10),
        },
    ]

    for ax, config in zip(axs, plot_configs):
        key = config["key"]

        # Plot Truth (assuming shared truth)
        if key in true_params:
            # Mask out UNSEEN
            true_vals = true_params[key]
            valid_true = true_vals[true_vals != hp.UNSEEN]

            if valid_true.size > 0:
                ax.hist(
                    valid_true,
                    bins=25,
                    histtype="step",
                    linewidth=2,
                    label="Truth",
                    density=True,
                    color="black",
                    linestyle="--",
                )

        # Plot Recovered (all realizations)
        for i, (name, all_params) in enumerate(zip(names, all_params_list)):
            if all_params is None:
                continue

            recovered_vals = []
            for p in all_params:
                if key in p:
                    val = p[key]
                    recovered_vals.append(val)

            if recovered_vals:
                all_recovered = np.concatenate(recovered_vals)
                color = get_run_color(i)

                idx_max = np.argmax(all_recovered)
                idx_max_unraveled = np.unravel_index(idx_max, all_recovered.shape)

                to_bin = all_recovered[:, idx_max_unraveled[1]]

                ax.hist(
                    to_bin,
                    bins=25,
                    histtype="step",
                    linewidth=2,
                    label=name,
                    density=True,
                    color=color,
                )

        ax.set_xlabel(config["label"])
        ax.set_ylabel("Density")
        ax.set_title(config["title"])
        ax.set_ylim(config["ylim"])

        ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    name_str = "_".join(names)
    save_or_show(
        f"minimize_histograms_{name_str}",
        output_format,
        output_dir=output_dir,
    )


def plot_all_cmb(
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Show reconstructed-minus-true Q/U differences for multiple runs."""
    nb_cmb = len(cmb_pytree_list)

    # Collect all Q and U maps for shared color limits
    all_maps = []
    diff_all = []

    for cmb_pytree in cmb_pytree_list:
        cmb_recon = get_min_variance(cmb_pytree["cmb_recon"])

        diff_q = cmb_pytree["cmb"].q - cmb_recon.q
        diff_u = cmb_pytree["cmb"].u - cmb_recon.u

        unseen_mask_q = cmb_pytree["cmb"].q == hp.UNSEEN
        unseen_mask_u = cmb_pytree["cmb"].u == hp.UNSEEN

        diff_q = np.where(unseen_mask_q, np.nan, diff_q)
        diff_u = np.where(unseen_mask_u, np.nan, diff_u)

        all_maps.extend([diff_q, diff_u])
        diff_all.append((diff_q, diff_u))

    vmin, vmax = get_symmetric_percentile_limits(all_maps)

    plt.figure(figsize=(10, 3.5 * nb_cmb))

    for i, (name, (diff_q, diff_u)) in enumerate(zip(names, diff_all)):
        hp.mollview(
            diff_q,
            title=rf"Difference (Q) - {name} ($\mu$K)",
            sub=(nb_cmb, 2, 2 * i + 1),
            cbar=True,
            min=vmin,
            max=vmax,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            notext=True,
        )
        hp.mollview(
            diff_u,
            title=rf"Difference (U) - {name} ($\mu$K)",
            sub=(nb_cmb, 2, 2 * i + 2),
            cbar=True,
            min=vmin,
            max=vmax,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            notext=True,
        )

    name_str = "_".join(names)
    save_or_show(f"cmb_recon_{name_str}", output_format, output_dir=output_dir)


def plot_all_variances(
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Histogram proxy metrics (variance, NLL, ∑Cℓ) across runs."""

    def get_all_variances(cmb_map: Stokes) -> Array:
        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance_sum = sum(jax.tree.leaves(variance))
        return variance_sum

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=False)

    metrics = {
        "Variance of Reconstructed CMB (Q + U)": [],
        "Negative Log-Likelihood": [],
        r"$\sum C_\ell^{BB}$": [],
    }

    for name, cmb_pytree in zip(names, cmb_pytree_list):
        metrics["Variance of Reconstructed CMB (Q + U)"].append(
            (name, get_all_variances(cmb_pytree["cmb_recon"]))
        )
        metrics["Negative Log-Likelihood"].append((name, np.array(cmb_pytree["nll_summed"])))
        metrics[r"$\sum C_\ell^{BB}$"].append((name, np.array(cmb_pytree["cl_bb_sum"])))

    for ax, (title, entries) in zip(axs, metrics.items()):
        for i, (name, values) in enumerate(entries):
            color = get_run_color(i)
            label = f"{name}"
            ax.hist(
                values,
                bins=20,
                alpha=0.5,
                label=label,
                color=color,
                edgecolor="black",
                histtype="stepfilled",
            )
            mean_val = np.mean(values)
            ax.axvline(
                mean_val,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"Mean of {name}",
            )

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize="small", loc="best")

        for label in ax.get_xticklabels():
            label.set_rotation(30)

    axs[-1].set_xlabel("Metric Value", fontsize=12)

    plt.tight_layout(pad=2.0)
    save_or_show("metric_distributions_histogram_all_metrics", output_format, output_dir=output_dir)


def plot_all_cl_residuals(
    names: list[str],
    cl_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Overlay residual BB spectra for all requested configurations."""
    _ = plt.figure(figsize=(10, 8))

    if len(cl_pytree_list) == 0:
        warning("No power spectra results to plot")
        return

    cl_bb_r1 = cl_pytree_list[0]["cl_bb_r1"]
    ell_range = cl_pytree_list[0]["ell_range"]
    cl_bb_lens = cl_pytree_list[0]["cl_bb_lens"]

    r_lo, r_hi = 1e-3, 4e-3
    plt.fill_between(
        ell_range,
        r_lo * cl_bb_r1,
        r_hi * cl_bb_r1,
        color="grey",
        alpha=0.35,
        label=r"$C_\ell^{BB},\; r\in[10^{-3},\,4\cdot10^{-3}]$",
    )

    plt.plot(
        ell_range,
        cl_bb_lens,
        label=r"$C_\ell^{BB}\,\mathrm{lens}$",
        color="grey",
        linestyle="-",
        linewidth=2,
    )

    for i, (name, cl_pytree) in enumerate(zip(names, cl_pytree_list)):
        color = get_run_color(i)
        linewidth = 1.5

        if cl_pytree["cl_total_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_total_res"],
                label=rf"{name} $C_\ell^{{\mathrm{{res}}}}$",
                color=color,
                linestyle="--",
            )
        if cl_pytree["cl_syst_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_syst_res"],
                label=rf"{name} $C_\ell^{{\mathrm{{syst}}}}$",
                color=color,
                linestyle="-",
                linewidth=linewidth,
            )
        if cl_pytree["cl_stat_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_stat_res"],
                label=rf"{name} $C_\ell^{{\mathrm{{stat}}}}$",
                color=color,
                linestyle=":",
                linewidth=linewidth,
            )

    plt.title(None)
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell^{BB}$ [$\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.legend(loc="best", ncol=2, framealpha=0.95, columnspacing=1.0)

    plt.tight_layout()

    name_str = "_".join(names)
    save_or_show(f"bb_spectra_{name_str}", output_format, output_dir=output_dir)


def plot_all_systematic_residuals(
    names: list[str],
    syst_map_list: list[Float[Array, " 3 npix"]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Plot systematic residual Q/U maps for multiple runs."""
    nb_runs = len(syst_map_list)
    if nb_runs == 0:
        warning("No systematic residual maps available to plot")
        return

    # Collect all Q and U maps for shared color limits
    all_maps = []
    processed_maps = []
    for syst_map in syst_map_list:
        syst_q = np.where(syst_map[1] == hp.UNSEEN, np.nan, syst_map[1])
        syst_u = np.where(syst_map[2] == hp.UNSEEN, np.nan, syst_map[2])
        all_maps.extend([syst_q, syst_u])
        processed_maps.append((syst_q, syst_u))

    vmin, vmax = get_symmetric_percentile_limits(all_maps)

    plt.figure(figsize=(12, 4 * nb_runs))

    for i, (name, (syst_q, syst_u)) in enumerate(zip(names, processed_maps)):
        hp.mollview(
            syst_q,
            title=rf"Systematic Residual (Q) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 1),
            min=vmin,
            max=vmax,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

        hp.mollview(
            syst_u,
            title=rf"Systematic Residual (U) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 2),
            min=vmin,
            max=vmax,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

    name_str = "_".join(names)
    save_or_show(f"all_systematic_residuals_{name_str}", output_format, output_dir=output_dir)


def plot_all_statistical_residuals(
    names: list[str],
    stat_map_list: list[list[Float[Array, " 3 npix"]]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Plot statistical residual Q/U maps for multiple runs."""
    nb_runs = len(stat_map_list)
    if nb_runs == 0:
        warning("No statistical residual maps available to plot")
        return

    # Collect all Q and U maps for shared color limits
    all_maps = []
    processed_maps = []
    for stat_maps in stat_map_list:
        stat_map_first = stat_maps[0]
        stat_q = np.where(stat_map_first[1] == hp.UNSEEN, np.nan, stat_map_first[1])
        stat_u = np.where(stat_map_first[2] == hp.UNSEEN, np.nan, stat_map_first[2])
        all_maps.extend([stat_q, stat_u])
        processed_maps.append((stat_q, stat_u))

    vmin, vmax = get_symmetric_percentile_limits(all_maps)

    plt.figure(figsize=(12, 4 * nb_runs))

    for i, (name, (stat_q, stat_u)) in enumerate(zip(names, processed_maps)):
        hp.mollview(
            stat_q,
            title=rf"Statistical Residual (Q) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 1),
            min=vmin,
            max=vmax,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

        hp.mollview(
            stat_u,
            title=rf"Statistical Residual (U) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 2),
            min=vmin,
            max=vmax,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

    name_str = "_".join(names)
    save_or_show(f"all_statistical_residuals_{name_str}", output_format, output_dir=output_dir)


def plot_all_r_estimation(
    names: list[str],
    r_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Compare r likelihood curves across runs in a single figure."""
    plt.figure(figsize=(6, 5))

    for i, (name, r_data) in enumerate(zip(names, r_pytree_list)):
        if r_data["r_best"] is None:
            warning(f"No r estimation for {name}, skipping plot.")
            continue

        r_grid = r_data["r_grid"]
        L_vals = r_data["L_vals"]
        r_best = r_data["r_best"]
        sigma_r_neg = r_data["sigma_r_neg"]
        sigma_r_pos = r_data["sigma_r_pos"]

        color = get_run_color(i)
        likelihood = L_vals / L_vals.max()

        plt.plot(
            r_grid,
            likelihood,
            label=rf"{name} $\hat{{r}} = {r_best:.2e}^{{+{sigma_r_pos:.1e}}}_{{-{sigma_r_neg:.1e}}}$",
            color=color,
        )

        plt.fill_between(
            r_grid,
            0,
            likelihood,
            where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
            color=color,
            alpha=0.2,
        )

        plt.axvline(
            x=r_best,
            color=color,
            linestyle="--",
            alpha=0.7,
        )

    plt.axvline(x=0.0, color="black", linestyle="--", alpha=0.7, label="True r=0")

    plt.title("Estimated $r$ residuals")
    plt.xlabel(r"$r$")
    plt.ylabel("Relative Likelihood")
    plt.grid(True, which="both", ls=":")

    plt.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()

    name_str = "_".join(names)
    save_or_show(f"r_likelihood_{name_str}", output_format, output_dir=output_dir)


def _create_r_vs_clusters_plot(
    patch_name: str,
    patch_key: str,
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    r_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Scatter plot of r + σ(r) vs clusters for one parameter."""
    method_dict = {}
    base_patch_keys = [
        "beta_dust_patches",
        "temp_dust_patches",
        "beta_pl_patches",
    ]

    for name, cmb_pytree, r_data in zip(names, cmb_pytree_list, r_pytree_list):
        if r_data["r_best"] is None:
            warning(f"No r estimation for {name}, skipping plot.")
            continue

        base_name = re.sub(r" \(\d+\)$", "", name)

        patches = cmb_pytree["patches_map"]
        if patch_key == "total":
            n_clusters = 0
            for key in base_patch_keys:
                patch_data = patches[key]
                n_clusters += np.unique(patch_data[patch_data != hp.UNSEEN]).size
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size

        if n_clusters in method_dict:
            existing_r_values = method_dict[n_clusters]["r_best"]
            if r_data["r_best"] > existing_r_values:
                continue

        method_dict[n_clusters] = {
            "name": base_name,
            "r_best": r_data["r_best"],
            "sigma_r_neg": r_data["sigma_r_neg"],
            "sigma_r_pos": r_data["sigma_r_pos"],
        }

    plt.figure(figsize=(8, 6))

    if len(method_dict) == 0:
        warning(f"No valid data points for {patch_key} in r_vs_clusters plot.")
        plt.close()
        return

    sorted_items = sorted(method_dict.items(), key=lambda item: item[0])

    cluster_points = []
    r_plus_sigma_vals = []
    min_point = None

    for idx, (n_clusters, data) in enumerate(sorted_items):
        r_best = data["r_best"]
        sigma_r_pos = data["sigma_r_pos"]
        r_plus_sigma = r_best + sigma_r_pos

        cluster_points.append(n_clusters)
        r_plus_sigma_vals.append(r_plus_sigma)

        if (min_point is None) or (r_plus_sigma < min_point["value"]):
            min_point = {
                "index": idx,
                "clusters": n_clusters,
                "value": r_plus_sigma,
            }

    scatter_all = plt.scatter(
        cluster_points,
        r_plus_sigma_vals,
        color="#1f77b4",
        s=100,
        edgecolors="black",
        linewidths=1,
    )

    if min_point is not None:
        min_idx = min_point["index"]
        min_clusters = cluster_points[min_idx]
        min_value = r_plus_sigma_vals[min_idx]

        scatter_min = plt.scatter(
            [min_clusters],
            [min_value],
            color="red",
            s=120,
            edgecolors="black",
            linewidths=1.5,
            zorder=3,
        )
        min_label = (
            r"Lowest residual $r+\sigma(r)$ at " f"{int(min_clusters)} clusters: {min_value:.2e}"
        )
    else:
        scatter_min = None
        min_label = None

    legend_handles = [scatter_all]
    legend_labels = [r"Residual $r+\sigma(r)$"]
    if scatter_min is not None and min_label is not None:
        legend_handles.append(scatter_min)
        legend_labels.append(min_label)

    plt.xlabel(f"Number of Clusters ({patch_name})")
    plt.ylabel(r"Residual $r + \sigma(r)$")
    plt.title(r"Residual $r + \sigma(r)$ vs. Number of Clusters" + f" ({patch_name})")
    # plt.ylim(-0.001, 0.01)
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.6)

    if legend_handles:
        plt.legend(legend_handles, legend_labels)

    plt.tight_layout()

    filename_suffix = patch_key.replace("_patches", "")
    save_or_show(f"residual_r_vs_clusters_{filename_suffix}", output_format, output_dir=output_dir)
    plt.close()


def _create_variance_vs_clusters_plot(
    patch_name: str,
    patch_key: str,
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Scatter plot of cluster count vs minimum variance for one parameter."""
    method_dict = {}
    base_patch_keys = [
        "beta_dust_patches",
        "temp_dust_patches",
        "beta_pl_patches",
    ]
    other_patch_keys = [k for k in base_patch_keys if k != patch_key]

    for name, cmb_pytree in zip(names, cmb_pytree_list):
        base_name = re.sub(r" \(\d+\)$", "", name)

        patches = cmb_pytree["patches_map"]
        if patch_key == "total":
            n_clusters = 0
            for key in base_patch_keys:
                patch_data = patches[key]
                n_clusters += np.unique(patch_data[patch_data != hp.UNSEEN]).size
            total_clusters = n_clusters
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size
            total_clusters = n_clusters
            for other_key in other_patch_keys:
                other_patch_data = patches[other_key]
                total_clusters += np.unique(other_patch_data[other_patch_data != hp.UNSEEN]).size

        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_pytree["cmb_recon"])
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_pytree["cmb_recon"], seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance_sum = sum(jax.tree.leaves(variance))
        min_variance = float(jnp.min(variance_sum))

        if n_clusters in method_dict:
            existing_variance = method_dict[n_clusters]["variance"]
            if min_variance > existing_variance:
                continue

        method_dict[n_clusters] = {
            "name": base_name,
            "variance": min_variance,
            "total_clusters": total_clusters,
        }

    plt.figure(figsize=(8, 6))

    if len(method_dict) == 0:
        warning(f"No valid data points for {patch_key} in variance_vs_clusters plot.")
        plt.close()
        return

    sorted_items = sorted(method_dict.items(), key=lambda item: item[0])
    total_cluster_values = np.array([data["total_clusters"] for _, data in sorted_items])
    total_min = float(total_cluster_values.min())
    total_max = float(total_cluster_values.max())
    if total_min == total_max:
        total_min -= 0.5
        total_max += 0.5

    cmap = plt.cm.viridis
    norm = Normalize(vmin=total_min, vmax=total_max)

    for (n_clusters, data), total_clusters in zip(sorted_items, total_cluster_values):
        variance = data["variance"]
        color = cmap(norm(total_clusters))

        plt.scatter(
            n_clusters,
            variance,
            color=color,
            s=100,
            edgecolors="black",
            linewidths=1,
        )

    plt.xlabel(f"Number of Clusters ({patch_name})")
    plt.ylabel(r"Minimum Variance (Q + U)")
    plt.title(f"Minimum Variance vs. Number of Clusters ({patch_name})")
    plt.grid(True, linestyle="--", alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Total Number of Clusters")

    plt.tight_layout()

    filename_suffix = patch_key.replace("_patches", "")
    save_or_show(f"variance_vs_clusters_{filename_suffix}", output_format, output_dir=output_dir)
    plt.close()


def plot_variance_vs_clusters(
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Plot minimum recovered-CMB variance vs cluster count for each parameter."""
    patch_configs = [
        (r"$\beta_d$", "beta_dust_patches"),
        (r"$T_d$", "temp_dust_patches"),
        (r"$\beta_s$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    for patch_name, patch_key in patch_configs:
        _create_variance_vs_clusters_plot(
            patch_name, patch_key, names, cmb_pytree_list, output_format, output_dir=output_dir
        )


def _create_variance_vs_r_plot(
    patch_name: str,
    patch_key: str,
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    r_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Helper to plot variance vs best-fit r for a given parameter/totals."""
    points = []
    is_total = patch_key == "total"

    for name, cmb_pytree, r_data in zip(names, cmb_pytree_list, r_pytree_list):
        if r_data["r_best"] is None:
            warning(f"No r estimation for {name}, skipping variance_vs_r point.")
            continue

        patches = cmb_pytree["patches_map"]
        cluster_counts = {}

        if is_total:
            n_clusters = 0
            for key in [
                "beta_dust_patches",
                "temp_dust_patches",
                "beta_pl_patches",
            ]:
                patch_data = patches[key]
                count = np.unique(patch_data[patch_data != hp.UNSEEN]).size
                n_clusters += count
                cluster_counts[key] = count
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size

        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_pytree["cmb_recon"])
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_pytree["cmb_recon"], seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance_sum = sum(jax.tree.leaves(variance))
        min_variance = float(jnp.min(variance_sum))

        points.append(
            (
                min_variance,
                float(r_data["r_best"]),
                int(n_clusters),
                float(r_data["sigma_r_neg"]),
                float(r_data["sigma_r_pos"]),
                cluster_counts,
            )
        )

    if len(points) == 0:
        warning("No valid data points for variance_vs_r plot.")
        return

    points.sort(key=lambda p: p[0])
    variances = [p[0] for p in points]
    r_values = [p[1] for p in points]
    k_values = np.array([p[2] for p in points])
    sigma_r_neg = [p[3] for p in points]
    sigma_r_pos = [p[4] for p in points]
    cluster_breakdowns = [p[5] for p in points]

    plt.figure(figsize=(8, 6))

    cmap = plt.cm.viridis
    norm = Normalize(vmin=k_values.min(), vmax=k_values.max())
    colors = cmap(norm(k_values))

    for i in range(len(variances)):
        plt.errorbar(
            variances[i],
            r_values[i],
            yerr=[[sigma_r_neg[i]], [sigma_r_pos[i]]],
            fmt="o",
            color=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            capsize=3,
            elinewidth=1.5,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    if is_total:
        cbar.set_label("Total Number of Clusters")
        var_indx = 0
        r_plus_sigma = np.array(r_values) + np.array(sigma_r_pos)
        r_indx = np.argmin(r_plus_sigma)

        info(
            f"Variance vs Residual r (Total): Least Variance Run: Variance={variances[var_indx]:.2e}, "
            f"r={r_values[var_indx]:.2e} +/- {sigma_r_pos[var_indx]:.2e}; "
            f"Least r Run: Variance={variances[r_indx]:.2e}, "
            f"r={r_values[r_indx]:.2e} +/- {sigma_r_pos[r_indx]:.2e}"
        )
        bd_v = cluster_breakdowns[var_indx].get("beta_dust_patches", 0)
        td_v = cluster_breakdowns[var_indx].get("temp_dust_patches", 0)
        bp_v = cluster_breakdowns[var_indx].get("beta_pl_patches", 0)
        bd_r = cluster_breakdowns[r_indx].get("beta_dust_patches", 0)
        td_r = cluster_breakdowns[r_indx].get("temp_dust_patches", 0)
        bp_r = cluster_breakdowns[r_indx].get("beta_pl_patches", 0)

        info(f"Least Variance Run Clusters: Beta_dust={bd_v}, Temp_dust={td_v}, Beta_pl={bp_v}")
        info(f"Least r Run Clusters: Beta_dust={bd_r}, Temp_dust={td_r}, Beta_pl={bp_r}")
    else:
        cbar.set_label(f"Number of Clusters ({patch_name})")

    plt.xlabel(r"Minimum Variance (Q + U)")
    plt.ylabel(r"Residual $r$")
    plt.ylim(-0.0005, 0.005)
    plt.title(f"Variance vs Residual $r$ ({patch_name})")
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filename_suffix = "total" if is_total else patch_key.replace("_patches", "")
    save_or_show(f"variance_vs_residual_r_{filename_suffix}", output_format, output_dir=output_dir)
    plt.close()


def plot_variance_vs_r(
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    r_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Plot variance vs best-fit r for each spectral parameter and combined."""
    patch_configs = [
        (r"$\beta_d$", "beta_dust_patches"),
        (r"$T_d$", "temp_dust_patches"),
        (r"$\beta_s$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    for patch_name, patch_key in patch_configs:
        _create_variance_vs_r_plot(
            patch_name,
            patch_key,
            names,
            cmb_pytree_list,
            r_pytree_list,
            output_format,
            output_dir=output_dir,
        )


def plot_r_vs_clusters(
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    r_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    """Plot r + σ(r) vs number of clusters for each parameter."""
    patch_configs = [
        (r"$\beta_d$", "beta_dust_patches"),
        (r"$T_d$", "temp_dust_patches"),
        (r"$\beta_s$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    for patch_name, patch_key in patch_configs:
        _create_r_vs_clusters_plot(
            patch_name,
            patch_key,
            names,
            cmb_pytree_list,
            r_pytree_list,
            output_format,
            output_dir=output_dir,
        )


def plot_systematic_residual_maps(
    name: str,
    syst_map: Float[Array, " 3 npix"],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
) -> None:
    """Plot systematic residual Q/U maps for a single configuration."""
    syst_q = np.where(syst_map[1] == hp.UNSEEN, np.nan, syst_map[1])
    syst_u = np.where(syst_map[2] == hp.UNSEEN, np.nan, syst_map[2])

    vmin, vmax = get_symmetric_percentile_limits([syst_q, syst_u])

    plt.figure(figsize=(12, 6))

    hp.mollview(
        syst_q,
        title=rf"Systematic Residual (Q) - {name} ($\mu$K)",
        sub=(1, 2, 1),
        min=vmin,
        max=vmax,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    hp.mollview(
        syst_u,
        title=rf"Systematic Residual (U) - {name} ($\mu$K)",
        sub=(1, 2, 2),
        min=vmin,
        max=vmax,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    save_or_show(
        f"systematic_residual_maps_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
    )


def plot_statistical_residual_maps(
    name: str,
    stat_maps: list[Float[Array, " 3 npix"]],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
) -> None:
    """Plot statistical residual Q/U maps for a single configuration."""
    stat_map_first = stat_maps[0]

    stat_q = np.where(stat_map_first[1] == hp.UNSEEN, np.nan, stat_map_first[1])
    stat_u = np.where(stat_map_first[2] == hp.UNSEEN, np.nan, stat_map_first[2])

    vmin, vmax = get_symmetric_percentile_limits([stat_q, stat_u])

    plt.figure(figsize=(12, 6))

    hp.mollview(
        stat_q,
        title=rf"Statistical Residual (Q) - {name} ($\mu$K)",
        sub=(1, 2, 1),
        min=vmin,
        max=vmax,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    hp.mollview(
        stat_u,
        title=rf"Statistical Residual (U) - {name} ($\mu$K)",
        sub=(1, 2, 2),
        min=vmin,
        max=vmax,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    save_or_show(
        f"statistical_residual_maps_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
    )


def plot_cmb_reconstructions(
    name: str,
    cmb_stokes: Stokes,
    cmb_recon: Stokes,
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
) -> None:
    """Plot reconstructed maps, inputs, and differences for Q/U."""

    cmb_recon_min = get_min_variance(cmb_recon)
    unseen_mask_q = cmb_recon_min.q == hp.UNSEEN
    diff_q = cmb_recon_min.q - cmb_stokes.q
    diff_q = np.where(unseen_mask_q, np.nan, diff_q)

    unseen_mask_u = cmb_recon_min.u == hp.UNSEEN
    diff_u = cmb_recon_min.u - cmb_stokes.u
    diff_u = np.where(unseen_mask_u, np.nan, diff_u)

    vmin, vmax = get_symmetric_percentile_limits([diff_q, diff_u])

    _ = plt.figure(figsize=(12, 12))
    hp.mollview(
        cmb_recon_min.q,
        title=r"Reconstructed CMB (Q) [$\mu$K]",
        sub=(3, 3, 1),
        bgcolor=(0,) * 4,
    )
    hp.mollview(
        cmb_stokes.q,
        title=r"Input CMB Map (Q) [$\mu$K]",
        sub=(3, 3, 2),
        bgcolor=(0,) * 4,
    )
    hp.mollview(
        diff_q,
        title=r"Difference (Q) [$\mu$K]",
        sub=(3, 3, 3),
        cbar=True,
        min=vmin,
        max=vmax,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
    )
    hp.mollview(
        cmb_recon_min.u,
        title=r"Reconstructed CMB (U) [$\mu$K]",
        sub=(3, 3, 4),
        bgcolor=(0,) * 4,
    )
    hp.mollview(
        cmb_stokes.u,
        title=r"Input CMB Map (U) [$\mu$K]",
        sub=(3, 3, 5),
        bgcolor=(0,) * 4,
    )
    hp.mollview(
        diff_u,
        title=r"Difference (U) [$\mu$K]",
        sub=(3, 3, 6),
        cbar=True,
        min=vmin,
        max=vmax,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
    )
    plt.title(f"{name} CMB Reconstruction")
    save_or_show(f"cmb_recon_{name}", output_format, output_dir=output_dir, subfolder=subfolder)


def plot_cl_residuals(
    name: str,
    cl_bb_obs: Float[Array, " ell"],
    cl_syst_res: Float[Array, " ell"],
    cl_total_res: Float[Array, " ell"],
    cl_stat_res: Float[Array, " ell"],
    cl_bb_r1: Float[Array, " ell"],
    cl_bb_lens: Float[Array, " ell"],
    cl_true: Float[Array, " ell"],
    ell_range: Float[Array, " ell"],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
) -> None:
    """Plot detailed BB spectrum decomposition for a single configuration."""
    _ = plt.figure(figsize=(10, 8))

    r_lo, r_hi = 1e-3, 4e-3
    plt.fill_between(
        ell_range,
        r_lo * cl_bb_r1,
        r_hi * cl_bb_r1,
        color="grey",
        alpha=0.35,
        label=r"$C_\ell^{BB},\; r\in[10^{-3},\,4\cdot10^{-3}]$",
    )

    plt.plot(
        ell_range,
        cl_bb_lens,
        label=r"$C_\ell^{BB}\,\mathrm{lens}$",
        color="grey",
        linestyle="-",
        linewidth=2,
    )

    plt.plot(ell_range, cl_bb_obs, label=rf"{name} $C_\ell^{{\mathrm{{obs}}}}$", color="green")
    plt.plot(
        ell_range,
        cl_total_res,
        label=rf"{name} $C_\ell^{{\mathrm{{res}}}}$",
        color="black",
    )
    plt.plot(
        ell_range,
        cl_syst_res,
        label=rf"{name} $C_\ell^{{\mathrm{{syst}}}}$",
        color="blue",
    )
    plt.plot(
        ell_range,
        cl_stat_res,
        label=rf"{name} $C_\ell^{{\mathrm{{stat}}}}$",
        color="orange",
    )
    plt.plot(
        ell_range,
        cl_true,
        label=rf"{name} $C_\ell^{{\mathrm{{true}}}}$",
        color="purple",
        linestyle="--",
    )

    plt.title(None)
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell^{BB}$ [$\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.legend(loc="best", ncol=2, framealpha=0.95, columnspacing=1.0)

    plt.tight_layout()

    save_or_show(f"bb_spectra_{name}", output_format, output_dir=output_dir, subfolder=subfolder)


def plot_r_estimator(
    name: str,
    r_best: float,
    sigma_r_neg: float,
    sigma_r_pos: float,
    r_grid: Float[Array, " r_grid"],
    L_vals: Float[Array, " r_grid"],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
) -> None:
    """Plot one-dimensional likelihood for r with highlighted estimate."""
    plt.figure(figsize=(6, 5))

    likelihood = L_vals / np.max(L_vals)

    plt.plot(
        r_grid,
        likelihood,
        label=rf"{name} $\hat{{r}} = {r_best:.2e}^{{+{sigma_r_pos:.1e}}}_{{-{sigma_r_neg:.1e}}}$",
        color="purple",
        linewidth=2,
    )
    plt.fill_between(
        r_grid,
        0,
        likelihood,
        where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
        color="purple",
        alpha=0.2,
    )
    plt.axvline(r_best, color="purple", linestyle="--", alpha=0.8)

    plt.axvline(0.0, color="purple", linestyle="--", alpha=0.8, label="True r=0")
    plt.title(f"{name} Estimated $r$ residual")
    plt.xlabel(r"$r$")
    plt.ylabel("Relative Likelihood")
    plt.grid(True)

    plt.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()

    save_or_show(f"r_likelihood_{name}", output_format, output_dir=output_dir, subfolder=subfolder)

    info(f"Estimated r (Reconstructed): {r_best:.4e} (+{sigma_r_pos:.1e}, -{sigma_r_neg:.1e})")


def get_plot_flags(args: Any) -> tuple[dict[str, bool], dict[str, bool]]:
    indiv_flags = {
        "plot_illustration": args.plot_illustrations,
        "plot_params": args.plot_params,
        "plot_patches": args.plot_patches,
        "plot_cl_spectra": args.plot_cl_spectra,
        "plot_cmb_recon": args.plot_cmb_recon,
        "plot_systematic_maps": args.plot_systematic_maps,
        "plot_statistical_maps": args.plot_statistical_maps,
        "plot_r_estimation": args.plot_r_estimation,
    }
    aggregate_flags = {
        "plot_all_spectra": args.plot_all_spectra,
        "plot_all_cmb_recon": args.plot_all_cmb_recon,
        "plot_all_histograms": args.plot_all_histograms,
        "plot_all_params_residuals": args.plot_all_params_residuals,
        "plot_all_systematic_maps": args.plot_all_systematic_maps,
        "plot_all_statistical_maps": args.plot_all_statistical_maps,
        "plot_all_r_estimation": args.plot_all_r_estimation,
        "plot_r_vs_c": args.plot_r_vs_c,
        "plot_v_vs_c": args.plot_v_vs_c,
        "plot_r_vs_v": args.plot_r_vs_v,
        "plot_all_metrics": args.plot_all_metrics,
    }

    if args.plot_all:
        for key in indiv_flags:
            indiv_flags[key] = True
        for key in aggregate_flags:
            aggregate_flags[key] = True

    return indiv_flags, aggregate_flags


def plot_indiv_results(
    name: str,
    computed_results: dict[str, Any],
    indiv_flags: dict[str, bool],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
) -> None:
    """Generate per-run plots according to CLI flags.

    Parameters
    ----------
    name : str
        Run identifier for labeling plots.
    computed_results : dict
        Dictionary containing cmb, cl, r, residual, and plotting_data.
    indiv_flags : dict
        Flags controlling which plots to generate.
    output_format : str
        'png', 'pdf', or 'show'.
    output_dir : str
        Directory to save plots to. Defaults to "plots".
    subfolder : str, optional
        Subdirectory under plots/ for grouped results.
    """

    cmb_pytree = computed_results.get("cmb", None)
    cl_pytree = computed_results.get("cl", None)
    r_pytree = computed_results.get("r", None)
    residual_pytree = computed_results.get("residual", None)
    plotting_data = computed_results.get("plotting_data", None)

    cmb_stokes = combined_cmb_recon = patches_map = None
    cl_bb_r1 = cl_true = ell_range = cl_bb_obs = cl_bb_lens = None
    cl_syst_res = cl_total_res = cl_stat_res = None
    r_best = sigma_r_neg = sigma_r_pos = r_grid = L_vals = None
    syst_map = stat_maps = None
    params_map = None

    if cmb_pytree is not None:
        cmb_stokes = cmb_pytree["cmb"]
        combined_cmb_recon = cmb_pytree["cmb_recon"]
        patches_map = cmb_pytree["patches_map"]

    if cl_pytree is not None:
        cl_bb_r1 = cl_pytree["cl_bb_r1"]
        cl_true = cl_pytree["cl_true"]
        ell_range = cl_pytree["ell_range"]
        cl_bb_obs = cl_pytree["cl_bb_obs"]
        cl_bb_lens = cl_pytree["cl_bb_lens"]
        cl_syst_res = cl_pytree["cl_syst_res"]
        cl_total_res = cl_pytree["cl_total_res"]
        cl_stat_res = cl_pytree["cl_stat_res"]

    if r_pytree is not None:
        r_best = r_pytree["r_best"]
        sigma_r_neg = r_pytree["sigma_r_neg"]
        sigma_r_pos = r_pytree["sigma_r_pos"]
        r_grid = r_pytree["r_grid"]
        L_vals = r_pytree["L_vals"]

    if residual_pytree is not None:
        syst_map = residual_pytree.get("syst_map")
        stat_maps = residual_pytree.get("stat_maps")

    if plotting_data is not None:
        params_map = plotting_data.get("params_map")

    if indiv_flags["plot_params"]:
        assert params_map is not None, "No params_map found for plotting."
        plot_params(name, params_map, output_format, output_dir=output_dir, subfolder=subfolder)
    if indiv_flags["plot_patches"]:
        assert patches_map is not None, "No patches_map found for plotting."
        plot_patches(name, patches_map, output_format, output_dir=output_dir, subfolder=subfolder)

    if indiv_flags["plot_cmb_recon"]:
        assert cmb_stokes is not None, "No cmb_stokes found for plotting."
        plot_cmb_reconstructions(
            name,
            cmb_stokes,
            combined_cmb_recon,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
        )

    if indiv_flags["plot_systematic_maps"]:
        assert syst_map is not None, "No systematic residual map found for plotting."
        plot_systematic_residual_maps(
            name, syst_map, output_format, output_dir=output_dir, subfolder=subfolder
        )

    if indiv_flags["plot_statistical_maps"]:
        assert stat_maps is not None, "No statistical residual maps found for plotting."
        plot_statistical_residual_maps(
            name, stat_maps, output_format, output_dir=output_dir, subfolder=subfolder
        )

    if indiv_flags["plot_cl_spectra"]:
        assert all(
            v is not None
            for v in [
                cl_bb_obs,
                cl_syst_res,
                cl_total_res,
                cl_stat_res,
                cl_bb_r1,
                cl_bb_lens,
                cl_true,
                ell_range,
            ]
        ), "Incomplete Cl data for plotting."
        plot_cl_residuals(
            name,
            cl_bb_obs,
            cl_syst_res,
            cl_total_res,
            cl_stat_res,
            cl_bb_r1,
            cl_bb_lens,
            cl_true,
            ell_range,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
        )

    if indiv_flags["plot_r_estimation"]:
        assert all(
            v is not None
            for v in [
                r_best,
                sigma_r_neg,
                sigma_r_pos,
                r_grid,
                L_vals,
            ]
        ), "Incomplete r estimation data for plotting."
        plot_r_estimator(
            name,
            r_best,
            sigma_r_neg,
            sigma_r_pos,
            r_grid,
            L_vals,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
        )


def plot_aggregate_results(
    names: list[str],
    computed_results: dict[str, dict[str, Any]],
    aggregate_flags: dict[str, bool],
    output_format: str,
    output_dir: str = "plots",
) -> None:
    # Aggregate plots
    # Stack all relevant data for aggregate plots
    stacked_titles = []
    stacked_cmb = []
    stacked_cl = []
    stacked_r = []
    stacked_syst = []
    stacked_stat = []
    stacked_all_params = []
    stacked_params_maps = []
    first_true_params = None

    for name, (kw, computed_res) in zip(names, computed_results.items()):
        cmb_pytree = computed_res.get("cmb", None)
        cl_pytree = computed_res.get("cl", None)
        r_pytree = computed_res.get("r", None)
        residual_pytree = computed_res.get("residual", None)
        plotting_data = computed_res.get("plotting_data", None)

        stacked_titles.append(name)
        if cmb_pytree is not None:
            stacked_cmb.append(cmb_pytree)

        if cl_pytree is not None:
            stacked_cl.append(cl_pytree)

        if r_pytree is not None:
            stacked_r.append(r_pytree)

        if residual_pytree is not None:
            syst_map = residual_pytree.get("syst_map", None)
            stat_maps = residual_pytree.get("stat_maps", None)
            if syst_map is not None:
                stacked_syst.append(syst_map)
            if stat_maps is not None:
                stacked_stat.append(stat_maps)

        if plotting_data is not None:
            stacked_all_params.append(plotting_data.get("all_params"))
            stacked_params_maps.append(plotting_data.get("params_map"))
            if first_true_params is None:
                first_true_params = plotting_data.get("true_params")
        else:
            stacked_all_params.append(None)
            stacked_params_maps.append(None)

    if aggregate_flags["plot_r_vs_c"]:
        plot_r_vs_clusters(
            stacked_titles, stacked_cmb, stacked_r, output_format, output_dir=output_dir
        )
        plt.close("all")
    if aggregate_flags["plot_v_vs_c"]:
        plot_variance_vs_clusters(stacked_titles, stacked_cmb, output_format, output_dir=output_dir)
        plt.close("all")
    if aggregate_flags["plot_r_vs_v"]:
        plot_variance_vs_r(
            stacked_titles, stacked_cmb, stacked_r, output_format, output_dir=output_dir
        )
        plt.close("all")

    if aggregate_flags["plot_all_systematic_maps"]:
        plot_all_systematic_residuals(
            stacked_titles, stacked_syst, output_format, output_dir=output_dir
        )
        plt.close("all")

    if aggregate_flags["plot_all_statistical_maps"]:
        plot_all_statistical_residuals(
            stacked_titles, stacked_stat, output_format, output_dir=output_dir
        )
        plt.close("all")

    if aggregate_flags["plot_all_cmb_recon"]:
        plot_all_cmb(stacked_titles, stacked_cmb, output_format, output_dir=output_dir)
        plt.close("all")

    if aggregate_flags["plot_all_params_residuals"]:
        if first_true_params:
            plot_all_params_residuals(
                stacked_titles,
                stacked_params_maps,
                first_true_params,
                output_format,
                output_dir=output_dir,
            )
            plt.close("all")

    if aggregate_flags["plot_all_histograms"]:
        if first_true_params:
            plot_all_histograms(
                stacked_titles,
                stacked_all_params,
                first_true_params,
                output_format,
                output_dir=output_dir,
            )
            plt.close("all")

    if aggregate_flags["plot_all_spectra"]:
        plot_all_cl_residuals(stacked_titles, stacked_cl, output_format, output_dir=output_dir)
        plt.close("all")

    if aggregate_flags["plot_all_r_estimation"]:
        plot_all_r_estimation(stacked_titles, stacked_r, output_format, output_dir=output_dir)
        plt.close("all")

    if aggregate_flags["plot_all_metrics"]:
        plot_all_variances(stacked_titles, stacked_cmb, output_format, output_dir=output_dir)
        plt.close("all")


def run_plot(
    matched_results: dict[str, Any],
    titles: list[str],
    nside: int,
    instrument: FGBusterInstrument,
    snapshot_path: str,
    flags: dict[str, bool],
    indiv_flags: dict[str, bool],
    aggregate_flags: dict[str, bool],
    solver_name: str,
    max_iter: int,
    output_format: str,
    font_size: int,
    output_dir: str,
    noise_selection: str = "min-value",
    sky_tag: str = "c1d0s0",
) -> int:
    if not output_dir:
        output_dir = "plots"

    if output_format != "show":
        os.makedirs(output_dir, exist_ok=True)

    if len(titles) != len(matched_results):
        error("Number of titles does not match number of existing results.")
        return -1

    existing, to_compute = load_and_filter_snapshot(snapshot_path, matched_results)

    if to_compute:
        warning(f"{len(to_compute)} run groups are not in the snapshot.")
        hint(
            "Consider running 'r_analysis snap ...' to cache these results "
            "for faster plotting next time."
        )
        computed = compute_all(
            to_compute,
            nside,
            instrument,
            flags,
            max_iter,
            solver_name,
            noise_selection=noise_selection,
            sky_tag=sky_tag,
        )
        # Serialize computed results for snapshot storage
        serialized_computed = {kw: serialize_snapshot_payload(res) for kw, res in computed.items()}
        existing.update(serialized_computed)

    for name, (kw, computed_results) in tqdm(
        zip(titles, existing.items()),
        desc="Generating per group plots",
        leave=False,
    ):
        plot_subfolder = kw
        if kw in matched_results:
            # matched_results[kw] is (folders, index_spec, root_dir)
            root_dir = matched_results[kw][2]
            if root_dir:
                plot_subfolder = os.path.join(root_dir, kw)

        plot_indiv_results(
            name,
            computed_results,
            indiv_flags,
            output_format,
            output_dir=output_dir,
            subfolder=plot_subfolder,
        )
        plt.close("all")

    plot_aggregate_results(
        titles,
        existing,
        aggregate_flags,
        output_format,
        output_dir=output_dir,
    )

    return 0
