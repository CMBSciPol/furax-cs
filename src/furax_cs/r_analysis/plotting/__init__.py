"""Plotting subpackage for r_analysis — shared utilities and orchestration."""

from __future__ import annotations

import os
import re
from typing import Any

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 # pyright: ignore[reportUnusedImport]
from furax.obs.stokes import Stokes
from jaxtyping import Array, Float
from tqdm.auto import tqdm

from ...logging_utils import warning
from ..snapshot import CompSepResult, _result_to_plot_dict

plt.style.use("science")

# Sets of aggregate plot flag keys
MULTIPLE_FILE_AGGREGATE_PLOTS = {
    "plot_all_spectra",
    "plot_all_histograms",
    "plot_all_r_estimation",
    "plot_r_vs_v",
}

SINGLE_FILE_AGGREGATE_PLOTS = {
    "plot_r_vs_c",
    "plot_v_vs_c",
    "plot_nll_vs_c",
}

font_size = 22
plt.rcParams.update(
    {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
        "text.usetex": True,
    }
)


# ---------------------------------------------------------------------------
# Color system
# ---------------------------------------------------------------------------
def get_run_color(index: int, colors: list[str] | None = None) -> str:
    """Get color for run by index.

    If *colors* is provided, cycles through it.
    Otherwise falls back to the matplotlib property cycle.

    Parameters
    ----------
    index : int
        Zero-based index of the run.
    colors : list[str] | None
        Optional user-specified color list.

    Returns
    -------
    str
        Color string.
    """
    if colors:
        return colors[index % len(colors)]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    cycle_colors = prop_cycle.by_key()["color"]
    return cycle_colors[index % len(cycle_colors)]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def get_symmetric_percentile_limits(
    data_list: list[Float[Array, " n"]], percentile: float = 99
) -> tuple[float, float]:
    """Compute symmetric vmin/vmax from percentile across multiple maps."""
    all_values = []
    for data in data_list:
        valid = data[~np.isnan(data) & (data != hp.UNSEEN)]
        all_values.append(valid)
    combined = np.concatenate(all_values)

    low = np.percentile(combined, 100 - percentile)
    high = np.percentile(combined, percentile)

    abs_max = max(abs(low), abs(high))
    return -abs_max, abs_max


def _truncate_name_if_too_long(name: str, max_length: int = 250) -> str:
    """Truncate long names for plot titles and filenames."""
    if len(name) > max_length:
        return name[: max_length - 3] + "..."
    return name


def set_font_size(size: int) -> None:
    """Set global font size for all plotting functions."""
    global font_size
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
    """Save figure to file or show inline based on output format."""
    from ...logging_utils import success

    if output_format == "show":
        plt.show()
    else:
        ext = "pdf" if output_format == "pdf" else "png"
        dpi = 1200 if ext == "png" else None
        filename = _truncate_name_if_too_long(filename)

        base_dir = output_dir
        if subfolder:
            base_dir = os.path.join(output_dir, subfolder)

        os.makedirs(base_dir, exist_ok=True)

        filepath = os.path.join(base_dir, f"{filename}.{ext}")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight", transparent=True)
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


def get_masked_residual(true_map, model_map):
    return np.where(true_map == hp.UNSEEN, hp.UNSEEN, true_map - model_map)


# ---------------------------------------------------------------------------
# Flag extraction
# ---------------------------------------------------------------------------
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
        "plot_params_residuals": args.plot_params_residuals,
    }
    aggregate_flags = {
        "plot_all_spectra": args.plot_all_spectra,
        "plot_all_histograms": args.plot_all_histograms,
        "plot_all_r_estimation": args.plot_all_r_estimation,
        "plot_r_vs_c": args.plot_r_vs_c,
        "plot_v_vs_c": args.plot_v_vs_c,
        "plot_nll_vs_c": args.plot_nll_vs_c,
        "plot_r_vs_v": args.plot_r_vs_v,
    }

    if args.plot_all:
        for key in indiv_flags:
            indiv_flags[key] = True
        for key in aggregate_flags:
            aggregate_flags[key] = True

    return indiv_flags, aggregate_flags


# ---------------------------------------------------------------------------
# Per-run dispatcher
# ---------------------------------------------------------------------------
def plot_indiv_results(
    name: str,
    computed_results: dict[str, Any],
    indiv_flags: dict[str, bool],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
    xlim: tuple[float, float] | None = None,
    r_legend_anchor: tuple[float, float] | None = None,
    r_figsize: tuple[float, float] | None = None,
) -> None:
    """Generate per-run plots according to CLI flags."""
    from .individual import (
        plot_cl_residuals,
        plot_cmb_reconstructions,
        plot_params,
        plot_params_residuals,
        plot_patches,
        plot_r_estimator,
        plot_statistical_residual_maps,
        plot_systematic_residual_maps,
    )

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
    true_params = None

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
        true_params = plotting_data.get("true_params")

    if indiv_flags.get("plot_params"):
        assert params_map is not None, "No params_map found for plotting."
        plot_params(name, params_map, output_format, output_dir=output_dir, subfolder=subfolder)
    if indiv_flags.get("plot_patches"):
        assert patches_map is not None, "No patches_map found for plotting."
        plot_patches(name, patches_map, output_format, output_dir=output_dir, subfolder=subfolder)

    if indiv_flags.get("plot_cmb_recon"):
        assert cmb_stokes is not None, "No cmb_stokes found for plotting."
        plot_cmb_reconstructions(
            name,
            cmb_stokes,
            combined_cmb_recon,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
        )

    if indiv_flags.get("plot_systematic_maps"):
        assert syst_map is not None, "No systematic residual map found for plotting."
        plot_systematic_residual_maps(
            name, syst_map, output_format, output_dir=output_dir, subfolder=subfolder
        )

    if indiv_flags.get("plot_statistical_maps"):
        assert stat_maps is not None, "No statistical residual maps found for plotting."
        plot_statistical_residual_maps(
            name, stat_maps, output_format, output_dir=output_dir, subfolder=subfolder
        )

    if indiv_flags.get("plot_cl_spectra"):
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

    if indiv_flags.get("plot_r_estimation"):
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
            xlim=xlim,
            legend_anchor=r_legend_anchor,
            figsize=r_figsize,
        )

    if indiv_flags.get("plot_params_residuals"):
        if params_map is not None and true_params is not None:
            plot_params_residuals(
                name,
                params_map,
                true_params,
                output_format,
                output_dir=output_dir,
                subfolder=subfolder,
            )


# ---------------------------------------------------------------------------
# Aggregate dispatcher
# ---------------------------------------------------------------------------
def plot_aggregate_results(
    names: list[str],
    computed_results: dict[str, dict[str, Any]],
    aggregate_flags: dict[str, bool],
    output_format: str,
    output_dir: str = "plots",
    group_name: str | None = None,
    colors: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    r_legend_anchor: tuple[float, float] | None = None,
    s_legend_anchor: tuple[float, float] | None = None,
    r_figsize: tuple[float, float] | None = None,
    s_figsize: tuple[float, float] | None = None,
    r_range: tuple[float, float] | None = None,
    r_plot: tuple[float, float] | None = None,
) -> None:
    from .group import plot_all_cl_residuals, plot_all_histograms, plot_all_r_estimation
    from .single import plot_variance_vs_r

    stacked_titles = []
    stacked_cmb = []
    stacked_cl = []
    stacked_r = []
    stacked_all_params = []
    first_true_params = None

    for name, (kw, computed_res) in zip(names, computed_results.items()):
        cmb_pytree = computed_res.get("cmb", None)
        cl_pytree = computed_res.get("cl", None)
        r_pytree = computed_res.get("r", None)
        plotting_data = computed_res.get("plotting_data", None)

        stacked_titles.append(name)
        if cmb_pytree is not None:
            stacked_cmb.append(cmb_pytree)

        if cl_pytree is not None:
            stacked_cl.append(cl_pytree)

        if r_pytree is not None:
            stacked_r.append(r_pytree)

        if plotting_data is not None:
            stacked_all_params.append(plotting_data.get("all_params"))
            if first_true_params is None:
                first_true_params = plotting_data.get("true_params")
        else:
            stacked_all_params.append(None)

    if aggregate_flags.get("plot_r_vs_v"):
        plot_variance_vs_r(
            stacked_titles, stacked_cmb, stacked_r, output_format, output_dir=output_dir
        )
        plt.close("all")

    if aggregate_flags.get("plot_all_histograms"):
        if first_true_params:
            plot_all_histograms(
                stacked_titles,
                stacked_all_params,
                first_true_params,
                output_format,
                output_dir=output_dir,
                group_name=group_name,
                colors=colors,
            )
            plt.close("all")

    if aggregate_flags.get("plot_all_spectra"):
        plot_all_cl_residuals(
            stacked_titles,
            stacked_cl,
            output_format,
            output_dir=output_dir,
            group_name=group_name,
            colors=colors,
            legend_anchor=s_legend_anchor,
            figsize=s_figsize,
            r_range=r_range,
            r_plot=r_plot,
        )
        plt.close("all")

    if aggregate_flags.get("plot_all_r_estimation"):
        plot_all_r_estimation(
            stacked_titles,
            stacked_r,
            output_format,
            output_dir=output_dir,
            group_name=group_name,
            colors=colors,
            xlim=xlim,
            legend_anchor=r_legend_anchor,
            figsize=r_figsize,
            r_plot=r_plot,
        )
        plt.close("all")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def run_grouped_plot(
    groups: list[tuple[str, Any]],
    indiv_flags: dict[str, bool],
    aggregate_flags: dict[str, bool],
    output_format: str,
    font_size: int,
    output_dir: str,
    group_titles: list[str] | None = None,
    row_titles: list[str] | None = None,
    colors: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    r_legend_anchor: tuple[float, float] | None = None,
    s_legend_anchor: tuple[float, float] | None = None,
    r_figsize: tuple[float, float] | None = None,
    s_figsize: tuple[float, float] | None = None,
    r_range: tuple[float, float] | None = None,
    r_plot: tuple[float, float] | None = None,
) -> int:
    """Run plots with one group per `-g` pattern."""
    from .single import plot_single_file_grouped

    if not output_dir:
        output_dir = "plots"

    set_font_size(font_size)
    if output_format != "show":
        os.makedirs(output_dir, exist_ok=True)

    per_group_flags = {
        k: (v if k in MULTIPLE_FILE_AGGREGATE_PLOTS else False) for k, v in aggregate_flags.items()
    }
    single_flags = {k: aggregate_flags.get(k, False) for k in SINGLE_FILE_AGGREGATE_PLOTS}

    all_groups_collected: list[tuple[str, list[str], dict[str, Any]]] = []
    row_idx = 0

    for idx, (pattern, group_ds) in enumerate(groups):
        group_label = group_titles[idx] if (group_titles and idx < len(group_titles)) else pattern
        safe = re.sub(r"[^\w\-]", "_", group_label).strip("_")
        group_dir = os.path.join(output_dir, safe)
        if output_format != "show":
            os.makedirs(group_dir, exist_ok=True)

        seen_in_group: set[str] = set()
        deduped_rows: list[Any] = []
        for row in group_ds:
            k = str(row["kw"])
            if k not in seen_in_group:
                seen_in_group.add(k)
                deduped_rows.append(row)

        names: list[str] = []
        kw_to_plot: dict[str, Any] = {}

        for row in tqdm(deduped_rows, desc=f"Group '{group_label}'"):
            result = CompSepResult.from_dataset(row)
            plot_dict = _result_to_plot_dict(result)
            if row_titles and row_idx < len(row_titles):
                row_label = row_titles[row_idx]
            else:
                row_label = result.name

            plot_indiv_results(
                row_label,
                plot_dict,
                indiv_flags,
                output_format,
                output_dir=group_dir,
                subfolder=result.kw,
                xlim=xlim,
                r_legend_anchor=r_legend_anchor,
                r_figsize=r_figsize,
            )
            plt.close("all")
            names.append(row_label)
            kw_to_plot[result.kw] = plot_dict
            row_idx += 1

        if not names:
            warning(f"Group '{group_label}' matched no parquet rows, skipping.")
            continue

        plot_aggregate_results(
            names,
            kw_to_plot,
            per_group_flags,
            output_format,
            output_dir=group_dir,
            group_name=group_label,
            colors=colors,
            xlim=xlim,
            r_legend_anchor=r_legend_anchor,
            s_legend_anchor=s_legend_anchor,
            r_figsize=r_figsize,
            s_figsize=s_figsize,
            r_range=r_range,
            r_plot=r_plot,
        )
        all_groups_collected.append((group_label, names, kw_to_plot))

    plot_single_file_grouped(
        all_groups_collected, single_flags, output_format, output_dir, colors=colors
    )
    return 0


# ---------------------------------------------------------------------------
# Re-exports for backward compatibility
# ---------------------------------------------------------------------------
from .group import plot_all_cl_residuals, plot_all_r_estimation  # noqa: E402, F401
from .individual import (  # noqa: E402, F401
    plot_cl_residuals,
    plot_cmb_reconstructions,
    plot_params,
    plot_params_residuals,
    plot_patches,
    plot_r_estimator,
    plot_statistical_residual_maps,
    plot_systematic_residual_maps,
)

__all__ = [
    "get_run_color",
    "get_symmetric_percentile_limits",
    "get_min_variance",
    "get_masked_residual",
    "set_font_size",
    "save_or_show",
    "get_plot_flags",
    "plot_indiv_results",
    "plot_aggregate_results",
    "run_grouped_plot",
    # re-exports
    "plot_params",
    "plot_patches",
    "plot_cmb_reconstructions",
    "plot_systematic_residual_maps",
    "plot_statistical_residual_maps",
    "plot_cl_residuals",
    "plot_r_estimator",
    "plot_params_residuals",
    "plot_all_cl_residuals",
    "plot_all_r_estimation",
]
