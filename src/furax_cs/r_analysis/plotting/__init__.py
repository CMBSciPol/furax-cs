"""Plotting subpackage for r_analysis — shared utilities and orchestration."""

from __future__ import annotations

import os
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 # pyright: ignore[reportUnusedImport]
from furax.obs.stokes import Stokes
from jaxtyping import Array, Float

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

font_size = 24
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
# Number formatting
# ---------------------------------------------------------------------------
# Decade used when an entry carries no usable magnitude of its own.
R_DEFAULT_EXPONENT = -3
# A panel keeps one shared exponent while its entries stay within this dynamic range.
# Beyond it a single exponent would round the smallest entry away, so each entry falls
# back to its own decade.
R_MAX_PANEL_SPAN = 20.0
# Significant digits allowed on each uncertainty.
R_SIGNIFICANT_DIGITS = 2


def resolve_r_exponents(
    entries: list[tuple[float, float, float]],
    max_panel_span: float = R_MAX_PANEL_SPAN,
) -> list[int]:
    """Choose the power of ten each (r, +sigma, -sigma) triplet is quoted in.

    The exponent comes from the largest quantity in the entry, which for a measurement
    consistent with zero is the uncertainty rather than the estimate. Scaling on the
    uncertainty keeps both printed to a comparable size; scaling on an estimate that is
    a small fraction of its own error turns the legend into things like 8.5 +65 -27.

    All entries of a panel share one exponent whenever their dynamic range allows it,
    so the curves can be compared digit by digit. When the panel spans more than
    *max_panel_span*, a shared exponent would print the smallest uncertainty as 0.0,
    and each entry falls back to its own decade instead.

    Parameters
    ----------
    entries : list of (float, float, float)
        One ``(r_best, sigma_r_pos, sigma_r_neg)`` triplet per curve.
    max_panel_span : float
        Largest ratio between the biggest quantity in the panel and the smallest
        uncertainty that still allows a shared exponent.

    Returns
    -------
    list of int
        One exponent per entry, in the order given.
    """
    if not entries:
        return []

    scales = [max(abs(float(r)), abs(float(sp)), abs(float(sn))) for r, sp, sn in entries]
    errors = [max(abs(float(sp)), abs(float(sn))) for _, sp, sn in entries]
    usable_scales = [s for s in scales if np.isfinite(s) and s > 0]
    usable_errors = [e for e in errors if np.isfinite(e) and e > 0]

    if not usable_scales or not usable_errors:
        return [R_DEFAULT_EXPONENT] * len(entries)

    largest = max(usable_scales)
    if largest / min(usable_errors) <= max_panel_span:
        shared = int(np.floor(np.log10(largest)))
        return [shared] * len(entries)

    return [
        int(np.floor(np.log10(s))) if np.isfinite(s) and s > 0 else R_DEFAULT_EXPONENT
        for s in scales
    ]


def _mantissa(value: float, exponent: int, significant: int | None = None) -> str:
    """Mantissa of *value* at 10**exponent, without a signed zero.

    With *significant* set the mantissa is rounded to that many significant digits,
    which is how the uncertainties are quoted; otherwise it keeps one decimal, which is
    how the estimate is quoted.

    Scales and rounds in decimal rather than binary. Dividing by ``10.0**-3`` turns a
    boundary value such as 6.5e-4 into 0.6499999999999999, which then rounds down and
    biases every half-way uncertainty towards zero.
    """
    scaled = Decimal(repr(float(value))).scaleb(-exponent)

    if significant is None or scaled == 0:
        quantum = Decimal("0.1")
    else:
        quantum = Decimal(1).scaleb(scaled.copy_abs().adjusted() - significant + 1)

    text = format(scaled.quantize(quantum, rounding=ROUND_HALF_UP), "f")
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def format_r_with_errors(
    r_best: float,
    sigma_r_pos: float,
    sigma_r_neg: float,
    exponent: int | None = None,
) -> str:
    r"""Typeset an r estimate and its 68% interval with one shared exponent.

    Renders ``(0.1\,^{+2.7}_{-2.6}) \times 10^{-3}``: the estimate and both
    uncertainties are scaled by the same power of ten and quoted to one decimal, so
    the dominant quantity carries two significant digits and no more. Quoting a mean
    and its error in different decades hides how many of the mean's digits are
    actually constrained.

    Parameters
    ----------
    r_best : float
        Maximum-likelihood estimate.
    sigma_r_pos, sigma_r_neg : float
        Upper and lower 68% offsets from *r_best*.
    exponent : int or None
        Power of ten to quote in. Chosen from this entry alone when omitted; pass the
        panel value from :func:`resolve_r_exponents` to keep a figure consistent.

    Returns
    -------
    str
        LaTeX math fragment, without surrounding ``$``.
    """
    if exponent is None:
        exponent = resolve_r_exponents([(r_best, sigma_r_pos, sigma_r_neg)])[0]

    centre = _mantissa(r_best, exponent)
    upper = _mantissa(abs(float(sigma_r_pos)), exponent, R_SIGNIFICANT_DIGITS)
    lower = _mantissa(abs(float(sigma_r_neg)), exponent, R_SIGNIFICANT_DIGITS)
    return rf"({centre}\,^{{+{upper}}}_{{-{lower}}}) \times 10^{{{exponent}}}"


def format_power_of_ten(value: float) -> str:
    r"""Typeset a lone number as ``3 \times 10^{-3}``, or ``0`` when it vanishes."""
    value = float(value)
    if value == 0.0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    if abs(mantissa - round(mantissa)) < 1e-6:
        mantissa_text = f"{round(mantissa):d}"
    else:
        mantissa_text = f"{mantissa:.1f}"

    if mantissa_text == "1":
        return rf"10^{{{exponent}}}"
    if mantissa_text == "-1":
        return rf"-10^{{{exponent}}}"
    return rf"{mantissa_text} \times 10^{{{exponent}}}"


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
    filename: str,
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
    transparent: bool = False,
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
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight", transparent=transparent)
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
    r_exponent: int | None = None,
    r_figsize: tuple[float, float] | None = None,
    transparent: bool = True,
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
        plot_params(
            name,
            params_map,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
            transparent=transparent,
        )
    if indiv_flags.get("plot_patches"):
        assert patches_map is not None, "No patches_map found for plotting."
        plot_patches(
            name,
            patches_map,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
            transparent=transparent,
        )

    if indiv_flags.get("plot_cmb_recon"):
        assert cmb_stokes is not None, "No cmb_stokes found for plotting."
        plot_cmb_reconstructions(
            name,
            cmb_stokes,
            combined_cmb_recon,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
            transparent=transparent,
        )

    if indiv_flags.get("plot_systematic_maps"):
        assert syst_map is not None, "No systematic residual map found for plotting."
        plot_systematic_residual_maps(
            name,
            syst_map,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
            transparent=transparent,
        )

    if indiv_flags.get("plot_statistical_maps"):
        assert stat_maps is not None, "No statistical residual maps found for plotting."
        plot_statistical_residual_maps(
            name,
            stat_maps,
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
            transparent=transparent,
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
            transparent=transparent,
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
            exponent=r_exponent,
            transparent=transparent,
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
                transparent=transparent,
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
    r_exponent: int | None = None,
    s_legend_anchor: tuple[float, float] | None = None,
    r_figsize: tuple[float, float] | None = None,
    s_figsize: tuple[float, float] | None = None,
    r_range: tuple[float, float] | None = None,
    r_plot: tuple[float, float] | None = None,
    transparent: bool = True,
    cl_obs_label: bool = False,
    no_tot_residuals: bool = False,
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
            stacked_titles,
            stacked_cmb,
            stacked_r,
            output_format,
            output_dir=output_dir,
            transparent=transparent,
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
                transparent=transparent,
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
            transparent=transparent,
            cl_obs_label=cl_obs_label,
            no_tot_residuals=no_tot_residuals,
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
            exponent=r_exponent,
            r_plot=r_plot,
            transparent=transparent,
        )
        plt.close("all")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def run_grouped_plot(
    ds: Any,
    runs_patterns: list[str] | None,
    groups_patterns: list[str] | None,
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
    r_exponent: int | None = None,
    s_legend_anchor: tuple[float, float] | None = None,
    r_figsize: tuple[float, float] | None = None,
    s_figsize: tuple[float, float] | None = None,
    r_range: tuple[float, float] | None = None,
    r_plot: tuple[float, float] | None = None,
    transparent: bool = True,
    cl_obs_label: bool = False,
    no_tot_residuals: bool = False,
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

    # 1. Setup groups
    groups_data = []
    if groups_patterns:
        for idx, pattern in enumerate(groups_patterns):
            group_label = (
                group_titles[idx] if (group_titles and idx < len(group_titles)) else pattern
            )
            safe = re.sub(r"[^\w\-]", "_", group_label).strip("_")
            group_dir = os.path.join(output_dir, safe)
            if output_format != "show":
                os.makedirs(group_dir, exist_ok=True)
            groups_data.append(
                {"pattern": pattern, "label": group_label, "dir": group_dir, "rows": []}
            )
    else:
        group_dir = os.path.join(output_dir, "ALL")
        if output_format != "show":
            os.makedirs(group_dir, exist_ok=True)
        groups_data.append({"pattern": ".*", "label": "ALL", "dir": group_dir, "rows": []})

    # 2. Iterate dataset once
    seen_in_group: set[str] = set()
    from tqdm import tqdm

    for row in tqdm(ds, desc="Filtering and grouping dataset"):
        name = str(row.get("name", row.get("kw", "")))

        if runs_patterns and not any(re.search(pat, name) for pat in runs_patterns):
            continue

        for gdata in groups_data:
            if re.search(gdata["pattern"], name):
                k = str(row["kw"])
                group_key = f"{gdata['label']}_{k}"
                if group_key not in seen_in_group:
                    seen_in_group.add(group_key)
                    gdata["rows"].append(row)

    # 3. Sort runs within groups based on runs_patterns order if given
    if runs_patterns:

        def _pattern_order(row):
            name = str(row.get("name", row.get("kw", "")))
            for i, pat in enumerate(runs_patterns):
                if re.search(pat, name):
                    return i
            return len(runs_patterns)

        for gdata in groups_data:
            gdata["rows"].sort(key=_pattern_order)

    # 4. Plot each group
    row_idx = 0
    for gdata in groups_data:
        group_label = gdata["label"]
        group_dir = gdata["dir"]
        deduped_rows = gdata["rows"]

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
                r_exponent=r_exponent,
                r_figsize=r_figsize,
                transparent=transparent,
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
            r_exponent=r_exponent,
            s_legend_anchor=s_legend_anchor,
            r_figsize=r_figsize,
            s_figsize=s_figsize,
            r_range=r_range,
            r_plot=r_plot,
            transparent=transparent,
            cl_obs_label=cl_obs_label,
            no_tot_residuals=no_tot_residuals,
        )
        all_groups_collected.append((group_label, names, kw_to_plot))

    plot_single_file_grouped(
        all_groups_collected,
        single_flags,
        output_format,
        output_dir,
        colors=colors,
        transparent=transparent,
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
