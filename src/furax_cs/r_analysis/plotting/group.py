"""Multi-run (group) aggregate plot functions — one file per group."""

from __future__ import annotations

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float

from ...logging_utils import debug, warning
from . import get_run_color, save_or_show

font_size = 22


def plot_all_cl_residuals(
    names: list[str],
    cl_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
    group_name: str | None = None,
    colors: list[str] | None = None,
    legend_anchor: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    r_range: tuple[float, float] | None = None,
    r_plot: tuple[float, float] | None = None,
) -> None:
    """Overlay residual BB spectra with a simple legend for line styles only."""

    rc_overrides = {
        "legend.fontsize": font_size * 0.55,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.labelsize": font_size,
    }

    with plt.rc_context(rc_overrides):
        plt.figure(figsize=figsize if figsize else (10, 8))

        if len(cl_pytree_list) == 0:
            warning("No power spectra results to plot")
            return

        cl_bb_r1 = cl_pytree_list[0]["cl_bb_r1"]
        ell_range = cl_pytree_list[0]["ell_range"]
        cl_bb_lens = cl_pytree_list[0]["cl_bb_lens"]

        if r_plot is not None:
            for r_val in r_plot:
                debug(f"Plotting r={r_val:.0e} line for BB spectrum")
                if r_val > 0:
                    plt.plot(
                        ell_range,
                        r_val * cl_bb_r1,
                        color="grey",
                        linestyle="--",
                        linewidth=1.5,
                        label=rf"$C_\ell^{{BB}},\; r={r_val:.0e}$",
                    )
        elif r_range is not None:
            r_lo, r_hi = r_range
            plt.fill_between(
                ell_range,
                r_lo * cl_bb_r1,
                r_hi * cl_bb_r1,
                color="grey",
                alpha=0.35,
                label=rf"$C_\ell^{{BB}},\; r\in[{r_lo:.0e},\,{r_hi:.0e}]$",
            )
        else:
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
            color="grey",
            linestyle="-",
            linewidth=2,
            label=r"$C_\ell^{BB}\,\mathrm{lens}$",
        )

        for i, (name, cl_pytree) in enumerate(zip(names, cl_pytree_list)):
            color = get_run_color(i, colors)
            linewidth = 1.5

            if cl_pytree["cl_total_res"] is not None:
                plt.plot(
                    ell_range, cl_pytree["cl_total_res"], color=color, linestyle="--", label=None
                )
            if cl_pytree["cl_syst_res"] is not None:
                plt.plot(
                    ell_range,
                    cl_pytree["cl_syst_res"],
                    color=color,
                    linestyle="-",
                    linewidth=linewidth,
                    label=None,
                )
            if cl_pytree["cl_stat_res"] is not None:
                plt.plot(
                    ell_range,
                    cl_pytree["cl_stat_res"],
                    color=color,
                    linestyle=":",
                    linewidth=linewidth,
                    label=None,
                )

        plt.plot([], [], color="black", linestyle="--", label=r"Total ($C_\ell^{\mathrm{res}}$)")
        plt.plot(
            [], [], color="black", linestyle="-", label=r"Systematic ($C_\ell^{\mathrm{syst}}$)"
        )
        plt.plot(
            [], [], color="black", linestyle=":", label=r"Statistical ($C_\ell^{\mathrm{stat}}$)"
        )
        for i, name in enumerate(names):
            plt.plot([], [], color=get_run_color(i, colors), linewidth=2, label=name)

        plt.title(None)
        plt.xlabel(r"Multipole $\ell$")
        plt.ylabel(r"$C_\ell^{BB}$ [$\mu K^2$]")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.4)

        legend_kwargs: dict = {"frameon": True, "framealpha": 0.95, "edgecolor": "grey"}
        if legend_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(legend_anchor)
            legend_kwargs["loc"] = "upper right"
        else:
            legend_kwargs["loc"] = "upper right"
        plt.legend(**legend_kwargs)
        plt.tight_layout()

        file_suffix = group_name if group_name else "_".join(names)
        save_or_show(f"bb_spectra_{file_suffix}", output_format, output_dir=output_dir)


def plot_all_histograms(
    names: list[str],
    all_params_list: list[list[dict[str, Float[Array, " npix"]]]],
    true_params: dict[str, Float[Array, " npix"]],
    output_format: str,
    output_dir: str = "plots",
    group_name: str | None = None,
    colors: list[str] | None = None,
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

        if key in true_params:
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

        for i, (name, all_params) in enumerate(zip(names, all_params_list)):
            if all_params is None:
                continue

            recovered_vals = []
            for p in all_params:
                if key in p:
                    val = p[key]
                    recovered_vals.append(val)

            if recovered_vals:
                all_recovered = np.concatenate(recovered_vals, axis=1)
                color = get_run_color(i, colors)

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
    file_suffix = group_name if group_name else "_".join(names)
    save_or_show(
        f"minimize_histograms_{file_suffix}",
        output_format,
        output_dir=output_dir,
    )


def plot_all_r_estimation(
    names: list[str],
    r_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
    group_name: str | None = None,
    colors: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    legend_anchor: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    r_plot: tuple[float, float] | None = None,
) -> None:
    """Compare r likelihood curves across runs in a single figure."""

    rc_overrides = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size * 0.7,
        "axes.titlesize": font_size,
    }

    with plt.rc_context(rc_overrides):
        plt.figure(figsize=figsize if figsize else (10, 8))
        for i, (name, r_data) in enumerate(zip(names, r_pytree_list)):
            if r_data["r_best"] is None:
                warning(f"No r estimation for {name}, skipping plot.")
                continue

            r_grid = r_data["r_grid"]
            L_vals = r_data["L_vals"]
            r_best = r_data["r_best"]
            sigma_r_neg = r_data["sigma_r_neg"]
            sigma_r_pos = r_data["sigma_r_pos"]

            color = get_run_color(i, colors)
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

        if r_plot is not None:
            for r_truth in r_plot:
                plt.axvline(
                    x=r_truth,
                    color="black",
                    linestyle="--",
                    alpha=0.7,
                    label=rf"Truth $r={r_truth:.0e}$",
                )

        plt.xlabel(r"$r$")
        plt.ylabel(r"$L_{\text{cosmo}}$")
        if xlim:
            plt.xlim(*xlim)
        else:
            plt.xlim(-0.002, 0.005)
        plt.grid(True, which="both", ls=":")

        legend_kwargs: dict = {
            "borderaxespad": 0,
            "frameon": True,
            "framealpha": 0.95,
        }
        if legend_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(legend_anchor)
            legend_kwargs["loc"] = "upper right"
        else:
            legend_kwargs["loc"] = "upper right"
        plt.legend(**legend_kwargs)
        plt.tight_layout()

        file_suffix = group_name if group_name else "_".join(names)
        save_or_show(f"r_likelihood_{file_suffix}", output_format, output_dir=output_dir)
