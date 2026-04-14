"""Per-run (individual) plot functions."""

from __future__ import annotations

import os

import healpy as hp
import jax
import matplotlib.pyplot as plt
import numpy as np
from furax.obs.stokes import Stokes
from jaxtyping import Array, Float, Int

from ...logging_utils import info
from . import get_masked_residual, get_min_variance, get_symmetric_percentile_limits, save_or_show


def plot_params(
    name: str,
    params: dict[str, Float[Array, " npix"]],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
    transparent: bool = True,
) -> None:
    """Plot recovered spectral parameter maps for a single configuration."""
    plot_vertical = os.environ.get("FURAX_CS_PLOT_VERTICAL", "0") == "1"
    if plot_vertical:
        fig_size = (8, 16)
        subplot_args = (3, 1, lambda i: i + 1)
    else:
        fig_size = (16, 8)
        subplot_args = (1, 3, lambda i: i + 1)

    _ = plt.figure(figsize=fig_size)

    keys = ["beta_dust", "temp_dust", "beta_pl"]
    names = [r"$\beta_d$", r"$T_d$ [K]", r"$\beta_s$"]

    for i, (key, param_name) in enumerate(zip(keys, names)):
        param_map = params[key]
        hp.mollview(
            param_map,
            title=param_name,
            sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
            bgcolor=(0.0,) * 4,
            cbar=True,
            format="%.4f",
        )

    save_or_show(
        f"params_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
        transparent=transparent,
    )
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
    subfolder: str | None = None,
    transparent: bool = True,
) -> None:
    """Visualise patch assignments (cluster labels) for each spectral parameter."""
    plot_vertical = os.environ.get("FURAX_CS_PLOT_VERTICAL", "0") == "1"
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
            title=patch_name,
            sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
            bgcolor=(0.0,) * 4,
            cbar=True,
        )
    save_or_show(
        f"patches_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
        transparent=transparent,
    )


def plot_cmb_reconstructions(
    name: str,
    cmb_stokes: Stokes,
    cmb_recon: Stokes,
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
    transparent: bool = True,
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
    plt.title("CMB Reconstruction")
    save_or_show(
        f"cmb_recon_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
        transparent=transparent,
    )


def plot_systematic_residual_maps(
    name: str,
    syst_map: Float[Array, " 3 npix"],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
    transparent: bool = True,
) -> None:
    """Plot systematic residual Q/U maps for a single configuration."""
    syst_q = np.where(syst_map[1] == hp.UNSEEN, np.nan, syst_map[1])
    syst_u = np.where(syst_map[2] == hp.UNSEEN, np.nan, syst_map[2])

    vmin, vmax = get_symmetric_percentile_limits([syst_q, syst_u])

    plt.figure(figsize=(12, 6))

    hp.mollview(
        syst_q,
        title=r"Systematic Residual (Q) ($\mu$K)",
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
        title=r"Systematic Residual (U) ($\mu$K)",
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
        transparent=transparent,
    )


def plot_statistical_residual_maps(
    name: str,
    stat_maps: list[Float[Array, " 3 npix"]],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
    transparent: bool = True,
) -> None:
    """Plot statistical residual Q/U maps for a single configuration."""
    indx = [0, 1, 2]
    for i in indx:
        if i >= len(stat_maps):
            info(f"Only {len(stat_maps)} statistical residual maps available. Skipping index {i}.")
            continue
        stat_map_first = stat_maps[i]

        stat_q = np.where(stat_map_first[1] == hp.UNSEEN, np.nan, stat_map_first[1])
        stat_u = np.where(stat_map_first[2] == hp.UNSEEN, np.nan, stat_map_first[2])

        vmin, vmax = get_symmetric_percentile_limits([stat_q, stat_u])

        plt.figure(figsize=(12, 6))

        hp.mollview(
            stat_q,
            title=r"Statistical Residual (Q) ($\mu$K)",
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
            title=r"Statistical Residual (U) ($\mu$K)",
            sub=(1, 2, 2),
            min=vmin,
            max=vmax,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

        save_or_show(
            f"statistical_residual_maps_{name}_realization_{i}",
            output_format,
            output_dir=output_dir,
            subfolder=subfolder,
            transparent=transparent,
        )


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
    transparent: bool = True,
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

    plt.plot(ell_range, cl_bb_obs, label=r"$C_\ell^{\mathrm{obs}}$", color="green")
    plt.plot(ell_range, cl_total_res, label=r"$C_\ell^{\mathrm{res}}$", color="black")
    plt.plot(ell_range, cl_syst_res, label=r"$C_\ell^{\mathrm{syst}}$", color="blue")
    plt.plot(ell_range, cl_stat_res, label=r"$C_\ell^{\mathrm{stat}}$", color="orange")
    plt.plot(ell_range, cl_true, label=r"$C_\ell^{\mathrm{true}}$", color="purple", linestyle="--")

    plt.title("")
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell^{BB}$ [$\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.legend(loc="best", ncol=2, framealpha=0.95, columnspacing=1.0)

    plt.tight_layout()

    save_or_show(
        f"bb_spectra_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
        transparent=transparent,
    )


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
    xlim: tuple[float, float] | None = None,
    legend_anchor: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    transparent: bool = True,
) -> None:
    """Plot one-dimensional likelihood for r with highlighted estimate."""
    plt.figure(figsize=figsize if figsize else (6, 5))

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
    plt.title(r"Estimated $r$ residual")
    plt.xlabel(r"$r$")
    plt.ylabel("Likelihood")
    plt.grid(True)
    if xlim:
        plt.xlim(*xlim)

    legend_kwargs: dict = {
        "frameon": True,
        "framealpha": 0.95,
        "fontsize": plt.rcParams["font.size"] * 0.7,
    }
    if legend_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = tuple(legend_anchor)
        legend_kwargs["loc"] = "upper left"
    else:
        legend_kwargs["loc"] = "upper right"
    plt.legend(**legend_kwargs)
    plt.tight_layout()

    save_or_show(
        f"r_likelihood_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
        transparent=transparent,
    )

    info(f"Estimated r (Reconstructed): {r_best:.4e} (+{sigma_r_pos:.1e}, -{sigma_r_neg:.1e})")


def plot_params_residuals(
    name: str,
    params_map: dict[str, Float[Array, " npix"]],
    true_params: dict[str, Float[Array, " npix"]],
    output_format: str,
    output_dir: str = "plots",
    subfolder: str | None = None,
    transparent: bool = True,
) -> None:
    """Plot individual param maps + residuals vs truth for one run.

    Layout: 3 rows (beta_d, T_d, beta_s) x 2 cols (map, residual).
    """
    from ...logging_utils import warning

    param_configs = [
        {"key": "beta_dust", "label": r"$\beta_{d}$"},
        {"key": "temp_dust", "label": r"$T_{d}$"},
        {"key": "beta_pl", "label": r"$\beta_{s}$"},
    ]

    fig = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(3, 2, hspace=0.3, wspace=0.1)

    for row, config in enumerate(param_configs):
        key = config["key"]
        label = config["label"]

        if key not in true_params or key not in params_map:
            warning(f"Missing data for {key}. Skipping row.")
            continue

        res = get_masked_residual(true_params[key], params_map[key])

        ax_map = fig.add_subplot(gs[row, 0])
        plt.sca(ax_map)
        hp.mollview(
            params_map[key],
            title=f"{name} {label}",
            hold=True,
            bgcolor=(0,) * 4,
            format="%.4f",
        )

        ax_res = fig.add_subplot(gs[row, 1])
        plt.sca(ax_res)
        hp.mollview(
            res,
            title=f"Residual (True - {name})",
            cmap="RdBu_r",
            hold=True,
            bgcolor=(0,) * 4,
            format="%.4f",
        )

    save_or_show(
        f"params_residuals_{name}",
        output_format,
        output_dir=output_dir,
        subfolder=subfolder,
        transparent=transparent,
    )
