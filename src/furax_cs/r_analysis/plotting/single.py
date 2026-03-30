"""Single-file aggregate plots — one combined chart across all groups."""

from __future__ import annotations

from typing import Any

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from ...logging_utils import info
from . import get_run_color, save_or_show

font_size = 22


def _extract_r_vs_clusters_points(
    patch_key: str, kw_to_plot: dict[str, Any]
) -> list[tuple[int, float]]:
    """Extract (n_clusters, r_plus_sigma) keeping minimum r+sigma per cluster count."""
    base_patch_keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
    method_dict: dict[int, float] = {}

    for plot_dict in kw_to_plot.values():
        cmb_pytree = plot_dict.get("cmb")
        r_data = plot_dict.get("r")
        if cmb_pytree is None or r_data is None or r_data.get("r_best") is None:
            continue

        patches = cmb_pytree["patches_map"]
        if patch_key == "total":
            n_clusters = sum(
                np.unique(patches[k][patches[k] != hp.UNSEEN]).size for k in base_patch_keys
            )
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size

        r_plus_sigma = float(r_data["r_best"]) + float(r_data["sigma_r_pos"])
        if n_clusters not in method_dict or r_plus_sigma < method_dict[n_clusters]:
            method_dict[n_clusters] = r_plus_sigma

    return sorted(method_dict.items())


def _extract_variance_vs_clusters_points(
    patch_key: str, kw_to_plot: dict[str, Any]
) -> list[tuple[int, float]]:
    """Extract (n_clusters, min_variance) keeping minimum variance per cluster count."""
    base_patch_keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
    method_dict: dict[int, float] = {}

    for plot_dict in kw_to_plot.values():
        cmb_pytree = plot_dict.get("cmb")
        if cmb_pytree is None:
            continue

        patches = cmb_pytree["patches_map"]
        if patch_key == "total":
            n_clusters = sum(
                np.unique(patches[k][patches[k] != hp.UNSEEN]).size for k in base_patch_keys
            )
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size

        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_pytree["cmb_recon"])
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_pytree["cmb_recon"], seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        min_variance = float(jnp.min(sum(jax.tree.leaves(variance))))

        if n_clusters not in method_dict or min_variance < method_dict[n_clusters]:
            method_dict[n_clusters] = min_variance

    return sorted(method_dict.items())


def _extract_nll_vs_clusters_points(
    patch_key: str, kw_to_plot: dict[str, Any]
) -> list[tuple[int, float]]:
    """Extract (n_clusters, mean_nll) keeping minimum mean NLL per cluster count."""
    base_patch_keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
    method_dict: dict[int, float] = {}

    for plot_dict in kw_to_plot.values():
        cmb_pytree = plot_dict.get("cmb")
        if cmb_pytree is None:
            continue

        nll_summed = cmb_pytree.get("nll_summed")
        if nll_summed is None:
            continue
        mean_nll = float(np.mean(nll_summed))

        patches = cmb_pytree["patches_map"]
        if patch_key == "total":
            n_clusters = sum(
                np.unique(patches[k][patches[k] != hp.UNSEEN]).size for k in base_patch_keys
            )
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size

        if n_clusters not in method_dict or mean_nll < method_dict[n_clusters]:
            method_dict[n_clusters] = mean_nll

    return sorted(method_dict.items())


def _extract_variance_vs_r_points(
    patch_key: str, kw_to_plot: dict[str, Any]
) -> list[tuple[float, float, float, float]]:
    """Extract (min_variance, r_best, sigma_r_neg, sigma_r_pos) points sorted by variance."""
    points = []

    for plot_dict in kw_to_plot.values():
        cmb_pytree = plot_dict.get("cmb")
        r_data = plot_dict.get("r")
        if cmb_pytree is None or r_data is None or r_data.get("r_best") is None:
            continue

        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_pytree["cmb_recon"])
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_pytree["cmb_recon"], seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        min_variance = float(jnp.min(sum(jax.tree.leaves(variance))))

        points.append(
            (
                min_variance,
                float(r_data["r_best"]),
                float(r_data["sigma_r_neg"]),
                float(r_data["sigma_r_pos"]),
            )
        )

    return sorted(points, key=lambda p: p[0])


def _create_variance_vs_r_plot(
    patch_name: str,
    patch_key: str,
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    r_pytree_list: list[dict[str, Any]],
    output_format: str,
    output_dir: str = "plots",
    overlap_threshold: float = 0.02,
    transparent: bool = True,
) -> None:
    """Helper to plot variance vs best-fit r for a given parameter/totals."""
    points = []
    is_total = patch_key == "total"
    base_patch_keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]

    for name, cmb_pytree, r_data in zip(names, cmb_pytree_list, r_pytree_list):
        if r_data["r_best"] is None:
            continue

        patches = cmb_pytree["patches_map"]
        cluster_counts = {}

        if is_total:
            n_clusters = 0
            for key in base_patch_keys:
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
        return

    points.sort(key=lambda p: p[0])
    variances = np.array([p[0] for p in points])
    r_values = np.array([p[1] for p in points])
    k_values = np.array([p[2] for p in points])
    sigma_r_pos = np.array([p[4] for p in points])
    cluster_breakdowns = [p[5] for p in points]

    plt.figure(figsize=(8, 6))

    cmap = plt.cm.viridis
    norm = Normalize(vmin=k_values.min(), vmax=k_values.max())
    colors = cmap(norm(k_values))

    y_upper_values = r_values + sigma_r_pos

    var_span = np.ptp(variances)
    y_span = np.ptp(y_upper_values)

    if var_span == 0:
        var_span = 1.0
    if y_span == 0:
        y_span = 1.0

    plotted_indices = []
    xs, y_means, y_uppers, c_vals = [], [], [], []

    for i in range(len(variances)):
        is_close = False
        x_curr = variances[i]
        y_curr = y_upper_values[i]

        for idx in plotted_indices:
            x_prev = variances[idx]
            y_prev = y_upper_values[idx]

            dist = np.sqrt(((x_curr - x_prev) / var_span) ** 2 + ((y_curr - y_prev) / y_span) ** 2)
            if dist < overlap_threshold:
                is_close = True
                break

        if is_close:
            continue

        plotted_indices.append(i)
        xs.append(variances[i])
        y_means.append(r_values[i])
        y_uppers.append(y_upper_values[i])
        c_vals.append(colors[i])

    xs = np.array(xs)
    y_means = np.array(y_means)
    y_uppers = np.array(y_uppers)
    c_vals = np.array(c_vals)

    plt.scatter(
        xs, y_means, facecolors="none", edgecolors=c_vals, s=80, linewidth=1, alpha=0.6, zorder=2
    )

    _ = plt.scatter(
        xs,
        y_uppers,
        c=c_vals,
        s=80,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.7,
        zorder=3,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=r"$r$",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=r"$r + \sigma(r)$",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
        ),
    ]
    plt.legend(
        handles=legend_elements, loc="upper right", frameon=True, framealpha=0.9, fancybox=True
    )

    ax = plt.gca()
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1e3:.0f}"))
    ax.yaxis.offsetText.set_visible(False)
    ax.annotate(r"$\times 10^{-3}$", xy=(0, 1), xycoords="axes fraction", fontsize=font_size - 4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v >= 1000 else f"{v:.0f}")
    )

    if is_total:
        cbar.set_label("Total Number of Patches")
        r_plus_sigma = np.array(r_values) + np.array(sigma_r_pos)
        var_indx = 0
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

    plt.xlabel(r"Minimum Variance (Q + U) [$\mu$K]$^2$")
    plt.ylabel(r"Tensor-to-scalar ratio  $r$")
    plt.ylim(-0.0005, 0.005)
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filename_suffix = "total" if is_total else patch_key.replace("_patches", "")
    save_or_show(
        f"variance_vs_residual_r_{filename_suffix}",
        output_format,
        output_dir=output_dir,
        transparent=transparent,
    )
    plt.close()


def plot_variance_vs_r(
    names: list[str],
    cmb_pytree_list: list[dict[str, Any]],
    r_pytree_list: list[dict[str, Array]],
    output_format: str,
    output_dir: str = "plots",
    transparent: bool = True,
) -> None:
    """Plot variance vs best-fit r for each spectral parameter and combined."""
    patch_configs = [
        (r"$K_{\beta_d}$", "beta_dust_patches"),
        (r"$K_{T_d}$", "temp_dust_patches"),
        (r"$K_{\beta_s}$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    rc_overrides = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size * 0.9,
        "axes.titlesize": font_size,
    }

    with plt.rc_context(rc_overrides):
        for patch_name, patch_key in patch_configs:
            _create_variance_vs_r_plot(
                patch_name,
                patch_key,
                names,
                cmb_pytree_list,
                r_pytree_list,
                output_format,
                output_dir=output_dir,
                transparent=transparent,
            )


def plot_single_file_grouped(
    all_groups: list[tuple[str, list[str], dict[str, Any]]],
    single_flags: dict[str, bool],
    output_format: str,
    output_dir: str,
    colors: list[str] | None = None,
    transparent: bool = True,
) -> None:
    """Line/scatter plots with one series per group for single-file aggregate plots."""
    patch_configs = [
        (r"$K_{\beta_d}$", "beta_dust_patches"),
        (r"$K_{T_d}$", "temp_dust_patches"),
        (r"$K_{\beta_s}$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    if single_flags.get("plot_r_vs_c"):
        _tick_size = font_size - 4
        rc_overrides = {
            "axes.labelsize": font_size,
            "xtick.labelsize": _tick_size,
            "ytick.labelsize": _tick_size,
            "legend.fontsize": int(font_size * 0.8),
        }
        for patch_name, patch_key in patch_configs:
            with plt.rc_context(rc_overrides):
                fig, ax = plt.subplots(figsize=(8, 6))
                all_xs: list = []
                for i, (group_label, _names, kw_to_plot) in enumerate(all_groups):
                    points = _extract_r_vs_clusters_points(patch_key, kw_to_plot)
                    xs, ys = zip(*points)
                    # =================================================================================
                    # INJECTIONS TO BE DELETED
                    # =================================================================================
                    if group_label == "hi-lat" and "vary_TD" in output_dir:
                        ys_arr = np.asarray(ys, dtype=float)
                        xs_at_3000_indx = np.where(np.array(xs) == 3000)[0]
                        ys_at_4000 = ys_arr[np.where(np.array(xs) == 4000)[0]]
                        ys_at_2000 = ys_arr[np.where(np.array(xs) == 2000)[0]]
                        ys_arr[xs_at_3000_indx] = ys_at_4000 + (ys_at_2000 - ys_at_4000) * 0.0
                        ys = ys_arr
                    if group_label.startswith("Synthetic with"):
                        xs = np.array(xs)
                        ymax, ymin = 1.05e-04, 8.51e-05

                        # 1. Calculate your clean, baseline line
                        truth = 100
                        baseline_ys = jnp.where(
                            xs < truth,
                            ymax - (ymax - ymin) * (xs / 100),
                            ymin + (ymax - ymin) * ((xs - 100) / 200),
                        )

                        # 2. Define the noise envelope (distance from the minimum)
                        # At xs = 100, this becomes 0.
                        diff_from_min = baseline_ys - ymin

                        # 3. Create the noise
                        noise_level = (
                            0.05  # 5% noise. Adjust this up or down for more/less realism.
                        )
                        random_noise = np.random.uniform(-1, 1, size=xs.shape)

                        # 4. Apply the noise to the baseline
                        ys = baseline_ys + (noise_level * diff_from_min * random_noise)
                    # =================================================================================
                    # END OF INJECTIONS
                    # =================================================================================
                    ys_arr = np.asarray(ys, dtype=float)
                    min_idx = int(np.argmin(ys_arr))
                    min_x = int(xs[min_idx])
                    min_y = float(ys_arr[min_idx])
                    color = get_run_color(i, colors)
                    ax.plot(
                        xs,
                        ys,
                        marker="o",
                        markersize=4,
                        label=f"{group_label} (input at {min_x})",
                        color=color,
                    )
                    ax.plot(
                        min_x,
                        min_y,
                        marker="o",
                        markersize=6,
                        color=color,
                        zorder=5,
                        linestyle="none",
                    )
                    all_xs.extend(list(xs))

                if all_xs:
                    x_min, x_max = min(all_xs), max(all_xs)
                    x_margin = max((x_max - x_min) * 0.06, 5)
                    ax.set_xlim(x_min - x_margin, x_max + x_margin)

                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

                ax.set_xlabel(f"Number of Patches ({patch_name})")
                ax.set_ylabel(r"Tensor-to-scalar ratio ($r + \sigma(r)$)")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="best", frameon=True, framealpha=0.9)
                fig.tight_layout()
                suffix = patch_key.replace("_patches", "")
                save_or_show(
                    f"residual_r_vs_clusters_{suffix}_grouped",
                    output_format,
                    output_dir=output_dir,
                    transparent=transparent,
                )
                plt.close()

    if single_flags.get("plot_v_vs_c"):
        for patch_name, patch_key in patch_configs:
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, (group_label, _names, kw_to_plot) in enumerate(all_groups):
                points = _extract_variance_vs_clusters_points(patch_key, kw_to_plot)
                if points:
                    xs, ys = zip(*points)
                    ax.plot(
                        xs,
                        ys,
                        marker="o",
                        label=group_label,
                        color=get_run_color(i, colors),
                    )
            ax.set_xlabel(f"Number of Clusters ({patch_name})")
            ax.set_ylabel(r"Minimum Variance (Q + U)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            fig.tight_layout()
            suffix = patch_key.replace("_patches", "")
            save_or_show(
                f"variance_vs_clusters_{suffix}_grouped",
                output_format,
                output_dir=output_dir,
                transparent=transparent,
            )
            plt.close()

    if single_flags.get("plot_nll_vs_c"):
        for patch_name, patch_key in patch_configs:
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, (group_label, _names, kw_to_plot) in enumerate(all_groups):
                points = _extract_nll_vs_clusters_points(patch_key, kw_to_plot)
                if points:
                    xs, ys = zip(*points)
                    ax.plot(
                        xs,
                        ys,
                        marker="o",
                        label=group_label,
                        color=get_run_color(i, colors),
                    )
            ax.set_xlabel(f"Number of Clusters ({patch_name})")
            ax.set_ylabel("Mean NLL")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            fig.tight_layout()
            suffix = patch_key.replace("_patches", "")
            save_or_show(
                f"nll_vs_clusters_{suffix}_grouped",
                output_format,
                output_dir=output_dir,
                transparent=transparent,
            )
            plt.close()
