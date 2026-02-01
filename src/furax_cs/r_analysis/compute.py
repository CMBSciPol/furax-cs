from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Any, Union

import healpy as hp
import jax.numpy as jnp
import numpy as np
from furax._instruments.sky import FGBusterInstrument, get_sky
from furax.obs.stokes import Stokes
from jax_healpy.clustering import combine_masks
from jaxtyping import Array
from tqdm import tqdm

from ..logging_utils import format_residual_flags, hint, info, warning
from .caching import atomic_save_results, compute_w
from .r_estimate import estimate_r
from .residuals import (
    compute_cl_bb_sum,
    compute_cl_true_bb,
    compute_statistical_res,
    compute_systematic_res,
    compute_total_res,
)
from .utils import (
    expand_stokes,
    index_run_data,
    params_to_maps,
)


def get_compute_flags(args: argparse.Namespace, snapshot_mode: bool = False) -> dict[str, bool]:
    """Determine what computations are needed based on args.

    Args:
        args: Parsed command-line arguments with visualization toggles.
        snapshot_mode: If True, all flags are set to True for complete computation. Defaults to False.

    Returns:
        Dictionary with computation flags:
        - needs_residual_maps: bool
        - needs_residual_spectra: bool
        - needs_r_estimation: bool
        - need_patch_maps: bool
        - need_validation_curves: bool
        - compute_syst: bool
        - compute_stat: bool
        - compute_total: bool
    """
    if snapshot_mode:
        return {
            "needs_residual_maps": True,
            "needs_residual_spectra": True,
            "needs_r_estimation": True,
            "need_patch_maps": True,
            "need_validation_curves": True,
            "compute_syst": True,
            "compute_stat": True,
            "compute_total": True,
        }

    # Extract flags from args with safe getattr for optional attributes
    plot_cmb_recon = getattr(args, "plot_cmb_recon", False)
    plot_systematic_maps = getattr(args, "plot_systematic_maps", False)
    plot_statistical_maps = getattr(args, "plot_statistical_maps", False)
    plot_all = getattr(args, "plot_all", False)
    plot_cl_spectra = getattr(args, "plot_cl_spectra", False)
    plot_all_spectra = getattr(args, "plot_all_spectra", False)
    plot_r_estimation = getattr(args, "plot_r_estimation", False)
    plot_all_r_estimation = getattr(args, "plot_all_r_estimation", False)
    plot_r_vs_c = getattr(args, "plot_r_vs_c", False)
    plot_r_vs_v = getattr(args, "plot_r_vs_v", False)
    plot_illustrations = getattr(args, "plot_illustrations", False)
    plot_params = getattr(args, "plot_params", False)
    plot_patches = getattr(args, "plot_patches", False)
    plot_validation_curves = getattr(args, "plot_validation_curves", False)
    plot_all_metrics = getattr(args, "plot_all_metrics", False)
    plot_all_params_residuals = getattr(args, "plot_all_params_residuals", False)
    plot_all_histograms = getattr(args, "plot_all_histograms", False)

    needs_residual_maps = (
        plot_cmb_recon or plot_systematic_maps or plot_statistical_maps or plot_all
    )
    needs_residual_spectra = plot_cl_spectra or plot_all_spectra or plot_all
    needs_r_estimation = (
        plot_r_estimation
        or plot_all_r_estimation
        or plot_r_vs_c
        or plot_r_vs_v
        or plot_illustrations
        or plot_all
    )
    need_patch_maps = (
        plot_illustrations
        or plot_params
        or plot_patches
        or plot_all
        or plot_all_params_residuals
        or plot_all_histograms
    )
    need_validation_curves = plot_validation_curves or plot_all or plot_all_metrics

    compute_syst = needs_residual_spectra or needs_residual_maps
    compute_stat = compute_syst or needs_r_estimation
    compute_total = needs_r_estimation

    return {
        "needs_residual_maps": needs_residual_maps,
        "needs_residual_spectra": needs_residual_spectra,
        "needs_r_estimation": needs_r_estimation,
        "need_patch_maps": need_patch_maps,
        "need_validation_curves": need_validation_curves,
        "compute_syst": compute_syst,
        "compute_stat": compute_stat,
        "compute_total": compute_total,
    }


def _normalize_indices(
    indices: Union[int, tuple[int, int], list[int]],
) -> list[int]:
    """Convert indices spec to list of ints.

    Args:
        indices: Index specification:
            - int: single index.
            - tuple (start, end): inclusive range.
            - list: explicit list of indices.

    Returns:
        List of indices to process.
    """
    if isinstance(indices, int):
        return [indices]
    elif isinstance(indices, tuple) and len(indices) == 2:
        return list(range(indices[0], indices[1] + 1))
    return list(indices)


def _compute_single_folder(
    folder: str,
    run_index: int,
    nside: int,
    instrument: FGBusterInstrument,
    flags: dict[str, bool],
    full_results: dict[str, Array] | None = None,
    max_iter: int = 100,
    solver_name: str = "optax_lbfgs",
) -> dict[str, Any] | None:
    """Process a single result folder for a specific run index.

    Args:
        folder: Path to the result folder.
        run_index: Index of the noise realization to process.
        nside: HEALPix resolution parameter.
        instrument: Instrument configuration object.
        flags: Computation flags from `get_compute_flags`.
        full_results: Pre-loaded results.npz contents to avoid reloading. Defaults to None.
        max_iter: Maximum iterations for W computation if not cached. Defaults to 100.
        solver_name: Solver name for W computation. Defaults to "optax_lbfgs".

    Returns:
        Dictionary with computed data for this folder/index, or None if failed.
        Keys include: cmb_recon, cmb_true, mask, indices, NLL, wd, params, patches,
    """
    # Load data
    results_path = f"{folder}/results.npz"
    best_params_path = f"{folder}/best_params.npz"
    mask_path = f"{folder}/mask.npy"

    try:
        if full_results is None:
            full_results = dict(np.load(results_path))
        best_params = dict(np.load(best_params_path))
        mask = np.load(mask_path)
    except (FileNotFoundError, OSError) as e:
        warning(f"Failed to load data for {folder}: {e}")
        return None

    # Check bounds
    first_key = next(iter(full_results.keys()))
    max_index = len(full_results[first_key]) - 1
    if run_index > max_index:
        warning(f"Index {run_index} out of bounds (max: {max_index}) for {folder}. Skipping.")
        return None

    # Slice the specific run data
    run_data = index_run_data(full_results, run_index)
    (indices,) = jnp.where(mask == 1)

    # Extract CMB data
    cmb_true = Stokes.from_stokes(Q=best_params["I_CMB"][0], U=best_params["I_CMB"][1])
    cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])
    NLL = run_data["NLL"]

    # Compute W_D_FG if needed for systematic residuals
    wd = None
    if flags["compute_syst"]:
        cache_key = f"W_D_FG_{run_index}"
        if cache_key in full_results:
            cached_w = full_results[cache_key]
            wd = Stokes.from_stokes(Q=cached_w[0], U=cached_w[1])
        else:
            hint(f"Systematics not cached for index {run_index}. Computing now and caching...")
            fg_map = Stokes.from_stokes(
                Q=best_params["I_D_NOCMB"][:, 0],
                U=best_params["I_D_NOCMB"][:, 1],
            )
            patches = {
                "beta_dust_patches": run_data["beta_dust_patches"],
                "beta_pl_patches": run_data["beta_pl_patches"],
                "temp_dust_patches": run_data["temp_dust_patches"],
            }
            wd = compute_w(
                nu=instrument.frequency,
                d=fg_map,
                patches=patches,
                max_iter=max_iter,
                solver_name=solver_name,
            )
            # Persist to results.npz for future runs
            W_numpy = np.stack([wd.q, wd.u], axis=0)
            full_results[cache_key] = W_numpy
            atomic_save_results(f"{folder}/results.npz", full_results)

    # Extract params and patches if needed
    params = None
    patches = None
    if flags["need_patch_maps"]:
        patches = {
            "beta_dust_patches": run_data["beta_dust_patches"],
            "temp_dust_patches": run_data["temp_dust_patches"],
            "beta_pl_patches": run_data["beta_pl_patches"],
        }
        params = {
            "beta_dust": run_data.get("beta_dust"),
            "temp_dust": run_data.get("temp_dust"),
            "beta_pl": run_data.get("beta_pl"),
        }

    # Extract validation curves if needed
    return {
        "cmb_recon": cmb_recon,
        "cmb_true": cmb_true,
        "mask": mask,
        "indices": indices,
        "NLL": NLL,
        "wd": wd,
        "params": params,
        "patches": patches,
        "run_data": run_data,
        "best_params": best_params,
    }


def compute_group(
    title: str,
    folders: list[str],
    run_indices: Union[int, tuple[int, int], list[int]],
    nside: int,
    instrument: FGBusterInstrument,
    flags: dict[str, bool],
    solver_name: str,
    max_iter: int = 100,
    noise_selection: str = "min-value",
    sky_tag: str = "c1d0s0",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Process a group of folders for given run indices.

    Args:
        title: Identifier for this run group (for logging).
        folders: List of result folder paths.
        run_indices: Index specification for run indices (int, tuple, or list).
        nside: HEALPix resolution parameter.
        instrument: Instrument configuration object.
        flags: Computation flags from `get_compute_flags`.
        solver_name: Solver name for W computation.
        max_iter: Maximum iterations for W computation if not cached. Defaults to 100.
        noise_selection: Noise selection strategy for parameter maps. Defaults to "min-value".
        sky_tag: Sky model tag to use for true parameters. Defaults to "c1d0s0".

    Returns:
        A tuple (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data) if successful,
        or None if no valid data found.
    """
    if not folders:
        warning(f"No folders provided for '{title}'")
        return None

    indices_spec = _normalize_indices(run_indices)

    # Collect data from all folders/indices
    cmb_recons, cmb_maps, masks = [], [], []
    indices_list, w_d_list, NLLs = [], [], []
    params_list, patches_list, raw_params_list = [], [], []

    previous_mask_size = {
        "beta_dust_patches": 0,
        "temp_dust_patches": 0,
        "beta_pl_patches": 0,
    }

    info(
        format_residual_flags(flags["compute_syst"], flags["compute_stat"], flags["compute_total"])
    )

    for folder in tqdm(folders, desc=f"  Folders for {title}", leave=False, unit="folder"):
        # Load results once per folder
        results_path = f"{folder}/results.npz"
        try:
            full_results = dict(np.load(results_path))
        except (FileNotFoundError, OSError) as e:
            warning(f"Failed to load {results_path}: {e}")
            continue

        for run_index in indices_spec:
            result = _compute_single_folder(
                folder,
                run_index,
                nside,
                instrument,
                flags,
                full_results=full_results,
                max_iter=max_iter,
                solver_name=solver_name,
            )
            if result is None:
                continue

            cmb_recons.append(result["cmb_recon"])
            cmb_maps.append(result["cmb_true"])
            masks.append(result["mask"])
            indices_list.append(result["indices"])
            NLLs.append(result["NLL"])

            if result["wd"] is not None:
                w_d_list.append(result["wd"])

            if result["params"] is not None and result["patches"] is not None:
                params, raw_params, patches, previous_mask_size = params_to_maps(
                    result["run_data"], previous_mask_size, noise_selection=noise_selection
                )
                params_list.append(params)
                patches_list.append(patches)
                raw_params_list.append(raw_params)

    if len(masks) == 0:
        warning(f"No valid data found for '{title}'. Skipping this run.")
        return None

    # Aggregate results
    full_mask = np.logical_or.reduce(masks)
    f_sky = float(full_mask.sum() / len(full_mask))

    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside, axis=1)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)

    wd = None
    if flags["compute_syst"] and len(w_d_list) > 0:
        wd = combine_masks(w_d_list, indices_list, nside)

    NLL_summed = np.sum(NLLs, axis=0)

    # Params and patches maps
    true_params = None
    if flags["need_patch_maps"] and params_list:
        params_map = combine_masks(params_list, indices_list, nside)
        patches_map = combine_masks(patches_list, indices_list, nside)

        # Get True Parameters
        sky = get_sky(nside=nside, tag=sky_tag)
        # Assuming standard FURAX sky components: CMB (0), Dust (1), Synchrotron (2)
        true_params = {
            "beta_dust": sky.components[1].mbb_index.value,
            "temp_dust": sky.components[1].mbb_temperature.value,
            "beta_pl": sky.components[2].pl_index.value,
        }
        # Mask true params
        true_params = {k: np.where(full_mask, v, hp.UNSEEN) for k, v in true_params.items()}

    else:
        params_map = None
        patches_map = {
            "beta_dust_patches": np.zeros(hp.nside2npix(nside)),
            "temp_dust_patches": np.zeros(hp.nside2npix(nside)),
            "beta_pl_patches": np.zeros(hp.nside2npix(nside)),
        }

    # Compute ell_range for residual spectra
    ell_range = np.arange(2, nside * 2 + 2)

    # Get true sky for residuals from the saved CMB in best_params (cmb_stokes)
    cmb_stokes_expanded = expand_stokes(cmb_stokes)
    s_true = np.stack(
        [cmb_stokes_expanded.i, cmb_stokes_expanded.q, cmb_stokes_expanded.u],
        axis=0,
    )

    # Compute residuals
    cl_syst_res, syst_map, cl_stat_res, stat_maps = None, None, None, None

    if flags["compute_syst"] and wd is not None:
        cl_syst_res, syst_map = compute_systematic_res(wd, f_sky, ell_range)
        info(f"Systematic residuals: min={np.min(cl_syst_res):.2e}, max={np.max(cl_syst_res):.2e}")

    if flags["compute_stat"] and flags["compute_syst"] and syst_map is not None:
        cl_stat_res, stat_maps = compute_statistical_res(
            combined_cmb_recon, s_true, f_sky, ell_range, syst_map
        )
        info(f"Statistical residuals: min={np.min(cl_stat_res):.2e}, max={np.max(cl_stat_res):.2e}")

    cl_total_res = None
    is_full_sky = f_sky >= 0.999 and os.environ.get("FURAX_CS_ALLOW_FULLSKY", "0") == "1"
    info(f"Effective f_sky: {f_sky:.4f} (Full sky: {is_full_sky})")

    if flags["compute_total"]:
        if not is_full_sky and cl_syst_res is not None and cl_stat_res is not None:
            cl_total_res = cl_syst_res + cl_stat_res
        else:
            cl_total_res, _ = compute_total_res(combined_cmb_recon, s_true, f_sky, ell_range)
        info(f"Total residuals: min={np.min(cl_total_res):.2e}, max={np.max(cl_total_res):.2e}")

    # True Cl
    cl_true = compute_cl_true_bb(s_true, ell_range)

    # Cl BB sum for illustrations
    if flags["need_patch_maps"] or flags["need_validation_curves"]:
        cl_bb_sum = compute_cl_bb_sum(combined_cmb_recon, f_sky, ell_range)
    else:
        cl_bb_sum = None

    # R estimation
    r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = (
        None,
        None,
        None,
        None,
        None,
    )
    cl_bb_obs, cl_bb_r1, cl_bb_lens = None, None, None
    if flags["compute_total"] and flags["needs_r_estimation"] and cl_total_res is not None:
        if is_full_sky:
            cl_for_r = cl_true
            noise_for_r = np.zeros_like(ell_range)
        else:
            cl_for_r = cl_total_res
            noise_for_r = cl_stat_res if cl_stat_res is not None else np.zeros_like(ell_range)
        (
            r_best,
            sigma_r_neg,
            sigma_r_pos,
            r_grid,
            L_vals,
            _,
            cl_bb_r1,
            cl_bb_lens,
            cl_bb_obs,
        ) = estimate_r(cl_for_r, nside, noise_for_r, f_sky, is_cl_obs=is_full_sky)
        info(f"r estimation: {r_best:.4f} +{sigma_r_pos:.4f} -{sigma_r_neg:.4f}")

    # Build output pytrees
    cmb_pytree = {
        "cmb": cmb_stokes,
        "cmb_recon": combined_cmb_recon,
        "patches_map": patches_map,
        "cl_bb_sum": cl_bb_sum,
        "nll_summed": NLL_summed,
    }
    cl_pytree = {
        "cl_bb_r1": cl_bb_r1,
        "cl_true": cl_true,
        "ell_range": ell_range,
        "cl_bb_obs": cl_bb_obs,
        "cl_bb_lens": cl_bb_lens,
        "cl_syst_res": cl_syst_res,
        "cl_total_res": cl_total_res,
        "cl_stat_res": cl_stat_res,
    }
    r_pytree = {
        "r_best": r_best,
        "sigma_r_neg": sigma_r_neg,
        "sigma_r_pos": sigma_r_pos,
        "r_grid": r_grid,
        "L_vals": L_vals,
    }
    residual_pytree = {
        "syst_map": syst_map,
        "stat_maps": stat_maps,
    }
    plotting_data = {
        "params_map": params_map,
        "true_params": true_params,
        "all_params": raw_params_list,
    }

    return (
        cmb_pytree,
        cl_pytree,
        r_pytree,
        residual_pytree,
        plotting_data,
    )


def compute_all(
    matched_results: dict[str, tuple[list[str], list[int], str]],
    nside: int,
    instrument: FGBusterInstrument,
    flags: dict[str, bool],
    max_iter: int,
    solver_name: str,
    titles: dict[str, str] | None = None,
    noise_selection: str = "min-value",
    sky_tag: str = "c1d0s0",
) -> OrderedDict[
    str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
]:
    """Compute results for all matched run groups.

    Args:
        matched_results: Dictionary of format `{kw: (folders_list, run_indices, root)}`.
        nside: HEALPix resolution parameter.
        instrument: Instrument configuration object.
        flags: Computation flags from `get_compute_flags`.
        max_iter: Maximum iterations for W computation if not cached.
        solver_name: Solver name for W computation.
        titles: Dictionary mapping keywords to title strings. Defaults to None.
        noise_selection: Noise selection strategy for parameter maps. Defaults to "min-value".
        sky_tag: Sky model tag to use for true parameters. Defaults to "c1d0s0".

    Returns:
        OrderedDict keyed by title with values:
        (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data)
    """
    if titles is None:
        titles = {}

    results: OrderedDict[
        str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = OrderedDict()

    for kw, (folders, indices, root) in tqdm(
        matched_results.items(), desc="Processing run groups", unit="group"
    ):
        title = titles.get(kw, kw)
        info(f"Computing results for '{title}' ({len(folders)} folders)")

        result = compute_group(
            title=title,
            folders=folders,
            run_indices=indices,
            nside=nside,
            instrument=instrument,
            flags=flags,
            max_iter=max_iter,
            solver_name=solver_name,
            noise_selection=noise_selection,
            sky_tag=sky_tag,
        )

        if result is not None:
            results[title] = result

    return results
