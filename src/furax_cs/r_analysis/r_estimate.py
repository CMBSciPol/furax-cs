from __future__ import annotations

import warnings
from typing import cast

import camb
import healpy as hp
import numpy as np
from furax.obs.stokes import Stokes, StokesIQU, StokesQU
from jaxtyping import (
    Array,
    Float,
    Int,
)
from tqdm import tqdm

from ..logging_utils import error, info, success
from .utils import expand_stokes


def _log_likelihood(
    r: Float[Array, " R"],
    ell_range: Int[Array, " L"],
    cl_obs: Float[Array, " L"],
    cl_bb_r1: Float[Array, " L"],
    cl_bb_lens: Float[Array, " L"],
    cl_noise: Float[Array, " L"],
    f_sky: float,
) -> Float[Array, " R"]:
    """Gaussian pseudo-Cℓ log-likelihood for the tensor-to-scalar ratio."""
    cl_model = r.reshape(-1, 1) * cl_bb_r1 + cl_bb_lens + cl_noise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        term = (2 * ell_range + 1) * (cl_obs / cl_model + np.log(cl_model))
    return -0.5 * f_sky * np.sum(term, axis=1)


def _get_camb_templates(
    nside: int,
) -> tuple[Int[Array, " L"], Float[Array, " L"], Float[Array, " L"]]:
    """Generate BB templates for r=1 and lensing using CAMB.

    Uses the same cosmological parameters as generate_custom_cmb in
    data/generate_maps.py to ensure consistency between map generation
    and template fitting.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    tuple
        (ell_range, cl_bb_r1, cl_bb_lens) where:
        - ell_range: multipole range array
        - cl_bb_r1: BB spectrum template for r=1
        - cl_bb_lens: lensing BB spectrum template
    """
    # Use same cosmology as generate_custom_cmb for consistency
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=1)
    pars.WantTensors = True
    pars.set_for_lmax(1024, lens_potential_accuracy=1)

    results = camb.get_results(pars)
    # Use raw_cl=True to get C_ell directly (not D_ell with ell(ell+1)/(2pi) factor)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=1024, raw_cl=True)
    cl_bb_r1_full, cl_bb_total = powers["tensor"][:, 2], powers["total"][:, 2]
    cl_bb_lens_full = cl_bb_total - cl_bb_r1_full

    ell_min, ell_max = 2, nside * 2 + 2
    ell_range = np.arange(ell_min, ell_max)
    # No need to divide by coeff since raw_cl=True gives C_ell directly
    cl_bb_r1 = cl_bb_r1_full[ell_range]
    cl_bb_lens = cl_bb_lens_full[ell_range]

    return ell_range, cl_bb_r1, cl_bb_lens


def estimate_r(
    cl: Float[Array, " L"],
    nside: int,
    cl_noise: Float[Array, " L"],
    f_sky: float,
    is_cl_obs: bool = False,
    max_point: float = 0.005,
) -> tuple[
    float,
    float,
    float,
    Float[Array, " R"],
    Float[Array, " R"],
    Int[Array, " L"],
    Float[Array, " L"],
    Float[Array, " L"],
    Float[Array, " L"],
]:
    """Estimate r and 68% uncertainties from a grid-evaluated likelihood.

    Args:
        cl: Input BB power spectrum. Interpretation depends on is_cl_obs:
            - If is_cl_obs=False (default): residual spectrum (lensing will be added).
            - If is_cl_obs=True: observed spectrum (already contains lensing).
        nside: HEALPix resolution for template generation.
        cl_noise: Noise power spectrum (e.g., statistical residuals).
        f_sky: Sky fraction.
        is_cl_obs: If True, input spectrum is observed (contains lensing).
            If False, input is residual and lensing is added internally. Defaults to False.
        max_point: Maximum absolute value for r grid. Defaults to 0.005.

    Returns:
        A tuple containing:
            - **r_best**: Best-fit r value.
            - **sigma_r_neg**: Negative uncertainty (68% CL).
            - **sigma_r_pos**: Positive uncertainty (68% CL).
            - **r_grid**: Grid of r values used for likelihood.
            - **L_vals**: Likelihood values.
            - **ell_range**: Multipole range.
            - **cl_bb_r1**: Template for r=1.
            - **cl_bb_lens**: Lensing template.
            - **cl_obs**: Observed spectrum used in likelihood.
    """
    # Generate CAMB templates internally
    ell_range, cl_bb_r1, cl_bb_lens = _get_camb_templates(nside)

    # Compute observed BB spectrum
    if is_cl_obs:
        cl_obs = cl  # Input already observed (contains lensing)
    else:
        cl_obs = cl + cl_bb_lens  # Input is residual, add lensing

    r_grid = np.linspace(-max_point, max_point, 1000)
    print(f"cl_noise is {cl_noise}")
    logL = _log_likelihood(r_grid, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky)
    finite_logL = logL[np.isfinite(logL)]
    finite_r = r_grid[np.isfinite(logL)]
    L = np.exp(finite_logL - np.max(finite_logL))
    r_best = float(finite_r[np.argmax(L)])

    rs_pos, L_pos = finite_r[finite_r > r_best], L[finite_r > r_best]
    rs_neg, L_neg = finite_r[finite_r < r_best], L[finite_r < r_best]
    cum_pos = np.cumsum(L_pos) / np.sum(L_pos)
    cum_neg = np.cumsum(L_neg[::-1]) / np.sum(L_neg)

    sigma_pos = float(rs_pos[np.argmin(np.abs(cum_pos - 0.68))] - r_best if len(rs_pos) > 0 else 0)
    sigma_neg = float(
        r_best - rs_neg[::-1][np.argmin(np.abs(cum_neg - 0.68))] if len(rs_neg) > 0 else 0
    )

    return (
        r_best,
        sigma_neg,
        sigma_pos,
        finite_r,
        L,
        ell_range,
        cl_bb_r1,
        cl_bb_lens,
        cl_obs,
    )


def estimate_r_from_maps(
    cmb: Stokes,
    cmb_hat: Stokes | None = None,
    syst_map: Stokes | None = None,
    nside: int | None = None,
    max_point: float = 0.005,
) -> tuple[
    float,
    float,
    float,
    Float[Array, " R"],
    Float[Array, " R"],
    Int[Array, " L"],
    Float[Array, " L"],
    Float[Array, " L"],
    Float[Array, " L"],
]:
    """Estimate r from CMB maps with automatic spectrum computation and masking detection.

    Args:
        cmb: True CMB map.
        cmb_hat: Reconstructed CMB maps from noise realizations. Defaults to None.
        syst_map: Systematic residual map. Defaults to None.
        nside: HEALPix resolution. If not provided, inferred from map size.
        max_point: Maximum absolute value for r grid. Defaults to 0.005.

    Returns:
        A tuple containing:
            - **r_best**: Best-fit r value.
            - **sigma_r_neg**: Negative uncertainty.
            - **sigma_r_pos**: Positive uncertainty.
            - **r_grid**: Grid of r values.
            - **L_vals**: Likelihood values.
            - **ell_range**: Multipole range.
            - **cl_bb_r1**: Template for r=1.
            - **cl_bb_lens**: Lensing template.
            - **cl_obs**: Observed spectrum.
    """

    def _to_numpy(stokes: Stokes) -> Float[Array, ...]:
        """Convert Stokes object to expanded IQU numpy array."""
        expanded = expand_stokes(stokes)
        return np.vstack([expanded.i, expanded.q, expanded.u])

    # 1. Expand Stokes format
    cmb_expanded = _to_numpy(cmb)

    # 2. Infer nside (if not provided)
    if nside is None:
        npix = cmb_expanded.shape[-1]
        nside = hp.npix2nside(npix)

    # 3. Generate CAMB templates
    ell_range, cl_bb_r1, cl_bb_lens = _get_camb_templates(nside)

    # 4. Compute BB spectrum from CMB map
    cl_all = cast(Float[Array, " 6 L"], hp.anafast(cmb_expanded))
    cl_cmb = cl_all[2][ell_range]  # Extract BB component

    # 5. Detect masking and compute f_sky

    q_map = cmb_expanded[1]
    has_mask = bool(np.any(q_map == hp.UNSEEN))
    if has_mask:
        npix_total = q_map.shape[0]
        # Count valid pixels in Q component
        npix_valid = np.sum(q_map != hp.UNSEEN)
        fsky = float(npix_valid / npix_total)
    else:
        fsky = 1.0

    # 6. Determine mode (is_cl_obs)
    is_cl_obs = not has_mask

    info(f"Mask detected: {has_mask}, f_sky = {fsky:.3f}, is_cl_obs = {is_cl_obs}")

    # 7. Expand auxiliary maps (if provided)
    cmb_hat_expanded = _to_numpy(cmb_hat) if cmb_hat is not None else None
    syst_map_expanded = _to_numpy(syst_map) if syst_map is not None else None

    # 8. Compute noise spectrum
    if cmb_hat_expanded is not None and syst_map_expanded is not None:
        # Statistical residuals
        res = cmb_hat_expanded - cmb_expanded[np.newaxis, ...]
        res_stat = res - syst_map_expanded[np.newaxis, ...]

        cl_list = []
        for i in tqdm(range(res_stat.shape[0]), desc="Computing residual spectra"):
            cl = cast(Float[Array, " 6 L"], hp.anafast(res_stat[i]))
            cl_list.append(cl[2][ell_range])
        cl_noise = np.mean(cl_list, axis=0) / fsky

        # If masked, need total residual for observed spectrum
        if has_mask:
            cl_list_total = []
            for i in tqdm(
                range(cmb_hat_expanded.shape[0]), desc="Computing total residual spectra"
            ):
                res_total = cmb_hat_expanded[i] - cmb_expanded
                cl = cast(Float[Array, " 6 L"], hp.anafast(res_total))
                cl_list_total.append(cl[2][ell_range])
            cl_total_res = np.mean(cl_list_total, axis=0) / fsky
            cl_input = cl_total_res
        else:
            cl_input = cl_cmb
    else:
        # No residual analysis: assume perfect reconstruction
        if has_mask:
            raise ValueError(
                "Cannot estimate r from masked maps without residual analysis. "
                "Provide --cmb-hat and --syst arguments."
            )
        cl_noise = np.zeros_like(cl_cmb)
        cl_input = cl_cmb

    # 9. Call estimate_r
    return estimate_r(cl_input, nside, cl_noise, fsky, is_cl_obs=is_cl_obs, max_point=max_point)


def run_estimate(
    cmb_path: str,
    cmb_hat_path: str | None = None,
    syst_path: str | None = None,
    fsky: float | None = None,
    nside: int | None = None,
    output_path: str | None = None,
    output_format: str = "png",
) -> None:
    """Entry point for 'estimate' subcommand.

    Estimates the tensor-to-scalar ratio r from input spectra or maps,
    following the pattern of run_validate.

    Args:
        cmb_path: Path to CMB data (.npy file).
        cmb_hat_path: Optional path to reconstructed CMB maps. Defaults to None.
        syst_path: Optional path to systematic residual map. Defaults to None.
        fsky: Sky fraction (required for spectrum input). Defaults to None.
        nside: HEALPix resolution (inferred from map if not provided). Defaults to None.
        output_path: Optional path to save results as .npz file. Defaults to None.
        output_format: Output format for plot: "png", "pdf", or "show". Defaults to "png".
    """
    from .plotting import plot_r_estimator

    # Load input data
    cmb_data = np.load(cmb_path)
    cmb_hat_raw = np.load(cmb_hat_path) if cmb_hat_path else None
    # Determine if input is spectrum (1D) or map (2D/3D)
    is_spectrum = cmb_data.ndim == 1

    if is_spectrum:
        # Spectrum mode
        if fsky is None:
            error("--fsky is required when input is a power spectrum")
            return
        if nside is None:
            error("--nside is required when input is a power spectrum")
            return

        info(f"Running r estimation from power spectrum (nside={nside}, fsky={fsky})")

        cl_noise = cmb_hat_raw
        cl_total_res = cmb_data
        (
            r_best,
            sigma_neg,
            sigma_pos,
            r_grid,
            L,
            ell_range,
            cl_bb_r1,
            cl_bb_lens,
            cl_obs,
        ) = estimate_r(cl_total_res, nside, cl_noise, fsky, is_cl_obs=True)

    else:
        info(f"Running r estimation from maps (shape={cmb_data.shape})")

        # Load optional auxiliary maps
        syst_map_raw = np.load(syst_path) if syst_path else None

        # Helper to wrap numpy arrays to Stokes
        def _wrap_stokes(arr: np.ndarray | None) -> Stokes | None:
            if arr is None:
                return None
            if arr.ndim == 2:
                if arr.shape[0] == 2:
                    return StokesQU(arr[0], arr[1])
                if arr.shape[0] == 3:
                    return StokesIQU(arr[0], arr[1], arr[2])
            if arr.ndim == 3:
                # Batched
                if arr.shape[1] == 2:
                    return StokesQU(arr[:, 0], arr[:, 1])
                if arr.shape[1] == 3:
                    return StokesIQU(arr[:, 0], arr[:, 1], arr[:, 2])
            return None

        # Estimate from maps
        (
            r_best,
            sigma_neg,
            sigma_pos,
            r_grid,
            L,
            ell_range,
            cl_bb_r1,
            cl_bb_lens,
            cl_obs,
        ) = estimate_r_from_maps(
            _wrap_stokes(cmb_data), _wrap_stokes(cmb_hat_raw), _wrap_stokes(syst_map_raw), nside
        )

    # Plot results using plot_r_estimator
    name = "r_estimate"
    plot_r_estimator(
        name=name,
        r_best=r_best,
        sigma_r_neg=sigma_neg,
        sigma_r_pos=sigma_pos,
        r_grid=r_grid,
        L_vals=L,
        output_format=output_format,
    )

    success(f"r estimation complete: r = {r_best:+.6f} +{sigma_pos:.6f} -{sigma_neg:.6f}")

    # Save results if requested
    if output_path:
        np.savez(
            output_path,
            r_best=r_best,
            sigma_neg=sigma_neg,
            sigma_pos=sigma_pos,
            r_grid=r_grid,
            likelihood=L,
            ell_range=ell_range,
            cl_bb_r1=cl_bb_r1,
            cl_bb_lens=cl_bb_lens,
            cl_obs=cl_obs,
        )
        success(f"Results saved to: {output_path}")
