from __future__ import annotations

import os
from typing import cast

import healpy as hp
import numpy as np
from furax.obs.stokes import Stokes
from jaxtyping import (
    Array,
    Float,
    Int,
)
from tqdm import tqdm

from .utils import expand_stokes


def compute_systematic_res(
    Wd_cmb: Stokes, fsky: float, ell_range: Int[Array, " L"]
) -> tuple[Float[Array, " L"], Float[Array, " 3 npix"]]:
    """Compute systematic residual BB power and map.

    Parameters
    ----------
    Wd_cmb : Stokes
        Foreground-only CMB reconstruction, i.e. W * d_fg.
    fsky : float
        Observed sky fraction used to debias the spectra.
    ell_range : Array
        Multipole array over which the spectrum is evaluated.

    Returns
    -------
    tuple
        (C_ell^syst, syst_map) where the spectrum is rescaled by f_sky and the
        map is a (3, Npix) array containing I, Q, U residuals.
    """
    Wd_cmb_expanded = expand_stokes(Wd_cmb)
    syst_map = np.stack([Wd_cmb_expanded.i, Wd_cmb_expanded.q, Wd_cmb_expanded.u], axis=0)

    cl_all = cast(Float[Array, " 6 L"], hp.anafast(syst_map))
    cl_bb = cl_all[2][ell_range]
    cl_bb_rescaled = cl_bb / fsky

    return cl_bb_rescaled, syst_map


def compute_statistical_res(
    s_hat: Stokes,
    s_true: Float[Array, " 3 npix"],
    fsky: float,
    ell_range: Int[Array, " L"],
    s_syst_map: Float[Array, " 3 npix"],
) -> tuple[Float[Array, " L"], Float[Array, " n_realizations 3 npix"]]:
    """Compute statistical residuals after subtracting systematic leakage.

    Parameters
    ----------
    s_hat : Stokes
        Reconstructed CMB map for all noise realizations.
    s_true : Array
        True input CMB map (I, Q, U stacked along axis=0).
    fsky : float
        Observed sky fraction.
    ell_range : Array
        Multipole range for spectral estimation.
    s_syst_map : Array
        Systematic residual map returned by :func:`compute_systematic_res`.

    Returns
    -------
    tuple
        (C_ell^stat, stat_maps) with averaged BB spectrum and residual maps per
        realization.
    """
    s_hat_expanded = expand_stokes(s_hat)
    s_hat_arr = np.stack([s_hat_expanded.i, s_hat_expanded.q, s_hat_expanded.u], axis=1)

    res = np.where(s_hat_arr == hp.UNSEEN, hp.UNSEEN, s_hat_arr - s_true[np.newaxis, ...])

    s_syst_arr = np.asarray(s_syst_map)
    res_stat = np.where(res == hp.UNSEEN, hp.UNSEEN, res - s_syst_arr[np.newaxis, ...])

    cl_list = []
    for i in tqdm(range(res_stat.shape[0]), desc="Computing Statistical BB Spectra"):
        cl = cast(Float[Array, " 6 L"], hp.anafast(res_stat[i]))
        cl_list.append(cl[2][ell_range])

    cl_mean = np.mean(cl_list, axis=0) / fsky

    return cl_mean, res_stat


def compute_total_res(
    s_hat: Stokes, s_true: Float[Array, " 3 npix"], fsky: float, ell_range: Int[Array, " L"]
) -> tuple[Float[Array, " L"], Float[Array, " n_realizations 3 npix"]]:
    """Compute total residual BB spectrum without separating components.

    Parameters
    ----------
    s_hat : Stokes
        Reconstructed CMB map for all noise realizations.
    s_true : Array
        True input CMB map (I, Q, U).
    fsky : float
        Observed sky fraction.
    ell_range : Array
        Multipole range for spectral estimation.

    Returns
    -------
    tuple
        (C_ell^res, residual_maps) where residual_maps has shape
        (n_realizations, 3, Npix).
    """
    s_hat_expanded = expand_stokes(s_hat)
    s_hat_arr = np.stack([s_hat_expanded.i, s_hat_expanded.q, s_hat_expanded.u], axis=1)

    if fsky >= 0.999 and os.environ.get("FURAX_CS_ALLOW_FULLSKY", "0") == "1":
        res = s_hat_arr
    else:
        res = np.where(s_hat_arr == hp.UNSEEN, hp.UNSEEN, s_hat_arr - s_true[np.newaxis, ...])

    cl_list = []
    for i in tqdm(range(res.shape[0]), desc="Computing Residual BB Spectra"):
        cl = cast(Float[Array, " 6 L"], hp.anafast(res[i]))
        cl_list.append(cl[2][ell_range])

    cl_mean = np.mean(cl_list, axis=0) / fsky

    return cl_mean, res


def compute_cl_bb_sum(
    cmb_out: Stokes, fsky: float, ell_range: Int[Array, " L"]
) -> Float[Array, " n_realizations"]:
    """Compute ∑ C_ell^{BB} across realizations for the recovered CMB."""
    cmb_out_expanded = expand_stokes(cmb_out)
    cmb_out_arr = np.stack([cmb_out_expanded.i, cmb_out_expanded.q, cmb_out_expanded.u], axis=1)

    cl_list = []
    for i in tqdm(range(cmb_out_arr.shape[0]), desc="Computing CL_BB_SUM"):
        cl = cast(Float[Array, " 6 L"], hp.anafast(cmb_out_arr[i]))
        cl_list.append(cl[2][ell_range])

    CL_BB_SUM = np.sum(cl_list, axis=1) / fsky
    return CL_BB_SUM


def compute_cl_obs_bb(
    cl_total_res: Float[Array, " L"], cl_bb_lens: Float[Array, " L"]
) -> Float[Array, " L"]:
    """Combine residual and lensing spectra to form observed BB power."""
    return cl_total_res + cl_bb_lens


def compute_cl_true_bb(
    s: Float[Array, " 3 npix"], ell_range: Int[Array, " L"]
) -> Float[Array, " L"]:
    """Compute the true sky BB spectrum over the requested ell range."""
    cl = cast(Float[Array, " 6 L"], hp.anafast(s))

    return cl[2][ell_range]
