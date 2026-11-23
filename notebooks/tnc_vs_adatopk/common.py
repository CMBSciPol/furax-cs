"""Shared code for the FGBuster TNC versus FURAX AdaTopK comparison (notebook 07).

Both optimisers see the same sky maps, patches, noise realisation, start point and bounds,
all defined here.
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import platform
import time
from functools import partial
from importlib.resources import files as pkg_files
from pathlib import Path

# Deterministic GPU reductions: without this flag the scatter-add atomics change the rounding
# from run to run, so the same fit on the same data stops at different points. XLA reads the
# flag when the backend starts, so this module must be imported before any jax computation.
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()

import healpy as hp  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pysm3.units as u  # noqa: E402

jax.config.update("jax_enable_x64", True)

from furax import Config, HomothetyOperator  # noqa: E402
from furax.linalg import BlockDiagonalSolver  # noqa: E402
from furax.obs import negative_log_likelihood, sky_signal  # noqa: E402
from furax.obs._likelihoods import _spectral_likelihood_core  # noqa: E402
from furax.obs.stokes import Stokes  # noqa: E402
from jax_healpy.clustering import normalize_by_first_occurrence  # noqa: E402

from furax_cs import (  # noqa: E402
    bin_parameter_map,
    generate_noise_operator,
    get_instrument,
    get_mask,
    load_cmb_map,
    load_fg_map,
    load_from_cache,
    minimize,
    save_cmb_map,
    save_fg_map,
    save_to_cache,
)
from furax_cs.r_analysis.r_estimate import estimate_r  # noqa: E402
from furax_cs.r_analysis.residuals import (  # noqa: E402
    compute_statistical_res,
    compute_systematic_res,
    compute_total_res,
)

# ---------------------------------------------------------------------------
# Problem definition, identical for both optimisers
# ---------------------------------------------------------------------------
NSIDE = 64
NPIX = 12 * NSIDE**2
MASK_NAME = "ALL-GALACTIC"
INSTRUMENT = "LiteBIRD"
DUST_NU0 = 160.0
SYNC_NU0 = 20.0
NOISE_RATIO = 1.0

PARAMS = ("beta_dust", "temp_dust", "beta_pl")
PATCH_KEY = {p: f"{p}_patches" for p in PARAMS}
START = {"beta_dust": 1.54, "temp_dust": 20.0, "beta_pl": -3.0}
LOWER = {"beta_dust": 0.5, "temp_dust": 10.0, "beta_pl": -7.0}
UPPER = {"beta_dust": 3.0, "temp_dust": 40.0, "beta_pl": -0.5}

# AdaTopK: solver, tolerances, iteration cap and cooldown of slurm/runners/10-kmeans_true_clusters.sh.
ADATOPK_SOLVER = "ADABK0"
ADATOPK_MAX_ITER = 10000
ADATOPK_ATOL = 1e-18
ADATOPK_RTOL = 1e-18
ADATOPK_OPTIONS = {"cooldown": 50}

# TNC: settings of scripts/fgbuster_model.py (run_fgbuster_comp_sep).
TNC_TOL = 1e-18
TNC_MAX_ITER = 2000
TNC_OPTIONS = {
    "disp": False,
    "gtol": TNC_TOL,
    "eps": TNC_TOL,
    "maxfun": TNC_MAX_ITER * 10,
    "ftol": TNC_TOL,
    "xtol": TNC_TOL,
}

# r grid: estimate_r evaluates the likelihood on linspace(-R_MAX_POINT, R_MAX_POINT, 1000).
# The furax-cs default (5e-3, step 1e-5) is coarse for r-hat of order 1e-6; 1e-3 gives a step
# of 2e-6 and contains the long upper tail of the one-realisation likelihood at nside 64.
R_MAX_POINT = 1e-3

# Linear solver used by furax for (A^T N^-1 A)^-1 inside the likelihood. The direct
# block solve is exact and its cost does not depend on the parameters, unlike the CG of
# scripts/kmeans_model.py. furax caches the first trace of the likelihood, so every
# likelihood call of a process must run under this configuration.
FURAX_CONFIG = {"solver": BlockDiagonalSolver()}

SKY_PRESETS = {
    "c1d1s1": ("d1", "s1"),
    "c1d10s5": ("d10", "s5"),
}

HERE = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("FX_WORK_DIR", HERE / "work"))

NLL = partial(
    negative_log_likelihood,
    dust_nu0=DUST_NU0,
    synchrotron_nu0=SYNC_NU0,
    analytical_gradient=True,
)
NLL_EVAL = partial(
    negative_log_likelihood,
    dust_nu0=DUST_NU0,
    synchrotron_nu0=SYNC_NU0,
    analytical_gradient=False,
)


def use_work_dir() -> Path:
    """Move to the directory that holds ``freq_maps_cache`` (furax-cs loaders use the cwd)."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK_DIR)
    return WORK_DIR


def hardware() -> dict[str, str]:
    devices = jax.devices()
    return {
        "host": platform.node(),
        "n_cpu": str(os.cpu_count()),
        "jax_backend": jax.default_backend(),
        "jax_device": str(devices[0].device_kind),
    }


# ---------------------------------------------------------------------------
# Sky maps and true parameters
# ---------------------------------------------------------------------------
def _pysm_frequency_maps(presets: list[str], nu: np.ndarray) -> np.ndarray:
    """Noise-free maps in uK_CMB, as furax ``get_observation`` builds them (delta bandpasses)."""
    import pysm3

    sky = pysm3.Sky(nside=NSIDE, preset_strings=presets)
    maps = []
    for freq in nu:
        emission = sky.get_emission(freq * u.GHz)
        emission = emission.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq * u.GHz))
        maps.append(np.asarray(emission.value, dtype=np.float64))
    return np.stack(maps, axis=0)  # (n_freq, 3, npix)


def _ensure_maps(tag: str, nu: np.ndarray) -> None:
    cache = Path("freq_maps_cache")
    full = cache / f"freq_maps_nside_{NSIDE}_no_noise_{tag}.pkl"
    dust, sync = SKY_PRESETS[tag]
    fg = cache / f"freq_maps_nside_{NSIDE}_no_noise_{dust}{sync}.pkl"
    cmb = cache / f"freq_maps_nside_{NSIDE}_c1.pkl"
    if tag == "c1d10s5":
        # The furax and furax-cs tag parsers split tags into 2-character chunks,
        # so d10 cannot go through save_to_cache. Same recipe, written to the same files.
        cache.mkdir(exist_ok=True)
        for path, presets in ((full, ["c1", dust, sync]), (fg, [dust, sync])):
            if not path.exists():
                tmp = path.with_suffix(".tmp")  # write then rename, so a killed run leaves no stub
                with open(tmp, "wb") as f:
                    pickle.dump(_pysm_frequency_maps(presets, nu), f)
                tmp.replace(path)
        if not cmb.exists():
            save_cmb_map(NSIDE, sky="c1d1s1")
        return
    if not full.exists():
        save_to_cache(NSIDE, instrument_name=INSTRUMENT, sky=tag)
    if not fg.exists():
        save_fg_map(NSIDE, instrument_name=INSTRUMENT, sky=tag)
    if not cmb.exists():
        save_cmb_map(NSIDE, sky=tag)


def true_parameter_maps(tag: str) -> dict[str, np.ndarray]:
    """Full-sky true spectral parameters of the PySM foreground model.

    Cached in ``freq_maps_cache`` next to the frequency maps (building the d10s5 sky takes
    minutes).
    """
    dust, sync = SKY_PRESETS[tag]
    path = Path("freq_maps_cache") / f"true_params_nside_{NSIDE}_{dust}{sync}.npz"
    if path.exists():
        with np.load(path) as stored:
            return {p: stored[p] for p in PARAMS}

    import pysm3

    sky = pysm3.Sky(nside=NSIDE, preset_strings=[dust, sync])

    def full_sky(quantity) -> np.ndarray:
        value = np.asarray(getattr(quantity, "value", quantity), dtype=np.float64)
        return np.broadcast_to(value, (NPIX,)).copy()

    true = {
        "beta_dust": full_sky(sky.components[0].mbb_index),
        "temp_dust": full_sky(sky.components[0].mbb_temperature),
        "beta_pl": full_sky(sky.components[1].pl_index),
    }
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(
        path.stem + "_tmp.npz"
    )  # write then rename, so a killed run leaves no stub
    np.savez(tmp, **true)
    tmp.replace(path)
    return true


def prepare_sky(tag: str) -> dict:
    """Masked data, foreground-only data, input CMB and true parameters for one sky."""
    instrument = get_instrument(INSTRUMENT)
    nu = np.asarray(instrument.frequency, dtype=np.float64)
    _ensure_maps(tag, nu)
    _, freq_maps = load_from_cache(NSIDE, instrument_name=INSTRUMENT, sky=tag)
    _, fg_maps = load_fg_map(NSIDE, instrument_name=INSTRUMENT, sky=tag)
    cmb_map = load_cmb_map(NSIDE, sky=tag)
    freq_maps = np.asarray(freq_maps, dtype=np.float64)
    fg_maps = np.asarray(fg_maps, dtype=np.float64)
    cmb_map = np.asarray(cmb_map, dtype=np.float64)

    mask = np.asarray(get_mask(MASK_NAME, nside=NSIDE))
    (indices,) = np.where(mask == 1)
    return {
        "tag": tag,
        "instrument": instrument,
        "nu": nu,
        "mask": mask,
        "indices": indices,
        "fsky": indices.size / NPIX,
        "d": Stokes.from_stokes(freq_maps[:, 1, indices], freq_maps[:, 2, indices]),
        "d_fg": Stokes.from_stokes(fg_maps[:, 1, indices], fg_maps[:, 2, indices]),
        "cmb": np.stack([cmb_map[1, indices], cmb_map[2, indices]]),
        "true": true_parameter_maps(tag),
    }


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
def renumber(labels: np.ndarray) -> np.ndarray:
    """Relabel to 0..K-1 so that every patch holds at least one pixel."""
    return np.unique(np.asarray(labels), return_inverse=True)[1].astype(np.int64)


_PACKAGED_D1S1 = {
    "beta_dust": "pixel_subsets_from_true_d1_s1_Bd_nbins100.npy",
    "temp_dust": "pixel_subsets_from_true_d1_s1_Td_nbins100.npy",
    "beta_pl": "pixel_subsets_from_true_d1_s1_Bs_nbins100.npy",
}


def packaged_d1s1_patches() -> dict[str, np.ndarray]:
    """Full-sky packaged ideal patches for d1s1 (used by ``kmeans-model -c true true true``)."""
    root = pkg_files("furax_cs").joinpath("data", "pixelsubset")
    return {p: np.load(str(root.joinpath(f))) for p, f in _PACKAGED_D1S1.items()}


def binned_true_patches(sky: dict, nbins: int = 100) -> dict[str, np.ndarray]:
    """Equal-width bins of the true parameters on the observed pixels
    (``furax_cs.bin_parameter_map``), before renumbering."""
    out = {}
    for p in PARAMS:
        labels, _, _ = bin_parameter_map(sky["true"][p][sky["indices"]], nbins)
        out[p] = labels
    return out


def ideal_patches(sky: dict, nbins: int = 100) -> dict[str, np.ndarray]:
    """Patches that follow the true parameters: 100 equal-width bins of each parameter."""
    if sky["tag"] == "c1d1s1":  # the packaged d1s1 patches (nside 64) of kmeans-model -c true
        packaged = packaged_d1s1_patches()
        labels = {}
        for p in PARAMS:
            masked = jnp.asarray(packaged[p][sky["indices"]])
            n = int(jnp.unique(masked).size)
            # Same renumbering as kmeans_model.load_precomputed_clusters
            labels[p] = np.asarray(normalize_by_first_occurrence(masked, n, n), dtype=np.int64)
        return labels
    return {p: renumber(v) for p, v in binned_true_patches(sky, nbins).items()}


def patch_counts(patches: dict[str, np.ndarray]) -> dict[str, int]:
    return {p: int(np.max(patches[p])) + 1 for p in PARAMS}


def as_patch_indices(patches: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {PATCH_KEY[p]: jnp.asarray(patches[p]) for p in PARAMS}


def initial_and_bounds(counts: dict[str, int]):
    init = {p: jnp.full((counts[p],), START[p]) for p in PARAMS}
    lower = {p: jnp.full((counts[p],), LOWER[p]) for p in PARAMS}
    upper = {p: jnp.full((counts[p],), UPPER[p]) for p in PARAMS}
    return init, lower, upper


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def noisy_data(sky: dict, seed: int):
    noised_d, N, _ = generate_noise_operator(
        jax.random.PRNGKey(seed),
        NOISE_RATIO,
        jnp.asarray(sky["indices"]),
        NSIDE,
        sky["d"],
        sky["instrument"],
    )
    return noised_d, N


def identity_noise(d: Stokes):
    """N = 1, used for the noise-free foreground fit that gives the systematic residual."""
    return HomothetyOperator(1.0, in_structure=d.structure)


# ---------------------------------------------------------------------------
# FURAX AdaTopK (GPU)
# ---------------------------------------------------------------------------
def pixel_terms(params, nu, N, d, patch_indices) -> jax.Array:
    """Per-pixel terms t_p of the furax spectral likelihood, with -2 ln L = -sum_p t_p."""
    AND, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, DUST_NU0, SYNC_NU0, None, None
    )
    prods = [a * b for a, b in zip(jax.tree.leaves(AND), jax.tree.leaves(s))]
    return sum(p.reshape(-1, p.shape[-1]).sum(axis=0) for p in prods)


def compensated_nll(params, nu, N, d, patch_indices, t_ref):
    """-2 ln L minus its value at the start point, summed as per-pixel differences.

    -2 ln L is about 1e9, so its float64 sum is resolved to about 1e-7, while the changes
    that decide the last iterations in one-pixel patches are of that size. Summing
    t_p(x) - t_p(x0) resolves the value to about 1e-9. The gradient is the analytical
    gradient of NLL (the constant shift does not change it).
    """
    base = NLL(params, nu=nu, N=N, d=d, patch_indices=patch_indices)
    t = pixel_terms(jax.lax.stop_gradient(params), nu, N, d, patch_indices)
    value = -jnp.sum(t - t_ref)
    return (base - jax.lax.stop_gradient(base)) + jax.lax.stop_gradient(value)


def make_adatopk_fit(sky: dict, patches: dict[str, np.ndarray]):
    """Jitted AdaTopK fit on the compensated objective; best_loss is relative to the start."""
    counts = patch_counts(patches)
    init, lower, upper = initial_and_bounds(counts)
    nu = jnp.asarray(sky["nu"])
    patch_indices = as_patch_indices(patches)

    def fit(d, N):
        t_ref = pixel_terms(init, nu, N, d, patch_indices)
        params, state = minimize(
            fn=compensated_nll,
            init_params=init,
            solver_name=ADATOPK_SOLVER,
            max_iter=ADATOPK_MAX_ITER,
            atol=ADATOPK_ATOL,
            rtol=ADATOPK_RTOL,
            lower_bound=lower,
            upper_bound=upper,
            precondition=False,
            options=dict(ADATOPK_OPTIONS),
            progress=False,
            nu=nu,
            N=N,
            d=d,
            patch_indices=patch_indices,
            t_ref=t_ref,
        )
        return params, state.iter_num, state.best_loss

    return jax.jit(fit)


def run_adatopk(sky: dict, patches: dict[str, np.ndarray], seeds) -> dict:
    """Noisy fits, one realisation at a time; compile time is reported separately."""
    fit = make_adatopk_fit(sky, patches)
    out = {p: [] for p in PARAMS}
    out.update(time=[], iters=[], best_loss=[])
    with Config(**FURAX_CONFIG):
        d0, N0 = noisy_data(sky, seeds[0])
        t0 = time.perf_counter()
        compiled = fit.lower(d0, N0).compile()
        out["compile_time"] = time.perf_counter() - t0
        for seed in seeds:
            d, N = noisy_data(sky, seed)
            jax.block_until_ready(d)
            t0 = time.perf_counter()
            params, iters, best_loss = jax.block_until_ready(compiled(d, N))
            out["time"].append(time.perf_counter() - t0)
            for p in PARAMS:
                out[p].append(np.asarray(params[p]))
            out["iters"].append(int(iters))
            out["best_loss"].append(float(best_loss))
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out.items()}


def run_adatopk_syst(sky: dict, patches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Noise-free fit to the foreground-only maps with N = 1."""
    fit = make_adatopk_fit(sky, patches)
    with Config(**FURAX_CONFIG):
        params, _, _ = jax.block_until_ready(fit(sky["d_fg"], identity_noise(sky["d_fg"])))
    return {p: np.asarray(params[p]) for p in PARAMS}


# ---------------------------------------------------------------------------
# FGBuster TNC (CPU)
# ---------------------------------------------------------------------------
def _fgbuster_full_sky(sky: dict, d: Stokes) -> np.ndarray:
    """(n_freq, 2, npix) full-sky maps with UNSEEN outside the mask, as fgbuster expects."""
    n_freq = sky["nu"].size
    full = np.full((n_freq, 2, NPIX), hp.UNSEEN)
    full[:, 0, sky["indices"]] = np.asarray(d.q)
    full[:, 1, sky["indices"]] = np.asarray(d.u)
    return full


def _fgbuster_patch_ids(sky: dict, patches: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Full-sky patch maps; masked pixels take the largest index (fgbuster_model.py)."""
    ids = []
    for p in PARAMS:
        full = np.full(NPIX, int(np.max(patches[p])), dtype=np.int64)
        full[sky["indices"]] = patches[p]
        ids.append(full)
    return ids


def _fgbuster_instrument(sky: dict, uniform: bool = False):
    from furax._instruments.sky import FGBusterInstrument

    inst = sky["instrument"]
    if uniform:
        ones = np.ones_like(np.asarray(inst.depth_p, dtype=np.float64))
        return FGBusterInstrument(np.asarray(inst.frequency), ones, ones)
    return inst


def _fgbuster_components():
    from fgbuster import CMB, Dust, Synchrotron

    components = [CMB(), Dust(DUST_NU0), Synchrotron(SYNC_NU0)]
    components[1]._set_default_of_free_symbols(beta_d=START["beta_dust"], temp=START["temp_dust"])
    components[2]._set_default_of_free_symbols(beta_pl=START["beta_pl"])
    return components


def _tnc_once(sky: dict, data: np.ndarray, patch_ids, instrument):
    from fgbuster import adaptive_comp_sep

    bounds = [(LOWER[p], UPPER[p]) for p in PARAMS]
    return adaptive_comp_sep(
        _fgbuster_components(),
        instrument,
        data,
        patch_ids,
        bounds=bounds,
        options=dict(TNC_OPTIONS),
        method="TNC",
        tol=TNC_TOL,
    )


def make_fgbuster_objective(sky: dict, patches: dict[str, np.ndarray], d: Stokes):
    """fgbuster's own -2 ln L and gradient, built as in ``adaptive_comp_sep`` (fgbuster 40ef92c).

    Returns ``f(params) -> (value, grad_dict)`` in the same units as the furax likelihood
    (fgbuster minimises -ln L, so both are multiplied by 2).
    """
    from fgbuster.algebra import (
        _A_dB_ev_and_comp_of_dB_as_compatible_list,
        _build_bound_inv_logL_and_logL_dB,
    )
    from fgbuster.mixingmatrix import MixingMatrix
    from fgbuster.observation_helpers import standardize_instrument
    from fgbuster.separation_recipes import _get_prewhiten_factors, _intersect_mask

    instrument = standardize_instrument(sky["instrument"])
    data = _fgbuster_full_sky(sky, d)
    mask = _intersect_mask(data)
    data = hp.pixelfunc.ma_to_array(data).copy()
    data[..., mask] = 0
    prewhiten = _get_prewhiten_factors(instrument, data.shape, hp.get_nside(data[0]))
    invN = np.zeros(prewhiten.shape + prewhiten.shape[-1:])
    np.einsum("...ii->...i", invN)[:] = prewhiten**2

    patch_ids = _fgbuster_patch_ids(sky, patches)
    n_clusters = [ids.max() + 1 for ids in patch_ids]

    def unpack(x):
        maps, i = [], 0
        for n, ids in zip(n_clusters, patch_ids):
            maps.append(x[i : i + n][ids].reshape(-1, 1))
            i += n
        return maps

    A = MixingMatrix(*_fgbuster_components())
    x0 = np.concatenate([np.full(n, START[p]) for n, p in zip(n_clusters, PARAMS)])
    A_dB_ev, comp_of_dB = _A_dB_ev_and_comp_of_dB_as_compatible_list(
        A.diff_evaluator(instrument.frequency, unpack), list(zip(A.comp_of_dB, patch_ids)), x0
    )
    fun, jac, _ = _build_bound_inv_logL_and_logL_dB(
        A.evaluator(instrument.frequency, unpack), data.T, invN, A_dB_ev, comp_of_dB, None
    )
    splits = np.cumsum(n_clusters)[:-1]

    def objective(params: dict) -> tuple[float, dict[str, np.ndarray]]:
        x = np.concatenate([np.asarray(params[p], dtype=np.float64) for p in PARAMS])
        value = 2.0 * float(fun(x))
        grad = 2.0 * np.asarray(jac(x))
        return value, dict(zip(PARAMS, np.split(grad, splits)))

    return objective


def projected_gradient(params: dict, grad: dict) -> dict[str, float]:
    """KKT residual of the box-constrained problem (same rule as ``make_evaluator``)."""
    pieces, n_low, n_up = [], 0, 0
    for p in PARAMS:
        x = np.asarray(params[p])
        g = np.asarray(grad[p])
        at_low = x <= LOWER[p] + 1e-7 * (abs(LOWER[p]) + 1.0)
        at_up = x >= UPPER[p] - 1e-7 * (abs(UPPER[p]) + 1.0)
        pieces.append(np.where(at_low, np.minimum(g, 0), np.where(at_up, np.maximum(g, 0), g)))
        n_low += int(at_low.sum())
        n_up += int(at_up.sum())
    flat = np.concatenate(pieces)
    return {
        "pg_norm": float(np.linalg.norm(flat)),
        "pg_max": float(np.max(np.abs(flat))),
        "n_at_lower": n_low,
        "n_at_upper": n_up,
    }


def evaluate_fgbuster(sky: dict, patches: dict[str, np.ndarray], runs: dict, seeds) -> dict:
    """fgbuster -2 ln L and its KKT residual at the final parameters of one optimiser."""
    rows = []
    for i, seed in enumerate(seeds):
        d, _ = noisy_data(sky, seed)
        params = {p: np.asarray(runs[p][i]) for p in PARAMS}
        value, grad = make_fgbuster_objective(sky, patches, d)(params)
        rows.append({"nll": value, **projected_gradient(params, grad)})
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def run_tnc(sky: dict, patches: dict[str, np.ndarray], seeds) -> dict:
    """Noisy fits with fgbuster ``adaptive_comp_sep`` (scipy TNC), one realisation at a time."""
    patch_ids = _fgbuster_patch_ids(sky, patches)
    out = {p: [] for p in PARAMS}
    out.update(time=[], iters=[], nfev=[], status=[], message=[], fun=[])
    for seed in seeds:
        d, _ = noisy_data(sky, seed)
        data = _fgbuster_full_sky(sky, d)
        t0 = time.perf_counter()
        res = _tnc_once(sky, data, patch_ids, sky["instrument"])
        out["time"].append(time.perf_counter() - t0)
        for p, x in zip(PARAMS, res.x):
            out[p].append(np.asarray(x, dtype=np.float64))
        out["iters"].append(int(getattr(res, "nit", -1)))
        out["nfev"].append(int(getattr(res, "nfev", -1)))
        out["status"].append(int(getattr(res, "status", -99)))
        out["message"].append(str(getattr(res, "message", "")))
        out["fun"].append(float(res.fun))
        print(
            f"TNC seed {seed}: {out['time'][-1]:.1f} s, nit={out['iters'][-1]}, "
            f"nfev={out['nfev'][-1]}, status={out['status'][-1]} ({out['message'][-1]})",
            flush=True,
        )
    return {k: np.asarray(v) for k, v in out.items()}


def run_tnc_syst(sky: dict, patches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Noise-free TNC fit to the foreground-only maps with uniform weights (N = 1)."""
    res = _tnc_once(
        sky,
        _fgbuster_full_sky(sky, sky["d_fg"]),
        _fgbuster_patch_ids(sky, patches),
        _fgbuster_instrument(sky, uniform=True),
    )
    return {p: np.asarray(x, dtype=np.float64) for p, x in zip(PARAMS, res.x)}


# ---------------------------------------------------------------------------
# Common evaluator (same function, data and weights for both optimisers)
# ---------------------------------------------------------------------------
def make_evaluator(sky: dict, patches: dict[str, np.ndarray]):
    nu = jnp.asarray(sky["nu"])
    patch_indices = as_patch_indices(patches)
    counts = patch_counts(patches)
    _, lower, upper = initial_and_bounds(counts)

    def nll(params, d, N):
        return NLL_EVAL(params, nu=nu, N=N, d=d, patch_indices=patch_indices)

    @jax.jit
    def evaluate(params, d, N):
        value, grad = jax.value_and_grad(nll)(params, d, N)
        tol = {p: 1e-7 * (jnp.abs(lower[p]) + 1.0) for p in PARAMS}
        at_lower = {p: params[p] <= lower[p] + tol[p] for p in PARAMS}
        at_upper = {p: params[p] >= upper[p] - 1e-7 * (jnp.abs(upper[p]) + 1.0) for p in PARAMS}
        # KKT residual: a parameter at a bound keeps only the gradient component that
        # points out of the feasible box along the descent direction.
        pg = {
            p: jnp.where(
                at_lower[p],
                jnp.minimum(grad[p], 0.0),
                jnp.where(at_upper[p], jnp.maximum(grad[p], 0.0), grad[p]),
            )
            for p in PARAMS
        }
        flat_pg = jnp.concatenate([pg[p] for p in PARAMS])
        cmb = sky_signal(
            params,
            nu=nu,
            d=d,
            N=N,
            dust_nu0=DUST_NU0,
            synchrotron_nu0=SYNC_NU0,
            patch_indices=patch_indices,
        )["cmb"]
        return {
            "nll": value,
            "pg_norm": jnp.linalg.norm(flat_pg),
            "pg_max": jnp.max(jnp.abs(flat_pg)),
            "n_at_lower": sum(jnp.sum(at_lower[p]) for p in PARAMS),
            "n_at_upper": sum(jnp.sum(at_upper[p]) for p in PARAMS),
            "finite": jnp.all(jnp.array([jnp.all(jnp.isfinite(params[p])) for p in PARAMS])),
            "cmb": jnp.stack([cmb.q, cmb.u]),
        }

    return evaluate


def evaluate_runs(sky: dict, patches: dict[str, np.ndarray], runs: dict, seeds) -> dict:
    """Evaluate the final parameters of one optimiser on every noise realisation."""
    evaluate = make_evaluator(sky, patches)
    rows = []
    with Config(**FURAX_CONFIG):
        for i, seed in enumerate(seeds):
            d, N = noisy_data(sky, seed)
            params = {p: jnp.asarray(runs[p][i]) for p in PARAMS}
            rows.append(jax.device_get(evaluate(params, d, N)))
    return {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}


def make_pixel_terms(sky: dict, patches: dict[str, np.ndarray]):
    """Per-pixel terms t_p of -2 ln L (``pixel_terms``) as a function of (params, d, N).

    Differences of -2 ln L between two points are summed as per-pixel differences, which
    resolves them to about 1e-9 although -2 ln L itself is about 1e9.
    """
    nu = jnp.asarray(sky["nu"])
    patch_indices = as_patch_indices(patches)
    terms = jax.jit(lambda params, d, N: pixel_terms(params, nu, N, d, patch_indices))

    def evaluate(params, d, N) -> np.ndarray:
        with Config(**FURAX_CONFIG):
            return np.asarray(terms({p: jnp.asarray(params[p]) for p in PARAMS}, d, N))

    return evaluate


def syst_cmb(sky: dict, patches: dict[str, np.ndarray], params: dict) -> np.ndarray:
    """W d_fg for the CMB component, with N = 1 (systematic residual map on the mask)."""
    evaluate = make_evaluator(sky, patches)
    with Config(**FURAX_CONFIG):
        out = evaluate(
            {p: jnp.asarray(params[p]) for p in PARAMS},
            sky["d_fg"],
            identity_noise(sky["d_fg"]),
        )
    return np.asarray(out["cmb"])


# ---------------------------------------------------------------------------
# Residuals and r
# ---------------------------------------------------------------------------
def _full_sky_qu(sky: dict, qu: np.ndarray) -> Stokes:
    """Cut-out (..., 2, n_valid) to full-sky StokesQU with UNSEEN outside the mask."""
    qu = np.asarray(qu)
    full = np.full(qu.shape[:-1] + (NPIX,), hp.UNSEEN)
    full[..., sky["indices"]] = qu
    return Stokes.from_stokes(full[..., 0, :], full[..., 1, :])


def r_estimate(sky: dict, cmb_hat: np.ndarray, wd_cmb: np.ndarray) -> dict:
    """r-hat and sigma(r) as in r_analysis/compute.py (f_sky-corrected likelihood)."""
    fsky = sky["fsky"]
    ell_range = np.arange(2, NSIDE * 2 + 2)
    s_true_qu = _full_sky_qu(sky, sky["cmb"])
    s_true = np.stack([np.zeros(NPIX), np.asarray(s_true_qu.q), np.asarray(s_true_qu.u)])
    cmb_hat_stokes = _full_sky_qu(sky, cmb_hat)
    cl_syst, syst_map = compute_systematic_res(_full_sky_qu(sky, wd_cmb), fsky, ell_range)
    cl_total, _ = compute_total_res(cmb_hat_stokes, s_true, fsky, ell_range)
    cl_stat, _ = compute_statistical_res(cmb_hat_stokes, s_true, fsky, ell_range, syst_map)
    with contextlib.redirect_stdout(io.StringIO()):  # estimate_r prints its noise spectrum
        r_best, sigma_neg, sigma_pos, r_grid, likelihood, _, cl_r1, cl_lens, _ = estimate_r(
            cl_total, NSIDE, np.maximum(cl_stat, 0), fsky, is_cl_obs=False, max_point=R_MAX_POINT
        )
    if max(likelihood[0], likelihood[-1]) > 1e-3:
        print(
            f"WARNING: r likelihood not negligible at the grid edge (+-{R_MAX_POINT})", flush=True
        )
    return {
        "ell": ell_range,
        "cl_syst": cl_syst,
        "cl_stat": cl_stat,
        "cl_total": cl_total,
        "r_best": r_best,
        "sigma_neg": sigma_neg,
        "sigma_pos": sigma_pos,
        "r_grid": r_grid,
        "likelihood": likelihood,
        "cl_r1": cl_r1,
        "cl_lens": cl_lens,
    }
