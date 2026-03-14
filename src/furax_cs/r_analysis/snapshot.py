"""Parquet-based snapshot storage for component separation results."""
from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from ..logging_utils import info, success
from .compute import compute_all

if TYPE_CHECKING:
    import datasets
    from PIL import Image as PILImage

try:
    import datasets  # noqa: F401
except ImportError:
    pass


def requires_datasets(func):
    try:
        import datasets  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred(*args, **kwargs):
        raise ImportError(
            "Missing optional library 'datasets'. Install with: pip install furax-cs[io]"
        )

    return deferred


@dataclass(frozen=True)
class CMBData:
    cmb_q: np.ndarray  # (npix,) true CMB Q
    cmb_u: np.ndarray  # (npix,) true CMB U
    cmb_recon_q: np.ndarray  # (n_real, npix) reconstructed Q
    cmb_recon_u: np.ndarray  # (n_real, npix) reconstructed U
    patches_beta_dust: np.ndarray  # (npix,) cluster label per pixel
    patches_temp_dust: np.ndarray
    patches_beta_pl: np.ndarray
    cl_bb_sum: np.ndarray | None  # (n_real,)
    nll_summed: np.ndarray  # (n_real,)


@dataclass(frozen=True)
class CLData:
    cl_bb_r1: np.ndarray | None  # (n_ell,)
    cl_true: np.ndarray | None
    ell_range: np.ndarray
    cl_bb_obs: np.ndarray | None
    cl_bb_lens: np.ndarray | None
    cl_syst_res: np.ndarray | None
    cl_total_res: np.ndarray | None
    cl_stat_res: np.ndarray | None


@dataclass(frozen=True)
class REstimate:
    r_best: float | None
    sigma_r_neg: float | None
    sigma_r_pos: float | None
    r_grid: np.ndarray | None  # (n_grid,)
    L_vals: np.ndarray | None


@dataclass(frozen=True)
class ResidualData:
    syst_map: np.ndarray | None  # (3, npix)
    stat_maps: np.ndarray | None  # (n_real, 3, npix)


@dataclass(frozen=True)
class ParamData:
    params_beta_dust: np.ndarray | None  # (npix,) best-realization map
    params_temp_dust: np.ndarray | None
    params_beta_pl: np.ndarray | None
    true_beta_dust: np.ndarray | None
    true_temp_dust: np.ndarray | None
    true_beta_pl: np.ndarray | None
    all_params_beta_dust: np.ndarray | None  # (n_real, n_patches)
    all_params_temp_dust: np.ndarray | None
    all_params_beta_pl: np.ndarray | None


@dataclass(frozen=True)
class CompSepResult:
    kw: str
    title: str
    nside: int
    sky_tag: str
    noise_selection: str
    cmb: CMBData
    cl: CLData
    r: REstimate
    residual: ResidualData
    params: ParamData
    version: int = 1
    # PIL images rendered at snap time (all optional)
    img_cmb_true_q: PILImage.Image | None = None
    img_cmb_true_u: PILImage.Image | None = None
    img_cmb_recon_q: PILImage.Image | None = None  # best realization
    img_cmb_recon_u: PILImage.Image | None = None
    img_syst_q: PILImage.Image | None = None
    img_syst_u: PILImage.Image | None = None
    img_stat_q: PILImage.Image | None = None
    img_stat_u: PILImage.Image | None = None
    img_params_beta_dust: PILImage.Image | None = None
    img_params_temp_dust: PILImage.Image | None = None
    img_params_beta_pl: PILImage.Image | None = None
    img_true_beta_dust: PILImage.Image | None = None
    img_true_temp_dust: PILImage.Image | None = None
    img_true_beta_pl: PILImage.Image | None = None
    img_patches_beta_dust: PILImage.Image | None = None
    img_patches_temp_dust: PILImage.Image | None = None
    img_patches_beta_pl: PILImage.Image | None = None

    @requires_datasets
    def to_dataset(self) -> datasets.Dataset:
        from datasets import Dataset

        features = _build_features(hp.nside2npix(self.nside), self.cmb.cmb_recon_q.shape[0])
        data = _result_to_row(self)
        return Dataset.from_dict(data, features=features)

    @requires_datasets
    def to_parquet(self, path: str | Path) -> None:
        self.to_dataset().to_parquet(str(path))

    @classmethod
    @requires_datasets
    def from_parquet(cls, path: str | Path) -> CompSepResult:
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=str(path), split="train").with_format("numpy")
        return cls.from_dataset(ds)

    @classmethod
    @requires_datasets
    def from_dataset(cls, ds: datasets.Dataset | datasets.IterableDataset | dict) -> CompSepResult:
        import datasets as hf_datasets

        if isinstance(ds, dict):
            return _row_to_result(ds)
        elif isinstance(ds, hf_datasets.IterableDataset):
            for row in ds:
                return _row_to_result(row)
            raise ValueError("Cannot reconstruct CompSepResult from an empty IterableDataset.")
        elif isinstance(ds, hf_datasets.Dataset):
            return _row_to_result(ds[0])
        else:
            raise ValueError(f"Unsupported dataset type: {type(ds)}")

    @classmethod
    @requires_datasets
    def stack(cls, results: list[CompSepResult]) -> datasets.Dataset:
        """Stack a list of CompSepResult into a multi-row Dataset (one row per result)."""
        from datasets import Dataset

        if not results:
            raise ValueError("Cannot stack an empty list of CompSepResult")

        npix = hp.nside2npix(results[0].nside)
        n_real = results[0].cmb.cmb_recon_q.shape[0]
        features = _build_features(npix, n_real)
        return Dataset.from_dict(_stack_rows(results), features=features)


def _stack_rows(results: list[CompSepResult]) -> dict:
    """Merge N _result_to_row() dicts into a single column-oriented dict."""
    combined: dict[str, list] = {}
    for result in results:
        for k, v in _result_to_row(result).items():
            combined.setdefault(k, []).extend(v)  # each v is a 1-element list
    return combined


def _render_mollview(data: np.ndarray | None, **kwargs) -> PILImage.Image | None:
    if data is None:
        return None
    fig = plt.figure(figsize=(10, 5))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Ignoring specified arguments in this call",
            category=UserWarning,
        )
        hp.mollview(data, fig=fig.number, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=72)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return Image.open(buf).copy()


@requires_datasets
def _build_features(npix: int, n_real: int) -> datasets.Features:
    from datasets import Array2D, Features, Image, Sequence, Value

    return Features(
        {
            # Metadata
            "kw": Value("string"),
            "title": Value("string"),
            "nside": Value("int32"),
            "sky_tag": Value("string"),
            "noise_selection": Value("string"),
            "version": Value("int32"),
            # Fixed-length 1D maps (npix,)
            "cmb_q": Sequence(Value("float64"), length=npix),
            "cmb_u": Sequence(Value("float64"), length=npix),
            "patches_beta_dust": Sequence(Value("float64"), length=npix),
            "patches_temp_dust": Sequence(Value("float64"), length=npix),
            "patches_beta_pl": Sequence(Value("float64"), length=npix),
            # nullable param maps — no fixed length to allow None storage
            "params_beta_dust": Sequence(Value("float64")),
            "params_temp_dust": Sequence(Value("float64")),
            "params_beta_pl": Sequence(Value("float64")),
            "true_beta_dust": Sequence(Value("float64")),
            "true_temp_dust": Sequence(Value("float64")),
            "true_beta_pl": Sequence(Value("float64")),
            # n_real-length sequences (nll_summed always present; cl_bb_sum may be None)
            "nll_summed": Sequence(Value("float64"), length=n_real),
            "cl_bb_sum": Sequence(Value("float64")),  # nullable — no fixed length
            # 2D (n_real, npix) — variable n_real → None as first dim
            "cmb_recon_q": Array2D(shape=(None, npix), dtype="float64"),
            "cmb_recon_u": Array2D(shape=(None, npix), dtype="float64"),
            # (3, npix) nullable — Sequence correctly stores None (Array2D does not)
            "syst_map": Sequence(Sequence(Value("float64"))),
            # (n_real, 3, npix) nullable
            "stat_maps": Sequence(Sequence(Sequence(Value("float64")))),
            # 1D spectra (variable n_ell) — Sequence handles None and variable length
            "cl_bb_r1": Sequence(Value("float64")),
            "cl_true": Sequence(Value("float64")),
            "ell_range": Sequence(Value("float64")),
            "cl_bb_obs": Sequence(Value("float64")),
            "cl_bb_lens": Sequence(Value("float64")),
            "cl_syst_res": Sequence(Value("float64")),
            "cl_total_res": Sequence(Value("float64")),
            "cl_stat_res": Sequence(Value("float64")),
            # R estimation
            "r_best": Value("float64"),
            "sigma_r_neg": Value("float64"),
            "sigma_r_pos": Value("float64"),
            "r_grid": Sequence(Value("float64")),
            "L_vals": Sequence(Value("float64")),
            # all_params (n_real, n_patches) — both dims variable → nested Sequence
            "all_params_beta_dust": Sequence(Sequence(Value("float64"))),
            "all_params_temp_dust": Sequence(Sequence(Value("float64"))),
            "all_params_beta_pl": Sequence(Sequence(Value("float64"))),
            # PIL images
            "img_cmb_true_q": Image(),
            "img_cmb_true_u": Image(),
            "img_cmb_recon_q": Image(),
            "img_cmb_recon_u": Image(),
            "img_syst_q": Image(),
            "img_syst_u": Image(),
            "img_stat_q": Image(),
            "img_stat_u": Image(),
            "img_params_beta_dust": Image(),
            "img_params_temp_dust": Image(),
            "img_params_beta_pl": Image(),
            "img_true_beta_dust": Image(),
            "img_true_temp_dust": Image(),
            "img_true_beta_pl": Image(),
            "img_patches_beta_dust": Image(),
            "img_patches_temp_dust": Image(),
            "img_patches_beta_pl": Image(),
        }
    )


def _result_to_row(result: CompSepResult) -> dict:
    """Convert CompSepResult to a 1-row column-oriented dict for Dataset.from_dict."""

    def _seq(arr):
        return arr.tolist() if arr is not None else None

    return {
        "kw": [result.kw],
        "title": [result.title],
        "nside": [result.nside],
        "sky_tag": [result.sky_tag],
        "noise_selection": [result.noise_selection],
        "version": [result.version],
        "cmb_q": [_seq(result.cmb.cmb_q)],
        "cmb_u": [_seq(result.cmb.cmb_u)],
        "cmb_recon_q": [np.asarray(result.cmb.cmb_recon_q)],
        "cmb_recon_u": [np.asarray(result.cmb.cmb_recon_u)],
        "patches_beta_dust": [_seq(result.cmb.patches_beta_dust)],
        "patches_temp_dust": [_seq(result.cmb.patches_temp_dust)],
        "patches_beta_pl": [_seq(result.cmb.patches_beta_pl)],
        "cl_bb_sum": [_seq(result.cmb.cl_bb_sum)],
        "nll_summed": [_seq(result.cmb.nll_summed)],
        "cl_bb_r1": [_seq(result.cl.cl_bb_r1)],
        "cl_true": [_seq(result.cl.cl_true)],
        "ell_range": [_seq(result.cl.ell_range)],
        "cl_bb_obs": [_seq(result.cl.cl_bb_obs)],
        "cl_bb_lens": [_seq(result.cl.cl_bb_lens)],
        "cl_syst_res": [_seq(result.cl.cl_syst_res)],
        "cl_total_res": [_seq(result.cl.cl_total_res)],
        "cl_stat_res": [_seq(result.cl.cl_stat_res)],
        "r_best": [result.r.r_best],
        "sigma_r_neg": [result.r.sigma_r_neg],
        "sigma_r_pos": [result.r.sigma_r_pos],
        "r_grid": [_seq(result.r.r_grid)],
        "L_vals": [_seq(result.r.L_vals)],
        "syst_map": [
            result.residual.syst_map.tolist() if result.residual.syst_map is not None else None
        ],
        "stat_maps": [
            result.residual.stat_maps.tolist() if result.residual.stat_maps is not None else None
        ],
        "params_beta_dust": [_seq(result.params.params_beta_dust)],
        "params_temp_dust": [_seq(result.params.params_temp_dust)],
        "params_beta_pl": [_seq(result.params.params_beta_pl)],
        "true_beta_dust": [_seq(result.params.true_beta_dust)],
        "true_temp_dust": [_seq(result.params.true_temp_dust)],
        "true_beta_pl": [_seq(result.params.true_beta_pl)],
        "all_params_beta_dust": [
            result.params.all_params_beta_dust.tolist()
            if result.params.all_params_beta_dust is not None
            else None
        ],
        "all_params_temp_dust": [
            result.params.all_params_temp_dust.tolist()
            if result.params.all_params_temp_dust is not None
            else None
        ],
        "all_params_beta_pl": [
            result.params.all_params_beta_pl.tolist()
            if result.params.all_params_beta_pl is not None
            else None
        ],
        # Images — datasets.Image() accepts PIL Images directly
        "img_cmb_true_q": [result.img_cmb_true_q],
        "img_cmb_true_u": [result.img_cmb_true_u],
        "img_cmb_recon_q": [result.img_cmb_recon_q],
        "img_cmb_recon_u": [result.img_cmb_recon_u],
        "img_syst_q": [result.img_syst_q],
        "img_syst_u": [result.img_syst_u],
        "img_stat_q": [result.img_stat_q],
        "img_stat_u": [result.img_stat_u],
        "img_params_beta_dust": [result.img_params_beta_dust],
        "img_params_temp_dust": [result.img_params_temp_dust],
        "img_params_beta_pl": [result.img_params_beta_pl],
        "img_true_beta_dust": [result.img_true_beta_dust],
        "img_true_temp_dust": [result.img_true_temp_dust],
        "img_true_beta_pl": [result.img_true_beta_pl],
        "img_patches_beta_dust": [result.img_patches_beta_dust],
        "img_patches_temp_dust": [result.img_patches_temp_dust],
        "img_patches_beta_pl": [result.img_patches_beta_pl],
    }


def _row_to_result(row: dict) -> CompSepResult:
    def _arr(key):
        v = row.get(key)
        if v is None:
            return None
        arr = np.asarray(v)
        return arr if arr.ndim > 0 else None

    def _float_or_none(key):
        v = row.get(key)
        if v is None:
            return None
        f = float(v)
        return None if (f != f) else f  # NaN → None

    cmb = CMBData(
        cmb_q=_arr("cmb_q"),
        cmb_u=_arr("cmb_u"),
        cmb_recon_q=_arr("cmb_recon_q"),
        cmb_recon_u=_arr("cmb_recon_u"),
        patches_beta_dust=_arr("patches_beta_dust"),
        patches_temp_dust=_arr("patches_temp_dust"),
        patches_beta_pl=_arr("patches_beta_pl"),
        cl_bb_sum=_arr("cl_bb_sum"),
        nll_summed=_arr("nll_summed"),
    )
    cl = CLData(
        cl_bb_r1=_arr("cl_bb_r1"),
        cl_true=_arr("cl_true"),
        ell_range=_arr("ell_range"),
        cl_bb_obs=_arr("cl_bb_obs"),
        cl_bb_lens=_arr("cl_bb_lens"),
        cl_syst_res=_arr("cl_syst_res"),
        cl_total_res=_arr("cl_total_res"),
        cl_stat_res=_arr("cl_stat_res"),
    )
    r = REstimate(
        r_best=_float_or_none("r_best"),
        sigma_r_neg=_float_or_none("sigma_r_neg"),
        sigma_r_pos=_float_or_none("sigma_r_pos"),
        r_grid=_arr("r_grid"),
        L_vals=_arr("L_vals"),
    )
    residual = ResidualData(syst_map=_arr("syst_map"), stat_maps=_arr("stat_maps"))

    params = ParamData(
        params_beta_dust=_arr("params_beta_dust"),
        params_temp_dust=_arr("params_temp_dust"),
        params_beta_pl=_arr("params_beta_pl"),
        true_beta_dust=_arr("true_beta_dust"),
        true_temp_dust=_arr("true_temp_dust"),
        true_beta_pl=_arr("true_beta_pl"),
        all_params_beta_dust=_arr("all_params_beta_dust"),
        all_params_temp_dust=_arr("all_params_temp_dust"),
        all_params_beta_pl=_arr("all_params_beta_pl"),
    )

    # Images come back as PIL Images from datasets.Image()
    def _img(key):
        return row.get(key)

    return CompSepResult(
        kw=str(row["kw"]),
        title=str(row["title"]),
        nside=int(row["nside"]),
        sky_tag=str(row["sky_tag"]),
        noise_selection=str(row["noise_selection"]),
        version=int(row["version"]),
        cmb=cmb,
        cl=cl,
        r=r,
        residual=residual,
        params=params,
        img_cmb_true_q=_img("img_cmb_true_q"),
        img_cmb_true_u=_img("img_cmb_true_u"),
        img_cmb_recon_q=_img("img_cmb_recon_q"),
        img_cmb_recon_u=_img("img_cmb_recon_u"),
        img_syst_q=_img("img_syst_q"),
        img_syst_u=_img("img_syst_u"),
        img_stat_q=_img("img_stat_q"),
        img_stat_u=_img("img_stat_u"),
        img_params_beta_dust=_img("img_params_beta_dust"),
        img_params_temp_dust=_img("img_params_temp_dust"),
        img_params_beta_pl=_img("img_params_beta_pl"),
        img_true_beta_dust=_img("img_true_beta_dust"),
        img_true_temp_dust=_img("img_true_temp_dust"),
        img_true_beta_pl=_img("img_true_beta_pl"),
        img_patches_beta_dust=_img("img_patches_beta_dust"),
        img_patches_temp_dust=_img("img_patches_temp_dust"),
        img_patches_beta_pl=_img("img_patches_beta_pl"),
    )


def _build_result_from_pytrees(
    kw: str,
    title: str,
    nside: int,
    sky_tag: str,
    noise_selection: str,
    cmb_pytree: dict[str, Any],
    cl_pytree: dict[str, Any],
    r_pytree: dict[str, Any],
    residual_pytree: dict[str, Any],
    plotting_data: dict[str, Any],
    skip_images: bool = False,
) -> CompSepResult:
    """Convert the 5 pytrees from compute_group() into a CompSepResult."""
    cmb_stokes = cmb_pytree["cmb"]
    cmb_recon = cmb_pytree["cmb_recon"]
    patches_map = cmb_pytree["patches_map"]
    cl_bb_sum = cmb_pytree.get("cl_bb_sum")
    nll_summed = cmb_pytree["nll_summed"]

    # Determine best realization index
    if noise_selection == "min-nll":
        best_idx = int(np.argmin(np.asarray(nll_summed)))
    elif noise_selection == "min-value":
        best_idx = int(np.argmin(np.asarray(cl_bb_sum))) if cl_bb_sum is not None else 0
    else:
        best_idx = int(noise_selection)

    # CMB arrays
    cmb_q = np.asarray(cmb_stokes.q)
    cmb_u = np.asarray(cmb_stokes.u)
    cmb_recon_q = np.asarray(cmb_recon.q)  # (n_real, npix)
    cmb_recon_u = np.asarray(cmb_recon.u)  # (n_real, npix)

    # Patch maps (always present; float64 to preserve hp.UNSEEN ≈ -1.6375e30)
    patches_beta_dust = np.asarray(patches_map["beta_dust_patches"])
    patches_temp_dust = np.asarray(patches_map["temp_dust_patches"])
    patches_beta_pl = np.asarray(patches_map["beta_pl_patches"])
    if (
        np.all(patches_beta_dust == 0)
        and np.all(patches_temp_dust == 0)
        and np.all(patches_beta_pl == 0)
    ):
        raise ValueError(
            f"All patch maps are zero for '{kw}'. params_list was empty "
            "(all result folders failed). Cannot build a valid snapshot."
        )

    # CL data
    def _opt_arr(d, key):
        v = d.get(key)
        return np.asarray(v) if v is not None else None

    cl = CLData(
        cl_bb_r1=_opt_arr(cl_pytree, "cl_bb_r1"),
        cl_true=_opt_arr(cl_pytree, "cl_true"),
        ell_range=np.asarray(cl_pytree["ell_range"]),
        cl_bb_obs=_opt_arr(cl_pytree, "cl_bb_obs"),
        cl_bb_lens=_opt_arr(cl_pytree, "cl_bb_lens"),
        cl_syst_res=_opt_arr(cl_pytree, "cl_syst_res"),
        cl_total_res=_opt_arr(cl_pytree, "cl_total_res"),
        cl_stat_res=_opt_arr(cl_pytree, "cl_stat_res"),
    )

    # R estimate
    r = REstimate(
        r_best=float(r_pytree["r_best"]) if r_pytree.get("r_best") is not None else None,
        sigma_r_neg=float(r_pytree["sigma_r_neg"])
        if r_pytree.get("sigma_r_neg") is not None
        else None,
        sigma_r_pos=float(r_pytree["sigma_r_pos"])
        if r_pytree.get("sigma_r_pos") is not None
        else None,
        r_grid=_opt_arr(r_pytree, "r_grid"),
        L_vals=_opt_arr(r_pytree, "L_vals"),
    )

    # Residuals
    syst_map_raw = residual_pytree.get("syst_map")
    stat_maps_raw = residual_pytree.get("stat_maps")
    syst_arr = np.asarray(syst_map_raw) if syst_map_raw is not None else None
    stat_arr = np.asarray(stat_maps_raw) if stat_maps_raw is not None else None
    residual = ResidualData(syst_map=syst_arr, stat_maps=stat_arr)

    # Params
    params_map = plotting_data.get("params_map")
    true_params = plotting_data.get("true_params")
    all_params_list = plotting_data.get("all_params") or []

    params_beta_dust = np.asarray(params_map["beta_dust"]) if params_map is not None else None
    params_temp_dust = np.asarray(params_map["temp_dust"]) if params_map is not None else None
    params_beta_pl = np.asarray(params_map["beta_pl"]) if params_map is not None else None
    true_beta_dust = np.asarray(true_params["beta_dust"]) if true_params is not None else None
    true_temp_dust = np.asarray(true_params["temp_dust"]) if true_params is not None else None
    true_beta_pl = np.asarray(true_params["beta_pl"]) if true_params is not None else None

    if all_params_list:
        all_params_beta_dust = np.concatenate(
            [np.asarray(d["beta_dust"]) for d in all_params_list], axis=1
        )
        all_params_temp_dust = np.concatenate(
            [np.asarray(d["temp_dust"]) for d in all_params_list], axis=1
        )
        all_params_beta_pl = np.concatenate(
            [np.asarray(d["beta_pl"]) for d in all_params_list], axis=1
        )
    else:
        all_params_beta_dust = all_params_temp_dust = all_params_beta_pl = None

    params = ParamData(
        params_beta_dust=params_beta_dust,
        params_temp_dust=params_temp_dust,
        params_beta_pl=params_beta_pl,
        true_beta_dust=true_beta_dust,
        true_temp_dust=true_temp_dust,
        true_beta_pl=true_beta_pl,
        all_params_beta_dust=all_params_beta_dust,
        all_params_temp_dust=all_params_temp_dust,
        all_params_beta_pl=all_params_beta_pl,
    )

    cmb_data = CMBData(
        cmb_q=cmb_q,
        cmb_u=cmb_u,
        cmb_recon_q=cmb_recon_q,
        cmb_recon_u=cmb_recon_u,
        patches_beta_dust=patches_beta_dust,
        patches_temp_dust=patches_temp_dust,
        patches_beta_pl=patches_beta_pl,
        cl_bb_sum=np.asarray(cl_bb_sum) if cl_bb_sum is not None else None,
        nll_summed=np.asarray(nll_summed),
    )

    # Render mollview images
    recon_q_best = cmb_recon_q[best_idx] if cmb_recon_q.ndim > 1 else cmb_recon_q
    recon_u_best = cmb_recon_u[best_idx] if cmb_recon_u.ndim > 1 else cmb_recon_u

    def _maybe_render(data, **kwargs):
        if skip_images:
            return None
        return _render_mollview(data, **kwargs)

    return CompSepResult(
        kw=kw,
        title=title,
        nside=nside,
        sky_tag=sky_tag,
        noise_selection=noise_selection,
        cmb=cmb_data,
        cl=cl,
        r=r,
        residual=residual,
        params=params,
        img_cmb_true_q=_maybe_render(cmb_q, title=f"{kw} CMB Q true"),
        img_cmb_true_u=_maybe_render(cmb_u, title=f"{kw} CMB U true"),
        img_cmb_recon_q=_maybe_render(recon_q_best, title=f"{kw} CMB Q recon"),
        img_cmb_recon_u=_maybe_render(recon_u_best, title=f"{kw} CMB U recon"),
        img_syst_q=_maybe_render(
            syst_arr[1] if syst_arr is not None else None, title=f"{kw} Syst Q"
        ),
        img_syst_u=_maybe_render(
            syst_arr[2] if syst_arr is not None else None, title=f"{kw} Syst U"
        ),
        img_stat_q=_maybe_render(
            stat_arr[best_idx, 1] if stat_arr is not None else None, title=f"{kw} Stat Q"
        ),
        img_stat_u=_maybe_render(
            stat_arr[best_idx, 2] if stat_arr is not None else None, title=f"{kw} Stat U"
        ),
        img_params_beta_dust=_maybe_render(params_beta_dust, title=f"{kw} beta_dust"),
        img_params_temp_dust=_maybe_render(params_temp_dust, title=f"{kw} T_dust"),
        img_params_beta_pl=_maybe_render(params_beta_pl, title=f"{kw} beta_pl"),
        img_true_beta_dust=_maybe_render(true_beta_dust, title=f"{kw} true beta_dust"),
        img_true_temp_dust=_maybe_render(true_temp_dust, title=f"{kw} true T_dust"),
        img_true_beta_pl=_maybe_render(true_beta_pl, title=f"{kw} true beta_pl"),
        img_patches_beta_dust=_maybe_render(patches_beta_dust, title=f"{kw} patches beta_dust"),
        img_patches_temp_dust=_maybe_render(patches_temp_dust, title=f"{kw} patches T_dust"),
        img_patches_beta_pl=_maybe_render(patches_beta_pl, title=f"{kw} patches beta_pl"),
    )


def _result_to_plot_dict(result: CompSepResult) -> dict:
    """Convert CompSepResult to the dict format expected by plot_indiv/aggregate."""
    from furax.obs.stokes import StokesQU

    return {
        "cmb": {
            "cmb": StokesQU(result.cmb.cmb_q, result.cmb.cmb_u),
            "cmb_recon": StokesQU(result.cmb.cmb_recon_q, result.cmb.cmb_recon_u),
            "patches_map": {
                "beta_dust_patches": result.cmb.patches_beta_dust,
                "temp_dust_patches": result.cmb.patches_temp_dust,
                "beta_pl_patches": result.cmb.patches_beta_pl,
            },
            "cl_bb_sum": result.cmb.cl_bb_sum,
            "nll_summed": result.cmb.nll_summed,
        },
        "cl": {f: getattr(result.cl, f) for f in CLData.__dataclass_fields__},
        "r": {f: getattr(result.r, f) for f in REstimate.__dataclass_fields__},
        "residual": {"syst_map": result.residual.syst_map, "stat_maps": result.residual.stat_maps},
        "plotting_data": {
            "params_map": {
                "beta_dust": result.params.params_beta_dust,
                "temp_dust": result.params.params_temp_dust,
                "beta_pl": result.params.params_beta_pl,
            },
            "true_params": {
                "beta_dust": result.params.true_beta_dust,
                "temp_dust": result.params.true_temp_dust,
                "beta_pl": result.params.true_beta_pl,
            },
            "all_params": (
                [
                    {
                        "beta_dust": result.params.all_params_beta_dust,
                        "temp_dust": result.params.all_params_temp_dust,
                        "beta_pl": result.params.all_params_beta_pl,
                    }
                ]
                if result.params.all_params_beta_dust is not None
                else []
            ),
        },
    }


def run_snapshot(
    matched_results: dict[str, Any],
    nside: int,
    instrument: Any,
    output_parquet: str,
    flags: dict[str, bool],
    max_iter: int,
    solver_name: str,
    noise_selection: str,
    sky_tag: str,
    skip_images: bool = False,
) -> int:
    """Entry point for 'snap' subcommand.

    Computes statistics for matched runs and saves results to the given parquet file.
    """
    from datasets import Dataset, concatenate_datasets, load_dataset

    combined_path = Path(output_parquet)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    existing_kws: set[str] = set()
    if combined_path.exists():
        existing_ds = load_dataset("parquet", data_files=str(combined_path), split="train")
        existing_kws = set(existing_ds["kw"])
        info(f"Skipping {len(existing_kws)} already-computed entries in {combined_path.name}")

    to_compute = {kw: v for kw, v in matched_results.items() if kw not in existing_kws}

    if to_compute:
        t0 = time.perf_counter()
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
        info(f"compute_all done in {time.perf_counter() - t0:.2f}s for {len(to_compute)} groups")

        new_results = []
        for kw, result_tuple in computed.items():
            t0b = time.perf_counter()
            result = _build_result_from_pytrees(
                kw,
                kw,
                nside,
                sky_tag,
                noise_selection,
                *result_tuple,
                skip_images=skip_images,
            )
            new_results.append(result)
            info(f"  [{kw}] build: {time.perf_counter() - t0b:.2f}s")
            success(f"Built [{kw}]")

        npix = hp.nside2npix(nside)
        n_real = new_results[0].cmb.cmb_recon_q.shape[0]
        new_ds = Dataset.from_dict(_stack_rows(new_results), features=_build_features(npix, n_real))

        if combined_path.exists():
            existing_full = load_dataset("parquet", data_files=str(combined_path), split="train")
            combined_ds = concatenate_datasets([existing_full, new_ds])
        else:
            combined_ds = new_ds

        t0p = time.perf_counter()
        combined_ds.to_parquet(str(combined_path))
        success(
            f"Saved {combined_path} ({len(combined_ds)} total rows, "
            f"parquet: {time.perf_counter() - t0p:.2f}s)"
        )

    return 0
