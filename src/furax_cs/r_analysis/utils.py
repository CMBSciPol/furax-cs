from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from furax.obs.stokes import Stokes, StokesI, StokesIQU, StokesQU
from jaxtyping import (
    Array,
)


def expand_stokes(stokes_map: Stokes) -> StokesIQU:
    """Promote a StokesI or StokesQU instance to StokesIQU.

    Parameters
    ----------
    stokes_map : Stokes
        Input Stokes data structure produced by the component separation pipeline.

    Returns
    -------
    StokesIQU
        Stokes object with I, Q, U components explicitly populated.
    """
    if isinstance(stokes_map, StokesIQU):
        return stokes_map

    zeros = np.zeros(shape=stokes_map.shape, dtype=stokes_map.dtype)

    if isinstance(stokes_map, StokesI):
        return StokesIQU(stokes_map, zeros, zeros)
    elif isinstance(stokes_map, StokesQU):
        return StokesIQU(zeros, stokes_map.q, stokes_map.u)
    else:
        # Fallback or error if passed something unexpected
        raise TypeError(f"Unsupported Stokes type: {type(stokes_map)}")


def filter_constant_param(input_dict: Mapping[str, Array], indx: int) -> Mapping[str, Array]:
    """Extract a specific entry from a tree of arrays.

    Parameters
    ----------
    input_dict : Mapping[str, Array]
        Tree containing clustering results indexed by realization.
    indx : int
        Index to select along the leading dimension of each array.

    Returns
    -------
    Mapping[str, Array]
        Tree with arrays sliced at the requested index.
    """
    return jax.tree.map(lambda x: x[indx], input_dict)


def index_run_data(run_data: Mapping[str, Array], run_index: int) -> Mapping[str, Array]:
    """Select the requested run across cached result arrays.

    Cached quantities whose keys start with ``W_D_FG_`` or ``CL_BB_SUM_`` already
    correspond to expensive per-run products, so they are returned unchanged.

    Parameters
    ----------
    run_data : Mapping[str, Array]
        Output dictionary for a single clustering configuration.
    run_index : int of the noise realization to extract.

    Returns
    -------
    Mapping[str, Array]
        Same structure as ``run_data`` with realizations sliced on demand.
    """

    def should_index(path: tuple[Any, ...], value: Array) -> Array:
        key = path[-1].key if path else None
        if key and (
            isinstance(key, str) and (key.startswith("W_D_FG_") or key.startswith("CL_BB_SUM_"))
        ):
            return value
        return value[run_index]

    return jax.tree_util.tree_map_with_path(should_index, run_data)


def sort_results(results: Mapping[str, Array], key: str) -> Mapping[str, Array]:
    """Sort a result tree by an array value.

    Parameters
    ----------
    results : Mapping[str, Array]
        Container produced by grid search evaluations.
    key : str
        Key whose values define the ordering.

    Returns
    -------
    Mapping[str, Array]
        Tree with entries reordered consistently.
    """
    indices = np.argsort(results[key])
    return jax.tree.map(lambda x: x[indices], results)


def params_to_maps(
    run_data: Mapping[str, Array],
    previous_mask_size: Mapping[str, int],
    noise_selection: str = "min-value",
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Mapping[str, int]]:
    """Convert per-cluster parameter arrays to HEALPix maps.

    Parameters
    ----------
    run_data : Mapping[str, Array]
        Output of a clustering run containing parameters and patch indices.
    previous_mask_size : Mapping[str, int]
        Offsets that keep cluster labels unique across disjoint sky regions.
    noise_selection : str
        Strategy to select noise realization: 'min-value', 'min-nll', 'idx'.

    Returns
    -------
    tuple
        (params, patches, updated_offsets) where params are maps of mean
        spectral parameters, patches contains normalized cluster indices and
        updated_offsets holds cumulative label offsets per parameter.
    """
    B_d_patches = run_data["beta_dust_patches"]
    T_d_patches = run_data["temp_dust_patches"]
    B_s_patches = run_data["beta_pl_patches"]

    # Select the noise realization based on strategy
    if noise_selection == "min-nll":
        indx = np.argmin(run_data["NLL"])
    elif noise_selection == "min-value":
        indx = np.argmin(run_data["value"])
    else:
        # Fallback: try parsing as int
        try:
            indx = int(noise_selection)
            assert 0 <= indx < run_data["value"].shape[0]
        except ValueError:
            raise ValueError(
                f"Unknown noise selection: {noise_selection}. Use 'min-value', 'min-nll', 'idx' or an integer."
            )

    B_d = run_data["beta_dust"]
    T_d = run_data["temp_dust"]
    B_s = run_data["beta_pl"]

    B_d_map = B_d[indx][B_d_patches]
    T_d_map = T_d[indx][T_d_patches]
    B_s_map = B_s[indx][B_s_patches]

    params = {"beta_dust": B_d_map, "temp_dust": T_d_map, "beta_pl": B_s_map}
    patches = {
        "beta_dust_patches": B_d_patches,
        "temp_dust_patches": T_d_patches,
        "beta_pl_patches": B_s_patches,
    }
    raw_params = {
        "beta_dust": B_d,
        "temp_dust": T_d,
        "beta_pl": B_s,
    }

    def normalize_array(arr: Array) -> Array:
        _, indices = np.unique(arr, return_inverse=True)
        return indices

    patches_normalized = jax.tree.map(normalize_array, patches)
    patches_final = jax.tree.map(lambda x, p: x + p, patches_normalized, previous_mask_size)
    previous_mask_size_updated = jax.tree.map(
        lambda x, p: p + np.unique(x).size, patches_final, previous_mask_size
    )

    return params, raw_params, patches_final, previous_mask_size_updated
