"""FURAX component separation utilities."""

from importlib import metadata

from cadre import (
    SOLVER_NAMES,
    ScipyMinimizeState,
    apply_projection,
    condition,
    get_solver,
    lbfgs_backtrack,
    lbfgs_zoom,
    minimize,
    scipy_minimize,
)

from . import r_analysis
from .binning import bin_parameter_map
from .data import (
    MASK_CHOICES,
    CMBLensedWithTensors,
    dump_default_search_space,
    generate_custom_cmb,
    generate_needed_maps,
    get_instrument,
    get_mask,
    get_mixin_matrix_operator,
    load_cmb_map,
    load_fg_map,
    load_from_cache,
    load_search_space,
    sanitize_mask_name,
    save_cmb_map,
    save_fg_map,
    save_to_cache,
    search_space_to_jax,
    simulate_D_from_params,
    validate_search_space,
)
from .kmeans_clusters import kmeans_clusters
from .multires_clusters import multires_clusters
from .noise import generate_noise_operator

__all__ = [
    "bin_parameter_map",
    "kmeans_clusters",
    "multires_clusters",
    "generate_noise_operator",
    # Data
    "CMBLensedWithTensors",
    "dump_default_search_space",
    "generate_custom_cmb",
    "generate_needed_maps",
    "get_instrument",
    "get_mask",
    "get_mixin_matrix_operator",
    "load_cmb_map",
    "load_fg_map",
    "load_from_cache",
    "load_search_space",
    "sanitize_mask_name",
    "save_cmb_map",
    "save_fg_map",
    "save_to_cache",
    "search_space_to_jax",
    "simulate_D_from_params",
    "validate_search_space",
    "MASK_CHOICES",
    # Optim
    "SOLVER_NAMES",
    "ScipyMinimizeState",
    "apply_projection",
    "condition",
    "get_solver",
    "lbfgs_backtrack",
    "lbfgs_zoom",
    "minimize",
    "scipy_minimize",
    # Modules
    "r_analysis",
]


def __getattr__(name: str) -> str:
    """Expose package metadata attributes lazily."""
    if name == "__version__":
        try:
            return metadata.version("furax-cs")
        except metadata.PackageNotFoundError:
            return "unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
