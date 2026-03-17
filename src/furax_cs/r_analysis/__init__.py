import os

os.environ["EQX_ON_ERROR"] = "nan"

from .plotting import (
    plot_all_cl_residuals,
    plot_all_r_estimation,
    plot_r_estimator,
)
from .r_estimate import estimate_r, estimate_r_from_maps
from .residuals import (
    compute_cl_bb_sum,
    compute_cl_obs_bb,
    compute_cl_true_bb,
    compute_statistical_res,
    compute_systematic_res,
    compute_total_res,
)
from .snapshot import CLData, CMBData, CompSepResult, ParamData, ResidualData, REstimate
from .utils import params_to_maps

__all__ = [
    "estimate_r",
    "estimate_r_from_maps",
    "compute_systematic_res",
    "compute_statistical_res",
    "compute_total_res",
    "plot_r_estimator",
    "compute_cl_bb_sum",
    "compute_cl_obs_bb",
    "compute_cl_true_bb",
    "plot_all_cl_residuals",
    "plot_all_r_estimation",
    "params_to_maps",
    "CompSepResult",
    "CMBData",
    "CLData",
    "REstimate",
    "ResidualData",
    "ParamData",
]
