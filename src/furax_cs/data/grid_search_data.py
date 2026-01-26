from itertools import product
from typing import Any

import numpy as np
from jaxtyping import Array, Float


def select_best_params(
    results: dict[str, Any], best_metric: str = "value", nb_best: int = 4
) -> tuple[Float[Array, "nb_best 2"], Float[Array, "nb_best"], Float[Array, "nb_best"]]:
    """Find the best parameter combinations from grid search results.

    Identifies the best combinations of (T_d, B_s) patches based on the chosen metric
    (function value or NLL), averaging over noise realizations and minimizing over
    B_d dimensions.

    Args:
        results: Dictionary containing grid search results. Expected keys:
            "T_d_patches", "B_s_patches", "value", "NLL".
        best_metric: Metric to use for ranking ("value" or "NLL"). Defaults to "value".
        nb_best: Number of top combinations to return. Defaults to 4.

    Returns:
        A tuple containing:
            - **combos**: Array of shape (nb_best, 2) with (T_d, B_s) values.
            - **best_vals**: Best metric values corresponding to combos.
            - **best_nlls**: NLL values corresponding to combos.

    Raises:
        ValueError: If best_metric is not "value" or "NLL".

    Example:
        >>> results = {
        ...     "T_d_patches": [10, 10, 20, 20],
        ...     "B_s_patches": [5, 5, 5, 5],
        ...     "value": np.array([[1.0], [0.9], [0.8], [0.7]]),
        ...     "NLL": np.array([[10.0], [9.0], [8.0], [7.0]]),
        ... }
        >>> best_combos, best_vals, best_nlls = select_best_params(results, nb_best=2)
    """
    T_d_arr = np.array(results["T_d_patches"])
    B_s_arr = np.array(results["B_s_patches"])

    T_d_unique = np.sort(np.unique(T_d_arr))
    B_s_unique = np.sort(np.unique(B_s_arr))

    combos = []
    combos_best_value = []
    combos_best_nll = []

    # Loop over all (T_d, B_s) combos
    for T_d, B_s in product(T_d_unique, B_s_unique):
        (indices,) = np.where((T_d_arr == T_d) & (B_s_arr == B_s))

        # For each index, we have multiple noise evaluations for 'value' and 'NLL'
        value = results["value"][indices]  # shape (#B_d_points, #noise_runs)
        nll = results["NLL"][indices]

        mean_value = value.mean(axis=1)  # average over noise runs, shape (#B_d_points,)
        mean_nll = nll.mean(axis=1)

        # The "best" for that combo is the lowest across the B_d dimension
        min_value_for_combo = np.min(mean_value)
        min_nll_for_combo = np.min(mean_nll)

        combos.append((T_d, B_s))
        combos_best_value.append(min_value_for_combo)
        combos_best_nll.append(min_nll_for_combo)

    combos_arr = np.array(combos)
    combos_best_value_arr = np.array(combos_best_value)
    combos_best_nll_arr = np.array(combos_best_nll)

    # Sort combos by the chosen best_metric
    if best_metric == "value":
        sorted_idx = np.argsort(combos_best_value_arr)
    elif best_metric == "NLL":
        sorted_idx = np.argsort(combos_best_nll_arr)
    else:
        raise ValueError("best_metric must be 'value' or 'NLL'.")

    chosen_idx = sorted_idx[: min(nb_best, len(sorted_idx))]

    return (
        combos_arr[chosen_idx],
        combos_best_value_arr[chosen_idx],
        combos_best_nll_arr[chosen_idx],
    )
