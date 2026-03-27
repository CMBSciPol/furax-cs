#!/usr/bin/env python3
"""3x3 mollview grid showing pixel subset processing stages.

Columns: beta_dust, temp_dust, beta_pl
Row 1: Raw full-sky files
Row 2: Masked (GAL020_U) + normalized by first occurrence
Row 3: Shuffled labels (randomized colors)
"""

from importlib.resources import files as pkg_files

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from furax_cs import get_mask
from jax_healpy.clustering import normalize_by_first_occurrence

# --- Configuration ---
PARAMS = [
    ("beta_dust", r"$\beta_d$", "pixel_subsets_from_true_d1_s1_Bd_nbins100.npy"),
    ("temp_dust", r"$T_d$", "pixel_subsets_from_true_d1_s1_Td_nbins100.npy"),
    ("beta_pl", r"$\beta_s$", "pixel_subsets_from_true_d1_s1_Bs_nbins100.npy"),
]

ROW_LABELS = ["Raw", "Masked + Normalized", "Shuffled"]


def shuffle_labels(arr: np.ndarray) -> np.ndarray:
    """Randomize cluster label values for better visual separation."""
    unique_vals = np.unique(arr[arr != hp.UNSEEN])
    shuffled_vals = np.random.permutation(unique_vals)
    mapping = dict(zip(unique_vals, shuffled_vals))
    shuffled_arr = np.vectorize(lambda x: mapping.get(x, hp.UNSEEN))(arr)
    return shuffled_arr.astype(np.float64)


def main():
    mask = get_mask("GAL020_U")
    (indices,) = jnp.where(mask == 1)
    npix = mask.shape[0]

    fig = plt.figure(figsize=(18, 14))

    np.random.seed(0)

    for col, (param_name, label, filename) in enumerate(PARAMS):
        path = str(pkg_files("furax_cs").joinpath("data", "pixelsubset", filename))
        raw = np.load(path)

        # Row 1: Raw
        hp.mollview(
            raw,
            title=f"{label} — Raw",
            sub=(3, 3, 1 + col),
            bgcolor=(0,) * 4,
            cbar=True,
        )

        # Row 2: Masked + Normalized
        masked = jnp.array(raw)[indices].astype(jnp.int64)
        n_clusters = int(jnp.unique(masked).size)
        normalized = normalize_by_first_occurrence(masked, n_clusters, n_clusters).astype(jnp.int64)
        fullsky_norm = np.full(npix, hp.UNSEEN)
        fullsky_norm[np.asarray(indices)] = np.asarray(normalized).astype(np.float64)

        hp.mollview(
            fullsky_norm,
            title=f"{label} — Masked + Normalized",
            sub=(3, 3, 4 + col),
            bgcolor=(0,) * 4,
            cbar=True,
        )

        # Row 3: Shuffled
        shuffled = shuffle_labels(fullsky_norm)

        hp.mollview(
            shuffled,
            title=f"{label} — Shuffled",
            sub=(3, 3, 7 + col),
            bgcolor=(0,) * 4,
            cbar=True,
        )

    fig.tight_layout()
    plt.savefig("pixel_subsets_3x3.png", dpi=300, bbox_inches="tight", transparent=True)
    print("Saved pixel_subsets_3x3.png")
    plt.show()


if __name__ == "__main__":
    main()
