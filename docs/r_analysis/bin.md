# bin — Post-Clustering Parameter Binning

The `bin` subcommand reads result folders produced by `kmeans-model` (or `ptep-model`), bins each spectral-parameter map into equal-width bins, and writes full-sky `.npy` patch files. These files can be fed directly to `kmeans-model -c` to re-run component separation with the binned clustering.

## Basic Usage

```bash
r_analysis bin \
    -n 64 \
    -r "kmeans_BD10000_TD500_BS500_GAL020" \
    -ird results/ \
    -o binned_patches/ \
    --bin-bd 50 --bin-td 20 --bin-bs 30
```

## How It Works

1. **Load** result folders matching the `-r` pattern (each must contain `results.npz` and `mask.npy`).
2. **Combine** disjoint masks from multiple folders into a single valid-pixel mask.
3. **Select** a reference noise realization (controlled by `--noise-selection`).
4. **Bin** each parameter's pixel-level values into equal-width bins using `bin_parameter_map`.
5. **Write** full-sky `.npy` files to `--output-dir`:
   - `patches_beta_dust.npy`
   - `patches_temp_dust.npy`
   - `patches_beta_pl.npy`
   - `mask.npy`

Each `.npy` file is a `float64` array of shape `(npix,)` where valid pixels contain 0-based bin indices and masked pixels are `hp.UNSEEN`.

## Visual Example

The following figures illustrate the binning process on a GAL060 mask with 200/$\beta_d$, 100/$T_d$, 100/$\beta_s$ clusters binned down to 10 bins each.

**Original K-means patches vs binned:**

![Original vs binned patches](../images/binning_original_vs_binned.png)

## Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output-dir` | `str` | *required* | Output directory for `.npy` patch files |
| `--bin-bd` | `int` | none | Number of equal-width bins for $\beta_\mathrm{dust}$ |
| `--bin-td` | `int` | none | Number of equal-width bins for $T_\mathrm{dust}$ |
| `--bin-bs` | `int` | none | Number of equal-width bins for $\beta_\mathrm{synch}$ |
| `--noise-selection` | `str` | `min-value` | Noise realization selection: `min-value`, `min-nll`, or integer index |

At least one `--bin-*` argument is required. Parameters without a `--bin-*` flag are preserved with their original (renumbered) cluster indices.

Plus all [common arguments](index.md#common-arguments) (`-n`, `-r`, `-ird`, `--sky`, `-mi`, `-s`, etc.).

## Workflow: Bin and Re-Run

The typical workflow is to first run a high-resolution component separation, then bin the resulting parameters and re-run with the binned patches:

```bash
# 1. Initial high-resolution run
kmeans-model -n 64 -pc 10000 500 500 -m GAL020 -tag c1d1s1

# 2. Bin the parameters
r_analysis bin \
    -n 64 \
    -r "kmeans_BD10000_TD500_BS500_GAL020" \
    -ird results/ \
    -o binned_patches/ \
    --bin-bd 50 --bin-td 20 --bin-bs 30

# 3. Re-run with binned patches
kmeans-model -n 64 \
    -c binned_patches/patches_beta_dust.npy \
       binned_patches/patches_temp_dust.npy \
       binned_patches/patches_beta_pl.npy \
    -m GAL020 -tag c1d1s1
```

## API Reference

The core binning function is exposed at the package level:

```python
from furax_cs import bin_parameter_map

patch_indices, bin_centers, bin_edges = bin_parameter_map(pixel_values, nbins=50)
```

See the [API documentation](../api/index.md#binning) for the full docstring with a reconstruction example.
