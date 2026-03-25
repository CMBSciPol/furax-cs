# CLI Reference

This guide covers all command-line tools provided by `furax-cs` for data generation, component separation, and analysis.

## Workflow Overview

```
generate_data  →  kmeans-model / ptep-model / fgbuster-model  →  r_analysis snap  →  r_analysis plot
```

1. **Generate** simulated frequency maps
2. **Run** component separation (produces result folders)
3. **Analyze** results with `r_analysis`

## `generate_data`

Pre-generate and cache frequency maps (CMB + Dust + Synchrotron).

```bash
generate_data --nside 64 --sky c1d1s1
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--nside` | `int` | `64` | HEALPix resolution |
| `--sky` | `str` | `c1d1s1` | PySM sky configuration tag |

The `--sky` tag follows PySM naming: `c` (CMB), `d` (dust), `s` (synchrotron), with a digit indicating the model index. Use `0` to disable a component.

**Custom CMB with tensor-to-scalar ratio:**

```bash
# Generate CMB with r = 0.003
generate_data --nside 64 --sky cr3d1s1
```

The `cr<N>` prefix generates a CMB map with $r = 0.001 \times N$.

---

## `kmeans-model`

Adaptive K-means clustering for spatially-varying component separation. The sky is partitioned into clusters, and spectral parameters are optimized per cluster.

### Usage

```bash
kmeans-model -n 64 -pc 10000 500 500 -m GAL020 -tag c1d1s1
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `-n`, `--nside` | `int` | `64` | HEALPix resolution |
| `-pc`, `--patch-count` | `int int int` | `10000 500 500` | Cluster counts for $[\beta_d, T_d, \beta_s]$ |
| `-c`, `--clusters` | `str str str` | none | Cluster source for $[\beta_d, T_d, \beta_s]$. Overrides `-pc`. See [below](#cluster-sources-c). |
| `-m`, `--mask` | `str` | `GAL020_U` | Galactic mask (see Masks section below) |
| `-tag`, `--tag` | `str` | `c1d1s1` | Sky simulation tag |
| `-ns`, `--noise-sim` | `int` | `1` | Number of noise realizations |
| `-nr`, `--noise-ratio` | `float` | `0.0` | Noise level as fraction of signal RMS |
| `-ss`, `--seed-start` | `int` | `0` | Starting seed for noise simulations |
| `-i`, `--instrument` | `str` | `LiteBIRD` | Instrument config: `LiteBIRD`, `Planck`, `default` |
| `-s`, `--solver` | `str` | `optax_lbfgs` | Optimizer (see [minimization](minimization.md)) |
| `-mi`, `--max-iter` | `int` | `1000` | Max optimization iterations |
| `-sp`, `--starting-params` | `float float float` | `1.54 20.0 -3.0` | Initial $[\beta_d, T_d, \beta_s]$ |
| `-ls`, `--linesearch` | `str` | `backtracking` | Linesearch: `backtracking` or `zoom` |
| `-cond`, `--cond` | flag | `False` | Enable gradient conditioning |
| `-b`, `--best-only` | flag | `False` | Only save best params (skip full results) |
| `-v`, `--use-vmap` | flag | `False` | Use `jax.vmap` for noise realizations |
| `-top_k`, `--top-k-release` | `float` | none | Fraction of constraints to release (active set) |
| `--atol` | `float` | `1e-18` | Absolute convergence tolerance |
| `--rtol` | `float` | `1e-16` | Relative convergence tolerance |
| `-o`, `--output` | `str` | `results` | Output directory |
| `--name` | `str` | auto | Override output folder name |

### Cluster Sources (`-c`)

The `-c` flag provides fine-grained control over how clusters are defined for each spectral parameter. It accepts exactly three values, one for each of $[\beta_d, T_d, \beta_s]$:

| Value | Meaning |
|---|---|
| `true` | Use precomputed pixel subsets derived from true parameter values. Only available for `-tag c1d1s1`. |
| *integer* | Run K-means clustering with this many clusters (same as `-pc`). |
| *path to `.npy`* | Load a full-sky patches file (e.g., produced by [`r_analysis bin`](r_analysis/bin.md)). |

Values can be mixed freely. When `-c` is provided, it overrides `-pc` entirely.

**Examples:**

```bash
# Use precomputed true-parameter subsets for all three parameters
kmeans-model -n 64 -c true true true -m GAL020 -tag c1d1s1

# Use binned patches from r_analysis bin
kmeans-model -n 64 \
    -c binned/patches_beta_dust.npy binned/patches_temp_dust.npy binned/patches_beta_pl.npy \
    -m GAL020 -tag c1d1s1

# Mix: true for beta_dust, 50 K-means clusters for temp_dust, file for beta_synch
kmeans-model -n 64 -c true 50 binned/patches_beta_pl.npy -m GAL020 -tag c1d1s1
```

---

## `ptep-model`

Multi-resolution component separation using HEALPix `ud_grade`. Each spectral parameter is optimized at a different angular resolution.

### Usage

```bash
# Beta_dust at nside=64, Temp_dust at 32, Beta_synch at 16
ptep-model -n 64 -ud 64 32 16 -m GAL020 -tag c1d1s1
```

### Arguments

Same as `kmeans-model` except:

| Flag | Type | Default | Description |
|---|---|---|---|
| `-ud`, `--target-ud-grade` | `float float float` | `64 32 16` | Target nside for $[\beta_d, T_d, \beta_s]$ (replaces `-pc`) |

No `--linesearch` argument. Otherwise shares all other flags with `kmeans-model`.

---

## `fgbuster-model`

Baseline comparison using the [FGBuster](https://github.com/fgbuster/fgbuster) framework with multi-resolution clustering.

### Usage

```bash
fgbuster-model -n 64 -ud 64 32 16 -m GAL020 -tag c1d1s1
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `-n`, `--nside` | `int` | `64` | HEALPix resolution |
| `-ud`, `--target-ud-grade` | `float float float` | `64 32 16` | Target nside for $[\beta_d, T_d, \beta_s]$ |
| `-m`, `--mask` | `str` | `GAL020_U` | Galactic mask |
| `-tag`, `--tag` | `str` | `c1d1s1` | Sky simulation tag |
| `-ns`, `--noise-sim` | `int` | `1` | Number of noise realizations |
| `-nr`, `--noise-ratio` | `float` | `0.1` | Noise level (default differs from other models) |
| `-ss`, `--seed-start` | `int` | `0` | Starting seed |
| `-i`, `--instrument` | `str` | `LiteBIRD` | Instrument config |
| `-mi`, `--max-iter` | `int` | `1000` | Max iterations (uses TNC solver internally) |
| `-sp`, `--starting-params` | `float float float` | `1.54 20.0 -3.0` | Initial parameters |
| `-b`, `--best-only` | flag | `False` | Only save best params |
| `-o`, `--output` | `str` | `results` | Output directory |
| `--name` | `str` | auto | Override output folder name |

:::{note}
`fgbuster-model` always uses SciPy's TNC solver internally. It does not support `--solver`, `--cond`, `--linesearch`, or `--use-vmap`.
:::

---

## Output Structure

All three models produce the same output structure:

```
results/<run_name>/
├── best_params.npz    # Optimized spectral parameters
├── mask.npy           # Binary sky mask used
└── results.npz        # Full results (maps, NLL, per-realization data)
```

The `<run_name>` encodes the configuration. For example:
```
results/kmeans_c1d1s1_BD10000_TD500_BS500_SP1.54_20.0_-3.0_LiteBIRD_GAL020_U_optax_lbfgs_condFalse_lsbacktracking_noise0/
```

Use `--name` to set a custom folder name instead.

These result folders are consumed by [`r_analysis`](r_analysis/index.md) for computing $r$ estimates, residuals, and generating plots.

---

## Masks

Available mask names:

| Mask | Description |
|---|---|
| `ALL` | Full sky |
| `GALACTIC` | Galactic plane only |
| `GAL020` | 20% Galactic cut (upper + lower) |
| `GAL020_U` | 20% upper Galactic cut |
| `GAL020_L` | 20% lower Galactic cut |
| `GAL040` | 40% Galactic cut |
| `GAL040_U` / `GAL040_L` | 40% upper/lower |
| `GAL060` | 60% Galactic cut |
| `GAL060_U` / `GAL060_L` | 60% upper/lower |

Masks can be combined with `+` (union) and `-` (subtract):

```bash
kmeans-model -m "GAL040-GAL020"   # 40% cut minus 20% cut
kmeans-model -m "GAL020+GAL040"   # union of both regions
```
