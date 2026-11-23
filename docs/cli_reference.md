# CLI Reference

This guide covers all command-line tools provided by `furax-cs` for data generation, component separation, and analysis.

## Workflow Overview

```
generate_data  →  kmeans-model / ptep-model / fgbuster-model  →  r_analysis snap  →  r_analysis plot
```

1. **Generate** simulated frequency maps
2. **Run** component separation (produces result folders)
3. **Analyze** results with `r_analysis`

Auxiliary tools: [`compute-mr`](#compute-mr) counts the patches of a multi-resolution setup, [`distributed-gridding`](#distributed-gridding) runs a grid search of the spectral parameters, and [`bench-bcp`](#bench-bcp) and [`bench-clusters`](#bench-clusters) benchmark FGBuster against FURAX.

## `generate_data`

Pre-generate and cache frequency maps (CMB + Dust + Synchrotron).

```bash
generate_data --nside 64 --sky c1d1s1
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--nside` | `int ...` | `4 8 32 64 128` | HEALPix resolution(s) |
| `--sky` | `str ...` | `c1d0s0 c1d1s1` | PySM sky configuration tag(s) |
| `--noise-ratio` | `float ...` | `0.0 1.0` | Noise level(s) to generate (`0.0` = no noise, `1.0` = nominal noise) |
| `--instrument` | `str` | `LiteBIRD` | Instrument configuration |

Each flag takes one or several values; the maps are generated for every combination.

The `--sky` tag follows PySM naming: `c` (CMB), `d` (dust), `s` (synchrotron), with a digit indicating the model index. Use `0` to disable a component.

**Synthetic skies with known patches:** a tag `<sky>_synt_bd<N>_td<N>_bs<N>` (e.g. `c1d1s1_synt_bd100_td50_bs50`) takes the dust and synchrotron templates of `<sky>` and rescales them with piecewise-constant spectral parameters: `N` full-sky K-means patches for each of $\beta_d$, $T_d$ and $\beta_s$, with values drawn around $(1.54, 20, -3)$ with a standard deviation of 0.2.

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
| `-ls`, `--linesearch` | `str` | `backtracking` | Linesearch of `optax_lbfgs`: `backtracking` or `zoom` |
| `-cond`, `--cond` | flag | `False` | Enable gradient conditioning (ignored by the self-conditioned solvers `ADABK{N}`, `scipy_tnc`, `scipy_cobyqa`) |
| `-b`, `--best-only` | flag | `False` | Skip the minimisation; only save the input maps (`best_params.npz`) and the mask |
| `--vmap-batch` | `int` | `1` | Noise realizations per `jax.vmap` batch; `1` runs them one at a time |
| `--cooldown` | `int` | `20` | AdaTopK (`ADABK{N}`): iterations after a constraint release before it may stop |
| `--min-steps` | `int` | `10` | No effect (kept for existing scripts); only appears in the output folder name |
| `-v`, `--verbose` | flag | `False` | AdaTopK (`ADABK{N}`): print per-iteration diagnostics |
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

The flags of `kmeans-model`, with these differences:

| Flag | Type | Default | Description |
|---|---|---|---|
| `-ud`, `--target-ud-grade` | `float float float` | `64 32 16` | Target nside for $[\beta_d, T_d, \beta_s]$ (replaces `-pc` and `-c`) |

`ptep-model` has no `-pc`, `-c` or `-ls` flag.

---

## `fgbuster-model`

Baseline comparison using the [FGBuster](https://github.com/fgbuster/fgbuster) framework, with multi-resolution patches (`-ud`) or, when `-pc` or `-c` is given, K-means or precomputed patches.

### Usage

```bash
fgbuster-model -n 64 -ud 64 32 16 -m GAL020 -tag c1d1s1
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `-n`, `--nside` | `int` | `64` | HEALPix resolution |
| `-ud`, `--target-ud-grade` | `float float float` | `64 32 16` | Target nside for $[\beta_d, T_d, \beta_s]$, used when `-pc` and `-c` are not given |
| `-pc`, `--patch-count` | `int int int` | none | K-means patch counts for $[\beta_d, T_d, \beta_s]$ (instead of `-ud`) |
| `-c`, `--clusters` | `str str str` | none | Cluster sources, as in [`kmeans-model`](#cluster-sources-c); overrides `-pc` |
| `-m`, `--mask` | `str` | `GAL020_U` | Galactic mask |
| `-tag`, `--tag` | `str` | `c1d1s1` | Sky simulation tag |
| `-ns`, `--noise-sim` | `int` | `1` | Number of noise realizations |
| `-nr`, `--noise-ratio` | `float` | `0.1` | Noise level (default differs from other models) |
| `-ss`, `--seed-start` | `int` | `0` | Starting seed |
| `-i`, `--instrument` | `str` | `LiteBIRD` | Instrument config |
| `-mi`, `--max-iter` | `int` | `1000` | Sets the TNC limit on function evaluations, `maxfun = 10 × max_iter` |
| `-sp`, `--starting-params` | `float float float` | `1.54 20.0 -3.0` | Initial parameters |
| `-b`, `--best-only` | flag | `False` | Run the fit but do not save `results.npz` |
| `-o`, `--output` | `str` | `results` | Output directory |
| `--name` | `str` | auto | Override output folder name |

:::{note}
`fgbuster-model` always uses SciPy's TNC solver internally, with `tol`, `gtol`, `ftol` and `xtol` at `1e-18`. It does not support `--solver`, `--cond`, `--linesearch`, `--vmap-batch`, `--cooldown`, `--min-steps`, `--verbose`, `--atol` or `--rtol`.
:::

---

## Output Structure

All three models produce the same output structure:

```
results/<run_name>/
├── best_params.npz    # Input maps: I_CMB, I_D (data) and I_D_NOCMB (foregrounds only)
├── mask.npy           # Binary sky mask used
└── results.npz        # Fitted spectral parameters, maps, NLL, per-realization data
```

The `<run_name>` encodes the configuration. For example:
```
results/kmeans_c1d1s1_BD10000_TD500_BS500_SP1.54_20.0_-3.0_LiteBIRD_GAL020_U_optax_lbfgs_condFalse_lsbacktracking_noise0_cd20_ms10/
```

`ptep-model` names its folders `ptep_<tag>_BD<nside>_TD<nside>_BS<nside>_SP..._<instrument>_<mask>_<solver>_cond<cond>_noise<N>_cd<cooldown>_ms<min_steps>`, and `fgbuster-model` `fgbuster_multires_...` or `fgbuster_kmeans_...` followed by the patches, instrument, mask and noise level.

Use `--name` to set a custom folder name instead.

These result folders are consumed by [`r_analysis`](r_analysis/index.md) for computing $r$ estimates, residuals, and generating plots.

---

## Masks

Each mask is the observed region. The names come from the Planck Galactic masks GAL020, GAL040 and GAL060, which keep the 20%, 40% and 60% of the sky with the least Galactic emission; furax-cs splits them into disjoint zones:

| Mask | Observed region |
|---|---|
| `ALL` | Full sky |
| `GAL020` | Planck GAL020 (the cleanest 20% of the sky) |
| `GAL040` | Planck GAL040 minus GAL020 (the next 20%) |
| `GAL060` | Planck GAL060 minus GAL040 (the next 20%) |
| `GALACTIC` | Everything outside Planck GAL060 (the Galactic plane, 40%) |
| `<zone>_U` / `<zone>_L` | Northern ($b > 0$) / southern ($b < 0$) half of `GAL020`, `GAL040` or `GAL060` |

Masks can be combined with `+` (union) and `-` (subtract):

```bash
kmeans-model -m "GAL020+GAL040"   # the cleanest 40% of the sky
kmeans-model -m "ALL-GALACTIC"    # everything except the Galactic plane (the cleanest 60%)
```

---

## `compute-mr`

Counts the patches of a multi-resolution (`ud_grade`) setup on a mask.

```bash
compute-mr --mask GAL040 --nside 256 --target-nside 8
compute-mr --mask GAL020 --nside 128 --target-nside 4 8 4
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `-m`, `--mask` | `str` | required | Mask name |
| `-n`, `--nside` | `int` | required | HEALPix resolution of the map |
| `-t`, `--target-nside` | `int ...` | required | One value for all three parameters, or three values for $[\beta_d, T_d, \beta_s]$; `0` gives a single patch |

---

## `distributed-gridding`

Grid search of the spectral parameters over the zones of a mask, distributed over the available GPUs. It can take several hours.

```bash
distributed-gridding -n 64 -ns 100 -nr 1.0 -tag c1d1s1 -m GAL020 -i LiteBIRD
distributed-gridding --dump-search-space my_search_space.yaml   # write the default grid and exit
distributed-gridding -n 64 -ss my_search_space.yaml             # use a custom grid
```

It takes the `kmeans-model` flags `-n`, `-ns`, `-nr`, `-tag`, `-m`, `-i`, `-mi`, `-sp`, `-s`, `-cond`, `-o`, `--atol` and `--rtol` (same defaults), plus:

| Flag | Type | Default | Description |
|---|---|---|---|
| `-ss`, `--search-space` | `str` | none | YAML file of the search space (here `-ss` is not `--seed-start`) |
| `-d`, `--dump-search-space` | `str` | none | Write the default search space to this YAML file and exit |
| `-c`, `--clean-up` | `str` | none | Clean up the output folder |
| `-b`, `--best-only` | flag | `False` | Only generate the best results |

---

## `bench-bcp`

Benchmarks the likelihood evaluation and the component-separation solvers of FGBuster and FURAX against nside.

| Flag | Type | Default | Description |
|---|---|---|---|
| `-n`, `--nsides` | `int ...` | `32 64 128 256 512` | Resolutions to benchmark |
| `-l`, `--likelihood` | flag | `False` | Benchmark the FGBuster and FURAX log-likelihoods |
| `-s`, `--solvers` | flag | `False` | Benchmark the solvers |
| `-c`, `--cache-run` | flag | `False` | Generate the map caches |
| `-mi`, `--max-iter` | `int` | `100` | Maximum iterations |
| `--jax-solver` | `str` | `optax_lbfgs` | FURAX solver (a `cadre.minimize` name) |
| `--fgbuster-solver` | `str` | `TNC` | FGBuster scipy method |
| `--precondition` | flag | `False` | Precondition the FURAX solve |
| `--noise` | `float` | `0.0` | Noise ratio |
| `-o`, `--output` | `str` | `comparison` | Prefix of the plot files |
| `--figsize` | `float float` | `10 6` | Figure size |

---

## `bench-clusters`

Benchmarks FGBuster and FURAX against the number of patches.

| Flag | Type | Default | Description |
|---|---|---|---|
| `-n`, `--nsides` | `int ...` | `32 64 128 256 512` | Resolutions to benchmark |
| `-cl`, `--clusters` | `int ...` | `1 5 10 ... 100` | Patch counts to benchmark |
| `-c`, `--cache-run` | flag | `False` | Generate the map caches |
| `-mi`, `--max-iter` | `int` | `1000` | Maximum iterations |
| `--tol` | `float` | `1e-15` | Convergence tolerance |
| `--jax-solver` | `str` | `SKIP` | FURAX solver (a `cadre.minimize` name), `SKIP` to leave it out |
| `--fgbuster-solver` | `str` | `SKIP` | FGBuster scipy method, `SKIP` to leave it out |
| `--precondition` | flag | `False` | Precondition the FURAX solve |
| `--noise` | `float` | `0.0` | Noise ratio |
| `--n-sims` | `int` | `3` | Noise realizations |
| `-o`, `--output` | `str` | `comparison` | Prefix of the plot files |
| `--figsize` | `float float` | `10 6` | Figure size |
