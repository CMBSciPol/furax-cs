# plot — Visualize Results

The `plot` subcommand reads `.parquet` snapshot files (produced by [`snap`](snap.md)) and generates publication-quality plots. No heavy computation is performed — it is purely a visualization tool.

## Basic Usage

```bash
# Plot r vs clusters and all spectra
r_analysis plot --parquet-dir snapshots/ -arc -as

# Plot everything
r_analysis plot --parquet-dir snapshots/ -a

# Filter to specific runs
r_analysis plot --parquet-dir snapshots/ -r "kmeans_BD4000" -ar -as
```

## Input

| Flag | Type | Default | Description |
|---|---|---|---|
| `--parquet-dir` | `str` (1+) | *required* | Directory(ies) containing `.parquet` files |
| `-r`, `--runs` | `str` (list) | all | Regex patterns to filter parquet rows by keyword |

## Visualization Toggles

### Per-Run Plots

These generate one plot per run entry in the parquet file:

| Flag | Description |
|---|---|
| `-pp`, `--plot-params` | Spectral parameter maps ($\beta_d$, $T_d$, $\beta_s$) |
| `-pt`, `--plot-patches` | Patch/cluster assignment maps |
| `-ps`, `--plot-cl-spectra` | $C_\ell^{BB}$ power spectra (one per run) |
| `-pc`, `--plot-cmb-recon` | CMB Q/U reconstruction maps |
| `-psm`, `--plot-systematic-maps` | Systematic residual maps |
| `-ptm`, `--plot-statistical-maps` | Statistical residual maps |
| `-pr`, `--plot-r-estimation` | $r$ likelihood curve (single run) |
| `-ppr`, `--plot-params-residuals` | Parameter maps with residuals vs truth |
| `-pi`, `--plot-illustrations` | Illustration plots |

### Aggregate Plots (Multi-Run)

These overlay multiple runs on a single figure:

| Flag | Description |
|---|---|
| `-as`, `--plot-all-spectra` | All $C_\ell^{BB}$ spectra overlaid |
| `-ar`, `--plot-all-r-estimation` | $r$ likelihood comparison across runs |
| `-ah`, `--plot-all-histograms` | Histograms of parameters across all runs |

### Correlation Plots

These plot metrics as a function of run properties (e.g., number of clusters):

| Flag | Description |
|---|---|
| `-arc`, `--plot-r-vs-c` | $r$ vs number of clusters |
| `-avc`, `--plot-v-vs-c` | Variance vs number of clusters |
| `-anlc`, `--plot-nll-vs-c` | NLL vs number of clusters |
| `-arv`, `--plot-r-vs-v` | $r$ vs variance |

### All-in-One

| Flag | Description |
|---|---|
| `-a`, `--plot-all` | Enable all plot types above |

## Group Plots

Group multiple runs together for aggregate comparison:

```bash
r_analysis plot --parquet-dir snapshots/ \
    -g "kmeans_BD4000" "ptep_BD64" \
    -gt "K-Means" "PTEP" \
    -ar -as
```

| Flag | Type | Description |
|---|---|---|
| `-g`, `--groups` | `str` (list) | Regex patterns defining named groups |
| `-gt`, `--group-titles` | `str` (list) | Human-readable titles for each group |
| `-t`, `--title` | `str` (list) | Per-row title overrides (curve labels) |

## Figure Customization

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output` | `str` | `plots/` | Output directory |
| `--output-format` | `str` | `png` | Output format: `png`, `pdf`, or `show` |
| `--font-size` | `int` | `14` | Font size for all text |
| `--color` | `str` (list) | auto | Custom color list for curves (cycles if fewer than runs) |
| `--xlim` | `float float` | auto | X-axis limits for $r$-estimation plots |
| `--r-legend-anchor` | `float float` | auto | Legend position `(x, y)` for $r$ plots |
| `--s-legend-anchor` | `float float` | auto | Legend position `(x, y)` for spectra plots |
| `--r-figsize` | `float float` | auto | Figure size `(w, h)` in inches for $r$ plots |
| `--s-figsize` | `float float` | auto | Figure size `(w, h)` in inches for spectra plots |
| `--r-range` | `float float` | none | $r$ fill-between range for spectra plot |
| `--r-plot` | `float float` | none | Two truth $r$ values shown as vertical lines |

## Examples

### Compare K-Means configurations

```bash
r_analysis plot --parquet-dir snapshots/ \
    -r "kmeans" \
    -arc -ar \
    --output-format pdf \
    -o figures/
```

### Side-by-side K-Means vs PTEP

```bash
r_analysis plot --parquet-dir snapshots/ \
    -g "kmeans" "ptep" \
    -gt "Adaptive K-Means" "Multi-Resolution (PTEP)" \
    -as -ar \
    --r-figsize 10 6 \
    --font-size 16
```

### Publication-quality r likelihood

```bash
r_analysis plot --parquet-dir snapshots/ \
    -r "kmeans_BD4000_GAL020" \
    -ar \
    --xlim -0.01 0.02 \
    --r-plot 0.0 0.004 \
    --output-format pdf \
    --font-size 18
```
