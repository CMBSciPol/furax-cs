# snap — Compute & Save Statistics

The `snap` subcommand reads result folders produced by `kmeans-model`, `ptep-model`, or `fgbuster-model`, computes statistics (residuals, power spectra, $r$ estimation), and saves them to lightweight `.parquet` files for later plotting.

## Basic Usage

```bash
r_analysis snap \
    -n 64 \
    -r "kmeans_BD10000_TD500_BS500_GAL020" \
    -ird results/ \
    -o snapshots/my_run.parquet
```

## Run Matching with `-r`

The `-r` flag controls which result folders are selected. It supports three matching modes depending on the pattern syntax.

### Token Matching (Exact)

When the pattern contains no regex metacharacters, it is split by `_` into tokens. A folder matches if **all tokens** are present in the folder name (AND logic).

```bash
# Match folders containing BOTH "kmeans" AND "BD10000"
r_analysis snap -r "kmeans_BD10000" -ird results/ -o out.parquet

# Match folders containing "kmeans", "BD4000", "TD500", "BS500", "GAL020"
r_analysis snap -r "BD4000_TD500_BS500_GAL020" -ird results/ -o out.parquet
```

The folder name is also split by `_`, and each token from the pattern must match at least one token in the folder name.

### Regex Matching (Expand)

When a token contains regex metacharacters (capture groups with `\d`, `\w`, etc.), each unique combination of captured values creates a **separate entry** in the output.

```bash
# Match all runs, extract BD/TD/BS/GAL values — each unique combination becomes a separate kw
r_analysis snap -r "BD(\d+)_TD(\d+)_BS(\d+)_GAL(\d+)" -ird results/ -o out.parquet
```

For example, if `results/` contains:
```
kmeans_c1d1s1_BD4000_TD500_BS500_..._GAL020_...
kmeans_c1d1s1_BD8000_TD500_BS500_..._GAL020_...
kmeans_c1d1s1_BD4000_TD500_BS500_..._GAL040_...
```

The pattern `BD(\d+)_TD(\d+)_BS(\d+)_GAL(\d+)` produces three separate entries:
- `BD4000_TD500_BS500_GAL020`
- `BD8000_TD500_BS500_GAL020`
- `BD4000_TD500_BS500_GAL040`

### Partial Matching (Merge Masks)

You can use a partial pattern to match folders across different mask configurations. All matched folders are merged into a single entry.

```bash
# Match all runs with BD4000_TD500_BS500 regardless of mask
# This effectively merges GAL020 + GAL040 + GAL060 masks → fsky ≈ 60%
r_analysis snap -r "BD4000_TD500_BS500" -ird results/ -o out.parquet
```

## Combining Runs

### `--combine`

Merge all matched result directories into a **single entry** rather than keeping them separate:

```bash
r_analysis snap -r "kmeans_BD4000" "ptep_BD64" \
    -ird results/ \
    --combine \
    --name "combined_run" \
    -o out.parquet
```

### `--name`

Set display names for each run group:

```bash
r_analysis snap -r "kmeans_BD4000" "ptep_BD64" \
    -ird results/ \
    --name "K-Means (4000)" "PTEP (64)" \
    -o out.parquet
```

by default, the display name is the matched pattern (e.g., `BD4000_TD500_BS500`).
For combined runs, it is recommended to give an explicit name so it easier to match when plotting using [plot](plot.md)

::::{seealso}
For reducing the number of clusters via post-clustering parameter binning, see [`bin`](bin.md).
::::

## All Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output-parquet` | `str` | *required* | Path to output `.parquet` file |
| `--noise-selection` | `str` | `min-value` | Which noise realization to use: `min-value`, `min-nll`, or an integer index |
| `--max-ns` | `int` | all | Maximum number of noise realizations |
| `--no-images` | flag | `False` | Skip rendering mollview images (faster) |
| `--combine` | flag | `False` | Merge all matched dirs into one entry |
| `--name` | `str` (list) | auto | Display names for run groups |
| `--max-size` | `int` | unlimited | Max entries per parquet file (splits into numbered files) |

Plus all [common arguments](index.md#common-arguments) (`-n`, `-r`, `-ird`, etc.).

## Output

The output is a `.parquet` file (powered by HuggingFace `datasets`) containing one row per matched run group. Each row stores:

- CMB reconstruction maps and patch assignments
- Power spectra ($C_\ell^{BB}$ observed, templates, residuals)
- $r$ estimation (best fit, confidence bounds, likelihood curve)
- Systematic and statistical residual maps
- Foreground parameter maps ($\beta_d$, $T_d$, $\beta_s$)
- Metadata (keyword, number of clusters, NLL, mask info)
