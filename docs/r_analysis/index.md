# r_analysis

The `r_analysis` CLI tool provides a complete workflow for analyzing component separation results: computing statistics, plotting metrics, validating optimization, and estimating the tensor-to-scalar ratio $r$.

## Subcommand Structure

```bash
r_analysis <subcommand> [arguments]
```

| Subcommand | Purpose |
|---|---|
| [`snap`](snap.md) | Compute statistics from result folders and save to `.parquet` files |
| [`bin`](bin.md) | Bin spectral parameters and produce `.npy` patch files for re-running |
| [`plot`](plot.md) | Generate plots from `.parquet` snapshot files |
| [`validate`](validate.md) | Run NLL perturbation analysis on results |
| [`estimate`](estimate.md) | Standalone $r$ estimation from a spectrum or map file |

## Typical Workflow

```bash
# 1. Run component separation (produces result folders)
kmeans-model -n 64 -pc 10000 500 500 -m GAL020 -tag c1d1s1

# 2. Compute statistics and save snapshots
r_analysis snap -n 64 -r "kmeans_BD10000" -ird results/ -o snapshots/kmeans_BD10000.parquet

# 2b. (Optional) Bin parameters and re-run with binned patches
r_analysis bin -n 64 -r "kmeans_BD10000" -ird results/ -o binned/ --bin-bd 50 --bin-td 20 --bin-bs 30
kmeans-model -n 64 -c binned/patches_beta_dust.npy binned/patches_temp_dust.npy binned/patches_beta_pl.npy -m GAL020 -tag c1d1s1

# 3. Plot from the snapshots
r_analysis plot --parquet-dir snapshots/ -arc -as -ar

# 4. (Optional) Validate optimization quality
r_analysis validate -n 64 -r "kmeans_BD10000" -ird results/ --steps 5
```

## Common Arguments

The `snap`, `bin`, and `validate` subcommands share these arguments:

| Flag | Type | Default | Description |
|---|---|---|---|
| `-n`, `--nside` | `int` | `64` | HEALPix resolution |
| `-i`, `--instrument` | `str` | `LiteBIRD` | Instrument config (`LiteBIRD`, `Planck`, `default`) |
| `-r`, `--runs` | `str` (list) | *required* | Run name patterns to filter result folders (see [snap](snap.md) for matching modes) |
| `-ird`, `--input-results-dir` | `str` (list) | *required* | Directory(ies) containing result folders |
| `--no-tex` | flag | `False` | Disable LaTeX rendering in plots |
| `--sky` | `str` | `c1d0s0` | Sky model tag for true parameter lookup |
| `-mi`, `--max-iterations` | `int` | `1000` | Max iterations for systematic residual computation |
| `-s`, `--solver` | `str` | `optax_lbfgs` | Solver for re-optimization |

```{toctree}
:maxdepth: 2

snap
bin
plot
validate
estimate
```
