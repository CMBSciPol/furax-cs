# validate — NLL Perturbation Analysis

The `validate` subcommand performs profile likelihood analysis by perturbing optimized parameters and measuring the NLL response. This helps verify that the optimizer has converged to a true minimum.

## Basic Usage

```bash
r_analysis validate \
    -n 64 \
    -r "kmeans_BD10000" \
    -ird results/ \
    --steps 5 \
    --scales 1e-3 1e-4
```

## Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--steps` | `int` | `5` | Number of perturbation steps in each direction |
| `--scales` | `float` (list) | `1e-3 1e-4` | Perturbation scale factors |
| `--noise-selection` | `str` | `min-value` | Noise realization: `min-value`, `min-nll`, or integer index |

### Perturbation Targets

Control which parameters are perturbed:

| Flag | Default | Description |
|---|---|---|
| `--perturb-beta-dust` | `all` | Which $\beta_d$ indices to perturb |
| `--perturb-temp-dust` | `all` | Which $T_d$ indices to perturb |
| `--perturb-beta-pl` | `all` | Which $\beta_s$ indices to perturb |

Perturbation spec values:
- `all` — perturb all indices
- `-1` — skip this parameter entirely
- `0:30` — slice notation (indices 0 through 29)
- `0,1,2` — specific indices
- `max` — only the index with the largest value
- `min` — only the index with the smallest value

### Plot Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--plot-type` | `str` (list) | `nll-grad` | Plot types: `nll-grad`, `nll`, `grad`, `grad-maps-{idx}` |
| `--aggregate` | flag | `False` | Overlay all runs on the same plot |
| `--no-vmap` | flag | `False` | Use for-loop instead of vmap (less memory) |
| `-t`, `--titles` | `str` (list) | auto | Custom titles for each plot |
| `-o`, `--output` | `str` | `plots/` | Output directory |
| `--output-format` | `str` | `png` | Format: `png`, `pdf`, `show` |
| `--font-size` | `int` | `14` | Font size |

Plus all [common arguments](index.md#common-arguments).

## Example: Validate specific parameters

```bash
r_analysis validate \
    -n 64 -r "kmeans_BD4000_GAL020" -ird results/ \
    --perturb-beta-dust "0:50" \
    --perturb-temp-dust "-1" \
    --perturb-beta-pl "max" \
    --plot-type nll grad \
    --aggregate
```
