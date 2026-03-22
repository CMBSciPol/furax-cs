# estimate — Standalone $r$ Estimation

The `estimate` subcommand computes the tensor-to-scalar ratio $r$ from a CMB spectrum or map file, without requiring full component separation results.

## Basic Usage

```bash
# From a pre-computed BB power spectrum
r_analysis estimate --cmb cl_bb.npy --fsky 0.8

# From a CMB QU map
r_analysis estimate --cmb cmb_qu_map.npy

# From a CMB map with reconstructed CMB and systematics
r_analysis estimate \
    --cmb cmb_true.npy \
    --cmb-hat cmb_reconstructed.npy \
    --syst systematic_residuals.npy
```

## Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cmb` | `str` | *required* | Path to CMB data (`.npy`). Accepts: 1D spectrum, 2D map `(2, npix)` for QU, or 3D `(3, npix)` for IQU |
| `--cmb-hat` | `str` | none | Path to reconstructed CMB maps `(n_real, 2, npix)` or `(n_real, 3, npix)` |
| `--syst` | `str` | none | Path to systematic residual map `(2, npix)` or `(3, npix)` |
| `--fsky` | `float` | inferred | Sky fraction. Required if input is a spectrum; inferred from map otherwise |
| `--nside` | `int` | inferred | HEALPix resolution. Inferred from map shape if not provided |
| `-o`, `--output` | `str` | none | Path to save results (`.npz` format) |
| `--output-format` | `str` | `png` | Plot format: `png`, `pdf`, `show` |

## Input Formats

The `--cmb` argument accepts three input shapes:

| Shape | Interpretation |
|---|---|
| `(N,)` | 1D $C_\ell^{BB}$ power spectrum. Requires `--fsky`. |
| `(2, npix)` | QU polarization map. $f_\mathrm{sky}$ inferred from non-zero pixels. |
| `(3, npix)` | IQU polarization map. Only Q and U are used. |

## Output

When `-o` is specified, saves an `.npz` file with:
- `r_best` — Maximum likelihood estimate of $r$
- `sigma_neg`, `sigma_pos` — Asymmetric 1$\sigma$ confidence bounds
- `r_grid` — Grid of $r$ values for the likelihood curve
- `L_vals` — Likelihood values on the grid
