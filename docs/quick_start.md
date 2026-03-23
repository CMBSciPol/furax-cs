# Quick Start (Python API)

This guide shows how to use the `furax-cs` Python API programmatically to load data, define masks, and run component separation routines.

## Data Loading

You can load generated data and masks directly in your scripts:

```python
from furax_cs.data import load_from_cache, get_mask, save_to_cache

# Load frequency maps (automatically generates if missing)
# Returns: frequencies (Hz), maps (shape: [freqs, 3, npix])
save_to_cache(nside=64, sky="c1d1s1")
nu, freq_maps = load_from_cache(nside=64, sky="c1d1s1")

# Load a Mask (e.g., 59% sky  for example all except the galactic plane)
mask = get_mask("ALL-GALACTIC", nside=64)
```

## Running Component Separation Programmatically

You can use the Python API to run routines like the adaptive K-means model. This offers more flexibility than the CLI.

```python
from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from furax.obs import (
    negative_log_likelihood,
    sky_signal,
)
from furax.obs.stokes import Stokes
from jax_healpy.clustering import get_cutout_from_mask, get_fullmap_from_cutout

from furax_cs import generate_noise_operator, kmeans_clusters, minimize
from furax_cs.data import (
    get_instrument,
    get_mask,
    load_cmb_map,
    load_from_cache,
    save_cmb_map,
    save_to_cache,
)

save_to_cache(nside=64, sky="c1d0s0")  # to be called only once
save_cmb_map(nside=64, sky="c1d0s0")  # to be called only once
nu, freq_maps = load_from_cache(nside=64, sky="c1d0s0")
cmb_map = load_cmb_map(nside=64, sky="c1d0s0")

# Create a StokesPytree out of the frequency maps
d = Stokes.from_stokes(freq_maps[:, 1], freq_maps[:, 2])
cmb_map = Stokes.from_stokes(cmb_map[1], cmb_map[2])
mask = get_mask(
    "ALL-GALACTIC", nside=64
)  # Supports logical + and - operations on masks (ALL,GALACTIC, GAL020, GAL040, GAL060)

# Recover valid indices for jit safe code
(indices,) = jnp.where(mask)
masked_d = get_cutout_from_mask(d, indices, axis=1)
cmb_map = get_cutout_from_mask(cmb_map, indices)

sky_signal_fn = partial(sky_signal, dust_nu0=160.0, synchrotron_nu0=23.0)
negative_log_likelihood_fn = partial(
    negative_log_likelihood,
    dust_nu0=160.0,
    synchrotron_nu0=23.0,
    analytical_gradient=True,  # Be sure to enable analytical gradients for stability
)

instrument = get_instrument("LiteBIRD")
noised_d, N,_ = generate_noise_operator(
    jax.random.key(0),
    noise_ratio=0.3,
    indices=indices,
    nside=64,
    masked_d=masked_d,
    instrument=instrument,
    stokes_type="QU",
)

patch_count = {
    "temp_dust_patches": 200,
    "beta_dust_patches": 20,
    "beta_pl_patches": 500,
}

# Max patches is used so this function is jittable
guess_clusters = kmeans_clusters(
    jax.random.key(1), mask, indices, patch_count, max_patches=patch_count
)
initial_params = {
    "beta_dust": jnp.full((20,), 1.6),
    "temp_dust": jnp.full((200,), 20.0),
    "beta_pl": jnp.full((500,), -3.0),
}
lower = {"beta_dust": 0.5, "temp_dust": 5.0, "beta_pl": -5.0}
upper = {"beta_dust": 3.0, "temp_dust": 40.0, "beta_pl": -1.0}

final_params, final_state = minimize(
    fn=negative_log_likelihood_fn,
    init_params=initial_params,
    solver_name="ADABK0",
    max_iter=1000,
    atol=1e-15,
    rtol=1e-10,
    lower_bound=lower,
    upper_bound=upper,
    nu=nu,
    N=N,
    d=noised_d,
    patch_indices=guess_clusters,
)

# Check convergence
print(f"Best loss: {final_state.best_loss:.6e}")

reconstructed_cmb = sky_signal_fn(
    final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters
)["cmb"]

# Calculate residuals on the valid pixels
residuals = cmb_map - reconstructed_cmb
rms_q = jnp.sqrt(jnp.mean(residuals.q**2))
rms_u = jnp.sqrt(jnp.mean(residuals.u**2))

print(f"Residual RMS (Q): {rms_q:.2e}")
print(f"Residual RMS (U): {rms_u:.2e}")

full_map = get_fullmap_from_cutout(reconstructed_cmb, indices, nside=64)
full_cmb = get_fullmap_from_cutout(cmb_map, indices, nside=64)

fig = plt.figure(figsize=(8, 8))
hp.mollview(full_map.q, title="Reconstructed CMB Q", sub=(2, 2, 1))
hp.mollview(full_map.u, title="Reconstructed CMB U", sub=(2, 2, 2))
hp.mollview(full_cmb.q - full_map.q, title="Residuals Q", sub=(2, 2, 3), cmap="RdBu_r")
hp.mollview(full_cmb.u - full_map.u, title="Residuals U", sub=(2, 2, 4), cmap="RdBu_r")
plt.show()
```
