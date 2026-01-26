"""Noise generation utilities for CMB component separation.

This module provides utilities for generating instrumental noise and
creating noise covariance operators used in likelihood computations.
"""

from functools import partial
from typing import Literal, Optional

import jax
import jax.numpy as jnp
from furax._instruments.sky import FGBusterInstrument, get_noise_sigma_from_instrument
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_healpy.clustering import get_cutout_from_mask
from jaxtyping import Array, Int, PRNGKeyArray


@partial(jax.jit, static_argnames=("nside", "stokes_type"))
def generate_noise_operator(
    key: PRNGKeyArray,
    noise_ratio: float,
    indices: Int[Array, " n_valid"],
    nside: int,
    masked_d: Stokes,
    instrument: FGBusterInstrument,
    stokes_type: Optional[Literal["QU", "IQU"]] = None,
) -> tuple[Stokes, NoiseDiagonalOperator]:
    """Generate noised data and corresponding noise covariance operator.

    This function creates a complete noise model for CMB component separation:

    1. Generates white noise scaled by instrument sensitivity.
    2. Adds noise to the input data.
    3. Creates a diagonal noise covariance operator for the likelihood.

    Args:
        key: JAX PRNG key for reproducible noise generation.
        noise_ratio: Noise level as fraction of signal. Use 0.0 for noiseless analysis.
            Typical values: 0.1 (10%), 1.0 (100%), etc.
        indices: Indices of unmasked pixels from the full-sky map.
        nside: HEALPix resolution parameter.
        masked_d: Input data (already masked/cutout), Stokes object with Q and U.
        instrument: Instrument configuration containing frequency bands and noise specs.
        stokes_type: Stokes parameters to use. If None, inferred from ``masked_d.stokes``.
            ``"QU"`` for polarization-only, ``"IQU"`` for intensity + polarization.

    Returns:
        A tuple containing:
            - **noised_d**: Data with added instrumental noise.
            - **N**: Diagonal noise covariance operator for use in likelihood computation.

    Example:
        >>> from furax_cs.data import get_instrument, get_mask
        >>> from jax_healpy.clustering import get_cutout_from_mask
        >>> instrument = get_instrument("LiteBIRD")
        >>> mask = get_mask("GAL020")
        >>> (indices,) = jnp.where(mask == 1)
        >>> masked_d = get_cutout_from_mask(d, indices, axis=1)
        >>> noised_d, N = generate_noise_operator(
        ...     key=jax.random.key(42),
        ...     noise_ratio=1.0,
        ...     indices=indices,
        ...     nside=64,
        ...     masked_d=masked_d,
        ...     instrument=instrument,
        ... )

    Notes:
        When ``noise_ratio=0``, the noise operator uses variance=1.0 to avoid
        singular matrices in the likelihood computation.

        The noise model assumes:
        - White (uncorrelated) noise between pixels and frequencies.
        - Diagonal noise covariance (independent noise per measurement).
        - Noise variance determined by instrument sensitivity.
    """
    # Infer stokes type from data structure if not provided
    if stokes_type is None:
        stokes_type = masked_d.stokes

    # Create frequency landscape for noise generation
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, stokes_type)

    # Generate white noise and scale by noise_ratio
    white_noise = f_landscapes.normal(key) * noise_ratio

    # Extract cutout (only unmasked pixels)
    white_noise = get_cutout_from_mask(white_noise, indices, axis=1)

    # Get instrument noise sigma
    sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type=stokes_type)

    # Scale noise by instrument sensitivity
    noise = white_noise * sigma

    # Add noise to data
    noised_d = masked_d + noise

    # Compute noise variance for covariance operator
    # When noise_ratio=0, use 1.0 to avoid singular N
    small_n = jax.tree.map(
        lambda s: jnp.where(noise_ratio == 0, 1.0, (s * noise_ratio) ** 2), sigma
    )
    # Create diagonal noise covariance operator
    N = NoiseDiagonalOperator(small_n, _in_structure=masked_d.structure)

    return noised_d, N, small_n
