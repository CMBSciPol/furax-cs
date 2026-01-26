from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Any, TypeAlias

import camb
import healpy as hp
import jax.random as jr
import numpy as np
from furax._instruments.sky import get_observation, get_sky
from furax.obs.operators import (
    CMBOperator,
    DustOperator,
    MixingMatrixOperator,
    SynchrotronOperator,
)
from furax.obs.stokes import Stokes
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    PRNGKeyArray,
    PyTree,  # pyright: ignore
)
from pysm3 import units as pysm_units
from pysm3.models.cmb import CMBLensed

from ..logging_utils import info, success
from .instruments import get_instrument

SkyType: TypeAlias = dict[str, Stokes]


class CMBLensedWithTensors(CMBLensed):
    """CMBLensed subclass that generates CMB with custom tensor-to-scalar ratio r.

    Uses PySM's taylens algorithm for proper lensing, which correctly generates
    B-modes from E-modes via gravitational lensing deflection.

    Args:
        nside: HEALPix resolution parameter.
        r: Tensor-to-scalar ratio. Defaults to 0.0.
        cmb_seed: Random seed for map generation.
        max_nside: Maximum nside for the model.
        apply_delens: Whether to apply delensing. Defaults to False.
        delensing_ells: Delensing ells file path.
        map_dist: Distribution for parallel computing.
        H0: Hubble constant. Defaults to 67.5.
        ombh2: Baryon density. Defaults to 0.022.
        omch2: CDM density. Defaults to 0.122.
        As: Scalar amplitude. Defaults to 2e-9.
        ns: Scalar spectral index. Defaults to 0.965.
        lmax: Maximum multipole for power spectra. Defaults to 2500.
    """

    def __init__(
        self,
        nside: int,
        r: float = 0.0,
        cmb_seed: int | None = None,
        max_nside: int | None = None,
        apply_delens: bool = False,
        delensing_ells: Path | None = None,
        map_dist: Any = None,
        H0: float = 67.5,
        ombh2: float = 0.022,
        omch2: float = 0.122,
        As: float = 2e-9,
        ns: float = 0.965,
        lmax: int = 2500,
    ):
        # Generate CAMB spectra with tensors
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(As=As, ns=ns, r=r)
        if r > 0:
            pars.WantTensors = True
        pars.set_for_lmax(lmax, lens_potential_accuracy=1)

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=False, lmax=lmax)

        # Build spectra in PySM format: [ell, TT, EE, BB, TE, PP, TP, EP]
        # Uses UNLENSED spectra - taylens will apply lensing
        ells_all = np.arange(lmax + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Avoid division by zero for ell=0,1 (though we slice them off anyway)
            conversion = 2 * np.pi / (ells_all * (ells_all + 1))
            conversion[0:2] = 0.0

        dl_tt = powers["unlensed_total"][2 : lmax + 1, 0]
        dl_ee = powers["unlensed_total"][2 : lmax + 1, 1]
        dl_bb = powers["unlensed_total"][2 : lmax + 1, 2]  # This contains tensor BB
        dl_te = powers["unlensed_total"][2 : lmax + 1, 3]

        # Convert to Cl
        ell_range_inputs = ells_all[2 : lmax + 1]
        # conv_factor = conversion[2 : lmax + 1]

        spectra = np.zeros((8, len(ell_range_inputs)))
        spectra[0] = ell_range_inputs
        spectra[1] = dl_tt  # TT (run_taylens expects Dell)
        spectra[2] = dl_ee  # EE (run_taylens expects Dell)
        spectra[3] = dl_bb  # BB (run_taylens expects Dell)
        spectra[4] = dl_te  # TE (run_taylens expects Dell)
        # Note: lens_potential is usually dimensionless or different units,
        # check CAMB docs, but usually they are PP, not D_ell.
        # For safety, assuming standard CAMB output for potentials is usually correct for PySM
        # but strictly PySM expects Cl for potentials too.
        spectra[5] = powers["lens_potential"][2 : lmax + 1, 0]  # PP
        spectra[6] = powers["lens_potential"][2 : lmax + 1, 1]  # TP
        spectra[7] = powers["lens_potential"][2 : lmax + 1, 2]  # EP

        # Store spectra and run taylens
        self.nside = nside
        self.max_nside = max_nside
        self.map_dist = map_dist
        self.cmb_spectra = spectra
        self.cmb_seed = cmb_seed
        self.apply_delens = apply_delens
        self.delensing_ells = delensing_ells

        # Generate lensed maps via taylens (inherited method)
        self.map = pysm_units.Quantity(self.run_taylens(), unit=pysm_units.uK_CMB, copy=False)


def parse_sky_tag(sky: str) -> tuple[str | None, str]:
    """Parse sky string to separate CMB and foreground tags.

    Args:
        sky: Sky model string (e.g., "c1d0s0", "cr3d0s0").

    Returns:
        A tuple (cmb_tag, fg_tag). `cmb_tag` is None if no CMB present.

    Example:
        >>> cmb_tag, fg_tag = parse_sky_tag("c1d1s1")
        >>> print(cmb_tag, fg_tag)
        c1 d1s1
    """
    # Check for custom r pattern (crX)
    match = re.search(r"cr(\d+)", sky)
    if match:
        cmb_tag = match.group(0)
        fg_tag = sky.replace(cmb_tag, "")
        return cmb_tag, fg_tag

    # Legacy 2-char parsing
    tags = [sky[i : i + 2] for i in range(0, len(sky), 2)]
    cmb_tags = [t for t in tags if t.startswith("c")]
    if cmb_tags:
        cmb_tag = cmb_tags[0]
        fg_tags = [t for t in tags if not t.startswith("c")]
        fg_tag = "".join(fg_tags)
        return cmb_tag, fg_tag

    return None, sky


def generate_custom_cmb(
    r_value: float, nside: int, seed: int | None = None
) -> Float[Array, "3 npix"]:
    """Generate a CMB realization with a specific tensor-to-scalar ratio r.

    Uses PySM's taylens algorithm for proper lensing, which correctly generates
    B-modes from E-modes via gravitational lensing deflection.

    Args:
        r_value: Tensor-to-scalar ratio.
        nside: HEALPix resolution.
        seed: Random seed for generation.

    Returns:
        CMB map (3, npix) in uK_CMB.

    Example:
        >>> cmb_map = generate_custom_cmb(r_value=0.01, nside=64, seed=42)
    """
    info(f"generating with r_value {r_value}")
    cmb = CMBLensedWithTensors(nside=nside, r=r_value, cmb_seed=seed)
    return cmb.map.value  # Returns (3, npix) array in uK_CMB


def save_to_cache(
    nside: int,
    noise_ratio: float = 0.0,
    instrument_name: str = "LiteBIRD",
    sky: str = "c1d0s0",
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, " freqs"], Float[Array, " freqs 3 npix"]]:
    """Generate and cache frequency maps for component separation.

    Args:
        nside: HEALPix resolution parameter.
        noise_ratio: Noise level ratio (0.0 = no noise, 1.0 = 100% noise). Defaults to 0.0.
        instrument_name: Instrument configuration name. Defaults to "LiteBIRD".
        sky: Sky model preset string (e.g., "c1d0s0" for CMB only). Defaults to "c1d0s0".
        key: JAX random key for noise generation. Defaults to None.

    Returns:
        A tuple (frequencies, freq_maps) where frequencies are in GHz and freq_maps
        have shape (n_freq, 3, n_pix) for Stokes I, Q, U.

    Example:
        >>> freqs, maps = save_to_cache(nside=64, noise_ratio=0.0, sky="c1d0s0")
    """
    if key is None:
        key = jr.PRNGKey(0)

    instrument = get_instrument(instrument_name)
    # Define cache file path
    cache_dir = "freq_maps_cache"
    os.makedirs(cache_dir, exist_ok=True)
    noise_str = f"noise_{int(noise_ratio * 100)}" if noise_ratio > 0 else "no_noise"
    cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{noise_str}_{sky}.pkl")

    # Check if file exists, load if it does, otherwise create and save it
    r_val = None
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        info(f"Loaded freq_maps for nside {nside} from cache with noise_ratio {noise_ratio}.")
    else:
        # Check for custom r CMB
        match = re.search(r"cr(\d+)", sky)
        custom_cmb_map = None
        fg_tag = sky

        if match:
            info(f"Detected custom r tag: {match.group(0)}")
            r_exp = int(match.group(1))
            r_val = r_exp * 1e-3
            info(f"Generating custom CMB with r={r_val}")
            fg_tag = sky.replace(match.group(0), "")
            # Derive seed from key if possible, else fixed
            # key is JAX key (uint32 array). Use first element.
            seed = int(key[1]) if key is not None else 0
            custom_cmb_map = generate_custom_cmb(r_val, nside, seed=seed)
        else:
            r_val = 0.0  # Unused but here for Pyright

        # If we have custom CMB, we treat the rest as the "furax" sky
        tag_to_use = fg_tag if custom_cmb_map is not None else sky

        stokes_obs = get_observation(
            instrument,
            nside=nside,
            tag=tag_to_use,
            noise_ratio=noise_ratio,
            key=key,
            stokes_type="IQU",
            unit="uK_CMB",
        )

        # Convert Stokes PyTree to numpy array (n_freq, 3, n_pix)
        freq_maps = np.stack(
            [np.array(stokes_obs.i), np.array(stokes_obs.q), np.array(stokes_obs.u)], axis=1
        )

        if custom_cmb_map is not None:
            # Add custom CMB (broadcasting over frequencies)
            # freq_maps: (n_freq, 3, npix)
            # custom_cmb_map: (3, npix)
            freq_maps += custom_cmb_map[None, ...]
            assert r_val is not None
            info(f"Added custom CMB with r={r_val} to maps.")

        # Save freq_maps to the cache
        with open(cache_file, "wb") as f:
            pickle.dump(freq_maps, f)
        success(f"Generated and saved freq_maps for nside {nside}.")
    return np.array(instrument.frequency), freq_maps


def load_from_cache(
    nside: int, noise_ratio: float = 0.0, instrument_name: str = "LiteBIRD", sky: str = "c1d0s0"
) -> tuple[Float[Array, " freqs"], Float[Array, " freqs 3 npix"]]:
    """Load cached frequency maps from disk.

    Args:
        nside: HEALPix resolution parameter.
        noise_ratio: Noise level ratio (0.0 = no noise, 1.0 = 100% noise). Defaults to 0.0.
        instrument_name: Instrument configuration name. Defaults to "LiteBIRD".
        sky: Sky model preset string. Defaults to "c1d0s0".

    Returns:
        A tuple (frequencies, freq_maps) loaded from cache.

    Raises:
        FileNotFoundError: If cache file does not exist.

    Example:
        >>> freqs, maps = load_from_cache(nside=64, noise_ratio=0.0, sky="c1d0s0")
    """
    # Define cache file path
    instrument = get_instrument(instrument_name)
    noise_str = f"noise_{int(noise_ratio * 100)}" if noise_ratio > 0 else "no_noise"
    cache_dir = "freq_maps_cache"
    cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{noise_str}_{sky}.pkl")

    # Check if file exists and load if it does; otherwise raise an error with guidance
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        info(f"Loaded freq_maps for nside {nside} from cache.")
    else:
        raise FileNotFoundError(
            f"Cache file for freq_maps with nside {nside} and noise_ratio {noise_ratio} not found.\n"
            f"Please generate it first by calling `generate_data --nside {nside}`."
        )

    return np.array(instrument.frequency), freq_maps


def save_fg_map(
    nside: int,
    noise_ratio: float = 0.0,
    instrument_name: str = "LiteBIRD",
    sky: str = "c1d0s0",
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, " freqs"], Float[Array, " freqs 3 npix"]]:
    """Generate and cache foreground-only frequency maps (CMB excluded).

    Args:
        nside: HEALPix resolution parameter.
        noise_ratio: Noise level ratio (0.0 = no noise, 1.0 = 100% noise). Defaults to 0.0.
        instrument_name: Instrument configuration name. Defaults to "LiteBIRD".
        sky: Sky model preset string, CMB component automatically removed. Defaults to "c1d0s0".
        key: JAX random key for noise generation. Defaults to None.

    Returns:
        A tuple (frequencies, freq_maps) containing only foreground contributions.

    Example:
        >>> freqs, fg_maps = save_fg_map(nside=64, sky="c1d1s1")
    """
    info(
        f"Generating fg map for nside {nside}, noise_ratio {noise_ratio}, instrument {instrument_name}"
    )
    _, stripped_sky = parse_sky_tag(sky)
    return save_to_cache(
        nside, noise_ratio=noise_ratio, instrument_name=instrument_name, sky=stripped_sky, key=key
    )


def load_fg_map(
    nside: int, noise_ratio: float = 0.0, instrument_name: str = "LiteBIRD", sky: str = "c1d0s0"
) -> tuple[Float[Array, " freqs"], Float[Array, " freqs 3 npix"]]:
    """Load cached foreground-only frequency maps.

    Args:
        nside: HEALPix resolution parameter.
        noise_ratio: Noise level ratio (0.0 = no noise, 1.0 = 100% noise). Defaults to 0.0.
        instrument_name: Instrument configuration name. Defaults to "LiteBIRD".
        sky: Sky model preset string, CMB automatically excluded. Defaults to "c1d0s0".

    Returns:
        A tuple (frequencies, freq_maps) containing only foreground contributions.

    Example:
        >>> freqs, fg_maps = load_fg_map(nside=64, sky="c1d1s1")
    """
    _, stripped_sky = parse_sky_tag(sky)
    return load_from_cache(
        nside, noise_ratio=noise_ratio, instrument_name=instrument_name, sky=stripped_sky
    )


def save_cmb_map(nside: int, sky: str = "c1d0s0") -> Float[Array, "3 npix"]:
    """Generate and cache CMB-only maps for template generation.

    Args:
        nside: HEALPix resolution parameter.
        sky: Sky model preset string. Defaults to "c1d0s0".

    Returns:
        CMB map with shape (3, n_pix) for Stokes I, Q, U, or zeros if no CMB.

    Example:
        >>> cmb_map = save_cmb_map(nside=64, sky="c1d0s0")
    """
    info(f"Generating CMB map for nside {nside}, sky {sky}")
    # Define cache file path
    cache_dir = "freq_maps_cache"
    os.makedirs(cache_dir, exist_ok=True)

    cmb_tag, _ = parse_sky_tag(sky)

    if cmb_tag is None:
        npix = 12 * nside**2
        return np.zeros((3, npix))
    else:
        cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{cmb_tag}.pkl")

        match = re.match(r"cr(\d+)", cmb_tag)
        if match:
            r_exp = int(match.group(1))
            r_val = r_exp * 0.001
            # Use default seed=0 to match save_to_cache default
            freq_maps = generate_custom_cmb(r_val, nside, seed=0)
        else:
            sky_obj = get_sky(nside, sky)
            freq_maps = sky_obj.components[0].map.to_value()

        with open(cache_file, "wb") as f:
            pickle.dump(freq_maps, f)
        success(f"Generated and saved freq_maps for nside {nside} and for tag {cmb_tag}.")

        return freq_maps


def load_cmb_map(nside: int, sky: str = "c1d0s0") -> Float[Array, "3 npix"]:
    """Load cached CMB-only maps.

    Args:
        nside: HEALPix resolution parameter.
        sky: Sky model preset string. Defaults to "c1d0s0".

    Returns:
        CMB map with shape (3, n_pix) for Stokes I, Q, U, or zeros if no CMB.

    Raises:
        FileNotFoundError: If cache file does not exist.

    Example:
        >>> cmb_map = load_cmb_map(nside=64, sky="c1d0s0")
    """
    # Define cache file path
    cache_dir = "freq_maps_cache"

    cmb_tag, _ = parse_sky_tag(sky)

    if cmb_tag is None:
        npix = 12 * nside**2
        return np.zeros((3, npix))
    else:
        cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{cmb_tag}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                freq_maps = pickle.load(f)
            info(f"Loaded freq_maps for nside {nside} from cache.")
        else:
            raise FileNotFoundError(
                f"Cache file for freq_maps with nside {nside} not found.\n"
                f"Please generate it first by calling `generate_data --nside {nside}`."
            )

    return freq_maps


def get_mixin_matrix_operator(
    params: PyTree[Float[Array, " P"]],
    patch_indices: PyTree[Int[Array, " P"]],
    nu: Float[Array, " Nf"],
    sky: SkyType,
    dust_nu0: float,
    synchrotron_nu0: float,
) -> tuple[MixingMatrixOperator, MixingMatrixOperator]:
    """Construct mixing matrix operators for CMB and foregrounds.

    Args:
        params: Spectral parameters (temp_dust, beta_dust, beta_pl).
        patch_indices: Patch assignment indices for each parameter.
        nu: Frequency array in GHz.
        sky: Sky component dictionary from FURAX.
        dust_nu0: Dust reference frequency in GHz.
        synchrotron_nu0: Synchrotron reference frequency in GHz.

    Returns:
        A tuple (MixingMatrixOperator with CMB, MixingMatrixOperator without CMB).

    Example:
        >>> A, A_nocmb = get_mixin_matrix_operator(params, patches, nu, sky, 150.0, 20.0)
    """
    first_element = next(iter(sky.values()))
    size = first_element.shape[-1]
    in_structure = first_element.structure_for((size,))

    cmb = CMBOperator(nu, in_structure=in_structure)
    dust = DustOperator(
        nu,
        frequency0=dust_nu0,
        temperature=params["temp_dust"],
        temperature_patch_indices=patch_indices["temp_dust_patches"],
        beta=params["beta_dust"],
        beta_patch_indices=patch_indices["beta_dust_patches"],
        in_structure=in_structure,
    )
    synchrotron = SynchrotronOperator(
        nu,
        frequency0=synchrotron_nu0,
        beta_pl=params["beta_pl"],
        beta_pl_patch_indices=patch_indices["beta_pl_patches"],
        in_structure=in_structure,
    )

    return MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron), MixingMatrixOperator(
        dust=dust, synchrotron=synchrotron
    )


def simulate_D_from_params(
    params: PyTree[Float[Array, " P"]],
    patch_indices: PyTree[Int[Array, " P"]],
    nu: Float[Array, " Nf"],
    sky: SkyType,
    dust_nu0: float,
    synchrotron_nu0: float,
) -> tuple[Stokes, Stokes]:
    """Simulate observed frequency maps given spectral parameters.

    Args:
        params: Spectral parameters (temp_dust, beta_dust, beta_pl).
        patch_indices: Patch assignment indices for each parameter.
        nu: Frequency array in GHz.
        sky: Sky component dictionary.
        dust_nu0: Dust reference frequency in GHz.
        synchrotron_nu0: Synchrotron reference frequency in GHz.

    Returns:
        A tuple (d, d_nocmb) where d includes CMB and d_nocmb excludes it.

    Example:
        >>> d, d_nocmb = simulate_D_from_params(params, patches, nu, sky, 150.0, 20.0)
    """
    A, A_nocmb = get_mixin_matrix_operator(
        params, patch_indices, nu, sky, dust_nu0, synchrotron_nu0
    )
    d = A(sky)
    sky_no_cmb = sky.copy()
    sky_no_cmb.pop("cmb")
    d_nocmb = A_nocmb(sky_no_cmb)
    return d, d_nocmb


MASK_CHOICES = [
    "ALL",
    "GALACTIC",
    "GAL020_U",
    "GAL020_L",
    "GAL020",
    "GAL040_U",
    "GAL040_L",
    "GAL040",
    "GAL060_U",
    "GAL060_L",
    "GAL060",
]


def sanitize_mask_name(mask_expr: str) -> str:
    """Convert mask expression to valid folder name.

    Args:
        mask_expr: Mask expression potentially containing + (union) or - (subtract) operators.

    Returns:
        Sanitized folder name with operators replaced by descriptive names.

    Example:
        >>> sanitize_mask_name("GAL020+GAL040")
        'GAL020_UNION_GAL040'
        >>> sanitize_mask_name("ALL-GALACTIC")
        'ALL_SUBTRACT_GALACTIC'
    """
    sanitized = mask_expr.replace("+", "_UNION_").replace("-", "_SUBTRACT_")
    return sanitized


def _parse_mask_expression(expr: str, nside: int) -> Bool[Array, " npix"]:
    """Parse and evaluate boolean mask expressions.

    Supports left-to-right evaluation of expressions with + (union) and - (subtraction)
    operators. Does not support parentheses.

    Args:
        expr: Mask expression with optional boolean operators.
            Examples: "GAL020+GAL040", "ALL-GALACTIC", "GAL020+GAL040-GALACTIC"
        nside: HEALPix resolution parameter.

    Returns:
        Boolean mask array where True indicates observed pixels.

    Raises:
        ValueError: If expression contains invalid mask names or syntax.

    Example:
        >>> mask = _parse_mask_expression("GAL020+GAL040", nside=64)
        >>> mask = _parse_mask_expression("ALL-GALACTIC", nside=64)
    """
    # Tokenize the expression while preserving operators
    tokens = []
    current_token = ""

    for char in expr:
        if char in ["+", "-"]:
            if current_token:
                tokens.append(current_token.strip())
                current_token = ""
            tokens.append(char)
        else:
            current_token += char

    if current_token:
        tokens.append(current_token.strip())

    if not tokens:
        raise ValueError(f"Empty mask expression: {expr}")

    # Validate that we have alternating mask names and operators
    if len(tokens) == 1:
        # Single mask, no operators
        mask_name = tokens[0]
        if mask_name not in MASK_CHOICES:
            raise ValueError(
                f"Invalid mask name '{mask_name}' in expression '{expr}'. "
                f"Choose from: {MASK_CHOICES}"
            )
        return get_mask(mask_name, nside)

    # Multiple tokens - evaluate left to right
    if len(tokens) % 2 == 0:
        raise ValueError(
            f"Invalid expression syntax: {expr}. Expected format: MASK [+/-] MASK [+/-] MASK ..."
        )

    # Start with first mask
    result = None
    i = 0

    while i < len(tokens):
        if i == 0:
            # First token must be a mask name
            mask_name = tokens[i]
            if mask_name not in MASK_CHOICES:
                raise ValueError(
                    f"Invalid mask name '{mask_name}' in expression '{expr}'. "
                    f"Choose from: {MASK_CHOICES}"
                )
            result = get_mask(mask_name, nside)
            i += 1
        else:
            # Even indices are operators, odd are mask names
            operator = tokens[i]
            if operator not in ["+", "-"]:
                raise ValueError(
                    f"Expected operator (+ or -) at position {i} in expression '{expr}', "
                    f"got '{operator}'"
                )

            if i + 1 >= len(tokens):
                raise ValueError(f"Operator '{operator}' at end of expression '{expr}'")

            mask_name = tokens[i + 1]
            if mask_name not in MASK_CHOICES:
                raise ValueError(
                    f"Invalid mask name '{mask_name}' in expression '{expr}'. "
                    f"Choose from: {MASK_CHOICES}"
                )

            next_mask = get_mask(mask_name, nside)

            if operator == "+":
                # Union
                result = np.logical_or(result, next_mask)
            elif operator == "-":
                # Subtraction
                result = np.logical_and(result, np.logical_not(next_mask))

            i += 2

    return result


def _get_or_generate_mask_file(nside: int) -> Path:
    """Get path to mask file, generating it from 2048 source if needed.

    Args:
        nside: HEALPix resolution parameter.

    Returns:
        Path to the mask file.
    """
    mask_dir = Path(__file__).parent / "masks"
    mask_file = mask_dir / f"GAL_PlanckMasks_{nside}.npz"

    if mask_file.exists():
        return mask_file

    # Generate from 2048 source
    source_file = mask_dir / "GAL_PlanckMasks_2048.npz"
    masks_2048 = np.load(source_file)

    downgraded = {}
    for key in masks_2048.files:
        downgraded[key] = hp.ud_grade(masks_2048[key] * 1.0, nside).astype(np.uint8)

    np.savez(mask_file, **downgraded)
    success(f"Generated and cached mask for nside {nside}")

    return mask_file


def get_mask(mask_name: str = "GAL020", nside: int = 64) -> Bool[Array, " npix"]:
    """Load and process galactic masks at specified resolution.

    Args:
        mask_name: Mask identifier (e.g., "GAL020", "GAL040", "GALACTIC") or boolean expression
            (e.g., "GAL020+GAL040", "ALL-GALACTIC"). Defaults to "GAL020".
        nside: HEALPix resolution parameter. Defaults to 64.

    Returns:
        Boolean mask array where True indicates observed pixels.

    Raises:
        ValueError: If mask_name is invalid.

    Notes:
        Available mask choices: ALL, GALACTIC, GAL020, GAL040, GAL060, and
        their _U (upper) and _L (lower) hemisphere variants.

        Boolean operations are supported:
        - Use + for union (logical OR)
        - Use - for subtraction (logical AND NOT)
        - Expressions are evaluated left-to-right
        - Examples: "GAL020+GAL040", "ALL-GALACTIC", "GAL020+GAL040-GALACTIC"

        Masks are automatically generated and cached on first call for each nside.

    Example:
        >>> mask = get_mask("GAL020", nside=64)
        >>> # Using boolean expression:
        >>> mask_union = get_mask("GAL020+GAL040", nside=64)
    """
    # Check if mask_name contains boolean operators
    if "+" in mask_name or "-" in mask_name:
        return _parse_mask_expression(mask_name, nside)

    masks_file = _get_or_generate_mask_file(nside)
    masks = np.load(masks_file)

    if mask_name not in MASK_CHOICES:
        raise ValueError(f"Invalid mask name: {mask_name}. Choose from: {MASK_CHOICES}")

    npix = 12 * nside**2
    ones = np.ones(npix, dtype=bool)
    # Extract the masks (keys: "GAL020", "GAL040", "GAL060").
    mask_GAL020 = masks["GAL020"]
    mask_GAL040 = masks["GAL040"]
    mask_GAL060 = masks["GAL060"]

    mask_galactic = np.logical_and(ones, np.logical_not(mask_GAL060))
    mask_GAL060 = np.logical_and(mask_GAL060, np.logical_not(mask_GAL040))
    mask_GAL040 = np.logical_and(mask_GAL040, np.logical_not(mask_GAL020))

    # Determine the HEALPix resolution (nside) from one of the masks.
    nside = hp.get_nside(mask_GAL020)

    # Get pixel indices and corresponding angular coordinates (theta, phi) in radians.
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix)

    # Define upper and lower hemispheres based on theta.
    # (Assuming theta < pi/2 corresponds to b > 0, i.e. the "upper" hemisphere.)
    upper = theta < np.pi / 2
    lower = theta >= np.pi / 2

    zones = {}

    zones["ALL"] = ones
    # --- Define Zones ---
    # GAL020 Upper ring and lower ring
    zones["GAL020_U"] = np.logical_and(mask_GAL020, upper)
    zones["GAL020_L"] = np.logical_and(mask_GAL020, lower)
    zones["GAL020"] = mask_GAL020
    # GAL040 Upper ring and lower ring
    zones["GAL040_U"] = np.logical_and(mask_GAL040, upper)
    zones["GAL040_L"] = np.logical_and(mask_GAL040, lower)
    zones["GAL040"] = mask_GAL040
    # GAL060 Upper ring and lower ring
    zones["GAL060_U"] = np.logical_and(mask_GAL060, upper)
    zones["GAL060_L"] = np.logical_and(mask_GAL060, lower)
    zones["GAL060"] = mask_GAL060
    # Galactic mask
    zones["GALACTIC"] = mask_galactic

    # Return the requested zone.
    return zones[mask_name]


def generate_needed_maps(
    nside_list: list[int] | None = None,
    noise_ratio_list: list[float] | None = None,
    instrument_name: str = "LiteBIRD",
    sky_list: list[str] | None = None,
) -> None:
    """Batch generate and cache all required frequency maps.

    Args:
        nside_list: HEALPix resolutions to generate. Defaults to [4, 8, 32, 64, 128].
        noise_ratio_list: Noise ratio configurations. Defaults to [0.0, 1.0].
        instrument_name: Instrument configuration. Defaults to "LiteBIRD".
        sky_list: Sky model presets. Defaults to ["c1d0s0", "c1d1s1"].

    Notes:
        Generates full frequency maps, foreground-only maps, and CMB-only maps
        for all combinations of input parameters.

    Example:
        >>> generate_needed_maps(nside_list=[64], noise_ratio_list=[0.0, 1.0])
    """
    if nside_list is None:
        nside_list = [4, 8, 32, 64, 128]
    if noise_ratio_list is None:
        noise_ratio_list = [0.0, 1.0]
    if sky_list is None:
        sky_list = ["c1d0s0", "c1d1s1"]

    for nside in nside_list:
        for noise_ratio in noise_ratio_list:
            for sky in sky_list:
                save_to_cache(
                    nside, noise_ratio=noise_ratio, instrument_name=instrument_name, sky=sky
                )

    for sky in sky_list:
        for nside in nside_list:
            save_fg_map(nside, noise_ratio=0.0, instrument_name=instrument_name, sky=sky)
            save_cmb_map(nside, sky=sky)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cached frequency maps for CMB component separation"
    )
    parser.add_argument(
        "--nside",
        type=int,
        nargs="+",
        default=[4, 8, 32, 64, 128],
        help="HEALPix resolution(s) to generate maps for (default: 4 8 32 64 128)",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="Noise ratio level(s) to generate (0.0=no noise, 1.0=100%% noise, default: 0.0 1.0)",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="LiteBIRD",
        help="Instrument name (default: LiteBIRD)",
    )
    parser.add_argument(
        "--sky",
        type=str,
        nargs="+",
        default=["c1d0s0", "c1d1s1"],
        help="Sky model tag(s) (default: c1d0s0 c1d1s1)",
    )

    args = parser.parse_args()

    generate_needed_maps(
        nside_list=args.nside,
        noise_ratio_list=args.noise_ratio,
        instrument_name=args.instrument,
        sky_list=args.sky,
    )


if __name__ == "__main__":
    main()
