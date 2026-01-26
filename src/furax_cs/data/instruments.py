from pathlib import Path

import numpy as np
import yaml
from furax._instruments.sky import FGBusterInstrument


def get_instrument(instrument_name: str) -> FGBusterInstrument:
    """Get an instrument configuration by name.

    Args:
        instrument_name: Name of the instrument (e.g., "LiteBIRD", "Planck").
            Must correspond to an entry in `instruments.yaml`.
            Use "default" for the FGBuster default instrument.

    Returns:
        The instrument configuration object with frequency bands and sensitivities.

    Raises:
        ValueError: If `instrument_name` is not found in the configuration.

    Example:
        >>> instrument = get_instrument("LiteBIRD")
        >>> print(instrument.frequency)
    """
    current_dir = Path(__file__).parent
    with open(f"{current_dir}/instruments.yaml") as f:
        instruments = yaml.safe_load(f)

    if instrument_name == "default":
        return FGBusterInstrument.default_instrument()

    if instrument_name not in instruments:
        raise ValueError(f"Unknown instrument {instrument_name}.")

    instrument_yaml = instruments[instrument_name]
    frequency = np.asarray(instrument_yaml["frequency"])
    depth_i = np.asarray(instrument_yaml["depth_i"])
    depth_p = np.asarray(instrument_yaml["depth_p"])

    return FGBusterInstrument(frequency, depth_i, depth_p)
