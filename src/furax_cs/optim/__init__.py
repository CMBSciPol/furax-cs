"""Compatibility shim — optimization has moved to the cadre package.

Import from cadre directly:
    from cadre import minimize, get_solver, condition, ...
"""

from cadre import *  # noqa: F401, F403
from cadre import __all__  # noqa: F401
