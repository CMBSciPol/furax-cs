"""Logging utilities for r_analysis package."""

import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager


class Colors:
    """ANSI color codes for terminal output."""

    BLUE: str = "\033[94m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    RED: str = "\033[91m"
    CYAN: str = "\033[96m"
    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"

    @classmethod
    def is_tty(cls) -> bool:
        """Check if stdout is a TTY (supports colors)."""
        return sys.stdout.isatty()

    @classmethod
    def disable(cls) -> None:
        """Disable all colors."""
        cls.BLUE = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.CYAN = ""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""


# Disable colors if not in a TTY
if not Colors.is_tty():
    Colors.disable()


def info(message: str) -> None:
    """Print an informational message.

    Args:
        message: Message to print.

    Example:
        >>> info("Loading data...")
        [INFO] Loading data...
    """
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")


def success(message: str) -> None:
    """Print a success message with checkmark.

    Args:
        message: Message to print.

    Example:
        >>> success("Data loaded!")
        ✓ Data loaded!
    """
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message to print.

    Example:
        >>> warning("Low memory.")
        [WARNING] Low memory.
    """
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")


def error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message to print.

    Example:
        >>> error("File not found.")
        [ERROR] File not found.
    """
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}", file=sys.stderr)


def hint(message: str) -> None:
    """Print a hint message for user guidance.

    Args:
        message: Hint message to print.

    Example:
        >>> hint("Try running with --verbose.")
        [HINT] Try running with --verbose.
    """
    print(f"{Colors.CYAN}[HINT]{Colors.RESET} {message}")


def debug(message: str) -> None:
    """Print a debug message (dimmed).

    Args:
        message: Debug message to print.

    Example:
        >>> debug("Variable x = 5")
        [DEBUG] Variable x = 5
    """
    print(f"{Colors.DIM}[DEBUG] " + "=" * 60 + Colors.RESET)
    print(f"{Colors.DIM}[DEBUG] {message}{Colors.RESET}")
    print(f"{Colors.DIM}[DEBUG] " + "=" * 60 + Colors.RESET)


def banner(message: str, char: str = "=", width: int = 60) -> None:
    """Print a banner message.

    Args:
        message: Message to display in banner.
        char: Character to use for banner lines. Defaults to "=".
        width: Width of the banner. Defaults to 60.

    Example:
        >>> banner("START")
        ============================================================
        START
        ============================================================
    """
    print(char * width)
    print(message)
    print(char * width)


def format_residual_flags(compute_syst: bool, compute_stat: bool, compute_total: bool) -> str:
    """Format residual computation flags into a readable message.

    Args:
        compute_syst: Whether computing systematic residuals.
        compute_stat: Whether computing statistical residuals.
        compute_total: Whether computing total residuals.

    Returns:
        Formatted message describing what will be computed.

    Example:
        >>> format_residual_flags(True, False, False)
        'Computing: systematic residuals'
    """
    components = []
    if compute_syst:
        components.append("systematic")
    if compute_stat:
        components.append("statistical")
    if compute_total and not (compute_syst and compute_stat):
        components.append("total")

    if not components:
        return "No residuals"

    return "Computing: " + ", ".join(components) + " residuals"


@contextmanager
def suppress_runtime_warnings() -> Generator[None, None, None]:
    """Context manager to suppress specific runtime warnings.

    Suppresses RuntimeWarning about invalid values in logarithm operations,
    which can occur during r estimation when cl_model has zeros.

    Example:
        >>> with suppress_runtime_warnings():
        ...     # code that might divide by zero
        ...     pass
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        yield
