"""Slide look for `r_analysis plot --slides`: the matplotlib style of the PhD defence deck.

Transparent background, sans-serif type, no TeX, text emitted as paths in SVG. The r-vs-clusters
and r-likelihood plots read ENABLED and TRUTH to adapt their layout (see single.py, group.py).
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

INK = "#2E2E2E"
GREY = "#6E6E6E"

ENABLED = False
TRUTH: float | None = None


def enable(truth: float | None = None) -> None:
    """Replace the `science` style by the deck's and remember the truth value of r."""
    global ENABLED, TRUTH
    ENABLED, TRUTH = True, truth
    plt.rcdefaults()
    scale = 1.25
    plt.rcParams.update(
        {
            "text.usetex": False,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "axes.facecolor": "none",
            "font.family": "DejaVu Sans",
            "font.size": 13 * scale,
            "axes.labelsize": 14 * scale,
            "axes.titlesize": 14 * scale,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "xtick.labelsize": 12 * scale,
            "ytick.labelsize": 12 * scale,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "svg.fonttype": "path",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    matplotlib.rcParams["axes.formatter.useoffset"] = False
