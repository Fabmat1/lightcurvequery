"""
Miscellaneous utility helpers.
"""
from __future__ import annotations

import os
import numpy as np

__all__ = [
    "magtoflux",
    "magerr_to_fluxerr",
    "calcpgramsamples",
    "ensure_directory_exists",
    "bcolors",
    "t_colors",
]

# ────────────────────────────────────────────────────────────────────
# colours for pretty terminal output
# ────────────────────────────────────────────────────────────────────
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ────────────────────────────────────────────────────────────────────
# telescope colour map – used by the plotting helpers
# ────────────────────────────────────────────────────────────────────
t_colors = {
    "ZTF": "green",
    "GAIA": "red",
    "TESS": "blue",
    "ATLAS": "darkorange",
    "BLACKGEM": "magenta",
}


# ────────────────────────────────────────────────────────────────────
# tiny maths helpers
# ────────────────────────────────────────────────────────────────────
def magtoflux(mag: np.ndarray | float):          # noqa: N802
    return 10 ** (-0.4 * mag)


def magerr_to_fluxerr(mag, magerr):              # noqa: N802
    flux = magtoflux(mag)
    return flux * np.log(10) * 0.4 * magerr


def calcpgramsamples(x_ptp, min_p, max_p):
    n = np.ceil(x_ptp / min_p)
    R_p = (x_ptp / (n - 1) - x_ptp / n)
    df = 1 / min_p - (1 / (min_p + R_p))
    return int(np.ceil((1 / min_p - 1 / max_p) / df)) * 10

def sinusoid(x, amplitude, period, offset, phase):
    result = amplitude * np.sin(2 * np.pi * (x/period+phase)) + offset;
    return result

def sinusoid_wrapper(phase):
    def sinusoid_wrapped(x, amplitude, period, offset):
        return sinusoid(x, amplitude, period, offset, phase)

    return sinusoid_wrapped


# ────────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────────
def ensure_directory_exists(directory: str | os.PathLike):
    if not os.path.exists(directory):
        os.makedirs(directory)