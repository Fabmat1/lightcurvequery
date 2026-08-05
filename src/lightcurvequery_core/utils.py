"""
Miscellaneous utility helpers.
"""
from __future__ import annotations

import os
import threading

import numpy as np

__all__ = [
    "magtoflux",
    "magerr_to_fluxerr",
    "calcpgramsamples",
    "ensure_directory_exists",
    "bcolors",
    "t_colors",
    "MAST_DOWNLOAD_LOCK",
    "patch_lightkurve_stdout",
]

# ────────────────────────────────────────────────────────────────────
# thread-safety shims
# ────────────────────────────────────────────────────────────────────

# ``gettesslc`` and ``crowding.fetch_tess_crowdsap`` both pull the SPOC
# LC / FAST-LC products of the same TIC, from different threads, into the very
# same ./mastDownload/TESS/... paths.  Without a lock both can decide a file is
# missing and open it 'wb' simultaneously, leaving a truncated / interleaved
# FITS behind.  Serialising them costs nothing – whoever gets there second just
# hits astroquery's on-disk cache.
MAST_DOWNLOAD_LOCK = threading.RLock()

_LK_PATCH_LOCK = threading.Lock()
_lk_patched = False


def patch_lightkurve_stdout() -> None:
    """Stop lightkurve from swapping the process-global ``sys.stdout``.

    ``lightkurve.utils.suppress_stdout`` decorates ``SearchResult.download``
    and ``SearchResult.download_all`` with ::

        with open(os.devnull, "w") as devnull:
            old_out = sys.stdout
            sys.stdout = devnull
            try:     return f(...)
            finally: sys.stdout = old_out

    so for the duration of a lightkurve download the *whole process* sees a
    devnull handle as ``sys.stdout`` – and that handle is **closed** when the
    ``with`` block ends.  We download from several threads at once, and
    astropy caches ``sys.stdout`` in ``ProgressBarOrSpinner.__init__``, which
    astroquery uses for every MAST download.  A spinner constructed inside
    that window still holds the devnull handle when it prints " [Done]" on
    exit, so an unrelated fetcher thread dies with ::

        ValueError: I/O operation on closed file.

    Undo the decoration – the only cost is a few more "Downloading URL ..."
    lines in the log.
    """
    global _lk_patched
    if _lk_patched:
        return
    with _LK_PATCH_LOCK:
        if _lk_patched:
            return
        _lk_patched = True
        try:
            from lightkurve.search import SearchResult
        except Exception:
            return
        for name in ("download", "download_all"):
            original = getattr(getattr(SearchResult, name, None), "__wrapped__", None)
            if original is not None:
                setattr(SearchResult, name, original)

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