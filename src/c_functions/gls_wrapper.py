"""
Thin ctypes wrapper around libgls_fast with multiband support.

Falls back gracefully: if the shared library is not found or fails to load,
``GLS_AVAILABLE`` is ``False`` and calling ``gls_power()`` raises
``RuntimeError``.  The caller is expected to check ``GLS_AVAILABLE`` and
use the astropy path when it is ``False``.
"""
from __future__ import annotations

import ctypes
import platform
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------ library
_LIB_DIR = Path(__file__).resolve().parent

_EXT = {
    "Linux":   ".so",
    "Darwin":  ".dylib",
    "Windows": ".dll",
}.get(platform.system(), ".so")

_LIB_PATH = _LIB_DIR / f"libgls_fast{_EXT}"

GLS_AVAILABLE: bool = False
_lib = None

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))

    _lib.gls_fast_extern.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # t
        ctypes.c_size_t,                  # t_size
        ctypes.POINTER(ctypes.c_double),  # y
        ctypes.c_size_t,                  # y_size
        ctypes.POINTER(ctypes.c_double),  # dy
        ctypes.c_size_t,                  # dy_size
        ctypes.c_double,                  # f0
        ctypes.c_double,                  # df
        ctypes.c_int,                     # Nf
        ctypes.c_int,                     # normalization
        ctypes.c_bool,                    # fit_mean
        ctypes.c_bool,                    # center_data
        ctypes.c_int,                     # nterms
        ctypes.POINTER(ctypes.c_double),  # output
    ]
    _lib.gls_fast_extern.restype = None

    GLS_AVAILABLE = True

except OSError:
    pass


# -------------------------------------------------------------- public API

# Normalization constants matching the C++ enum convention
NORM_UNNORMALIZED = 0
NORM_CHI2         = 1
NORM_LOG          = 2
NORM_PSD          = 3


def gls_power(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    f0: float,
    df: float,
    Nf: int,
    *,
    normalization: int = NORM_UNNORMALIZED,
    fit_mean: bool = True,
    center_data: bool = True,
    nterms: int = 1,
) -> np.ndarray:
    """
    Compute the generalised Lomb-Scargle periodogram via the C extension.

    Parameters
    ----------
    t, y, dy : 1-D float64 arrays (same length)
    f0       : starting frequency
    df       : frequency step
    Nf       : number of frequencies
    normalization : 0=unnorm, 1=chi2, 2=log, 3=psd
    fit_mean : fit a constant offset
    center_data : subtract weighted mean before fitting
    nterms   : number of Fourier terms

    Returns
    -------
    power : ndarray of shape (Nf,)
    """
    if not GLS_AVAILABLE:
        raise RuntimeError(
            "GLS C extension not available. "
            f"Expected shared library at {_LIB_PATH}"
        )

    # Ensure contiguous float64
    t  = np.ascontiguousarray(t,  dtype=np.float64)
    y  = np.ascontiguousarray(y,  dtype=np.float64)
    dy = np.ascontiguousarray(dy, dtype=np.float64)

    output = np.empty(Nf, dtype=np.float64)

    _lib.gls_fast_extern(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(t.size),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(y.size),
        dy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(dy.size),
        ctypes.c_double(f0),
        ctypes.c_double(df),
        ctypes.c_int(Nf),
        ctypes.c_int(normalization),
        ctypes.c_bool(fit_mean),
        ctypes.c_bool(center_data),
        ctypes.c_int(nterms),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return output


def gls_power_multiband(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    bands: np.ndarray,
    f0: float,
    df: float,
    Nf: int,
    *,
    normalization: int = NORM_CHI2,
    fit_mean: bool = True,
    center_data: bool = True,
    nterms: int = 1,
    min_band_points: int = 10,
) -> np.ndarray:
    """
    Multiband Lomb-Scargle via independent per-band GLS fits combined
    by inverse-variance weight.

    This mirrors astropy's ``LombScargleMultiband(..., method='fast')``,
    which fits each band independently and combines them by weight.

    The combination formula for chi2-normalized single-band powers p_k is:

        P_combined = sum_k  w_k * p_k
        where  w_k = sum(1/dy_k^2) / sum_all(1/dy^2)

    Parameters
    ----------
    t, y, dy   : 1-D float64 arrays (same length)
    bands      : 1-D array of band labels (same length)
    f0, df, Nf : frequency grid parameters
    normalization : applied to the *combined* result
    min_band_points : skip bands with fewer points

    Returns
    -------
    power : ndarray of shape (Nf,)
    """
    if not GLS_AVAILABLE:
        raise RuntimeError(
            "GLS C extension not available. "
            f"Expected shared library at {_LIB_PATH}"
        )

    t     = np.asarray(t, dtype=np.float64)
    y     = np.asarray(y, dtype=np.float64)
    dy    = np.asarray(dy, dtype=np.float64)
    bands = np.asarray(bands)

    unique_bands = np.unique(bands)

    # Per-band inverse-variance sums (for weighting)
    band_weights = {}
    total_weight = 0.0
    valid_bands = []

    for b in unique_bands:
        mask = bands == b
        if mask.sum() < min_band_points:
            continue
        w = np.sum(1.0 / dy[mask] ** 2)
        band_weights[b] = w
        total_weight += w
        valid_bands.append(b)

    if len(valid_bands) == 0:
        return np.zeros(Nf, dtype=np.float64)

    # Normalize weights to sum to 1
    for b in valid_bands:
        band_weights[b] /= total_weight

    # Accumulate weighted per-band periodograms
    combined = np.zeros(Nf, dtype=np.float64)

    for b in valid_bands:
        mask = bands == b

        power_b = gls_power(
            t[mask], y[mask], dy[mask],
            f0=f0, df=df, Nf=Nf,
            normalization=NORM_CHI2,  # always chi2-normalize per band
            fit_mean=fit_mean,
            center_data=center_data,
            nterms=nterms,
        )

        combined += band_weights[b] * power_b

    # Re-normalize the combined result if the caller wants something
    # other than chi2
    if normalization == NORM_CHI2:
        pass  # already done
    elif normalization == NORM_UNNORMALIZED:
        # Undo chi2 normalization: approximate by scaling with total chi2_ref
        # This is only approximate; chi2 normalization is recommended for
        # multiband use.
        chi2_ref = np.sum(y ** 2 / dy ** 2)
        combined *= chi2_ref * 0.5
    elif normalization == NORM_LOG:
        combined = -np.log(1.0 - combined)
    elif normalization == NORM_PSD:
        combined = combined / (1.0 - combined)

    return combined