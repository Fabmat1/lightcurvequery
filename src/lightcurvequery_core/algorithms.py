"""
Period-finding algorithm registry.

Each algorithm computes a power spectrum on a frequency grid.
Add new algorithms by:
  1. implementing a ``compute_fn(t, y, dy, freqs, bands=None, **kw)``
  2. registering an ``AlgorithmSpec`` in ``ALGORITHM_REGISTRY``
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

__all__ = [
    "AlgorithmResult",
    "AlgorithmSpec",
    "ALGORITHM_REGISTRY",
    "get_algorithm",
    "resolve_algorithm_names",
    "check_algorithm_deps",
]


# ─────────────────────────── result container ───────────────────────

@dataclass
class AlgorithmResult:
    """Full output of one period-finding algorithm run."""
    name: str
    display_name: str
    periodograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    common_periods: Optional[np.ndarray] = None
    common_power: Optional[np.ndarray] = None
    period: Optional[float] = None
    period_loerr: Optional[float] = None
    period_hierr: Optional[float] = None


# ─────────────────────────── spec container ─────────────────────────

@dataclass
class AlgorithmSpec:
    """Registered period-finding algorithm."""
    name: str
    display_name: str
    compute_fn: callable
    supports_fal: bool = True   # are Lomb-Scargle false-alarm levels meaningful?


# ─────────────────────── Lomb-Scargle backend ───────────────────────

try:
    from ..c_functions.gls_wrapper import (
        GLS_AVAILABLE as _GLS_AVAILABLE,
        gls_power as _gls_power,
        gls_power_multiband as _gls_power_multiband,
    )
except ImportError:
    _GLS_AVAILABLE = False
    _gls_power = None
    _gls_power_multiband = None

if not _GLS_AVAILABLE:
    try:
        from astropy.timeseries import LombScargle as _LS
        from astropy.timeseries import LombScargleMultiband as _LSMB
    except ImportError:
        _LS = None
        _LSMB = None
else:
    _LS = None
    _LSMB = None


def _lombscargle_power(t, y, dy, freqs, bands=None, **kwargs):
    """Lomb-Scargle periodogram on an arbitrary frequency grid."""
    f0 = float(freqs[0])
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    Nf = len(freqs)

    if bands is not None:
        bands = np.asarray(bands)
        unique, counts = np.unique(bands, return_counts=True)
        valid = unique[counts >= 10]
        mask = np.isin(bands, valid)
        t, y, dy, bands = t[mask], y[mask], dy[mask], bands[mask]

        if _GLS_AVAILABLE:
            return _gls_power_multiband(
                t, y, dy, bands,
                f0=f0, df=df, Nf=Nf,
                normalization=0, fit_mean=True,
                center_data=True, nterms=1,
            )
        ls = _LSMB(t, y, dy=dy, bands=bands)
        return ls.power(freqs, method="fast")

    if _GLS_AVAILABLE:
        return _gls_power(
            t, y, dy,
            f0=f0, df=df, Nf=Nf,
            normalization=0, fit_mean=True,
            center_data=True, nterms=1,
        )
    ls = _LS(t, y, dy)
    return ls.power(freqs, method="fast")


# ──────────────────────────── FPW backend ───────────────────────────

def _fpw_power(t, y, dy, freqs, bands=None, n_bins=None, **kwargs):
    """
    FPW periodogram (Finkbeiner, Prince & Whitebook 2025, PASP 137, 054504).

    For multi-band data the algorithm runs per band; results are combined
    via a weighted sum with weights ∝ number of data points per band.
    """
    try:
        import fpw
    except ImportError:
        raise ImportError(
            "The FPW algorithm requires the fpwperiodic package.\n"
            "Install with:  pip install fpwperiodic"
        )

    if bands is not None:
        bands = np.asarray(bands)
        unique_bands, counts = np.unique(bands, return_counts=True)
        valid = unique_bands[counts >= 10]

        combined = np.zeros(len(freqs), dtype=float)
        total_w = 0.0
        for band in valid:
            m = bands == band
            t_b, y_b, dy_b = t[m], y[m], dy[m]
            nb = _optimal_fpw_bins(len(t_b)) if n_bins is None else n_bins
            power_b = np.asarray(
                fpw.run_fpw(t_b, y_b, dy_b, freqs, nb), dtype=float,
            )
            w = float(len(t_b))
            combined += w * power_b
            total_w += w
        return combined / total_w if total_w > 0 else combined

    nb = _optimal_fpw_bins(len(t)) if n_bins is None else n_bins
    return np.asarray(fpw.run_fpw(t, y, dy, freqs, nb), dtype=float)


def _optimal_fpw_bins(n_points: int, minimum: int = 25) -> int:
    """Optimal FPW bin count: max(*minimum*, √N)."""
    return max(minimum, int(np.sqrt(n_points)))


# ─────────────────────────── registry ───────────────────────────────

ALGORITHM_REGISTRY: Dict[str, AlgorithmSpec] = {
    "lombscargle": AlgorithmSpec(
        name="lombscargle",
        display_name="Lomb-Scargle",
        compute_fn=_lombscargle_power,
        supports_fal=True,
    ),
    "fpw": AlgorithmSpec(
        name="fpw",
        display_name="FPW",
        compute_fn=_fpw_power,
        supports_fal=False,
    ),
}


def get_algorithm(name: str) -> AlgorithmSpec:
    """Look up an algorithm by its CLI key."""
    if name not in ALGORITHM_REGISTRY:
        available = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return ALGORITHM_REGISTRY[name]


def resolve_algorithm_names(names: list[str]) -> list[str]:
    """Expand ``'all'``, validate, and deduplicate."""
    out: list[str] = []
    for n in names:
        if n == "all":
            out.extend(ALGORITHM_REGISTRY.keys())
        elif n in ALGORITHM_REGISTRY:
            out.append(n)
        else:
            available = ", ".join(ALGORITHM_REGISTRY.keys())
            raise ValueError(f"Unknown algorithm '{n}'. Available: {available}")
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def check_algorithm_deps(name: str):
    """Raise ``ImportError`` early if a required package is missing."""
    if name == "fpw":
        try:
            import fpw  # noqa: F401
        except ImportError:
            raise ImportError(
                "The FPW algorithm requires the fpwperiodic package.\n"
                "Install with:  pip install fpwperiodic"
            )