"""
Self-contained Star container used throughout lightcurve_query.
Now includes a *spectra* attribute for complete compatibility with
legacy code that might expect it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gatspy import periodic

from .utils import calcpgramsamples

__all__ = ["Star"]


class Star:
    """Light-curve and periodogram container."""

    # ─────────── initialisation ───────────
    def __init__(self, gaia_id: str | int):
        self.gaia_id = str(gaia_id)
        self.name = self.gaia_id
        self.alias = None

        # light-curve & periodogram storage
        self.lightcurves: dict[str, pd.DataFrame] = {}
        self.periodograms: dict[str, np.ndarray] = {}

        # ─ compatibility attributes (may stay None) ─
        self.times = np.array([])
        self.datapoints = np.array([])
        self.datapoint_errors = np.array([])
        self.associations = np.array([])
        self.spectra: dict = {}          # <─ new: keep legacy code happy

        # orbital / fit parameters (used by plot_phot)
        self.period = None
        self.amplitude = None
        self.offset = None
        self.phase = None

        # generic metadata – arbitrary key/value pairs filled in by
        # higher-level code (average crowding, etc.)
        self.metadata: dict[str, object] = {}

    # ─────────── private helpers ───────────
    def _single_band_pgram(self, tel, min_p, max_p, nsamp):
        lc = self.lightcurves[tel]
        x, y, yerr = lc[0].to_numpy(), lc[1].to_numpy(), lc[2].to_numpy()
        model = periodic.LombScargleFast().fit(x, y, yerr)
        nsamp = nsamp or calcpgramsamples(np.ptp(x), min_p, max_p)
        freqs = np.linspace(1 / max_p, 1 / min_p, nsamp)
        pwr = model.score_frequency_grid(freqs.min(), freqs[1] - freqs[0], nsamp)
        self.periodograms[tel] = np.vstack((pwr, 1 / freqs))

    def _multi_band_pgram(self, tel, min_p, max_p, nsamp):
        lc = self.lightcurves[tel]
        x, y, yerr = lc[0].to_numpy(), lc[1].to_numpy(), lc[2].to_numpy()
        bands = lc[3]
        model = periodic.LombScargleMultibandFast().fit(x, y, yerr, bands)
        nsamp = nsamp or calcpgramsamples(np.ptp(x), min_p, max_p)
        freqs = np.linspace(1 / max_p, 1 / min_p, nsamp)
        pwr = model.score_frequency_grid(freqs.min(), freqs[1] - freqs[0], nsamp)
        self.periodograms[tel] = np.vstack((pwr, 1 / freqs))

    # ─────────── public API ───────────
    def calculate_periodograms(self, min_p=None, max_p=None, nsamp=None, telescope=None):
        targets = [telescope] if telescope else list(self.lightcurves)
        for tel in targets:
            fn = self._multi_band_pgram if len(self.lightcurves[tel].columns) == 4 else self._single_band_pgram
            fn(tel, min_p, max_p, nsamp)

    def get_display_name(self) -> str:
        """Get display name for titles/output."""
        return self.alias or "Gaia DR3 {self.gaia_id}"