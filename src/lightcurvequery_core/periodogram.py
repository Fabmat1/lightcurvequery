"""
Periodogram utilities + the public ``plot_common_pgram`` function that the
main pipeline calls.

Only the functionality actually used by the refactored code base is kept.
The huge amount of demo / CLI code in the original *periodogramplot.py*
has been removed.
"""
from __future__ import annotations

import ctypes          # still used when the GLS C-extension is uncommented
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager as fm
from astropy.timeseries.periodograms.lombscargle._statistics import (
    false_alarm_level,
)
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.stats import norm
from astropy.timeseries import LombScargle

from .star import Star
from .utils import t_colors, ensure_directory_exists, sinusoid


# ---------------------------------------------------------------------- fonts
try:
    helvetica_available = any("Helvetica" in fm.FontProperties(fname=p).get_name()
                             for p in fm.findSystemFonts(fontext="ttf"))
except Exception:
    helvetica_available = False

if helvetica_available:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica"]
else:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["DejaVu Sans"]

# ------------------------------------------------------------------ constants
TELESCOPE_ZORDER = {"TESS": 10, "ZTF": 8, "GAIA": 6, "ATLAS": 9, "BLACKGEM": 7}

# alias intervals – unchanged
atlas_aliases = [
    [0.25177, 0.251795], [0.3323, 0.337], [0.49025, 0.4904],
    [0.50787, 0.50795], [0.5703, 0.57065], [0.7979, 0.7985],
    [0.96462, 0.9647], [0.996, 1.001], [1.031, 1.036],
    [1.3266, 1.335], [3.99, 4.01], [27.2, 27.45], [29.45, 29.65],
]
ztf_aliases = [
    [0.1108, 0.1121], [0.1424, 0.1434], [0.1659, 0.167],
    [0.1994, 0.2028], [0.2495, 0.252], [0.33, 0.338],
    [0.4975, 0.5035], [0.995, 1.005],
]
bg_aliases = [
    [0.04998, 0.05002], [0.1108, 0.1121], [0.1424, 0.1434],
    [0.1659, 0.167],   [0.1994, 0.2028], [0.33, 0.338],
    [0.4975, 0.5035],  [0.995, 1.005],
]

# default parameters (mirrors originals)
MIN_P, MAX_P, NSAMP = 0.05, 50.0, None
MED_FILTER, MED_SEGMENTS = False, 500
CORRECT_ZTF, CORRECT_ATLAS = True, True
IGNORE_SOURCES: List[str] = ["GAIA"]


# ------------------------------------------------------------------ utilities
def round_to_significant_digits(val, err):
    if val is None:
        return np.nan
    if err in (None, 0):
        return round(val, 2)
    try:
        order = np.floor(np.log10(abs(err)))
        fac = 10 ** (order - 1)
        rounded = round(val / fac) * fac
        dec = int(max(0, -order + 1))
        return f"{rounded:.{dec}f}" if dec else int(rounded)
    except ValueError:
        return val


def sinus_fix_period(period):
    def wrapped(x, amplitude, offset, phase):
        return sinusoid(x, amplitude, period, offset, phase)
    return wrapped


def gen_optimal_samples(t, sample_factor, min_p, max_p, Nsamp):
    if min_p is None:
        min_p = np.min(np.diff(np.sort(t))) * 2
    if max_p is None:
        max_p = np.ptp(t) / 2

    if Nsamp is None:
        n = np.ceil(np.ptp(t) / min_p)
        R_p = (np.ptp(t) / (n - 1) - np.ptp(t) / n)
        df = (1 / min_p - (1 / (min_p + R_p))) / sample_factor
        Nsamp = int(np.ceil((1 / min_p - 1 / max_p) / df))

    f0 = 1 / max_p
    df = (1 / min_p - f0) / Nsamp
    return f0, df, Nsamp


def fast_pgram(t, y, dy, min_p=None, max_p=None, N=None):
    if min_p is None or max_p is None or N is None:
        f0, df, Nf = gen_optimal_samples(t, 20, min_p, max_p, N)
        freqs = np.linspace(f0, f0 + df * Nf, Nf)
    else:
        freqs = np.flip(1 / np.linspace(min_p, max_p, int(N)))
    power = LombScargle(t, y, dy).power(freqs)
    return power, 1 / freqs


def alias_key_wrapper(p_x, p_y):
    def alias_key(interval):
        if interval[1] < p_x.min() or interval[0] > p_x.max():
            return 0
        mask = (p_x > interval[0]) & (p_x < interval[1])
        return np.max(p_y[mask])
    return alias_key


def sigma_percentage(sigma):
    return norm.cdf(sigma) - norm.cdf(-sigma)


# ---------------------------------------------------------------- core logic
def calc_pgrams(
    star: Star,
    *,
    ignore_source=None,
    min_p=MIN_P,
    max_p=MAX_P,
    Nsamp=NSAMP,
    whitening=True,
    plot=True,
    plot_as_bg=False,
    axs=None,
):
    ignore_source = ignore_source or []
    common_periods, n_samp = None, 0

    # determine a common period grid
    for tel, lc in star.lightcurves.items():
        if tel in ignore_source:
            continue
        psamp = gen_optimal_samples(lc[0].to_numpy(), 20, min_p, max_p, Nsamp)
        if psamp[-1] > n_samp:
            f0, df, Nf = psamp
            freqs = np.linspace(f0, f0 + df * Nf, Nf)
            common_periods = 1 / freqs
            n_samp = Nf
    if common_periods is None:   # nothing to do
        raise RuntimeError("No lightcurves left to analyse")
    common_power = np.ones_like(common_periods)

    # ------------------------------------------------------------------ loop
    row = 0
    for tel, lc in star.lightcurves.items():
        if tel in ignore_source:
            continue

        power, periods = fast_pgram(
            lc[0].to_numpy(), lc[1].to_numpy(), lc[2].to_numpy(),
            min_p, max_p, Nsamp,
        )

        # optional pre-whitening for ZTF / ATLAS / BlackGEM
        if whitening:
            aliases = (
                ztf_aliases if tel == "ZTF" else
                atlas_aliases if tel == "ATLAS" else
                bg_aliases if tel == "BLACKGEM" else
                []
            )
            aliases = sorted(aliases, key=alias_key_wrapper(periods, power))
            for lo, hi in aliases:
                if min_p and hi < min_p or max_p and lo > max_p:
                    continue
                sub = (periods > lo) & (periods < hi)
                if not sub.any():
                    continue
                mp = periods[sub][np.argmax(power[sub])]
                pars, _ = curve_fit(
                    sinus_fix_period(mp),
                    lc[0].to_numpy(), lc[1].to_numpy(),
                    sigma=lc[2].to_numpy(),
                )
                lc[1] -= sinus_fix_period(mp)(lc[0], *pars)
                power, periods = fast_pgram(
                    lc[0].to_numpy(), lc[1].to_numpy(), lc[2].to_numpy(),
                    min_p, max_p, Nsamp,
                )

        # accumulate multiplied periodogram
        f = interp1d(periods, power, bounds_error=False, fill_value=0)
        common_power *= f(common_periods)

        # store individual pgrams
        star.periodograms[tel] = (periods, power)

        # ----- plotting of this individual telescope ------------------------
        if plot:
            ax = axs[row] if axs is not None else plt.gca()
            col = t_colors.get(tel, 'k') if not plot_as_bg else 'gray'
            style = '-' if not plot_as_bg else '--'
            ax.plot(periods, power, color=col, linestyle=style,
                    label=f"{tel} photometry",
                    zorder=TELESCOPE_ZORDER[tel])

            # confidence levels ---------------------------------------------
            try:
                onesig, twosig, threesig = false_alarm_level(
                    [1 - 0.682689, 1 - 0.954499, 1 - 0.997300],
                    1 / min_p,
                    lc[0].to_numpy(), lc[1].to_numpy(), lc[2].to_numpy(),
                    "standard",
                )
                ax.axhline(onesig, ls='--', color='#F7B267',
                           label=r'$1\sigma$ limit')
                ax.axhline(twosig, ls='--', color='#F4845F',
                           label=r'$2\sigma$ limit')
                ax.axhline(threesig, ls='--', color='#F25C54',
                           label=r'$3\sigma$ limit')
            except (ValueError, ZeroDivisionError):
                # fall back silently – no FAP for this panel
                pass

             # ------------------------------ annotate average TESS crowding metric
            if tel.upper() == "TESS" and "TESS_CROWD" in star.metadata:
                ax.annotate(f"CROWDSAP = {star.metadata['TESS_CROWD']:.2f}",
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

            row += 1

    return common_periods, common_power


# ------------------------------------------------------------------ main API
def plot_common_pgram(
    star: Star,
    *,
    ignore_source=None,
    min_p_given=MIN_P,
    max_p_given=MAX_P,
    nsamp_given=NSAMP,
    whitening=True,
    title_fontsize=12,
    label_fontsize=12,
    legend_fontsize=8,
    tick_fontsize=10,
):
    ignore_source = ignore_source or []

    ensure_directory_exists(f"./periodograms/{star.gaia_id}")
    ensure_directory_exists("pgramplots")

    n_plots = (
        len(star.lightcurves)
        - len([s for s in ignore_source if s in star.lightcurves])
        + 2
    )

    # create all panels without sharing …
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(8.27, 11.69),
        dpi=100,
        sharex=False,
    )
    axes = np.atleast_1d(axes)  # ensure iterable

    # run calculations and draw upper panels
    common_periods, common_power = calc_pgrams(
        star,
        ignore_source=ignore_source,
        min_p=min_p_given,
        max_p=max_p_given,
        Nsamp=nsamp_given,
        whitening=whitening,
        axs=axes[:-2],
    )

    for ax in axes[1:-1]:
        ax.sharex(axes[0])

    # multiplied periodogram (log scale) -----------------------------
    axes[-2].plot(common_periods, common_power, color='#6D23B6',
                  label="Multiplied periodogram")
    axes[-2].set_xscale('log')
    axes[-2].set_xlim(common_periods[-1], common_periods[0])

    # zoom panel (independent x-axis) --------------------------------
    peak_p = common_periods[np.argmax(common_power)]
    local = (common_periods > peak_p * 0.999) & (common_periods < peak_p * 1.001)
    lerr, herr = common_periods[local][0], common_periods[local][-1]

    axes[-1].plot(common_periods, common_power, color='#6D23B6')
    axes[-1].axvline(peak_p, ls='--', color='red', label='Measured period')
    axes[-1].axvline(lerr, ls='--', color='black', label=r'$1\sigma$ bound')
    axes[-1].axvline(herr, ls='--', color='black')
    axes[-1].set_xlim(lerr, herr)

    # -------------- aesthetics (no grid, tick font sizes, legends) ----------
    for ax in axes:
        ax.tick_params(labelsize=tick_fontsize)
        ax.set_ylabel("Power", fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)

    axes[-1].set_xlabel("Period [d]", fontsize=label_fontsize)
    fig.suptitle(f"Periodograms for Gaia DR3 {star.gaia_id}",
                 fontsize=title_fontsize)

    plt.tight_layout()
    plt.savefig(f"pgramplots/{star.gaia_id}_periodograms.pdf",
                bbox_inches="tight", pad_inches=0)
    plt.show()

    # save multiplied pgram & update star
    np.savetxt(f"./periodograms/{star.gaia_id}/multiplied_pgram.txt",
               np.vstack((common_periods, common_power)).T, delimiter=",")
    star.period = peak_p