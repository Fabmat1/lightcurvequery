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
from astropy.timeseries import LombScargle, LombScargleMultiband

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


def fast_pgram(t, y, dy, min_p=None, max_p=None, N=None, bands=None):
    if min_p is None or max_p is None or N is None:
        f0, df, Nf = gen_optimal_samples(t, 20, min_p, max_p, N)
        freqs = np.linspace(f0, f0 + df * Nf, Nf)
    else:
        freqs = np.flip(1 / np.linspace(min_p, max_p, int(N)))

    if bands is not None:
        ls = LombScargleMultiband(t, y, dy=dy, bands=bands)
    else:
        ls = LombScargle(t, y, dy)

    power = ls.power(freqs, method="fast")
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


def measure_peak_period_HPD(common_periods, common_power, star, ignore_source=None,
                            alpha=0.683, k_rayleigh=5, smooth_win=9, error_floor_frac=0.5):
    """
    Measure peak period and asymmetric 1σ HPD interval using an adaptive local window.

    - Brackets the main lobe around the peak using local minima and ±k*dP_rayleigh.
    - Computes a contiguous HPD interval that contains 'alpha' probability mass.
    - Imposes an error floor = error_floor_frac * dP_rayleigh.

    Returns:
        peak_p, lerr_p, herr_p, (bracket_lo, bracket_hi), meta_dict
        where lerr_p <= peak_p <= herr_p, and (bracket_lo, bracket_hi) are the plotted x-limits
        in the original order of the period array (likely descending).
    """
    import numpy as np

    ignore_source = ignore_source or []

    # Handle degenerate cases
    if common_periods.size < 5:
        idx = int(np.argmax(common_power))
        peak = float(common_periods[idx])
        return peak, peak, peak, (float(common_periods.min()), float(common_periods.max())), {
            "status": "too_few_points"
        }

    # Determine effective time span across used lightcurves
    tmin, tmax = np.inf, -np.inf
    for tel, lc in star.lightcurves.items():
        if tel in ignore_source:
            continue
        t = np.asarray(lc[0].to_numpy())
        if t.size == 0:
            continue
        t_f = t[np.isfinite(t)]
        if t_f.size == 0:
            continue
        tmin = min(tmin, np.nanmin(t_f))
        tmax = max(tmax, np.nanmax(t_f))
    T_span = tmax - tmin if np.isfinite(tmin) and np.isfinite(tmax) else None

    # Identify global peak
    idx_peak = int(np.argmax(common_power))
    peak_p = float(common_periods[idx_peak])
    peak_pow = float(common_power[idx_peak])

    # Rayleigh period resolution; fallback if T_span not available
    if T_span is None or T_span <= 0:
        dP_rayleigh = abs(peak_p) * 1e-3  # fallback: 0.1% of P if no times available
        rayleigh_ok = False
    else:
        dP_rayleigh = (peak_p**2) / T_span
        rayleigh_ok = True

    # Smooth the power lightly to stabilize minima detection
    smooth_win = int(smooth_win)
    if smooth_win < 3:
        smooth_win = 3
    if smooth_win % 2 == 0:
        smooth_win += 1
    kernel = np.ones(smooth_win, dtype=float) / smooth_win
    pow_smooth = np.convolve(common_power, kernel, mode="same")

    # Find local minima around the peak (valley-to-valley bracketing)
    left_idx = idx_peak
    for i in range(idx_peak - 1, 0, -1):
        if pow_smooth[i] <= pow_smooth[i - 1] and pow_smooth[i] <= pow_smooth[i + 1]:
            left_idx = i
            break

    right_idx = idx_peak
    for i in range(idx_peak + 1, common_power.size - 1):
        if pow_smooth[i] <= pow_smooth[i - 1] and pow_smooth[i] <= pow_smooth[i + 1]:
            right_idx = i
            break

    # If minima were not found near the peak, fall back to Rayleigh-based window
    if left_idx == idx_peak or right_idx == idx_peak:
        # ±k*dP around the peak (period array is likely descending)
        target_left_P = peak_p + k_rayleigh * dP_rayleigh   # larger period (lower frequency)
        target_right_P = peak_p - k_rayleigh * dP_rayleigh  # smaller period (higher frequency)
        left_idx = int(np.argmin(np.abs(common_periods - target_left_P)))
        right_idx = int(np.argmin(np.abs(common_periods - target_right_P)))

    # Ensure bracket indices make sense and include peak
    i_lo = min(left_idx, right_idx, idx_peak)
    i_hi = max(left_idx, right_idx, idx_peak)
    if i_hi - i_lo < 4:
        # expand minimally if too narrow
        i_lo = max(0, idx_peak - 3)
        i_hi = min(common_power.size - 1, idx_peak + 3)

    # Build local arrays
    x = common_periods[i_lo:i_hi + 1]
    y = common_power[i_lo:i_hi + 1]

    # Numerical integration weights (absolute widths)
    dx = np.abs(np.diff(x))
    w = np.empty_like(x, dtype=float)
    if x.size == 1:
        w[:] = 1.0
    else:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        w[0] = dx[0]
        w[-1] = dx[-1]

    # Local PDF normalized within bracket
    area = float(np.sum(y * w))
    if not np.isfinite(area) or area <= 0:
        # fallback to small symmetric window if normalization fails
        j0 = idx_peak
        j_lo = max(0, j0 - 5)
        j_hi = min(common_power.size - 1, j0 + 5)
        x = common_periods[j_lo:j_hi + 1]
        y = common_power[j_lo:j_hi + 1]
        dx = np.abs(np.diff(x))
        w = np.empty_like(x, dtype=float)
        if x.size == 1:
            w[:] = 1.0
        else:
            w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
            w[0] = dx[0]
            w[-1] = dx[-1]
        area = float(np.sum(y * w))
        if not np.isfinite(area) or area <= 0:
            return peak_p, peak_p, peak_p, (float(x.min()), float(x.max())), {
                "status": "normalization_failed"
            }

    pdf = y / area

    # HPD: contiguous region around peak containing 'alpha' mass
    local_peak = int(np.argmax(y))
    target = float(alpha)

    def contiguous_mass_for_tau(tau):
        mask = pdf >= tau
        # ensure peak bin is included (nudge if needed)
        if not mask[local_peak]:
            mask = pdf >= (tau * 0.999)

        # Expand contiguously around peak
        l = r = local_peak
        while l - 1 >= 0 and mask[l - 1]:
            l -= 1
        while r + 1 < mask.size and mask[r + 1]:
            r += 1

        mass = float(np.sum(pdf[l:r + 1] * w[l:r + 1]))
        return mass, l, r

    # Bisection on tau in [0, max_pdf]
    lo_tau, hi_tau = 0.0, float(pdf[local_peak])
    l_sel = r_sel = local_peak
    for _ in range(50):
        mid = 0.5 * (lo_tau + hi_tau)
        mass, l_tmp, r_tmp = contiguous_mass_for_tau(mid)
        if mass >= target:
            # threshold too low (region too wide); increase tau
            lo_tau = mid
            l_sel, r_sel = l_tmp, r_tmp
        else:
            hi_tau = mid

    # Convert to period bounds; enforce lerr_p <= peak_p <= herr_p
    p_left = float(x[l_sel])
    p_right = float(x[r_sel])
    lerr_p = min(p_left, p_right)
    herr_p = max(p_left, p_right)

    bracket_lo = float(x[0])
    bracket_hi = float(x[-1])

    # Enforce an error floor tied to Rayleigh resolution (optional but recommended)
    if rayleigh_ok and error_floor_frac > 0:
        floor = float(error_floor_frac) * dP_rayleigh
        # Ensure lerr_p <= peak_p <= herr_p while applying the floor
        lerr_p = min(lerr_p, peak_p - floor)
        herr_p = max(herr_p, peak_p + floor)

    meta = {
        "status": "ok",
        "T_span": T_span,
        "dP_rayleigh": dP_rayleigh,
        "indices": (i_lo, i_hi, idx_peak),
        "selected_indices": (l_sel + i_lo, r_sel + i_lo),
        "peak_power": peak_pow,
    }
    return peak_p, lerr_p, herr_p, (bracket_lo, bracket_hi), meta


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
    legend_fontsize=8,
):
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

        asnp = lambda x: x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
        # detect multiband labels
        bands_arg = None
        try:
            if hasattr(lc, "columns") and (3 in lc.columns):
                ba = asnp(lc[3])
                if ba.size == asnp(lc[0]).size:
                    is_str = np.fromiter((isinstance(b, (str, np.str_)) for b in ba), dtype=bool, count=ba.size)
                    if is_str.all():
                        bands_arg = ba
        except Exception:
            bands_arg = None

        power, periods = fast_pgram(
            asnp(lc[0]).astype(float),
            asnp(lc[1]).astype(float),
            asnp(lc[2]).astype(float),
            min_p, max_p, Nsamp,
            bands=bands_arg,
        )
        # optional pre-whitening for ZTF / ATLAS / BlackGEM

        if whitening:
            aliases = (
                ztf_aliases if tel == "ZTF" else
                atlas_aliases if tel == "ATLAS" else
                bg_aliases if tel == "BLACKGEM" else
                []
            )
            if aliases != []:
                print(f"[{star.gaia_id}] Pre-whitening for {tel}...")
                aliases = sorted(aliases, key=alias_key_wrapper(periods, power))

                for lo, hi in aliases:
                    if (min_p and hi < min_p) or (max_p and lo > max_p):
                        continue

                    # use the current period grid to pick mp inside the alias window
                    sub = (periods > lo) & (periods < hi)
                    if not sub.any():
                        continue
                    mp = periods[sub][np.argmax(power[sub])]
                    print(f"[{star.gaia_id}] {tel}: Eliminating {mp} d...")

                    t = asnp(lc[0]).astype(float)
                    y = asnp(lc[1]).astype(float)
                    dy = asnp(lc[2]).astype(float)
                    filt = asnp(lc[3]) if bands_arg is not None else None

                    phase = (t % mp) / mp

                    unique_filters = np.unique(filt) if filt is not None else [None]
                    for f in unique_filters:
                        mask = (filt == f) & np.isfinite(y) & np.isfinite(dy) if filt is not None \
                               else (np.isfinite(y) & np.isfinite(dy))
                        if mask.sum() <= 10:
                            continue

                        x_fit = phase[mask]
                        y_fit = y[mask]
                        dy_fit = dy[mask]

                        try:
                            pars, _ = curve_fit(
                                sinus_fix_period(1.0),
                                x_fit,
                                y_fit,
                                sigma=dy_fit,
                            )
                        except Exception:
                            continue

                        model_unfolded = sinus_fix_period(mp)(t[mask], *pars)

                        y_resid = y[mask] - (model_unfolded - 1.0)
                        med = np.nanmedian(y_resid)
                        if np.isfinite(med) and med != 0.0:
                            y_resid /= med
                        y[mask] = y_resid

                        # plt.scatter(x_fit.astype(float), y_fit.astype(float), s=10, label=f)
                        # phi = np.linspace(0, 1, 500)
                        # plt.plot(phi, sinus_fix_period(1.0)(phi, *pars), "r")
                        # plt.legend(); plt.show()

                    # single-step assignment to avoid chained assignment warnings
                    lc.loc[:, 1] = y

                    power, periods = fast_pgram(
                        t, y, dy,
                        min_p, max_p, Nsamp,
                        bands=bands_arg,  # reuse detected bands
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

            ax.legend(fontsize=legend_fontsize, loc="best") 

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
    show_plots=True
):
    ensure_directory_exists(f"./periodograms/{star.gaia_id}")
    ensure_directory_exists("pgramplots")
    n_pgram_panels = (
        len(star.lightcurves)
        - len([s for s in ignore_source if s in star.lightcurves])
    )
    nrows_total = n_pgram_panels + 2           # multiplied + zoom

    fig, axes = plt.subplots(
        nrows=nrows_total,
        ncols=1,
        figsize=(8.27, 11.69),
        dpi=100,
        sharex=False,          # we wire the sharing manually below
    )
    axes = np.atleast_1d(axes)

    # ---------------------------------------------------------------- calc & plot individual + multiplied
    common_periods, common_power = calc_pgrams(
        star,
        ignore_source=ignore_source,
        min_p=min_p_given,
        max_p=max_p_given,
        Nsamp=nsamp_given,
        whitening=whitening,
        axs=axes[:n_pgram_panels],
        legend_fontsize=legend_fontsize,
    )

    # bottom-most *periodogram* axis (before zoom)
    ax_master = axes[n_pgram_panels]           # == axes[-2]

    # add all telescopes + multiplied pgram share the same x-axis
    for ax in axes[: n_pgram_panels]:
        ax.sharex(ax_master)
        ax.tick_params(labelbottom=False)      # remove duplicate labels

    # multiplied pgram --------------------------------------------------------
    ax_master.plot(common_periods, common_power, color='#6D23B6',
                   label="Multiplied periodogram")
    ax_master.set_xscale('log')
    ax_master.set_xlim(common_periods[-1], common_periods[0])
    ax_master.legend(fontsize=legend_fontsize)
    ax_master.set_ylabel("Power", fontsize=label_fontsize)
    ax_master.tick_params(labelsize=tick_fontsize)

    # ---------------------------------------------------------------- zoom panel
    peak_p, lerr_p, herr_p, (lwin, rwin), _meta = measure_peak_period_HPD(
        common_periods,
        common_power,
        star,
        ignore_source=ignore_source,
    )

    # make sure limits run from small → large period (left → right)
    xlo, xhi = sorted([lwin, rwin])

    sel = (common_periods >= xlo) & (common_periods <= xhi)
    axes[-1].plot(common_periods[sel], common_power[sel], color='#6D23B6')
    axes[-1].axvline(peak_p,  ls='--', color='red',   label='Measured period')
    axes[-1].axvline(lerr_p, ls='--', color='black', label='1σ HPD bounds')
    axes[-1].axvline(herr_p, ls='--', color='black')
    axes[-1].set_xlim(xlo, xhi)                # correct orientation
    axes[-1].set_xlabel("Period [d]", fontsize=label_fontsize)
    axes[-1].set_ylabel("Power",      fontsize=label_fontsize)
    axes[-1].tick_params(labelsize=tick_fontsize)
    axes[-1].legend(fontsize=legend_fontsize, loc="best")

    # --------------- period annotation on the zoom panel ---------------------
    plus  = herr_p - peak_p
    minus = peak_p - lerr_p
    txt   = f"P = {peak_p:.6f} (+{plus:.6f} / -{minus:.6f}) d"
    axes[-1].annotate(
        txt,
        xy=(0.02, 0.95), xycoords='axes fraction',
        ha='left', va='top', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)
    )

    # ---------------------------------------------------------------- figure cosmetics
    fig.suptitle(f"Periodograms for Gaia DR3 {star.gaia_id}",
                 fontsize=title_fontsize)
    fig.subplots_adjust(hspace=0)      # remove vertical white-space
    plt.tight_layout()
    plt.savefig(f"pgramplots/{star.gaia_id}_periodograms.pdf",
                bbox_inches="tight", pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close('all')
        
    # --------- console output & bookkeeping ----------------------------------
    print(f"[{star.gaia_id}] Measured period: "
          f"{peak_p:.6f} (+{plus:.6f} / -{minus:.6f}) d")

    np.savetxt(f"./periodograms/{star.gaia_id}/multiplied_pgram.txt",
               np.vstack((common_periods, common_power)).T, delimiter=",")
    star.period        = peak_p
    star.period_loerr  = minus
    star.period_hierr  = plus
