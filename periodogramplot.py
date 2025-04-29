import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from astropy.time import Time
from astropy.timeseries.periodograms.lombscargle._statistics import false_alarm_level
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.stats import norm
from astropy.timeseries import LombScargle

from models import Star
import ctypes

from makervplot import sinusoid
from common_functions import *
from tablemaker import round_to_significant_digits

fm.findSystemFonts(fontpaths=None, fontext='ttf')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
GAIA_ID = 5205381551075658624
SET_P = None
ADD_RV = False
MED_FILTER = False
MED_SEGMENTS = 500
IGNORE_SOURCES = ["GAIA"]
MIN_P = 0.05
MAX_P = 50
NSAMP = None
PLO = None
PHI = None
CORRECT_ZTF = True
CORRECT_ATLAS = True
TELESCOPE_ZORDER = {
    "TESS": 10,
    "ZTF": 8,
    "GAIA": 6,
    "ATLAS": 9,
    "BLACKGEM": 7
}

atlas_aliases = [
    [0.25177, 0.251795],
    [0.3323, 0.337],
    [0.49025, 0.4904],
    [0.50787, 0.50795],
    [0.5703, 0.57065],
    [0.7979, 0.7985],
    [0.96462, 0.9647],
    [0.996, 1.001],
    [1.031, 1.036],
    [1.3266, 1.335],
    [3.99, 4.01],
    [27.2, 27.45],
    [29.45, 29.65],
]

ztf_aliases = [
    [0.1108, 0.1121],
    [0.1424, 0.1434],
    [0.1659, 0.167],
    [0.1994, 0.2028],
    [0.33, 0.338],
    [0.4975, 0.5035],
    [0.995, 1.005],
]

bg_aliases = [
    [0.04998, 0.05002],
    [0.1108, 0.1121],
    [0.1424, 0.1434],
    [0.1659, 0.167],
    [0.1994, 0.2028],
    [0.33, 0.338],
    [0.4975, 0.5035],
    [0.995, 1.005],
]


def sinus_fix_period(period):
    def sinusoid_wrapped(x, amplitude, offset, phase):
        return sinusoid(x, amplitude, period, offset, phase)

    return sinusoid_wrapped


def genOptimalPeriodogramSamples(t, sample_factor, forced_min_p, forced_max_p, forced_N):
    if forced_min_p is None:
        min_p = np.min(np.diff(np.sort(t))) * 2
    else:
        min_p = forced_min_p

    if forced_max_p is None:
        max_p = (np.ptp(t)) / 2
    else:
        max_p = forced_max_p

    if forced_N is None:
        n = np.ceil(np.ptp(t) / min_p)
        R_p = (np.ptp(t) / (n - 1) - np.ptp(t) / n)
        df = (1 / min_p - (1 / (min_p + R_p))) / sample_factor

        Npoints = np.ceil((1 / min_p - 1 / max_p) / df)

    else:
        df = (1 / min_p - 1 / max_p) / forced_N
        Npoints = forced_N

    result = []
    result.append(1 / max_p)
    result.append(df)
    result.append(int(Npoints))
    return result


#
# def fast_pgram(t, y, dy, min_p=None, max_p=None, N=None):
#     # Load the shared library
#     lib = ctypes.CDLL("./libgls_shared.so")  # Use .dll for Windows
#
#     # Set up the return type and argument types for the function
#     lib.gls_fast_extern.restype = ctypes.POINTER(ctypes.c_double)
#     lib.gls_fast_extern.argtypes = [
#         ctypes.POINTER(ctypes.c_double),  # t_pointer
#         ctypes.c_size_t,  # t_size
#         ctypes.POINTER(ctypes.c_double),  # y_pointer
#         ctypes.c_size_t,  # y_size
#         ctypes.POINTER(ctypes.c_double),  # dy_pointer
#         ctypes.c_size_t,  # dy_size
#         ctypes.c_double,  # f0
#         ctypes.c_double,  # df
#         ctypes.c_int,  # Nf
#         ctypes.c_int,  # normalization (as C string)
#         ctypes.c_bool,  # fit_mean
#         ctypes.c_bool,  # center_data
#         ctypes.c_int  # nterms
#     ]
#
#     # Convert NumPy arrays to ctypes pointers
#     t_pointer = t.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     y_pointer = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     dy_pointer = dy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     f0, df, Nf = genOptimalPeriodogramSamples(t, 20, min_p, max_p, N)
#
#     print(f"Calculating periodogram from {1 / (f0 + df * Nf)} to {1 / f0} using {Nf} samples...")
#     fit_mean = True
#     center_data = True
#     nterms = 1
#
#     result_pointer = lib.gls_fast_extern(
#         t_pointer, len(t),
#         y_pointer, len(y),
#         dy_pointer, len(dy),
#         f0, df, Nf,
#         1, fit_mean, center_data, nterms
#     )
#
#     result_array = np.ctypeslib.as_array(result_pointer, shape=(Nf,))
#
#     freqs = np.linspace(f0, f0 + df * Nf, Nf)
#     periods = 1 / freqs
#
#     return result_array, periods


def fast_pgram(t, y, dy, min_p=None, max_p=None, N=None):
    """
    Calculate the Lomb-Scargle periodogram.

    Parameters:
    t : array-like
        Time values.
    y : array-like
        Flux values.
    dy : array-like
        Flux errors.
    min_p : float, optional
        Minimum period to search.
    max_p : float, optional
        Maximum period to search.
    N : int, optional
        Number of frequencies to evaluate.

    Returns:
    result_array : array-like
        Power values of the periodogram.
    periods : array-like
        Corresponding periods.
    """
    # Convert periods to frequencies
    if min_p is not None and max_p is not None:
        min_f = 1.0 / max_p
        max_f = 1.0 / min_p
    else:
        # Use default frequency range
        min_f, max_f = None, None

    f0, df, Nf = genOptimalPeriodogramSamples(t, 20, min_p, max_p, N)

    # Create frequency grid
    frequencies = np.linspace(f0, f0 + df * Nf, Nf)

    ls = LombScargle(t, y, dy)
    power = ls.power(frequencies)

    # Convert frequencies to periods
    periods = 1.0 / frequencies

    return power, periods


def alias_key_wrapper(pgram_x, pgram_y):
    def alias_key(interval):
        if interval[0] > pgram_x.max():
            return 0
        if interval[1] < pgram_x.min():
            return 0
        mask = np.logical_and(pgram_x > interval[0], pgram_x < interval[1])
        return np.max(pgram_y[mask])

    return alias_key


def calc_pgrams(star, ignore_source=[], min_p=MIN_P, max_p=MAX_P, Nsamp=NSAMP, plot=True, plot_as_bg=False, nocorr=False, axs=None):
    global atlas_aliases
    global ztf_aliases
    global bg_aliases

    common_periods = None
    n_samp_periods = 0
    if min_p is None or max_p is None or Nsamp is None:
        for telescope in star.lightcurves.keys():
            psamp = genOptimalPeriodogramSamples(star.lightcurves[telescope][0].to_numpy(), 20, min_p, max_p, Nsamp)
            if psamp[-1] > n_samp_periods:
                f0, df, Nf = psamp
                freqs = np.linspace(f0, f0 + df * Nf, Nf)
                common_periods = 1 / freqs
    else:
        common_periods = np.linspace(min_p, max_p, Nsamp)

    common_power = np.ones_like(common_periods)

    if ADD_RV:
        print("Calculating RV Periodogram...")
        result_array, periods = fast_pgram(star.times, star.datapoints, star.datapoint_errors, min_p, max_p, Nsamp)

        f = interp1d(periods, result_array, bounds_error=False, fill_value=0)
        common_power *= f(common_periods)

        try:
            onesig, twosig, threesig = false_alarm_level([1 - 0.682689, 1 - 0.954499, 1 - 0.997300],
                                                         1 / min_p,
                                                         star.times, star.datapoints, star.datapoint_errors,
                                                         "standard")
        except ValueError:
            try:
                onesig = false_alarm_level([1 - 0.682689],
                                           1 / min_p,
                                           star.times, star.datapoints, star.datapoint_errors,
                                           "standard")
                twosig, threesig = None, None
            except ValueError:
                onesig, twosig, threesig = None, None, None
        except ZeroDivisionError:
            onesig, twosig, threesig = None, None, None
        if plot:
            if axs is not None:
                axs[0].plot(periods, result_array, color='darkred' if not plot_as_bg else "gray", linestyle="-" if not plot_as_bg else "--", label="Radial Velocity")
                if onesig is not None:
                    axs[0].axhline(onesig, linestyle="--", color="#F7B267", label=r"$1\sigma$ limit")
                if twosig is not None:
                    axs[0].axhline(twosig, linestyle="--", color="#F4845F", label=r"$2\sigma$ limit")
                if threesig is not None:
                    axs[0].axhline(threesig, linestyle="--", color="#F25C54", label=r"$3\sigma$ limit")
            else:
                plt.plot(periods, result_array, color='darkred' if not plot_as_bg else "gray", linestyle="-" if not plot_as_bg else "--", label="Radial Velocity")

    print("Calculating Photometric Periodograms...")
    if ADD_RV:
        n = 1
    else:
        n = 0
    for telescope in star.lightcurves.keys():
        if telescope in ignore_source:
            continue

        result_array, periods = fast_pgram(star.lightcurves[telescope][0].to_numpy(),
                                           star.lightcurves[telescope][1].to_numpy(),
                                           star.lightcurves[telescope][2].to_numpy(),
                                           min_p, max_p, Nsamp)

        # plt.close()
        # plt.cla()
        # plt.clf()
        # fig, axs = plt.subplots(2, 1, figsize=(8.27, 0.3*11.69), sharex=True)

        if telescope == "ZTF" and CORRECT_ZTF and not nocorr:
            # axs[0].plot(periods, result_array, color="navy", label="Raw ZTF data")
            # axs[0].legend()
            # axs[0].set_xscale("log")
            ztf_aliases = sorted(ztf_aliases, key=alias_key_wrapper(periods, result_array))
            for [l, h] in ztf_aliases:
                if min_p is not None:
                    if l < min_p:
                        continue
                if max_p is not None:
                    if h > max_p:
                        continue
                subper = periods[np.logical_and(periods > l, periods < h)]
                subpow = result_array[np.logical_and(periods > l, periods < h)]
                mp = subper[np.argmax(subpow)]
                print(f"ZTF: Correcting Signal @ {mp} days")
                params, errs = curve_fit(sinus_fix_period(mp),
                                         star.lightcurves[telescope][0].to_numpy(),
                                         star.lightcurves[telescope][1].to_numpy(),
                                         sigma=star.lightcurves[telescope][2].to_numpy()
                                         )

                star.lightcurves[telescope][1] -= sinus_fix_period(mp)(star.lightcurves[telescope][0].to_numpy(), *params)
                result_array, periods = fast_pgram(star.lightcurves[telescope][0].to_numpy(),
                                                   star.lightcurves[telescope][1].to_numpy(),
                                                   star.lightcurves[telescope][2].to_numpy(),
                                                   min_p, max_p, Nsamp)
            # axs[1].plot(periods, result_array, color="navy", label="Pre-whitened ZTF data")
            # axs[1].legend()
            # axs[1].set_xscale("log")
            # fig.supxlabel("Period [d]")
            # fig.supylabel("Power [No Unit]")
            # plt.tight_layout()
            # plt.savefig("other_plots/prewhitening_ztf.pdf")
            # plt.show()
        if telescope == "ATLAS" and CORRECT_ATLAS and not nocorr:
            atlas_aliases = sorted(atlas_aliases, key=alias_key_wrapper(periods, result_array))
            for [l, h] in atlas_aliases:
                if min_p is not None:
                    if l < min_p:
                        continue
                if max_p is not None:
                    if h > max_p:
                        continue
                subper = periods[np.logical_and(periods > l, periods < h)]
                subpow = result_array[np.logical_and(periods > l, periods < h)]
                mp = subper[np.argmax(subpow)]
                print(f"ATLAS: Correcting Signal @ {mp} days")
                params, errs = curve_fit(sinus_fix_period(mp),
                                         star.lightcurves[telescope][0].to_numpy(),
                                         star.lightcurves[telescope][1].to_numpy(),
                                         sigma=star.lightcurves[telescope][2].to_numpy()
                                         )

                star.lightcurves[telescope][1] -= sinus_fix_period(mp)(star.lightcurves[telescope][0].to_numpy(), *params)
                result_array, periods = fast_pgram(star.lightcurves[telescope][0].to_numpy(),
                                                   star.lightcurves[telescope][1].to_numpy(),
                                                   star.lightcurves[telescope][2].to_numpy(),
                                                   min_p, max_p, Nsamp)
        if telescope == "BLACKGEM" and not nocorr:
            bg_aliases = sorted(bg_aliases, key=alias_key_wrapper(periods, result_array))
            for [l, h] in bg_aliases:
                if min_p is not None and max_p is not None:
                    if (l < min_p and h < max_p) or (l > max_p and h > max_p):
                        continue
                subper = periods[np.logical_and(periods > l, periods < h)]
                subpow = result_array[np.logical_and(periods > l, periods < h)]
                mp = subper[np.argmax(subpow)]
                print(f"BLACKGEM: Correcting Signal @ {mp} days")
                params, errs = curve_fit(sinus_fix_period(mp),
                                         star.lightcurves[telescope][0].to_numpy(),
                                         star.lightcurves[telescope][1].to_numpy(),
                                         sigma=star.lightcurves[telescope][2].to_numpy()
                                         )

                star.lightcurves[telescope][1] -= sinus_fix_period(mp)(star.lightcurves[telescope][0].to_numpy(), *params)
                result_array, periods = fast_pgram(star.lightcurves[telescope][0].to_numpy(),
                                                   star.lightcurves[telescope][1].to_numpy(),
                                                   star.lightcurves[telescope][2].to_numpy(),
                                                   min_p, max_p, Nsamp)
                
        f = interp1d(periods, result_array, bounds_error=False, fill_value=0)
        # if not ADD_RV and telescope == [k for k in star.lightcurves.keys() if k not in ignore_source][0]:
        #     common_periods = periods
        #     common_power = result_array

        common_power *= f(common_periods)

        if plot:
            if axs is not None:
                onesig, twosig, threesig = false_alarm_level([1 - 0.682689, 1 - 0.954499, 1 - 0.997300],
                                                             1 / min_p,
                                                             star.lightcurves[telescope][0].to_numpy(),
                                                             star.lightcurves[telescope][1].to_numpy(),
                                                             star.lightcurves[telescope][2].to_numpy(),
                                                             "standard")

                axs[n].plot(periods, result_array, color=t_colors[telescope] if not plot_as_bg else "gray", linestyle="-" if not plot_as_bg else "--", label=f"{telescope} Photometry", zorder=TELESCOPE_ZORDER[telescope])
                axs[n].axhline(onesig, linestyle="--", color="#F7B267", label=r"$1\sigma$ limit")
                axs[n].axhline(twosig, linestyle="--", color="#F4845F", label=r"$2\sigma$ limit")
                axs[n].axhline(threesig, linestyle="--", color="#F25C54", label=r"$3\sigma$ limit")
            else:
                plt.plot(periods, result_array, color=t_colors[telescope] if not plot_as_bg else "gray", linestyle="-" if not plot_as_bg else "--", label=f"{telescope} Photometry", zorder=TELESCOPE_ZORDER[telescope])
        n += 1
    return common_periods, common_power


def sigma_percentage(sigma):
    # Get the cumulative distribution function for the positive sigma
    return norm.cdf(sigma) - norm.cdf(-sigma)


def plot_common_pgram(star: Star, ignore_source=[], nsamp_given=NSAMP, min_p_given=MIN_P, max_p_given=MAX_P, whitening=True, title_fontsize=12, label_fontsize=12, legend_fontsize=8, tick_fontsize=10):
    try:
        solved_table = pd.read_csv("solved_orbit_tracker.txt")
        plo = solved_table.loc[star.gaia_id == solved_table["gaia_id"]]["p_window_low"].iloc[0]
        phi = solved_table.loc[star.gaia_id == solved_table["gaia_id"]]["p_window_hi"].iloc[0]
        nsamp = solved_table.loc[star.gaia_id == solved_table["gaia_id"]]["p_samp"].iloc[0]
    except:
        plo = PLO
        phi = PHI
        nsamp = None

    if pd.isnull(plo):
        plo = None
    if pd.isnull(phi):
        phi = None
    if pd.isnull(nsamp):
        nsamp = None

    axs: list[plt.Axes]
    nignored = [i for i in ignore_source if i in star.lightcurves.keys()]
    fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    # fig, axs = plt.subplots(nrows=len(star.lightcurves) + 3 - len(nignored), ncols=1, figsize=(8.27, 11.69), dpi=100, sharex=True)

    if not ADD_RV:
        nignored.append(1)

    axs = []
    for i in range(len(star.lightcurves) + 3 - len(nignored)):
        isfirstorlast = i == 0 or i == len(star.lightcurves) + 2 - len(nignored)
        axs.append(fig.add_subplot(len(star.lightcurves) + 3 - len(nignored), 1, i + 1, sharex=axs[-1] if not isfirstorlast else None))

    common_periods, common_power = calc_pgrams(star, ignore_source, min_p=min_p_given, max_p=max_p_given, Nsamp=nsamp_given, nocorr=~whitening, axs=axs)

    # plt.ylabel("Relative Power [arb. Unit]", fontsize=label_fontsize)
    # plt.xlabel("Period [d]", fontsize=label_fontsize)
    # plt.xlim(1.4, 1.8)
    # plt.gca().tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # legend = plt.legend(title="Data sources", fontsize=legend_fontsize)
    # plt.setp(legend.get_title(), fontsize=legend_fontsize)
    # plt.tight_layout()
    # plt.savefig(f"pgramplots/{GAIA_ID}_allpgplot_window.pdf", bbox_inches='tight', pad_inches=0)
    # plt.show()

    fig.supylabel("Relative Power [no Unit]", fontsize=label_fontsize)
    fig.supxlabel("Period [d]", fontsize=label_fontsize)

    for i, ax in enumerate(axs):
        if i != len(axs) - 1:
            if plo is not None:
                ax.axvline(plo, color="darkgrey", linestyle="--")
                ax.axvline(phi, color="darkgrey", linestyle="--")
            else:
                ax.axvline(common_periods[np.argmax(common_power)] * 0.999, color="darkgrey", linestyle="--")
                ax.axvline(common_periods[np.argmax(common_power)] * 1.001, color="darkgrey", linestyle="--")
            ax.set_xscale("log")
            ax.set_xlim(common_periods[-1], common_periods[0])
        if i == len(axs) - 1:
            common_periods, common_power = calc_pgrams(star, ignore_source,
                                                       plo if plo is not None else common_periods[np.argmax(common_power)] * 0.999, phi if phi is not None else common_periods[np.argmax(common_power)] * 1.001,
                                                       nsamp if nsamp is not None else 10000, plot=False, nocorr=True)

            if MED_FILTER:
                common_power = median_filter(common_power, size=len(common_power) // MED_SEGMENTS)
                common_power /= common_power.max()

            if not SET_P:
                measured_p = common_periods[common_power.argmax()]
            else:
                measured_p = SET_P
            n_sigma = 1
            sp = sigma_percentage(n_sigma)

            upper_periods = common_periods > measured_p
            upper_power = common_power[upper_periods]
            upper_power /= np.sum(upper_power)
            upper_periods = common_periods[upper_periods]

            try:
                upper_cumsum = 1 - np.cumsum(upper_power)
                upper_id = np.where(upper_cumsum >= sp)[0][-1]
                herr = upper_periods[upper_id]
            except IndexError:
                herr = 0

            lower_periods = common_periods < measured_p
            lower_power = common_power[lower_periods]
            lower_power /= np.sum(lower_power)
            lower_periods = common_periods[lower_periods]
            lower_power = np.flip(lower_power)
            lower_periods = np.flip(lower_periods)

            try:
                lower_cumsum = 1 - np.cumsum(lower_power)
                lower_id = np.where(lower_cumsum >= sp)[0][-1]
                lerr = lower_periods[lower_id]
            except IndexError:
                lerr = 0

            print(f"Measured Period: {round_to_significant_digits(measured_p, (herr - lerr)/2)}+{round_to_significant_digits(herr - measured_p,herr - measured_p)}-{round_to_significant_digits(measured_p - lerr, measured_p - lerr)} d")
            star.period = measured_p
            ax.plot(common_periods, common_power, color='#6D23B6', label="Multiplied Periodogram")
            ax.axvline(measured_p, linestyle="--", color="red", label="Measured Period")
            ax.axvline(lerr, linestyle="--", color="black", label=r"$1\sigma$ bound")
            ax.axvline(herr, linestyle="--", color="black")
            ax.ticklabel_format(useOffset=False)
            ax.set_xlim(plo, phi)
        elif i == len(axs) - 2:
            ax.plot(common_periods, common_power, color='#6D23B6', label="Multiplied Periodogram")
        # plt.title(f"Periodogram \n for Gaia DR3 {GAIA_ID}", fontsize=title_fontsize)

        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(f"pgramplots/{GAIA_ID}_allpgplot.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    star = load_star(GAIA_ID)
    print("Star loaded")
    plot_common_pgram(star, ignore_source=IGNORE_SOURCES)
    # plot_rv_periodogram(star)
    # plot_pgram(star)
