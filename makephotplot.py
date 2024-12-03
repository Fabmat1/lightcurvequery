import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from astropy.time import Time
from matplotlib import rc
from models import Star
import ctypes

from makervplot import phasefoldplot
from common_functions import load_star

fm.findSystemFonts(fontpaths=None, fontext='ttf')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
GAIA_ID = 5205381551075658624
ADD_RV = False
DOFIT = False

OVERWRITE_P = 0.670867509035588
bandcolors = ["darkred", "navy", "darkgreen", "violet"]
IGNORE_SOURCES = []
IGNOREZI = True
IGNOREH = True

def load_single_band_lc(lc, star, binned, normalized, telescope, fold=True):
    lc: pd.DataFrame
    lc_x = lc[0].to_numpy()

    if telescope == "TESS":
        lc_x = Time(lc_x + 2457000, format="jd").mjd

    if star.period is not None and fold:
        lc_x = (lc_x % star.period) / star.period

        lc_x += star.phase
        lc_x[lc_x < 0] += 1
        lc_x[lc_x > 1] -= 1

    lc_y = lc[1].to_numpy()
    lc_y_err = lc[2].to_numpy()

    if normalized:
        lc_y_err /= np.median(lc_y)
        lc_y /= np.median(lc_y)

    if binned and len(lc_x) > 100:
        nbins = round(int(np.sqrt(len(lc_x))))
        if nbins % 2 != 0:
            nbins += 1

        # Create bin edges and digitize the x data
        bin_edges = np.linspace(lc_x.min(), lc_x.max(), nbins + 1)
        bin_indices = np.digitize(lc_x, bin_edges) - 1

        # Initialize arrays for binned results
        binned_lc_x = np.zeros(nbins)
        binned_lc_y = np.zeros(nbins)
        binned_lc_y_err = np.zeros(nbins)

        for j in range(nbins):
            mask = bin_indices == j
            if np.any(mask):  # Avoid empty bins
                binned_lc_x[j] = np.mean(lc_x[mask])  # Mean of x in the bin
                binned_lc_y[j] = np.mean(lc_y[mask])  # Mean of y in the bin
                binned_lc_y_err[j] = np.sqrt(np.sum(lc_y_err[mask] ** 2)) / np.sum(mask)  # Propagate errors
        if fold:
            binned_lc_x = np.concatenate([binned_lc_x-1, binned_lc_x])
            binned_lc_y = np.concatenate([binned_lc_y, binned_lc_y])
            binned_lc_y_err = np.concatenate([binned_lc_y_err, binned_lc_y_err])

        return binned_lc_x, binned_lc_y, binned_lc_y_err

    else:
        if fold:
            lc_x = np.concatenate([lc_x - 1, lc_x])
            lc_y = np.concatenate([lc_y, lc_y])
            lc_y_err = np.concatenate([lc_y_err, lc_y_err])

        return lc_x, lc_y, lc_y_err


def load_multi_band_lc(lc, star, binned, normalized, telescope, fold=True):
    lc: pd.DataFrame
    lc_x = lc[0].to_numpy()

    if telescope == "GAIA":
        lc_x += 2455197.5
        lc_x = Time(lc_x, format="jd").mjd

    if star.period is not None and fold:
        lc_x = (lc_x % star.period) / star.period

        lc_x += star.phase
        lc_x[lc_x < 0] += 1
        lc_x[lc_x > 1] -= 1

    lc_y = lc[1].to_numpy()
    lc_y_err = lc[2].to_numpy()

    lc_x_out = []
    lc_y_out = []
    lc_y_err_out = []
    bands = []

    for band in lc[3].unique():
        bandmask = lc[3] == band
        bandmask = bandmask.to_numpy()

        curr_lc_x = lc_x[bandmask]
        curr_lc_y = lc_y[bandmask]
        curr_lc_y_err = lc_y_err[bandmask]

        if normalized:
            curr_lc_y_err /= np.median(curr_lc_y)
            curr_lc_y /= np.median(curr_lc_y)

            if telescope == "GAIA":
                gmask = curr_lc_y < 1.15
                curr_lc_x = curr_lc_x[gmask]
                curr_lc_y = curr_lc_y[gmask]
                curr_lc_y_err = curr_lc_y_err[gmask]

        if binned and len(curr_lc_x) > 100:
            nbins = round(int(np.sqrt(len(curr_lc_x))))
            if nbins % 2 != 0:
                nbins += 1

            # Create bin edges and digitize the x data
            bin_edges = np.linspace(curr_lc_x.min(), curr_lc_x.max(), nbins + 1)
            bin_indices = np.digitize(curr_lc_x, bin_edges) - 1

            # Initialize arrays for binned results
            binned_lc_x = np.zeros(nbins)
            binned_lc_y = np.zeros(nbins)
            binned_lc_y_err = np.zeros(nbins)

            for j in range(nbins):
                mask = bin_indices == j
                if np.any(mask):  # Avoid empty bins
                    binned_lc_x[j] = np.mean(curr_lc_x[mask])  # Mean of x in the bin
                    binned_lc_y[j] = np.mean(curr_lc_y[mask])  # Mean of y in the bin
                    binned_lc_y_err[j] = np.sqrt(np.sum(curr_lc_y_err[mask] ** 2)) / np.sum(mask)  # Propagate errors

            if fold:
                binned_lc_x = np.concatenate([binned_lc_x - 1, binned_lc_x])
                binned_lc_y = np.concatenate([binned_lc_y, binned_lc_y])
                binned_lc_y_err = np.concatenate([binned_lc_y_err, binned_lc_y_err])

            lc_x_out.append(binned_lc_x)
            lc_y_out.append(binned_lc_y)
            lc_y_err_out.append(binned_lc_y_err)
            bands.append(band)
        else:
            if fold:
                curr_lc_x = np.concatenate([curr_lc_x - 1, curr_lc_x])
                curr_lc_y = np.concatenate([curr_lc_y, curr_lc_y])
                curr_lc_y_err = np.concatenate([curr_lc_y_err, curr_lc_y_err])

            lc_x_out.append(curr_lc_x)
            lc_y_out.append(curr_lc_y)
            lc_y_err_out.append(curr_lc_y_err)
            bands.append(band)

    return lc_x_out, lc_y_out, lc_y_err_out, bands


def plot_phot(star: Star, binned=True, normalized=True, add_rv_plot=False, title_fontsize=12, label_fontsize=12, legend_fontsize=8, tick_fontsize=10, ignore_sources=IGNORE_SOURCES, ignorezi=IGNOREZI, ignoreh=IGNOREH, dofit=DOFIT):
    for s in ignore_sources:
        del star.lightcurves[s]

    nrows = len(star.lightcurves)
    if add_rv_plot:
        nrows += 1

    fig: plt.Figure
    axs: list[plt.Axes]

    if nrows == 2:
        figsize = (8.27, 11.69/2)
    elif nrows == 3:
        figsize = (8.27, 2*11.69 / 3)
    elif nrows == 1:
        figsize = (8.27, 11.69 / 3)
    else:
        figsize = (8.27, 11.69)
    if not add_rv_plot:
        fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, dpi=100, sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, dpi=100, sharex=True)

    if isinstance(axs, plt.Axes):
        axs = [axs]


    if add_rv_plot:
        phasefoldplot(star, title_fontsize=12, label_fontsize=12, legend_fontsize=8, tick_fontsize=10, for_phfold=True, ax_for_phfold=axs[0], do_fit=dofit)


    for i, (telescope, lc) in enumerate(star.lightcurves.items()):
        if add_rv_plot:
            i += 1
        if len(lc.columns) < 4:
            lc_x, lc_y, lc_y_err = load_single_band_lc(lc, star, binned, normalized, telescope)
            axs[i].errorbar(lc_x, lc_y, yerr=lc_y_err, color="darkred", linestyle='None', zorder=5)
            axs[i].scatter(lc_x, lc_y, color="darkred", linestyle='None', s=5, zorder=6, label=f"{telescope} flux")
            axs[i].legend(fontsize=legend_fontsize, loc="lower right").set_zorder(7)
        else:
            lc_xs, lc_ys, lc_y_errs, bands = load_multi_band_lc(lc, star, binned, normalized, telescope)

            for ind, (lc_x, lc_y, lc_y_err, band) in enumerate(zip(lc_xs, lc_ys, lc_y_errs, bands)):
                if band == "zi" and ignorezi:
                    continue
                if band == "H" and ignoreh:
                    continue
                axs[i].errorbar(lc_x, lc_y, yerr=np.abs(lc_y_err), color=bandcolors[ind], linestyle='None', zorder=5)
                axs[i].scatter(lc_x, lc_y, color=bandcolors[ind], linestyle='None', s=5, zorder=6, label=f"{telescope} {band} flux")
                axs[i].legend(fontsize=legend_fontsize, loc="lower right").set_zorder(7)

        axs[i].grid(True, linestyle='--', color="darkgrey")
        axs[i].set_ylabel("Normalized Flux", fontsize=label_fontsize)
        axs[i].set_xlim((-1, 1))
        axs[i].tick_params(labelsize=tick_fontsize)

    # fig.suptitle(f"Phasefolded photometric{' and RV' if ADD_RV else ''} measurements \n for Gaia DR3 {GAIA_ID}",
    #                  fontsize=title_fontsize)

    axs[-1].set_xlabel("Phase", fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig(f"lcplots/{star.gaia_id}_lcplot.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    orbit_tracking_table = pd.read_csv("solved_orbit_tracker.txt", sep=",")
    try:
        star = load_star(GAIA_ID, orbit_tracking_table[orbit_tracking_table["gaia_id"] == GAIA_ID].iloc[0])
    except:
        star = load_star(GAIA_ID)
        if OVERWRITE_P is None:
            raise AssertionError
        else:
            star.phase = 0
    print("$P$ &", star.period, star.period_err_lo, star.period_err_hi)
    print("$K$ &", star.amplitude, star.amplitude_err)
    print("$\gamma$ &", star.offset, star.offset_err)
    print("$M_1$ &", star.m_1, star.m_1_err_lo, star.m_1_err_hi)
    print("$M_{2, \min}$ &", star.m_2, star.m_2_err_lo, star.m_2_err_hi)

    if OVERWRITE_P is not None:
        star.period = OVERWRITE_P
    # star.times = star.times[star.associations == 2]
    # star.datapoints = star.datapoints[star.associations == 2]
    # star.datapoint_errors = star.datapoint_errors[star.associations == 2]
    plot_phot(star, add_rv_plot=ADD_RV)
    # star.calculate_periodograms()
