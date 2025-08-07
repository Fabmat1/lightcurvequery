"""
Light-curve plotting helpers.

The public function here is ``plot_phot(star, …)`` – identical signature
to the one that the original *makephotplot.py* exposed, so the rest of the
pipeline (``process_lightcurves``) keeps working unchanged.

Only the imports were modernised (``Star`` now comes from the local
package) and all references to the old *common_functions* / *models*
modules have been removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from astropy.time import Time
from pathlib import Path


from .star import Star

# ---------------------------------------------------------------------- setup
fm.findSystemFonts(fontpaths=None, fontext='ttf')
rc('font', family='sans-serif', **{'sans-serif': ['Helvetica']})

# Per-telescope display colours (feel free to edit)
bandcolors = [
    "darkred", "navy", "darkgreen", "violet", "magenta",
    "gold", "salmon", "brown", "black", "lime", "red",
]

# Default behaviour flags – the values equal those in the old script
IGNORE_SOURCES: list[str] = []
IGNOREZI = True
IGNOREH = True
DOFIT = False  # whether to run the RV sine-fit in phasefoldplot


# ------------------------------------------------------------------ utilities
def _load_single_band(
    lc: pd.DataFrame,
    star: Star,
    *,
    telescope: str,
    binned: bool,
    normalized: bool,
    fold: bool = True,
):
    x = lc[0].to_numpy()
    y = lc[1].to_numpy()
    yerr = lc[2].to_numpy()

    # convert TESS time stamps
    if telescope.upper() == "TESS":
        x = Time(x + 2457000, format="jd").mjd

    # phase-fold
    if star.period is not None and fold:
        x = (x % star.period) / star.period
        x = (x + star.phase) % 1.0

    # normalise
    if normalized:
        scale = np.median(y)
        y /= scale
        yerr /= scale

    # optional binning
    if binned and len(x) > 100:
        nbins = int(np.sqrt(len(x)))
        nbins += nbins % 2  # make it even
        edges = np.linspace(x.min(), x.max(), nbins + 1)
        idx = np.digitize(x, edges) - 1

        bx = np.array([x[idx == i].mean() for i in range(nbins)])
        by = np.array([y[idx == i].mean() for i in range(nbins)])
        be = np.array([
            np.sqrt((yerr[idx == i] ** 2).sum()) / max(1, (idx == i).sum())
            for i in range(nbins)
        ])
        x, y, yerr = bx, by, be

    if fold:
        x = np.concatenate([x - 1, x])
        y = np.concatenate([y, y])
        yerr = np.concatenate([yerr, yerr])

    return x, y, yerr


def _load_multi_band(
    lc: pd.DataFrame,
    star: Star,
    *,
    telescope: str,
    binned: bool,
    normalized: bool,
    fold: bool = True,
):
    out_x, out_y, out_e, bands = [], [], [], []

    base_x = lc[0].to_numpy()
    if telescope.upper() == "GAIA":               # Gaia JD → MJD
        base_x += 2455197.5
        base_x = Time(base_x, format="jd").mjd

    for band in lc[3].unique():
        sel = lc[3] == band
        x = base_x[sel.to_numpy()]
        y = lc[1][sel].to_numpy()
        e = lc[2][sel].to_numpy()

        if normalized:
            scale = np.median(y)
            y, e = y / scale, e / scale

            if telescope.upper() == "GAIA":
                mask = y < 1.15        # old script’s “gmask” cut
                x, y, e = x[mask], y[mask], e[mask]

        if binned and len(x) > 100:
            nbins = int(np.sqrt(len(x))) + int(np.sqrt(len(x))) % 2
            edges = np.linspace(x.min(), x.max(), nbins + 1)
            idx = np.digitize(x, edges) - 1

            bx = np.array([x[idx == i].mean() for i in range(nbins)])
            by = np.array([y[idx == i].mean() for i in range(nbins)])
            be = np.array([
                np.sqrt((e[idx == i] ** 2).sum()) / max(1, (idx == i).sum())
                for i in range(nbins)
            ])
            x, y, e = bx, by, be

        if fold and star.period is not None:
            x = (x % star.period) / star.period
            x = (x + star.phase) % 1.0
            x = np.concatenate([x - 1, x])
            y = np.concatenate([y, y])
            e = np.concatenate([e, e])

        out_x.append(x)
        out_y.append(y)
        out_e.append(e)
        bands.append(band)

    return out_x, out_y, out_e, bands


# ------------------------------------------------------------------ main plot
def phasefoldplot(star: Star, N_samples=5000, title_fontsize=14, label_fontsize=14, legend_fontsize=12, tick_fontsize=12, for_phfold=False, ax_for_phfold: plt.Axes = None, do_fit=True, custom_saveloc=None, custom_ylim=None, figsize=None):
    if not for_phfold and figsize is None:
        plt.figure(figsize=(6, 4))
    elif figsize is not None:
        plt.figure(figsize=figsize)
    # params, errs = curve_fit(sinusoid,
    #                          star.times,
    #                          star.datapoints,
    #                          p0=[star.amplitude, star.period, star.offset, star.phase],
    #                          bounds=[
    #                              [0, star.period*0.99, -np.inf, -1],
    #                              [np.inf, star.period*1.01, np.inf, 2]
    #                          ],
    #                          sigma=star.datapoint_errors,
    #                          maxfev=100000)
    # errs = np.sqrt(np.diag(errs))

    # if np.any(~np.isfinite(errs)):
    if do_fit:
        params, errs = curve_fit(sinusoid_wrapper(star.phase),
                                 star.times,
                                 star.datapoints,
                                 p0=[star.amplitude, star.period, star.offset],
                                 bounds=[
                                     [0, star.period * 0.99, -np.inf],
                                     [np.inf, star.period * 1.01, np.inf]
                                 ],
                                 sigma=star.datapoint_errors,
                                 maxfev=100000)
        errs = np.sqrt(np.diag(errs))
        errs = np.append(errs, 0)
        params = np.append(params, star.phase)

        star.amplitude, star.period, star.offset, star.phase = params
    folded_times = (star.times % star.period) / star.period

    folded_times = np.concatenate([folded_times - 1, folded_times])
    folded_times += star.phase
    folded_times[folded_times < -1] += 2
    folded_times[folded_times > 1] -= 2

    folded_points = np.concatenate([star.datapoints, star.datapoints])
    folded_errors = np.concatenate([star.datapoint_errors, star.datapoint_errors])
    folded_associations = np.concatenate([star.associations, star.associations])

    mask = np.argsort(folded_times)
    folded_times = folded_times[mask]
    folded_points = folded_points[mask]
    folded_errors = folded_errors[mask]
    folded_associations = folded_associations[mask]

    t_space = np.linspace(-1, 1, 10000)

    # Color map for associations
    color_map = {0: '#800E13', 1: '#47126B', 2: '#0F392B', 3: '#0F392B'}
    color_map_err = {0: '#640D14', 1: '#6D23B6', 2: '#185339', 3: '#185339'}
    assoc_label_map = {0: 'LAMOST', 1: 'SDSS', 2: 'SOAR', 3: 'WHT/CAHA/NOT'}

    # Plot the errorbars with different colors for each association
    for assoc_value in np.unique(folded_associations):
        assoc_mask = folded_associations == assoc_value
        if not for_phfold:
            plt.errorbar(folded_times[assoc_mask], folded_points[assoc_mask], folded_errors[assoc_mask],
                         linestyle="none", capsize=3,
                         color=color_map_err[assoc_value], label=f'{assoc_label_map[assoc_value]}', zorder=9)
            plt.scatter(folded_times[assoc_mask], folded_points[assoc_mask], 10, marker="o",
                        color=color_map[assoc_value], label="_nolegend_", zorder=10)
        else:
            ax_for_phfold.errorbar(folded_times[assoc_mask], folded_points[assoc_mask], folded_errors[assoc_mask],
                                   linestyle="none", capsize=3,
                                   color=color_map_err[assoc_value], label=f'{assoc_label_map[assoc_value]}', zorder=9)
            ax_for_phfold.scatter(folded_times[assoc_mask], folded_points[assoc_mask], 10, marker="o",
                                  color=color_map[assoc_value], label="_nolegend_", zorder=10)

    if not for_phfold:
        plt.plot(t_space, sinusoid(t_space, star.amplitude, 1, star.offset, 0), color="mediumblue", zorder=6, label="Best Fit")
    else:
        ax_for_phfold.plot(t_space, sinusoid(t_space, star.amplitude, 1, star.offset, 0), color="mediumblue", zorder=6, label="Best Fit")

    if do_fit:
        sinusoids = np.zeros((N_samples, len(t_space)))
        for i in range(N_samples):
            amplitude_varied = star.amplitude + np.random.normal(0, errs[0])
            period_varied = (star.period + np.random.normal(0, errs[1])) / star.period
            offset_varied = star.offset + np.random.normal(0, errs[2])

            sinusoids[i, :] = sinusoid(t_space, amplitude_varied, period_varied, offset_varied, 0)

        # Calculate the 16th and 84th percentiles (1-sigma bounds)
        lower_bound = np.percentile(sinusoids, 16, axis=0)
        upper_bound = np.percentile(sinusoids, 84, axis=0)

        # Calculate the 2.5th and 97.5th percentiles (2-sigma bounds)
        lower_bound_2s = np.percentile(sinusoids, 2.5, axis=0)
        upper_bound_2s = np.percentile(sinusoids, 97.5, axis=0)

    # Plot the 68% confidence interval using fill_between
    if not for_phfold:
        if do_fit:
            plt.fill_between(t_space, lower_bound, upper_bound, color="lightsteelblue", alpha=0.75, label="1σ region", zorder=2, edgecolor="None")
            plt.fill_between(t_space, lower_bound_2s, upper_bound_2s, color="lightblue", alpha=0.5, label="2σ region", zorder=2, edgecolor="None")
    else:
        if do_fit:
            ax_for_phfold.fill_between(t_space, lower_bound, upper_bound, color="lightsteelblue", alpha=0.75, label="1σ region", zorder=2, edgecolor="None")
            ax_for_phfold.fill_between(t_space, lower_bound_2s, upper_bound_2s, color="lightblue", alpha=0.5, label="2σ region", zorder=2, edgecolor="None")

        ax_for_phfold.set_xlabel("Phase", fontsize=label_fontsize)
        ax_for_phfold.set_ylabel("Radial Velocity [$kms^{-1}$]", fontsize=label_fontsize)
        ax_for_phfold.legend(fontsize=legend_fontsize, loc="upper right")
        ax_for_phfold.set_xlim(-1, 1)
        ax_for_phfold.grid(True, linestyle="--", color="darkgrey", zorder=1)
        ax_for_phfold.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        return

    # plt.title(f"Phasefolded RV curve for Gaia DR3 {GAIA_ID}", fontsize=title_fontsize)
    plt.xlabel("Phase", fontsize=label_fontsize)
    plt.ylabel("Radial Velocity [$kms^{-1}$]", fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize, loc="lower right")
    plt.xlim(-1, 1)
    plt.grid(True, linestyle="--", color="darkgrey", zorder=1)
    if custom_ylim is not None:
        plt.ylim(custom_ylim)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    if custom_saveloc is not None:
        plt.savefig(custom_saveloc, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f"rvplots/{GAIA_ID}_rv_phasefold.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()#
    if do_fit:
        np.savetxt(RVVD_PATH + f"/phasefolds/{GAIA_ID}/orbit_fit.txt",
                   np.array(
                       [[star.amplitude, star.period, star.offset, star.phase], [*errs]]
                   ))
        print(np.array(
            [[star.amplitude, star.period, star.offset, star.phase], [*errs]]
        ))


def plot_phot(
    star: Star,
    *,
    binned: bool = True,
    normalized: bool = True,
    add_rv_plot: bool = False,
    title_fontsize: int = 12,
    label_fontsize: int = 12,
    legend_fontsize: int = 8,
    tick_fontsize: int = 10,
    ignore_sources: list[str] | None = None,
    ignorezi: bool = IGNOREZI,
    ignoreh: bool = IGNOREH,
    dofit: bool = DOFIT,
):
    """
    Replicates the behaviour of the original makephotplot.plot_phot.
    """

    ignore_sources = ignore_sources or []
    # drop unwanted telescopes
    for src in ignore_sources:
        star.lightcurves.pop(src, None)

    nrows = len(star.lightcurves) + int(add_rv_plot)
    if nrows == 0:
        raise RuntimeError("No lightcurves attached to Star object")

    # figure height follows the original heuristic
    if nrows == 1:
        height = 11.69 / 3
    elif nrows == 2:
        height = 11.69 / 2
    elif nrows == 3:
        height = 2 * 11.69 / 3
    else:
        height = 11.69

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(8.27, height),
        dpi=100,
        sharex=True,
        sharey=not add_rv_plot,
    )
    axes = np.atleast_1d(axes)

    row = 0
    if add_rv_plot:
        phasefoldplot(
            star,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            tick_fontsize=tick_fontsize,
            for_phfold=True,
            ax_for_phfold=axes[0],
            do_fit=dofit,
        )
        row += 1

    for tel, lc in star.lightcurves.items():
        ax = axes[row]
        row += 1

        if len(lc.columns) < 4:  # single-band
            x, y, e = _load_single_band(
                lc, star, telescope=tel, binned=binned, normalized=normalized
            )
            ax.errorbar(x, y, yerr=e, fmt='.', color="darkred", ms=3, zorder=5)
            
            # ------------------------------------------------------- CROWD SAP
            if tel.upper() == "TESS" and "TESS_CROWD" in star.metadata:
                lbl = f"TESS flux  (CROWDSAP={star.metadata['TESS_CROWD']:.2f})"
            else:
                lbl = f"{tel} flux"
            
            ax.legend([lbl], fontsize=legend_fontsize,
                      loc="lower right").set_zorder(6)

        else:
            xs, ys, es, bands = _load_multi_band(
                lc, star, telescope=tel, binned=binned, normalized=normalized
            )
            for i, (x, y, e, band) in enumerate(zip(xs, ys, es, bands)):
                if (band == "zi" and ignorezi) or (band == "H" and ignoreh):
                    continue
                col = bandcolors[i % len(bandcolors)]
                ax.errorbar(x, y, yerr=np.abs(e), fmt='.', color=col, ms=3, zorder=5,
                            label=f"{tel} {band} flux")

            ax.legend(fontsize=legend_fontsize, loc="lower right").set_zorder(6)

        ax.set_ylabel("Normalized flux", fontsize=label_fontsize)
        ax.set_xlim(-1, 1)
        ax.grid(True, linestyle='--', color="darkgrey")
        ax.tick_params(labelsize=tick_fontsize)

    axes[-1].set_xlabel("Phase", fontsize=label_fontsize)
    plt.tight_layout()
    Path("lcplots").mkdir(exist_ok=True)
    plt.savefig(f"lcplots/{star.gaia_id}_lcplot.pdf",
                bbox_inches='tight', pad_inches=0)
    plt.show()