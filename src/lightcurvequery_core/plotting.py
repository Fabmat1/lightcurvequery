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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from pathlib import Path
import matplotlib


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
    if telescope.upper() == "GAIA":
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
        
        # FOLD FIRST, before binning
        if fold and star.period is not None:
            x = (x % star.period) / star.period
            x = (x + star.phase) % 1.0
            # Sort by phase for proper binning
            sort_idx = np.argsort(x)
            x, y, e = x[sort_idx], y[sort_idx], e[sort_idx]
        
        # THEN BIN (now in phase space if folded)
        if binned and len(x) > 100:
            nbins = int(np.sqrt(len(x))) + int(np.sqrt(len(x))) % 2
            edges = np.linspace(x.min(), x.max(), nbins + 1)
            idx = np.digitize(x, edges) - 1
            
            # Handle empty bins
            bx = np.array([x[idx == i].mean() if (idx == i).any() else np.nan 
                          for i in range(nbins)])
            by = np.array([y[idx == i].mean() if (idx == i).any() else np.nan 
                          for i in range(nbins)])
            be = np.array([
                np.sqrt((e[idx == i] ** 2).sum()) / max(1, (idx == i).sum())
                if (idx == i).any() else np.nan
                for i in range(nbins)
            ])
            
            # Remove NaN bins
            valid = ~np.isnan(bx)
            x, y, e = bx[valid], by[valid], be[valid]
        
        # Add duplicate phase coverage for plotting (after binning)
        if fold and star.period is not None:
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
    show_plots: bool = True,
):
    """
    Light-curve plotting with the following upgrades:

    4. Robust y-limits: central 99 % of points (0.5–99.5 percentile)
       around the median plus a ±5 % margin.  The same limits are
       applied to *all* LC panels.
    5. No vertical white-space between panels.
    6. The period used for folding (± error, if available) is annotated
       on every LC panel.
    """
    ignore_sources = ignore_sources or []
    for src in ignore_sources:
        star.lightcurves.pop(src, None)

    nrows = len(star.lightcurves) + int(add_rv_plot)
    if nrows == 0:
        raise RuntimeError("No lightcurves attached to Star object")

    # ---------------------------------------------------------------- gather y-values first (for global ylim)
    y_collect: list[np.ndarray] = []

    for tel, lc in star.lightcurves.items():
        if len(lc.columns) < 4:                 # single band

            _, y, _ = _load_single_band(
                lc, star, telescope=tel, binned=binned,
                normalized=normalized, fold=True
            )
            y_collect.append(y)
            print(np.nanmax(y), np.nanmin(y))
        else:                                   # multi band
            xs, ys, es, bands = _load_multi_band(
                lc, star, telescope=tel, binned=binned,
                normalized=normalized, fold=True
            )
            # ignore requested bands while collecting
            for band, arr in zip(bands, ys):
                if (band == "zi" and ignorezi) or (band == "H" and ignoreh):
                    continue
                y_collect.append(arr)

    if not y_collect:
        raise RuntimeError("No usable photometric data found.")
    y_all = np.concatenate(y_collect)
    y_all = y_all[np.isfinite(y_all)]

    p_lo, p_hi = np.percentile(y_all, [0.5, 99.5])
    span = p_hi - p_lo
    ylim_global = (p_lo - 0.05 * span, p_hi + 0.05 * span)

    # ---------------------------------------------------------------- figure & axes
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
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    fig.subplots_adjust(hspace=0)        # remove vertical gaps

    # ---------------------------------------------------------------- optional RV panel
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

    # ---------------------------------------------------------------- plot each LC
    per_string = None
    if star.period is not None:
        if getattr(star, "period_loerr", None) is not None \
           and getattr(star, "period_hierr", None) is not None:
            per_string = (f"P = {star.period:.6f} "
                          f"(+{star.period_hierr:.6f} / "
                          f"-{star.period_loerr:.6f}) d")
        else:
            per_string = f"P = {star.period:.6f} d"

    for tel, lc in star.lightcurves.items():
        ax = axes[row]
        row += 1

        if len(lc.columns) < 4:  # single-band -----------------------------
            x, y, e = _load_single_band(
                lc, star, telescope=tel,
                binned=binned, normalized=normalized
            )
            ax.errorbar(x, y, yerr=e, fmt='.', color="darkred", ms=3, zorder=5)
            label = (f"TESS flux (CROWDSAP={star.metadata['TESS_CROWD']:.2f})"
                     if tel.upper() == "TESS" and "TESS_CROWD" in star.metadata
                     else f"{tel} flux")
            ax.legend([label], fontsize=legend_fontsize,
                      loc="lower right").set_zorder(6)

        else:                     # multi-band ------------------------------
            xs, ys, es, bands = _load_multi_band(
                lc, star, telescope=tel,
                binned=binned, normalized=normalized
            )
            for i, (x, y, e, band) in enumerate(zip(xs, ys, es, bands)):
                if (band == "zi" and ignorezi) or (band == "H" and ignoreh):
                    continue
                col = bandcolors[i % len(bandcolors)]
                ax.errorbar(x, y, yerr=np.abs(e), fmt='.', color=col,
                            ms=3, zorder=5,
                            label=f"{tel} {band} flux")
            ax.legend(fontsize=legend_fontsize, loc="lower right").set_zorder(6)

        # ------------------ cosmetics shared by *all* LC panels -------------
        ax.set_ylabel("Normalized flux", fontsize=label_fontsize)
        ax.set_xlim(-1, 1)
        ax.set_ylim(*ylim_global)
        ax.grid(True, linestyle='--', color="darkgrey")
        ax.tick_params(labelsize=tick_fontsize)

        if per_string:
            ax.annotate(per_string,
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8,
                        bbox=dict(facecolor='white',
                                  edgecolor='none', alpha=0.6))

    # hide duplicate x-tick labels (all but bottom LC panel)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Phase", fontsize=label_fontsize)
    plt.suptitle(f"Lightcurve for Gaia DR3 {star.gaia_id}")

    plt.tight_layout()
    Path("lcplots").mkdir(exist_ok=True)
    plt.savefig(f"lcplots/{star.gaia_id}_lcplot.pdf",
                bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close('all')

# ───────────────────────── ZTF preview (moved from fetchers.py) ─────────────
def plot_sky_coords_window(gaia_id, zc, coord, *, arcsec_radius=5, figsize=(10, 8)):
    import matplotlib, numpy as np
    prev_backend = matplotlib.get_backend()
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt
    from astropy import units as u

    seps_to_gaia = zc.separation(coord)
    idx_closest = np.argmin(seps_to_gaia)
    closest_coord = zc[idx_closest]

    mask = zc.separation(closest_coord) < arcsec_radius * u.arcsec
    window_size = 20/3600
    ra0, dec0 = closest_coord.ra.deg, closest_coord.dec.deg
    ra_min, ra_max = ra0 - window_size, ra0 + window_size
    dec_min, dec_max = dec0 - window_size, dec0 + window_size

    fig, ax = plt.subplots(figsize=figsize)
    win = ((zc.ra.deg >= ra_min) & (zc.ra.deg <= ra_max) &
           (zc.dec.deg >= dec_min) & (zc.dec.deg <= dec_max))
    ax.scatter(zc.ra.deg[win], zc.dec.deg[win], c='lightgray', s=20, alpha=0.5)
    ax.scatter(zc.ra.deg[mask], zc.dec.deg[mask], c='blue', s=40, alpha=0.7)
    ax.scatter(coord.ra.deg, coord.dec.deg, c='red', s=100, marker='*', edgecolors='k')
    ax.scatter(ra0, dec0, c='green', s=80, marker='s', edgecolors='k')
    circ = plt.Circle((ra0, dec0), arcsec_radius/3600, fill=False,
                      color='green', linestyle='--')
    ax.add_patch(circ)
    ax.set_xlim(ra_max, ra_min)
    ax.set_ylim(dec_min, dec_max)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.grid(True, alpha=0.3)

    out_dir = f"lightcurves/{gaia_id}"
    Path(out_dir).mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ztf_query_view.pdf", dpi=300)
    plt.close()
    matplotlib.use(prev_backend, force=True)