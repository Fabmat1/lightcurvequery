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
from .terminal_style import *
from .plotconfig import *

# ---------------------------------------------------------------------- setup
fm.findSystemFonts(fontpaths=None, fontext='ttf')
rc('font', family='sans-serif', **{'sans-serif': ['Helvetica']})

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
    min_points: int = 10,  # Add this parameter
):
    """Load multi-band data, filtering out bands with fewer than min_points."""
    out_x, out_y, out_e, bands = [], [], [], []
    base_x = lc[0].to_numpy()
    
    if telescope.upper() == "GAIA" and lc[0].mean() < 25000: # Prevent applying base conversion more than once (Gaia timestamps are never > 4000 or so)
        base_x += 2455197.5
        base_x = Time(base_x, format="jd").mjd
    # Get unique bands and their counts
    unique_bands, counts = np.unique(lc[3], return_counts=True)
    valid_bands = unique_bands[counts >= min_points]
    
    if len(valid_bands) == 0:
        print_warning(f"Warning: {telescope} has no bands with >= {min_points} points", star.gaia_id, telescope)
        return [], [], [], []
    
    # Filter out bands with too few points
    filtered_out = unique_bands[counts < min_points]
    if len(filtered_out) > 0:
        print_info(f"Skipping bands {list(filtered_out)} (<{min_points} points)", star.gaia_id, telescope)
    
    for band in valid_bands:
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


def phasefoldplot(
    star: Star,
    N_samples=5000,
    config: Optional[PlotConfig] = None,
    for_phfold=False,
    ax_for_phfold: plt.Axes = None,
    do_fit=True,
    custom_saveloc=None,
    custom_ylim=None,
    figsize=None,
):
    """Phase-folded RV plot with PlotConfig integration."""
    if config is None:
        config = PlotConfig()
    
    # Handle figsize precedence: explicit > config > default
    if figsize is None:
        figsize = config.figsize or (6, 4)
    
    if not for_phfold:
        plt.figure(figsize=figsize, dpi=config.dpi, facecolor=config.facecolor)
    
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
            plt.errorbar(
                folded_times[assoc_mask], folded_points[assoc_mask],
                folded_errors[assoc_mask],
                linestyle="none", capsize=config.errorbar_capsize,
                color=color_map_err[assoc_value],
                label=f'{assoc_label_map[assoc_value]}', zorder=9,
                elinewidth=config.errorbar_width,
                markeredgewidth=config.errorbar_width
            )
            plt.scatter(folded_times[assoc_mask], folded_points[assoc_mask], 
                       config.marker_size, marker="o",
                       color=color_map[assoc_value], label="_nolegend_", zorder=10)
        else:
            ax_for_phfold.errorbar(
                folded_times[assoc_mask], folded_points[assoc_mask],
                folded_errors[assoc_mask],
                linestyle="none", capsize=config.errorbar_capsize,
                color=color_map_err[assoc_value],
                label=f'{assoc_label_map[assoc_value]}', zorder=9,
                elinewidth=config.errorbar_width,
                markeredgewidth=config.errorbar_width
            )
            ax_for_phfold.scatter(folded_times[assoc_mask], folded_points[assoc_mask],
                                 config.marker_size, marker="o",
                                 color=color_map[assoc_value], label="_nolegend_", zorder=10)

    # Plot best fit
    if not for_phfold:
        plt.plot(
            t_space, sinusoid(t_space, star.amplitude, 1, star.offset, 0),
            color=config.fit_line_color, linewidth=config.fit_line_width,
            alpha=config.fit_line_alpha, zorder=6, label="Best Fit"
        )
    else:
        ax_for_phfold.plot(
            t_space, sinusoid(t_space, star.amplitude, 1, star.offset, 0),
            color=config.fit_line_color, linewidth=config.fit_line_width,
            alpha=config.fit_line_alpha, zorder=6, label="Best Fit"
        )

    # Confidence regions
    if do_fit:
        sinusoids = np.zeros((N_samples, len(t_space)))
        for i in range(N_samples):
            amplitude_varied = star.amplitude + np.random.normal(0, errs[0])
            period_varied = (star.period + np.random.normal(0, errs[1])) / star.period
            offset_varied = star.offset + np.random.normal(0, errs[2])
            sinusoids[i, :] = sinusoid(t_space, amplitude_varied, period_varied, offset_varied, 0)

        lower_bound = np.percentile(sinusoids, 16, axis=0)
        upper_bound = np.percentile(sinusoids, 84, axis=0)
        lower_bound_2s = np.percentile(sinusoids, 2.5, axis=0)
        upper_bound_2s = np.percentile(sinusoids, 97.5, axis=0)

    # Fill confidence intervals
    if not for_phfold:
        if do_fit:
            plt.fill_between(t_space, lower_bound, upper_bound,
                           color="lightsteelblue", alpha=0.75,
                           label="1σ region", zorder=2, edgecolor="None")
            plt.fill_between(t_space, lower_bound_2s, upper_bound_2s,
                           color="lightblue", alpha=0.5,
                           label="2σ region", zorder=2, edgecolor="None")
    else:
        if do_fit:
            ax_for_phfold.fill_between(t_space, lower_bound, upper_bound,
                                      color="lightsteelblue", alpha=0.75,
                                      label="1σ region", zorder=2, edgecolor="None")
            ax_for_phfold.fill_between(t_space, lower_bound_2s, upper_bound_2s,
                                      color="lightblue", alpha=0.5,
                                      label="2σ region", zorder=2, edgecolor="None")

        if config.show_legend and config.legend_fontsize > 0:
            ax_for_phfold.legend(fontsize=config.legend_fontsize, loc="upper right")
        ax_for_phfold.set_xlim(-1, 1)
        ax_for_phfold.grid(config.grid, linestyle=config.grid_linestyle,
                          color=config.grid_color, alpha=config.grid_alpha, zorder=1)
        ax_for_phfold.tick_params(axis='both', which='major', labelsize=config.tick_fontsize)
        ax_for_phfold.xaxis.offsetText.set_fontsize(config.tick_fontsize)
        ax_for_phfold.yaxis.offsetText.set_fontsize(config.tick_fontsize)
        return

    # Standalone plot cosmetics
    if config.show_legend and config.legend_fontsize > 0:
        plt.legend(fontsize=config.legend_fontsize, loc="lower right")
    
    plt.xlim(-1, 1)
    plt.grid(config.grid, linestyle=config.grid_linestyle,
            color=config.grid_color, alpha=config.grid_alpha, zorder=1)
    
    if custom_ylim is not None:
        plt.ylim(custom_ylim)
    
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    plt.gca().xaxis.offsetText.set_fontsize(config.tick_fontsize)
    plt.gca().yaxis.offsetText.set_fontsize(config.tick_fontsize)
    
    if config.tight_layout:
        plt.tight_layout()
    
    if custom_saveloc is not None:
        plt.savefig(custom_saveloc, bbox_inches='tight', pad_inches=0)
    else:
        Path("rvplots").mkdir(exist_ok=True)
        plt.savefig(f"rvplots/{star.gaia_id}_rv_phasefold.pdf", bbox_inches='tight', pad_inches=0)
    
    plt.show()
    
    if do_fit:
        Path(f"./phasefolds/{star.gaia_id}").mkdir(parents=True, exist_ok=True)
        np.savetxt(f"./phasefolds/{star.gaia_id}/orbit_fit.txt",
                   np.array([[star.amplitude, star.period, star.offset, star.phase], [*errs]]))

def plot_phot(
    star: Star,
    *,
    binned: bool = True,
    normalized: bool = True,
    config: Optional[PlotConfig] = None,
    add_rv_plot: bool = False,
    ignore_sources: list[str] | None = None,
    ignorezi: bool = IGNOREZI,
    ignoreh: bool = IGNOREH,
    dofit: bool = DOFIT,
    show_plots: bool = True,
    min_points: int = 10,
    title: Optional[str] = None, 
):
    """Light-curve plotting with filter handling."""
    # Use default config if none provided
    if config is None:
        config = PlotConfig()

    # Set-up
    ignore_sources = ignore_sources or []
    for src in ignore_sources:
        star.lightcurves.pop(src, None)

    # First pass: check which telescopes have valid data
    valid_telescopes = []
    for tel, lc in star.lightcurves.items():
        if len(lc.columns) < 4:
            # Single band - always valid if has data
            if len(lc) > 0:
                valid_telescopes.append(tel)
        else:
            # Multi-band - check if any bands have enough points
            unique_bands, counts = np.unique(lc[3], return_counts=True)
            valid_bands = unique_bands[counts >= min_points]
            # Also apply ignorezi and ignoreh filters
            valid_bands = [b for b in valid_bands 
                          if not ((b == "zi" and ignorezi) or (b == "H" and ignoreh))]
            if len(valid_bands) > 0:
                valid_telescopes.append(tel)

    nrows = len(valid_telescopes) + int(add_rv_plot)
    if nrows == 0:
        raise RuntimeError("No lightcurves with sufficient data points")

    # Gather y-values (only from valid telescopes)
    per_tel_y: dict[str, list[np.ndarray]] = {tel: [] for tel in valid_telescopes}
    y_collect: list[np.ndarray] = []

    for tel in valid_telescopes:
        lc = star.lightcurves[tel]
        if len(lc.columns) < 4:
            _, y, _ = _load_single_band(
                lc, star, telescope=tel,
                binned=binned, normalized=normalized, fold=True
            )
            per_tel_y[tel].append(y)
            y_collect.append(y)
        else:
            xs, ys, es, bands = _load_multi_band(
                lc, star, telescope=tel,
                binned=binned, normalized=normalized, fold=True,
                min_points=min_points
            )
            for band, arr in zip(bands, ys):
                if (band == "zi" and ignorezi) or (band == "H" and ignoreh):
                    continue
                per_tel_y[tel].append(arr)
                y_collect.append(arr)

    if not y_collect:
        raise RuntimeError("No usable photometric data found.")

    # Global limits
    y_all = np.concatenate(y_collect)
    y_all = y_all[np.isfinite(y_all)]
    
    Q1 = np.percentile(y_all, 25)
    Q3 = np.percentile(y_all, 75)
    IQR = Q3 - Q1
    p_lo_glob = Q1 - 1.5 * IQR
    p_hi_glob = Q3 + 1.5 * IQR
    span_glob = p_hi_glob - p_lo_glob
    ylim_global = (p_lo_glob - 0.05 * span_glob,
                   p_hi_glob + 0.05 * span_glob)

    # Per-telescope limits
    ylim_dict: dict[str, tuple[float, float]] = {"global": ylim_global}
    divergent_telescopes: set[str] = set()

    for tel, arrs in per_tel_y.items():
        y_tel = np.concatenate(arrs)
        y_tel = y_tel[np.isfinite(y_tel)]
        Q1 = np.percentile(y_tel, 25)
        Q3 = np.percentile(y_tel, 75)
        IQR = Q3 - Q1
        p_lo_tel = Q1 - 1.5 * IQR
        p_hi_tel = Q3 + 1.5 * IQR
        span_tel = p_hi_tel - p_lo_tel

        if span_tel > 1.5 * span_glob or span_tel < 0.5 * span_glob:
            ylim_tel = (p_lo_tel - 0.05 * span_tel,
                        p_hi_tel + 0.05 * span_tel)
            ylim_dict[tel] = ylim_tel
            divergent_telescopes.add(tel)
        else:
            ylim_dict[tel] = ylim_global

    # Figure & axes
    if nrows == 1:
        height = 11.69 / 3
    elif nrows == 2:
        height = 11.69 / 2
    elif nrows == 3:
        height = 2 * 11.69 / 3
    else:
        height = 11.69

    figsize = config.figsize or (8.27, height)
    
    # Create figure with constrained_layout parameters if enabled
    if config.constrained_layout:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=1, 
            figsize=figsize, 
            dpi=config.dpi,
            sharex=True, 
            sharey=False, 
            facecolor=config.facecolor,
            layout="constrained"
        )
    else:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=1, 
            figsize=figsize, 
            dpi=config.dpi,
            sharex=True, 
            sharey=False, 
            facecolor=config.facecolor,
        )

    axes = np.atleast_1d(axes)
    
    # Manual adjustment if not using constrained_layout
    if not config.constrained_layout:
        fig.subplots_adjust(
            hspace=config.hspace,
            left=config.left_margin,
            right=config.right_margin,
            top=config.top_margin,
            bottom=config.bottom_margin,
        )

    # Optional RV panel
    row = 0
    if add_rv_plot:
        phasefoldplot(
            star,
            config=config,
            for_phfold=True,
            ax_for_phfold=axes[0],
            do_fit=dofit,
        )
        row += 1

    # Period text
    per_string = None
    if star.period is not None:
        if getattr(star, "period_loerr", None) is not None \
           and getattr(star, "period_hierr", None) is not None:
            per_string = (f"P = {star.period:.6f} "
                          f"(+{star.period_hierr:.6f} / "
                          f"-{star.period_loerr:.6f}) d")
        else:
            per_string = f"P = {star.period:.6f} d"

    # Plotting loop (only valid telescopes)
    for tel in valid_telescopes:
        lc = star.lightcurves[tel]
        ax = axes[row]
        row += 1

        if len(lc.columns) < 4:
            x, y, e = _load_single_band(
                lc, star, telescope=tel,
                binned=binned, normalized=normalized
            )
            ax.errorbar(
                x, y, yerr=e, fmt=config.marker_style, 
                color=config.band_colors[0], ms=config.marker_size, 
                alpha=config.marker_alpha, capsize=config.errorbar_capsize,
                elinewidth=config.errorbar_width, zorder=5, markeredgewidth=config.errorbar_width
            )
            label = (f"TESS flux (CROWDSAP={star.metadata['TESS_CROWD']:.2f})"
                     if tel.upper() == "TESS" and "TESS_CROWD" in star.metadata
                     else f"{tel} flux")
            if config.show_legend and config.legend_fontsize > 0:
                ax.legend([label], fontsize=config.legend_fontsize,
                          loc="lower right").set_zorder(6)

        else:
            xs, ys, es, bands = _load_multi_band(
                lc, star, telescope=tel,
                binned=binned, normalized=normalized,
                min_points=min_points
            )
            
            if len(bands) == 0:
                continue
                
            labels = []
            for i, (x, y, e, band) in enumerate(zip(xs, ys, es, bands)):
                if (band == "zi" and ignorezi) or (band == "H" and ignoreh):
                    continue
                col = config.band_colors[i % len(config.band_colors)]
                ax.errorbar(
                    x, y, yerr=np.abs(e), fmt=config.marker_style, 
                    color=col, ms=config.marker_size, 
                    alpha=config.marker_alpha, capsize=config.errorbar_capsize,
                    elinewidth=config.errorbar_width, zorder=5, markeredgewidth=config.errorbar_width,
                    label=f"{tel} {band} flux"
                )
                labels.append(f"{tel} {band} flux")
            
            if config.show_legend and config.legend_fontsize > 0 and labels:
                ax.legend(fontsize=config.legend_fontsize, loc="lower right").set_zorder(6)

        # Cosmetics
        ax.set_xlim(-1, 1)
        ax.set_ylim(*ylim_dict[tel])
        if config.grid:
            ax.grid(config.grid, linestyle=config.grid_linestyle, 
                    color=config.grid_color, alpha=config.grid_alpha)
        ax.tick_params(labelsize=config.tick_fontsize)
        ax.xaxis.offsetText.set_fontsize(config.tick_fontsize)
        ax.yaxis.offsetText.set_fontsize(config.tick_fontsize)

        if tel in divergent_telescopes and config.annotation_fontsize > 0:
            ax.annotate("(divergent)", xy=(1.02, 0.5), xycoords='axes fraction',
                        rotation=90, va='center', ha='left',
                        fontsize=config.annotation_fontsize, color='crimson')

        if per_string and config.annotation_fontsize > 0:
            ax.annotate(per_string,
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        ha='left', va='top', fontsize=config.annotation_fontsize,
                        bbox=dict(facecolor=config.annotation_bgcolor,
                                  edgecolor='none', 
                                  alpha=config.annotation_alpha))

    # Hide duplicate x-tick labels
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Add labels with proper padding
    if config.show_xlabel:
        fig.supxlabel(
            config.xlabel, 
            fontsize=config.label_fontsize,
            y=config.xlabel_pad if not config.constrained_layout else None,
            ha='center'
        )
    
    if config.show_ylabel:
        fig.supylabel(
            config.ylabel, 
            fontsize=config.label_fontsize,
            x=config.ylabel_pad if not config.constrained_layout else None,
            ha='center'
        )

    if config.show_title and title:
        fig.suptitle(
            title, 
            fontsize=config.title_fontsize,
            y=config.title_pad if not config.constrained_layout else None
        )
    
    # Only apply tight_layout if specified and not using constrained_layout
    if config.tight_layout and not config.constrained_layout:
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Leave room for labels
        except:
            pass  # Sometimes tight_layout fails with complex layouts

    # Save & show
    Path("lcplots").mkdir(exist_ok=True)
    plt.savefig(f"lcplots/{star.gaia_id}_lcplot.pdf",
                bbox_inches='tight', pad_inches=0.02)
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

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
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