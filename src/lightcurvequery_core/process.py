"""
High-level orchestration: download lightcurves, run periodogram, plot.
The progress table now reproduces the exact spinner / ✓ / ✗ / * behaviour
of the original monolithic script.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

import numpy as np
import pandas as pd
from rich import box
from rich.align import Align
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .star import Star
from .utils import ensure_directory_exists
from .fetchers import FETCHERS, getnone
from .plotting import plot_phot              
from .periodogram import plot_common_pgram          # external – kept intact


# ────────────────────────────────────────────────────────────────────
def process_lightcurves(
    gaia_id,
    *,                        # unchanged keyword interface
    skip_tess=False,
    skip_ztf=False,
    skip_atlas=False,
    skip_gaia=False,
    skip_bg=False,
    nsamp=None,
    minp=0.05,
    maxp=50,
    coord=None,
    forced_period=None,
    no_whitening=False,
    binning=True,
    enable_plotting=True,
    show_plots=True,
    ztf_inner_radius: float = 5.0,
    ztf_outer_radius: float = 20.0,
    ztf_preview: bool = False,
    ignore_h: bool = True,
    ignore_zi: bool = True,
    trim_tess: float = 0.0,
):
    """
    Fetch data, build a Star instance, compute periodograms, and create plots.
    Falls back to sequential fetching without Rich table if ~/.ztfquery is missing
    and ZTF is not skipped (to avoid interfering with interactive login prompts).
    """

    # ------------------------------------------------------------------ setup
    base_dir = f"./lightcurves/{gaia_id}"
    ensure_directory_exists(f"./periodograms/{gaia_id}")
    ensure_directory_exists(base_dir)

    surveys = {
        "TESS":      (FETCHERS["TESS"],      os.path.join(base_dir, "tess_lc.txt")),
        "ZTF":       (FETCHERS["ZTF"],       os.path.join(base_dir, "ztf_lc.txt")),
        "ATLAS":     (FETCHERS["ATLAS"],     os.path.join(base_dir, "atlas_lc.txt")),
        "Gaia":      (FETCHERS["Gaia"],      os.path.join(base_dir, "gaia_lc.txt")),
        "BlackGEM":  (FETCHERS["BlackGEM"],  os.path.join(base_dir, "bg_lc.txt")),
    }

    # apply skip flags by replacing the fetcher with the no-op placeholder
    if skip_tess:   surveys["TESS"]     = (getnone, surveys["TESS"][1])
    if skip_ztf:    surveys["ZTF"]      = (getnone, surveys["ZTF"][1])
    if skip_atlas:  surveys["ATLAS"]    = (getnone, surveys["ATLAS"][1])
    if skip_gaia:   surveys["Gaia"]     = (getnone, surveys["Gaia"][1])
    if skip_bg:     surveys["BlackGEM"] = (getnone, surveys["BlackGEM"][1])

    ignore_source = [
        "TESS" if skip_tess else None,
        "ZTF" if skip_ztf else None,
        "ATLAS" if skip_atlas else None,
        "GAIA" if skip_gaia else None,
        "BLACKGEM" if skip_bg else None,
    ]

    ignore_source = [n for n in ignore_source if n is not None]

    spinner = cycle(["\\", "|", "/", "-"])
    status = {name: {'symbol': '', 'style': 'grey50'} for name in surveys}
    pending = set(surveys)
    lock = threading.Lock()

    # ---------------------------------------------------------------- helpers
    def fetch(name, func, filepath, gid):
        if func is getnone:
            sym, style = '*', 'grey50'
        else:
            if os.path.isfile(filepath):
                try:
                    with open(filepath) as f:
                        first = f.readline().strip()
                    ok = bool(first) and 'NaN' not in first
                    sym, style = ('✓', 'green') if ok else ('✗', 'red')
                except Exception:
                    sym, style = '✗', 'red'
            else:
                try:
                    if name == "ZTF":
                        ok = func(gid,
                                  inner_arcsec=ztf_inner_radius,
                                  outer_arcsec=ztf_outer_radius,
                                  do_preview=ztf_preview)
                    else:
                        ok = func(gid)
                    sym, style = ('✓', 'green') if ok else ('✗', 'red')
                except Exception:
                    print(f"Exception in {func.__name__} for {gid}:")
                    print(traceback.format_exc())
                    sym, style = '✗', 'red'
        with lock:
            status[name] = {'symbol': sym, 'style': style}
            pending.discard(name)

    def make_table():
        tbl = Table(
            title=f"Surveying lightcurves for Gaia DR3 {gaia_id}",
            box=box.ROUNDED, show_header=False, show_lines=True, padding=(0, 1)
        )
        for name in surveys:
            tbl.add_column(name, justify="center")
        title_row = [Align.center(Text(n, style="bold")) for n in surveys]

        with lock:
            frame = next(spinner)
            data_row = [
                Text(frame, style="yellow") if n in pending
                else Text(status[n]['symbol'], style=status[n]['style'])
                for n in surveys
            ]
        tbl.add_row(*title_row)
        tbl.add_row(*data_row)
        return tbl

    # ---------------------------------------------------------------- execute
    ztfquery_path = os.path.expanduser("~/.ztfquery")
    run_parallel = os.path.exists(ztfquery_path) or skip_ztf

    if run_parallel:
        # --- parallel mode with Rich table ---
        with Live(make_table(), refresh_per_second=8) as live:
            with ThreadPoolExecutor(max_workers=len(surveys)) as pool:
                for name, (fn, path) in surveys.items():
                    pool.submit(fetch, name, fn, path, gaia_id)

                while pending:
                    live.update(make_table())
                    time.sleep(0.1)
            live.update(make_table())
    else:
        # --- sequential mode without Rich table ---
        print(f"No ~/.ztfquery found and ZTF not skipped — running sequentially.")
        for name, (fn, path) in surveys.items():
            print(f"Fetching {name}...")
            fetch(name, fn, path, gaia_id)
            print(f"  Done: {status[name]['symbol']}")

    # ---------------------------------------------------------------- assemble data into Star object
    star = Star(gaia_id)

    for tel, (_, path) in surveys.items():
        if not os.path.isfile(path):
            continue
        try:
            lc = pd.read_csv(path, header=None)
        except pd.errors.EmptyDataError:
            continue

        if not lc.empty and not lc[0].isna().any():
            star.lightcurves[tel.upper()] = lc

    # ---------------------------------------------------------------- CROWD SAP
    crowd_file = os.path.join(base_dir, "tess_crowdsap.txt")
    if os.path.isfile(crowd_file):
        try:
            star.metadata["TESS_CROWD"] = float(open(crowd_file).read().strip())
        except Exception:
            pass

    # ---------------------------------------------------------------- period analysis & plotting
    if enable_plotting and star.lightcurves:
        plot_common_pgram(
            star,
            ignore_source=ignore_source,
            min_p_given=minp,
            max_p_given=maxp,
            nsamp_given=nsamp,
            whitening=not no_whitening,
            show_plots=show_plots
        )

    # save periodograms
    for name, arr in star.periodograms.items():
        np.savetxt(f"./periodograms/{gaia_id}/{name.lower()}_pgram.txt",
                   np.vstack(arr).T, delimiter=",")

    # final phase-folded lightcurve plot
    if enable_plotting and star.lightcurves:
        star.period = forced_period or star.period
        star.phase = 0
        star.amplitude = star.offset = 0
        plot_phot(
            star,
            add_rv_plot=False,
            ignore_sources=ignore_source,
            ignoreh=ignore_h,
            ignorezi=ignore_zi,
            normalized=True,
            binned=binning,
            show_plots=show_plots,
        )
