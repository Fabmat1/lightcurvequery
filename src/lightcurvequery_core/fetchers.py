"""
All “getXXXlc” functions live here.  They are copied verbatim from the original
script, with only one change: the utility helpers are imported relatively
(`from .utils import magtoflux, …`).
"""
from __future__ import annotations

# --- keep original imports ---------------------------------------------------
import os
import re
import time
import traceback
import subprocess
from io import StringIO
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time  # noqa: F401 (used in the untouched code)
from astroquery.exceptions import InvalidQueryError
from astroquery.mast import Observations
from astroquery.vizier import Vizier
from gatspy import periodic                                    # noqa: F401
import lightkurve as lk
from ztfquery import lightcurve

from .utils import (
    bcolors,
    magtoflux,
    magerr_to_fluxerr,
    ensure_directory_exists,
)
from .terminal_style import *

import json


ATLASBASEURL = "https://fallingstar-data.com/forcedphot"
CACHE_FILE = Path.home() / ".atlas_forcedphot_cache.json"



def opentessfile(flist):
    if isinstance(flist, str):
        flist = [flist]
    crowdsap = []
    with fits.open(flist[0]) as TESSdata:
        data = TESSdata[1].data
        BJD = np.array(data['TIME'])
        flux = np.array(data['PDCSAP_FLUX'])
        err_flux = np.array(data['PDCSAP_FLUX_ERR'])
        err_flux = err_flux / np.nanmean(flux)
        flux = flux / np.nanmean(flux)
        header = TESSdata[1].header
        crowdsap.append(header['CROWDSAP'])

        if len(flist) > 1:
            for i in range(1, len(flist)):
                with fits.open(flist[i]) as TESSdata:
                    data = TESSdata[1].data
                    BJD = np.append(BJD, np.array(data['TIME']))
                    f = np.array(data['PDCSAP_FLUX'])
                    ef = np.array(data['PDCSAP_FLUX_ERR'])
                    flux = np.append(flux, f / np.nanmean(f))
                    err_flux = np.append(err_flux, ef / np.nanmean(f))
                    header = TESSdata[1].header
                    crowdsap.append(header['CROWDSAP'])

    err_flux = err_flux / np.nanmean(flux)
    flux = flux / np.nanmean(flux)

    bjd = np.array(BJD)
    flux = np.array(flux)
    flux_err = np.array(err_flux)
    crowdsap = np.array(crowdsap)
    return bjd, flux, flux_err, crowdsap

def get_tic(gaia_id: int) -> str:
    """
    Query the IV/39/tic82 catalog from Vizier using a Gaia ID to retrieve the TIC ID.

    Parameters:
    - gaia_id (int): The Gaia ID to query.

    Returns:
    - str: The TIC ID if found, or a message indicating no match was found.
    """
    # Define the catalog and the columns to retrieve
    catalog = "IV/39/tic82"
    columns = ["GAIA", "TIC"]

    # Initialize Vizier with a row limit and column selection
    Vizier.ROW_LIMIT = -1  # No row limit
    Vizier.columns = columns

    # Perform the query
    try:
        result = Vizier.query_constraints(catalog=catalog, GAIA=str(gaia_id))
        if result:
            # Extract the TIC ID from the first table's results
            tic_table: Table = result[0]
            tic_id = tic_table["TIC"][0]  # Assume first match is sufficient
            return str(tic_id)
        else:
            return "No TIC found for Gaia ID: {}".format(gaia_id)
    except Exception as e:
        return f"Error during query: {str(e)}"



def getztflc(gaia_id, *, inner_arcsec: float = 5.0, outer_arcsec: float = 20.0,
             do_preview: bool = False):
    from astropy import units as u
    from .plotting import plot_sky_coords_window        # local import avoids circularity

    coord = SkyCoord.from_name(f'GAIA DR3 {gaia_id}')
    print_info(f"Getting ZTF data (outer radius {outer_arcsec}″)...", gaia_id, "ZTF")

    lcq = lightcurve.LCQuery().from_position(coord.ra.deg, coord.dec.deg, outer_arcsec)

    try:
        if str(lcq.data['ra'].mean()).lower() == "nan":
            print_warning(f"No ZTF data available for this star!", gaia_id)
            if not os.path.isdir(f"lightcurves/{gaia_id}"):
                os.mkdir(f"lightcurves/{gaia_id}")
            with open(f"lightcurves/{gaia_id}/ztf_lc.txt", "w") as file:
                file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
            return False
    except KeyError as e:
        #print(lcq.data)
        print_error(f"Exception while calling getztflc with gid={gid}:\n{traceback.format_exc()}", gaia_id, "ZTF")
        return False

    
    tbl = lcq.data[np.isfinite(lcq.data['ra']) & np.isfinite(lcq.data['dec'])]
    zc = SkyCoord(ra=tbl['ra'], dec=tbl['dec'], unit='deg', frame='icrs')

    # 2) Filter out any bad/missing RA or DEC
    mask_finite = np.isfinite(tbl['ra']) & np.isfinite(tbl['dec'])
    if not np.any(mask_finite):
        print_warning(f"All RA/DEC values are invalid — skipping.", gaia_id, "ZTF")
        return False

    tbl = tbl[mask_finite]
    n_removed = len(mask_finite) - np.count_nonzero(mask_finite)
    if n_removed:
        print_info(f"Filtered out {n_removed} rows with invalid coordinates", gaia_id, "ZTF")

    idx_closest = np.argmin(zc.separation(coord))
    closest_coord = zc[idx_closest]
    mask_keep = zc.separation(closest_coord) < inner_arcsec * u.arcsec
    filtered_tbl = tbl[mask_keep]

    if do_preview:
        plot_sky_coords_window(gaia_id, zc, coord, arcsec_radius=inner_arcsec)

    print_info(f"Best match {zc.separation(coord)[idx_closest].arcsec:.2f}″; "
          f"keeping {len(filtered_tbl)} points within {inner_arcsec}″.", gaia_id, "ZTF")

    mask = filtered_tbl["catflags"] == 0

    data = filtered_tbl[mask]
    dates = data["mjd"].to_numpy()

    if len(dates) == 0:
        print_warning(f"No ZTF data available for this star!", gaia_id, "ZTF")
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/ztf_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        return False
    else:
        print_info(f"Got {len(dates)} samples of ZTF data", gaia_id, "ZTF")
    mags = data["mag"].to_numpy()
    mag_err = data["magerr"].to_numpy()
    filters = data["filtercode"].to_numpy()

    flx = np.zeros_like(mags, dtype=float)
    flx_err = np.zeros_like(mag_err, dtype=float)

    for fil in np.unique(filters):
        mask = (filters == fil) & np.isfinite(mags) & np.isfinite(mag_err)

        flx[mask] = magtoflux(mags[mask])
        flx_err[mask] = magerr_to_fluxerr(mags[mask], mag_err[mask])

        median_flux = np.median(flx[mask])
        flx[mask] /= median_flux
        flx_err[mask] /= median_flux

    table = pd.DataFrame({"mjd": dates, "flx": flx, "flx_err": flx_err, "filter": filters})
    if not os.path.isdir(f"lightcurves/{gaia_id}"):
        os.mkdir(f"lightcurves/{gaia_id}")
    table.to_csv(f"lightcurves/{gaia_id}/ztf_lc.txt", index=False, header=False)
    print_success(f"ZTF data saved!", gaia_id, "ZTF")
    return True


def _load_atlas_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as fh:
                return json.load(fh)
        except Exception:
            pass                       # corrupt file → start with empty cache
    return {}


def _save_atlas_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as fh:
            json.dump(cache, fh)
    except Exception as exc:
        print_warning(f"Warning: could not write ATLAS cache: {exc}", None, "ATLAS")
# ---------------------------------------------------------------------------

def getatlaslc(gaia_id):
    """
    Retrieve an ATLAS forced–photometry light-curve.

    • A small JSON cache in ~/.atlas_forcedphot_cache.json remembers
      {gaia_id : task_id}.  
    • If the result for a cached task already exists we download it
      immediately – no new queue entry is created.  
    • Finished tasks stay in the cache so that subsequent runs will keep
      re-using the same task/result.  The entry is purged only when both
      the queue record *and* the result file have vanished on the server.  
    • Polling is limited to 20 minutes.
    """
    print_info(f"Getting ATLAS data…", gaia_id, "ATLAS")

    token = os.environ.get("ATLASFORCED_SECRET_KEY")
    if not token:
        # Try loading from ~/.atlaskey file
        atlas_key_path = os.path.expanduser("~/.atlaskey")
        if os.path.exists(atlas_key_path):
            try:
                with open(atlas_key_path, 'r') as f:
                    token = f.read().strip()
                if token:
                    print_info(f"Using token from ~/.atlaskey", gaia_id, "ATLAS")
                else:
                    token = None
            except Exception as e:
                print_warning(f"Error reading ~/.atlaskey: {e}", gaia_id, "ATLAS")
                token = None
        
        if not token:
            print_warning("GENERATE AN ATLAS TOKEN AND ADD IT TO YOUR shell rc FILE:", gaia_id, "ATLAS")
            print_warning('   export ATLASFORCED_SECRET_KEY="YOURTOKEN"', gaia_id, "ATLAS")
            print_warning("OR save it to ~/.atlaskey:", gaia_id, "ATLAS")
            print_warning('   echo "YOURTOKEN" > ~/.atlaskey', gaia_id, "ATLAS")
            return False
    else:
        print_info(f"ATLAS: Using stored token", gaia_id, "ATLAS")

    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    cache = _load_atlas_cache()

    task_id: str | None = cache.get(str(gaia_id))
    task_url: str | None = None
    result_url: str | None = None

    # ---------------------------------------------------------------------
    # 1) ——— Try to reuse a cached task ————————————————————————————————
    if task_id:
        task_url = f"{ATLASBASEURL}/queue/{task_id}/"
        result_url = f"{ATLASBASEURL}/static/results/job{task_id}.txt"
        print_info(f"Found cached task {task_id}", gaia_id, "ATLAS")

        # 1a) Does the results file still exist?
        with requests.Session() as s:
            head = s.head(result_url, headers=headers)
        if head.status_code == 200:
            print_info(f"Result already on server – downloading", gaia_id, "ATLAS")
            # go straight to the download step further below
        else:
            # 1b) Results file is gone – check if the queue entry still exists
            with requests.Session() as s:
                r = s.get(task_url, headers=headers)

            if r.status_code == 404:
                print_info(f"Cached task vanished on server – "
                      "submitting a new one", gaia_id, "ATLAS")
                cache.pop(str(gaia_id), None)
                _save_atlas_cache(cache)
                task_id = None
                task_url = None
                result_url = None
            else:
                # keep polling this existing task
                result_url = None
    # ---------------------------------------------------------------------
    # 2) ——— Submit a new task if necessary ————————————————————————————
    if task_url is None:
        coord = SkyCoord.from_name(f"GAIA DR3 {gaia_id}")
        while True:
            with requests.Session() as s:
                resp = s.post(f"{ATLASBASEURL}/queue/",
                              headers=headers,
                              data=dict(ra=coord.ra.deg,
                                        dec=coord.dec.deg,
                                        mjd_min=0,
                                        use_reduced=True))

            if resp.status_code == 201:
                task_url = resp.json()["url"]
                task_id = task_url.rstrip("/").split("/")[-1]
                cache[str(gaia_id)] = task_id
                _save_atlas_cache(cache)
                print_info(f"The task URL is {task_url}", gaia_id, "ATLAS")
                break

            elif resp.status_code == 429:
                detail = resp.json()["detail"]
                m_sec = re.search(r"available in (\d+) seconds", detail)
                m_min = re.search(r"available in (\d+) minutes", detail)
                wait = int(m_sec.group(1)) if m_sec else \
                       int(m_min.group(1)) * 60 if m_min else 10
                print_warning(f"429 Too Many Requests – waiting {wait}s", gaia_id, "ATLAS")
                time.sleep(wait)
            else:
                print_error(f"ERROR {resp.status_code}\n{resp.text}", gaia_id, "ATLAS")
                return False
    # ---------------------------------------------------------------------
    # 3) ——— Poll the task until it finishes (unless result_url already set)
    if result_url is None:
        t0 = time.monotonic()
        taskstarted_printed = False

        while True:
            if time.monotonic() - t0 > 1200:          # 20-min timeout
                print_error(f"Timed-out after 20 min; will try later.", gaia_id, "ATLAS")
                return False

            with requests.Session() as s:
                r = s.get(task_url, headers=headers)

            if r.status_code != 200:
                print_error(f"ERROR {r.status_code}\n{r.text}", gaia_id, "ATLAS")
                return False

            data = r.json()
            if data["finishtimestamp"]:
                result_url = data["result_url"]
                print_info(f"Task is complete with results at {result_url}", gaia_id, "ATLAS")
                break
            elif data["starttimestamp"]:
                if not taskstarted_printed:
                    print_info(f"Task is running (started at {data['starttimestamp']})", gaia_id, "ATLAS")
                    taskstarted_printed = True
                time.sleep(2)
            else:
                print_info(f"Waiting for job to start "
                      f"(queued at {data['timestamp']})", gaia_id, "ATLAS")
                time.sleep(4)
    # ---------------------------------------------------------------------
    # 4) ——— Download the result text and write atlas_lc.txt ————————
    with requests.Session() as s:
        textdata = s.get(result_url, headers=headers).text

    df = pd.read_csv(StringIO(textdata.replace("###", "")), sep=r"\s+")

    #print(f" Got {len(df)} points of ATLAS data!")

    # --------------- quality mask (unchanged) ----------------------------
    good = (
         (df["uJy"] > 0) &
         (df["m"] > 0) &
         (df["err"] == 0) &
         (np.abs(df["uJy"]) / np.abs(df["duJy"]) > 3)
    #     (df["duJy"] < 1e4) &
    #     (df["x"].between(100, 10460)) &
    #     (df["y"].between(100, 10460)) &
    #     (df["maj"].between(1.6, 5)) &
    #     (df["min"].between(1.6, 5)) &
    #     (df["apfit"].between(-1, -0.1)) &
    #     (df["mag5sig"] > 17) &
    #     (df["Sky"] > 17)
    )
    df = df[good].reset_index(drop=True)

    table = df[["MJD", "uJy", "duJy", "F"]].rename(
    columns={"MJD": "mjd", "uJy": "flx",
             "duJy": "flx_err", "F": "filter"}
    )

    # Only convert numeric columns
    numeric_cols = ["mjd", "flx", "flx_err"]
    table[numeric_cols] = table[numeric_cols].astype(float)

    for f in table["filter"].unique():
        m = table["filter"] == f
        med = np.median(table.loc[m, "flx"])
        table.loc[m, "flx"]     /= med
        table.loc[m, "flx_err"] /= med

    outdir = f"lightcurves/{gaia_id}"
    os.makedirs(outdir, exist_ok=True)
    table.to_csv(f"{outdir}/atlas_lc.txt", index=False, header=False)
    print_success(f" {len(table)} points of ATLAS data saved! ({sum(~good)} discarded)", gaia_id, "ATLAS")

    # Keep the finished task in the cache so it can be re-used next time
    # (nothing to do – we never removed it).
    return True

def getbglc(gaia_id):
    # Define current directory path
    current_dir = Path.cwd()

    # Define input and output file paths
    lightcurve_dir = current_dir / "lightcurves" / gaia_id
    output_csv = lightcurve_dir / "output.csv"
    output_txt = lightcurve_dir / "bg_lc.txt"

    # Construct the command to run the query script
    command = [
        "python",
        os.path.expanduser("~/workspace/query_fullsource/query_fullsource.py"),
        str(output_csv),
        "--source_ids",
        gaia_id,
        "--output_type",
        "detections"
    ]

    try:
        # Execute the command without printing anything
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print_error(f"You have no access to the BlackGEM database.", gaia_id, "BLACKGEM")
        return False


    try:
        # Read the generated output.csv
        df = pd.read_csv(output_csv)

        # Keep only the desired columns in the specified order
        cols = ["MJD_OBS", "FNU_OPT", "FNUERRTOT_OPT", "FILTER"]
        df_selected = df[cols]
        if len(df_selected) <= 10:
            print_error(f"Not enough BlackGEM data ({len(df_selected)}) to save.", gaia_id, "BLACKGEM")
            raise FileNotFoundError
        elif len(df_selected) < 50:
            print_warning(f"Warning: Only {len(df_selected)} BlackGEM data points available.", gaia_id, "BLACKGEM")

        # Save the DataFrame to bg_lc.txt without headers and index
        df_selected.to_csv(output_txt, sep=",", header=False, index=False)
        print_success(f"Got BlackGEM data!", gaia_id, "BLACKGEM")
        os.remove(output_csv)
        return True
    except FileNotFoundError:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/bg_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        print_error(f"No BlackGEM data found!", gaia_id, "BLACKGEM")
        return False



def gettesslc(gaia_id):
    print_info(f"Getting MAST data...", gaia_id, "TESS")
    tic = get_tic(gaia_id)


    if "No TIC" in tic or "Error" in tic:
        print_error(f"Invalid TIC: {tic}.", gaia_id, "TESS")
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        return False
    else:
        print_info(f"TIC is {tic}.", gaia_id, "TESS")

    obsTable = Observations.query_criteria(dataproduct_type="timeseries",
                                           project="TESS",
                                           target_name=tic)

    try:
        data = Observations.get_product_list(obsTable)
    except InvalidQueryError as e:
        print_warning(f"No TESS lightcurve available!", gaia_id, "TESS")
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        return False

    times = []
    fluxes = []
    flux_errors = []
    crowdsaps = []

    print_info(f" Looking for short cadence data...", gaia_id, "TESS")
    short_c_lc = Observations.download_products(data, productSubGroupDescription="FAST-LC")
    
    used_paths = []
    if short_c_lc is not None:
        for s in short_c_lc["Local Path"]:
            t2, f2, ef2, cs2 = opentessfile(s)
            times.append(t2)
            fluxes.append(f2)
            flux_errors.append(ef2)
            crowdsaps.append(cs2)
            used_paths.append(s.replace("-a_fast-lc.fits", "-s_lc.fits").replace("-a_fast", "-s"))

    print_info(f"Looking for long cadence data...", gaia_id, "TESS")
    long_c_lc = Observations.download_products(data, productSubGroupDescription="LC")

    if long_c_lc is not None:
        for l in long_c_lc["Local Path"]:
            if l in used_paths:
                print_info(f"Short cadence data used already for TESS, skipping the long cadence data: {l}", gaia_id, "TESS")
                continue
            t1, f1, ef1, cs1 = opentessfile(l)
            times.append(t1)
            fluxes.append(f1)
            flux_errors.append(ef1)
            crowdsaps.append(cs1)

    print_info(f"Looking for FFIs...", gaia_id, "TESS")
    search_result = lk.search_tesscut(f'Gaia DR3{gaia_id}')
    print_info(f"{len(search_result)} FFI datasets found!", gaia_id, "TESS")
    if len(search_result) != 1 and search_result is not None:
        print_warning(f"Downloading TESS FFIs is not implemented yet, sorry!", gaia_id, "TESS") #TODO: implement downloading

    try:
        times = np.concatenate(times)
        flux = np.concatenate(fluxes)
        flux_error = np.concatenate(flux_errors)

        # ----------------------------------------------------------- CROWD SAP
        # ``crowdsaps`` holds one numpy array *per* downloaded FITS file.
        # Flatten it and compute the average so that downstream code
        # (periodograms / light-curve panels) can show this number.
        if crowdsaps:
            try:
                avg_crowd = float(np.nanmean(np.concatenate(crowdsaps)))
            except Exception:
                avg_crowd = np.nan
        else:
            avg_crowd = np.nan
        
        # Persist the value next to the light-curve file so that it
        # can be picked up later without having to re-open the FITSes.
        with open(f"lightcurves/{gaia_id}/tess_crowdsap.txt", 'w') as fh:
            fh.write(f"{avg_crowd:.6f}\n")
        print_success(f"Got {len(times)} datapoints", gaia_id, "TESS")
    except ValueError:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        print_error("TESS data error!", gaia_id, "TESS")
        return False


    if len(times) > 0:
        mask = np.logical_and(np.logical_and(~np.isnan(times), ~np.isnan(flux)), ~np.isnan(flux_error))
        times = times[mask]
        flux = flux[mask]
        flux_error = flux_error[mask]

        sorted_indices = np.argsort(times)
        times = times[sorted_indices]
        flux = flux[sorted_indices]
        flux_error = flux_error[sorted_indices]
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        np.savetxt(f"lightcurves/{gaia_id}/tess_lc.txt", np.vstack((times, flux, flux_error)).T, delimiter=",")
        return True
    else:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        print_error("No TESS data found!", gaia_id, "TESS")
        return False


def getgaialc(gaia_id):
    photquery = Vizier(
        columns=["Source", "TimeG", "TimeBP", "TimeRP", "FG", "FBP", "FRP", "e_FG", "e_FBP", "e_FRP", "noisyFlag"]
    ).query_region(
        SkyCoord.from_name(f'GAIA DR3 {gaia_id}'),
        radius=5 * u.arcsec,
        catalog='I/355/epphot'  # I/355/epphot is the designation for the Gaia photometric catalogue on Vizier
    )
    
    # Create directory if it doesn't exist
    if not os.path.isdir(f"lightcurves/{gaia_id}"):
        os.makedirs(f"lightcurves/{gaia_id}", exist_ok=True)
    
    if len(photquery) != 0:
        table = photquery[0].to_pandas()
        
        # Filter out noisy measurements and select the correct source
        # table = table[table["noisyFlag"] != 1]
        table = table[table["Source"] == int(gaia_id)]
        
        if len(table) == 0:
            print_error("No valid Gaia data found after filtering!", gaia_id, "GAIA")
            # Create empty CSV with proper headers
            empty_df = pd.DataFrame(columns=['MJD', 'Flux', 'Flux_error', 'Filter'])
            empty_df.to_csv(f"lightcurves/{gaia_id}/gaia_lc.txt", index=False, header=False)
            return False
        
        # Reshape data from wide to long format
        lc_data = []
        
        # Process G band
        g_mask = ~pd.isna(table["FG"]) & ~pd.isna(table["TimeG"])
        if g_mask.any():
            for _, row in table[g_mask].iterrows():
                lc_data.append({
                    'MJD': row["TimeG"],
                    'Flux': row["FG"], 
                    'Flux_error': row["e_FG"],
                    'Filter': 'G'
                })
        
        # Process BP band
        bp_mask = ~pd.isna(table["FBP"]) & ~pd.isna(table["TimeBP"])
        if bp_mask.any():
            for _, row in table[bp_mask].iterrows():
                lc_data.append({
                    'MJD': row["TimeBP"],
                    'Flux': row["FBP"],
                    'Flux_error': row["e_FBP"], 
                    'Filter': 'BP'
                })
        
        # Process RP band  
        rp_mask = ~pd.isna(table["FRP"]) & ~pd.isna(table["TimeRP"])
        if rp_mask.any():
            for _, row in table[rp_mask].iterrows():
                lc_data.append({
                    'MJD': row["TimeRP"],
                    'Flux': row["FRP"],
                    'Flux_error': row["e_FRP"],
                    'Filter': 'RP'
                })
        
        if lc_data:
            # Convert to DataFrame and sort by MJD
            lc_df = pd.DataFrame(lc_data)
            lc_df = lc_df.sort_values('MJD').reset_index(drop=True)
            
            # Save as CSV
            lc_df.to_csv(f"lightcurves/{gaia_id}/gaia_lc.txt", index=False, header=False)
            print_success(f" Saved {len(lc_df)} Gaia photometric measurements", gaia_id, "GAIA")
            return True
        else:
            print_error(f"No valid photometric measurements found!", gaia_id, "GAIA")
            # Create empty CSV with proper headers
            empty_df = pd.DataFrame(columns=['MJD', 'Flux', 'Flux_error', 'Filter'])
            empty_df.to_csv(f"lightcurves/{gaia_id}/gaia_lc.txt", index=False, header=False)
            return False
            
    else:
        print_error(f"No Gaia data found!", gaia_id, "GAIA")
        # Create empty CSV with proper headers
        empty_df = pd.DataFrame(columns=['MJD', 'Flux', 'Flux_error', 'Filter'])
        empty_df.to_csv(f"lightcurves/{gaia_id}/gaia_lc.txt", index=False, header=False)
        return False


# the “do nothing” placeholder
def getnone(gaia_id):
    return False


# expose a mapping that the rest of the code can import
FETCHERS: dict[str, Callable[[str], bool]] = {
    "TESS": gettesslc,
    "ZTF": getztflc,
    "ATLAS": getatlaslc,
    "Gaia": getgaialc,
    "BlackGEM": getbglc,
}