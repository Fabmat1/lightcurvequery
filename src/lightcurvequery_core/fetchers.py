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

ATLASBASEURL = "https://fallingstar-data.com/forcedphot"



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
# To keep this example concise, we only provide stubs.  Replace these stubs
# with your full original implementations!
def getztflc(gaia_id):
    coord = SkyCoord.from_name(f'GAIA DR3 {gaia_id}')

    print(f"[{gaia_id}] Getting ZTF data...")
    lcq = lightcurve.LCQuery().from_position((coord.ra * u.deg).value, (coord.dec * u.deg).value, 20)

    try:
        if str(lcq.data['ra'].mean()).lower() == "nan":
            print(f"[{gaia_id}] No ZTF data available for this star!")
            if not os.path.isdir(f"lightcurves/{gaia_id}"):
                os.mkdir(f"lightcurves/{gaia_id}")
            with open(f"lightcurves/{gaia_id}/ztf_lc.txt", "w") as file:
                file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
            return False
    except KeyError as e:
        print(lcq.data)
        print(f"Exception while calling getztflc with gid={gid}:\n{traceback.format_exc()}")
        return False

    
    # 1) Grab the raw table
    tbl = lcq.data

    # 2) Filter out any bad/missing RA or DEC
    mask_finite = np.isfinite(tbl['ra']) & np.isfinite(tbl['dec'])
    if not np.any(mask_finite):
        print(f"[{gaia_id}] All RA/DEC values are invalid — skipping.")
        return False

    tbl = tbl[mask_finite]
    n_removed = len(mask_finite) - np.count_nonzero(mask_finite)
    if n_removed:
        print(f"[{gaia_id}] Filtered out {n_removed} rows with invalid coordinates")

    # 3) Build a SkyCoord array from the cleaned table
    zc = SkyCoord(ra=np.array(tbl['ra']) * u.deg,
                  dec=np.array(tbl['dec']) * u.deg,
                  frame='icrs')

    # 4) Find which detection is closest to your queried Gaia position
    seps_to_gaia = zc.separation(coord)
    idx_closest = np.argmin(seps_to_gaia)
    closest_coord = zc[idx_closest]

    # 5) Now keep only those rows within 2″ of that “best” detection
    seps_to_closest = zc.separation(closest_coord)
    mask_within_2as = seps_to_closest < 2 * u.arcsec
    filtered_tbl = tbl[mask_within_2as]

    print(f"[{gaia_id}] Best match was {seps_to_gaia[idx_closest].arcsec:.2f}″ from Gaia; "
          f"keeping {len(filtered_tbl)} points within 2″ of that source.")

    mask = filtered_tbl["catflags"] == 0

    data = filtered_tbl[mask]
    dates = data["mjd"].to_numpy()

    if len(dates) == 0:
        print(f"[{gaia_id}] No ZTF data available for this star!")
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/ztf_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        return False
    else:
        print(f"[{gaia_id}] Got {len(dates)} samples of ZTF data")
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

        print(f"Filter {fil}: median flux = {median_flux}")

    table = pd.DataFrame({"mjd": dates, "flx": flx, "flx_err": flx_err, "filter": filters})
    if not os.path.isdir(f"lightcurves/{gaia_id}"):
        os.mkdir(f"lightcurves/{gaia_id}")
    table.to_csv(f"lightcurves/{gaia_id}/ztf_lc.txt", index=False, header=False)
    print(f"[{gaia_id}] ZTF data saved!")
    return True


def getatlaslc(gaia_id):
    print(f"[{gaia_id}] Getting ATLAS data...")
    if os.environ.get("ATLASFORCED_SECRET_KEY"):
        token = os.environ.get("ATLASFORCED_SECRET_KEY")
        print("ATLAS: Using stored token")
    else:
        print("GENERATE AN ATLAS TOKEN AND ADD IT TO YOUR .bashrc FILE:")
        print(f'export ATLASFORCED_SECRET_KEY="**YOURTOKEN**"')
        return False

    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    coord = SkyCoord.from_name(f'GAIA DR3 {gaia_id}')

    task_url = None
    while not task_url:
        with requests.Session() as s:
            # alternative to token auth
            # s.auth = ('USERNAME', 'PASSWORD')
            resp = s.post(f"{ATLASBASEURL}/queue/", headers=headers, data={"ra": (coord.ra * u.deg).value, "dec": (coord.dec * u.deg).value, "mjd_min": 0, "use_reduced": True})

            if resp.status_code == 201:  # successfully queued
                task_url = resp.json()["url"]
                print(f"The task URL is {task_url}")
            elif resp.status_code == 429:  # throttled
                message = resp.json()["detail"]
                print(f"{resp.status_code} {message}")
                t_sec = re.findall(r"available in (\d+) seconds", message)
                t_min = re.findall(r"available in (\d+) minutes", message)
                if t_sec:
                    waittime = int(t_sec[0])
                elif t_min:
                    waittime = int(t_min[0]) * 60
                else:
                    waittime = 10
                print(f"Waiting {waittime} seconds")
                time.sleep(waittime)
            else:
                print(f"ERROR {resp.status_code}")
                print(resp.text)
                return False

    result_url = None
    taskstarted_printed = False
    while not result_url:
        with requests.Session() as s:
            resp = s.get(task_url, headers=headers)

            if resp.status_code == 200:  # HTTP OK
                if resp.json()["finishtimestamp"]:
                    result_url = resp.json()["result_url"]
                    print(f"Task is complete with results available at {result_url}")
                elif resp.json()["starttimestamp"]:
                    if not taskstarted_printed:
                        print(f"Task is running (started at {resp.json()['starttimestamp']})")
                        taskstarted_printed = True
                    time.sleep(2)
                else:
                    print(f"Waiting for job to start (queued at {resp.json()['timestamp']})")
                    time.sleep(4)
            else:
                print(f"ERROR {resp.status_code}")
                print(resp.text)
                with open(f"lightcurves/{gaia_id}/atlas_lc.txt", "w") as file:
                    file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
                return False

    with requests.Session() as s:
        textdata = s.get(result_url, headers=headers).text

        # if we'll be making a lot of requests, keep the web queue from being
        # cluttered (and reduce server storage usage) by sending a delete operation
        # s.delete(task_url, headers=headers).json()

    print(f"[{gaia_id}] Got ATLAS data!")
    dfresult = pd.read_csv(StringIO(textdata.replace("###", "")), delim_whitespace=True)

    dfresult = dfresult.reset_index(drop=True)

    flx = dfresult["uJy"].to_numpy()

    dfresult = dfresult[flx > 0]
    dfresult = dfresult[dfresult["m"] > 0]
    dfresult = dfresult[dfresult["err"] == 0]
    flx = dfresult["uJy"].to_numpy()
    flx_err = dfresult["duJy"].to_numpy()
    dfresult = dfresult[np.abs(flx) / np.abs(flx_err) > 3]
    flx = dfresult["uJy"].to_numpy()
    flx_err = dfresult["duJy"].to_numpy()
    filter = dfresult["F"].to_numpy()
    lctime = dfresult["MJD"].to_numpy()
    table = pd.DataFrame({"mjd": lctime.astype(float), "flx": flx.astype(float), "flx_err": flx_err.astype(float), "filter": filter})
    if not os.path.isdir(f"lightcurves/{gaia_id}"):
        os.mkdir(f"lightcurves/{gaia_id}")

    table.to_csv(f"lightcurves/{gaia_id}/atlas_lc.txt", index=False, header=False)
    print(f"[{gaia_id}] ATLAS data saved!")
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

    # Execute the command without printing anything
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    try:
        # Read the generated output.csv
        df = pd.read_csv(output_csv)

        # Keep only the desired columns in the specified order
        cols = ["MJD_OBS", "FNU_OPT", "FNUERRTOT_OPT", "FILTER"]
        df_selected = df[cols]
        if len(df_selected) <= 10:
            print(f"Not enough BlackGEM data ({len(df_selected)}) to save.")
            raise FileNotFoundError
        elif len(df_selected) < 50:
            print(f"Warning: Only {len(df_selected)} BlackGEM data points available.")

        # Save the DataFrame to bg_lc.txt without headers and index
        df_selected.to_csv(output_txt, sep=",", header=False, index=False)
        print(f"[{gaia_id}] Got BlackGEM data!")
        os.remove(output_csv)
        return True
    except FileNotFoundError:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/bg_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        print(f"[{gaia_id}] No BlackGEM data found!")
        return False



def gettesslc(gaia_id):
    print(f"[{gaia_id}] Getting MAST data...")
    tic = get_tic(gaia_id)


    if "No TIC" in tic or "Error" in tic:
        print(bcolors.FAIL+tic+bcolors.ENDC)
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        return False
    else:
        print(f"[{gaia_id}] TIC is {tic}.")

    obsTable = Observations.query_criteria(dataproduct_type="timeseries",
                                           project="TESS",
                                           target_name=tic)

    try:
        data = Observations.get_product_list(obsTable)
    except InvalidQueryError as e:
        print("No TESS lightcurve available!")
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        return False

    times = []
    fluxes = []
    flux_errors = []
    crowdsaps = []

    print(f"[{gaia_id}] Looking for short cadence data...")
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

    print(f"[{gaia_id}] Looking for long cadence data...")
    long_c_lc = Observations.download_products(data, productSubGroupDescription="LC")

    if long_c_lc is not None:
        for l in long_c_lc["Local Path"]:
            print(l, "\n", used_paths)
            if l in used_paths:
                print("Short cadence data used already for TESS, skipping the long cadence data: ", l)
                continue
            t1, f1, ef1, cs1 = opentessfile(l)
            times.append(t1)
            fluxes.append(f1)
            flux_errors.append(ef1)
            crowdsaps.append(cs1)

    print(f"[{gaia_id}] Looking for FFIs...")
    search_result = lk.search_tesscut(f'Gaia DR3{gaia_id}')
    print(f"[{gaia_id}] {len(search_result)} FFI datasets found!")
    if len(search_result) != 1 and search_result is not None:
        print(f"[{gaia_id}] Downloading TESS FFI...")
        ffi_download = search_result.download(cutout_size=10)
        print(ffi_download.time.shape)
        print(ffi_download.flux.shape)
        print(ffi_download.flux_err.shape)
        #times.append(ffi_download.time)
        #fluxes.append(ffi_download.flux)
        #flux_errors.append(ffi_download.flux_err)


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
        print(f"[{gaia_id}] Got {len(times)} datapoints")
    except ValueError:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        print("TESS data error!")
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
        print("No TESS data found!")
        return False


def getgaialc(gaia_id):
    photquery = Vizier(
        columns=["Source", "TimeG", "TimeBP", "TimeRP", "FG", "FBP", "FRP", "e_FG", "e_FBP", "e_FRP", "noisyFlag"]
    ).query_region(
        SkyCoord.from_name(f'GAIA DR3 {gaia_id}'),
        radius=5 * u.arcsec,
        catalog='I/355/epphot'  # I/355/epphot is the designation for the Gaia photometric catalogue on Vizier
    )
    if len(photquery) != 0:
        table = photquery[0].to_pandas()
        table = table[table["noisyFlag"] != 1]
        table = table.drop(columns=["noisyFlag"])
        table.columns = ["Source", "TimeG", "TimeBP", "TimeRP", "FG", "FBP", "FRP", "e_FG", "e_FBP", "e_FRP"]
        table = table[table["Source"] == int(gaia_id)]
        table = table.drop(columns=["Source"])
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        table.to_csv(f"lightcurves/{gaia_id}/gaia_lc.txt", index=False, header=False)
        return True
    else:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/gaia_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
            print("No Gaia data found!")
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