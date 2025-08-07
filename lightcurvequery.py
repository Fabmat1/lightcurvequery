
import os.path
import re
import time
import sys
from io import StringIO
import argparse
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
#from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from gatspy import periodic
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from ztfquery import lightcurve
import lightkurve as lk

from astroquery.mast import Observations
from astroquery.exceptions import InvalidQueryError
import numpy as np
import pandas as pd
from astropy import units as u

from makephotplot import plot_phot
from models import Star
from periodogramplot import plot_common_pgram
import subprocess
from pathlib import Path
import traceback

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

# Optional: install with pip install rich
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich import box
from rich.text import Text

ATLASBASEURL = "https://fallingstar-data.com/forcedphot"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def magtoflux(mag):
    return 10 ** (-0.4 * mag)

def magerr_to_fluxerr(mag, magerr):
    """Propagate magnitude error to flux error."""
    flux = magtoflux(mag)
    return flux * np.log(10) * 0.4 * magerr


def calcpgramsamples(x_ptp, min_p, max_p):
    n = np.ceil(x_ptp / min_p)
    R_p = (x_ptp / (n - 1) - x_ptp / n)

    df = 1 / min_p - (1 / (min_p + R_p))
    return int(np.ceil((1 / min_p - 1 / max_p) / df)) * 10

class Star:
    def __init__(self, gaia_id):
        self.times = np.array([])
        self.datapoints = np.array([])
        self.datapoint_errors = np.array([])
        self.associations = np.array([])

        self.ra = None
        self.dec = None

        self.period = None
        self.period_err_lo = None
        self.period_err_hi = None
        self.amplitude = None
        self.amplitude_err = None
        self.offset = None
        self.offset_err = None
        self.phase = None
        self.phase_err = None

        self.teff = None
        self.teff_err = None
        self.logg = None
        self.logg_err = None
        self.logy = None
        self.logy_err = None

        self.m_1 = None
        self.m_1_err_lo = None
        self.m_1_err_hi = None
        self.m_2 = None
        self.m_2_err_lo = None
        self.m_2_err_hi = None

        self.t_merger = None

        self.gaia_id = gaia_id
        self.name = gaia_id
        self.lc_classification = None

        self.lightcurves = {}
        self.periodograms = {}

    def sortself(self):
        mask = np.argsort(self.times)

        self.times = self.times[mask]
        self.datapoints = self.datapoints[mask]
        self.datapoint_errors = self.datapoint_errors[mask]

    def make_single_periodogram(self, telescope, min_p, max_p, nsamp):
        lc: pd.DataFrame = self.lightcurves[telescope]

        lc_x = lc[0].to_numpy()
        lc_y = lc[1].to_numpy()
        lc_y_err = lc[2].to_numpy()

        model = periodic.LombScargleFast()

        model.fit(lc_x,
                  lc_y,
                  lc_y_err)
        tdiffs = np.diff(lc_x)

        if min_p is None:
            min_p = np.max([np.min(tdiffs[tdiffs > 0]), 0.01])
        if max_p is None:
            max_p = np.ptp(lc_x) / 2

        if nsamp is None:
            nsamp = calcpgramsamples(np.ptp(lc_x), min_p, max_p)

        pgram = model.score_frequency_grid(1 / max_p, (1 / min_p - 1 / max_p) / nsamp, nsamp)

        freqs = np.linspace(1 / max_p, 1 / min_p, nsamp)
        ps = 1 / freqs

        self.periodograms[telescope] = np.vstack((pgram, ps))

    def make_multiband_periodogram(self, telescope, min_p, max_p, nsamp):
        lc: pd.DataFrame = self.lightcurves[telescope]

        lc_x = lc[0].to_numpy()
        lc_y = lc[1].to_numpy()
        lc_y_err = lc[2].to_numpy()

        model = periodic.LombScargleMultibandFast()

        model.fit(lc_x,
                  lc_y,
                  lc_y_err,
                  lc[3])
        tdiffs = np.diff(lc_x)

        if min_p is None:
            min_p = np.max([np.min(tdiffs[tdiffs > 0]), 0.01])
        if max_p is None:
            max_p = np.ptp(lc_x) / 2

        if nsamp is None:
            nsamp = calcpgramsamples(np.ptp(lc_x), min_p, max_p)

        pgram = model.score_frequency_grid(1 / max_p, (1 / min_p - 1 / max_p) / nsamp, nsamp)

        freqs = np.linspace(1 / max_p, 1 / min_p, nsamp)
        ps = 1 / freqs

        self.periodograms[telescope] = np.vstack((pgram, ps))

    def calculate_periodograms(self, min_p=None, max_p=None, nsamp=None, telescope=None):
        if telescope is None:
            for telescope, lc in self.lightcurves.items():
                if len(lc.columns) == 4:
                    self.make_multiband_periodogram(telescope, min_p, max_p, nsamp)
                else:
                    self.make_single_periodogram(telescope, min_p, max_p, nsamp)
        else:
            if len(self.lightcurves[telescope].columns) == 4:
                self.make_multiband_periodogram(telescope, min_p, max_p, nsamp)
            else:
                self.make_single_periodogram(telescope, min_p, max_p, nsamp)


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



def ensure_directory_exists(directory):
    """Ensure that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def getnone(gaia_id):
    pass


def process_lightcurves(
    gaia_id,
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
    enable_plotting=True
):
    """Fetch and plot lightcurves for a given Gaia ID."""
    base_dir = f"./lightcurves/{gaia_id}"
    ensure_directory_exists(f"./periodograms/{gaia_id}")
    ensure_directory_exists(base_dir)

    # Map survey name to function and filename
    surveys = {
        "TESS": (gettesslc, os.path.join(base_dir, "tess_lc.txt")),
        "ZTF": (getztflc, os.path.join(base_dir, "ztf_lc.txt")),
        "ATLAS": (getatlaslc, os.path.join(base_dir, "atlas_lc.txt")),
        "Gaia": (getgaialc, os.path.join(base_dir, "gaia_lc.txt")),
        "BlackGEM": (getbglc, os.path.join(base_dir, "bg_lc.txt")),
    }
    # Apply skip flags
    if skip_tess:   surveys["TESS"]   = (getnone, surveys["TESS"][1])
    if skip_ztf:    surveys["ZTF"]    = (getnone, surveys["ZTF"][1])
    if skip_atlas:  surveys["ATLAS"]  = (getnone, surveys["ATLAS"][1])
    if skip_gaia:   surveys["Gaia"]   = (getnone, surveys["Gaia"][1])
    if skip_bg:     surveys["BlackGEM"] = (getnone, surveys["BlackGEM"][1])

    spinner = cycle(["\\", "|", "/", "-"])
    status = {name: {'symbol': '', 'style': 'grey50'} for name in surveys}
    pending = set(surveys)
    lock = threading.Lock()

    def fetch(name, func, filepath, gid):
        # If file exists, inspect content
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r') as f:
                    first_line = f.readline().strip()
                # Empty or contains NaN
                if not first_line or 'NaN' in first_line:
                    sym, style = '✗', 'red'
                else:
                    sym, style = '✓', 'green'
            except Exception as e:
                print(f"Exception while reading file {filepath}:\n{traceback.format_exc()}")
                sym, style = '✗', 'red'
        else:
            try:
                result = func(gid)
                sym, style = ('*', 'grey50') if func is getnone else (('✓', 'green') if result else ('✗', 'red'))
            except Exception as e:
                print(f"Exception while calling {func.__name__} with gid={gid}:\n{traceback.format_exc()}")
                sym, style = '✗', 'red'
        with lock:
            status[name] = {'symbol': sym, 'style': style}
            pending.discard(name)

    def build_table():
        tbl = Table(
            title=f"Surveying lightcurves for Gaia DR3 {gaia_id}...",
            box=box.ROUNDED,
            show_header=False,
            show_lines=True,
            expand=False,
            padding=(0,1),
            title_style = "Bold Cyan"
        )
        for name in surveys:
            tbl.add_column(name, justify="center", no_wrap=True)

        names = [Text(name, style="bold") for name in surveys]
        with lock:
            frame = next(spinner)
            symbols = [Text(frame, style="yellow") if name in pending else Text(status[name]['symbol'], style=status[name]['style']) for name in surveys]
        tbl.add_row(*names)
        tbl.add_row(*symbols)
        return tbl

    with Live(build_table(), refresh_per_second=8) as live:
        with ThreadPoolExecutor(max_workers=len(surveys)) as executor:
            futures = [executor.submit(fetch, name, fn, path, gaia_id) for name, (fn, path) in surveys.items()]
            while pending:
                live.update(build_table())
                time.sleep(0.1)
        live.update(build_table())

    star = Star(gaia_id)

    lc_paths = {
        "TESS": f"lightcurves/{gaia_id}/tess_lc.txt",
        "GAIA": f"lightcurves/{gaia_id}/gaia_lc.txt",
        "ZTF": f"lightcurves/{gaia_id}/ztf_lc.txt",
        "ATLAS": f"lightcurves/{gaia_id}/atlas_lc.txt",
        "BLACKGEM": f"lightcurves/{gaia_id}/bg_lc.txt",
    }

    if skip_gaia:
        lc_paths.pop('GAIA', None)
    if skip_atlas:
        lc_paths.pop('ATLAS', None)
    if skip_tess:
        lc_paths.pop('TESS', None)
    if skip_ztf:
        lc_paths.pop('ZTF', None)
    if skip_bg:
        lc_paths.pop('BLACKGEM', None)

    for telescope, fpath in lc_paths.items():
        if os.path.isfile(fpath):
            # print(f"Found lightcurve at {fpath}, reading...")
            try:
                lc = pd.read_csv(fpath, delimiter=",", header=None)
            except pd.errors.EmptyDataError:
                print(f"Error reading {fpath}! Is the file empty? If this is about gaia_lc.txt then this can be safely ignored...")
                print(f"Continuing anyways...")
                continue

            if telescope == "GAIA":
                if " NaN" not in lc.iloc[0].to_list():
                    # "TimeBP", "FG", "e_FG", "FBP", "e_FBP", "FRP", "e_FRP"
                    lc = pd.DataFrame({
                        0: pd.concat([lc[0], lc[1], lc[2]], ignore_index=True),
                        1: pd.concat([lc[3], lc[4], lc[5]], ignore_index=True),
                        2: pd.concat([lc[6], lc[7], lc[8]], ignore_index=True),
                        3: ["G"] * len(lc) + ["BP"] * len(lc) + ["RP"] * len(lc)
                    })
                    lc = lc.dropna()
                else:
                    continue

            if pd.isnull(lc[0].to_numpy()).any():
                continue

            star.lightcurves[telescope] = lc

    if not enable_plotting:
        return
    #try:
    plot_common_pgram(star, ignore_source=[], min_p_given=minp, max_p_given=maxp, nsamp_given=nsamp, whitening=~no_whitening)
    #except TypeError:
    #    print("No lightcurve data was found!")
    #    exit()

    for telescope, fpath in lc_paths.items():
        if os.path.isfile(fpath):
            try:
                lc = pd.read_csv(fpath, delimiter=",", header=None)
            except pd.errors.EmptyDataError:
                print(f"Error reading {fpath}! Is the file empty?")
                print(f"Continuing anyways...")

            if telescope == "GAIA":
                try:
                    if " NaN" not in lc.iloc[0].to_list():
                        # "TimeBP", "FG", "e_FG", "FBP", "e_FBP", "FRP", "e_FRP"
                        lc = pd.DataFrame({
                            0: pd.concat([lc[0], lc[1], lc[2]], ignore_index=True),
                            1: pd.concat([lc[3], lc[4], lc[5]], ignore_index=True),
                            2: pd.concat([lc[6], lc[7], lc[8]], ignore_index=True),
                            3: ["G"] * len(lc) + ["BP"] * len(lc) + ["RP"] * len(lc)
                        })
                        lc = lc.dropna()
                    else:
                        continue
                except KeyError:
                    print("Gaia data was empty! This probably means that none was found...")
                    continue

            if pd.isnull(lc[0].to_numpy()).any():
                continue

            star.lightcurves[telescope] = lc

    for k, v in star.periodograms.items():
        np.savetxt(f"./periodograms/{gaia_id}/{k.lower()}_pgram.txt", np.vstack(v).T, delimiter=",")

    star.phase = 0
    star.amplitude = 0
    star.offset = 0
    if forced_period is not None:
        star.period = forced_period
    plot_phot(star, add_rv_plot=False, ignore_sources=[], ignoreh=True, ignorezi=True, normalized=True, binned=binning)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process lightcurves for astronomical objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lightcurvequery.py 1234567890123456789
  python lightcurvequery.py 1234567890123456789 9876543210987654321
  python lightcurvequery.py --coords 123.456 -45.678
  python lightcurvequery.py --file gaia_ids.txt
        """
    )
    
    # Positional: list of Gaia IDs or RA/DEC
    parser.add_argument(
        'targets',
        nargs='*',
        help='List of Gaia DR3 source IDs or RA DEC coordinates'
    )

    # Optional input alternatives
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--coords',
        nargs=2,
        metavar=('RA', 'DEC'),
        type=float,
        help='Coordinates in degrees (RA DEC)'
    )
    input_group.add_argument(
        '--file', '-i',
        type=str,
        help='Path to file with list of Gaia DR3 source IDs'
    )

    # Data source exclusion flags
    data_group = parser.add_argument_group('Data source options')
    data_group.add_argument('--skip-tess', '-t', action='store_true', help='Exclude TESS data')
    data_group.add_argument('--skip-ztf', '-z', action='store_true', help='Exclude ZTF data')
    data_group.add_argument('--skip-atlas', '-a', action='store_true', help='Exclude ATLAS data')
    data_group.add_argument('--skip-gaia', '-g', action='store_true', help='Exclude Gaia data')
    data_group.add_argument('--skip-bg', '-b', action='store_true', help='Exclude BlackGEM data')
    
    # Processing options
    processing_group = parser.add_argument_group('Processing options')
    processing_group.add_argument('--no-binning', '-B', action='store_true', help='Disable binning')
    processing_group.add_argument('--no-whitening', '-W', action='store_true', help='Disable pre-whitening')
    processing_group.add_argument('--no-plot', '-P', action='store_true', help='Disable plotting output')
    processing_group.add_argument('--plot', '-p', action='store_true', help='Enable plotting output (default)')
    
    # Period analysis options
    period_group = parser.add_argument_group('Period analysis options')
    period_group.add_argument('--min-p', '-m', type=float, default=0.05, metavar='PERIOD', help='Minimum period (d)')
    period_group.add_argument('--max-p', '-M', type=float, default=50.0, metavar='PERIOD', help='Maximum period (d)')
    period_group.add_argument('--force-nsamp', '-n', type=int, metavar='N', help='Force number of samples')
    period_group.add_argument('--force-period', '-f', type=float, metavar='PERIOD', help='Use fixed period (d)')
    
    return parser


def resolve_coordinates_and_gaia_id(args):
    targets = []

    # Handle --file input
    if args.file:
        targets.extend(load_gaia_ids_from_file(args.file))

    # Handle --coords input
    elif args.coords:
        ra, dec = args.coords
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        gaia_id = query_gaia_by_coordinates(coord)
        targets.append((gaia_id, coord))

    # Handle positional targets
    elif args.targets:
        if len(args.targets) == 2:
            try:
                ra, dec = float(args.targets[0]), float(args.targets[1])
                if -360 <= ra <= 360 and -90 <= dec <= 90:
                    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                    gaia_id = query_gaia_by_coordinates(coord)
                    targets.append((gaia_id, coord))
                    return targets
            except ValueError:
                pass  # Fall back to treating as Gaia IDs

        # Treat as Gaia IDs or file paths
        for target in args.targets:
            if target.endswith('.txt') or '/' in target or '\\' in target:
                targets.extend(load_gaia_ids_from_file(target))
            else:
                validate_gaia_id(target)
                targets.append((target, None))
    
    else:
        print("Error: No targets provided. Use positional Gaia IDs, --file or --coords.")
        sys.exit(1)

    return targets


def load_gaia_ids_from_file(filename):
    """
    Load Gaia IDs from a file (one per line).
    Returns list of tuples: [(gaia_id, None), ...]
    """
    targets = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    try:
                        validate_gaia_id(line)
                        targets.append((line, None))
                    except SystemExit:
                        print(f"Error in file {filename}, line {line_num}: Invalid Gaia ID '{line}'")
                        sys.exit(1)
        
        if not targets:
            print(f"Error: No valid Gaia IDs found in file {filename}")
            sys.exit(1)
            
        print(f"Loaded {len(targets)} Gaia IDs from {filename}")
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    
    return targets

def validate_gaia_id(gaia_id):
    """Validate that the Gaia ID looks reasonable."""
    try:
        # Gaia DR3 source IDs are typically 19-digit integers
        int(gaia_id)
        if len(gaia_id) < 10:  # Basic sanity check
            raise ValueError("Gaia ID seems too short")
    except ValueError:
        print(f"Error: Invalid Gaia ID '{gaia_id}'!")
        print("Gaia IDs should be numeric (e.g., 1234567890123456789)")
        sys.exit(1)

def query_gaia_by_coordinates(coord):
    """Query Gaia catalog by coordinates and return the source ID."""
    radius = 5 * u.arcsec
    
    try:
        job = Gaia.cone_search_async(coord, radius=radius)
        results = job.get_results()
        
        if len(results) == 0:
            print("No star found within 5 arcsec! Please check coordinates.")
            sys.exit(1)
        else:
            gaia_id = str(results[0]["SOURCE_ID"])
            print(f"\033[92mStar identified as Gaia DR3 {gaia_id}!\033[0m")  # Green text
            return gaia_id
    except Exception as e:
        print(f"Error querying Gaia catalog: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    
    # Handle coordinates and Gaia ID resolution
    targets = resolve_coordinates_and_gaia_id(args)
    
    # Determine plotting behavior
    enable_plotting = True  # Default to plotting enabled
    if args.no_plot:
        enable_plotting = False
    elif args.plot:
        enable_plotting = True
    
    # Print forced period if specified
    if args.force_period:
        print(f"Period: {args.force_period}")
    
    # Process all targets
    total_targets = len(targets)
    print(f"\nProcessing {total_targets} target{'s' if total_targets != 1 else ''}...\n")
    
    for i, (gaia_id, coord) in enumerate(targets, 1):
        if total_targets > 1:
            print(f"\n{'='*60}")
            print(f"Processing target {i}/{total_targets}: Gaia DR3 {gaia_id}")
            print(f"{'='*60}")
        
        try:
            # Call the main processing function with parsed arguments
            process_lightcurves(
                gaia_id=gaia_id,
                skip_tess=args.skip_tess,
                skip_ztf=args.skip_ztf, 
                skip_atlas=args.skip_atlas,
                skip_gaia=args.skip_gaia,
                skip_bg=args.skip_bg,
                nsamp=args.force_nsamp,
                minp=args.min_p,
                maxp=args.max_p,
                coord=coord,
                forced_period=args.force_period,
                no_whitening=args.no_whitening,
                binning=not args.no_binning,  # Note: inverted logic
                enable_plotting=enable_plotting
            )
            
            if total_targets > 1:
                print(f"✓ Completed processing for Gaia DR3 {gaia_id}")
                
        except Exception as e:
            print(f"✗ Error processing Gaia DR3 {gaia_id}: {e}")
            if total_targets > 1:
                print("Continuing with next target...\n")
                continue
            else:
                raise
    
    if total_targets > 1:
        print(f"\n{'='*60}")
        print(f"Completed processing all {total_targets} targets!")
        print(f"{'='*60}")