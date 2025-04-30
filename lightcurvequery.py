
import os.path
import re
import time
import sys
from io import StringIO

import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from gatspy import periodic
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from ztfquery import lightcurve

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


def magtoflux(mags):
    """
    Convert magnitudes to relative flux centered at 1.

    Parameters:
        mags (array-like): Array of magnitudes.

    Returns:
        np.ndarray: Relative flux array centered at 1.
    """
    mags = np.asarray(mags)
    mag_ref = np.median(mags)
    rel_flux = 10**(-0.4 * (mags - mag_ref))
    return rel_flux

def magerr_to_fluxerr(mag, magerr):
    """
    Convert magnitude errors to relative flux errors, centered at 1.

    Parameters:
        mag (array-like): Array of magnitudes.
        magerr (array-like): Array of magnitude errors.

    Returns:
        np.ndarray: Array of relative flux errors.
    """
    mag = np.asarray(mag)
    magerr = np.asarray(magerr)
    mag_ref = np.median(mag)
    rel_flux = 10**(-0.4 * (mag - mag_ref))
    flux_err = 0.4 * np.log(10) * rel_flux * magerr
    return flux_err
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
    lcq = lightcurve.LCQuery().from_position((coord.ra * u.deg).value, (coord.dec * u.deg).value, 5)

    c2 = SkyCoord(lcq.data["ra"].mean(),lcq.data["dec"].mean(), unit=(u.deg, u.deg), frame='icrs')

    sep = coord.separation(c2)
    print(f"Found object at {lcq.data['ra'].mean()} {lcq.data['dec'].mean()}, with a separation of {sep.to(u.arcsec)}")
    
    mask = lcq.data["catflags"] == 0

    data = lcq.data[mask]
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

    flx = magtoflux(mags)
    flx_err = magerr_to_fluxerr(mags, mag_err)

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
        return

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
                sys.exit()

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
    except FileNotFoundError:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/bg_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        print(f"[{gaia_id}] No BlackGEM data found!")


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
        return
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
        return

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
        return


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
    else:
        if not os.path.isdir(f"lightcurves/{gaia_id}"):
            os.mkdir(f"lightcurves/{gaia_id}")
        with open(f"lightcurves/{gaia_id}/tess_lc.txt", "w") as file:
            file.write("NaN, NaN, NaN, NaN, NaN, NaN, NaN")
        print("No TESS data found!")


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


def process_lightcurves(gaia_id, skip_tess=False, skip_ztf=False, skip_atlas=False, skip_gaia=False, skip_bg=False, nsamp=None, minp=0.05, maxp=50, coord=None, forced_period=None, no_whitening=False, binning=True):
    """Fetch and plot lightcurves for a given Gaia ID."""
    base_dir = f"./lightcurves/{gaia_id}"
    ensure_directory_exists(base_dir)

    # Survey-specific functions and file naming
    surveys = {
        "tess": gettesslc if not skip_tess else getnone,
        "ztf": getztflc if not skip_ztf else getnone,
        "atlas": getatlaslc if not skip_atlas else getnone,
        "gaia": getgaialc if not skip_gaia else getnone,
        "bg": getbglc if not skip_bg else getnone
    }

    for survey_name, fetch_function in surveys.items():
        file_path = os.path.join(base_dir, f"{survey_name}_lc.txt")
        if not os.path.exists(file_path):
            if fetch_function == getnone:
                print(f"Skipping {survey_name}!")
            else:
                print(f"{file_path} not found. Downloading data...")
                fetch_function(gaia_id)
        else:
            if fetch_function == getnone:
                print(f"Skipping {survey_name}!")
            else:
                print(f"{file_path} already exists. Skipping download.")

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

    plot_common_pgram(star, ignore_source=[], min_p_given=minp, max_p_given=maxp, nsamp_given=nsamp, whitening=~no_whitening)

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

    star.phase = 0
    star.amplitude = 0
    star.offset = 0

    if forced_period is not None:
        star.period = forced_period
    plot_phot(star, add_rv_plot=False, ignore_sources=[], ignoreh=True, ignorezi=True, normalized=True, binned=binning)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_lightcurves.py <gaia_id> [OPTIONS]")
        print("OR: python process_lightcurves.py <ra [deg]> <dec [deg]> [OPTIONS]")
        sys.exit(1)

    gaia_id = sys.argv[1]
    if "-" in gaia_id:
        if gaia_id == "--help":
            print("""Usage: python process_lightcurves.py <gaia_id> [OPTIONS]
OR: python process_lightcurves.py <ra [deg]> <dec [deg]> [OPTIONS]

Available commands:
--skip-tess     excludes TESS data
--skip-ztf      excludes ZTF data
--skip-atlas    excludes ATLAS data
--skip-gaia     excludes Gaia data

--no-binning    Disable binning
--no-whitening  Disable pre-whitening

--min-p <period [d]>        Define minimum period for periodogram (Default = 0.05d))
--max-p <period [d]>        Define maximum period for periodogram (Default = 50d)
--force-nsamp <int>         Force define number of samples used in periodogram
--force-period <period [d]> Forcibly define the period of the system
""")
            sys.exit(0)
        else:
            raise ValueError("Invalid Gaia ID! (Did you mean --help?)")

    coord = None
    if len(sys.argv) > 2 and "-" not in sys.argv[2]:
        print(f"Querying for star at RA={sys.argv[1]} DEC={sys.argv[2]}")
        try:
            coord = SkyCoord(ra=float(sys.argv[1])*u.deg, dec=float(sys.argv[2])*u.deg)
        except ValueError:
            print("Invalid coordinate!")
            sys.exit(1)

        # Define the search radius (e.g., 5 arcseconds)
        radius = 5 * u.arcsec

        # Query Gaia around the coordinates
        job = Gaia.cone_search_async(coord, radius=radius)
        results = job.get_results()

        if len(results) == 0:
            print("No star within 5 arcsec! Are the coordinates correct?")
        else:
            gaia_id = results[0]["SOURCE_ID"]
            print(bcolors.OKGREEN+"Star was identified as Gaia DR3 {gaia_id}!".format(gaia_id=gaia_id)+bcolors.ENDC)

    skip_tess = False
    skip_ztf = False
    skip_atlas = False
    skip_gaia = False
    skip_bg = False
    period = None

    min_p = 0.05
    max_p = 50
    Nsamp = None

    no_whitening = False
    binning = True

    for i, arg in enumerate(sys.argv[2:]):
        if "-" in arg:
            if arg == "--skip-tess":
                skip_tess = True
            elif arg == "--skip-ztf":
                skip_ztf = True
            elif arg == "--skip-atlas":
                skip_atlas = True
            elif arg == "--skip-gaia":
                skip_gaia = True
            elif arg == "--skip-bg":
                skip_bg = True
            elif arg == "--min-p":
                min_p = float(sys.argv[2:][i+1])
            elif arg == "--max-p":
                max_p = float(sys.argv[2:][i+1])
            elif arg == "--force-nsamp":
                Nsamp = float(sys.argv[2:][i+1])
            elif arg == "--no-whitening":
                no_whitening = True
            elif arg == "--no-binning":
                binning = False
            elif arg == "--force-period":
                period = float(sys.argv[2:][i+1])
                print(f"Period: {period}")
            elif arg == "--help":
                print("""Usage: python process_lightcurves.py <gaia_id> [OPTIONS]
           
OR: python process_lightcurves.py <ra [deg]> <dec [deg]> [OPTIONS]     
Available commands:
--skip-tess     excludes TESS data
--skip-ztf      excludes ZTF data
--skip-atlas    excludes ATLAS data
--skip-gaia     excludes Gaia data
--skip-bg       excludes BlackGEM data

--no-binning    Disable binning
--no-whitening  Disable pre-whitening

--min-p <period [d]>        Define minimum period for periodogram (Default = 0.05d))
--max-p <period [d]>        Define maximum period for periodogram (Default = 50d)
--force-nsamp <int>         Force define number of samples used in periodogram
--force-period <period [d]> Forcibly define the period of the system
""")
                sys.exit(0)

    process_lightcurves(gaia_id, skip_tess, skip_ztf, skip_atlas, skip_gaia, skip_bg, nsamp=Nsamp, minp=min_p, maxp=max_p, coord=coord, forced_period=period, no_whitening=no_whitening, binning=binning)
