import pickle
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.constants import G

from models import Star
import pandas as pd
import numpy as np
import os

warnings.filterwarnings('ignore')


def is_wsl():
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            return "microsoft" in version_info
    except FileNotFoundError:
        return False


if os.name == 'nt':
    RVVD_PATH = "C:/Users/fabia/PycharmProjects/RVVD_plus"
    BASE_PATH = "C:/Users/fabia/"
    DOC_PATH = "C:/Users/fabia/Dokumente"
elif os.name == 'posix':
    if is_wsl():
        RVVD_PATH = "/mnt/c/Users/fabia/PycharmProjects/RVVD_plus"
        BASE_PATH = "/mnt/c/Users/fabia"
        DOC_PATH = "/mnt/c/Users/fabia/Dokumente"
    else:
        RVVD_PATH = "/home/fabian/RVVD"
        BASE_PATH = "/home/fabian"
        DOC_PATH = "/home/fabian/Documents"

matplotlib.use("QtAgg")
try:
    res_table = pd.read_csv(RVVD_PATH + f"/result_parameters.csv")
except FileNotFoundError:
    res_table = None

LAMOST_PATTERNS = [
    r"med-\d+-[A-Za-z0-9]+_sp\d+-\d+_mjd\.txt",
    r"spec-\d+-[A-Za-z0-9]+_sp\d+-\d+_mjd\.txt",
    r"spec-\d+-[A-Za-z0-9]+_\d+_sp\d+-\d+_mjd\.txt",
    r"spec-\d+-[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_\d+_sp\d+-\d+_mjd\.txt",
    r"spec-\d+-[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_sp\d+-\d+_mjd\.txt",
    r"med-\d+-[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_sp\d+-\d+_mjd\.txt",
    r"dummy_spec_mjd\.txt"
]
SDSS_PATTERNS = [
    r"spec-\d+-\d+-\d+_mjd\.txt",
    r"SDSSspec-\d+-\d+_mjd\.txt",
]
SOAR_PATTERNS = [
    r"\d+_\d+_[gG]ai[au]_\d+_\d+_mjd\.txt",
    r"\d+_[gG]ai[au]_\d+_\d+_mjd\.txt",
    r"\d+_\d+_[gG]ai[au]_[dD][rR]3_\d+_\d+_mjd\.txt",
    r"\d+_[gG]ai[au]_[dD][rR]3_\d+_\d+_mjd\.txt",
    r"\d+_[gG]ai[au]_\d+_\d+_mjd\.txt",
    r"\d+_[gG]ai[au]\d+_\d+_mjd\.txt",
    r"\d+_[gG]ai[au]\d+_\d+_red_mjd\.txt",
    r"\d+_[gG]ai[au]\d+_\d+_red_test_mjd\.txt",
    r"\d+_[gG]ai[au]\d+_\d+_blue_mjd\.txt",
    r"\d+_[gG]ai[au]\d+_\d+_blue_test_mjd\.txt",
    r"\d+_TYC_\d+[-_]\d+_\d+_mjd\.txt",
]

t_colors = {
    "ZTF": "green",
    "GAIA": "red",
    "TESS": "blue",
    "ATLAS": "darkorange",
    "BLACKGEM": "magenta"
}


def histogram_with_errorbars(data, uncertainties, outpath, num_bins=20, num_samples=100,
                             xlim=None, hist_color='skyblue', errorbar_color='red',
                             alpha=0.7, xlabel='Value', ylabel='Frequency', figsize=None, **kwargs):
    """
    Plots a histogram with error bars derived from Monte Carlo sampling.

    Parameters:
    - data (np.ndarray): Array of data points.
    - uncertainties (np.ndarray): Array of uncertainties corresponding to each data point.
    - num_bins (int): Number of bins for the histogram (default: 20).
    - num_samples (int): Number of Monte Carlo samples to estimate error bars (default: 100).
    - range (tuple): The range for the histogram (default: range of the data).
    - hist_color (str): Color for the histogram bars (default: 'skyblue').
    - errorbar_color (str): Color for the error bars (default: 'red').
    - alpha (float): Transparency level for the histogram bars (default: 0.7).
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - **kwargs: Additional keyword arguments for plt.errorbar().

    Returns:
    - None
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    # Set default range if not specified
    if xlim is None:
        xlim = (data.min(), data.max())

    # Initialize an array to store histogram counts for each sample
    hist_samples = np.zeros((num_samples, num_bins))

    # Monte Carlo simulation: generate histograms for varied data samples
    for i in range(num_samples):
        # Add random noise to data based on uncertainties
        varied_data = data + np.random.normal(0, uncertainties)

        # Compute histogram for this variation
        hist, bin_edges = np.histogram(varied_data, bins=num_bins)

        hist_samples[i] = hist

    hist, bin_edges = np.histogram(data, bins=num_bins)
    # Calculate the mean and standard deviation for each bin
    # hist_mean = np.mean(hist_samples, axis=0)
    hist_std = np.vstack([np.percentile(hist_samples, 50 - 68.2689492 / 2, axis=0), np.percentile(hist_samples, 50 + 68.2689492 / 2, axis=0)])
    hist_std = (hist_std[1, :] - hist_std[0, :]) / 2

    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.xlim(xlim)
    plt.ylim(0, np.max(hist + hist_std) * 1.1)
    # Plotting the histogram with error bars
    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], color=hist_color, alpha=alpha, label='Histogram', **kwargs)
    plt.errorbar(bin_centers, hist, yerr=hist_std, fmt='None', color=errorbar_color, label='Error bars')

    # Labels and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0)
    plt.show()


def match_telescope(filename):
    for pattern in LAMOST_PATTERNS:
        if re.match(pattern, filename):
            return 0  # LAMOST
    for pattern in SDSS_PATTERNS:
        if re.match(pattern, filename):
            return 1  # SDSS
    for pattern in SOAR_PATTERNS:
        if re.match(pattern, filename):
            return 2  # SOAR
    print("No match for " + filename)
    return None


def gen_telescope_time_associations(file_list):
    associations = []
    times = []

    for f in file_list:
        filepath = os.path.join(RVVD_PATH + "/spectra_processed", f)

        # Match the file name to a telescope pattern
        telescope_id = match_telescope(f)
        if telescope_id is not None:
            # Load data from the file (assuming 1D array or scalar)
            data = np.loadtxt(filepath)
            # Ensure data is a 1D array, even for single values
            if np.isscalar(data) or data.ndim == 0:
                data = np.array([data])
            # Save the telescope and number of points
            associations.append(np.repeat(telescope_id, np.size(data)))
            times.append(data)
    try:
        associations = np.concatenate(associations)
    except ValueError:
        print(file_list)
        raise ValueError("No match for telescope files")
    times = np.concatenate(times)

    mask = np.argsort(times)

    times = times[mask]
    associations = associations[mask]
    return times, associations


def m2(K, P, i, m1):
    K *= 1000
    P *= 24
    P *= 60
    P *= 60
    i = i / 360 * 2 * np.pi

    b = K ** 3 * P * 1 / (2 * np.pi * G * np.sin(i) ** 3)

    m1 *= 1.989e+30
    p1 = (2 * b ** 3 + 18 * b ** 2 * m1 + 5.1962 * np.sqrt(4 * b ** 3 * m1 ** 3 + 27 * b ** 2 * m1 ** 4) + 27 * b * m1 ** 2) ** (1 / 3)
    return (0.26457 * p1 - (0.41997 * (-b ** 2 - 6 * b * m1) / p1) + 1 / 3 * b) / (1.989e+30)


def asymmetrical_random(mean, lower_error, upper_error, size=1):
    combined_weight = np.sqrt(lower_error ** 2 * np.pi) + np.sqrt(upper_error ** 2 * np.pi)

    lo_weight = np.sqrt(lower_error ** 2 * np.pi) / combined_weight
    hi_weight = np.sqrt(upper_error ** 2 * np.pi) / combined_weight

    # Generate uniform random numbers to decide whether to use lower or upper error
    random_signs = np.random.choice([-1, 1], p=[lo_weight, hi_weight], size=size)

    # Apply lower_error for values below the mean and upper_error for values above the mean
    random_values = np.where(random_signs < 0,
                             -np.abs(np.random.normal(0, lower_error, size)) + mean,
                             np.abs(np.random.normal(0, upper_error, size)) + mean)

    return random_values


def masscalc(K, K_err, P, P_err, i, i_err, M_1, M_1_err, N=100000):
    if isinstance(K_err, list) or isinstance(K_err, tuple):
        Ks = asymmetrical_random(K, *K_err, N)
    else:
        Ks = np.random.normal(K, K_err, N)
    if isinstance(P_err, list) or isinstance(P_err, tuple):
        Ps = asymmetrical_random(P, *P_err, N)
    else:
        Ps = np.random.normal(P, P_err, N)
    if isinstance(i_err, list) or isinstance(i_err, tuple):
        iss = asymmetrical_random(i, *i_err, N)
    else:
        iss = np.random.normal(i, i_err, N)
    if isinstance(M_1_err, list) or isinstance(M_1_err, tuple):
        m1s = asymmetrical_random(M_1, *M_1_err, N)
    else:
        m1s = np.random.normal(M_1, M_1_err, N)

    m2s = m2(Ks, Ps, iss, m1s)
    m2s = m2s[~np.isnan(m2s)]
    m2s_lower = m2s[m2s < np.median(m2s)]
    m2s_high = m2s[m2s > np.median(m2s)]
    m2_lolim = np.percentile(m2s_lower, 31.7310508)
    m2_hilim = np.percentile(m2s_high, 68.2689492)

    return np.median(m2s), np.median(m2s) - m2_lolim, m2_hilim - np.median(m2s)


def merger_time(M_1, M_2, p):
    p = 86400 * p
    return 3.22e-3 * ((M_1 + M_2) ** (1 / 3)) / (M_1 * M_2) * p ** (8 / 3)


def load_star(gaia_id, row=None, ignore_med=False):
    rvs = pd.read_csv(RVVD_PATH + f"/output/{gaia_id}/RV_variation.csv")

    res_comparison = res_table.loc[res_table["source_id"] == gaia_id]
    associated_files = res_comparison["associated_files"].iloc[0].split(";")
    file_list = os.listdir(RVVD_PATH + "/spectra_processed")
    file_list = [f for f in file_list if "mjd" in f and "_".join(f.split("_")[:-1]) in associated_files]
    if len(file_list) == 0:
        print("No MJDs found for " + str(gaia_id))
    star = Star(gaia_id)

    spec_file_list_fits = [f for f in os.listdir(RVVD_PATH + "/spectra_processed") if "mjd" not in f and "_".join(f.split("_")[:-1]) in associated_files]
    spec_file_list = []

    for spec_file in spec_file_list_fits:
        if "med" in spec_file and ignore_med:
            continue
        n = 1
        file_name = spec_file.replace(".fits", f"_{n:02d}.txt")
        while os.path.exists(RVVD_PATH + "/spectra_processed/" + file_name):
            nstr = f"{n:02d}"
            spec_file_list.append(file_name)
            n += 1
            file_name = "_".join(spec_file.split("_")[:-1]) + "_" + nstr + ".txt"

    for spec_file in spec_file_list:
        try:
            spec_mjd = np.loadtxt(os.path.join(RVVD_PATH, "spectra_processed", "_".join(spec_file.split("_")[:-1]) + "_mjd.txt"))[int(spec_file.split("_")[-1].replace(".txt", "")) - 1]
        except IndexError:
            spec_mjd = np.loadtxt(os.path.join(RVVD_PATH, "spectra_processed", "_".join(spec_file.split("_")[:-1]) + "_mjd.txt"))
            if not np.ndim(spec_mjd) == 0:
                raise AssertionError
            else:
                spec_mjd = float(spec_mjd)
        star.spectra[spec_mjd] = np.loadtxt(RVVD_PATH + "/spectra_processed/" + spec_file, delimiter=" ")

    star.ra = res_table.loc[res_table["source_id"] == gaia_id]["ra"].iloc[0]
    star.dec = res_table.loc[res_table["source_id"] == gaia_id]["dec"].iloc[0]
    try:
        alias = res_table.loc[res_table["source_id"] == gaia_id]["alias"].iloc[0]
        star.name = alias
        if star.name == "-" or pd.isnull(star.name):
            raise IndexError
    except IndexError:
        coord = SkyCoord(ra=star.ra * u.degree, dec=star.dec * u.degree, frame='icrs')
        # Convert to HMS/DMS format
        ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':').replace(":", "")
        dec_dms = coord.dec.to_string(unit=u.degree, sep=':').replace(":", "")
        star.name = "SDSSJ" + (ra_hms + "-" if dec_dms[0] == "-" else ra_hms + "+") + dec_dms
    finally:
        if "SDSS" in star.name or "LAMOST" in star.name or star.name[0] == "J":
            namepattern = r'(SDSSJ|LAMOSTJ)(\d+)\.\d+([+-]\d+)\.\d+'

            # Replace the matched pattern with just the integer parts
            star.name = "J" + re.sub(namepattern, r'\2\3', star.name)

    perr_temp = None

    if os.path.isfile(RVVD_PATH + f"/phasefolds/{gaia_id}/orbit_fit.txt"):
        params = np.loadtxt(RVVD_PATH + f"/phasefolds/{gaia_id}/orbit_fit.txt", delimiter=" ")
        star.amplitude, star.period, star.offset, star.phase = params[0, :]
        star.amplitude_err, perr_temp, star.offset_err, star.phase_err = params[1, :]

        if star.phase > 1:
            star.phase -= 1
        elif star.phase < 0:
            star.phase += 1
    elif os.path.isfile(RVVD_PATH + f"/phasefolds/{gaia_id}/orbit.txt"):
        params = np.loadtxt(RVVD_PATH + f"/phasefolds/{gaia_id}/orbit.txt", skiprows=1, delimiter=",")
        star.period, star.amplitude, star.offset, star.phase = params
        if star.phase > 1:
            star.phase -= 1
        elif star.phase < 0:
            star.phase += 1

    if row is not None:
        if not pd.isnull(row["p_err_lo"]):
            star.period_err_lo = row["p_err_lo"]
            star.period_err_hi = row["p_err_hi"]
        else:
            if perr_temp is not None:
                star.period_err_lo = perr_temp
                star.period_err_hi = perr_temp

    try:
        star.times = rvs["mjd"].to_numpy()
        star.datapoints = rvs["culum_fit_RV"].to_numpy()
        star.datapoint_errors = rvs["u_culum_fit_RV"].to_numpy()
    except KeyError:
        print(star.gaia_id)
        print(rvs.columns)
        raise KeyError

    star.sortself()

    if os.path.isfile(BASE_PATH + f"/spektralfits/{gaia_id}/results_conf.fits"):
        try:
            hdul = fits.open(BASE_PATH + f"/spektralfits/{gaia_id}/results_conf.fits")
            atmos_param_table = Table(hdul[1].data).to_pandas()

            teffrow = atmos_param_table[atmos_param_table["name"].str.contains("teff")]
            loggrow = atmos_param_table[atmos_param_table["name"].str.contains("logg")]
            logyrow = atmos_param_table[atmos_param_table["name"].str.contains("HE")]

            star.teff = teffrow["value"].iloc[0]
            star.logg = loggrow["value"].iloc[0]
            star.logy = logyrow["value"].iloc[0]

            star.teff_err = (teffrow["conf_max"].iloc[0] - teffrow["conf_min"].iloc[0]) / 2
            star.logg_err = (loggrow["conf_max"].iloc[0] - loggrow["conf_min"].iloc[0]) / 2
            star.logy_err = (logyrow["conf_max"].iloc[0] - logyrow["conf_min"].iloc[0]) / 2
            star.logg_err = np.sqrt(star.logg_err ** 2 + 0.1 ** 2)
            star.logy_err = np.sqrt(star.logy_err ** 2 + 0.1 ** 2)
            star.teff_err = np.sqrt(star.teff_err ** 2 + 1000 ** 2)
        except IndexError:
            try:
                if os.path.isfile(RVVD_PATH + f"/models/{gaia_id}/results_conf.fits"):
                    hdul = fits.open(RVVD_PATH + f"/models/{gaia_id}/results_conf.fits")
                    atmos_param_table = Table(hdul[1].data).to_pandas()

                    teffrow = atmos_param_table[atmos_param_table["name"].str.contains("teff")]
                    loggrow = atmos_param_table[atmos_param_table["name"].str.contains("logg")]
                    logyrow = atmos_param_table[atmos_param_table["name"].str.contains("HE")]

                    star.teff = teffrow["value"].iloc[0]
                    star.logg = loggrow["value"].iloc[0]
                    star.logy = logyrow["value"].iloc[0]

                    star.teff_err = (teffrow["conf_max"].iloc[0] - teffrow["conf_min"].iloc[0]) / 2
                    star.logg_err = (loggrow["conf_max"].iloc[0] - loggrow["conf_min"].iloc[0]) / 2
                    star.logy_err = (logyrow["conf_max"].iloc[0] - logyrow["conf_min"].iloc[0]) / 2
                    star.logg_err = np.sqrt(star.logg_err ** 2 + 0.1 ** 2)
                    star.logy_err = np.sqrt(star.logy_err ** 2 + 0.1 ** 2)
                    star.teff_err = np.sqrt(star.teff_err ** 2 + 1000 ** 2)
            except IndexError:
                pass
    elif os.path.isfile(RVVD_PATH + f"/models/{gaia_id}/results_conf.fits"):
        try:
            hdul = fits.open(RVVD_PATH + f"/models/{gaia_id}/results_conf.fits")
            atmos_param_table = Table(hdul[1].data).to_pandas()

            teffrow = atmos_param_table[atmos_param_table["name"].str.contains("teff")]
            loggrow = atmos_param_table[atmos_param_table["name"].str.contains("logg")]
            logyrow = atmos_param_table[atmos_param_table["name"].str.contains("HE")]

            star.teff = teffrow["value"].iloc[0]
            star.logg = loggrow["value"].iloc[0]
            star.logy = logyrow["value"].iloc[0]

            star.teff_err = (teffrow["conf_max"].iloc[0] - teffrow["conf_min"].iloc[0]) / 2
            star.logg_err = (loggrow["conf_max"].iloc[0] - loggrow["conf_min"].iloc[0]) / 2
            star.logy_err = (logyrow["conf_max"].iloc[0] - logyrow["conf_min"].iloc[0]) / 2

            star.logg_err = np.sqrt(star.logg_err ** 2 + 0.1 ** 2)
            star.logy_err = np.sqrt(star.logy_err ** 2 + 0.1 ** 2)
            star.teff_err = np.sqrt(star.teff_err ** 2 + 1000 ** 2)
        except IndexError:
            pass

    mjds, association = gen_telescope_time_associations(file_list)

    star.associations = association[np.isin(np.round(mjds, 6), np.round(star.times, 6))]

    lc_paths = {
        "TESS": RVVD_PATH + f"/lightcurves/{gaia_id}/tess_lc.txt",
        "GAIA": RVVD_PATH + f"/lightcurves/{gaia_id}/gaia_lc.txt",
        "ZTF": RVVD_PATH + f"/lightcurves/{gaia_id}/ztf_lc.txt",
        "ATLAS": RVVD_PATH + f"/lightcurves/{gaia_id}/atlas_lc.txt",
    }
    for telescope, fpath in lc_paths.items():
        if os.path.isfile(fpath):
            lc = pd.read_csv(fpath, delimiter=",", header=None)

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

    if os.path.isfile(RVVD_PATH + f"/SEDs/{gaia_id}/photometry_results_stellar_c1.txt"):
        sed_table = pd.read_csv(RVVD_PATH + f"/SEDs/{gaia_id}/photometry_results_stellar_c1.txt", delimiter=r"\s+")

        m_1_row = sed_table[sed_table["name"] == "c1_M"]
        star.m_1 = m_1_row["value"].iloc[0]
        star.m_1_err_lo = m_1_row["value"].iloc[0] - m_1_row["conf_min"].iloc[0]
        star.m_1_err_hi = m_1_row["conf_max"].iloc[0] - m_1_row["value"].iloc[0]

        if star.m_1 is not None and not pd.isnull(star.m_1):

            try:
                # print(star.gaia_id)
                star.m_2, star.m_2_err_lo, star.m_2_err_hi = masscalc(star.amplitude,
                                                                      star.amplitude_err,
                                                                      star.period,
                                                                      (star.period_err_lo, star.period_err_hi),
                                                                      90,
                                                                      0,
                                                                      star.m_1,
                                                                      (star.m_1_err_lo, star.m_1_err_hi))
                star.t_merger = merger_time(star.m_1, star.m_2, star.period)
            except TypeError:
                pass

    return star


def load_solved_stars(err_limit=0.5, orbit_tracking_table_path=None):
    orbit_tracking_table = pd.read_csv("solved_orbit_tracker.txt" if orbit_tracking_table_path is None else orbit_tracking_table_path, sep=",")

    stars = []
    for ind, row in orbit_tracking_table.iterrows():
        if row["solved"] == "y":
            star = load_star(row["gaia_id"], row)
            if row["phot_type"] is not None:
                star.lc_classification = row["phot_type"]
            else:
                star.lc_classification = "unknown"
            stars.append(star)

    len_before = len(stars)

    for star in stars:
        if star.m_1_err_lo is not None and star.m_2_err_hi is not None:
            if (star.m_1_err_lo + star.m_1_err_hi) / 2 > err_limit or (star.m_2_err_lo + star.m_2_err_hi) / 2 > err_limit:
                stars.remove(star)

    if len_before - len(stars) != 0:
        print("Excluded " + str(len_before - len(stars)) + " stars")

    return stars


def load_all_stars(orbit_tracking_table_path=None):
    orbit_tracking_table = pd.read_csv("solved_orbit_tracker.txt" if orbit_tracking_table_path is None else orbit_tracking_table_path, sep=",")

    stars = []
    for ind, row in orbit_tracking_table.iterrows():
        star = load_star(row["gaia_id"], row)
        stars.append(star)

    return stars


def really_load_all_stars(cache_path="stars_cache.pkl"):
    """
    Load Star objects from a cache file if it exists, otherwise create them and cache the result.

    Args:
        cache_path (str, optional): Path where the cached stars will be saved

    Returns:
        list: List of Star objects
    """

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                print(f"Loading stars from cache: {cache_path}")
                return pickle.load(f)
        except (pickle.PickleError, EOFError) as e:
            print(f"Error loading cache, will regenerate: {e}")
            # If there's any error loading the cache, continue to regenerate it
            pass

    # If we get here, either cache doesn't exist, is outdated, or failed to load
    print("Generating stars from orbit tracking table...")
    non_bhbs = pd.read_csv("non_bhb.txt", header=None)
    non_bhbs = non_bhbs[non_bhbs.columns[0]].to_numpy().astype(str)

    stars = []
    for i, dir in enumerate(os.listdir(RVVD_PATH + "/output")):
        print(f"{i}/{len(os.listdir(RVVD_PATH + '/output'))}")
        if dir not in non_bhbs:
            continue
        star = load_star(int(dir))
        stars.append(star)

    # Save to cache
    try:
        with open(cache_path, 'wb') as f:
            print(f"Saving stars to cache: {cache_path}")
            pickle.dump(stars, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return stars


def load_unsolved_stars(orbit_tracking_table_path=None):
    orbit_tracking_table = pd.read_csv("solved_orbit_tracker.txt" if orbit_tracking_table_path is None else orbit_tracking_table_path, sep=",")

    stars = []
    for ind, row in orbit_tracking_table.iterrows():
        if row["solved"] != "y" and row["solved"] != "e":
            star = load_star(row["gaia_id"], row)
            if row["phot_type"] is not None:
                star.lc_classification = row["phot_type"]
            else:
                star.lc_classification = "unknown"
            stars.append(star)

    return stars
