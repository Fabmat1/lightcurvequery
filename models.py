import numpy as np
import pandas as pd
from gatspy import periodic



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
