from traceback import print_tb

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from common_functions import RVVD_PATH
from periodogramplot import fast_pgram

def plot_loghist(x, bins):
    logbins = np.geomspace(x.min(), x.max(), bins)
    plt.hist(x, bins=logbins, edgecolor="black", facecolor="darkred")
    plt.xscale('log')

ndir = 0
for dir in os.listdir(RVVD_PATH + "/lightcurves"):
    if os.path.isdir(RVVD_PATH + "/lightcurves/" + dir):
        ndir += 1

peaks_atlas = []
peaks_ztf = []
j = 0
for dir in os.listdir(RVVD_PATH + "/lightcurves"):
    if os.path.isdir(RVVD_PATH + "/lightcurves/" + dir):
        j += 1
        print(f"{j}/{ndir}")
        if os.path.isfile(RVVD_PATH + "/lightcurves/" + dir + "/atlas_lc.txt"):
            print("reading csv")
            atlas_lc = pd.read_csv(RVVD_PATH + "/lightcurves/" + dir + "/atlas_lc.txt", header=None)
            print("read csv")
            for f in np.unique(atlas_lc[atlas_lc.columns[3]].to_numpy()):
                if not os.path.isfile(RVVD_PATH + "/lightcurves/" + dir + "/atlas_lc_periodogram.txt"):
                    if f == "H":
                        continue
                    fmask = atlas_lc[atlas_lc.columns[3]].to_numpy() == f
                    x = atlas_lc[atlas_lc.columns[0]].to_numpy()[fmask]
                    y = atlas_lc[atlas_lc.columns[1]].to_numpy()[fmask]
                    dy = atlas_lc[atlas_lc.columns[2]].to_numpy()[fmask]
                    power, periods = fast_pgram(x, y, dy, 0.01, 50, 2000000)
                else:
                    pdata = np.loadtxt(RVVD_PATH + "/lightcurves/" + dir + "/atlas_lc_periodogram.txt", delimiter=",")
                    if pdata.ndim == 1:
                        continue
                    periods = pdata[:, 0]
                    power = pdata[:, 1]
                    print("loop...")
                    for i in range(20):
                        peaks_atlas.append(periods[np.argmax(power)])
                        mask = np.logical_and(periods > peaks_atlas[-1] * 0.99, periods < peaks_atlas[-1] * 1.01)
                        # plt.scatter(periods[np.argmax(power)], power.max(), color="red")
                        periods = periods[~mask]
                        power = power[~mask]
                    print("loop end...")
                    continue
                # plt.plot(periods, power)
                print("loop...")
                for i in range(20):
                    peaks_atlas.append(periods[np.argmax(power)])
                    mask = np.logical_and(periods > peaks_atlas[-1]*0.99, periods < peaks_atlas[-1]*1.01)
                    # plt.scatter(periods[np.argmax(power)], power.max(), color="red")
                    periods = periods[~mask]
                    power = power[~mask]
                print("loop end...")
                # plt.xscale("log")
                # plt.show()
        if os.path.isfile(RVVD_PATH + "/lightcurves/" + dir + "/ztf_lc.txt"):
            print("reading csv")
            ztf_lc = pd.read_csv(RVVD_PATH + "/lightcurves/" + dir + "/ztf_lc.txt", header=None)
            print("read csv")
            if ztf_lc.ndim == 1:
                continue
            for f in np.unique(ztf_lc[ztf_lc.columns[3]].to_numpy()):
                if not os.path.isfile(RVVD_PATH + "/lightcurves/" + dir + "/ztf_lc_periodogram.txt"):
                    fmask = ztf_lc[ztf_lc.columns[3]].to_numpy() == f
                    x = ztf_lc[ztf_lc.columns[0]].to_numpy()[fmask]
                    y = ztf_lc[ztf_lc.columns[1]].to_numpy()[fmask]
                    dy = ztf_lc[ztf_lc.columns[2]].to_numpy()[fmask]
                    power, periods = fast_pgram(x, y, dy, 0.04, 50, 1000000)
                else:
                    pdata = np.loadtxt(RVVD_PATH + "/lightcurves/" + dir + "/ztf_lc_periodogram.txt", delimiter=",")
                    if pdata.ndim <= 1:
                        continue
                    periods = pdata[:, 0]
                    if np.ptp(periods) < 10:
                        fmask = ztf_lc[ztf_lc.columns[3]].to_numpy() == f
                        x = ztf_lc[ztf_lc.columns[0]].to_numpy()[fmask]
                        y = ztf_lc[ztf_lc.columns[1]].to_numpy()[fmask]
                        dy = ztf_lc[ztf_lc.columns[2]].to_numpy()[fmask]
                        power, periods = fast_pgram(x, y, dy, 0.04, 50, 1000000)
                    else:
                        power = pdata[:, 1]
                        print("loop...")
                        for i in range(10):
                            peaks_ztf.append(periods[np.argmax(power)])
                            mask = np.logical_and(periods > peaks_ztf[-1] * 0.99, periods < peaks_ztf[-1] * 1.01)
                            # plt.scatter(periods[np.argmax(power)], power.max(), color="red")
                            periods = periods[~mask]
                            power = power[~mask]
                        print("loop end...")
                        continue
                # plt.plot(periods, power)
                print("loop...")
                for i in range(10):
                    peaks_ztf.append(periods[np.argmax(power)])
                    mask = np.logical_and(periods > peaks_ztf[-1] * 0.99, periods < peaks_ztf[-1] * 1.01)
                    # plt.scatter(periods[np.argmax(power)], power.max(), color="red")
                    periods = periods[~mask]
                    power = power[~mask]
                print("loop end...")
                # plt.xscale("log")
                # plt.show()

peaks_atlas = np.array(peaks_atlas)
peaks_atlas = peaks_atlas[peaks_atlas < 50]
peaks_atlas = peaks_atlas[peaks_atlas > 0.05]

plot_loghist(peaks_atlas, 500)
plt.scatter(peaks_atlas, np.zeros_like(peaks_atlas))
plt.xlabel("Period [d]")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("other_plots/ztfpeaks.pdf")
plt.show()


peaks_ztf = np.array(peaks_ztf)
peaks_ztf = peaks_ztf[peaks_ztf < 50]
peaks_ztf = peaks_ztf[peaks_ztf > 0.05]

plot_loghist(peaks_ztf, 500)
plt.scatter(peaks_ztf, np.zeros_like(peaks_ztf))
plt.xlabel("Period [d]")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("other_plots/atlaspeaks.pdf")
plt.show()
