import os
import re

from common_functions import load_star, BASE_PATH, RVVD_PATH
from models import Star
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
import pandas as pd
from scipy.optimize import curve_fit

fm.findSystemFonts(fontpaths=None, fontext='ttf')

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
GAIA_ID = 3157967411257049728


def sinusoid(x, amplitude, period, offset, phase):
    result = amplitude * np.sin(2 * np.pi * (x/period+phase)) + offset;
    return result


def sinusoid_wrapper(phase):
    def sinusoid_wrapped(x, amplitude, period, offset):
        return sinusoid(x, amplitude, period, offset, phase)

    return sinusoid_wrapped


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


if __name__ == "__main__":
    if os.path.isfile(RVVD_PATH + f"/phasefolds/{GAIA_ID}/orbit_fit.txt"):
        os.remove(RVVD_PATH + f"/phasefolds/{GAIA_ID}/orbit_fit.txt")
    star = load_star(GAIA_ID)

    phasefoldplot(star)
