import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_functions import *


def half_amplitude(P, M_1, M_2, i):
    P = P * 24 * 60 * 60
    M_1 = M_1 * 1.989e+30
    M_2 = M_2 * 1.989e+30
    return ((2 * np.pi * G / P * M_2 ** 3 * np.sin(i) ** 3 * 1 / ((M_1 + M_2) ** 2)) ** (1 / 3)) / 1000


class StarSet:
    def __init__(self):
        self.periods = []
        self.period_errs = []
        self.amplitudes = []
        self.amplitude_errs = []

        self.color = None
        self.label = None

    def numpyify(self):
        self.amplitudes = np.array(self.amplitudes)
        self.amplitude_errs = np.array(self.amplitude_errs)
        self.periods = np.array(self.periods)
        self.period_errs = np.array(self.period_errs)

    def make_color(self):
        type_colors = {
            "beaming": "red",
            "reflection": "blue",
            "ellipsoidal": "magenta",
            "hwvir": "green",
            "none": "black",
        }

        labels = {
            "beaming": "Beaming",
            "reflection": "Reflection",
            "ellipsoidal": "Ellipsoidal",
            "hwvir": "HW Vir",
            "none": "None",
        }

        if self.label in list(type_colors.keys()):
            self.color = type_colors[self.label]
            self.label = labels[self.label]
        else:
            self.color = "darkgrey"

    def plot(self):
        mask = ~np.logical_or(np.logical_or(pd.isnull(self.periods), pd.isnull(self.amplitudes)), np.logical_or(pd.isnull([p[0] for p in self.period_errs]), pd.isnull(self.amplitude_errs)))
        periods = self.periods[mask]
        period_errs = self.period_errs[mask]
        amplitudes = self.amplitudes[mask]
        amplitude_errs = self.amplitude_errs[mask]

        plt.errorbar(periods, amplitudes, amplitude_errs, period_errs.T, color=self.color, fmt="None", capsize=3, linestyle='none', zorder=45)
        plt.scatter(periods, amplitudes, c=self.color, marker="o", s=50, zorder=50, label=self.label)


if __name__ == '__main__':
    title_fontsize = 16
    label_fontsize = 16
    legend_fontsize = 12
    tick_fontsize = 14

    stars = load_solved_stars(err_limit=np.inf)

    lc_types = []

    for star in stars:
        lc_types.append(star.lc_classification)
    lc_types = np.unique(lc_types)

    starsets = []
    for t in lc_types:
        starset = StarSet()

        if t is not None:
            starset.label = t
        else:
            starset.label = "unknown"

        for star in stars:
            if star.lc_classification == t:
                if star.m_1 is not None and star.m_2 is not None:
                    if star.m_1 + star.m_2 < 1.45 or (star.m_1_err_hi + star.m_1_err_lo + star.m_2_err_hi + star.m_2_err_lo) / 4 > 0.5:
                        starset.periods.append(star.period)
                        starset.period_errs.append((star.period_err_lo, star.period_err_hi))
                        starset.amplitudes.append(star.amplitude)
                        starset.amplitude_errs.append(star.amplitude_err)
        starset.numpyify()
        starset.make_color()
        starsets.append(starset)

    plt.figure(figsize=(8.27, 4*11.69 / 10))
    pspace = np.logspace(-2, 2, 10000)
    plt.plot(pspace, half_amplitude(pspace, 0.4, 0.8, np.pi / 2), color="black", linewidth=1, zorder=1, label="$M_1=0.4$, $M_2=0.8$")
    plt.plot(pspace, half_amplitude(pspace, 0.4, 0.4, np.pi / 2), color="black", linestyle="--", linewidth=1, zorder=1, label="$M_1=0.4$, $M_2=0.4$")
    plt.plot(pspace, half_amplitude(pspace, 0.47, 0.08, np.pi / 2), color="black", linestyle="-.", linewidth=1, zorder=1, label="$M_1=0.4$, $M_2=0.08$")

    for starset in starsets:
        starset.plot()

    prev_known_stars = pd.read_csv("binaries_lit_topcat.tsv", sep=r"\s+")
    comparison_K = prev_known_stars["K"].to_numpy()
    comparison_P = prev_known_stars["Period"].to_numpy()
    comp = prev_known_stars["Comp"].to_numpy()
    print(np.unique(comp))
    for i, comparison_type in enumerate(np.unique(comp)):
        subset_K = comparison_K[comp == comparison_type]
        subset_P = comparison_P[comp == comparison_type]

        plt.scatter(subset_P, subset_K, s=10, marker="D", color=["salmon", "lightgreen"][i], alpha=1, edgecolor="none", label=f"{comparison_type} (Literature)")


    plt.xscale('log')
    plt.xlim(0.05, 10)
    plt.ylim(0, 400)
    # plt.title(r"Period-Amplitude Diagram")
    plt.xlabel(r"$P$ [d]")
    plt.ylabel(r"$K$ [kms$^{-1}$]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("other_plots/period_amplitude_plot.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
