import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from common_functions import *
from read_evol import *

class StarGroup:
    def __init__(self, name: str, color: str, label: str):
        self.name = name
        self.color = color
        self.label = label
        self.stars: List[Star] = []
        self.teffs: np.ndarray = np.array([])
        self.teff_errs: np.ndarray = np.array([])
        self.loggs: np.ndarray = np.array([])
        self.logg_errs: np.ndarray = np.array([])

    def add_star(self, star: Star):
        self.stars.append(star)

    def process_data(self):
        """Convert star list to numpy arrays and clean null values"""
        self.teffs = np.array([star.teff for star in self.stars])
        self.teff_errs = np.array([star.teff_err for star in self.stars])
        self.loggs = np.array([star.logg for star in self.stars])
        self.logg_errs = np.array([star.logg_err for star in self.stars])

        # Clean null values
        mask = ~np.logical_or(
            np.logical_or(pd.isnull(self.loggs), pd.isnull(self.teffs)),
            np.logical_or(pd.isnull(self.teff_errs), pd.isnull(self.logg_errs))
        )
        self.teffs = self.teffs[mask]
        self.teff_errs = self.teff_errs[mask]
        self.loggs = self.loggs[mask]
        self.logg_errs = self.logg_errs[mask]

    def plot(self, ax):
        """Plot the star group on the given axes"""
        ax.errorbar(
            self.teffs, self.loggs,
            self.logg_errs, self.teff_errs,
            color=self.color, fmt="o",
            capsize=3, linestyle='none',
            label=self.label
        )


class TeffLoggDiagram:
    def __init__(self):
        self.star_groups: List[StarGroup] = []
        self.fig, self.ax = plt.subplots(figsize=(8.27, 4*11.69 / 10))

    def add_group(self, group: StarGroup):
        self.star_groups.append(group)

    def plot_background(self):
        """Plot HEZAMS and EHB regions"""
        hezams = get_hezams()
        x, y = fill_dorman()
        hezams_teff = 10 ** hezams["log_Teff"]
        hezams_logg = hezams["log_g"]

        # Plot He-ZAMS line
        self.ax.plot(
            [10 ** 4.53, *hezams_teff],
            [6.125, *hezams_logg],
            color="red", alpha=0.5,
            linewidth=5, label="He-ZAMS"
        )

        for teff, logg, mass in zip(10**hezams["log_Teff"], hezams["log_g"], hezams["mass"]):
            if mass <= 1 and mass != 0.85:
                print(teff, logg, mass)
                print(fr"${mass} \,M_{{\odot}}$")
                self.ax.annotate(fr"${mass} \,M_{{\odot}}$", (teff, logg), (0, 0), va="center", ha="center", xycoords='data', textcoords='offset points', color="black", size=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", edgecolor="none"))

        # Plot EHB region
        self.ax.fill(
            10 ** np.array(x), y,
            color="darkgrey",
            alpha=0.5, label="EHB"
        )

    def setup_plot(self):
        """Configure plot settings"""
        self.ax.set_xlim(60000, 20000)
        self.ax.set_ylim(6.75, 4.5)
        self.ax.set_xscale('log')
        self.ax.grid(True, linestyle='--', color='darkgrey', which="both")
        self.ax.set_xlabel(r"$T_\mathrm{eff}$ [K]")
        self.ax.set_ylabel(r"$\log g$ [dex]")
        self.ax.legend(loc="upper left")
        plt.tight_layout()

    def plot(self, save_path: Optional[str] = None):
        """Create the complete plot"""
        self.plot_background()

        for group in self.star_groups:
            group.process_data()
            group.plot(self.ax)

        self.setup_plot()

        if save_path:
            plt.savefig(save_path)
        plt.show()


def classify_star(star: Star) -> str:
    """Classify a star based on its properties"""
    if star.m_1 is not None and star.m_2 is not None:
        total_mass = star.m_1 + star.m_2
        avg_mass_err = (star.m_1_err_hi + star.m_1_err_lo +
                        star.m_2_err_hi + star.m_2_err_lo) / 4

        if total_mass >= 1.45 and avg_mass_err <= 0.5:
            return "super_chandrasekhar"
    return "normal"


def main():
    # Initialize star groups
    type_colors = {
        "beaming": "red",
        "reflection": "blue",
        "ellipsoidal": "magenta",
        "hwvir": "green",
        "none": "black",
    }
    beaming_stars = StarGroup(
        name="beaming",
        color="red",
        label="Beaming"
    )
    reflection_stars = StarGroup(
        name="reflection",
        color="blue",
        label="Reflection"
    )
    ellipsoidal_stars = StarGroup(
        name="ellipsoidal",
        color="magenta",
        label="Ellipsoidal"
    )
    hwvir_stars = StarGroup(
        name="hwvir",
        color="green",
        label="HW Vir"
    )
    none_stars = StarGroup(
        name="none",
        color="black",
        label="None"
    )
    super_chandrasekhar = StarGroup(
        name="super_chandrasekhar",
        color="darkorange",
        label="J065816+094343"
    )

    # Load and classify stars
    stars = load_solved_stars(err_limit=np.inf)
    solved_table = pd.read_csv("solved_orbit_tracker.txt")
    for star in stars:
        if star.gaia_id == 2551900379931546240:
            star.teff = 26270
            star.teff_err = 1000
            star.logg = 5.443
            star.logg_err = 0.1
        if star.teff_err < 1000:
            star.teff_err = 1000
        if star.logg_err < 0.1:
            star.logg_err = 0.1
        if classify_star(star) == "super_chandrasekhar":
            super_chandrasekhar.add_star(star)
        else:
            row = solved_table.loc[solved_table["gaia_id"] == star.gaia_id]
            if row["phot_type"].iloc[0] == "beaming":
                beaming_stars.add_star(star)
            elif row["phot_type"].iloc[0] == "reflection":
                reflection_stars.add_star(star)
            elif row["phot_type"].iloc[0] == "ellipsoidal":
                ellipsoidal_stars.add_star(star)
            elif row["phot_type"].iloc[0] == "none":
                none_stars.add_star(star)
            elif row["phot_type"].iloc[0] == "hwvir":
                hwvir_stars.add_star(star)

    # Create and plot diagram
    diagram = TeffLoggDiagram()
    diagram.add_group(beaming_stars)
    diagram.add_group(reflection_stars)
    diagram.add_group(ellipsoidal_stars)
    diagram.add_group(none_stars)
    diagram.add_group(hwvir_stars)
    diagram.add_group(super_chandrasekhar)
    diagram.plot(save_path="other_plots/teff_logg_plot.pdf")


if __name__ == '__main__':
    main()