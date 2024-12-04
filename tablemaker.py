import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import G
import sys
from math import log10, floor

from common_functions import load_star, load_solved_stars

from warnings import filterwarnings

filterwarnings('ignore')


def round_to_significant_digits(value, err):
    if value is None:
        return np.nan
    if err is None:
        if value is None:
            return np.nan
        else:
            return round(value, 2)
    if err == 0:
        return value

    try:
        # Determine the order of magnitude of the error
        err_order = np.floor(np.log10(abs(err)))

        # Calculate the factor to round the value to two significant digits of the error
        factor = 10 ** (err_order - 1)

        # Round the value to the nearest multiple of this factor
        rounded_value = round(value / factor) * factor
        # Format the result to avoid floating-point precision issues
        # Use the number of decimals corresponding to the factor's order of magnitude
        # Determine the number of decimal places for formatting
        decimal_places = int(max(0, -err_order + 1))

        # Format the result to avoid floating-point precision issues
        formatted_value = f"{rounded_value:.{decimal_places}f}"

        # Convert to float if the result has decimal places, else return as int
        return formatted_value if '.' in formatted_value else int(formatted_value)

    except ValueError:
        return value


if __name__ == '__main__':
    orbit_tracking_table = pd.read_csv("solved_orbit_tracker.txt", sep=",")

    stars = load_solved_stars()

    for s in stars:
        if s.m_1 is not None and s.m_2 is not None:
            if s.m_1 + s.m_2 > 1.4:
                print(s.gaia_id)

    stars.sort(key=lambda x: x.m_2 if x.m_2 is not None else x.amplitude - 9999 if x.amplitude is not None else -100000000, reverse=True)

    # print(r"""\begin{tabularx}{\linewidth}{Xrrrrrrrr}
    # \toprule
    # Target ID & $K\ [\mathrm{kms}^{-1}]$ & $P\ [d]$ & $\gamma\ [\mathrm{kms}^{-1}]$ & $M_{1,\mathrm{SED}}\ [M_\astrosun]$ & $M_{2, \min}\ [M_\astrosun]$ & Lightcurve Type & Companion Type & $t_{\mathrm{merger}}\ [\mathrm{Gyr}]$ \\
    # \midrule""")
    # for i, star in enumerate(stars):
    #     row = orbit_tracking_table.loc[orbit_tracking_table.gaia_id == star.gaia_id]
    #     if star.amplitude is None or i >= np.inf:
    #         print("skipped!")
    #         continue
    #     # period_err_str = ((str(round_to_significant_digits(star.period_err, star.period_err)).replace('e-', r'\cdot 10^{-')+'}')
    #     #                   .replace('01', '1')
    #     #                   .replace('02', '2')
    #     #                   .replace('03', '3')
    #     #                   .replace('04', '4')
    #     #                   .replace('05', '5')
    #     #                   .replace('06', '6')
    #     #                   .replace('07', '7')
    #     #                   .replace('08', '8')
    #     #                   .replace('09', '9'))
    #     # print(star.gaia_id)
    #     print(fr"{star.name}"
    #           fr" & ${round_to_significant_digits(star.amplitude, 0.1):.1f} \pm {round_to_significant_digits(star.amplitude_err, 0.1):.1f}$"
    #           fr" & ${round_to_significant_digits(star.period, (star.period_err_lo + star.period_err_hi) / 2)}^{{+{round_to_significant_digits(star.period_err_hi, star.period_err_hi):.15f}}}_{{-{round_to_significant_digits(star.period_err_lo, star.period_err_lo):.15f}}}$"
    #           fr" & ${round_to_significant_digits(star.offset, 0.1):.1f} \pm {round_to_significant_digits(star.offset_err, 0.1):.1f}$"
    #           fr" & ${round_to_significant_digits(star.m_1, 0.001):.3f}^{{+{round_to_significant_digits(star.m_1_err_hi, 0.001):.3f}}}_{{-{round_to_significant_digits(star.m_1_err_lo, star.m_1_err_lo):.3f}}}$"
    #           fr" & ${round_to_significant_digits(star.m_2, 0.001):.3f}^{{+{round_to_significant_digits(star.m_2_err_hi, 0.001):.3f}}}_{{-{round_to_significant_digits(star.m_2_err_lo, star.m_2_err_lo):.3f}}}$"
    #           fr" & {row['phot_type'].iloc[0]}"
    #           fr" & {'dM/BD' if row['phot_type'].iloc[0] in ['reflection', 'hwvir'] else 'WD'}"
    #           fr" & ${round_to_significant_digits(star.t_merger, 1e8) / 1e9 if not star.t_merger / 1e9 > 13.5 else ">13.5" if not pd.isna(star.t_merger) else 0}$ \\")
    #
    # print(r"\end{tabularx}")

    stars.sort(key=lambda x: x.ra)
    print(r"""\begin{tabularx}{\linewidth}{Xrrrrrrrr}
    \toprule 
    Name & $N_{\mathrm{spec}}$ & of which SDSS & of which LAMOST & of which SOAR & TESS LC & GAIA LC & ATLAS LC & ZTF LC\\
    \midrule""")
    for i, star in enumerate(stars):
        row = orbit_tracking_table.loc[orbit_tracking_table.gaia_id == star.gaia_id]
        # period_err_str = ((str(round_to_significant_digits(star.period_err, star.period_err)).replace('e-', r'\cdot 10^{-')+'}')
        #                   .replace('01', '1')
        #                   .replace('02', '2')
        #                   .replace('03', '3')
        #                   .replace('04', '4')
        #                   .replace('05', '5')
        #                   .replace('06', '6')
        #                   .replace('07', '7')
        #                   .replace('08', '8')
        #                   .replace('09', '9'))
        # print(star.gaia_id)

        check = r'{\unicodefont ✓}'
        cross = r'{\unicodefont ✗}'
        print(fr"{star.name}"
              fr" & {len(star.datapoints)}"
              fr" & {np.sum(star.associations == 1) if np.sum(star.associations == 1) != 0 else '-'}"
              fr" & {np.sum(star.associations == 0) if np.sum(star.associations == 0) != 0 else '-'}"
              fr" & {np.sum(star.associations == 2) if np.sum(star.associations == 2) != 0 else '-'}"
              fr" & {check if 'TESS' in star.lightcurves.keys() else cross}"
              fr" & {check if 'GAIA' in star.lightcurves.keys() else cross}"
              fr" & {check if 'ATLAS' in star.lightcurves.keys() else cross}"
              fr" & {check if 'ZTF' in star.lightcurves.keys() else cross} \\"
              )
    print(r"\end{tabularx}")

    print(r"""\begin{tabularx}{\linewidth}{Xrrrrrrrr}
        \toprule 
        Name & $T_{\mathrm{eff}}$ & $\log g$ & $\log y$ & Primary Spectral Type\\
        \midrule""")

    spec_types = ["sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdOB", "sdOB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", "sdB", ]
    is_lit = [False, True, True, False, False, False, False, False, False, False, True, False, True, False, False, True, True, True, True, True, False]

    for i, star in enumerate(stars):
        row = orbit_tracking_table.loc[orbit_tracking_table.gaia_id == star.gaia_id]
        # if star.teff_err < 1000:
        #     # print(star.teff_err)
        #     star.teff_err = np.sqrt(1000**2+star.teff_err**2)
        #     # print(star.teff_err)
        # if star.logg_err < 0.1:
        #     star.logg_err = np.sqrt(0.1**2+star.logg_err**2)
        # if star.logy_err < 0.1:
        #     star.logy_err = np.sqrt(0.1**2+star.logy_err**2)

        print(fr"{star.name}{r'$^1$' if is_lit[i] else ''}"
              fr" & ${round_to_significant_digits(star.teff, star.teff_err)} \pm {round_to_significant_digits(star.teff_err, 10)}$"
              fr" & ${round_to_significant_digits(star.logg, star.logg_err)} \pm {round_to_significant_digits(star.logg_err, 0.01)}$"
              fr" & ${round_to_significant_digits(star.logy, star.logy_err)} \pm {round_to_significant_digits(star.logy_err, 0.01)}$"
              fr" & {spec_types[i]}\\"
              )
    print(r"\end{tabularx}")
