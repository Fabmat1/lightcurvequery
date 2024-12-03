import shutil

from common_functions import *


if __name__ == "__main__":
    ma_dir = DOC_PATH + "/LaTeXProjects/Masterarbeit/Bilder/"
    print(ma_dir)
    stars = load_solved_stars(np.inf)

    stars = sorted(stars, key=lambda x: x.ra)
    for star in stars:
        try:
            shutil.copyfile(f"lcplots/{star.gaia_id}_lcplot.pdf", ma_dir+f"{star.gaia_id}_lcplot.pdf")
        except FileNotFoundError:
            pass
        try:
            shutil.copyfile(f"pgramplots/{star.gaia_id}_allpg_mashed.pdf", ma_dir+f"{star.gaia_id}_allpg_mashed.pdf")
        except FileNotFoundError:
            pass
        try:
            shutil.copyfile(f"pgramplots/{star.gaia_id}_allpgplot.pdf", ma_dir+f"{star.gaia_id}_allpgplot.pdf")
        except FileNotFoundError:
            pass
        try:
            shutil.copyfile(f"pgramplots/{star.gaia_id}_allpg_window.pdf", ma_dir+f"{star.gaia_id}_allpg_window.pdf")
        except FileNotFoundError:
            pass
        try:
            shutil.copyfile(f"/home/fabian/RVVD/SEDs/{star.gaia_id}/photometry_SED.pdf", ma_dir+f"{star.gaia_id}_photometry_SED.pdf")
        except FileNotFoundError:
            pass

        print(rf"""\subsection{{{star.name}}}
\begin{{figure}}[H]
	\centering
	\label{{fig:{star.gaia_id}lcplot_appendix}}
	\includegraphics[width=\linewidth]{{Bilder/{star.gaia_id}_lcplot}}
	\caption{{Phasefolded lightcurves and RV curves for {star.name}.}}
\end{{figure}}
\begin{{figure}}[H]
	\centering
	\label{{fig:{star.gaia_id}allpgplot}}
	\includegraphics[width=\linewidth]{{Bilder/{star.gaia_id}_allpgplot}}
	\caption{{Periodograms constructed from TESS and RV data available for {star.name}.}}
\end{{figure}}
""")