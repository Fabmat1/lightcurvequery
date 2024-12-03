import shutil
import subprocess
import traceback
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from common_functions import *

GAIA_ID = 4190630033436379776


def candidate_plot(star):
    # Load data
    mcmc_amp = np.loadtxt(BASE_PATH + f'/CLionProjects/subdwarf_rv_simulation{"s" if os.name == "nt" else "" if not is_wsl() else "s"}/out/pamp_full{str(star.gaia_id)[:5]}.csv', delimiter=",")
    pgram = np.loadtxt(BASE_PATH + f'/CLionProjects/subdwarf_rv_simulation{"s" if os.name == "nt" else "" if not is_wsl() else "s"}/out/{str(star.gaia_id)[:5]}.csv', delimiter=",")
    pgram_x = pgram[:, 0]
    pgram_y = pgram[:, 1]

    # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8.27/2, 11.69/3), dpi=100, gridspec_kw={'height_ratios': [2, 1]})

    plt.figure(figsize=(8.27 / 2, 11.69 / 3), dpi=100)

    # Upper plot: imshow for mcmc_amp
    # To ensure proper scaling, we map the extent of the x-axis (log-scaled) to the periodogram range
    extent = [np.log10(pgram_x.min()), np.log10(pgram_x.max()), 0, 500]
    norm = plt.Normalize(vmin=0, vmax=np.percentile(mcmc_amp, 99.9))
    # axs[0].imshow(mcmc_amp, aspect='auto', cmap='afmhot', extent=extent, origin='lower', norm=norm)  # Flip y-axis with origin='lower'
    plt.imshow(mcmc_amp, aspect='auto', cmap='Greys', extent=extent, origin='lower', norm=norm)  # Flip y-axis with origin='lower'
    # axs[0].set_ylabel("Amplitude [km/s]")
    plt.ylabel("Amplitude [km/s]")

    # Remove x-axis ticks from the upper plot
    #
    # axs[0].set_xticks([])

    # Lower plot: periodogram
    # axs[1].plot(pgram_x, pgram_y, color='blue')
    # axs[1].set_ylabel("Power [no unit]")
    # axs[1].set_xlabel("Period [d]")
    plt.xlabel("Period [d]")
    # axs[1].set_xlim(pgram_x.min(), pgram_x.max())

    # axs[1].set_xscale('log')
    # plt.xscale('log')

    plt.xticks([-1, 0, 1], ["$10^{-1}$", "$10^{0}$", "$10^{1}$"])
    plt.xlim(np.log10(pgram_x.min()), np.log10(pgram_x.max()))
    # plt.xlim(np.log10(pgram_x.min()), 0)
    # plt.ylim(0, 125)
    #
    plt.subplots_adjust(hspace=0)  # Set hspace to zero

    # Show the plot
    plt.tight_layout()
    # plt.savefig("other_plots/example_mcmc.pdf")
    plt.show()

    # Create amp_x and amp_y
    amp_x = 10 ** np.flip(np.linspace(np.log10(pgram_x[0]), np.log10(pgram_x[-1]), mcmc_amp.shape[1]))
    amp_y = np.linspace(0, 500, mcmc_amp.shape[0])

    print(star.m_1)
    index = np.unravel_index(np.argmax(mcmc_amp, axis=None), mcmc_amp.shape)

    print(amp_x[index[1]], amp_y[index[0]])

    # Create meshgrid to compute all (x, y) combinations
    amp_x_grid, amp_y_grid = np.meshgrid(amp_x, amp_y)  #, indexing='ij')

    # Vectorize the m2 function over the grid
    m2_matrix = m2(amp_y_grid, amp_x_grid, 90, star.m_1)

    # Flatten the arrays
    m2_flat = m2_matrix.flatten()
    weights_flat = mcmc_amp.flatten()

    weights_flat = weights_flat[~np.isnan(m2_flat)]
    m2_flat = m2_flat[~np.isnan(m2_flat)]

    # Normalize the weights after filtering
    weights_prob = weights_flat / np.sum(weights_flat)

    # Calculate the weighted percentage above the threshold
    threshold = 1.45 - star.m_1
    super_chandra_prob = 100 * np.sum(weights_prob[m2_flat > threshold])

    print(f"Super-Chandra probability: {super_chandra_prob}%")

    max_perc = np.percentile(m2_flat, 75.0)
    weights_flat = weights_flat[m2_flat < max_perc]
    m2_flat = m2_flat[m2_flat < max_perc]

    # Create the weighted histogram
    hist, bin_edges = np.histogram(m2_flat, bins=500, weights=weights_flat, density=True)

    # Plot the histogram
    # Plot the histogram using bar
    plt.figure(figsize=(8.27 / 2, 11.69 / 3), dpi=100)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Calculate bin centers for plotting
    plt.bar(bin_centers, hist, width=np.diff(bin_edges), color='darkred', edgecolor='black', align='center')
    plt.axvline(1.45 - star.m_1, color='red', linestyle='dashed')
    plt.xlim(0, 1.5)
    plt.xlabel(r'$M_{2, \min}$')
    plt.ylabel('Weighted count')
    # plt.title('Weighted Histogram of m2 values')
    plt.tight_layout()
    # plt.savefig("other_plots/example_m2dist.pdf")
    plt.show()


canonical_stars = [2562664186451182464, 1572205081151512192, 620091435278404608]
exclude_stars= [2500824388329728256]

def cand_probability(star):
    global canonical_stars
    try:
        mcmc_amp = np.loadtxt(BASE_PATH + f'/CLionProjects/subdwarf_rv_simulation{"s" if os.name == "nt" else "" if not is_wsl() else "s"}/out/pamp_full{str(star.gaia_id)[:5]}.csv', delimiter=",")
        pgram = np.loadtxt(BASE_PATH + f'/CLionProjects/subdwarf_rv_simulation{"s" if os.name == "nt" else "" if not is_wsl() else "s"}/out/{str(star.gaia_id)[:5]}.csv', delimiter=",")
    except FileNotFoundError:
        print(star.gaia_id, " - File Not found!")
        return {"gaia_id": str(star.gaia_id),
                "K_lo": None,
                "K_hi": None,
                "P_lo": None,
                "P_hi": None,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}

    if len(star.times) <= 4:
        return {"gaia_id": str(star.gaia_id),
                "K_lo": None,
                "K_hi": None,
                "P_lo": None,
                "P_hi": None,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}

    if star.gaia_id in canonical_stars:
        star.m_1 = 0.47

    if star.gaia_id in exclude_stars:
        return {"gaia_id": str(star.gaia_id),
                "K_lo": None,
                "K_hi": None,
                "P_lo": None,
                "P_hi": None,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}


    if star.m_1 is None and not os.path.isfile(RVVD_PATH+f"/SEDs/{star.gaia_id}/photometry.dat"):
        print(f"Missing SED: {star.gaia_id}")
        if star.teff is None:
            previous_dir = os.getcwd()

            try:
                # Change to the ~/spektralfits directory
                os.chdir(os.path.expanduser("~/spektralfits"))
                print(f"{star.gaia_id}: Doing model fit!")

                # Run the ./auto_isis.sh {ID} command
                subprocess.run(["./auto_isis.sh", str(star.gaia_id)])

            finally:
                # Change back to the previous working directory
                os.chdir(previous_dir)

        print("Created model!")
        star = load_star(star.gaia_id)

        if star.teff is None or star.logg is None or star.logy is None:
            print(f"{star.gaia_id}: no valid model!")
            return {"gaia_id": str(star.gaia_id),
                "K_lo": None,
                "K_hi": None,
                "P_lo": None,
                "P_hi": None,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}

        previous_dir = os.getcwd()

        try:
            # Change to the ~/RVVD/SEDs directory
            base_dir = os.path.expanduser("~/RVVD/SEDs")
            os.chdir(base_dir)

            # Create the directory named after the Gaia ID
            gaia_dir = os.path.join(base_dir, f"{star.gaia_id}")
            print(f"Gaia dir: {gaia_dir}")
            if not os.path.isfile(os.path.join(gaia_dir, f"photometry.dat")):
                try:
                    os.mkdir(gaia_dir)
                except FileExistsError:
                    pass
                print(f"Created {gaia_dir}")

                # Copy photometry.sl from the SEDs directory to the Gaia ID directory
                shutil.copy("sedscriptbase.sl", os.path.join(gaia_dir, "photometry.sl"))

                # Define the content to prepend
                to_prepend = f"""variable star = "GAIA DR3 {star.gaia_id}";
variable par = struct{{name = ["c*_xi","c*_z","c1_teff","c1_logg","c1_HE"],
    value = [0,0,{round(star.teff, 1)},{round(star.logg, 4)},{round(star.logy, 2)}],
    freeze = [1,1,1,1,1]}};
variable griddirectories, bpaths;
griddirectories = ["sdB/processed"];
bpaths = ["/home/fabian/ISIS_models/"];
                """

                # Full path to the copied photometry.sl file
                photometry_file_path = os.path.join(gaia_dir, "photometry.sl")

                # Read the original contents of photometry.sl
                with open(photometry_file_path, "r") as f:
                    original_content = f.read()

                # Write the new content (to_prepend + original content) to the file
                with open(photometry_file_path, "w") as f:
                    f.write(to_prepend + original_content)

                print("Created script!")
                os.chdir(gaia_dir)
                # Run the ./auto_isis.sh {ID} command
                subprocess.run(["isis", "photometry.sl"])
        except Exception as e:
            print("Exception occurred:", e)
            traceback.print_exc()  # This prints the full traceback to stderr
        finally:
            # Change back to the previous working directory
            star = load_star(star.gaia_id)
            os.chdir(previous_dir)

        print("Star M_1:", star.m_1)
    if star.m_1 is None:
        print(f"{star.gaia_id}: no valid M1!")
        return {"gaia_id": str(star.gaia_id),
                "K_lo": None,
                "K_hi": None,
                "P_lo": None,
                "P_hi": None,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}
    elif star.m_1_err_lo is None or star.m_1_err_hi is None and star.gaia_id not in canonical_stars:
        print(f"{star.gaia_id}: no valid M1_error!")
        return {"gaia_id": str(star.gaia_id),
                "K_lo": None,
                "K_hi": None,
                "P_lo": None,
                "P_hi": None,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}
    elif (star.m_1_err_lo+star.m_1_err_hi)/2 > 0.5 and star.gaia_id not in canonical_stars:
        print(f"{star.gaia_id}: M1 error too high!")
        return {"gaia_id": str(star.gaia_id),
                "K_lo": None,
                "K_hi": None,
                "P_lo": None,
                "P_hi": None,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}

    pgram_x = pgram[:, 0]

    amp_x = 10 ** np.flip(np.linspace(np.log10(pgram_x[0]), np.log10(pgram_x[-1]), mcmc_amp.shape[1]))
    amp_y = np.linspace(0, 500, mcmc_amp.shape[0])

    mcmc_amp_xsum = np.sum(mcmc_amp, axis=0)
    mcmc_amp_ysum = np.sum(mcmc_amp, axis=1)

    cdf_x = np.cumsum(mcmc_amp_xsum) / np.sum(mcmc_amp_xsum)
    cdf_y = np.cumsum(mcmc_amp_ysum) / np.sum(mcmc_amp_ysum)

    # Find the percentiles for the 1-sigma bounds (15.87% and 84.13%)
    # percentile_1sigma = [0.15865525393, 1 - 0.15865525393]
    # percentile_2sigma = [0.02275013195, 1 - 0.02275013195]
    percentile_3sigma = [0.00134989803, 1 - 0.00134989803]

    # Interpolate the values of amp_x and amp_y at these percentiles
    plo, phi = np.interp(percentile_3sigma, cdf_x, amp_x)
    klo, khi = np.interp(percentile_3sigma, cdf_y, amp_y)

    # Create meshgrid to compute all (x, y) combinations
    amp_x_grid, amp_y_grid = np.meshgrid(amp_x, amp_y)  # , indexing='ij')

    # Vectorize the m2 function over the grid
    m2_matrix = m2(amp_y_grid, amp_x_grid, 90, star.m_1)

    # Flatten the arrays
    m2_flat = m2_matrix.flatten()
    weights_flat = mcmc_amp.flatten()

    weights_flat = weights_flat[~np.isnan(m2_flat)]
    m2_flat = m2_flat[~np.isnan(m2_flat)]

    # Normalize the weights after filtering
    weights_prob = weights_flat / np.sum(weights_flat)

    # Calculate the weighted percentage above the threshold
    threshold = 1.45 - star.m_1
    super_chandra_prob = 100 * np.sum(weights_prob[m2_flat > threshold])

    sorted_indices = np.argsort(m2_flat)
    m2_flat_sorted = m2_flat[sorted_indices]
    weights_flat_sorted = weights_flat[sorted_indices]

    # Compute the weighted CDF
    cdf = np.cumsum(weights_flat_sorted) / np.sum(weights_flat_sorted)

    # Find the percentiles for the 1-sigma bounds (15.87% and 84.13%)
    percentile_1sigma = [0.1587, 0.8413]

    # Interpolate the values of m2_flat at these percentiles
    try:
        m2_lo, m2_hi = np.interp(percentile_1sigma, cdf, m2_flat_sorted)
    except ValueError:
        print(f"{star.gaia_id}: Value Error!")
        return {"gaia_id": str(star.gaia_id),
                "K_lo": klo,
                "K_hi": khi,
                "P_lo": plo,
                "P_hi": phi,
                "m2_lo": None,
                "m2_hi": None,
                "chandr_prob": None}

    if star.gaia_id == 1223920926079402112:
        print(star.gaia_id)
        print(m2_lo, m2_hi)
        print(klo, khi)
    return {
        "gaia_id": str(star.gaia_id),
        "K_lo": klo,
        "K_hi": khi,
        "P_lo": plo,
        "P_hi": phi,
        "m2_lo": m2_lo,
        "m2_hi": m2_hi,
        "chandr_prob": round(super_chandra_prob / 100, 4)
    }


if __name__ == "__main__":

    candidate_plot(load_star(GAIA_ID))
    # Load your stars here
    stars = load_unsolved_stars()
    # stars = [load_star(4440145853560129408)]
    # input()
    with Pool(processes=16) as pool:
        results = list(pool.map(cand_probability, stars))

    print("Process Done!")
    # Convert the list of dictionaries to a DataFrame
    outtable = pd.DataFrame(results)

    # Sort and save the DataFrame
    outtable.sort_values(by="chandr_prob", ascending=False, inplace=True)
    outtable.to_csv("binary_candidates.csv", index=False)
