#!/usr/bin/env python3
import os
import csv
import argparse
import math
import numpy as np
from astropy.time import Time

def bin_lightcurve(data, nbins):
    """
    Bin the lightcurve data into nbins bins.
    The data is assumed to be sorted. Each bin’s value is the average of its points.
    Each data point is a tuple: (x, d_val, flux, flux_error).
    """
    n = len(data)
    if n == 0:
        return data

    # Determine bin sizes
    binned = []
    # Calculate base bin size and the remainder for uneven division
    base_bin_size = n // nbins
    remainder = n % nbins
    start = 0
    for i in range(nbins):
        # Distribute remainder one by one into the first few bins
        extra = 1 if i < remainder else 0
        end = start + base_bin_size + extra
        bin_data = data[start:end]
        if not bin_data:
            continue
        avg_x = np.mean([row[0] for row in bin_data])
        avg_d = np.sum([row[1] for row in bin_data])
        avg_flux = np.average([row[2] for row in bin_data], weights=[1/row[3] for row in bin_data])
        avg_flux_error = np.sqrt(np.sum([row[3]*row[3] for row in bin_data]))/len(bin_data)
        binned.append((avg_x, avg_d, avg_flux, avg_flux_error))
        start = end
    return binned

def convert_lightcurve(gaia_id, period=None, phase_offset=None, do_bin=False, nbins=None):
    # Construct file paths
    input_path = os.path.join("lightcurves", gaia_id, "tess_lc.txt")
    output_path = os.path.join("lightcurves", gaia_id, "tess_lc_space.txt")

    # Read the CSV file data; expects three columns: mjd, flux, flux_error.
    data = []
    with open(input_path, "r", newline="") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                mjd = float(row[0])
                flux = float(row[1])
                flux_error = float(row[2])
                data.append((mjd, flux, flux_error))
            except ValueError:
                continue

    if not data:
        print("No valid data found in the file.")
        return

    # Compute delta_mjd for each datapoint.
    delta_mjd = []
    for i in range(len(data)):
        if i < len(data) - 1:
            delta = data[i+1][0] - data[i][0]
        else:
            delta = delta_mjd[-1] if i > 0 else 0.0
        delta_mjd.append(delta)

    # Prepare the lightcurve data.
    # When phasing, compute phase and sort; otherwise use the mjd.
    processed_data = []
    if period is not None and phase_offset is not None:
        print(phase_offset * period, phase_offset)
        for (mjd, flux, flux_error), d_mjd in zip(data, delta_mjd):
            x = ((mjd + (phase_offset * period)) % period) / period
            d_val = d_mjd / period
            processed_data.append((x, d_val, flux, flux_error))
        processed_data.sort(key=lambda tup: tup[0])
    else:
        for (mjd, flux, flux_error), d_mjd in zip(data, delta_mjd):
            processed_data.append((mjd, d_mjd, flux, flux_error))

    # If binning is requested, bin the data.
    if do_bin:
        if nbins is None:
            # Use cube root of the number of data points, rounded up.
            nbins = math.ceil(len(processed_data) ** (1/3))
        processed_data = bin_lightcurve(processed_data, nbins)
        print(f"Binned data into {nbins} bins.")

    # Write the space-separated file with columns: x/mjd, delta, flux, flux_error, weight, factor.
    with open(output_path, "w", newline="") as outfile:
        for entry in processed_data:
            line = f"{entry[0]} {entry[1]} {entry[2]} {entry[3]} 1 1\n"
            outfile.write(line)

    print(f"Converted lightcurve saved to: {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert and optionally bin a lightcurve."
    )
    parser.add_argument("gaia_id", type=str, help="Gaia ID for the lightcurve directory.")
    parser.add_argument("--period", type=float, default=None, help="Period for phasing the lightcurve.")
    parser.add_argument("--phase_offset", type=float, default=None,
                        help="Phase offset for phasing (as a fraction of the period, between 0 and 1).")
    parser.add_argument("--bin", dest="do_bin", action="store_true",
                        help="Enable binning of the lightcurve data.")
    parser.add_argument("--nbins", type=int, default=None,
                        help="Number of bins to use if binning is enabled. If not provided, cube root of N is used.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Validate phasing arguments.
    if (args.period is None) ^ (args.phase_offset is None):
        print("Error: Both --period and --phase_offset must be provided for phasing.")
        exit(1)
    if args.period is not None and args.period <= 0:
        print("Error: period must be positive.")
        exit(1)
    if args.phase_offset is not None and not (0 <= args.phase_offset <= 1):
        print("Error: phase_offset must be between 0 and 1 (as a fraction of the period).")
        exit(1)

    convert_lightcurve(args.gaia_id, args.period, args.phase_offset, args.do_bin, args.nbins)
