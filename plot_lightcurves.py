#!/usr/bin/env python3
"""
Plot lightcurves from ASCII files in a specified folder with dynamic filter-based colors.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define colors for specific filters (astronomical convention)
FILTER_COLORS = {
    # Standard astronomical filters
    'u': '#7F00FF',  # ultraviolet -> violet
    'g': '#00FF00',  # green band -> green
    'r': '#FF0000',  # red band -> red
    'i': '#8B0000',  # infrared -> dark red
    'z': '#4B0082',  # near-infrared -> indigo
    'y': '#FFD700',  # y-band -> gold
    'b': '#0000FF',  # blue -> blue
    'v': '#90EE90',  # visual -> light green
    'B': '#0000FF',  # Johnson B -> blue
    'V': '#90EE90',  # Johnson V -> light green
    'R': '#FF0000',  # Johnson R -> red
    'I': '#8B0000',  # Johnson I -> dark red
    
    # Survey-specific filters
    'c': '#00CED1',  # cyan (ATLAS cyan)
    'o': '#FF8C00',  # orange (ATLAS orange)
    'w': '#808080',  # white/clear -> gray
    'G': '#32CD32',  # Gaia G -> lime green
    'BP': '#4169E1', # Gaia BP -> royal blue
    'RP': '#DC143C', # Gaia RP -> crimson
    
    # ZTF filters (compound names)
    'zg': '#00FF00', # ZTF g -> green
    'zr': '#FF0000', # ZTF r -> red
    'zi': '#8B0000', # ZTF i -> dark red
    
    # Pan-STARRS filters
    'pg': '#00FF00',
    'pr': '#FF0000',
    'pi': '#8B0000',
    'pz': '#4B0082',
    'py': '#FFD700',
}

def get_filter_color(filter_name):
    """
    Get color for a filter, with fallback for unknown filters.
    """
    # Direct match
    if filter_name in FILTER_COLORS:
        return FILTER_COLORS[filter_name]
    
    # Try to extract the actual filter letter from compound names
    # e.g., 'zg' -> 'g', 'atlas_c' -> 'c'
    filter_lower = filter_name.lower()
    
    # Check if last character is a known filter
    if len(filter_lower) > 0:
        last_char = filter_lower[-1]
        if last_char in FILTER_COLORS:
            return FILTER_COLORS[last_char]
    
    # If still no match, generate a color from a colormap
    # Use hash to get consistent color for same filter name
    hash_val = hash(filter_name) % 256
    return cm.tab20(hash_val / 256)

def read_lightcurve(filepath):
    """
    Read lightcurve data from ASCII file.
    
    Returns:
        tuple: (time, flux, flux_error, filters) arrays or None if file is empty/invalid
    """
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return None
        
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle both comma and space/tab separated values
                    if ',' in line:
                        parts = line.split(',')
                    else:
                        parts = line.split()
                    
                    if len(parts) >= 4:
                        try:
                            time = float(parts[0])
                            flux = float(parts[1])
                            flux_error = float(parts[2])
                            # Clean filter name
                            filt = parts[3].strip().strip('"').strip("'")
                            data.append([time, flux, flux_error, filt])
                        except ValueError:
                            continue
        
        if not data:
            return None
            
        data = np.array(data, dtype=object)
        times = data[:, 0].astype(float)
        fluxes = data[:, 1].astype(float)
        flux_errors = data[:, 2].astype(float)
        filters = data[:, 3]
        
        return times, fluxes, flux_errors, filters
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def plot_lightcurves(folder_path, surveys_to_plot):
    """
    Plot lightcurves from specified surveys.
    
    Args:
        folder_path: Path to folder containing lightcurve files
        surveys_to_plot: List of survey names to plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    plotted_any = False
    all_filters_seen = set()
    
    # Different marker for each survey
    survey_markers = {
        'atlas': 'o',
        'bg': 's',
        'gaia': '^',
        'tess': 'D',
        'ztf': 'v'
    }
    
    for survey in surveys_to_plot:
        filename = f"{survey}_lc.txt"
        filepath = os.path.join(folder_path, filename)
        
        print(f"\nProcessing {survey}...")
        
        data = read_lightcurve(filepath)
        
        if data is None:
            print(f"  - No data found in {filename}")
            continue
        
        times, fluxes, flux_errors, filters = data
        
        # Get unique filters for this survey
        unique_filters = np.unique(filters)
        
        for filt in unique_filters:
            mask = filters == filt
            t = times[mask]
            f = fluxes[mask]
            e = flux_errors[mask]
            
            # Get color for this filter
            color = get_filter_color(filt)
            
            # Get marker for this survey
            marker = survey_markers.get(survey, 'o')
            
            # Create label
            label = f"{survey}-{filt}"
            
            # Plot with error bars
            ax.errorbar(t, f, yerr=e, 
                       fmt=marker, 
                       color=color,
                       label=label,
                       markersize=5,
                       alpha=0.7,
                       capsize=2,
                       elinewidth=1,
                       markeredgecolor='black',
                       markeredgewidth=0.3)
            
            print(f"  - Plotted {len(t)} points from filter '{filt}' with color {color if isinstance(color, str) else 'auto'}")
            all_filters_seen.add(filt)
            plotted_any = True
    
    if plotted_any:
        ax.set_xlabel('Time (MJD)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.set_title(f'Lightcurves from {os.path.basename(folder_path)}', fontsize=14)
        
        # Create a sorted legend
        handles, labels = ax.get_legend_handles_labels()
        sorted_pairs = sorted(zip(handles, labels), key=lambda x: x[1])
        if sorted_pairs:
            handles, labels = zip(*sorted_pairs)
            ax.legend(handles, labels, loc='best', fontsize=9, ncol=2 if len(labels) > 10 else 1)
        
        ax.grid(True, alpha=0.3)
        
        # Add filter color reference in title area
        if all_filters_seen:
            filter_info = f"Filters: {', '.join(sorted(all_filters_seen))}"
            fig.text(0.5, 0.95, filter_info, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Optional: Invert y-axis if dealing with magnitudes
        # Uncomment the next line if your flux values are actually magnitudes
        # ax.invert_yaxis()
        
        plt.show()
    else:
        print("\nNo data to plot!")

def main():
    parser = argparse.ArgumentParser(
        description='Plot astronomical lightcurves from ASCII files',
        epilog='Example: python plot_lightcurves.py lightcurves/3377315754750660992 --ztf --atlas'
    )
    
    parser.add_argument('folder', 
                       help='Path to folder containing lightcurve files')
    
    parser.add_argument('--atlas', action='store_true',
                       help='Plot ATLAS lightcurve')
    parser.add_argument('--bg', action='store_true',
                       help='Plot BG lightcurve')
    parser.add_argument('--gaia', action='store_true',
                       help='Plot Gaia lightcurve')
    parser.add_argument('--tess', action='store_true',
                       help='Plot TESS lightcurve')
    parser.add_argument('--ztf', action='store_true',
                       help='Plot ZTF lightcurve')
    parser.add_argument('--all', action='store_true',
                       help='Plot all available lightcurves')
    parser.add_argument('--list-filters', action='store_true',
                       help='List all unique filters found in the data')
    
    args = parser.parse_args()
    
    # If --list-filters, just scan and report
    if args.list_filters:
        print("\nScanning for filters in all files...")
        all_filters = set()
        for survey in ['atlas', 'bg', 'gaia', 'tess', 'ztf']:
            filepath = os.path.join(args.folder, f"{survey}_lc.txt")
            data = read_lightcurve(filepath)
            if data:
                _, _, _, filters = data
                unique = np.unique(filters)
                for f in unique:
                    all_filters.add(f)
                    print(f"  {survey}: {', '.join(unique)}")
        print(f"\nAll unique filters: {', '.join(sorted(all_filters))}")
        return
    
    # Determine which surveys to plot
    surveys_to_plot = []
    
    if args.all:
        surveys_to_plot = ['atlas', 'bg', 'gaia', 'tess', 'ztf']
    else:
        if args.atlas:
            surveys_to_plot.append('atlas')
        if args.bg:
            surveys_to_plot.append('bg')
        if args.gaia:
            surveys_to_plot.append('gaia')
        if args.tess:
            surveys_to_plot.append('tess')
        if args.ztf:
            surveys_to_plot.append('ztf')
    
    # If no surveys specified, default to all
    if not surveys_to_plot:
        print("No surveys specified. Use --all to plot all, or specify individual surveys.")
        print("Available options: --atlas, --bg, --gaia, --tess, --ztf")
        print("\nExample usage:")
        print(f"  python {os.path.basename(__file__)} {args.folder} --all")
        print(f"  python {os.path.basename(__file__)} {args.folder} --ztf --atlas")
        return
    
    print(f"Plotting lightcurves from: {', '.join(surveys_to_plot)}")
    plot_lightcurves(args.folder, surveys_to_plot)

if __name__ == "__main__":
    main()