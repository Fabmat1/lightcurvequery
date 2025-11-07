"""
Command-line interface with advanced title and alias handling.
"""

from __future__ import annotations
import warnings
import csv

warnings.filterwarnings(
    'ignore',
    message='Warning: the tpfmodel submodule is not available without oktopus installed',
    category=UserWarning,
    module='lightkurve.prf'
)

import sys, argparse
from typing import List, Tuple
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from .process import process_lightcurves
from .utils import bcolors
from .terminal_style import *
from .update_checker import *
from .plotconfig import PlotConfig, STYLE_PRESETS
from .title_manager import TitleTemplate

try:
    from astroquery.gaia import Gaia
except ModuleNotFoundError:
    Gaia = None


# ────────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process lightcurves for astronomical objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ALIAS & TITLE CUSTOMIZATION:
  Store target aliases in a file with format: gaia_id [alias]
    1234567890 ProximaCen
    9876543210 TRAPPIST-1
  
  For custom Plot titles use: {gaia_id}, {alias}, {display_name}
    - gaia_id: Gaia ID for the star
    - alias: alias name for the star (blank if N/A)
    - display_name: alias if available, otherwise "Gaia DR3 XXXXXXXXXXXXXXXXXXX"
    Examples:
      --title-template "LC for {display_name}"
      --title-template "Gaia DR3 {gaia_id} ({alias})"
      --title-template "{alias} Lightcurve"

EXAMPLE USAGE:
    # Fetch everything that is publicly available for Gaia DR3 XXXXXXXXXXXXXXXXXXX
    lightcurvequery XXXXXXXXXXXXXXXXXXX

    # Same, but provide coordinates (deg) instead of an ID
    lightcurvequery --coords 269.4521 -24.8801

    # Use a text file that contains one Gaia ID per line 
    # (and optional whitespace-separated alias)
    lightcurvequery --file targets.txt

    # Only TESS + Gaia, search periods between 0.1 and 5 d, no plots
    lightcurvequery XXXXXXXXXXXXXXXXXXX --skip-ztf --skip-atlas --skip-bg \
                            --min-p 0.1 --max-p 5 --no-plot
        """,
    )

    parser.add_argument('targets', nargs='*')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--coords', nargs=2, metavar=('RA', 'DEC'), type=float)
    g.add_argument('--file', '-i', type=str)

    # Survey flags
    parser.add_argument('--skip-tess', '-t', action='store_true')
    parser.add_argument('--skip-ztf', '-z', action='store_true')
    parser.add_argument('--skip-atlas', '-a', action='store_true')
    parser.add_argument('--skip-gaia', '-g', action='store_true')
    parser.add_argument('--skip-bg', '-b', action='store_true')

    # Processing options
    parser.add_argument('--no-binning', '-B', action='store_true')
    parser.add_argument('--no-whitening', '-W', action='store_true')
    parser.add_argument('--no-plot', '-P', action='store_true')
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--show-plots', type=str2bool, default=True)

    # Period options
    parser.add_argument('--min-p', '-m', type=float, default=0.05)
    parser.add_argument('--max-p', '-M', type=float, default=50.0)
    parser.add_argument('--force-nsamp', '-n', type=int)
    parser.add_argument('--force-period', '-f', type=float)

    parser.add_argument('--trim-tess', type=float, default=0.00,
                        help='Percent of data to trim from beginning/end of TESS sectors [0.0,0.5]')

    # ZTF-specific options
    parser.add_argument('--ztf-inner-radius', type=float, default=5.0, metavar='ARCSEC',
                        help='Inner radius (arcsec) kept around closest ZTF source')
    parser.add_argument('--ztf-outer-radius', type=float, default=20.0, metavar='ARCSEC',
                        help='Outer radius (arcsec) for initial ZTF query')
    parser.add_argument('--plot-ztf-preview', action='store_true',
                        help='Enable preview plot of ZTF sources')

    parser.add_argument('--include-h', action='store_true',
                        help='Show H-band photometry (ignored by default)')
    parser.add_argument('--include-zi', action='store_true',
                        help='Show zi-band photometry (ignored by default)')

    # NEW: Plot configuration
    parser.add_argument('--plot-style', type=str, default='default',
                        choices=list(STYLE_PRESETS.keys()) + ['default'],
                        help='Use a preset plot style')
    parser.add_argument('--plot-config', type=str, default=None,
                        help='Path to custom YAML/JSON plot config file')

    # NEW: Alias and title customization
    parser.add_argument('--alias', type=str, default=None,
                        help='Alias name for single target (used in plot titles)')
    parser.add_argument('--title-template', type=str, default=None,
                        help='Template for plot titles. Variables: {gaia_id}, {alias}, {display_name}')
    parser.add_argument('--title-preset', type=str, default='default',
                        choices=['default', 'paper', 'minimal'],
                        help='Preset title configuration')

    return parser


# ────────────────────────────────────────────────────────────────────
def validate_gaia_id(gid: str):
    try:
        int(gid)
        if len(gid) < 10:
            raise ValueError
    except ValueError:
        print_error(f"Invalid Gaia ID: {gid}")
        sys.exit(1)


def load_gaia_ids_from_file(path) -> List[Tuple[str, Optional[str]]]:
    """Load Gaia IDs with optional aliases from file.
    
    Supports formats:
    - Simple: one gaia_id per line
    - With aliases: gaia_id alias (space-separated)
    - YAML: targets: [{gaia_id: ..., alias: ...}, ...]
    - CSV: gaia_id,alias
    """
    ids: list[Tuple[str, Optional[str]]] = []
    
    try:
        with open(path) as f:
            # Try YAML format first
            if path.endswith('.yaml') or path.endswith('.yml'):
                import yaml
                data = yaml.safe_load(f)
                if isinstance(data, dict) and 'targets' in data:
                    for target in data['targets']:
                        gid = target.get('gaia_id')
                        alias = target.get('alias')
                        validate_gaia_id(str(gid))
                        ids.append((str(gid), alias))
                    return ids
            
            # CSV format
            if path.endswith('.csv'):
                f.seek(0)
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith('#'):
                        continue
                    gid = row[0].strip()
                    alias = row[1].strip() if len(row) > 1 else None
                    validate_gaia_id(gid)
                    ids.append((gid, alias))
                return ids
            
            # Plain text format: one per line, optionally with alias
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(None, 1)  # Split on whitespace, max 2 parts
                gid = parts[0]
                alias = parts[1] if len(parts) > 1 else None
                
                validate_gaia_id(gid)
                ids.append((gid, alias))
        
        return ids
    
    except FileNotFoundError:
        print_error(f"File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error loading targets from {path}: {e}")
        sys.exit(1)


def query_gaia_by_coordinates(coord: SkyCoord) -> str:
    if Gaia is None:
        print_error("astroquery.gaia not installed – cannot resolve coordinates.")
        sys.exit(1)

    job = Gaia.cone_search_async(coord, radius=5*u.arcsec)
    res = job.get_results()
    if len(res) == 0:
        print_error("No Gaia source within 5\".")
        sys.exit(1)
    return str(res[0]["SOURCE_ID"])


def resolve_targets(args) -> List[Tuple[str, Optional[SkyCoord], Optional[str]]]:
    """Resolve targets to (gaia_id, coord, alias) tuples."""
    targets: list[Tuple[str, Optional[SkyCoord], Optional[str]]] = []

    # --file
    if args.file:
        for gid, alias in load_gaia_ids_from_file(args.file):
            targets.append((gid, None, alias))
        return targets

    # --coords
    if args.coords:
        ra, dec = map(float, args.coords)
        coord = SkyCoord(ra*u.deg, dec*u.deg)
        gid = query_gaia_by_coordinates(coord)
        targets.append((gid, coord, args.alias))
        return targets

    # Positional
    if not args.targets:
        print_error("No targets provided.")
        sys.exit(1)

    if len(args.targets) == 2:
        try:
            ra, dec = map(float, args.targets)
            coord = SkyCoord(ra*u.deg, dec*u.deg)
            gid = query_gaia_by_coordinates(coord)
            return [(gid, coord, args.alias)]
        except ValueError:
            pass

    for tok in args.targets:
        if tok.endswith('.txt') or tok.endswith('.yaml') or tok.endswith('.yml') or tok.endswith('.csv'):
            for gid, alias in load_gaia_ids_from_file(tok):
                targets.append((gid, None, alias))
        else:
            validate_gaia_id(tok)
            targets.append((tok, None, args.alias))
    
    return targets


def load_plot_config(args) -> PlotConfig:
    """Load plot configuration from args."""
    config = PlotConfig()
    
    # If custom config file is provided, use it
    if args.plot_config:
        try:
            config = PlotConfig.from_file(args.plot_config)
            print_success(f"Loaded plot config from {args.plot_config}")
        except FileNotFoundError:
            print_error(f"Plot config file not found: {args.plot_config}")
            sys.exit(1)
        except Exception as e:
            print_error(f"Error loading plot config: {e}")
            sys.exit(1)
    # If preset is specified, use it
    elif args.plot_style != 'default':
        config = STYLE_PRESETS[args.plot_style]
        print_success(f"Using '{args.plot_style}' plot style preset")
    else:
        print_success("Using default plot configuration")
    
    return config


def load_title_template(args) -> TitleTemplate:
    """Create title template from arguments."""
    # If explicit template provided, use it
    if args.title_template:
        return TitleTemplate(
            photometry_template=args.title_template,
            periodogram_template=args.title_template,
            rv_template=args.title_template,
            show_titles=True,
        )
    
    # Otherwise use preset
    title_template = TitleTemplate.from_preset(args.title_preset)
    print_success(f"Using '{args.title_preset}' title preset")
    return title_template


# ────────────────────────────────────────────────────────────────────
def main():
    check_for_update(current_version="0.2.0", repo="Fabmat1/lightcurvequery")

    parser = parse_arguments()
    args = parser.parse_args()

    targets = resolve_targets(args)
    enable_plotting = not args.no_plot
    plot_config = load_plot_config(args)
    title_template = load_title_template(args)
    
    # Attach title template to plot config
    plot_config.title_template = title_template

    total = len(targets)
    for idx, (gid, coord, alias) in enumerate(targets, 1):
        if total > 1:
            display_name = alias or gid
            print_header(f"Target {idx}/{total}: {display_name}")

        try:
            process_lightcurves(
                gaia_id=gid,
                coord=coord,
                alias=alias,
                skip_tess=args.skip_tess,
                skip_ztf=args.skip_ztf,
                skip_atlas=args.skip_atlas,
                skip_gaia=args.skip_gaia,
                skip_bg=args.skip_bg,
                nsamp=args.force_nsamp,
                minp=args.min_p,
                maxp=args.max_p,
                forced_period=args.force_period,
                no_whitening=args.no_whitening,
                binning=not args.no_binning,
                enable_plotting=enable_plotting,
                ztf_inner_radius=args.ztf_inner_radius,
                ztf_outer_radius=args.ztf_outer_radius,
                ztf_preview=args.plot_ztf_preview,
                ignore_h=not args.include_h,
                ignore_zi=not args.include_zi,
                show_plots=args.show_plots,
                trim_tess=args.trim_tess,
                plot_config=plot_config,
            )
        except Exception as exc:
            print_error(f"Error while processing {gid}: {exc}", gid)
            if total == 1:
                raise
    
    if total > 1:
        print_success(f"\nCompleted {total} targets.")


if __name__ == "__main__":
    main()