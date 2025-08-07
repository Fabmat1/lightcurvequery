"""
Command-line interface – exactly the same behaviour as the old script.
"""
from __future__ import annotations

import sys
import argparse
from typing import List, Tuple

from astropy import units as u
from astropy.coordinates import SkyCoord

from .process import process_lightcurves
from .utils import bcolors

# --- optional: only imported when --coords is used --------------------------
try:
    from astroquery.gaia import Gaia
except ModuleNotFoundError:
    Gaia = None   # type: ignore


# ────────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process lightcurves for astronomical objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # positional
    parser.add_argument('targets', nargs='*',
                        help='List of Gaia source IDs or RA DEC')

    # mutually exclusive input options
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--coords', nargs=2, metavar=('RA', 'DEC'), type=float)
    g.add_argument('--file', '-i', type=str)

    # survey flags
    parser.add_argument('--skip-tess', '-t', action='store_true')
    parser.add_argument('--skip-ztf', '-z', action='store_true')
    parser.add_argument('--skip-atlas', '-a', action='store_true')
    parser.add_argument('--skip-gaia', '-g', action='store_true')
    parser.add_argument('--skip-bg', '-b', action='store_true')

    # processing options
    parser.add_argument('--no-binning', '-B', action='store_true')
    parser.add_argument('--no-whitening', '-W', action='store_true')
    parser.add_argument('--no-plot', '-P', action='store_true')
    parser.add_argument('--plot', '-p', action='store_true')

    # period options
    parser.add_argument('--min-p', '-m', type=float, default=0.05)
    parser.add_argument('--max-p', '-M', type=float, default=50.0)
    parser.add_argument('--force-nsamp', '-n', type=int)
    parser.add_argument('--force-period', '-f', type=float)

    return parser


# ────────────────────────────────────────────────────────────────────
def validate_gaia_id(gid: str):
    try:
        int(gid)
        if len(gid) < 10:
            raise ValueError
    except ValueError:
        print(f"Invalid Gaia ID: {gid}")
        sys.exit(1)


def load_gaia_ids_from_file(path) -> List[Tuple[str, None]]:
    ids: list[Tuple[str, None]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    validate_gaia_id(line)
                    ids.append((line, None))
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)
    return ids


def query_gaia_by_coordinates(coord: SkyCoord) -> str:
    if Gaia is None:
        print("astroquery.gaia not installed – cannot resolve coordinates.")
        sys.exit(1)

    job = Gaia.cone_search_async(coord, radius=5*u.arcsec)
    res = job.get_results()
    if len(res) == 0:
        print("No Gaia source within 5\".")
        sys.exit(1)
    return str(res[0]["SOURCE_ID"])


def resolve_targets(args) -> List[Tuple[str, SkyCoord | None]]:
    targets: list[Tuple[str, SkyCoord | None]] = []

    # --file
    if args.file:
        targets.extend(load_gaia_ids_from_file(args.file))
        return targets

    # --coords
    if args.coords:
        ra, dec = map(float, args.coords)
        coord = SkyCoord(ra*u.deg, dec*u.deg)
        gid = query_gaia_by_coordinates(coord)
        targets.append((gid, coord))
        return targets

    # positional
    if not args.targets:
        print("No targets provided.")
        sys.exit(1)

    if len(args.targets) == 2:
        try:
            ra, dec = map(float, args.targets)
            coord = SkyCoord(ra*u.deg, dec*u.deg)
            gid = query_gaia_by_coordinates(coord)
            return [(gid, coord)]
        except ValueError:
            pass

    for tok in args.targets:
        if tok.endswith('.txt'):
            targets.extend(load_gaia_ids_from_file(tok))
        else:
            validate_gaia_id(tok)
            targets.append((tok, None))
    return targets


# ────────────────────────────────────────────────────────────────────
def main():
    parser = parse_arguments()
    args = parser.parse_args()

    targets = resolve_targets(args)
    enable_plotting = not args.no_plot

    if args.force_period:
        print(f"Forced period: {args.force_period}")

    total = len(targets)
    for idx, (gid, coord) in enumerate(targets, 1):
        if total > 1:
            print(f"\n{'='*60}\nTarget {idx}/{total}: Gaia DR3 {gid}\n{'='*60}")

        try:
            process_lightcurves(
                gaia_id=gid,
                coord=coord,
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
            )
        except Exception as exc:
            print(f"Error while processing {gid}: {exc}")
            if total == 1:
                raise

    if total > 1:
        print(f"\nCompleted {total} targets.")


if __name__ == "__main__":
    main()