"""
Gaia DR3 ``source_id`` ⇄ sky coordinate resolution.

Every fetcher used to call ``SkyCoord.from_name("GAIA DR3 <id>")``, i.e. it
asked CDS/Sesame (SIMBAD, NED, VizieR name tables).  Sesame only knows objects
that actually have a catalogue *entry* under that designation, so a perfectly
ordinary Gaia source that nobody ever wrote a paper about makes every single
survey fail with ``NameResolveError`` – even though the Gaia archive knows its
position perfectly well.

So we ask the archives that are indexed by ``source_id`` first and keep Sesame
as the last resort:

    1. Gaia archive TAP  (gaiadr3.gaia_source, authoritative, epoch 2016.0)
    2. VizieR I/355/gaiadr3  (mirror of the same catalogue)
    3. CDS/Sesame via ``SkyCoord.from_name``  (old behaviour)

Results are memoised per process: the five fetcher threads of one star each
need the coordinates, and there is no point in querying five times.
"""

from __future__ import annotations

import threading
from typing import Optional

import requests
from astropy import units as u
from astropy.coordinates import SkyCoord

from .terminal_style import print_warning

__all__ = [
    "CoordinateResolutionError",
    "register_coord",
    "resolve_gaia_coord",
    "resolve_source_id",
]

GAIA_TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
TAP_TIMEOUT = 60          # seconds; the archive is occasionally slow, not dead

_cache: dict[str, SkyCoord] = {}
_cache_lock = threading.Lock()
_id_locks: dict[str, threading.Lock] = {}


class CoordinateResolutionError(RuntimeError):
    """Raised when no service could turn a Gaia ID into coordinates."""


# ────────────────────────────────────────────────────────────────────
# low level: one query per service
# ────────────────────────────────────────────────────────────────────
def _tap_query(adql: str, timeout: float = TAP_TIMEOUT) -> list[list[str]]:
    """Run a synchronous ADQL query and return the CSV rows without header."""
    resp = requests.get(
        GAIA_TAP_SYNC,
        params={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": adql,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    return [ln.split(",") for ln in lines[1:]]


def _from_gaia_tap(gaia_id: str) -> Optional[SkyCoord]:
    rows = _tap_query(
        "SELECT TOP 1 ra, dec FROM gaiadr3.gaia_source "
        f"WHERE source_id = {int(gaia_id)}"
    )
    if not rows:
        return None
    ra, dec = rows[0][0], rows[0][1]
    return SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")


def _from_vizier(gaia_id: str) -> Optional[SkyCoord]:
    from astroquery.vizier import Vizier

    # an *instance* – the module level ``Vizier`` class attributes are mutated
    # elsewhere (see fetchers.get_tic) and we do not want to inherit that.
    viz = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS"], row_limit=5)
    res = viz.query_constraints(catalog="I/355/gaiadr3",
                                Source=str(int(gaia_id)))
    if not res:
        return None
    tbl = res[0]
    if "Source" in tbl.colnames:
        tbl = tbl[tbl["Source"] == int(gaia_id)]
    if len(tbl) == 0:
        return None
    return SkyCoord(float(tbl["RA_ICRS"][0]) * u.deg,
                    float(tbl["DE_ICRS"][0]) * u.deg,
                    frame="icrs")


def _from_sesame(gaia_id: str) -> Optional[SkyCoord]:
    return SkyCoord.from_name(f"GAIA DR3 {gaia_id}")


_RESOLVERS = (
    ("Gaia archive", _from_gaia_tap),
    ("VizieR I/355/gaiadr3", _from_vizier),
    ("CDS/Sesame", _from_sesame),
)


# ────────────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────────────
def register_coord(gaia_id, coord: Optional[SkyCoord]) -> None:
    """Seed the cache with coordinates we already know (e.g. ``--coords``)."""
    if coord is None:
        return
    with _cache_lock:
        _cache[str(gaia_id)] = coord


def resolve_gaia_coord(gaia_id,
                       coord: Optional[SkyCoord] = None,
                       *,
                       instrument: Optional[str] = None) -> SkyCoord:
    """Return the ICRS position of a Gaia DR3 source.

    ``coord`` short-circuits everything, so callers that were handed a
    position by the user can pass it straight through.  Raises
    :class:`CoordinateResolutionError` if every service failed.
    """
    if coord is not None:
        register_coord(gaia_id, coord)
        return coord

    key = str(gaia_id)
    with _cache_lock:
        hit = _cache.get(key)
        lock = _id_locks.setdefault(key, threading.Lock())
    if hit is not None:
        return hit

    # only one thread per star talks to the network, the others wait and
    # then find the answer in the cache
    with lock:
        with _cache_lock:
            hit = _cache.get(key)
        if hit is not None:
            return hit

        failures: list[str] = []
        for i, (name, resolver) in enumerate(_RESOLVERS):
            try:
                found = resolver(key)
            except Exception as exc:                      # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            if found is None:
                failures.append(f"{name}: no match")
                continue
            if i:  # something before this one did not work – say so once
                print_warning(
                    f"Coordinates resolved via {name} "
                    f"(tried: {', '.join(n for n, _ in _RESOLVERS[:i])}).",
                    gaia_id, instrument,
                )
            with _cache_lock:
                _cache[key] = found
            return found

    raise CoordinateResolutionError(
        f"Could not resolve coordinates for Gaia DR3 {key}. Attempts:\n  "
        + "\n  ".join(failures)
    )


def resolve_source_id(coord: SkyCoord, radius_arcsec: float = 5.0) -> Optional[str]:
    """Nearest ``gaiadr3.gaia_source`` ID within ``radius_arcsec`` of ``coord``.

    A drop-in for ``Gaia.cone_search_async`` that does not need astroquery's
    gaia module (whose import likes to hang on a cold TAP server).
    """
    ra, dec = float(coord.icrs.ra.deg), float(coord.icrs.dec.deg)
    radius_deg = float(radius_arcsec) / 3600.0
    rows = _tap_query(
        "SELECT TOP 1 source_id, "
        f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS d "
        "FROM gaiadr3.gaia_source "
        "WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) "
        "ORDER BY d ASC"
    )
    if not rows:
        return None
    return rows[0][0].strip()
