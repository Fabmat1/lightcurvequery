"""
crowding.py
-----------
Fetch and save preview images of a target's field from multiple surveys
(ATLAS-equivalent, Gaia, ZTF, TESS) to help judge how badly blended /
crowded the source is.  Also computes & caches the mean TESS CROWDSAP
metric across all available sectors.

All outputs go to ``lightcurves/{gaia_id}/``:
    tess_preview.png     – TESScut FFI cutout with the target marked
    ztf_preview.png      – ZTF reference image cutout
    atlas_preview.png    – DSS2-Red cutout (ATLAS pixel/PSF reference)
    gaia_preview.png     – Scatter plot of nearby Gaia DR3 sources
    tess_crowdsap.txt    – Mean CROWDSAP across all downloaded sectors
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize

from .terminal_style import (
    print_info, print_success, print_warning, print_error
)

import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings(
    "ignore",
    message="Query returned no results",
    module="astroquery.mast.discovery_portal",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ensure_dir(gaia_id) -> Path:
    outdir = Path(f"lightcurves/{gaia_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _resolve_coord(gaia_id, coord: Optional[SkyCoord] = None) -> SkyCoord:
    if coord is not None:
        return coord
    return SkyCoord.from_name(f"GAIA DR3 {gaia_id}")


def _imshow_fits(data, wcs, coord, title, outpath, marker_arcsec=3):
    """Display a FITS array with the target marked by a red cross + circle."""
    fig = plt.figure(figsize=(6, 6))
    if wcs is not None:
        ax = fig.add_subplot(111, projection=wcs)
    else:
        ax = fig.add_subplot(111)

    safe = np.nan_to_num(data, nan=np.nanmedian(data))
    norm = ImageNormalize(safe, interval=ZScaleInterval())
    ax.imshow(safe, origin='lower', cmap='gray_r', norm=norm)

    if wcs is not None:
        try:
            x, y = wcs.world_to_pixel(coord)
            ax.plot(x, y, '+', color='red', markersize=18, markeredgewidth=2)
            try:
                from astropy.wcs.utils import proj_plane_pixel_scales
                pix_scale = float(proj_plane_pixel_scales(wcs)[0]) * 3600
                r_pix = marker_arcsec / pix_scale
                ax.add_patch(Circle((x, y), r_pix, edgecolor='red',
                                    facecolor='none', linewidth=1.5,
                                    linestyle='--'))
            except Exception:
                pass
        except Exception:
            pass
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# TESS – cutout + mean CROWDSAP
# ---------------------------------------------------------------------------
def fetch_tess_preview(gaia_id, coord, outdir) -> bool:
    """TESScut FFI cutout via lightkurve."""
    print_info("Fetching TESScut preview ...", gaia_id, "CROWDING")
    try:
        import lightkurve as lk
        sr = lk.search_tesscut(coord)
        if sr is None or len(sr) == 0:
            print_warning("No TESS coverage", gaia_id, "CROWDING")
            return False

        tpf = sr[0].download(cutout_size=15)
        if tpf is None:
            print_warning("TESScut download returned None", gaia_id, "CROWDING")
            return False

        fig, ax = plt.subplots(figsize=(6, 6))
        tpf.plot(ax=ax, show_colorbar=True)
        try:
            x, y = tpf.wcs.world_to_pixel(coord)
            ax.plot(x + tpf.column, y + tpf.row, 'r+',
                    markersize=20, markeredgewidth=2)
        except Exception:
            pass
        ax.set_title(f"TESS Sector {tpf.sector} (≈21″/px)")
        fig.savefig(outdir / "tess_preview.png", dpi=120, bbox_inches='tight')
        plt.close(fig)
        print_success("TESS preview saved", gaia_id, "CROWDING")
        return True
    except Exception as e:
        print_warning(f"TESS preview failed: {e}", gaia_id, "CROWDING")
        return False


def _tic_contratio(gaia_id) -> Optional[float]:
    """Return TIC v8 contratio (contaminant flux / target flux) or None."""
    try:
        from astroquery.vizier import Vizier
        from .fetchers import get_tic

        tic = get_tic(gaia_id)
        if "No TIC" in str(tic) or "Error" in str(tic):
            return None

        v = Vizier(columns=["TIC", "Cont"])
        v.ROW_LIMIT = -1
        res = v.query_constraints(catalog="IV/39/tic82", TIC=str(tic))
        if not res or len(res[0]) == 0:
            return None
        contratio = float(res[0]["Cont"][0])
        if not np.isfinite(contratio) or contratio < 0:
            return None
        return contratio
    except Exception:
        return None


def fetch_tess_crowdsap(gaia_id, coord, outdir) -> Optional[float]:
    """
    Mean TESS CROWDSAP over all SPOC sectors; if no SPOC LCs exist,
    fall back to an estimate from the TIC v8 ``contratio`` column:
        CROWDSAP ≈ 1 / (1 + contratio)
    """
    crowd_file = outdir / "tess_crowdsap.txt"
    if crowd_file.exists():
        try:
            val = float(open(crowd_file).read().splitlines()[0].strip())
            print_success(f"Mean TESS CROWDSAP = {val:.4f}  (cached)",
                          gaia_id, "CROWDING")
            return val
        except Exception:
            pass

    # ---------- 1) Try genuine SPOC CROWDSAP -------------------------------
    print_info("Computing mean TESS CROWDSAP from SPOC LCs …",
               gaia_id, "CROWDING")
    crowdsaps: list[float] = []
    try:
        from astroquery.mast import Observations
        from .fetchers import get_tic

        tic = get_tic(gaia_id)
        if "No TIC" in str(tic) or "Error" in str(tic):
            raise RuntimeError("no TIC")

        obsTable = Observations.query_criteria(
            dataproduct_type="timeseries",
            project="TESS",
            target_name=str(tic),
        )
        if len(obsTable) > 0:
            prods = Observations.get_product_list(obsTable)
            mask = np.isin(prods["productSubGroupDescription"],
                           ["LC", "FAST-LC"])
            prods_lc = prods[mask]
            if len(prods_lc) > 0:
                downloaded = Observations.download_products(prods_lc)
                if downloaded is not None:
                    for path in downloaded["Local Path"]:
                        try:
                            with fits.open(path) as hdul:
                                cs = hdul[1].header.get("CROWDSAP", None)
                                if cs is not None and np.isfinite(cs):
                                    crowdsaps.append(float(cs))
                        except Exception:
                            continue
    except Exception as e:
        print_warning(f"SPOC CROWDSAP query failed: {e}",
                      gaia_id, "CROWDING")

    if crowdsaps:
        mean_cs = float(np.mean(crowdsaps))
        with open(crowd_file, "w") as f:
            f.write(f"{mean_cs:.6f}\n"
                    f"# source: SPOC, N={len(crowdsaps)} sectors\n")
        print_success(
            f"Mean TESS CROWDSAP = {mean_cs:.4f}  "
            f"(SPOC, N={len(crowdsaps)})", gaia_id, "CROWDING")
        return mean_cs

    # ---------- 2) Fallback: TIC v8 contratio ------------------------------
    print_info("No SPOC LCs found – estimating crowding from TIC contratio …",
               gaia_id, "CROWDING")
    contratio = _tic_contratio(gaia_id)
    if contratio is None:
        print_warning("TIC contratio unavailable; no crowdsap saved.",
                      gaia_id, "CROWDING")
        return None

    crowdsap_est = 1.0 / (1.0 + contratio)
    with open(crowd_file, "w") as f:
        f.write(f"{crowdsap_est:.6f}\n"
                f"# source: TIC contratio={contratio:.4f}  "
                f"(estimate, no SPOC LC available)\n")
    print_success(
        f"Estimated CROWDSAP = {crowdsap_est:.4f}  "
        f"(from TIC contratio={contratio:.4f})", gaia_id, "CROWDING")
    return crowdsap_est


# ---------------------------------------------------------------------------
# ZTF – reference image cutout via IRSA SIA
# ---------------------------------------------------------------------------
def fetch_ztf_preview(gaia_id, coord, outdir) -> bool:
    print_info("Fetching ZTF reference cutout ...", gaia_id, "CROWDING")
    try:
        ra, dec = coord.ra.deg, coord.dec.deg
        size_deg = 60.0 / 3600.0     # 1 arcmin
        sia_url = (
            "https://irsa.ipac.caltech.edu/ibe/sia/ztf/products/ref"
            f"?POS={ra},{dec}&SIZE={size_deg}&MCEN&INTERSECT=CENTER"
        )
        r = requests.get(sia_url, timeout=60)
        r.raise_for_status()

        from astropy.io.votable import parse_single_table
        table = parse_single_table(BytesIO(r.content)).to_table()
        if len(table) == 0:
            print_warning("No ZTF reference image found", gaia_id, "CROWDING")
            return False

        # Find URL column
        url_col = None
        for cand in ('access_url', 'sia_url', 'URL', 'access_estsize'):
            if cand in table.colnames:
                url_col = cand
                break
        if url_col is None:
            for col in table.colnames:
                if 'url' in col.lower():
                    url_col = col
                    break
        if url_col is None:
            print_warning("No URL in ZTF SIA response", gaia_id, "CROWDING")
            return False

        img_url = str(table[0][url_col])
        # IRSA cutout parameters
        if 'center=' not in img_url.lower():
            join = '&' if '?' in img_url else '?'
            img_url += f"{join}center={ra},{dec}&size={size_deg}deg&gzip=false"

        imgr = requests.get(img_url, timeout=120)
        imgr.raise_for_status()

        data, wcs = None, None
        with fits.open(BytesIO(imgr.content)) as hdul:
            for h in hdul:
                if h.data is not None:
                    data = h.data
                    wcs = WCS(h.header)
                    break
        if data is None:
            print_warning("ZTF cutout had no image data", gaia_id, "CROWDING")
            return False

        _imshow_fits(data, wcs, coord, "ZTF reference image (~1′)",
                     outdir / "ztf_preview.png", marker_arcsec=2)
        print_success("ZTF preview saved", gaia_id, "CROWDING")
        return True
    except Exception as e:
        print_warning(f"ZTF preview failed: {e}", gaia_id, "CROWDING")
        return False


# ---------------------------------------------------------------------------
# ATLAS – DSS2 Red used as a "matched-bandpass / resolution" proxy
# ---------------------------------------------------------------------------
def fetch_atlas_preview(gaia_id, coord, outdir) -> bool:
    print_info("Fetching ATLAS-equivalent preview (DSS2 Red) ...",
               gaia_id, "CROWDING")
    try:
        from astroquery.skyview import SkyView
        imgs = SkyView.get_images(
            position=coord,
            survey=['DSS2 Red'],
            radius=2 * u.arcmin,
            pixels=500,
        )
        if not imgs:
            print_warning("No SkyView image returned", gaia_id, "CROWDING")
            return False
        hdu = imgs[0][0]
        data = hdu.data
        wcs = WCS(hdu.header)

        _imshow_fits(
            data, wcs, coord,
            "ATLAS field (DSS2 Red, ATLAS PSF ≈ 5–8″)",
            outdir / "atlas_preview.png",
            marker_arcsec=5,
        )
        print_success("ATLAS preview saved", gaia_id, "CROWDING")
        return True
    except Exception as e:
        print_warning(f"ATLAS preview failed: {e}", gaia_id, "CROWDING")
        return False

# ---------------------------------------------------------------------------
# Gaia – high-resolution Pan-STARRS cutout (with DSS fallback)
# ---------------------------------------------------------------------------
def _panstarrs_image_url(ra: float, dec: float, size_pix: int = 480,
                         filt: str = "r") -> Optional[str]:
    """Query the PS1 filename service and build a fitscut URL."""
    try:
        meta = requests.get(
            "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py",
            params={"ra": ra, "dec": dec, "filters": filt, "type": "stack"},
            timeout=30,
        )
        meta.raise_for_status()
        lines = meta.text.strip().splitlines()
        if len(lines) < 2:
            return None
        header = lines[0].split()
        idx = header.index("filename")
        filename = lines[1].split()[idx]
        return (
            "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
            f"?ra={ra}&dec={dec}&size={size_pix}&format=fits&red={filename}"
        )
    except Exception:
        return None


def fetch_gaia_preview(gaia_id, coord, outdir) -> bool:
    """High-resolution image (Pan-STARRS-r, DSS2-Red fallback) of the field.

    Pan-STARRS gives ~0.25″/px — a good match for Gaia's resolution, so
    visually inspecting it tells you exactly what Gaia sees as separate
    sources vs. blends.
    """
    print_info("Fetching Gaia-resolution preview …", gaia_id, "CROWDING")
    ra, dec = coord.ra.deg, coord.dec.deg

    # ---- try Pan-STARRS first --------------------------------------------
    url = _panstarrs_image_url(ra, dec, size_pix=480, filt="r")
    if url is not None:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with fits.open(BytesIO(r.content)) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
            if data is not None and np.any(np.isfinite(data)):
                _imshow_fits(
                    data, wcs, coord,
                    "Gaia-resolution view (Pan-STARRS r, ~0.25″/px)",
                    outdir / "gaia_preview.png",
                    marker_arcsec=1,
                )
                print_success("Gaia preview (PS1) saved",
                              gaia_id, "CROWDING")
                return True
        except Exception as e:
            print_warning(f"Pan-STARRS fetch failed ({e}); "
                          "falling back to DSS2.", gaia_id, "CROWDING")

    # ---- fallback: DSS2 Red via SkyView ----------------------------------
    try:
        from astroquery.skyview import SkyView
        imgs = SkyView.get_images(
            position=coord,
            survey=['DSS2 Red'],
            radius=1 * u.arcmin,
            pixels=600,
        )
        if not imgs:
            print_warning("No DSS fallback image", gaia_id, "CROWDING")
            return False
        hdu = imgs[0][0]
        _imshow_fits(
            hdu.data, WCS(hdu.header), coord,
            "Gaia-resolution view (DSS2 Red, fallback)",
            outdir / "gaia_preview.png",
            marker_arcsec=2,
        )
        print_success("Gaia preview (DSS2) saved", gaia_id, "CROWDING")
        return True
    except Exception as e:
        print_warning(f"Gaia preview failed: {e}", gaia_id, "CROWDING")
        return False


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------
def fetch_crowding_data(gaia_id,
                        coord: Optional[SkyCoord] = None,
                        skip_tess: bool = False,
                        skip_ztf: bool = False,
                        skip_atlas: bool = False,
                        skip_gaia: bool = False) -> dict:
    """Fetch every preview image + mean TESS CROWDSAP for a single target.

    Saves outputs to ``lightcurves/{gaia_id}/``.

    Returns
    -------
    dict
        Mapping of the things that were attempted to either bool (image
        saved) or float (CROWDSAP).
    """
    coord = _resolve_coord(gaia_id, coord)
    outdir = _ensure_dir(gaia_id)

    print_info(f"=== Crowding preview ({gaia_id}) ===", gaia_id, "CROWDING")

    results: dict = {}
    if not skip_tess:
        results['tess_preview']  = fetch_tess_preview(gaia_id, coord, outdir)
        results['tess_crowdsap'] = fetch_tess_crowdsap(gaia_id, coord, outdir)
    if not skip_ztf:
        results['ztf_preview']   = fetch_ztf_preview(gaia_id, coord, outdir)
    if not skip_atlas:
        results['atlas_preview'] = fetch_atlas_preview(gaia_id, coord, outdir)
    if not skip_gaia:
        results['gaia_preview']  = fetch_gaia_preview(gaia_id, coord, outdir)

    print_success(f"Crowding products written to {outdir}/",
                  gaia_id, "CROWDING")
    return results