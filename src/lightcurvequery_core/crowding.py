"""
crowding.py
-----------
Fetch and save preview images of a target's field from multiple, independent
surveys (TESS, ZTF, Gaia, DSS, Pan-STARRS) to help judge how badly blended /
crowded the source is.  Also computes & caches the mean TESS CROWDSAP metric
across all available sectors.

All outputs go to ``lightcurves/{gaia_id}/``:
    tess_preview.png     – TESScut FFI cutout                  (TESS)
    ztf_preview.png      – ZTF reference image cutout          (ZTF)
    gaia_preview.png     – Scatter of nearby Gaia DR3 sources  (Gaia)
    dss_preview.png      – DSS2 colour composite               (DSS, independent)
    ps1_preview.png      – Pan-STARRS y/i/g composite          (PS1, independent)
    tess_crowdsap.txt    – Mean CROWDSAP across all downloaded sectors
"""

from __future__ import annotations

import csv
import warnings
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
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ZScaleInterval, ImageNormalize

from .terminal_style import (
    print_info, print_success, print_warning, print_error
)

warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings(
    "ignore",
    message="Query returned no results",
    module="astroquery.mast.discovery_portal",
)

# ---------------------------------------------------------------------------
# Common look-and-feel for every preview image
# ---------------------------------------------------------------------------
COMMON_FOV_ARCSEC = 180.0   # (B) every preview shows the same 3' FOV
OUTPUT_DATA_PX    = 1000    # data array side after reprojection
FIG_INCHES        = 8       # (E) crisper output:
FIG_DPI           = 200     #     8 × 200 = 1600 px PNGs

RETICLE_COLOR     = 'red'        # (F) clean N/S/E/W ticks, target is never occluded
SCALEBAR_COLOR    = 'magenta'    # (G) high-contrast on both dark & light bgs
COMPASS_N_COLOR   = 'blue'       # (C) N arrow – blue
COMPASS_E_COLOR   = 'red'        # (C) E arrow – red

# ===========================================================================
#                              utility helpers
# ===========================================================================
def _ensure_dir(gaia_id) -> Path:
    outdir = Path(f"lightcurves/{gaia_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _resolve_coord(gaia_id, coord: Optional[SkyCoord] = None) -> SkyCoord:
    if coord is not None:
        return coord
    return SkyCoord.from_name(f"GAIA DR3 {gaia_id}")


def _choose_scale(fov_arcsec: float) -> float:
    """Pick a round scale-bar length so it spans ~15-25 % of the FOV."""
    target = 0.20 * fov_arcsec
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120,
                  180, 300, 600, 1200, 1800, 3600]
    valid = [c for c in candidates if c <= 0.30 * fov_arcsec]
    if not valid:
        return candidates[0]
    return min(valid, key=lambda c: abs(c - target))


def _scale_label(arcsec: float) -> str:
    if arcsec >= 60:
        v = arcsec / 60.0
        return f"{v:g}'"
    return f'{arcsec:g}"'


def _reproject_north_up(data, wcs, coord,
                        fov_arcsec: float = COMMON_FOV_ARCSEC,
                        output_pix: int = OUTPUT_DATA_PX,
                        order: str = 'bilinear'):
    """Reproject to a common north-up / east-left FOV centred on ``coord``."""
    if data is None or wcs is None:
        return data, wcs
    try:
        from reproject import reproject_interp
        size  = output_pix
        cdelt = (fov_arcsec / 3600.0) / size       # deg / pix
        new_wcs = WCS(naxis=2)
        new_wcs.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
        new_wcs.wcs.crval = [coord.ra.deg, coord.dec.deg]
        new_wcs.wcs.cdelt = [-cdelt, cdelt]
        new_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        new_data, _ = reproject_interp(
            (data, wcs), new_wcs, shape_out=(size, size), order=order,
        )
        return new_data, new_wcs
    except Exception:
        return data, wcs

def _draw_reticle(ax, x, y, nx, ny,
                  color: str = RETICLE_COLOR,
                  gap_frac: float = 0.018,
                  length_frac: float = 0.055,
                  lw: float = 2.2):
    """4 short ticks N/S/E/W with a gap in the middle – never covers the target."""
    s = min(nx, ny)
    gap, length = gap_frac * s, length_frac * s
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ax.plot([x + dx * gap, x + dx * (gap + length)],
                [y + dy * gap, y + dy * (gap + length)],
                color=color, lw=lw, solid_capstyle='round')


def _draw_compass(ax, nx, ny, arm_frac: float = 0.08,
                  pad_frac: float = 0.04):
    """Two perpendicular arrows in the top-right: blue=N (up), red=E (left)."""
    import matplotlib.patheffects as pe
    arm = arm_frac * min(nx, ny)
    pad = pad_frac * min(nx, ny)
    x0  = nx - pad - arm
    y0  = ny - pad - arm
    halo = [pe.withStroke(linewidth=3.5, foreground='black')]
    txt_halo = [pe.withStroke(linewidth=2, foreground='black')]

    # North arrow (up) – blue
    ax.annotate('', xy=(x0, y0 + arm), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=COMPASS_N_COLOR,
                                lw=2.5, shrinkA=0, shrinkB=0,
                                mutation_scale=15,
                                path_effects=halo))
    ax.text(x0, y0 + arm * 1.08, 'N',
            color=COMPASS_N_COLOR, ha='center', va='bottom',
            fontsize=13, fontweight='bold', path_effects=txt_halo)

    # East arrow (left) – red
    ax.annotate('', xy=(x0 - arm, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=COMPASS_E_COLOR,
                                lw=2.5, shrinkA=0, shrinkB=0,
                                mutation_scale=15,
                                path_effects=halo))
    ax.text(x0 - arm * 1.08, y0, 'E',
            color=COMPASS_E_COLOR, ha='right', va='center',
            fontsize=13, fontweight='bold', path_effects=txt_halo)

def _stretch_to_unit(arr) -> np.ndarray:
    """ZScale-normalise an array to [0, 1] for RGB compositing."""
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
    vmin, vmax = ZScaleInterval().get_limits(arr)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)

def _skip_if_exists(path: Path, gaia_id, label: str) -> bool:
    """Return True (and log) if ``path`` already exists."""
    if path.exists():
        print_info(f"{label} already exists – skipping ({path.name})",
                   gaia_id, "CROWDING")
        return True
    return False

# ---------------------------------------------------------------------------
# Common image display: no axes, no colorbar, target marker, scale bar
# ---------------------------------------------------------------------------
def _display_image(data, wcs, coord, outpath,
                   cmap: str = 'gray_r',
                   is_rgb: bool = False,
                   reproject_order: str = 'bilinear'):
    """Save a preview at the common FOV, with reticle, compass and scale bar."""
    import matplotlib.patheffects as pe

    # ---- reproject to the shared FOV (B) ---------------------------------
    if wcs is not None:
        if is_rgb:
            chans, new_wcs = [], None
            for i in range(3):
                arr, w = _reproject_north_up(
                    data[..., i], wcs, coord, order=reproject_order,
                )
                chans.append(arr)
                new_wcs = new_wcs or w
            data = np.nan_to_num(
                np.clip(np.stack(chans, axis=-1), 0.0, 1.0), nan=0.0,
            )
            wcs = new_wcs
        else:
            data, wcs = _reproject_north_up(
                data, wcs, coord, order=reproject_order,
            )

    ny, nx = data.shape[:2]
    fig = plt.figure(figsize=(FIG_INCHES, FIG_INCHES), dpi=FIG_DPI)
    ax  = fig.add_axes([0, 0, 1, 1])

    if is_rgb:
        ax.imshow(data, origin='lower', interpolation='nearest')
    else:
        safe = np.nan_to_num(data, nan=np.nanmedian(data))
        norm = ImageNormalize(safe, interval=ZScaleInterval())
        ax.imshow(safe, origin='lower', cmap=cmap, norm=norm,
                  interpolation='nearest')

    if wcs is not None:
        pix_scale_as = float(
            np.mean(np.abs(proj_plane_pixel_scales(wcs)))
        ) * 3600.0
        actual_fov = nx * pix_scale_as

        # ---- (F) clean reticle -------------------------------------------
        try:
            x, y = wcs.world_to_pixel(coord)
            _draw_reticle(ax, x, y, nx, ny)
        except Exception:
            pass

        # ---- (C) compass --------------------------------------------------
        _draw_compass(ax, nx, ny)

        # ---- (G) magenta scale bar with high-contrast halo ----------------
        bar_as  = _choose_scale(actual_fov)
        bar_pix = bar_as / pix_scale_as
        x0 = 0.05 * nx
        y0 = 0.06 * ny
        halo   = [pe.withStroke(linewidth=5, foreground='black')]
        t_halo = [pe.withStroke(linewidth=2.5, foreground='black')]
        ax.plot([x0, x0 + bar_pix], [y0, y0],
                color=SCALEBAR_COLOR, lw=4, solid_capstyle='butt',
                path_effects=halo)
        ax.text(x0 + bar_pix / 2, y0 + 0.022 * ny,
                _scale_label(bar_as),
                color=SCALEBAR_COLOR, ha='center', va='bottom',
                fontsize=15, fontweight='bold',
                path_effects=t_halo)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)

# ===========================================================================
#                              TESS
# ===========================================================================

def fetch_tess_preview(gaia_id, coord, outdir) -> bool:
    out = outdir / "tess_preview.png"
    if _skip_if_exists(out, gaia_id, "TESS preview"):
        return True
    print_info("Fetching TESScut preview ...", gaia_id, "CROWDING")
    try:
        import lightkurve as lk
        sr = lk.search_tesscut(coord)
        if sr is None or len(sr) == 0:
            print_warning("No TESS coverage", gaia_id, "CROWDING")
            return False
        # need ≥ COMMON_FOV / 21″ pix = 9 pix → 15 px gives margin
        tpf = sr[0].download(cutout_size=15)
        if tpf is None:
            print_warning("TESScut download returned None",
                          gaia_id, "CROWDING")
            return False
        img = np.nanmedian(np.asarray(tpf.flux.value), axis=0)
        # Use nearest-neighbour so the 21″ pixels stay visible as blocks.
        _display_image(img, tpf.wcs, coord, out,
                       reproject_order='nearest-neighbor')
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
    """Mean TESS CROWDSAP from SPOC, with TIC contratio as a fallback."""
    crowd_file = outdir / "tess_crowdsap.txt"
    if crowd_file.exists():
        try:
            val = float(open(crowd_file).read().splitlines()[0].strip())
            print_success(f"Mean TESS CROWDSAP = {val:.4f}  (cached)",
                          gaia_id, "CROWDING")
            return val
        except Exception:
            pass

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


# ---------------- ZTF ---------------------------------------------------------
def fetch_ztf_preview(gaia_id, coord, outdir,
                      size_arcsec: float = COMMON_FOV_ARCSEC + 60) -> bool:
    out = outdir / "ztf_preview.png"
    if _skip_if_exists(out, gaia_id, "ZTF preview"):
        return True
    print_info("Fetching ZTF reference cutout ...", gaia_id, "CROWDING")
    try:
        ra, dec = coord.ra.deg, coord.dec.deg
        search_url = (
            "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/ref"
            f"?POS={ra},{dec}&ct=csv"
        )
        r = requests.get(search_url, timeout=60); r.raise_for_status()
        lines = r.text.strip().splitlines()
        if len(lines) < 2:
            print_warning("No ZTF reference image found",
                          gaia_id, "CROWDING")
            return False
        rows = list(csv.DictReader(lines))
        priority = {"zr": 0, "zg": 1, "zi": 2}
        rows.sort(key=lambda row: priority.get(row.get("filtercode", ""), 9))
        row = rows[0]
        field, ccdid, qid = int(row["field"]), int(row["ccdid"]), int(row["qid"])
        filt   = row["filtercode"]
        padded = f"{field:06d}"; prefix = padded[:3]
        fname  = f"ztf_{padded}_{filt}_c{ccdid:02d}_q{qid}_refimg.fits"
        url = (
            f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/"
            f"{prefix}/field{padded}/{filt}/ccd{ccdid:02d}/q{qid}/{fname}"
            f"?center={ra},{dec}&size={size_arcsec}arcsec&gzip=false"
        )
        imgr = requests.get(url, timeout=120); imgr.raise_for_status()
        data, wcs = None, None
        with fits.open(BytesIO(imgr.content)) as hdul:
            for h in hdul:
                if h.data is not None:
                    data, wcs = h.data, WCS(h.header)
                    break
        if data is None:
            print_warning("ZTF cutout had no image data",
                          gaia_id, "CROWDING")
            return False
        _display_image(data, wcs, coord, out)
        print_success("ZTF preview saved", gaia_id, "CROWDING")
        return True
    except Exception as e:
        print_warning(f"ZTF preview failed: {e}", gaia_id, "CROWDING")
        return False


# ---------------- DSS  (BW DSS2 Red — point D) --------------------------------
def fetch_dss_preview(gaia_id, coord, outdir,
                      radius_arcmin: float = COMMON_FOV_ARCSEC / 60.0) -> bool:
    out = outdir / "dss_preview.png"
    if _skip_if_exists(out, gaia_id, "DSS preview"):
        return True
    print_info("Fetching DSS2 Red preview ...", gaia_id, "CROWDING")
    try:
        from astroquery.skyview import SkyView
        for survey in ("DSS2 Red", "DSS"):
            try:
                imgs = SkyView.get_images(
                    position=coord, survey=[survey],
                    radius=radius_arcmin * u.arcmin, pixels=800,
                )
                if imgs:
                    hdu = imgs[0][0]
                    _display_image(hdu.data, WCS(hdu.header), coord, out)
                    print_success(f"DSS preview saved ({survey})",
                                  gaia_id, "CROWDING")
                    return True
            except Exception:
                continue
        print_warning("No DSS image available", gaia_id, "CROWDING")
        return False
    except Exception as e:
        print_warning(f"DSS preview failed: {e}", gaia_id, "CROWDING")
        return False


# ===========================================================================
#                       Pan-STARRS  (y/i/g RGB composite)
# ===========================================================================
def _ps1_files(ra, dec, filters="yig"):
    meta = requests.get(
        "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py",
        params={"ra": ra, "dec": dec, "filters": filters, "type": "stack"},
        timeout=30,
    )
    meta.raise_for_status()
    lines = meta.text.strip().splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split()
    idx_filt = header.index("filter")
    idx_file = header.index("filename")
    out = {}
    for line in lines[1:]:
        parts = line.split()
        out[parts[idx_filt]] = parts[idx_file]
    return out


# ---------------- Pan-STARRS  (y/i/g RGB, BUT only if all 3 OK; else g BW) ----
def fetch_ps1_preview(gaia_id, coord, outdir,
                      size_arcmin: float = COMMON_FOV_ARCSEC / 60.0 + 1) -> bool:
    out = outdir / "ps1_preview.png"
    if _skip_if_exists(out, gaia_id, "Pan-STARRS preview"):
        return True
    print_info("Fetching Pan-STARRS preview ...", gaia_id, "CROWDING")
    try:
        ra, dec = coord.ra.deg, coord.dec.deg
        size_pix = int(size_arcmin * 60 / 0.25)
        files = _ps1_files(ra, dec, "yig")
        if not files:
            print_warning("PS1 has no coverage here", gaia_id, "CROWDING")
            return False
        order = ['y', 'i', 'g']
        bands, wcs = [], None
        for f in order:
            if f not in files:
                bands.append(None); continue
            url = (
                "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
                f"?ra={ra}&dec={dec}&size={size_pix}&format=fits"
                f"&red={files[f]}"
            )
            try:
                rr = requests.get(url, timeout=120); rr.raise_for_status()
                with fits.open(BytesIO(rr.content)) as hdul:
                    bands.append(hdul[0].data.astype(float))
                    if wcs is None:
                        wcs = WCS(hdul[0].header)
            except Exception:
                bands.append(None)

        if all(b is not None for b in bands):
            rgb = np.stack([_stretch_to_unit(b) for b in bands], axis=-1)
            _display_image(rgb, wcs, coord, out, is_rgb=True)
            print_success("Pan-STARRS preview (RGB) saved",
                          gaia_id, "CROWDING")
            return True
        valid = [b for b in bands if b is not None]
        if valid:
            _display_image(valid[0], wcs, coord, out)
            print_success("Pan-STARRS preview (single band) saved",
                          gaia_id, "CROWDING")
            return True
        print_warning("PS1 fitscut returned nothing", gaia_id, "CROWDING")
        return False
    except Exception as e:
        print_warning(f"PS1 preview failed: {e}", gaia_id, "CROWDING")
        return False

# ===========================================================================
#                              Gaia DR3 (actual)
# ===========================================================================
def fetch_gaia_preview(gaia_id, coord, outdir, radius_arcmin: float = 1.0) -> bool:
    out = outdir / "gaia_preview.png"
    if _skip_if_exists(out, gaia_id, "Gaia preview"):
        return True
    print_info("Fetching Gaia DR3 sources ...", gaia_id, "CROWDING")
    try:
        from astroquery.gaia import Gaia
        radius_deg = (radius_arcmin * u.arcmin).to(u.deg).value
        q = (
            "SELECT ra, dec, phot_g_mean_mag, source_id "
            "FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
            "POINT('ICRS', ra, dec),"
            f"CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius_deg}))"
        )
        job = Gaia.launch_job(q)
        tab = job.get_results()
        if len(tab) == 0:
            print_warning("No Gaia sources in field", gaia_id, "CROWDING")
            return False

        # tangent-plane offsets in arcsec (east-LEFT, north-UP)
        src = SkyCoord(tab['ra'], tab['dec'], unit='deg')
        d_lon, d_lat = coord.spherical_offsets_to(src)
        x = -d_lon.to(u.arcsec).value
        y =  d_lat.to(u.arcsec).value

        g = np.array(tab['phot_g_mean_mag'])
        g = np.where(np.isfinite(g), g, 21.0)
        sizes = np.clip(220.0 * 10.0 ** (-0.4 * (g - 13.0)), 4, 600)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor('white')
        ax.scatter(x, y, s=sizes, c='black', alpha=0.85, edgecolor='none')
        ax.plot(0, 0, '+', color='red', markersize=18, markeredgewidth=2)
        ax.add_patch(Circle((0, 0), 1.5, edgecolor='red', facecolor='none',
                             linewidth=1.3, linestyle='--'))

        lim = radius_arcmin * 60.0   # arcsec half-width
        ax.set_xlim(-lim, lim)       # east-left (x = -d_lon already flips)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_axis_off()

        # scale bar – bottom-left
        bar_as = _choose_scale(2 * lim)
        x0, y0 = -lim * 0.90, -lim * 0.85
        ax.plot([x0, x0 + bar_as], [y0, y0],
                color='black', lw=3, solid_capstyle='butt')
        ax.text(x0 + bar_as / 2, y0 + lim * 0.03,
                _scale_label(bar_as), color='black',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(outdir / "gaia_preview.png", dpi=140,
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print_success(f"Gaia preview saved ({len(tab)} sources)",
                      gaia_id, "CROWDING")
        return True
    except Exception as e:
        print_warning(f"Gaia preview failed: {e}", gaia_id, "CROWDING")
        return False


# ===========================================================================
#                              top-level entry
# ===========================================================================
def fetch_crowding_data(gaia_id,
                        coord: Optional[SkyCoord] = None,
                        skip_tess: bool = False,
                        skip_ztf: bool = False,    # kept for API compat
                        skip_atlas: bool = False,  # kept for API compat
                        skip_gaia: bool = False    # kept for API compat
                        ) -> dict:
    """Fetch every preview image + (optionally) mean TESS CROWDSAP.

    Notes
    -----
    The ``skip_*`` flags are kept for backwards compatibility but **do not**
    skip preview-image fetches – every preview that has data for the field
    is always produced (point (c) of the redesign).  ``skip_tess`` is the
    only one that has an effect: it suppresses the catalogue-level CROWDSAP
    computation.
    """
    coord = _resolve_coord(gaia_id, coord)
    outdir = _ensure_dir(gaia_id)

    print_info(f"=== Crowding preview ({gaia_id}) ===", gaia_id, "CROWDING")

    results: dict = {}

    # ---- (c) all previews are always attempted, regardless of skips ------
    results['tess_preview'] = fetch_tess_preview(gaia_id, coord, outdir)
    results['ztf_preview']  = fetch_ztf_preview(gaia_id, coord, outdir)
    results['gaia_preview'] = fetch_gaia_preview(gaia_id, coord, outdir)
    results['dss_preview']  = fetch_dss_preview(gaia_id, coord, outdir)
    results['ps1_preview']  = fetch_ps1_preview(gaia_id, coord, outdir)

    # ---- only the catalogue-level metric is gated by a skip flag ---------
    if not skip_tess:
        results['tess_crowdsap'] = fetch_tess_crowdsap(gaia_id, coord, outdir)

    print_success(f"Crowding products written to {outdir}/",
                  gaia_id, "CROWDING")
    return results