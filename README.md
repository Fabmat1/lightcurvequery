# lightcurvequery

*A one–stop Python toolkit to download, clean, and analyse time-series photometry for any Gaia DR3 source.*

lightcurvequery talks to the public archives of  
• TESS (MAST) • ZTF • ATLAS • Gaia DR3 epoch photometry
as well as accessing BlackGEM data if you are authorized to do so, combines the light-curves, measures the period through a multiplied Lomb–Scargle periodogram, and produces publication-ready plots – all from a single command.

---

## Key features
* End-to-end pipeline: fetch → quality cut → pre-whiten aliases → period search → plotting.
* Supports Gaia IDs, equatorial coordinates, or plain text lists.
* Fully scriptable **command line interface** with sane defaults.
* Interactive Rich table that shows live download status (falls back to quiet mode on headless systems).
* Highly customisable: skip individual surveys, tweak binning, whitening, sample grid, ZTF radius, …
* Generates three kinds of output ready for inspection or further processing:

```
lightcurves/<GAIA_ID>/       raw ASCII light-curves (one file per survey)
periodograms/<GAIA_ID>/      multiplied & per-survey periodograms (CSV)
lcplots/                     phase-folded light-curve PDFs
pgramplots/                  periodogram overview PDFs
```

---

## Installation

```bash
git clone https://github.com/your-user/lightcurvequery.git
cd lightcurvequery
python -m venv venv            # optional
source venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt   # astropy, astroquery, lightkurve, ztfquery, …
pip install -e .                 # editable install
```

Optional extras  
- ZTF: create a `~/.ztfquery` config (interactive on first run)  
- ATLAS forced photometry: export your personal API token to use ATLAS photometry (add this to your `.bashrc`, `.zshrc` or similar)  
  `export ATLASFORCED_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
  ATLAS tokens can also be stored in `~/.atlaskey`


---

## Quick start

Some usage examples:

```bash
# Fetch everything that is publicly available for Gaia DR3 XXXXXXXXXXXXXXXXXXX
lightcurvequery XXXXXXXXXXXXXXXXXXX

# Same, but provide coordinates (deg) instead of an ID
lightcurvequery --coords 269.4521 -24.8801

# Use a text file that contains one Gaia ID per line
lightcurvequery --file targets.txt

# Only TESS + Gaia, search periods between 0.1 and 5 d, no plots
lightcurvequery XXXXXXXXXXXXXXXXXXX --skip-ztf --skip-atlas --skip-bg \
                         --min-p 0.1 --max-p 5 --no-plot
```

While the program runs you will see

```
Surveying lightcurves for Gaia DR3 XXXXXXXXXXXXXXXXXXX
┏━━━━━━┳━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━━━━┓
┃ TESS ┃ ZTF ┃ ATLAS ┃ Gaia ┃ BlackGEM ┃
┡━━━━━━╇━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━━━━┩
│  ✓   │  /  │   ✓   │   ✓  │     *    │
└──────┴─────┴───────┴──────┴──────────┘
```

where `✓` = downloaded/available, `✗` = no data, `*` = skipped.

---

## Command-line reference (abridged)

```
lightcurvequery  [TARGETS …]              Gaia IDs or coordinates

Input options
  --coords RA DEC           supply coordinates (deg)
  --file  path.txt          read IDs from text file

Survey toggles
  --skip-tess  -t
  --skip-ztf   -z
  --skip-atlas -a
  --skip-gaia  -g
  --skip-bg    -b

Period search
  --min-p / --max-p FLOAT   search window in days   (0.05‒50 default)
  --force-nsamp   INT       override #frequency samples
  --force-period  FLOAT     skip periodogram and use given period

Processing flags
  --no-binning              leave raw cadence
  --no-whitening            turn off alias pre-whitening
  --no-plot                 do not generate any PDFs
  --show-plots BOOL         do not show plots in interactive window (default True)

ZTF fine-tuning
  --ztf-inner-radius 5      arcsec kept around best source
  --ztf-outer-radius 20     initial query radius
  --plot-ztf-preview        PDF sky view of all ZTF matches

Photometry filters
  --include-h               keep ATLAS H-band (off by default)
  --include-zi              keep ZTF z_i band (off by default)
```

Run `lightcurvequery -h` for the full help text.

---

## Python API

Everything used by the CLI is available programmatically:

```python
from lightcurvequery_core.process import process_lightcurves

process_lightcurves(
    gaia_id      = "5853498713190526720",
    skip_ztf     = True,
    minp         = 0.1,
    maxp         = 5,
    enable_plotting = False,
)
```

The returned `Star` object (see `lightcurvequery_core.star.Star`) contains
```python
star.lightcurves        # dict of pandas DataFrames
star.periodograms       # dict of [power, period] arrays
star.period             # measured period (or None)
star.metadata["TESS_CROWD"]   # average CROWDSAP crowding metric (if available)
```

### Utility scripts

* `convert_lightcurve.py` – bin and phase an existing ASCII TESS light-curve  
  into *x Δx flux σ* columns for use with LCURVE.

* `getcrowdsap.py` – query MAST via lightkurve for the CROWDSAP aperture-contamination
  factor of any TIC. (Now included in all plots too)

Both are standalone executables independent of the main pipeline.

---

## Contributing & Support

Bug reports, feature requests or pull requests are welcome!  
Open an issue on GitHub or contact me directly.

If you use lightcurvequery in your research, please include it in the acknowledgments.
I leave the wording up to you, since I hate forced acknowledgement blocks.

---

## License

This project is released under the MIT License – see [`LICENSE`](LICENSE) for details.
