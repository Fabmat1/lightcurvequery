"""
Light-curve query mini-package.

Besides exposing the public API of the new modular code, this file *also*
creates *virtual* legacy modules so that **every** old import line such as

    import models
    from models import Star
    from common_functions import t_colors, load_star

continues to work – without cluttering the repository root with stub files.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# expose the real, refactored implementation
# ────────────────────────────────────────────────────────────────────
from .star import Star                                # noqa: F401 (re-exported)
from .utils import calcpgramsamples, t_colors, bcolors  # noqa: F401

__all__ = ["Star", "calcpgramsamples", "t_colors", "bcolors"]

# ────────────────────────────────────────────────────────────────────
# build “virtual” legacy modules
# ────────────────────────────────────────────────────────────────────

# --- 1) legacy  *models*  module -----------------------------------
models_mod = types.ModuleType("models")
models_mod.Star = Star
models_mod.calcpgramsamples = calcpgramsamples
sys.modules["models"] = models_mod

# --- 2) legacy  *common_functions*  module -------------------------
#
# The plotting code only ever needs:
#   • t_colors
#   • bcolors
#   • load_star   (but only in the __main__ demo section of makephotplot.py)
#
# To stay lightweight we provide a *minimal* load_star stub that creates an
# empty Star object.  Users who relied on the huge original implementation can
# still supply their own replacement elsewhere on the PYTHONPATH; that version
# would override this stub automatically thanks to Python’s import rules.
#
cf_mod = types.ModuleType("common_functions")
cf_mod.t_colors = t_colors
cf_mod.bcolors = bcolors


def _load_star_stub(gaia_id, *args, **kwargs):
    """
    Minimal replacement for the colossal `common_functions.load_star`.

    It only returns an empty Star instance containing the requested Gaia ID,
    just enough for the demo section of *makephotplot.py* to run.

    If you need the full original behaviour, drop your own implementation
    named ``common_functions`` *ahead* of ``lightcurve_query`` on
    ``PYTHONPATH`` and it will shadow this lightweight stub.
    """
    print(
        "[lightcurve_query] WARNING: using lightweight load_star stub – "
        "no spectra / RV data are loaded"
    )
    return Star(gaia_id)


cf_mod.load_star = _load_star_stub
sys.modules["common_functions"] = cf_mod