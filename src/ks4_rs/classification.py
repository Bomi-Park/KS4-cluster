"""
classification.py

Star–galaxy separation utilities for the KS4 RS pipeline.

This module is designed to:
1) provide a clean, reusable interface for star/galaxy classification
2) optionally cross-match to Gaia DR3 for robust star identification
3) apply field-level quality cuts based on star counts

Notes
-----
- This file intentionally avoids hard-coding survey-specific paths.
- Provide Gaia data either as:
    (a) an already-loaded Table/DataFrame of Gaia sources for the field, or
    (b) a user-provided `gaia_loader(field_id)` callable that returns Gaia sources.

Expected catalog columns
------------------------
KS4 catalog (cat_KS4):
- RA, DEC (degrees)
- CLASS_STAR (SExtractor stellarity index, 0..1)  [configurable]

Gaia catalog:
- ra, dec (degrees)  (or RA/DEC; configurable)

Outputs
-------
- galcat : galaxy-only table
- starcat: star-only table
- meta   : diagnostic information
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


GaiaLoaderFn = Callable[[Any], Any]


@dataclass
class ClassifyResult:
    galcat: Any
    starcat: Any
    meta: Dict[str, Any]


def classify_star_galaxy(
    cat_KS4: Any,
    config: Dict[str, Any],
    *,
    field_id: Optional[Any] = None,
    gaia_sources: Optional[Any] = None,
    gaia_loader: Optional[GaiaLoaderFn] = None,
) -> ClassifyResult:
    """
    Classify stars and galaxies using SExtractor CLASS_STAR and optional Gaia cross-match.

    Parameters
    ----------
    cat_KS4 : Table-like
        Full source catalog for a field.
    config : dict
        Configuration dictionary. Expected section: config['classification'].
    field_id : optional
        Field identifier (used only if gaia_loader is provided).
    gaia_sources : optional, Table-like
        Gaia sources for this field.
    gaia_loader : optional, callable(field_id) -> gaia_sources
        Loader to fetch Gaia sources if not provided.

    Returns
    -------
    ClassifyResult
        galcat : galaxy-only catalog
        starcat: star-only catalog
        meta   : diagnostics

    Raises
    ------
    ValueError
        If Gaia cross-match is enabled but neither gaia_sources nor gaia_loader is provided.
    """
    cfg = config.get("classification", {})

    ra_col = cfg.get("ra_col", "RA")
    dec_col = cfg.get("dec_col", "DEC")
    cs_col = cfg.get("class_star_col", "CLASS_STAR")

    cs_star_thresh = float(cfg.get("class_star_star_thresh", 0.98))
    cs_gal_thresh = float(cfg.get("class_star_gal_thresh", 0.50))
    # If CLASS_STAR in-between, default to galaxy unless Gaia says star (configurable)
    mid_as_gal = bool(cfg.get("mid_as_galaxy", True))

    use_gaia = bool(cfg.get("use_gaia", True))
    gaia_match_radius_arcsec = float(cfg.get("gaia_match_radius_arcsec", 1.0))

    gaia_ra_col = cfg.get("gaia_ra_col", "ra")
    gaia_dec_col = cfg.get("gaia_dec_col", "dec")

    # ------------------------------
    # 0) Basic masks from CLASS_STAR
    # ------------------------------
    cs = np.asarray(cat_KS4[cs_col], dtype=float)

    star_mask_cs = cs >= cs_star_thresh
    gal_mask_cs = cs <= cs_gal_thresh
    mid_mask = (~star_mask_cs) & (~gal_mask_cs)

    if mid_as_gal:
        gal_mask = gal_mask_cs | mid_mask
    else:
        gal_mask = gal_mask_cs
    star_mask = star_mask_cs

    meta: Dict[str, Any] = {
        "field_id": str(field_id) if field_id is not None else None,
        "ra_col": ra_col,
        "dec_col": dec_col,
        "class_star_col": cs_col,
        "class_star_star_thresh": cs_star_thresh,
        "class_star_gal_thresh": cs_gal_thresh,
        "n_total": int(len(cat_KS4)),
        "n_star_cs": int(star_mask.sum()),
        "n_gal_cs": int(gal_mask.sum()),
        "use_gaia": use_gaia,
    }

    # ------------------------------
    # 1) Optional Gaia cross-match
    # ------------------------------
    gaia_star_mask = None
    if use_gaia:
        if gaia_sources is None:
            if gaia_loader is None:
                raise ValueError(
                    "Gaia cross-match enabled (classification.use_gaia=true) but no gaia_sources "
                    "or gaia_loader provided."
                )
            if field_id is None:
                raise ValueError("field_id must be provided when using gaia_loader.")
            gaia_sources = gaia_loader(field_id)

        gaia_star_mask = _match_to_gaia(
            cat_KS4,
            gaia_sources,
            ra_col=ra_col,
            dec_col=dec_col,
            gaia_ra_col=gaia_ra_col,
            gaia_dec_col=gaia_dec_col,
            radius_arcsec=gaia_match_radius_arcsec,
        )
        meta["n_gaia_match"] = int(np.sum(gaia_star_mask))
        meta["gaia_match_radius_arcsec"] = gaia_match_radius_arcsec

        # Apply Gaia info:
        # - Any Gaia match is considered a star by default
        # - This is conservative for removing stars from galaxy selection
        prefer_gaia_star = bool(cfg.get("prefer_gaia_star", True))
        if prefer_gaia_star:
            star_mask = star_mask | gaia_star_mask
            gal_mask = gal_mask & (~gaia_star_mask)

    # ------------------------------
    # 2) Build outputs
    # ------------------------------
    starcat = cat_KS4[star_mask]
    galcat = cat_KS4[gal_mask]

    meta["n_star_final"] = int(len(starcat))
    meta["n_gal_final"] = int(len(galcat))

    return ClassifyResult(galcat=galcat, starcat=starcat, meta=meta)


def field_pass_starcount_cut(
    starcat: Any,
    config: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Field-level selection based on star counts.

    Parameters
    ----------
    starcat : Table-like
        Star catalog returned from classify_star_galaxy.
    config : dict
        Expected section: config['field_selection'].

    Returns
    -------
    passed : bool
        Whether the field passes the star count cut.
    meta : dict
        Diagnostic info.
    """
    cfg = config.get("field_selection", {})
    max_stars = cfg.get("max_stars", None)
    min_stars = cfg.get("min_stars", None)

    nstar = int(len(starcat))
    passed = True

    if max_stars is not None and nstar > int(max_stars):
        passed = False
    if min_stars is not None and nstar < int(min_stars):
        passed = False

    meta = {
        "n_star": nstar,
        "max_stars": int(max_stars) if max_stars is not None else None,
        "min_stars": int(min_stars) if min_stars is not None else None,
        "passed": passed,
    }
    return passed, meta


# =============================================================================
# Internal helpers
# =============================================================================

def _match_to_gaia(
    cat: Any,
    gaia: Any,
    *,
    ra_col: str,
    dec_col: str,
    gaia_ra_col: str,
    gaia_dec_col: str,
    radius_arcsec: float,
) -> np.ndarray:
    """
    Cross-match catalog sources to Gaia sources within a radius.

    Returns
    -------
    matched_mask : np.ndarray(bool)
        True for sources in `cat` that have a Gaia counterpart within radius.
    """
    ra = np.asarray(cat[ra_col], dtype=float)
    dec = np.asarray(cat[dec_col], dtype=float)

    # Gaia col name fallbacks (some tables use RA/DEC)
    if gaia_ra_col not in gaia.colnames and "RA" in getattr(gaia, "colnames", []):
        gaia_ra_col = "RA"
    if gaia_dec_col not in gaia.colnames and "DEC" in getattr(gaia, "colnames", []):
        gaia_dec_col = "DEC"

    gra = np.asarray(gaia[gaia_ra_col], dtype=float)
    gdec = np.asarray(gaia[gaia_dec_col], dtype=float)

    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    g = SkyCoord(ra=gra * u.deg, dec=gdec * u.deg)

    idx, sep2d, _ = c.match_to_catalog_sky(g)
    matched = sep2d < (radius_arcsec * u.arcsec)
    return np.asarray(matched, dtype=bool)
