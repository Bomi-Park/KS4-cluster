"""
utils.py

Shared helper utilities for the KS4 RS pipeline.

Principles
----------
- Keep functions small and reusable
- Avoid survey-path hardcoding
- Prefer pure functions (no global state)
- Provide astronomy-safe utilities (RA wrap, angular separation, etc.)

This module is where you move:
- optimal color selection logic
- zbin helpers
- safe nearest-value selection (your earlier request: if z not in list, replace with nearest)
- robust angle handling (RA wrap-around)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u


# =============================================================================
# Redshift helpers
# =============================================================================

def nearest_value(x: float, values: Sequence[float]) -> float:
    """
    Return the element in `values` closest to `x`.

    Useful when z is not exactly in a pre-defined grid (z_range_fin),
    and you want to replace it with the nearest available value.
    """
    if values is None or len(values) == 0:
        raise ValueError("values must be a non-empty sequence")
    arr = np.asarray(values, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(x)))])


def make_zbins(z_list: Sequence[float], *, dz: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Make simple z-bins around each z in z_list.

    If dz is provided, each bin is (z - dz/2, z + dz/2).
    If dz is None, bins are computed from neighbor spacing (midpoints).

    Returns
    -------
    zbins : list of (zmin, zmax)
    """
    z = np.sort(np.asarray(z_list, dtype=float))
    if z.size == 0:
        return []

    if dz is not None:
        half = 0.5 * float(dz)
        return [(float(zz - half), float(zz + half)) for zz in z]

    # midpoint bins
    mids = 0.5 * (z[1:] + z[:-1])
    zmins = np.empty_like(z)
    zmaxs = np.empty_like(z)
    zmins[0] = z[0] - (mids[0] - z[0])
    zmaxs[-1] = z[-1] + (z[-1] - mids[-1])
    zmins[1:] = mids
    zmaxs[:-1] = mids
    return [(float(a), float(b)) for a, b in zip(zmins, zmaxs)]


def find_zbin(z: float, z_list: Sequence[float], zbins: Optional[Sequence[Tuple[float, float]]] = None) -> Tuple[float, float]:
    """
    Return the z-bin for a given z.

    If zbins not provided, it is constructed from z_list using midpoint method.
    If z is not exactly in z_list, it is replaced by nearest z in z_list.
    """
    z_near = nearest_value(z, z_list)
    if zbins is None:
        zbins = make_zbins(z_list, dz=None)
    # Find index of z_near in sorted z_list
    z_sorted = np.sort(np.asarray(z_list, dtype=float))
    idx = int(np.where(np.isclose(z_sorted, z_near))[0][0])
    return tuple(map(float, zbins[idx]))  # (zmin, zmax)


# =============================================================================
# Color selection helpers (your "optimal_color_fin" logic)
# =============================================================================

def get_optimal_color(z: float) -> str:
    """
    Choose the optimal color index for RS selection based on redshift.

    This follows the logic you used before:
    - z < 0.25  : B - I
    - 0.25 <= z < 0.47 : V - I
    - 0.47 <= z <= 0.8 : R - I
    """
    z = float(z)
    if z < 0.25:
        return "B-I"
    elif z < 0.47:
        return "V-I"
    elif z <= 0.80:
        return "R-I"
    else:
        raise ValueError("Optimal color is undefined for z > 0.8 (update logic if needed).")


def parse_color_to_mags(color: str) -> Tuple[str, str]:
    """
    Convert a color string like 'R-I' into ('R','I').

    Accepts variants with spaces.
    """
    c = color.replace(" ", "")
    if "-" not in c:
        raise ValueError(f"Invalid color format: {color}. Expected like 'R-I'.")
    a, b = c.split("-", 1)
    return a, b


# =============================================================================
# Angle / coordinate helpers
# =============================================================================

def wrap_ra_deg(ra_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Wrap RA into [0, 360) degrees.
    """
    ra = np.asarray(ra_deg, dtype=float)
    ra_wrapped = np.mod(ra, 360.0)
    return float(ra_wrapped) if np.isscalar(ra_deg) else ra_wrapped


def angular_separation_arcmin(
    ra1: Union[float, np.ndarray],
    dec1: Union[float, np.ndarray],
    ra2: Union[float, np.ndarray],
    dec2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Great-circle angular separation in arcmin.

    Works for scalars or arrays (broadcasting supported by astropy).
    """
    c1 = SkyCoord(ra=np.asarray(ra1, dtype=float) * u.deg, dec=np.asarray(dec1, dtype=float) * u.deg)
    c2 = SkyCoord(ra=np.asarray(ra2, dtype=float) * u.deg, dec=np.asarray(dec2, dtype=float) * u.deg)
    sep = c1.separation(c2)
    return sep.to_value(u.arcmin)


def within_radius(
    ra: np.ndarray,
    dec: np.ndarray,
    ra0: float,
    dec0: float,
    radius_arcmin: float,
) -> np.ndarray:
    """
    Return boolean mask for sources within radius_arcmin of (ra0, dec0).
    """
    c = SkyCoord(ra=np.asarray(ra, dtype=float) * u.deg, dec=np.asarray(dec, dtype=float) * u.deg)
    c0 = SkyCoord(ra=float(ra0) * u.deg, dec=float(dec0) * u.deg)
    return c.separation(c0) < (float(radius_arcmin) * u.arcmin)


# =============================================================================
# Table helpers (works with astropy.table.Table or numpy structured arrays)
# =============================================================================

def safe_column(table: Any, col: str, default: Any = None):
    """
    Safely get a column from a Table-like object.
    Returns `default` if not present.
    """
    try:
        if hasattr(table, "colnames") and col in table.colnames:
            return table[col]
        # numpy structured array or dict-like
        if isinstance(table, dict) and col in table:
            return table[col]
        return default
    except Exception:
        return default


def require_columns(table: Any, columns: Sequence[str], *, name: str = "table") -> None:
    """
    Raise a KeyError if any required columns are missing.
    """
    missing = []
    for c in columns:
        ok = False
        if hasattr(table, "colnames"):
            ok = c in table.colnames
        elif isinstance(table, dict):
            ok = c in table
        else:
            try:
                _ = table[c]
                ok = True
            except Exception:
                ok = False
        if not ok:
            missing.append(c)
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


# =============================================================================
# Simple logging helper (optional)
# =============================================================================

def format_meta(info: Dict[str, Any]) -> str:
    """
    Compact string for logging.
    """
    field_id = info.get("field_id", "")
    z = info.get("z", None)
    if z is None:
        return f"{field_id}"
    return f"{field_id} | z={float(z):.3f}"
