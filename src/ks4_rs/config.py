"""
config.py

Load and manage pipeline configuration.

Role of config
--------------
- Separate "parameters" from "code"
- Make pipeline runs reproducible (config file = run recipe)
- Avoid hard-coding environment-specific paths
- Provide defaults and sanity checks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os

import yaml


# ----------------------------- Public API ----------------------------- #

def load_config(path: Union[str, Path], *, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML configuration and apply defaults + overrides.

    Parameters
    ----------
    path : str or Path
        Path to YAML config file.
    overrides : dict, optional
        Dictionary to override config values after loading.
        Useful for quick experiments in notebooks / scripts.

    Returns
    -------
    config : dict
        Fully merged configuration dictionary.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _apply_defaults(cfg)
    cfg = _expand_paths(cfg)

    if overrides:
        cfg = deep_update(cfg, overrides)

    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Minimal validation to catch common mistakes early.
    Keep this lightweight and extend as needed.
    """
    # red_sequence
    rs = cfg.get("red_sequence", {})
    if "mag1" not in rs or "mag2" not in rs:
        raise ValueError("config['red_sequence'] must define 'mag1' and 'mag2' (e.g., 'MAG_AUTO_R').")

    # density_map
    dm = cfg.get("density_map", {})
    if dm.get("coord_mode", "radec") not in ("radec", "xy"):
        raise ValueError("config['density_map']['coord_mode'] must be 'radec' or 'xy'.")

    # sextractor
    sex = cfg.get("sextractor", {})
    # exe can be missing (defaults to 'sex' in the step), so no hard check.
    # But config path(s) are usually needed for real runs:
    # allow missing if user wants to pass extra_args only.
    # still, warn-ish behavior is avoided (no prints) – raise only if explicitly required.
    if sex.get("require_config_files", False):
        for k in ("config", "parameters"):
            if not sex.get(k):
                raise ValueError(f"sextractor.require_config_files=true but config['sextractor']['{k}'] is missing.")

    # cluster_detect
    cd = cfg.get("cluster_detect", {})
    if cd.get("member_radius_arcmin", 1.0) <= 0:
        raise ValueError("cluster_detect.member_radius_arcmin must be > 0.")
    if int(cd.get("min_members", 5)) < 1:
        raise ValueError("cluster_detect.min_members must be >= 1.")


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary (like yaml merge).
    """
    out = dict(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# ----------------------------- Internals ----------------------------- #

def _apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide sensible defaults so missing keys don't crash the pipeline.
    Users can override them in YAML.
    """
    defaults = {
        "red_sequence": {
            # IMPORTANT: set these to your catalog column names
            "mag1": "MAG_AUTO_R",
            "mag2": "MAG_AUTO_I",
            "cutsig": 0.5,
        },
        "density_map": {
            "coord_mode": "radec",
            "ra_col": "RA",
            "dec_col": "DEC",
            "nbins": 256,
            "smooth_sigma": 1.0,
            "smooth_truncate": 3.0,
            "globdensity": True,
            "norm_mode": "max",
            "pad_frac": 0.01,
        },
        "plot_density": {
            "figsize": (6, 5),
            "cmap": "viridis",
            "vmin": None,
            "vmax": None,
        },
        "fits": {
            "tag": "density",
        },
        "sextractor": {
            "exe": "sex",
            "out_dir": "outputs/sex",
            "tag": "sex",
            "catalog_ext": ".cat",
            "make_segmentation": True,
            "make_background": False,
            "write_log": True,
            "extra_args": "-VERBOSE_TYPE QUIET",
            "require_config_files": False,
        },
        "cluster_detect": {
            "member_radius_arcmin": 1.0,
            "min_members": 5,
        },
        "cluster_images": {
            "out_dir": "outputs/cluster_images",
            "figsize": (6, 5),
            "cmap": "viridis",
            "member_color": "cyan",
            "center_color": "red",
        },
        "classification": {
            "ra_col": "RA",
            "dec_col": "DEC",
            "class_star_col": "CLASS_STAR",
            "class_star_star_thresh": 0.98,
            "class_star_gal_thresh": 0.50,
            "mid_as_galaxy": True,
            "use_gaia": True,
            "gaia_match_radius_arcsec": 1.0,
            "gaia_ra_col": "ra",
            "gaia_dec_col": "dec",
            "prefer_gaia_star": True,
        },
        "field_selection": {
            # If None: disabled
            "max_stars": None,
            "min_stars": None,
        },
        "consolidation": {
            "match_radius_arcmin": 2.0,
            "dz_max": 0.05,
            "adaptive_dz": False,
            "dz_base": 0.02,
            "dz_slope": 0.02,
            "use_member_overlap": False,
            "min_member_jaccard": 0.2,
            "member_id_col": "OBJID",
            "representative": "max_n_member",
        },
    }

    return deep_update(defaults, cfg)


def _expand_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand user (~) and environment variables in any string path-like values.
    This keeps configs portable across machines.
    """
    def expand(v):
        if isinstance(v, str):
            # Expand env vars and ~
            return os.path.expandvars(str(Path(v).expanduser()))
        if isinstance(v, dict):
            return {kk: expand(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [expand(x) for x in v]
        return v

    return expand(cfg)
