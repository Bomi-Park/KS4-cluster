"""
Step 2: Density map construction

Build a 2D density map from red-sequence galaxy candidates.
This map is later converted into a FITS image and fed to SExtractor.

Design notes
------------
- This step should be pure: no file I/O, no global state.
- Default coordinate system: RA/Dec (degrees). You may swap to pixel coords by config.
- Supports optional Gaussian smoothing.
- Supports optional global normalization ("globdensity") for consistent scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class GridSpec:
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    extent: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)


def make_density_map(
    RScand,
    cat_data: Tuple[Any, Any],
    info: Dict[str, Any],
    config: Dict[str, Any],
    figure: bool = False,
) -> Dict[str, Any]:
    """
    Construct density map from RS candidates.

    Parameters
    ----------
    RScand : astropy.table.Table (or pandas-like)
        Red-sequence candidate table.
    cat_data : tuple
        (cat_KS4, galcat). Included for compatibility with original pipeline,
        but not strictly required in this step.
    info : dict
        Field metadata (e.g., RA/Dec limits). Optional keys:
            - 'ra_min', 'ra_max', 'dec_min', 'dec_max' : floats (deg)
            - 'field_id' : str
    config : dict
        Configuration dictionary. Expected section: config['density_map'].
    figure : bool
        If True, show a quick-look plot.

    Returns
    -------
    result : dict
        Keys:
          - 'density_map' : 2D numpy array (float)
          - 'grid'        : GridSpec
          - 'meta'        : dict (settings, stats)
    """
    dens_cfg = config.get("density_map", {})

    # ------------------------------------------------------------
    # 0) Settings
    # ------------------------------------------------------------
    # Coordinate columns
    coord_mode = dens_cfg.get("coord_mode", "radec")  # 'radec' or 'xy'
    if coord_mode not in ("radec", "xy"):
        raise ValueError("density_map.coord_mode must be 'radec' or 'xy'.")

    if coord_mode == "radec":
        x_col = dens_cfg.get("ra_col", "RA")
        y_col = dens_cfg.get("dec_col", "DEC")
    else:
        x_col = dens_cfg.get("x_col", "X_IMAGE")
        y_col = dens_cfg.get("y_col", "Y_IMAGE")

    # Binning / grid
    # You can set either:
    # - nbins (int) and bounds, or
    # - bin_size (float) and bounds (deg or pixel), or
    # - provide edges directly (advanced)
    nbins = dens_cfg.get("nbins", None)
    bin_size = dens_cfg.get("bin_size", None)  # deg (radec) or pixel (xy)

    # Bounds priority: config > info > data min/max
    bounds = dens_cfg.get("bounds", None)  # [xmin, xmax, ymin, ymax]
    xmin = xmax = ymin = ymax = None
    if bounds is not None:
        if len(bounds) != 4:
            raise ValueError("density_map.bounds must be [xmin, xmax, ymin, ymax].")
        xmin, xmax, ymin, ymax = bounds
    else:
        # Try info
        xmin = info.get("ra_min") if coord_mode == "radec" else info.get("x_min")
        xmax = info.get("ra_max") if coord_mode == "radec" else info.get("x_max")
        ymin = info.get("dec_min") if coord_mode == "radec" else info.get("y_min")
        ymax = info.get("dec_max") if coord_mode == "radec" else info.get("y_max")

    # Smoothing
    smooth_sigma = dens_cfg.get("smooth_sigma", 0.0)  # in bins (pixel units of density map)
    smooth_truncate = dens_cfg.get("smooth_truncate", 3.0)

    # Normalization options
    globdensity = bool(dens_cfg.get("globdensity", False))
    norm_mode = dens_cfg.get("norm_mode", "none")  # 'none', 'max', 'sum', 'zscore'
    if norm_mode not in ("none", "max", "sum", "zscore"):
        raise ValueError("density_map.norm_mode must be one of: none|max|sum|zscore")

    # ------------------------------------------------------------
    # 1) Extract coordinates
    # ------------------------------------------------------------
    if RScand is None or len(RScand) == 0:
        # Return empty map with safe defaults
        empty = np.zeros((10, 10), dtype=float)
        grid = _make_default_grid()
        return {
            "density_map": empty,
            "grid": grid,
            "meta": {
                "N_input": 0,
                "coord_mode": coord_mode,
                "x_col": x_col,
                "y_col": y_col,
                "note": "Empty RScand; returned 10x10 zeros",
            },
        }

    try:
        x = np.asarray(RScand[x_col], dtype=float)
        y = np.asarray(RScand[y_col], dtype=float)
    except Exception as e:
        raise KeyError(
            f"Could not read coordinate columns from RScand: {x_col}, {y_col}. "
            f"Check config['density_map'] column names."
        ) from e

    # Remove non-finite
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size == 0:
        empty = np.zeros((10, 10), dtype=float)
        grid = _make_default_grid()
        return {
            "density_map": empty,
            "grid": grid,
            "meta": {
                "N_input": 0,
                "coord_mode": coord_mode,
                "x_col": x_col,
                "y_col": y_col,
                "note": "All coordinates non-finite; returned 10x10 zeros",
            },
        }

    # ------------------------------------------------------------
    # 2) Determine bounds
    # ------------------------------------------------------------
    if xmin is None:
        xmin = float(np.nanmin(x))
    if xmax is None:
        xmax = float(np.nanmax(x))
    if ymin is None:
        ymin = float(np.nanmin(y))
    if ymax is None:
        ymax = float(np.nanmax(y))

    # Expand bounds slightly to avoid edge clipping (optional)
    pad_frac = float(dens_cfg.get("pad_frac", 0.01))
    xmin, xmax, ymin, ymax = _pad_bounds(xmin, xmax, ymin, ymax, pad_frac=pad_frac)

    # ------------------------------------------------------------
    # 3) Build edges
    # ------------------------------------------------------------
    if dens_cfg.get("x_edges") is not None and dens_cfg.get("y_edges") is not None:
        x_edges = np.asarray(dens_cfg["x_edges"], dtype=float)
        y_edges = np.asarray(dens_cfg["y_edges"], dtype=float)
    else:
        if nbins is None and bin_size is None:
            # Sensible default: 256 bins
            nbins = int(dens_cfg.get("default_nbins", 256))

        if bin_size is not None:
            # Compute number of bins from bin_size
            bin_size = float(bin_size)
            if bin_size <= 0:
                raise ValueError("density_map.bin_size must be > 0.")
            nx = max(1, int(np.ceil((xmax - xmin) / bin_size)))
            ny = max(1, int(np.ceil((ymax - ymin) / bin_size)))
            x_edges = np.linspace(xmin, xmax, nx + 1)
            y_edges = np.linspace(ymin, ymax, ny + 1)
        else:
            # nbins can be scalar or (nx, ny)
            if isinstance(nbins, (list, tuple)) and len(nbins) == 2:
                nx, ny = int(nbins[0]), int(nbins[1])
            else:
                nx = ny = int(nbins)
            x_edges = np.linspace(xmin, xmax, nx + 1)
            y_edges = np.linspace(ymin, ymax, ny + 1)

    # ------------------------------------------------------------
    # 4) 2D histogram (density map)
    # ------------------------------------------------------------
    # Note: numpy.histogram2d returns shape (nx, ny) with x bins first.
    H, x_e, y_e = np.histogram2d(x, y, bins=[x_edges, y_edges])
    density = H.astype(float)

    # Optional smoothing
    if smooth_sigma and float(smooth_sigma) > 0:
        density = _gaussian_smooth(density, sigma=float(smooth_sigma), truncate=float(smooth_truncate))

    # ------------------------------------------------------------
    # 5) Normalization
    # ------------------------------------------------------------
    meta_norm = {"globdensity": globdensity, "norm_mode": norm_mode}

    if globdensity:
        # "global" scaling to stabilize SExtractor thresholds across fields/z
        # Typical choices: max-normalize or zscore. We'll follow config.
        density = _normalize(density, mode=norm_mode)
    else:
        # If not global normalization, you may still want a mild scaling; default: none.
        density = _normalize(density, mode=norm_mode) if norm_mode != "none" else density

    # ------------------------------------------------------------
    # 6) Grid metadata
    # ------------------------------------------------------------
    x_centers = 0.5 * (x_e[:-1] + x_e[1:])
    y_centers = 0.5 * (y_e[:-1] + y_e[1:])
    grid = GridSpec(
        x_edges=x_e,
        y_edges=y_e,
        x_centers=x_centers,
        y_centers=y_centers,
        extent=(float(x_e[0]), float(x_e[-1]), float(y_e[0]), float(y_e[-1])),
    )

    meta = {
        "N_input": int(x.size),
        "coord_mode": coord_mode,
        "x_col": x_col,
        "y_col": y_col,
        "bounds": [float(grid.extent[0]), float(grid.extent[1]), float(grid.extent[2]), float(grid.extent[3])],
        "shape": list(density.shape),
        "smooth_sigma": float(smooth_sigma),
        "smooth_truncate": float(smooth_truncate),
        **meta_norm,
        "field_id": info.get("field_id"),
        "z": info.get("z"),
    }

    if figure:
        _plot_density_quicklook(density, grid, info, coord_mode)

    return {
        "density_map": density,
        "grid": grid,
        "meta": meta,
    }


# =============================================================================
# Helpers
# =============================================================================

def _pad_bounds(xmin: float, xmax: float, ymin: float, ymax: float, pad_frac: float = 0.01):
    dx = (xmax - xmin) if (xmax > xmin) else 1.0
    dy = (ymax - ymin) if (ymax > ymin) else 1.0
    pad_x = dx * pad_frac
    pad_y = dy * pad_frac
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def _gaussian_smooth(arr: np.ndarray, sigma: float, truncate: float = 3.0) -> np.ndarray:
    """
    Gaussian smoothing using SciPy if available; fallback to a simple separable kernel.
    Sigma is in pixel units of the density map.
    """
    if sigma <= 0:
        return arr

    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma, truncate=truncate, mode="nearest")
    except Exception:
        # Fallback: build a 1D kernel and do separable convolution
        radius = int(np.ceil(truncate * sigma))
        x = np.arange(-radius, radius + 1)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()

        # Convolve along axis 0 then axis 1
        tmp = _convolve1d(arr, k, axis=0)
        out = _convolve1d(tmp, k, axis=1)
        return out


def _convolve1d(arr: np.ndarray, kernel: np.ndarray, axis: int = 0) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            out[i, :] = np.sum(padded[i:i + len(kernel), :] * kernel[:, None], axis=0)
        return out
    elif axis == 1:
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
        out = np.empty_like(arr, dtype=float)
        for j in range(arr.shape[1]):
            out[:, j] = np.sum(padded[:, j:j + len(kernel)] * kernel[None, :], axis=1)
        return out
    else:
        raise ValueError("axis must be 0 or 1")


def _normalize(arr: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode == "none":
        return arr
    if arr.size == 0:
        return arr

    a = arr.astype(float)
    if mode == "max":
        m = np.nanmax(a)
        return a / m if m > 0 else a
    if mode == "sum":
        s = np.nansum(a)
        return a / s if s > 0 else a
    if mode == "zscore":
        mu = np.nanmean(a)
        sig = np.nanstd(a)
        return (a - mu) / sig if sig > 0 else (a - mu)
    raise ValueError("Unknown normalization mode")


def _make_default_grid() -> GridSpec:
    x_edges = np.linspace(0, 1, 11)
    y_edges = np.linspace(0, 1, 11)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return GridSpec(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        extent=(0.0, 1.0, 0.0, 1.0),
    )


def _plot_density_quicklook(density: np.ndarray, grid: GridSpec, info: Dict[str, Any], coord_mode: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    # imshow expects (ny, nx) as image; our density is (nx, ny) from histogram2d,
    # so transpose for intuitive orientation.
    plt.imshow(
        density.T,
        origin="lower",
        extent=grid.extent,
        aspect="auto",
    )
    plt.colorbar(label="Density (a.u.)")
    plt.xlabel("RA (deg)" if coord_mode == "radec" else "X")
    plt.ylabel("Dec (deg)" if coord_mode == "radec" else "Y")
    z = info.get("z")
    fid = info.get("field_id", "")
    title = f"Density map"
    if fid:
        title += f" | {fid}"
    if z is not None:
        title += f" | z={z:.3f}"
    plt.title(title)
    plt.tight_layout()
    plt.show()
