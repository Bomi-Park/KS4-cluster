"""
Step 4: Convert density map to FITS image

This step converts the numerical density map into a FITS image
that can be ingested by SExtractor.

Design principles
-----------------
- Minimal but valid FITS + WCS
- Compatible with SExtractor
- No SExtractor execution here (pure I/O)
"""

from typing import Any, Dict
from pathlib import Path

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS


def make_density_fits(
    density_result: Dict[str, Any],
    info: Dict[str, Any],
    image_path: str | Path,
    config: Dict[str, Any],
    figure: bool = False,
) -> Dict[str, Any]:
    """
    Create a FITS image from the density map.

    Parameters
    ----------
    density_result : dict
        Output from step2_density_map.make_density_map.
    info : dict
        Field metadata (field_id, z, etc.).
    image_path : str or Path
        Output directory for FITS files.
    config : dict
        Configuration dictionary.
    figure : bool
        Reserved for future diagnostic plots.

    Returns
    -------
    result : dict
        Keys
        ----
        'fits_path' : Path
            Path to the created FITS file.
        'header' : astropy.io.fits.Header
            FITS header (including WCS).
    """

    dens = density_result.get("density_map")
    grid = density_result.get("grid")
    meta = density_result.get("meta", {})

    if dens is None or grid is None:
        raise ValueError("density_result must contain 'density_map' and 'grid'.")

    fits_cfg = config.get("fits", {})

    # ------------------------------------------------------------
    # 0) Output path
    # ------------------------------------------------------------
    image_path = Path(image_path)
    image_path.mkdir(parents=True, exist_ok=True)

    field_id = info.get("field_id", "field")
    z = info.get("z", None)

    tag = fits_cfg.get("tag", "density")
    if z is not None:
        fname = f"{field_id}_{tag}_z{z:.3f}.fits"
    else:
        fname = f"{field_id}_{tag}.fits"

    fits_path = image_path / fname

    # ------------------------------------------------------------
    # 1) Prepare data array
    # ------------------------------------------------------------
    # IMPORTANT:
    # histogram2d → density is (nx, ny)
    # FITS image convention → (ny, nx)
    data = np.asarray(dens.T, dtype=np.float32)

    # ------------------------------------------------------------
    # 2) Build WCS
    # ------------------------------------------------------------
    coord_mode = meta.get("coord_mode", "radec")

    w = WCS(naxis=2)

    if coord_mode == "radec":
        # World-coordinate system
        xmin, xmax, ymin, ymax = grid.extent
        nx, ny = data.shape[1], data.shape[0]

        cdelt1 = (xmax - xmin) / nx
        cdelt2 = (ymax - ymin) / ny

        w.wcs.crpix = [1.0, 1.0]
        w.wcs.cdelt = [cdelt1, cdelt2]
        w.wcs.crval = [xmin, ymin]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    else:
        # Pixel coordinate system (no sky projection)
        w.wcs.crpix = [1.0, 1.0]
        w.wcs.cdelt = [1.0, 1.0]
        w.wcs.crval = [0.0, 0.0]
        w.wcs.ctype = ["X", "Y"]

    header = w.to_header()

    # ------------------------------------------------------------
    # 3) Add metadata to header
    # ------------------------------------------------------------
    header["FIELD"] = (field_id, "Field identifier")
    if z is not None:
        header["REDSHIFT"] = (float(z), "Redshift slice")

    header["NRS"] = (meta.get("N_input", -1), "Number of RS galaxies")
    header["SMOOTH"] = (meta.get("smooth_sigma", 0.0), "Gaussian smoothing sigma (pix)")
    header["GLOB"] = (meta.get("globdensity", False), "Global density normalization")
    header["NORM"] = (meta.get("norm_mode", "none"), "Density normalization mode")

    header["BUNIT"] = ("arb.", "Arbitrary density unit")
    header["ORIGIN"] = ("ks4_rs pipeline", "Created by KS4 RS pipeline")

    # ------------------------------------------------------------
    # 4) Write FITS
    # ------------------------------------------------------------
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(fits_path, overwrite=True)

    return {
        "fits_path": fits_path,
        "header": header,
    }
