"""
Step 6: Cluster candidate detection and member assignment

This step converts SExtractor detections on the density map into
physical cluster candidates by assigning red-sequence galaxy members
and measuring basic cluster properties.

This module is the scientific core of the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


def detect_clusters(
    sex_result: Dict[str, Any],
    fits_result: Dict[str, Any],
    density_result: Dict[str, Any],
    rs_result: Dict[str, Any],
    info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect cluster candidates and assign member galaxies.

    Parameters
    ----------
    sex_result : dict
        Output from step5_source_extractor.run_source_extractor.
    fits_result : dict
        Output from step4_make_fits.make_density_fits.
    density_result : dict
        Output from step2_density_map.make_density_map.
    rs_result : dict
        Output from step1_find_rs.find_red_sequence.
    info : dict
        Field metadata (field_id, z, etc.).
    config : dict
        Configuration dictionary.

    Returns
    -------
    result : dict
        Keys
        ----
        'clusters' : astropy.table.Table
            Cluster candidate catalog.
        'members' : astropy.table.Table
            Member galaxy catalog.
    """

    # ------------------------------------------------------------------
    # 0. Load SExtractor catalog
    # ------------------------------------------------------------------
    cat_path = sex_result.get("catalog_path", None)
    if cat_path is None:
        raise ValueError("sex_result must contain 'catalog_path'.")

    sexcat = _load_sextractor_catalog(cat_path)

    if len(sexcat) == 0:
        return {
            "clusters": Table(),
            "members": Table(),
        }

    # ------------------------------------------------------------------
    # 1. Convert detection positions to sky coordinates
    # ------------------------------------------------------------------
    # NOTE:
    # SExtractor outputs positions in pixel coordinates (X_IMAGE, Y_IMAGE)
    # We convert them back to RA/Dec using the WCS from the FITS header.
    from astropy.wcs import WCS

    wcs = WCS(fits_result["header"])

    x = sexcat["X_IMAGE"]
    y = sexcat["Y_IMAGE"]

    ra, dec = wcs.wcs_pix2world(x, y, 0)

    sexcat["RA"] = ra
    sexcat["DEC"] = dec

    det_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # ------------------------------------------------------------------
    # 2. Prepare RS galaxy coordinates
    # ------------------------------------------------------------------
    RScand = rs_result.get("RScand")
    if RScand is None or len(RScand) == 0:
        return {
            "clusters": Table(),
            "members": Table(),
        }

    gal_coords = SkyCoord(
        ra=RScand["RA"] * u.deg,
        dec=RScand["DEC"] * u.deg,
    )

    # ------------------------------------------------------------------
    # 3. Define matching radius
    # ------------------------------------------------------------------
    clt_cfg = config.get("cluster_detect", {})

    # Radius for member assignment
    r_member = clt_cfg.get("member_radius_arcmin", 1.0) * u.arcmin

    # Minimum number of RS members to define a cluster
    min_members = int(clt_cfg.get("min_members", 5))

    # ------------------------------------------------------------------
    # 4. Assign members to each detection
    # ------------------------------------------------------------------
    cluster_rows = []
    member_rows = []

    for i, coord in enumerate(det_coords):
        sep = coord.separation(gal_coords)

        mem_mask = sep < r_member
        if mem_mask.sum() < min_members:
            continue

        members = RScand[mem_mask]

        # --------------------------------------------------------------
        # 4-1. Basic cluster properties
        # --------------------------------------------------------------
        ra_cen = np.mean(members["RA"])
        dec_cen = np.mean(members["DEC"])

        z = info.get("z")

        richness = len(members)

        # You can add:
        # - weighted centroid
        # - luminosity-weighted richness
        # - radial profile fits
        # here, based on your original code.

        cluster_rows.append(
            {
                "CLUSTER_ID": f"{info.get('field_id','F')}_{i:04d}",
                "RA": ra_cen,
                "DEC": dec_cen,
                "Z": z,
                "N_MEMBER": richness,
            }
        )

        # --------------------------------------------------------------
        # 4-2. Member table
        # --------------------------------------------------------------
        for mem in members:
            member_rows.append(
                {
                    "CLUSTER_ID": f"{info.get('field_id','F')}_{i:04d}",
                    "OBJID": mem.get("OBJID", -1),
                    "RA": mem["RA"],
                    "DEC": mem["DEC"],
                    "Z": z,
                }
            )

    # ------------------------------------------------------------------
    # 5. Construct output tables
    # ------------------------------------------------------------------
    clusters = Table(rows=cluster_rows)
    members = Table(rows=member_rows)

    return {
        "clusters": clusters,
        "members": members,
    }


# ======================================================================
# Helper functions
# ======================================================================

def _load_sextractor_catalog(path) -> Table:
    """
    Load a SExtractor catalog.

    Supports:
    - ASCII_HEAD / ASCII
    - FITS_LDAC / FITS

    Returns
    -------
    astropy.table.Table
    """
    path = str(path)

    try:
        if path.lower().endswith(".fits"):
            return Table.read(path)
        else:
            return Table.read(path, format="ascii")
    except Exception as e:
        raise RuntimeError(f"Failed to read SExtractor catalog: {path}") from e
