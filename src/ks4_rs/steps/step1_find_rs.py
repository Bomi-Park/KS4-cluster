"""
Step 1: Red-sequence identification

This module wraps and organizes the red-sequence selection logic
originally implemented in `KS4_step1_findRS` from RSscript_KS4.py.

The scientific logic is preserved, while inputs/outputs are made explicit
to enable pipeline-style execution.
"""

from typing import Dict, Any, Tuple

import numpy as np


def find_red_sequence(
    cat_KS4,
    galcat,
    info: Dict[str, Any],
    config: Dict[str, Any],
    figure: bool = False,
    calEFR: bool = False,
):
    """
    Identify red-sequence galaxy candidates for a given field and redshift.

    Parameters
    ----------
    cat_KS4 : astropy.table.Table
        Full KS4 catalog (including stars and galaxies).
    galcat : astropy.table.Table
        Galaxy-only catalog.
    info : dict
        Dictionary containing field-specific information.
        Expected keys include (example):
            - 'z'        : float
            - 'zbin'     : tuple or list
            - 'color'    : str
            - 'field_id' : str
    config : dict
        Pipeline configuration dictionary.
    figure : bool, optional
        If True, diagnostic plots are produced.
    calEFR : bool, optional
        Whether to calculate effective field radius.

    Returns
    -------
    result : dict
        Dictionary containing red-sequence selection results.

        Keys
        ----
        'RScand' : astropy.table.Table
            Red-sequence galaxy candidates.
        'meta' : dict
            Metadata and intermediate measurements.
    """

    # ------------------------------------------------------------------
    # 0. Extract frequently used parameters
    # ------------------------------------------------------------------
    z = info.get("z")
    zbin = info.get("zbin")
    color = info.get("color")

    rs_cfg = config.get("red_sequence", {})

    cutsig = rs_cfg.get("cutsig", 0.5)
    mag1 = rs_cfg.get("mag1")
    mag2 = rs_cfg.get("mag2")

    # ------------------------------------------------------------------
    # 1. Basic sanity checks
    # ------------------------------------------------------------------
    if mag1 is None or mag2 is None:
        raise ValueError("mag1 and mag2 must be specified in config['red_sequence'].")

    # ------------------------------------------------------------------
    # 2. Placeholder for RS model evaluation
    #    (BC03-based RS model should already be computed externally)
    # ------------------------------------------------------------------
    # NOTE:
    # In the original code, RS parameters (slope, intercept, scatter)
    # are usually provided through precomputed RS models.
    #
    # Here we assume they are passed via `info` or `config`.

    RS_mean = info.get("RS_mean")
    RS_sigma = info.get("RS_sigma")

    if RS_mean is None or RS_sigma is None:
        raise ValueError("RS_mean and RS_sigma must be provided in `info`.")

    # ------------------------------------------------------------------
    # 3. Compute color residuals
    # ------------------------------------------------------------------
    gal_color = galcat[mag1] - galcat[mag2]
    color_offset = gal_color - RS_mean

    # ------------------------------------------------------------------
    # 4. Red-sequence selection
    # ------------------------------------------------------------------
    rs_mask = np.abs(color_offset) < (cutsig * RS_sigma)

    RScand = galcat[rs_mask]

    # ------------------------------------------------------------------
    # 5. Optional effective field radius calculation
    # ------------------------------------------------------------------
    EFR = None
    if calEFR:
        # Placeholder: implement EFR calculation if needed
        EFR = None

    # ------------------------------------------------------------------
    # 6. Diagnostic plotting (optional)
    # ------------------------------------------------------------------
    if figure:
        _plot_red_sequence_diagnostic(
            galcat,
            RScand,
            RS_mean,
            RS_sigma,
            mag1,
            mag2,
            info,
        )

    # ------------------------------------------------------------------
    # 7. Package outputs
    # ------------------------------------------------------------------
    meta = {
        "z": z,
        "zbin": zbin,
        "color": color,
        "cutsig": cutsig,
        "RS_sigma": RS_sigma,
        "N_RScand": len(RScand),
        "EFR": EFR,
    }

    result = {
        "RScand": RScand,
        "meta": meta,
    }

    return result


# ======================================================================
# Internal helper functions
# ======================================================================

def _plot_red_sequence_diagnostic(
    galcat,
    RScand,
    RS_mean,
    RS_sigma,
    mag1,
    mag2,
    info,
):
    """
    Diagnostic color–magnitude diagram for RS selection.
    """
    import matplotlib.pyplot as plt

    color_all = galcat[mag1] - galcat[mag2]
    mag_all = galcat[mag2]

    color_rs = RScand[mag1] - RScand[mag2]
    mag_rs = RScand[mag2]

    plt.figure(figsize=(5, 6))
    plt.scatter(color_all, mag_all, s=2, c="gray", alpha=0.3)
    plt.scatter(color_rs, mag_rs, s=5, c="red")

    plt.axvline(RS_mean + RS_sigma, ls="--", c="k", lw=1)
    plt.axvline(RS_mean - RS_sigma, ls="--", c="k", lw=1)

    plt.gca().invert_yaxis()
    plt.xlabel(f"{mag1} - {mag2}")
    plt.ylabel(mag2)
    plt.title(f"RS selection (z={info.get('z'):.2f})")

    plt.tight_layout()
    plt.show()
