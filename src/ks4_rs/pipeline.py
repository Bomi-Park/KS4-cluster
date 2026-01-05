"""
KS4 Red-Sequence Cluster Detection Pipeline

This module orchestrates the full 7-step red-sequence cluster detection
pipeline for a single field and a single redshift slice.

Pipeline flow
-------------
1. Red-sequence selection
2. Density map construction
3. Density map visualization (optional)
4. Density map -> FITS
5. SExtractor execution
6. Cluster detection & member assignment
7. Diagnostic cluster images
"""

from __future__ import annotations

from typing import Any, Dict

# Step imports
from ks4_rs.steps.step1_find_rs import find_red_sequence
from ks4_rs.steps.step2_density_map import make_density_map
from ks4_rs.steps.step3_plot_density import plot_density_map
from ks4_rs.steps.step4_make_fits import make_density_fits
from ks4_rs.steps.step5_source_extractor import run_source_extractor
from ks4_rs.steps.step6_cluster_detect import detect_clusters
from ks4_rs.steps.step7_cluster_images import make_cluster_images


def run_pipeline(
    cat_KS4,
    galcat,
    image_path,
    info: Dict[str, Any],
    config: Dict[str, Any],
    *,
    figure: bool = False,
    calEFR: bool = False,
    ref_clusters=None,
) -> Dict[str, Any]:
    """
    Run the full KS4 red-sequence cluster detection pipeline.

    Parameters
    ----------
    cat_KS4 : astropy.table.Table
        Full KS4 catalog for the field.
    galcat : astropy.table.Table
        Galaxy-only catalog.
    image_path : str or pathlib.Path
        Directory where FITS and diagnostic images will be saved.
    info : dict
        Field / redshift metadata.
        Expected keys typically include:
            - 'field_id'
            - 'z'
            - 'zbin'
            - 'RS_mean'
            - 'RS_sigma'
    config : dict
        Pipeline configuration dictionary.
    figure : bool, optional
        If True, generate diagnostic plots (Step 3 & Step 7).
    calEFR : bool, optional
        Whether to compute effective field radius in Step 1.
    ref_clusters : optional
        Reference cluster catalog for overlay in Step 3.

    Returns
    -------
    result : dict
        Dictionary containing outputs from all pipeline steps.
        Keys:
            - 'rs'
            - 'density'
            - 'fits'
            - 'sex'
            - 'cluster'
    """

    # ------------------------------------------------------------
    # Step 1: Red-sequence selection
    # ------------------------------------------------------------
    rs_result = find_red_sequence(
        cat_KS4,
        galcat,
        info=info,
        config=config,
        figure=figure,
        calEFR=calEFR,
    )

    # If no RS galaxies, abort early
    if rs_result["RScand"] is None or len(rs_result["RScand"]) == 0:
        return {
            "rs": rs_result,
            "density": None,
            "fits": None,
            "sex": None,
            "cluster": None,
        }

    # ------------------------------------------------------------
    # Step 2: Density map
    # ------------------------------------------------------------
    density_result = make_density_map(
        rs_result["RScand"],
        (cat_KS4, galcat),
        info=info,
        config=config,
        figure=False,
    )

    # ------------------------------------------------------------
    # Step 3: Density map visualization (optional)
    # ------------------------------------------------------------
    plot_density_map(
        density_result,
        ref_clusters=ref_clusters,
        info=info,
        config=config,
        figure=figure,
    )

    # ------------------------------------------------------------
    # Step 4: Density FITS
    # ------------------------------------------------------------
    fits_result = make_density_fits(
        density_result,
        info=info,
        image_path=image_path,
        config=config,
        figure=False,
    )

    # ------------------------------------------------------------
    # Step 5: SExtractor
    # ------------------------------------------------------------
    sex_result = run_source_extractor(
        fits_result,
        density_result,
        info=info,
        config=config,
    )

    # ------------------------------------------------------------
    # Step 6: Cluster detection
    # ------------------------------------------------------------
    cluster_result = detect_clusters(
        sex_result,
        fits_result,
        density_result,
        rs_result,
        info=info,
        config=config,
    )

    # ------------------------------------------------------------
    # Step 7: Diagnostic cluster images (optional)
    # ------------------------------------------------------------
    if figure:
        make_cluster_images(
            cluster_result,
            all_results=[
                rs_result,
                density_result,
                fits_result,
                sex_result,
            ],
            info=info,
            config=config,
        )

    # ------------------------------------------------------------
    # Package outputs
    # ------------------------------------------------------------
    return {
        "rs": rs_result,
        "density": density_result,
        "fits": fits_result,
        "sex": sex_result,
        "cluster": cluster_result,
    }
