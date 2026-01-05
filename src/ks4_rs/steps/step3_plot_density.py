"""
Step 3: Density map visualization (diagnostic)

This step provides visualization utilities for the density map
constructed in Step 2. It is intended for *diagnostic and validation*
purposes only and does not modify the density map itself.

Design principles
-----------------
- Optional (can be turned off entirely)
- No file I/O by default
- Safe to skip in batch / multiprocessing runs
"""

from typing import Any, Dict, Optional

import numpy as np


def plot_density_map(
    density_result: Dict[str, Any],
    ref_clusters: Optional[Any],
    info: Dict[str, Any],
    config: Dict[str, Any],
    figure: bool = False,
) -> Dict[str, Any]:
    """
    Plot density map and (optionally) reference clusters.

    Parameters
    ----------
    density_result : dict
        Output from step2_density_map.make_density_map.
        Must contain:
            - 'density_map' : 2D numpy array
            - 'grid'        : GridSpec
            - 'meta'        : dict
    ref_clusters : optional
        Reference cluster catalog (e.g., known clusters for validation).
        Expected to have RA/Dec or X/Y columns.
        If None, reference overlay is skipped.
    info : dict
        Field metadata (field_id, z, etc.).
    config : dict
        Configuration dictionary.
    figure : bool
        If False, this function returns immediately without plotting.

    Returns
    -------
    density_result : dict
        Unmodified input dictionary (pass-through).
    """

    if not figure:
        return density_result

    density = density_result.get("density_map")
    grid = density_result.get("grid")
    meta = density_result.get("meta", {})

    if density is None or grid is None:
        # Nothing to plot
        return density_result

    plot_cfg = config.get("plot_density", {})

    coord_mode = meta.get("coord_mode", "radec")
    vmin = plot_cfg.get("vmin", None)
    vmax = plot_cfg.get("vmax", None)
    cmap = plot_cfg.get("cmap", "viridis")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=plot_cfg.get("figsize", (6, 5)))

    # histogram2d convention: density is (nx, ny) → transpose for display
    im = plt.imshow(
        density.T,
        origin="lower",
        extent=grid.extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, label="Density (a.u.)")

    # ------------------------------------------------------------
    # Overlay reference clusters (optional)
    # ------------------------------------------------------------
    if ref_clusters is not None:
        try:
            if coord_mode == "radec":
                xref = ref_clusters[plot_cfg.get("ra_col", "RA")]
                yref = ref_clusters[plot_cfg.get("dec_col", "DEC")]
            else:
                xref = ref_clusters[plot_cfg.get("x_col", "X")]
                yref = ref_clusters[plot_cfg.get("y_col", "Y")]

            plt.scatter(
                xref,
                yref,
                s=plot_cfg.get("ref_size", 40),
                facecolors="none",
                edgecolors=plot_cfg.get("ref_color", "red"),
                linewidths=plot_cfg.get("ref_lw", 1.5),
                label="Reference clusters",
            )
            plt.legend(loc="upper right")

        except Exception:
            # Fail silently: plotting must never break pipeline
            pass

    # ------------------------------------------------------------
    # Labels / title
    # ------------------------------------------------------------
    if coord_mode == "radec":
        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")
    else:
        plt.xlabel("X")
        plt.ylabel("Y")

    field_id = info.get("field_id", "")
    z = info.get("z", None)

    title = "Density map"
    if field_id:
        title += f" | {field_id}"
    if z is not None:
        title += f" | z={z:.3f}"

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return density_result
