#!/usr/bin/env python
"""
scripts/run_pipeline.py

CLI runner for the KS4 RS pipeline.

What this script does
---------------------
- Loads a YAML config (default: configs/pipeline_default.yaml)
- Loads one field catalog + galaxy catalog (YOU implement loader())
- Looks up RS model values (RS_mean, RS_sigma) for the requested redshift (YOU implement lookup())
- Runs the full pipeline (Step 1–7)
- Prints a compact summary and writes outputs under an output directory

Why we keep loader/lookup as placeholders
-----------------------------------------
Your KS4 catalogs + BC03 RS model live in your environment (server paths, formats).
Hard-coding them would break portability and is not GitHub-friendly.

You only need to fill TWO small functions:
- loader(field_id) -> cat_KS4, galcat, info
- rs_model_lookup(z, info, config) -> (RS_mean, RS_sigma)

Then the rest works.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

from ks4_rs import run_pipeline, load_config


# =============================================================================
# USER IMPLEMENTATIONS (fill these)
# =============================================================================

def loader(field_id: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load KS4 catalogs for a given field.

    Returns
    -------
    cat_KS4 : astropy.table.Table
        Full KS4 catalog for the field.
    galcat : astropy.table.Table
        Galaxy-only catalog.
    info : dict
        Field metadata (should include at least 'field_id', and ideally RA/Dec bounds).
    """
    # Example (replace with your real logic):
    #
    # from astropy.table import Table
    # cat_KS4 = Table.read(f"/data/KS4/{field_id}.fits")
    # galcat = cat_KS4[cat_KS4["CLASS_STAR"] < 0.5]
    # info = {"field_id": field_id}
    # return cat_KS4, galcat, info

    raise NotImplementedError("Fill loader(field_id) with your KS4 catalog loading logic.")


def rs_model_lookup(z: float, info: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Lookup or compute RS_mean and RS_sigma for the given redshift.

    In your original pipeline this comes from BC03-based RS model table.

    Returns
    -------
    RS_mean : float
        Mean RS color at this z (in the same definition as mag1-mag2).
    RS_sigma : float
        Scatter (sigma) of RS color at this z.
    """
    # Example dummy:
    # RS_mean = 1.0
    # RS_sigma = 0.1
    # return RS_mean, RS_sigma

    raise NotImplementedError("Fill rs_model_lookup(z, info, config) using your BC03 RS model products.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KS4 RS pipeline for one field and one redshift.")
    p.add_argument("--field", required=True, help="Field identifier (e.g., KS4_0123)")
    p.add_argument("--z", required=True, type=float, help="Redshift slice (e.g., 0.35)")
    p.add_argument("--config", default="configs/pipeline_default.yaml", help="YAML config path")
    p.add_argument("--outdir", default="outputs", help="Base output directory")
    p.add_argument("--figure", action="store_true", help="Enable diagnostic plots/images (slower)")
    p.add_argument("--calEFR", action="store_true", help="Compute EFR in Step 1 (if implemented)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    field_id = args.field
    z = float(args.z)

    # Output dir structure: <outdir>/<field>/z<z>
    out_dir = Path(args.outdir) / str(field_id) / f"z{z:.3f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load catalogs
    # ------------------------------------------------------------
    cat_KS4, galcat, info = loader(field_id)
    info = dict(info)
    info["field_id"] = str(field_id)
    info["z"] = z

    # ------------------------------------------------------------
    # RS model lookup
    # ------------------------------------------------------------
    RS_mean, RS_sigma = rs_model_lookup(z, info, cfg)
    info["RS_mean"] = float(RS_mean)
    info["RS_sigma"] = float(RS_sigma)

    # ------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------
    result = run_pipeline(
        cat_KS4=cat_KS4,
        galcat=galcat,
        image_path=out_dir,     # where FITS and diagnostic images go
        info=info,
        config=cfg,
        figure=bool(args.figure),
        calEFR=bool(args.calEFR),
        ref_clusters=None,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    n_rs = 0
    if result.get("rs") and result["rs"].get("RScand") is not None:
        n_rs = len(result["rs"]["RScand"])

    n_cl = 0
    n_mem = 0
    if result.get("cluster"):
        if result["cluster"].get("clusters") is not None:
            n_cl = len(result["cluster"]["clusters"])
        if result["cluster"].get("members") is not None:
            n_mem = len(result["cluster"]["members"])

    print(f"[DONE] field={field_id} z={z:.3f}")
    print(f"  N_RS     : {n_rs}")
    print(f"  N_cluster: {n_cl}")
    print(f"  N_member : {n_mem}")

    if result.get("fits"):
        print(f"  FITS     : {result['fits']['fits_path']}")
    if result.get("sex"):
        print(f"  SEx cat  : {result['sex']['catalog_path']}")
        if result["sex"].get("log_path"):
            print(f"  SEx log  : {result['sex']['log_path']}")


if __name__ == "__main__":
    main()
