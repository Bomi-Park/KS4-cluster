#!/usr/bin/env python
"""
scripts/run_single_field.py

Run the KS4 RS pipeline for ONE field across MANY redshift slices.

- Loads YAML config
- Uses a user-provided loader(field_id) to load catalogs once
- Uses a user-provided rs_model_lookup(z, info, config) to get RS_mean/RS_sigma per z
- Iterates over z_list (from CLI args)
- Writes outputs under: <outdir>/<field_id>/z<z>/

Notes
-----
- This script runs SERIAL by default (safe, debuggable).
- For multiprocessing across z, use ks4_rs.runners.run_jobs_multiprocess
  (and make sure loader/lookup are picklable and figure=False).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ks4_rs import run_pipeline, load_config
from ks4_rs.utils import nearest_value


# =============================================================================
# USER IMPLEMENTATIONS (fill these)
# =============================================================================

def loader(field_id: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load catalogs for a field ONCE.

    Returns
    -------
    cat_KS4 : astropy.table.Table
    galcat  : astropy.table.Table
    info    : dict (field-level metadata; can include RA/Dec bounds)
    """
    raise NotImplementedError("Fill loader(field_id) with your KS4 catalog loading logic.")


def rs_model_lookup(z: float, info: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Lookup RS_mean and RS_sigma for this z (from BC03 RS model table).
    """
    raise NotImplementedError("Fill rs_model_lookup(z, info, config) using your BC03 RS model products.")


# =============================================================================
# CLI
# =============================================================================

def _parse_z_list(z_text: str) -> List[float]:
    """
    Parse z list from CLI input.

    Supported:
    - comma list: "0.2,0.25,0.3"
    - range: "0.2:0.8:0.05" meaning start:stop:step (stop inclusive-ish)
    """
    s = z_text.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("zlist range must be 'start:stop:step' e.g. 0.2:0.8:0.05")
        a, b, dz = map(float, parts)
        # inclusive-ish range
        n = int((b - a) / dz) + 1
        return [a + i * dz for i in range(max(n, 0))]
    # comma list
    return [float(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KS4 RS pipeline for one field across many z slices.")
    p.add_argument("--field", required=True, help="Field identifier (e.g., KS4_0123)")
    p.add_argument(
        "--zlist",
        required=True,
        help="Redshift list. Examples: '0.2,0.25,0.3' or '0.2:0.8:0.05'",
    )
    p.add_argument("--config", default="configs/pipeline_default.yaml", help="YAML config path")
    p.add_argument("--outdir", default="outputs", help="Base output directory")
    p.add_argument("--figure", action="store_true", help="Enable diagnostic plots/images (slower)")
    p.add_argument("--calEFR", action="store_true", help="Compute EFR in Step 1 (if implemented)")
    p.add_argument("--resume", action="store_true", help="Skip z runs if output FITS already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    field_id = args.field
    z_list = _parse_z_list(args.zlist)

    # Load catalogs once
    cat_KS4, galcat, base_info = loader(field_id)
    base_info = dict(base_info)
    base_info["field_id"] = str(field_id)

    base_out = Path(args.outdir) / str(field_id)
    base_out.mkdir(parents=True, exist_ok=True)

    # Run per z
    for z in z_list:
        z = float(z)

        out_dir = base_out / f"z{z:.3f}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Basic resume logic: if density fits already exists, skip
        if args.resume:
            # matches step4 file naming: <field>_density_z0.xxx.fits by default
            expected = out_dir / f"{field_id}_{cfg.get('fits', {}).get('tag', 'density')}_z{z:.3f}.fits"
            if expected.exists():
                print(f"[SKIP] field={field_id} z={z:.3f} (found {expected.name})")
                continue

        info = dict(base_info)
        info["z"] = z

        # RS model lookup
        RS_mean, RS_sigma = rs_model_lookup(z, info, cfg)
        info["RS_mean"] = float(RS_mean)
        info["RS_sigma"] = float(RS_sigma)

        # Run pipeline
        try:
            result = run_pipeline(
                cat_KS4=cat_KS4,
                galcat=galcat,
                image_path=out_dir,
                info=info,
                config=cfg,
                figure=bool(args.figure),
                calEFR=bool(args.calEFR),
                ref_clusters=None,
            )

            n_rs = len(result["rs"]["RScand"]) if result.get("rs") and result["rs"].get("RScand") is not None else 0
            n_cl = len(result["cluster"]["clusters"]) if result.get("cluster") and result["cluster"].get("clusters") is not None else 0

            print(f"[DONE] field={field_id} z={z:.3f} | N_RS={n_rs} | N_cluster={n_cl}")

        except Exception as e:
            print(f"[FAIL] field={field_id} z={z:.3f} | {e}")


if __name__ == "__main__":
    main()
