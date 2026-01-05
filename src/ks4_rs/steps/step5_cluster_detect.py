"""
Step 5: Run SExtractor on density FITS

This step executes SExtractor (Source Extractor) on the density map FITS image
created in Step 4 and returns detection catalogs.

Requirements
------------
- SExtractor installed and accessible (sex or source-extractor)
- Configuration files prepared (default.sex, default.param, default.conv, etc.)

Design principles
-----------------
- Keep external tool invocation isolated here
- Fail loudly with helpful stderr output
- Return file paths for downstream steps
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import shlex
import subprocess


def run_source_extractor(
    fits_result: Dict[str, Any],
    density_result: Dict[str, Any],
    info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run SExtractor on the density FITS image.

    Parameters
    ----------
    fits_result : dict
        Output from step4_make_fits.make_density_fits.
        Must contain 'fits_path'.
    density_result : dict
        Output from step2_density_map.make_density_map (for metadata).
    info : dict
        Field metadata (field_id, z, etc.).
    config : dict
        Configuration dictionary. Expected: config['sextractor'].

    Returns
    -------
    result : dict
        Keys
        ----
        'cmd' : list[str]
            Executed command.
        'catalog_path' : Path
            Output catalog path (.cat or .fits depending on config).
        'segmentation_path' : Path or None
            Optional segmentation map.
        'background_path' : Path or None
            Optional background map.
        'log_path' : Path or None
            Optional log file containing stdout/stderr.
    """
    sex_cfg = config.get("sextractor", {})

    fits_path = Path(fits_result.get("fits_path", ""))
    if not fits_path.exists():
        raise FileNotFoundError(f"Input FITS not found: {fits_path}")

    # ------------------------------------------------------------
    # 0) Locate executable
    # ------------------------------------------------------------
    exe = sex_cfg.get("exe", "sex")  # common: 'sex' or 'source-extractor'
    # We'll just call it; if missing, subprocess will error.

    # ------------------------------------------------------------
    # 1) Output directory and filenames
    # ------------------------------------------------------------
    out_dir = Path(sex_cfg.get("out_dir", fits_path.parent))
    out_dir.mkdir(parents=True, exist_ok=True)

    field_id = info.get("field_id", fits_path.stem)
    z = info.get("z", None)

    tag = sex_cfg.get("tag", "sex")
    suffix = f"_z{z:.3f}" if z is not None else ""
    catalog_ext = sex_cfg.get("catalog_ext", ".cat")

    catalog_path = out_dir / f"{field_id}_{tag}{suffix}{catalog_ext}"

    # Optional outputs
    make_seg = bool(sex_cfg.get("make_segmentation", False))
    make_bkg = bool(sex_cfg.get("make_background", False))

    segmentation_path = out_dir / f"{field_id}_{tag}{suffix}_seg.fits" if make_seg else None
    background_path = out_dir / f"{field_id}_{tag}{suffix}_bkg.fits" if make_bkg else None

    # Optional logging
    log_path = None
    if sex_cfg.get("write_log", True):
        log_path = out_dir / f"{field_id}_{tag}{suffix}.sex.log"

    # ------------------------------------------------------------
    # 2) SExtractor config files
    # ------------------------------------------------------------
    # You can provide either:
    # - a single .sex config file, plus overrides via -PARAMETERS_NAME etc.
    # or
    # - set all via command-line.
    sex_config = sex_cfg.get("config", None)           # e.g., "configs/sex/default.sex"
    params_name = sex_cfg.get("parameters", None)      # e.g., "configs/sex/default.param"
    conv_name = sex_cfg.get("filter", None)            # e.g., "configs/sex/default.conv"
    nnw_name = sex_cfg.get("nnw", None)                # e.g., "configs/sex/default.nnw"

    # ------------------------------------------------------------
    # 3) Build command
    # ------------------------------------------------------------
    cmd: List[str] = [exe, str(fits_path)]

    if sex_config is not None:
        cmd += ["-c", str(sex_config)]

    # Core output
    cmd += ["-CATALOG_NAME", str(catalog_path)]

    # Optional output images (segmentation/background)
    # SExtractor uses -CHECKIMAGE_TYPE and -CHECKIMAGE_NAME
    check_types = []
    check_names = []
    if segmentation_path is not None:
        check_types.append("SEGMENTATION")
        check_names.append(str(segmentation_path))
    if background_path is not None:
        check_types.append("BACKGROUND")
        check_names.append(str(background_path))

    if check_types:
        cmd += ["-CHECKIMAGE_TYPE", ",".join(check_types)]
        cmd += ["-CHECKIMAGE_NAME", ",".join(check_names)]

    # Attach auxiliary config files if provided
    if params_name is not None:
        cmd += ["-PARAMETERS_NAME", str(params_name)]
    if conv_name is not None:
        cmd += ["-FILTER_NAME", str(conv_name)]
    if nnw_name is not None:
        cmd += ["-STARNNW_NAME", str(nnw_name)]

    # Threshold-related overrides (common tuning knobs)
    # These are examples; override only if present.
    for key, flag in [
        ("detect_minarea", "-DETECT_MINAREA"),
        ("detect_thresh", "-DETECT_THRESH"),
        ("analysis_thresh", "-ANALYSIS_THRESH"),
        ("deblend_nthresh", "-DEBLEND_NTHRESH"),
        ("deblend_mincount", "-DEBLEND_MINCONT"),
        ("clean", "-CLEAN"),
        ("clean_param", "-CLEAN_PARAM"),
        ("back_size", "-BACK_SIZE"),
        ("back_filtersize", "-BACK_FILTERSIZE"),
        ("filter", "-FILTER"),
    ]:
        if key in sex_cfg and sex_cfg[key] is not None:
            cmd += [flag, str(sex_cfg[key])]

    # Allow arbitrary extra args
    extra_args = sex_cfg.get("extra_args", None)
    if extra_args:
        # extra_args can be a string ("-VERBOSE_TYPE FULL") or list
        if isinstance(extra_args, str):
            cmd += shlex.split(extra_args)
        elif isinstance(extra_args, (list, tuple)):
            cmd += [str(x) for x in extra_args]
        else:
            raise ValueError("sextractor.extra_args must be str or list/tuple")

    # ------------------------------------------------------------
    # 4) Execute
    # ------------------------------------------------------------
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"SExtractor executable not found: '{exe}'. "
            f"Install SExtractor or set config['sextractor']['exe'] correctly."
        ) from e

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if log_path is not None:
        log_path.write_text(
            f"CMD: {' '.join(cmd)}\n\n--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}\n",
            encoding="utf-8",
        )

    if proc.returncode != 0:
        msg = (
            "SExtractor failed.\n"
            f"Return code: {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- STDERR ---\n{stderr.strip()}\n"
        )
        raise RuntimeError(msg)

    # Quick existence check
    if not catalog_path.exists():
        raise RuntimeError(
            "SExtractor finished without errors but catalog file was not created.\n"
            f"Expected: {catalog_path}\n"
            f"Command: {' '.join(cmd)}\n"
        )

    return {
        "cmd": cmd,
        "catalog_path": catalog_path,
        "segmentation_path": segmentation_path if (segmentation_path and segmentation_path.exists()) else None,
        "background_path": background_path if (background_path and background_path.exists()) else None,
        "log_path": log_path if (log_path and log_path.exists()) else None,
        "stdout": stdout,
        "stderr": stderr,
    }
