"""
runners.py

Execution helpers for running the KS4 RS pipeline across:
- one field over many redshift slices
- many fields at one redshift slice
- optionally with multiprocessing

This module intentionally does NOT implement I/O specifics for KS4 catalogs.
Instead, it expects a user-provided `loader(field, z)` callable that returns:
    cat_KS4, galcat, image_path, info

and optionally a `saver(result, field, z, out_dir)` callable to persist outputs.

Typical usage
-------------
from ks4_rs.runners import run_single_field_all_z
from ks4_rs.config import load_config

config = load_config("configs/pipeline_default.yaml")

def loader(field, z):
    # load tables + build info dict
    return cat_KS4, galcat, out_dir, info

run_single_field_all_z("KS4_field_001", z_list, loader, config, out_dir="outputs")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import traceback

from ks4_rs.pipeline import run_pipeline


LoaderFn = Callable[[Any, float], Tuple[Any, Any, Union[str, Path], Dict[str, Any]]]
SaverFn = Callable[[Dict[str, Any], Any, float, Union[str, Path]], None]


@dataclass
class RunRecord:
    field: Any
    z: float
    ok: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Small, JSON-safe summary for logging."""
    out = {
        "has_rs": bool(result.get("rs") and result["rs"].get("RScand") is not None),
        "n_rs": int(len(result["rs"]["RScand"])) if result.get("rs") and result["rs"].get("RScand") is not None else 0,
        "has_cluster": bool(result.get("cluster") and result["cluster"].get("clusters") is not None),
        "n_clusters": int(len(result["cluster"]["clusters"])) if result.get("cluster") and result["cluster"].get("clusters") is not None else 0,
        "n_members": int(len(result["cluster"]["members"])) if result.get("cluster") and result["cluster"].get("members") is not None else 0,
    }
    return out


def _write_run_record(out_dir: Path, record: RunRecord, fname: str = "run_log.jsonl") -> None:
    """Append one record as JSONL."""
    path = out_dir / fname
    obj = {
        "field": str(record.field),
        "z": float(record.z),
        "ok": bool(record.ok),
        "error": record.error,
        "traceback": record.traceback,
        "result_summary": record.result_summary,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _done_marker_path(out_dir: Path, field: Any, z: float) -> Path:
    safe_field = str(field).replace("/", "_").replace(" ", "_")
    return out_dir / "done" / f"{safe_field}_z{z:.3f}.done"


def _is_done(out_dir: Path, field: Any, z: float) -> bool:
    return _done_marker_path(out_dir, field, z).exists()


def _mark_done(out_dir: Path, field: Any, z: float, summary: Dict[str, Any]) -> None:
    done_dir = _ensure_dir(out_dir / "done")
    p = _done_marker_path(out_dir, field, z)
    p.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


def run_single_field_all_z(
    field: Any,
    z_list: Sequence[float],
    loader: LoaderFn,
    config: Dict[str, Any],
    *,
    out_dir: Union[str, Path] = "outputs",
    figure: bool = False,
    calEFR: bool = False,
    ref_clusters: Any = None,
    saver: Optional[SaverFn] = None,
    resume: bool = True,
    log_jsonl: bool = True,
) -> List[RunRecord]:
    """
    Run pipeline for one field across multiple redshift slices.

    Parameters
    ----------
    field : any
        Field identifier.
    z_list : sequence of float
        Redshift slices.
    loader : callable(field, z) -> (cat_KS4, galcat, image_path, info)
        User-provided loader.
    config : dict
        Pipeline config.
    out_dir : path
        Output directory for logs and done markers.
    figure : bool
        If True, generate plots/images (slower).
    calEFR : bool
        Passed to step1.
    ref_clusters : optional
        Reference clusters for overlays.
    saver : callable(result, field, z, out_dir), optional
        If provided, called after each successful run to persist outputs.
    resume : bool
        If True, skip (field,z) that already has a done marker.
    log_jsonl : bool
        If True, write JSONL run logs.

    Returns
    -------
    records : list[RunRecord]
    """
    out_dir = _ensure_dir(out_dir)

    records: List[RunRecord] = []
    for z in z_list:
        if resume and _is_done(out_dir, field, z):
            rec = RunRecord(field=field, z=float(z), ok=True, result_summary={"skipped": True})
            records.append(rec)
            if log_jsonl:
                _write_run_record(out_dir, rec)
            continue

        try:
            cat_KS4, galcat, image_path, info = loader(field, float(z))
            # Ensure required info fields are set
            info = dict(info)
            info.setdefault("field_id", str(field))
            info["z"] = float(z)

            result = run_pipeline(
                cat_KS4,
                galcat,
                image_path=image_path,
                info=info,
                config=config,
                figure=figure,
                calEFR=calEFR,
                ref_clusters=ref_clusters,
            )

            summary = _default_summary(result)

            if saver is not None:
                saver(result, field, float(z), out_dir)

            _mark_done(out_dir, field, float(z), summary)

            rec = RunRecord(field=field, z=float(z), ok=True, result_summary=summary)
            records.append(rec)
            if log_jsonl:
                _write_run_record(out_dir, rec)

        except Exception as e:
            tb = traceback.format_exc()
            rec = RunRecord(field=field, z=float(z), ok=False, error=str(e), traceback=tb)
            records.append(rec)
            if log_jsonl:
                _write_run_record(out_dir, rec)

    return records


def run_single_z_all_fields(
    z: float,
    field_list: Sequence[Any],
    loader: LoaderFn,
    config: Dict[str, Any],
    *,
    out_dir: Union[str, Path] = "outputs",
    figure: bool = False,
    calEFR: bool = False,
    ref_clusters: Any = None,
    saver: Optional[SaverFn] = None,
    resume: bool = True,
    log_jsonl: bool = True,
) -> List[RunRecord]:
    """
    Run pipeline for one redshift slice across multiple fields.
    (Serial execution)
    """
    out_dir = _ensure_dir(out_dir)

    records: List[RunRecord] = []
    for field in field_list:
        if resume and _is_done(out_dir, field, float(z)):
            rec = RunRecord(field=field, z=float(z), ok=True, result_summary={"skipped": True})
            records.append(rec)
            if log_jsonl:
                _write_run_record(out_dir, rec)
            continue

        try:
            cat_KS4, galcat, image_path, info = loader(field, float(z))
            info = dict(info)
            info.setdefault("field_id", str(field))
            info["z"] = float(z)

            result = run_pipeline(
                cat_KS4,
                galcat,
                image_path=image_path,
                info=info,
                config=config,
                figure=figure,
                calEFR=calEFR,
                ref_clusters=ref_clusters,
            )

            summary = _default_summary(result)

            if saver is not None:
                saver(result, field, float(z), out_dir)

            _mark_done(out_dir, field, float(z), summary)

            rec = RunRecord(field=field, z=float(z), ok=True, result_summary=summary)
            records.append(rec)
            if log_jsonl:
                _write_run_record(out_dir, rec)

        except Exception as e:
            tb = traceback.format_exc()
            rec = RunRecord(field=field, z=float(z), ok=False, error=str(e), traceback=tb)
            records.append(rec)
            if log_jsonl:
                _write_run_record(out_dir, rec)

    return records


# ----------------------------- Multiprocessing ----------------------------- #

def run_jobs_multiprocess(
    jobs: Sequence[Tuple[Any, float]],
    loader: LoaderFn,
    config: Dict[str, Any],
    *,
    out_dir: Union[str, Path] = "outputs",
    nproc: int = 4,
    figure: bool = False,
    calEFR: bool = False,
    ref_clusters: Any = None,
    saver: Optional[SaverFn] = None,
    resume: bool = True,
    log_jsonl: bool = True,
) -> List[RunRecord]:
    """
    Run arbitrary (field, z) jobs in parallel using multiprocessing.

    Notes
    -----
    - `loader` must be picklable (top-level function), because it's sent to workers.
    - For stability, keep `figure=False` in parallel runs (matplotlib not fork-safe).
    - External SExtractor calls are fine in parallel, but mind I/O contention.
    """
    out_dir = _ensure_dir(out_dir)

    # Filter jobs if resume
    if resume:
        jobs = [(f, float(z)) for (f, z) in jobs if not _is_done(out_dir, f, float(z))]

    if len(jobs) == 0:
        return []

    import multiprocessing as mp

    ctx = mp.get_context("spawn")  # safer than fork for matplotlib/scipy in many environments

    # Use initializer to pass config safely (copied into each worker)
    with ctx.Pool(processes=int(nproc)) as pool:
        args_iter = [
            (
                f,
                float(z),
                loader,
                config,
                out_dir,
                figure,
                calEFR,
                ref_clusters,
                saver,
                log_jsonl,
            )
            for (f, z) in jobs
        ]
        records = pool.starmap(_run_one_job_worker, args_iter)

    return records


def _run_one_job_worker(
    field: Any,
    z: float,
    loader: LoaderFn,
    config: Dict[str, Any],
    out_dir: Path,
    figure: bool,
    calEFR: bool,
    ref_clusters: Any,
    saver: Optional[SaverFn],
    log_jsonl: bool,
) -> RunRecord:
    """
    Worker-safe single job runner.
    Writes logs/done markers from each worker (append-only JSONL is OK).
    """
    try:
        cat_KS4, galcat, image_path, info = loader(field, float(z))
        info = dict(info)
        info.setdefault("field_id", str(field))
        info["z"] = float(z)

        result = run_pipeline(
            cat_KS4,
            galcat,
            image_path=image_path,
            info=info,
            config=config,
            figure=figure,
            calEFR=calEFR,
            ref_clusters=ref_clusters,
        )

        summary = _default_summary(result)

        if saver is not None:
            saver(result, field, float(z), out_dir)

        _mark_done(out_dir, field, float(z), summary)

        rec = RunRecord(field=field, z=float(z), ok=True, result_summary=summary)
        if log_jsonl:
            _write_run_record(out_dir, rec)
        return rec

    except Exception as e:
        tb = traceback.format_exc()
        rec = RunRecord(field=field, z=float(z), ok=False, error=str(e), traceback=tb)
        if log_jsonl:
            _write_run_record(out_dir, rec)
        return rec
