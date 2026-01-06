from __future__ import annotations

from pathlib import Path

from ks4_rs.config import load_config
from ks4_rs.runners import run_single_field_all_z


def loader(field_id: str, z: float):
    """
    TODO: 환경에 맞게 여기만 채우면 됨.

    Returns
    -------
    cat_KS4, galcat, image_path, info
    """
    raise NotImplementedError("Fill the loader() with your KS4 catalog loading logic.")


def main():
    config = load_config("configs/pipeline_test.yaml")

    field_id = "TEST_FIELD_001"
    z_list = [0.20, 0.25, 0.30, 0.35, 0.40]  # TEST: 몇 개만

    out_dir = Path("outputs_test") / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = run_single_field_all_z(
        field=field_id,
        z_list=z_list,
        loader=loader,
        config=config,
        out_dir=out_dir,
        figure=False,   # TEST: 빠르게/안정적으로
        calEFR=False,
        ref_clusters=None,
        saver=None,
        resume=True,
        log_jsonl=True,
    )

    ok = sum(r.ok for r in records)
    fail = len(records) - ok
    print(f"[SUMMARY] ok={ok} fail={fail} (log: {out_dir/'run_log.jsonl'})")


if __name__ == "__main__":
    main()
