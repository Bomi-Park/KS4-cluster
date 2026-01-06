from __future__ import annotations

from pathlib import Path

from ks4_rs.config import load_config
from ks4_rs.pipeline import run_pipeline


def loader(field_id: str, z: float):
    """
    TODO: 너 환경에 맞게 여기만 채우면 됨.

    Returns
    -------
    cat_KS4 : astropy.table.Table
    galcat  : astropy.table.Table
    image_path : str | Path
    info : dict (must include RS_mean, RS_sigma at minimum)
    """
    # 예시(가짜):
    # from astropy.table import Table
    # cat_KS4 = Table.read(f"data/{field_id}_ks4.fits")
    # galcat = cat_KS4[cat_KS4["IS_GAL"] == 1]

    raise NotImplementedError("Fill the loader() with your KS4 catalog loading logic.")


def main():
    field_id = "TEST_FIELD_001"
    z = 0.35

    config = load_config("configs/pipeline_test.yaml")
    out_dir = Path("outputs_test") / field_id / f"z{z:.3f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_KS4, galcat, image_path, info = loader(field_id, z)

    # Step1 requires RS_mean / RS_sigma in info
    # You can compute/lookup them from your BC03 RS model table.
    info = dict(info)
    info["field_id"] = field_id
    info["z"] = float(z)

    result = run_pipeline(
        cat_KS4,
        galcat,
        image_path=image_path,
        info=info,
        config=config,
        figure=False,   # TEST: 빠르게
        calEFR=False,
        ref_clusters=None,
    )

    n_rs = len(result["rs"]["RScand"]) if result.get("rs") and result["rs"].get("RScand") is not None else 0
    n_cl = len(result["cluster"]["clusters"]) if result.get("cluster") and result["cluster"].get("clusters") is not None else 0

    print(f"[DONE] field={field_id} z={z:.3f} | N_RS={n_rs} | N_cluster={n_cl}")
    print("FITS:", result["fits"]["fits_path"] if result.get("fits") else None)
    print("SEx cat:", result["sex"]["catalog_path"] if result.get("sex") else None)


if __name__ == "__main__":
    main()
