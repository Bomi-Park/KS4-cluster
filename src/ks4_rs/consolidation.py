"""
consolidation.py

Post-processing: consolidate (merge) duplicated cluster candidates.

Why consolidation is needed
---------------------------
In RS-based detection, especially for rich clusters, the RS color distribution
can broaden and trigger detections in adjacent redshift slices. This can create
multiple "new" candidates at different redshifts that are actually the same
physical cluster.

This module merges cluster candidates that are close in:
- sky position (angular separation)
- redshift (|dz|), optionally adaptive
and (optionally) share member galaxies.

Inputs/Outputs
--------------
Input cluster catalog:
- columns: CLUSTER_ID, RA, DEC, Z, N_MEMBER (minimum)
Optional member catalog:
- columns: CLUSTER_ID, OBJID (or unique source id), RA, DEC, Z

Output:
- consolidated cluster catalog with a new group id (GROUP_ID)
- mapping table of original CLUSTER_ID -> GROUP_ID
- (optional) consolidated members with GROUP_ID
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


@dataclass
class ConsolidationResult:
    clusters: Table
    mapping: Table
    members: Optional[Table] = None


def consolidate_clusters(
    clusters: Table,
    members: Optional[Table],
    config: Dict[str, Any],
) -> ConsolidationResult:
    """
    Consolidate duplicated cluster candidates.

    Parameters
    ----------
    clusters : astropy.table.Table
        Candidate clusters table. Required columns:
            - 'CLUSTER_ID', 'RA', 'DEC', 'Z'
        Optional:
            - 'N_MEMBER' (used for choosing representative)
    members : astropy.table.Table or None
        Member table. If provided and config enables it,
        member-overlap will be used as an extra merging criterion.
        Expected columns:
            - 'CLUSTER_ID', and either 'OBJID' or ('RA','DEC')
    config : dict
        Configuration dictionary. Expected section: config['consolidation'].

    Returns
    -------
    ConsolidationResult
        clusters: consolidated cluster table (adds GROUP_ID and REP flags)
        mapping : table mapping CLUSTER_ID -> GROUP_ID
        members : (optional) member table with GROUP_ID added
    """
    cfg = config.get("consolidation", {})

    if clusters is None or len(clusters) == 0:
        return ConsolidationResult(
            clusters=Table(),
            mapping=Table(names=("CLUSTER_ID", "GROUP_ID"), dtype=("U64", "i8")),
            members=members if members is not None else None,
        )

    # ------------------------------------------------------------
    # Required columns check
    # ------------------------------------------------------------
    for c in ("CLUSTER_ID", "RA", "DEC", "Z"):
        if c not in clusters.colnames:
            raise KeyError(f"clusters must contain column '{c}'")

    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------
    # Angular match threshold
    match_arcmin = float(cfg.get("match_radius_arcmin", 2.0))
    match_radius = match_arcmin * u.arcmin

    # Redshift threshold (static)
    dz_max = cfg.get("dz_max", 0.05)
    dz_max = float(dz_max) if dz_max is not None else None

    # Optional adaptive dz: dz_max(z) = base + slope*(1+z)
    adaptive = bool(cfg.get("adaptive_dz", False))
    dz_base = float(cfg.get("dz_base", 0.02))
    dz_slope = float(cfg.get("dz_slope", 0.02))

    # Member overlap criterion (optional)
    use_member_overlap = bool(cfg.get("use_member_overlap", False))
    min_jaccard = float(cfg.get("min_member_jaccard", 0.2))  # 0..1
    member_id_col = cfg.get("member_id_col", "OBJID")  # preferred id

    # Representative selection
    rep_by = cfg.get("representative", "max_n_member")  # or "closest_to_median_z"

    # ------------------------------------------------------------
    # Build coordinates
    # ------------------------------------------------------------
    coords = SkyCoord(ra=clusters["RA"] * u.deg, dec=clusters["DEC"] * u.deg)
    zarr = np.asarray(clusters["Z"], dtype=float)

    # ------------------------------------------------------------
    # Union-Find (Disjoint Set) for grouping
    # ------------------------------------------------------------
    parent = np.arange(len(clusters), dtype=int)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Precompute member sets per cluster if enabled
    member_sets = None
    if use_member_overlap:
        if members is None or len(members) == 0:
            # If enabled but missing, silently fallback to position+z only
            use_member_overlap = False
        else:
            if "CLUSTER_ID" not in members.colnames:
                raise KeyError("members must contain 'CLUSTER_ID'")
            member_sets = _build_member_sets(members, member_id_col=member_id_col)

    # ------------------------------------------------------------
    # Pairwise grouping (O(N^2) by default)
    # For typical candidate counts per field/z this is fine.
    # If N is huge, we can later add a spatial index (k-d tree / healpix).
    # ------------------------------------------------------------
    N = len(clusters)
    for i in range(N):
        for j in range(i + 1, N):
            # 1) angular separation
            if coords[i].separation(coords[j]) > match_radius:
                continue

            # 2) redshift proximity
            dz_thr = None
            if adaptive:
                zmid = 0.5 * (zarr[i] + zarr[j])
                dz_thr = dz_base + dz_slope * (1.0 + zmid)
            else:
                dz_thr = dz_max

            if dz_thr is not None:
                if abs(zarr[i] - zarr[j]) > dz_thr:
                    continue

            # 3) member overlap (optional)
            if use_member_overlap and member_sets is not None:
                s1 = member_sets.get(str(clusters["CLUSTER_ID"][i]), set())
                s2 = member_sets.get(str(clusters["CLUSTER_ID"][j]), set())
                if len(s1) == 0 or len(s2) == 0:
                    # If one has no members, do not require overlap
                    pass
                else:
                    jac = _jaccard(s1, s2)
                    if jac < min_jaccard:
                        continue

            # If all checks passed → same group
            union(i, j)

    # ------------------------------------------------------------
    # Assign GROUP_ID
    # ------------------------------------------------------------
    roots = np.array([find(i) for i in range(N)], dtype=int)
    unique_roots = {r: k for k, r in enumerate(np.unique(roots), start=1)}
    group_id = np.array([unique_roots[r] for r in roots], dtype=int)

    out_clusters = clusters.copy()
    out_clusters["GROUP_ID"] = group_id

    # ------------------------------------------------------------
    # Choose representative cluster per group
    # ------------------------------------------------------------
    rep_mask = np.zeros(N, dtype=bool)
    for gid in np.unique(group_id):
        idx = np.where(group_id == gid)[0]
        rep_idx = _choose_representative(out_clusters[idx], rep_by=rep_by)
        rep_mask[idx[rep_idx]] = True

    out_clusters["IS_REP"] = rep_mask

    # ------------------------------------------------------------
    # Mapping table: CLUSTER_ID -> GROUP_ID
    # ------------------------------------------------------------
    mapping = Table()
    mapping["CLUSTER_ID"] = out_clusters["CLUSTER_ID"]
    mapping["GROUP_ID"] = out_clusters["GROUP_ID"]

    # ------------------------------------------------------------
    # Members: add GROUP_ID if provided
    # ------------------------------------------------------------
    out_members = None
    if members is not None and len(members) > 0:
        # Build dict from CLUSTER_ID -> GROUP_ID
        gid_map = {str(cid): int(gid) for cid, gid in zip(out_clusters["CLUSTER_ID"], out_clusters["GROUP_ID"])}
        out_members = members.copy()
        out_members["GROUP_ID"] = [gid_map.get(str(cid), -1) for cid in out_members["CLUSTER_ID"]]

    return ConsolidationResult(clusters=out_clusters, mapping=mapping, members=out_members)


# =============================================================================
# Helpers
# =============================================================================

def _jaccard(a: set, b: set) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _build_member_sets(members: Table, member_id_col: str = "OBJID") -> Dict[str, set]:
    """
    Build member id sets for each cluster.
    If member_id_col is missing, fallback to RA/DEC rounded hashing.
    """
    sets: Dict[str, set] = {}

    has_id = member_id_col in members.colnames
    for row in members:
        cid = str(row["CLUSTER_ID"])
        sets.setdefault(cid, set())
        if has_id and row[member_id_col] is not None:
            sets[cid].add(str(row[member_id_col]))
        else:
            # fallback: hash position
            ra = float(row["RA"]) if "RA" in members.colnames else np.nan
            dec = float(row["DEC"]) if "DEC" in members.colnames else np.nan
            key = f"{ra:.6f}_{dec:.6f}"
            sets[cid].add(key)

    return sets


def _choose_representative(sub: Table, rep_by: str = "max_n_member") -> int:
    """
    Choose representative cluster index within a group.
    Returns index *within sub-table*.
    """
    if len(sub) == 1:
        return 0

    if rep_by == "max_n_member" and "N_MEMBER" in sub.colnames:
        return int(np.argmax(np.asarray(sub["N_MEMBER"], dtype=float)))

    if rep_by == "closest_to_median_z":
        z = np.asarray(sub["Z"], dtype=float)
        zmed = np.nanmedian(z)
        return int(np.argmin(np.abs(z - zmed)))

    # fallback
    return 0
