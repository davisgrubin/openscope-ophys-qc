"""
motion_loader.py
----------------
Loads per-frame motion-correction data for a single session from S3.

Reads the motion_transform.csv and registration_summary_metric.json
for every imaging plane in a session, returning a tidy dict of DataFrames
ready for plotting or metric computation.

Typical usage
-------------
    import motion_loader as ml

    motion_data = ml.load_motion_data(SESSION_SOURCE)
    # motion_data["planes"]    → {plane_name: DataFrame}
    # motion_data["reg_metrics"] → {plane_name: dict}
    # motion_data["session_id"]  → str
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from s3_utils import (
    list_plane_names,
    parse_s3_path,
    parse_session_id,
    parse_subject_id,
    read_csv,
    read_json,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIXEL_SIZE_UM = 0.78   # µm per pixel
IMAGING_RATE  = 9.48   # Hz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_motion_data(
    s3_path: str,
    planes: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load motion-correction data for every plane in a session.

    Parameters
    ----------
    s3_path : s3:// URL for the processed session asset
    planes  : restrict to these plane names; None = all discovered planes
    verbose : print progress

    Returns
    -------
    dict with keys:
        "session_id"   : str
        "subject_id"   : str
        "planes"       : {plane_name: DataFrame}
            Each DataFrame has columns:
                framenumber, x, y, correlation, is_valid,
                displacement_px, displacement_um, time_s
        "reg_metrics"  : {plane_name: dict}   (registration summary JSON)
        "plane_names"  : list[str]             (ordered)
    """
    session_id = parse_session_id(s3_path)
    subject_id = parse_subject_id(s3_path)
    bucket, session_key = parse_s3_path(s3_path)

    discovered = list_plane_names(bucket, session_key)
    if planes is not None:
        discovered = [p for p in discovered if p in planes]

    if verbose:
        print(f"Session  : {session_id}")
        print(f"Planes   : {discovered}")

    plane_dfs:   dict[str, pd.DataFrame] = {}
    reg_metrics: dict[str, dict]         = {}

    for plane in discovered:
        mc_prefix = f"{session_key}/{plane}/motion_correction"

        # --- displacement CSV ---
        csv_key = f"{mc_prefix}/{plane}_motion_transform.csv"
        df = read_csv(bucket, csv_key)

        if df is None or df.empty:
            if verbose:
                print(f"  {plane} ✗  (motion_transform.csv missing)")
            continue

        df = _enrich_motion_df(df)
        plane_dfs[plane] = df

        # --- registration metrics JSON ---
        json_key = f"{mc_prefix}/{plane}_registration_summary_metric.json"
        metrics  = read_json(bucket, json_key)
        if metrics is not None:
            reg_metrics[plane] = metrics

        if verbose:
            med = df["displacement_um"].median()
            bad = (~df["is_valid"]).sum()
            print(f"  {plane} ✓  "
                  f"median={med:.2f} µm  "
                  f"invalid={bad}/{len(df)}")

    return {
        "session_id"  : session_id,
        "subject_id"  : subject_id,
        "planes"      : plane_dfs,
        "reg_metrics" : reg_metrics,
        "plane_names" : list(plane_dfs.keys()),
    }


def displacement_matrix(
    motion_data: dict[str, Any],
    plane_order: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Stack per-plane displacement arrays into a (n_planes, T) matrix.

    Planes are trimmed to the shortest common length to handle
    occasional frame-count mismatches.

    Parameters
    ----------
    motion_data : output of load_motion_data()
    plane_order : desired row order; defaults to motion_data["plane_names"]

    Returns
    -------
    (disp_matrix, plane_names)
    """
    planes = plane_order or motion_data["plane_names"]
    planes = [p for p in planes if p in motion_data["planes"]]

    arrays  = [motion_data["planes"][p]["displacement_um"].values for p in planes]
    min_len = min(len(a) for a in arrays)
    matrix  = np.stack([a[:min_len] for a in arrays])

    return matrix, planes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enrich_motion_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a raw motion_transform DataFrame."""
    df = df.copy()
    df["is_valid"]        = df["is_valid"].astype(str).str.lower() == "true"
    df["displacement_px"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["displacement_um"] = df["displacement_px"] * PIXEL_SIZE_UM
    df["time_s"]          = df["framenumber"] / IMAGING_RATE
    return df
