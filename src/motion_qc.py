"""
motion_qc.py
------------
Computes the full set of motion QC metrics across a sample of sessions.

Reads motion_transform.csv and quality_control.json directly from S3
for each session × plane. Does not load the NWB file.

Outputs
-------
outputs/stage2/motion_plane_metrics.csv   — one row per session × plane
outputs/stage2/motion_session_metrics.csv — one row per session

Typical usage
-------------
    import motion_qc

    plane_df, session_df = motion_qc.run(all_paths, n=20, seed=42)
"""

from __future__ import annotations

import random
import re
from pathlib import Path
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

PIXEL_SIZE_UM  = 0.78    # µm per pixel
IMAGING_RATE   = 9.48    # Hz
BAD_DISP_UM    = 5.0     # threshold for "bad" frame
SETTLE_UM      = 3.0     # threshold for "settled" baseline
SETTLE_WIN_S   = 60.0    # consecutive seconds below threshold to declare settled
BURST_DILATION = 2       # frames to dilate around bad-frame detections
MIN_CROSS_CORR = 0.3     # below this, a plane is an "outlier"

OUT_DIR = Path("outputs/stage2")


# ---------------------------------------------------------------------------
# quality_control.json parsing
# ---------------------------------------------------------------------------

def _parse_qc_json(qc: dict) -> dict[str, dict]:
    """
    Extract per-plane z_drift_um and intensity_stability_pct
    from the quality_control.json evaluations block.

    Returns
    -------
    {plane_name: {"z_drift_um": float, "intensity_stability_pct": float}}
    """
    per_plane: dict[str, dict] = {}

    for evaluation in qc.get("evaluations", []):
        for metric in evaluation.get("metrics", []):
            name  = metric.get("name", "")
            value = metric.get("value", {})

            zdrift_match = re.match(r"(\w+_\d+)\s+Z-drift", name)
            if zdrift_match and isinstance(value, dict):
                plane = zdrift_match.group(1)
                per_plane.setdefault(plane, {})
                per_plane[plane]["z_drift_um"] = value.get("z_drift_um", np.nan)

            intensity_match = re.match(r"Intensity stability - (\w+_\d+)", name)
            if intensity_match and isinstance(value, str):
                plane = intensity_match.group(1)
                per_plane.setdefault(plane, {})
                try:
                    per_plane[plane]["intensity_stability_pct"] = float(
                        value.replace("%", "")
                    )
                except ValueError:
                    per_plane[plane]["intensity_stability_pct"] = np.nan

    return per_plane


# ---------------------------------------------------------------------------
# Per-plane metric computation
# ---------------------------------------------------------------------------

def _compute_plane_metrics(
    df: pd.DataFrame,
    session_id: str,
    subject_id: str,
    plane: str,
    qc_plane: dict,
) -> dict[str, Any]:
    """
    Derive all motion QC metrics for one plane from its motion_transform DataFrame.
    """
    df = df.copy()
    df["is_valid"]        = df["is_valid"].astype(str).str.lower() == "true"
    df["displacement_px"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["displacement_um"] = df["displacement_px"] * PIXEL_SIZE_UM

    T       = len(df)
    disp    = df["displacement_um"].values
    corr    = df["correlation"].values
    valid   = df["is_valid"].values
    t_s     = np.arange(T) / IMAGING_RATE
    t_min   = t_s / 60.0
    n5      = min(int(5 * 60 * IMAGING_RATE), T // 4)

    # --- basic displacement ---
    median_disp_um      = float(np.median(disp))
    mean_disp_um        = float(np.mean(disp))
    std_disp_um         = float(np.std(disp))
    p95_disp_um         = float(np.percentile(disp, 95))
    p99_disp_um         = float(np.percentile(disp, 99))
    max_disp_um         = float(np.max(disp))
    displacement_cv     = float(std_disp_um / mean_disp_um) if mean_disp_um > 0 else np.nan
    bad_frame_frac      = float(np.mean(disp > BAD_DISP_UM))
    invalid_frame_frac  = float(np.mean(~valid))

    # --- registration correlation ---
    mean_reg_corr   = float(np.mean(corr))
    median_reg_corr = float(np.median(corr))
    min_reg_corr    = float(np.min(corr))
    std_reg_corr    = float(np.std(corr))

    # --- temporal: early vs late ---
    first_5min_median_um = float(np.median(disp[:n5])) if n5 > 0 else np.nan
    last_5min_median_um  = float(np.median(disp[-n5:])) if n5 > 0 else np.nan
    early_late_ratio     = (
        float(first_5min_median_um / last_5min_median_um)
        if last_5min_median_um and last_5min_median_um > 0
        else np.nan
    )

    # --- temporal: linear trend ---
    displacement_slope_um_per_min = (
        float(np.polyfit(t_min, disp, 1)[0]) if T > 10 else np.nan
    )

    # --- settling time ---
    win_frames      = int(SETTLE_WIN_S * IMAGING_RATE)
    settling_time_s = np.nan
    if win_frames < T:
        below   = (disp <= SETTLE_UM).astype(int)
        cumsum  = np.cumsum(np.concatenate(([0], below)))
        rolling = cumsum[win_frames:] - cumsum[:-win_frames]
        settled = np.where(rolling == win_frames)[0]
        if len(settled) > 0:
            settling_time_s = float(t_s[settled[0]])

    # --- motion bursts ---
    bad     = (disp > BAD_DISP_UM).astype(int)
    dilated = bad.copy()
    for shift in range(1, BURST_DILATION + 1):
        dilated[shift:]  |= bad[:-shift]
        dilated[:-shift] |= bad[shift:]
    transitions      = np.diff(np.concatenate(([0], dilated, [0])))
    n_motion_bursts  = int(np.sum(transitions == 1))

    # --- longest clean run ---
    clean       = (disp <= BAD_DISP_UM).astype(int)
    trans_clean = np.diff(np.concatenate(([0], clean, [0])))
    starts      = np.where(trans_clean == 1)[0]
    ends        = np.where(trans_clean == -1)[0]
    longest_clean_run_s = (
        float((ends - starts).max() / IMAGING_RATE) if len(starts) > 0 else 0.0
    )

    return {
        "session_id"                   : session_id,
        "subject_id"                   : subject_id,
        "plane"                        : plane,
        "structure"                    : plane.split("_")[0],
        "plane_idx"                    : int(plane.split("_")[1]),
        "n_frames"                     : T,
        # displacement
        "median_disp_um"               : median_disp_um,
        "mean_disp_um"                 : mean_disp_um,
        "std_disp_um"                  : std_disp_um,
        "p95_disp_um"                  : p95_disp_um,
        "p99_disp_um"                  : p99_disp_um,
        "max_disp_um"                  : max_disp_um,
        "displacement_cv"              : displacement_cv,
        "bad_frame_frac"               : bad_frame_frac,
        "invalid_frame_frac"           : invalid_frame_frac,
        # registration
        "mean_reg_corr"                : mean_reg_corr,
        "median_reg_corr"              : median_reg_corr,
        "min_reg_corr"                 : min_reg_corr,
        "std_reg_corr"                 : std_reg_corr,
        # temporal
        "first_5min_median_um"         : first_5min_median_um,
        "last_5min_median_um"          : last_5min_median_um,
        "early_late_ratio"             : early_late_ratio,
        "displacement_slope_um_per_min": displacement_slope_um_per_min,
        "settling_time_s"              : settling_time_s,
        "n_motion_bursts"              : n_motion_bursts,
        "longest_clean_run_s"          : longest_clean_run_s,
        # from quality_control.json
        "z_drift_um"                   : qc_plane.get("z_drift_um", np.nan),
        "intensity_stability_pct"      : qc_plane.get("intensity_stability_pct", np.nan),
        "error"                        : None,
    }


# ---------------------------------------------------------------------------
# Session-level aggregation
# ---------------------------------------------------------------------------

def _session_aggregate(
    plane_rows: list[dict],
    session_id: str,
    subject_id: str,
) -> dict[str, Any]:
    """
    Aggregate plane-level metrics to a single session row.
    Also computes cross-plane displacement correlation.
    """
    ok_rows = [r for r in plane_rows if r.get("error") is None]
    if not ok_rows:
        return {
            "session_id"     : session_id,
            "subject_id"     : subject_id,
            "n_planes_loaded": 0,
            "error"          : "all planes failed",
        }

    keys_to_aggregate = [
        "median_disp_um", "p95_disp_um", "bad_frame_frac",
        "invalid_frame_frac", "mean_reg_corr", "min_reg_corr",
        "first_5min_median_um", "last_5min_median_um",
        "displacement_slope_um_per_min", "settling_time_s",
        "n_motion_bursts", "longest_clean_run_s",
        "z_drift_um", "intensity_stability_pct", "displacement_cv",
    ]

    agg = {
        "session_id"      : session_id,
        "subject_id"      : subject_id,
        "n_planes_loaded" : len(ok_rows),
        "error"           : None,
    }

    for k in keys_to_aggregate:
        vals = [
            r[k] for r in ok_rows
            if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))
        ]
        agg[f"session_{k}_median"] = float(np.median(vals)) if vals else np.nan
        agg[f"session_{k}_max"]    = float(np.max(vals))    if vals else np.nan

    # cross-plane displacement correlation
    disp_arrays = [r.get("_disp_arr") for r in ok_rows if r.get("_disp_arr") is not None]
    if len(disp_arrays) >= 2:
        min_len  = min(len(a) for a in disp_arrays)
        mat      = np.stack([a[:min_len] for a in disp_arrays])
        corr_mat = np.corrcoef(mat)
        n        = corr_mat.shape[0]
        off_diag = corr_mat[np.triu_indices(n, k=1)]
        mean_corr_per_plane = [(corr_mat[i].sum() - 1) / (n - 1) for i in range(n)]
        agg["mean_cross_plane_corr"] = float(np.mean(off_diag))
        agg["outlier_plane_count"]   = int(sum(c < MIN_CROSS_CORR for c in mean_corr_per_plane))
    else:
        agg["mean_cross_plane_corr"] = np.nan
        agg["outlier_plane_count"]   = np.nan

    return agg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    all_paths: list[str],
    n: int = 20,
    seed: int = 42,
    out_dir: Path = OUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run motion QC across a random sample of sessions.

    Parameters
    ----------
    all_paths : list of s3:// paths from session_loader.discover_sessions()
    n         : number of sessions to sample
    seed      : random seed for reproducible sampling
    out_dir   : directory for output CSVs

    Returns
    -------
    plane_df   : one row per session × plane
    session_df : one row per session (aggregated across planes)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    sampled = random.sample(all_paths, min(n, len(all_paths)))
    print(f"Running motion QC on {len(sampled)} sessions\n")

    all_plane_rows:   list[dict] = []
    all_session_rows: list[dict] = []

    for i, s3_path in enumerate(sampled):
        session_id = parse_session_id(s3_path)
        subject_id = parse_subject_id(s3_path)
        bucket, session_key = parse_s3_path(s3_path)
        print(f"[{i+1}/{len(sampled)}] {session_id}")

        # quality_control.json
        qc_json      = read_json(bucket, f"{session_key}/quality_control.json")
        qc_per_plane = _parse_qc_json(qc_json) if qc_json else {}

        # discover planes
        planes = list_plane_names(bucket, session_key)
        if not planes:
            print("  ✗  no planes found")
            all_session_rows.append({
                "session_id"     : session_id,
                "subject_id"     : subject_id,
                "n_planes_loaded": 0,
                "error"          : "no planes found",
            })
            continue

        print(f"  planes: {planes}")
        session_plane_rows: list[dict] = []

        for plane in planes:
            csv_key = (
                f"{session_key}/{plane}/motion_correction"
                f"/{plane}_motion_transform.csv"
            )
            df = read_csv(bucket, csv_key)

            if df is None or df.empty:
                session_plane_rows.append({
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "plane"     : plane,
                    "structure" : plane.split("_")[0],
                    "error"     : "csv not found or empty",
                })
                print(f"    {plane} ✗  (csv missing)")
                continue

            try:
                row = _compute_plane_metrics(
                    df, session_id, subject_id, plane,
                    qc_per_plane.get(plane, {}),
                )
                row["_disp_arr"] = (
                    np.sqrt(df["x"].values ** 2 + df["y"].values ** 2) * PIXEL_SIZE_UM
                )
                session_plane_rows.append(row)
                print(
                    f"    {plane} ✓  "
                    f"median={row['median_disp_um']:.2f} µm  "
                    f"bad_frac={row['bad_frame_frac']:.3f}"
                )
            except Exception as e:
                session_plane_rows.append({
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "plane"     : plane,
                    "structure" : plane.split("_")[0],
                    "error"     : str(e),
                })
                print(f"    {plane} ✗  ({e})")

        session_row = _session_aggregate(session_plane_rows, session_id, subject_id)
        all_session_rows.append(session_row)
        print(
            f"  → session median_disp="
            f"{session_row.get('session_median_disp_um_median', np.nan):.2f} µm  "
            f"cross_plane_corr="
            f"{session_row.get('mean_cross_plane_corr', np.nan):.3f}\n"
        )

        for row in session_plane_rows:
            row.pop("_disp_arr", None)
        all_plane_rows.extend(session_plane_rows)

    plane_df   = pd.DataFrame(all_plane_rows)
    session_df = pd.DataFrame(all_session_rows)

    plane_df.to_csv(out_dir / "motion_plane_metrics.csv",   index=False)
    session_df.to_csv(out_dir / "motion_session_metrics.csv", index=False)
    print(f"Saved to {out_dir}/")

    return plane_df, session_df
