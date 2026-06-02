"""
roi_classifier.py
-----------------
Extracts per-ROI soma and dendrite classification scores from the NWB Zarr
files produced by the Allen Institute's suite2p segmentation pipeline.

For each session × plane the segmentation table provides, for every detected
ROI, a probability that it is a cell body (soma) and a probability that it is
a dendrite process. This module reads those scores and computes per-plane
summary statistics used downstream for quality assessment.

Outputs
-------
outputs/stage2/roi_classifications.csv  — one row per ROI
outputs/stage2/roi_plane_summary.csv    — one row per session × plane

Typical usage
-------------
    import roi_classifier

    roi_df, summary_df = roi_classifier.run(all_paths, n=20, seed=42)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from hdmf_zarr import NWBZarrIO

from s3_utils import parse_session_id, parse_subject_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUT_DIR = Path("outputs/stage2")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _nwb_url(s3_path: str) -> str:
    """Return the s3:// URL for the pophys.nwb.zarr inside an asset."""
    base = s3_path.rstrip("/") if s3_path.startswith("s3://") else "s3://" + s3_path.rstrip("/")
    return base + "/pophys.nwb.zarr"


# ---------------------------------------------------------------------------
# Per-session extraction
# ---------------------------------------------------------------------------

def extract_session(
    s3_path: str,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Open the NWB Zarr for one session and extract ROI classification data.

    Reads is_soma, soma_probability, and dendrite_probability from the
    segmentation table in each plane's processing module.

    Parameters
    ----------
    s3_path : s3:// URL for the processed session asset
    verbose : print per-plane progress

    Returns
    -------
    roi_rows     : list of per-ROI dicts
    summary_rows : list of per-plane summary dicts
    """
    session_id = parse_session_id(s3_path)
    subject_id = parse_subject_id(s3_path)
    nwb_url    = _nwb_url(s3_path)

    if verbose:
        print(f"  opening {nwb_url.split('/')[-2]} ...", end=" ", flush=True)

    try:
        io  = NWBZarrIO(
            path=nwb_url, mode="r",
            load_namespaces=True,
            storage_options={"anon": True},
        )
        nwb = io.read()
    except Exception as e:
        if verbose:
            print(f"✗  ({e})")
        return [], []

    available_planes = list(nwb.processing.keys())
    if verbose:
        print(f"✓  planes: {available_planes}")

    roi_rows:     list[dict] = []
    summary_rows: list[dict] = []

    for plane in available_planes:
        structure = plane.split("_")[0]
        try:
            mod       = nwb.processing[plane]
            roi_table = mod["image_segmentation"]["roi_table"]

            is_soma       = np.array(roi_table["is_soma"].data[:]).astype(bool)
            soma_prob     = np.array(roi_table["soma_probability"].data[:]).astype(float)
            dendrite_prob = np.array(roi_table["dendrite_probability"].data[:]).astype(float)
            n_rois        = len(is_soma)

            # per-ROI rows
            for i in range(n_rois):
                roi_rows.append({
                    "session_id"           : session_id,
                    "subject_id"           : subject_id,
                    "plane"                : plane,
                    "structure"            : structure,
                    "roi_index"            : i,
                    "is_soma"              : bool(is_soma[i]),
                    "soma_probability"     : float(soma_prob[i]),
                    "dendrite_probability" : float(dendrite_prob[i]),
                })

            # per-plane summary
            summary_rows.append({
                "session_id"             : session_id,
                "subject_id"             : subject_id,
                "plane"                  : plane,
                "structure"              : structure,
                "n_rois"                 : n_rois,
                "soma_frac"              : float(is_soma.mean()),
                "dendrite_frac"          : float((~is_soma).mean()),
                "mean_soma_prob"         : float(soma_prob.mean()),
                "median_soma_prob"       : float(np.median(soma_prob)),
                "std_soma_prob"          : float(soma_prob.std()),
                "mean_dendrite_prob"     : float(dendrite_prob.mean()),
                "median_dendrite_prob"   : float(np.median(dendrite_prob)),
                "std_dendrite_prob"      : float(dendrite_prob.std()),
            })

            if verbose:
                print(
                    f"    {plane}: {n_rois} ROIs  "
                    f"{is_soma.sum()} soma ({100*is_soma.mean():.0f}%)  "
                    f"mean soma_prob={soma_prob.mean():.3f}"
                )

        except Exception as e:
            if verbose:
                print(f"    {plane} ✗  ({e})")

    try:
        io.close()
    except Exception:
        pass

    return roi_rows, summary_rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    all_paths: list[str],
    n: int = 20,
    seed: int = 42,
    out_dir: Path = OUT_DIR,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract ROI classifications across a random sample of sessions.

    Parameters
    ----------
    all_paths : list of s3:// paths from session_loader.discover_sessions()
    n         : number of sessions to sample
    seed      : random seed
    out_dir   : directory for output CSVs
    verbose   : print per-session and per-plane progress

    Returns
    -------
    roi_df     : one row per ROI
    summary_df : one row per session × plane
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    sampled = random.sample(all_paths, min(n, len(all_paths)))
    print(f"Extracting ROI classifications from {len(sampled)} sessions\n")

    all_roi_rows:     list[dict] = []
    all_summary_rows: list[dict] = []

    for i, s3_path in enumerate(sampled):
        session_id = parse_session_id(s3_path)
        print(f"[{i+1}/{len(sampled)}] {session_id}")

        roi_rows, summary_rows = extract_session(s3_path, verbose=verbose)
        all_roi_rows.extend(roi_rows)
        all_summary_rows.extend(summary_rows)
        print()

    roi_df     = pd.DataFrame(all_roi_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    roi_path     = out_dir / "roi_classifications.csv"
    summary_path = out_dir / "roi_plane_summary.csv"
    roi_df.to_csv(roi_path,     index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved:")
    print(f"  {roi_path}  ({len(roi_df):,} ROI rows)")
    print(f"  {summary_path}  ({len(summary_df)} plane rows)")

    return roi_df, summary_df
