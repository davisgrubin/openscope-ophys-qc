"""
session_loader.py
-----------------
Loading and visualization utilities for OpenScope planar-ophys QC.

Naming convention (AIND docs):
    Raw:     <modality>_<subject-id>_<acquisition-date>_<acquisition-time>
    Derived: <source-asset-name>_<label>_<processing-date>_<processing-time>

Usage
-----
    import session_loader as sl

    # Discover sessions (deduplicated to latest processing run per raw session)
    paths = sl.discover_sessions()

    # Load one session
    session = sl.load_session("s3://aind-open-data/multiplane-ophys_...")

    # Load a random sample
    sessions = sl.load_sessions(paths, n=3, seed=42)

    # Always close when done
    sl.close_sessions(sessions)
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mesoscope_qc_pipeline as qc


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_sessions(
    bucket: str = "aind-open-data",
    prefix: str = "multiplane-ophys_",
    deduplicate: bool = True,
) -> list[str]:
    """
    Return S3 paths for all processed planar-ophys sessions in the bucket.

    If deduplicate=True (default), keeps only the latest processing run
    for each raw session (same acquisition timestamp, reprocessed multiple times).
    """
    client    = qc.get_s3_client()
    paginator = client.get_paginator("list_objects_v2")

    all_paths = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            p = cp["Prefix"].rstrip("/")
            if "_processed_" in p:
                all_paths.append(f"s3://{bucket}/{p}")

    return _keep_latest(all_paths) if deduplicate else sorted(all_paths)


def _parse_path(s3_path: str) -> tuple[str, str, str]:
    """
    Split a derived asset path into (session_key, processing_timestamp, subject_id).

    "multiplane-ophys_837568_2026-03-05_14-14-51_processed_2026-03-06_11-31-22"
      session_key  = "multiplane-ophys_837568_2026-03-05_14-14-51"
      processed_ts = "2026-03-06_11-31-22"
      subject_id   = "837568"
    """
    name         = s3_path.rstrip("/").split("/")[-1]
    parts        = name.split("_processed_")
    session_key  = parts[0]
    processed_ts = parts[1] if len(parts) > 1 else ""
    subject_id   = session_key.split("_")[1]
    return session_key, processed_ts, subject_id


def _keep_latest(paths: list[str]) -> list[str]:
    """Keep only the path with the latest processing timestamp per raw session."""
    latest: dict[str, tuple[str, str]] = {}
    for p in paths:
        key, ts, _ = _parse_path(p)
        if key not in latest or ts > latest[key][1]:
            latest[key] = (p, ts)
    return sorted(v[0] for v in latest.values())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_session(s3_path: str) -> dict[str, Any]:
    """
    Open a single NWB session from S3.

    Returns a dict:
        path, nwb, meta, plane_meta, planes, io

    Call close_session() when done.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        open_result = qc.open_nwb_from_s3(s3_path)

    nwb = open_result.nwb
    return {
        "path"      : s3_path,
        "nwb"       : nwb,
        "meta"      : qc.extract_session_metadata(open_result),
        "plane_meta": qc.get_plane_metadata(nwb),
        "planes"    : qc.get_plane_names(nwb),
        "io"        : open_result.io,
    }


def load_sessions(
    paths: list[str],
    n: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Load a sample of sessions from a list of S3 paths.

    Parameters
    ----------
    paths : list of S3 paths, e.g. from discover_sessions()
    n     : number to load; loads all if None
    seed  : random seed for reproducible sampling
    """
    if n is not None and n < len(paths):
        random.seed(seed)
        paths = random.sample(paths, n)

    sessions = []
    for path in paths:
        label = path.split("/")[-1]
        print(f"Loading {label} ...", end=" ", flush=True)
        try:
            sessions.append(load_session(path))
            print("✓")
        except Exception as e:
            print(f"✗  ({e})")

    print(f"\nLoaded {len(sessions)} / {len(paths)} sessions")
    return sessions


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def close_session(session: dict[str, Any]) -> None:
    try:
        session["io"].close()
    except Exception:
        pass


def close_sessions(sessions: list[dict[str, Any]]) -> None:
    for s in sessions:
        close_session(s)


# ---------------------------------------------------------------------------
# Visualization — single session
# ---------------------------------------------------------------------------

def plot_reference_distributions(
    roi_qc: pd.DataFrame,
    title: str = "Reference distributions",
    save_path: Path | None = None,
) -> plt.Figure:
    """Histogram grid of key QC metrics, split by all ROIs vs soma-only."""
    METRICS = [
        ("soma_probability",               "Soma probability",          (0.0, 1.0)),
        ("dff_std",                        "dF/F std",                  None),
        ("dff_robust_range_5_95",          "dF/F robust range (p5–p95)", None),
        ("event_rate_hz",                  "Event rate (Hz)",           None),
        ("dff_frac_nan",                   "Fraction NaN in dF/F",      (0.0, 1.0)),
        ("dff_drift_delta_last10_first10", "Slow dF/F drift",           None),
    ]

    soma_mask = roi_qc["is_soma"].astype(bool)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (col, label, xlim) in zip(axes.flat, METRICS):
        if col not in roi_qc.columns:
            ax.set_visible(False)
            continue
        vals_all  = pd.to_numeric(roi_qc[col],                  errors="coerce").dropna()
        vals_soma = pd.to_numeric(roi_qc.loc[soma_mask, col],   errors="coerce").dropna()
        bins = np.linspace(vals_all.min(), vals_all.max(), 50) if len(vals_all) > 1 else 30
        ax.hist(vals_all,  bins=bins, alpha=0.5, label="all ROIs",  color="steelblue")
        ax.hist(vals_soma, bins=bins, alpha=0.7, label="soma only", color="darkorange")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("ROI count", fontsize=9)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig


def plot_plane_summary(
    plane_summary: pd.DataFrame,
    title: str = "Per-plane summary",
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar charts of ROI yield, signal amplitude, and activity level per plane."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    x      = np.arange(len(plane_summary))
    labels = plane_summary["plane"].astype(str).tolist()

    axes[0].bar(x, plane_summary["n_rois"], color="steelblue")
    axes[0].bar(x, plane_summary["n_soma"], color="darkorange", label="soma")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("count"); axes[0].set_title("ROI yield"); axes[0].legend()

    axes[1].bar(x, plane_summary["dff_std_median"], color="mediumseagreen")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("median dF/F std"); axes[1].set_title("Signal amplitude")

    axes[2].bar(x, plane_summary["event_rate_hz_median"], color="orchid")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].set_ylabel("median event rate (Hz)"); axes[2].set_title("Activity level")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig


# ---------------------------------------------------------------------------
# Visualization — multi-session
# ---------------------------------------------------------------------------

def plot_dataset_overview(
    agg_df: pd.DataFrame,
    title: str = "Dataset overview",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Overview scatter/distribution plots across all sessions.

    Expects columns produced by aggregate_results():
        dff_std_median, event_rate_hz_median, soma_frac, n_rois_total, error
    """
    ok = agg_df[agg_df["error"].isna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].hist(ok["dff_std_median"].dropna(),      bins=20, color="steelblue")
    axes[0].set_xlabel("median dF/F std");  axes[0].set_ylabel("sessions")
    axes[0].set_title("Signal amplitude")

    axes[1].hist(ok["event_rate_hz_median"].dropna(), bins=20, color="orchid")
    axes[1].set_xlabel("median event rate (Hz)"); axes[1].set_ylabel("sessions")
    axes[1].set_title("Activity level")

    axes[2].scatter(ok["n_rois_total"], ok["soma_frac"], alpha=0.6, color="darkorange")
    axes[2].set_xlabel("total ROIs"); axes[2].set_ylabel("soma fraction")
    axes[2].set_title("ROI yield vs soma fraction")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig