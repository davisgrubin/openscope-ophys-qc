#!/usr/bin/env python3
"""
Mesoscope session QC pipeline for the OpenScope Community Predictive Processing project.

Purpose
-------
Load a processed Mesoscope planar-ophys NWB/Zarr session from the public
s3://aind-open-data bucket, extract ROI/plane metadata, compute basic trace QC,
and write session-level CSV/JSON summaries and figures.

This is intentionally a validation-first pipeline:
    1. Can we open the session?
    2. How many ROIs/putative somata are present per plane?
    3. Are ROI sizes/soma probabilities plausible?
    4. What do dF/F distributions and event rates look like?
    5. Are there missing timestamps, NaNs, or obvious plane/session failures?

Example
-------
python mesoscope_qc_pipeline.py \
    --s3-path multiplane-ophys_837568_2026-03-05_14-14-51_processed_2026-03-06_11-31-22 \
    --out outputs/837568_20260305 \
    --max-frames 10000 \
    --make-plots

For full-session metrics, omit --max-frames.

Notes
-----
- Pass the processed planar-ophys asset folder, not the raw asset.
- The planar-ophys asset is usually the processed folder containing VISp_*/VISl_* folders
  and pophys.nwb.zarr.
- Stimulus/trial alignment may require the behavior/timing partial NWB or a complete NWB,
  depending on which assets are currently available for the session.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import boto3
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import remfile
from botocore import UNSIGNED
from botocore.config import Config
from pynwb import NWBHDF5IO

S3_BUCKET = "aind-open-data"


@dataclass
class OpenResult:
    nwb: Any
    io: Any
    nwb_s3_path: str
    is_zarr: bool


def get_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def normalize_s3_path(s3_path: str, default_bucket: str = S3_BUCKET) -> tuple[str, str]:
    """Return bucket, key_or_prefix from s3://bucket/key or plain key."""
    s3_path = s3_path.strip().rstrip("/")
    if s3_path.startswith("s3://"):
        without = s3_path[len("s3://") :]
        bucket, _, key = without.partition("/")
        return bucket, key.rstrip("/")
    return default_bucket, s3_path.rstrip("/")


def list_all_keys(prefix: str, bucket: str = S3_BUCKET) -> list[str]:
    prefix = prefix.rstrip("/") + "/"
    paginator = get_s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def find_nwb_key(keys: Iterable[str]) -> tuple[str | None, bool | None]:
    """
    Find an NWB file/folder in a list of S3 keys.

    Preference order:
    1. pophys.nwb.zarr
    2. any *.nwb.zarr
    3. any *.nwb/ Zarr-style folder
    4. any HDF5 *.nwb file
    """
    zarr_prefixes: set[str] = set()
    hdf5_keys: list[str] = []

    for key in keys:
        if ".nwb.zarr/" in key:
            zarr_prefixes.add(key[: key.index(".nwb.zarr/") + len(".nwb.zarr")])
        elif ".nwb/" in key:
            zarr_prefixes.add(key[: key.index(".nwb/") + len(".nwb")])
        elif key.endswith(".nwb"):
            hdf5_keys.append(key)

    if zarr_prefixes:
        preferred = sorted(
            zarr_prefixes,
            key=lambda k: (0 if k.endswith("pophys.nwb.zarr") else 1, k),
        )[0]
        return preferred, True

    if hdf5_keys:
        return sorted(hdf5_keys)[0], False

    return None, None


def open_nwb_from_s3(s3_path: str, bucket: str = S3_BUCKET) -> OpenResult:
    """
    Open an NWB from S3.

    s3_path may be:
    - a processed asset folder containing an NWB/Zarr folder
    - a direct path to an NWB file/folder
    - a full s3://bucket/path
    """
    bucket, key = normalize_s3_path(s3_path, bucket)

    direct_is_zarr = key.endswith(".nwb.zarr") or key.endswith(".nwb")
    if direct_is_zarr:
        nwb_key = key
        is_zarr = key.endswith(".zarr") or key.endswith(".nwb")
        # Ambiguous: .nwb can be HDF5 file or Zarr folder. Check by listing.
        if key.endswith(".nwb") and not key.endswith(".zarr"):
            keys = list_all_keys(key, bucket=bucket)
            if keys:
                is_zarr = True
            else:
                is_zarr = False
    else:
        keys = list_all_keys(key, bucket=bucket)
        nwb_key, is_zarr = find_nwb_key(keys)
        if nwb_key is None:
            raise FileNotFoundError(f"No NWB found under s3://{bucket}/{key}")

    if is_zarr:
        from hdmf_zarr import NWBZarrIO

        io = NWBZarrIO(
            path=f"s3://{bucket}/{nwb_key}",
            mode="r",
            load_namespaces=True,
            storage_options={"anon": True},
        )
    else:
        url = f"https://s3.amazonaws.com/{bucket}/{nwb_key}"
        rem = remfile.File(url)
        h5_file = h5py.File(rem, "r")
        io = NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)

    nwb = io.read()
    return OpenResult(
        nwb=nwb,
        io=io,
        nwb_s3_path=f"s3://{bucket}/{nwb_key}",
        is_zarr=bool(is_zarr),
    )


def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, attr)
        if value is None:
            return default
        return value
    except Exception:
        return default


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)


def parse_structure_depth(location: Any) -> tuple[str | None, float | None]:
    """
    Parse plane.location strings like:
        'Structure: VISp Depth: 60'
    """
    text = str(location) if location is not None else ""
    structure = None
    depth = None

    structure_match = re.search(r"Structure:\s*([A-Za-z0-9_/-]+)", text)
    if structure_match:
        structure = structure_match.group(1)

    depth_match = re.search(r"Depth:\s*([-+]?\d+\.?\d*)", text)
    if depth_match:
        depth = float(depth_match.group(1))

    return structure, depth


def get_plane_names(nwb: Any) -> list[str]:
    if not hasattr(nwb, "imaging_planes") or nwb.imaging_planes is None:
        return []
    return sorted(list(nwb.imaging_planes.keys()))


def get_plane_metadata(nwb: Any) -> pd.DataFrame:
    rows = []
    for plane_name in get_plane_names(nwb):
        plane = nwb.imaging_planes[plane_name]
        structure, depth_um = parse_structure_depth(safe_getattr(plane, "location"))
        rows.append(
            {
                "plane": plane_name,
                "structure": structure,
                "depth_um": depth_um,
                "location": str(safe_getattr(plane, "location", "")),
                "imaging_rate_hz": safe_getattr(plane, "imaging_rate", np.nan),
                "excitation_lambda_nm": safe_getattr(plane, "excitation_lambda", np.nan),
                "indicator": str(safe_getattr(plane, "indicator", "")),
                "device": str(safe_getattr(safe_getattr(plane, "device", None), "name", "")),
            }
        )
    return pd.DataFrame(rows)


def read_small_column(roi_table: Any, colname: str, n: int, default: Any = np.nan) -> np.ndarray:
    if colname not in list(getattr(roi_table, "colnames", [])):
        return np.full(n, default)
    try:
        col = roi_table[colname].data
        return np.asarray(col[:])
    except Exception:
        try:
            return np.asarray(roi_table[colname][:])
        except Exception:
            return np.full(n, default)


def compute_mask_geometry(roi_table: Any, n: int) -> pd.DataFrame:
    """
    Compute ROI area and centroid from image_mask without bulk-loading all masks.
    This can be slow over S3, so it is optional.
    """
    if "image_mask" not in list(getattr(roi_table, "colnames", [])):
        return pd.DataFrame(
            {
                "roi_area_pix": np.full(n, np.nan),
                "roi_centroid_x_pix": np.full(n, np.nan),
                "roi_centroid_y_pix": np.full(n, np.nan),
            }
        )

    masks = roi_table["image_mask"].data
    areas = np.full(n, np.nan)
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)

    for i in range(n):
        try:
            mask = np.asarray(masks[i])
            valid = mask > 0
            areas[i] = float(np.count_nonzero(valid))
            if areas[i] > 0:
                yy, xx = np.nonzero(valid)
                weights = mask[yy, xx].astype(float)
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    xs[i] = float(np.sum(xx * weights) / weight_sum)
                    ys[i] = float(np.sum(yy * weights) / weight_sum)
                else:
                    xs[i] = float(np.mean(xx))
                    ys[i] = float(np.mean(yy))
        except Exception:
            continue

    return pd.DataFrame(
        {
            "roi_area_pix": areas,
            "roi_centroid_x_pix": xs,
            "roi_centroid_y_pix": ys,
        }
    )


def get_roi_metadata_for_plane(nwb: Any, plane_name: str, load_masks: bool = False) -> pd.DataFrame:
    proc = nwb.processing[plane_name]
    roi_table = proc["image_segmentation"]["roi_table"]
    n = len(roi_table)

    df = pd.DataFrame(
        {
            "plane": plane_name,
            "roi_index": np.arange(n, dtype=int),
            "is_soma": read_small_column(roi_table, "is_soma", n, default=False).astype(bool),
            "is_dendrite": read_small_column(roi_table, "is_dendrite", n, default=False).astype(bool),
            "soma_probability": read_small_column(roi_table, "soma_probability", n, default=np.nan),
            "dendrite_probability": read_small_column(roi_table, "dendrite_probability", n, default=np.nan),
        }
    )

    # Pull in any other scalar columns if present, while skipping large masks.
    skip = {
        "image_mask",
        "pixel_mask",
        "voxel_mask",
        "is_soma",
        "is_dendrite",
        "soma_probability",
        "dendrite_probability",
    }
    for col in list(getattr(roi_table, "colnames", [])):
        if col in skip:
            continue
        try:
            arr = read_small_column(roi_table, col, n, default=np.nan)
            if arr.ndim == 1 and len(arr) == n:
                df[col] = arr
        except Exception:
            pass

    if load_masks:
        geom = compute_mask_geometry(roi_table, n)
        df = pd.concat([df, geom], axis=1)
    else:
        df["roi_area_pix"] = np.nan
        df["roi_centroid_x_pix"] = np.nan
        df["roi_centroid_y_pix"] = np.nan

    return df


def get_timeseries_from_proc(proc: Any, paths: list[tuple[str, ...]]) -> Any | None:
    for path in paths:
        obj = proc
        try:
            for key in path:
                obj = obj[key]
            if hasattr(obj, "data"):
                return obj
        except Exception:
            continue
    return None


def load_timeseries_matrix(series: Any, max_frames: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = series.data
    n_time = data.shape[0]
    stop = min(n_time, max_frames) if max_frames is not None else n_time
    arr = np.asarray(data[:stop, :], dtype=np.float32)

    try:
        ts = np.asarray(series.timestamps[:stop], dtype=float)
    except Exception:
        ts = np.arange(stop, dtype=float)

    return arr, ts


def robust_trace_qc(dff: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else np.nan
    sampling_rate_hz = float(1.0 / np.nanmedian(np.diff(timestamps))) if len(timestamps) > 2 else np.nan

    finite = np.isfinite(dff)
    frac_nan = 1.0 - np.mean(finite, axis=0)

    p01, p05, p50, p95, p99 = np.nanpercentile(dff, [1, 5, 50, 95, 99], axis=0)
    mean = np.nanmean(dff, axis=0)
    std = np.nanstd(dff, axis=0)
    mad = np.nanmedian(np.abs(dff - p50[None, :]), axis=0)

    # Simple slow drift estimate: compare first and last 10% medians.
    n = dff.shape[0]
    k = max(1, int(0.1 * n))
    first_med = np.nanmedian(dff[:k, :], axis=0)
    last_med = np.nanmedian(dff[-k:, :], axis=0)
    drift_delta = last_med - first_med

    return pd.DataFrame(
        {
            "duration_s": duration_s,
            "sampling_rate_hz": sampling_rate_hz,
            "n_timepoints_loaded": dff.shape[0],
            "dff_mean": mean,
            "dff_std": std,
            "dff_mad": mad,
            "dff_p01": p01,
            "dff_p05": p05,
            "dff_median": p50,
            "dff_p95": p95,
            "dff_p99": p99,
            "dff_robust_range_5_95": p95 - p05,
            "dff_frac_nan": frac_nan,
            "dff_drift_delta_last10_first10": drift_delta,
        }
    )


def event_qc(events: np.ndarray, timestamps: np.ndarray, threshold: float = 0.0) -> pd.DataFrame:
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else np.nan
    is_event = np.isfinite(events) & (events > threshold)
    event_counts = np.sum(is_event, axis=0)
    event_rate_hz = event_counts / duration_s if duration_s and duration_s > 0 else np.full(events.shape[1], np.nan)

    # Mean amplitude over detected event samples only.
    event_amp_mean = np.full(events.shape[1], np.nan)
    for j in range(events.shape[1]):
        vals = events[is_event[:, j], j]
        if vals.size:
            event_amp_mean[j] = float(np.nanmean(vals))

    return pd.DataFrame(
        {
            "event_count": event_counts,
            "event_rate_hz": event_rate_hz,
            "event_amplitude_mean": event_amp_mean,
        }
    )


def fluorescence_source_qc(proc: Any, max_frames: int | None = None) -> pd.DataFrame | None:
    """
    Optional check comparing raw, neuropil, and neuropil-corrected fluorescence.
    This is useful but costs extra S3 reads.
    """
    raw = get_timeseries_from_proc(proc, [("raw_timeseries", "ROI_fluorescence_timeseries")])
    neuropil = get_timeseries_from_proc(proc, [("neuropil_fluorescence_timeseries",)])
    corrected = get_timeseries_from_proc(proc, [("neuropil_corrected_timeseries",)])

    if raw is None or neuropil is None:
        return None

    raw_arr, _ = load_timeseries_matrix(raw, max_frames=max_frames)
    neuropil_arr, _ = load_timeseries_matrix(neuropil, max_frames=max_frames)

    eps = 1e-12
    raw_med = np.nanmedian(raw_arr, axis=0)
    neuropil_med = np.nanmedian(neuropil_arr, axis=0)
    out = pd.DataFrame(
        {
            "raw_fluorescence_median": raw_med,
            "neuropil_fluorescence_median": neuropil_med,
            "neuropil_to_raw_median_ratio": neuropil_med / (raw_med + eps),
        }
    )

    if corrected is not None:
        corrected_arr, _ = load_timeseries_matrix(corrected, max_frames=max_frames)
        out["neuropil_corrected_median"] = np.nanmedian(corrected_arr, axis=0)

    return out


def analyze_plane(
    nwb: Any,
    plane_name: str,
    load_masks: bool = False,
    max_frames: int | None = None,
    event_threshold: float = 0.0,
    include_fluorescence_sources: bool = False,
) -> pd.DataFrame:
    proc = nwb.processing[plane_name]
    roi_df = get_roi_metadata_for_plane(nwb, plane_name, load_masks=load_masks)

    dff_series = get_timeseries_from_proc(proc, [("dff_timeseries", "dff_timeseries"), ("dff_timeseries",)])
    if dff_series is None:
        raise KeyError(f"No dF/F timeseries found for plane {plane_name}")

    dff, dff_ts = load_timeseries_matrix(dff_series, max_frames=max_frames)
    trace_df = robust_trace_qc(dff, dff_ts)

    out = pd.concat([roi_df.reset_index(drop=True), trace_df.reset_index(drop=True)], axis=1)

    event_series = get_timeseries_from_proc(proc, [("event_timeseries",), ("events", "event_timeseries")])
    if event_series is not None:
        events, event_ts = load_timeseries_matrix(event_series, max_frames=max_frames)
        events_df = event_qc(events, event_ts, threshold=event_threshold)
        out = pd.concat([out.reset_index(drop=True), events_df.reset_index(drop=True)], axis=1)
    else:
        out["event_count"] = np.nan
        out["event_rate_hz"] = np.nan
        out["event_amplitude_mean"] = np.nan

    if include_fluorescence_sources:
        fl_df = fluorescence_source_qc(proc, max_frames=max_frames)
        if fl_df is not None:
            out = pd.concat([out.reset_index(drop=True), fl_df.reset_index(drop=True)], axis=1)

    return out


def summarize_planes(roi_qc: pd.DataFrame, plane_meta: pd.DataFrame) -> pd.DataFrame:
    summary = (
        roi_qc.groupby("plane", dropna=False)
        .agg(
            n_rois=("roi_index", "count"),
            n_soma=("is_soma", "sum"),
            n_dendrite=("is_dendrite", "sum"),
            soma_probability_mean=("soma_probability", "mean"),
            soma_probability_median=("soma_probability", "median"),
            roi_area_pix_median=("roi_area_pix", "median"),
            dff_std_median=("dff_std", "median"),
            dff_robust_range_5_95_median=("dff_robust_range_5_95", "median"),
            dff_frac_nan_max=("dff_frac_nan", "max"),
            event_rate_hz_median=("event_rate_hz", "median"),
            event_rate_hz_p95=("event_rate_hz", lambda x: np.nanpercentile(x, 95)),
            duration_s=("duration_s", "median"),
            sampling_rate_hz=("sampling_rate_hz", "median"),
        )
        .reset_index()
    )
    if not plane_meta.empty:
        summary = summary.merge(plane_meta, on="plane", how="left")
    return summary


def extract_session_metadata(open_result: OpenResult) -> dict[str, Any]:
    nwb = open_result.nwb
    subj = safe_getattr(nwb, "subject", None)
    meta = {
        "nwb_s3_path": open_result.nwb_s3_path,
        "is_zarr": open_result.is_zarr,
        "session_id": safe_getattr(nwb, "session_id", None),
        "session_description": safe_getattr(nwb, "session_description", None),
        "session_start_time": safe_getattr(nwb, "session_start_time", None),
        "institution": safe_getattr(nwb, "institution", None),
        "experimenter": safe_getattr(nwb, "experimenter", None),
        "subject": {
            "subject_id": safe_getattr(subj, "subject_id", None),
            "species": safe_getattr(subj, "species", None),
            "sex": safe_getattr(subj, "sex", None),
            "age": safe_getattr(subj, "age", None),
            "genotype": safe_getattr(subj, "genotype", None),
            "date_of_birth": safe_getattr(subj, "date_of_birth", None),
        },
        "processing_modules": list(getattr(nwb, "processing", {}).keys()),
        "imaging_planes": get_plane_metadata(nwb).to_dict(orient="records"),
        "interval_tables": list(getattr(nwb, "intervals", {}).keys()) if hasattr(nwb, "intervals") else [],
    }
    return to_jsonable(meta)


def save_interval_tables(nwb: Any, out_dir: Path) -> list[str]:
    """
    Save any available NWB interval tables.
    For some partial planar-ophys NWBs, stimulus intervals may be absent; in that case
    this simply writes nothing.
    """
    written = []
    if not hasattr(nwb, "intervals") or nwb.intervals is None:
        return written

    intervals_dir = out_dir / "intervals"
    intervals_dir.mkdir(parents=True, exist_ok=True)

    for name in nwb.intervals.keys():
        try:
            df = nwb.intervals[name].to_dataframe()
            path = intervals_dir / f"{name}.csv"
            df.to_csv(path, index=True)
            written.append(str(path))
        except Exception as exc:
            print(f"[WARN] Could not save interval table {name}: {exc}")
    return written


def plot_histogram(df: pd.DataFrame, column: str, out_path: Path, title: str, bins: int = 60):
    vals = pd.to_numeric(df[column], errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=bins)
    plt.xlabel(column)
    plt.ylabel("ROI count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_plane_counts(plane_summary: pd.DataFrame, out_path: Path):
    if plane_summary.empty:
        return

    x = np.arange(len(plane_summary))
    plt.figure(figsize=(max(8, 0.75 * len(x)), 5))
    plt.bar(x, plane_summary["n_rois"].to_numpy())
    plt.xticks(x, plane_summary["plane"].astype(str).to_list(), rotation=45, ha="right")
    plt.ylabel("ROI count")
    plt.title("ROI yield by imaging plane")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_qc_plots(roi_qc: pd.DataFrame, plane_summary: pd.DataFrame, out_dir: Path):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_plane_counts(plane_summary, fig_dir / "plane_roi_counts.png")

    for col, title in [
        ("soma_probability", "Soma probability distribution"),
        ("roi_area_pix", "ROI area distribution"),
        ("dff_std", "dF/F standard deviation distribution"),
        ("dff_robust_range_5_95", "dF/F robust range distribution"),
        ("event_rate_hz", "Detected event-rate distribution"),
        ("dff_frac_nan", "Fraction NaN in dF/F traces"),
        ("dff_drift_delta_last10_first10", "Slow dF/F drift: last 10% minus first 10%"),
    ]:
        if col in roi_qc.columns:
            plot_histogram(roi_qc, col, fig_dir / f"{col}.png", title)


def run_session_qc(
    s3_path: str,
    out_dir: Path,
    planes: list[str] | None = None,
    load_masks: bool = False,
    max_frames: int | None = None,
    event_threshold: float = 0.0,
    include_fluorescence_sources: bool = False,
    make_plots: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    open_result = open_nwb_from_s3(s3_path)
    nwb = open_result.nwb

    try:
        session_meta = extract_session_metadata(open_result)
        (out_dir / "session_metadata.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")

        plane_meta = get_plane_metadata(nwb)
        plane_meta.to_csv(out_dir / "plane_metadata.csv", index=False)

        all_planes = get_plane_names(nwb)
        if planes is None or planes == ["all"]:
            planes_to_run = all_planes
        else:
            missing = sorted(set(planes) - set(all_planes))
            if missing:
                raise ValueError(f"Requested planes not present in NWB: {missing}. Available: {all_planes}")
            planes_to_run = planes

        roi_dfs = []
        for plane_name in planes_to_run:
            print(f"[INFO] Analyzing plane {plane_name}")
            try:
                plane_df = analyze_plane(
                    nwb,
                    plane_name,
                    load_masks=load_masks,
                    max_frames=max_frames,
                    event_threshold=event_threshold,
                    include_fluorescence_sources=include_fluorescence_sources,
                )
                roi_dfs.append(plane_df)
            except Exception as exc:
                print(f"[ERROR] Plane {plane_name} failed: {exc}")

        if not roi_dfs:
            raise RuntimeError("No planes were successfully analyzed.")

        roi_qc = pd.concat(roi_dfs, ignore_index=True)
        roi_qc = roi_qc.merge(plane_meta, on="plane", how="left")
        roi_qc.to_csv(out_dir / "roi_qc.csv", index=False)

        plane_summary = summarize_planes(roi_qc, plane_meta)
        plane_summary.to_csv(out_dir / "plane_qc.csv", index=False)

        interval_files = save_interval_tables(nwb, out_dir)

        if make_plots:
            make_qc_plots(roi_qc, plane_summary, out_dir)

        summary = {
            "out_dir": str(out_dir),
            "nwb_s3_path": open_result.nwb_s3_path,
            "session_id": session_meta.get("session_id"),
            "subject_id": session_meta.get("subject", {}).get("subject_id"),
            "planes_analyzed": planes_to_run,
            "n_planes_analyzed": len(planes_to_run),
            "n_rois_total": int(len(roi_qc)),
            "n_soma_total": int(np.nansum(roi_qc["is_soma"].astype(int))),
            "roi_qc_csv": str(out_dir / "roi_qc.csv"),
            "plane_qc_csv": str(out_dir / "plane_qc.csv"),
            "session_metadata_json": str(out_dir / "session_metadata.json"),
            "interval_files_written": interval_files,
        }
        (out_dir / "run_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
        return summary

    finally:
        try:
            open_result.io.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC a Mesoscope planar-ophys NWB/Zarr session from S3.")
    parser.add_argument(
        "--s3-path",
        required=True,
        help="Processed planar-ophys asset folder or direct NWB path in s3://aind-open-data.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--planes",
        nargs="+",
        default=["all"],
        help="Plane names to analyze, e.g. VISp_0 VISl_4. Default: all.",
    )
    parser.add_argument(
        "--load-masks",
        action="store_true",
        help="Compute ROI area/centroids from image masks. Useful but slower over S3.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only load the first N timepoints per plane for quick tests. Omit for full session.",
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.0,
        help="Threshold for counting detected event samples. Default: >0.",
    )
    parser.add_argument(
        "--include-fluorescence-sources",
        action="store_true",
        help="Also load raw/neuropil/corrected fluorescence and compute neuropil/raw metrics.",
    )
    parser.add_argument("--make-plots", action="store_true", help="Write basic QC PNG plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_session_qc(
        s3_path=args.s3_path,
        out_dir=Path(args.out),
        planes=args.planes,
        load_masks=args.load_masks,
        max_frames=args.max_frames,
        event_threshold=args.event_threshold,
        include_fluorescence_sources=args.include_fluorescence_sources,
        make_plots=args.make_plots,
    )
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
