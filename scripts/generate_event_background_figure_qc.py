#!/usr/bin/env python3
"""Generate figure-oriented event/background QC assets for a materialized session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mesoscope_snr import (  # noqa: E402
    calculate_roi_snr_metrics,
    event_cluster_amplitude_table,
    event_composition_kmeans,
    event_composition_labels,
    long_event_window_flags,
    roi_event_composition_from_cluster_table,
    trace_threshold_event_metrics,
)


def available_planes(session_dir: Path) -> list[str]:
    return sorted(
        p.name for p in session_dir.iterdir() if p.is_dir() and (p / "dff.npy").exists()
    )


def load_plane_arrays(session_dir: Path, plane: str, max_frames: int | None = None):
    plane_dir = session_dir / plane
    dff = np.load(plane_dir / "dff.npy", mmap_mode="r")
    events_path = plane_dir / "events.npy"
    events = np.load(events_path, mmap_mode="r") if events_path.exists() else None
    timestamps = np.load(plane_dir / "dff_timestamps.npy")
    if max_frames is not None:
        n = min(int(max_frames), dff.shape[0])
        dff = dff[:n]
        events = None if events is None else events[:n]
        timestamps = timestamps[:n]
    return np.asarray(dff), None if events is None else np.asarray(events), np.asarray(timestamps)


def _read_optional_roi_metadata(session_dir: Path, plane: str) -> pd.DataFrame:
    for name in ["roi_metadata.csv", "roi_table.csv"]:
        path = session_dir / plane / name
        if path.exists():
            table = pd.read_csv(path)
            if "roi_index" not in table.columns:
                table = table.reset_index(names="roi_index")
            keep = [
                col
                for col in [
                    "roi_index",
                    "roi_area_pixels",
                    "soma_probability",
                    "dendrite_probability",
                    "is_soma",
                    "is_dendrite",
                    "roi_classifier_confidence",
                    "roi_classifier_margin",
                ]
                if col in table.columns
            ]
            return table[keep].copy()
    return pd.DataFrame(columns=["roi_index"])


def _read_optional_pixel_mask_area(session_dir: Path, plane: str) -> pd.DataFrame:
    plane_dir = session_dir / plane
    for name in ["pixel_masks.npz", "roi_pixel_masks.npz"]:
        path = plane_dir / name
        if not path.exists():
            continue
        masks = np.load(path, allow_pickle=True)
        if {"indptr", "indices"}.issubset(masks.files):
            indptr = np.asarray(masks["indptr"], dtype=int)
            return pd.DataFrame(
                {
                    "roi_index": np.arange(max(0, len(indptr) - 1), dtype=int),
                    "roi_area_pixels": np.diff(indptr).astype(float),
                }
            )
        if "pixel_masks" in masks.files:
            rows = []
            for roi_index, pix in enumerate(masks["pixel_masks"]):
                rows.append({"roi_index": roi_index, "roi_area_pixels": len(pix)})
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=["roi_index", "roi_area_pixels"])


def attach_plane_context(session_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    context_rows = []
    for plane in sorted(out["plane"].dropna().astype(str).unique()):
        meta = _read_optional_roi_metadata(session_dir, plane)
        area = _read_optional_pixel_mask_area(session_dir, plane)
        context = meta.merge(area, on="roi_index", how="outer", suffixes=("", "_from_mask"))
        if "roi_area_pixels_from_mask" in context.columns:
            context["roi_area_pixels"] = context.get("roi_area_pixels").combine_first(
                context["roi_area_pixels_from_mask"]
            )
            context = context.drop(columns=["roi_area_pixels_from_mask"])
        if not context.empty:
            context["plane"] = plane
            context_rows.append(context)
    if context_rows:
        context = pd.concat(context_rows, ignore_index=True)
        out = out.merge(context, on=["plane", "roi_index"], how="left", suffixes=("", "_context"))
        for col in list(out.columns):
            if col.endswith("_context"):
                base = col[: -len("_context")]
                out[base] = out[base].combine_first(out[col]) if base in out else out[col]
                out = out.drop(columns=[col])
    return out


def add_trace_threshold_metrics(session_dir: Path, metrics: pd.DataFrame, max_frames: int | None) -> pd.DataFrame:
    needed = "trace_event_fraction_gt_4sd"
    if needed in metrics.columns and pd.to_numeric(metrics[needed], errors="coerce").notna().any():
        return metrics
    rows = []
    for plane in sorted(metrics["plane"].dropna().astype(str).unique()):
        print(f"[INFO] trace-threshold metrics for {plane}", flush=True)
        dff, _, timestamps = load_plane_arrays(session_dir, plane, max_frames=max_frames)
        for roi_index in range(dff.shape[1]):
            row = {"plane": plane, "roi_index": roi_index}
            row.update(trace_threshold_event_metrics(dff[:, roi_index], timestamps))
            rows.append(row)
    trace_metrics = pd.DataFrame(rows)
    drop_cols = [col for col in trace_metrics.columns if col in metrics.columns and col not in {"plane", "roi_index"}]
    metrics = metrics.drop(columns=drop_cols)
    return metrics.merge(trace_metrics, on=["plane", "roi_index"], how="left")


def compute_metrics(session_dir: Path, planes: list[str], max_frames: int | None) -> pd.DataFrame:
    rows = []
    for plane in planes:
        print(f"[INFO] ROI metrics for {plane}", flush=True)
        dff, events, timestamps = load_plane_arrays(session_dir, plane, max_frames=max_frames)
        table = calculate_roi_snr_metrics(dff, timestamps=timestamps, events=events)
        table.insert(0, "plane", plane)
        table.insert(0, "session", session_dir.name)
        table.insert(0, "session_dir", str(session_dir))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def compute_cluster_table(
    session_dir: Path,
    planes: list[str],
    metrics: pd.DataFrame,
    max_frames: int | None,
) -> pd.DataFrame:
    rows = []
    for plane in planes:
        print(f"[INFO] event clusters for {plane}", flush=True)
        dff, events, timestamps = load_plane_arrays(session_dir, plane, max_frames=max_frames)
        if events is None:
            continue
        plane_metrics = metrics.loc[metrics["plane"].astype(str) == plane]
        rows.append(
            event_cluster_amplitude_table(
                dff,
                events,
                timestamps,
                roi_metrics=plane_metrics,
                plane=plane,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_figure_metrics(
    session_dir: Path,
    output_dir: Path,
    *,
    metrics_csv: Path | None = None,
    cluster_csv: Path | None = None,
    planes: list[str] | None = None,
    max_frames: int | None = None,
) -> tuple[Path, Path]:
    planes = planes or available_planes(session_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = session_dir.name
    metrics_out = output_dir / f"{stem}_figure_event_qc_metrics.csv"
    clusters_out = output_dir / f"{stem}_figure_event_clusters.csv"

    if metrics_csv and metrics_csv.exists():
        metrics = pd.read_csv(metrics_csv)
    else:
        metrics = compute_metrics(session_dir, planes, max_frames)
    metrics = metrics.loc[metrics["plane"].astype(str).isin(planes)].copy()
    metrics = add_trace_threshold_metrics(session_dir, metrics, max_frames=max_frames)
    metrics = attach_plane_context(session_dir, metrics)

    if cluster_csv and cluster_csv.exists():
        clusters = pd.read_csv(cluster_csv)
    else:
        clusters = compute_cluster_table(session_dir, planes, metrics, max_frames=max_frames)
    if not clusters.empty:
        clusters.to_csv(clusters_out, index=False)
        composition = roi_event_composition_from_cluster_table(clusters)
        flags = long_event_window_flags(clusters)
        metrics = metrics.merge(composition, on=["plane", "roi_index"], how="left", suffixes=("", "_cluster"))
        metrics = metrics.merge(flags, on=["plane", "roi_index"], how="left", suffixes=("", "_flag"))

    metrics = event_composition_labels(metrics)
    metrics, _ = event_composition_kmeans(metrics)
    nonlong_cols = [
        "nonlong_event_fraction_lt_2sd",
        "nonlong_event_fraction_2_4sd",
        "nonlong_event_fraction_gt_4sd",
    ]
    if all(col in metrics.columns for col in nonlong_cols):
        original = metrics[["background_event_fraction_lt_2sd", "background_event_fraction_2_4sd", "background_event_fraction_gt_4sd"]].copy()
        metrics_for_nonlong = metrics.rename(
            columns={
                "nonlong_event_fraction_lt_2sd": "background_event_fraction_lt_2sd",
                "nonlong_event_fraction_2_4sd": "background_event_fraction_2_4sd",
                "nonlong_event_fraction_gt_4sd": "background_event_fraction_gt_4sd",
            }
        )
        metrics_for_nonlong, _ = event_composition_kmeans(metrics_for_nonlong)
        metrics["nonlong_event_composition_cluster"] = metrics_for_nonlong["event_composition_cluster"]
        metrics[["background_event_fraction_lt_2sd", "background_event_fraction_2_4sd", "background_event_fraction_gt_4sd"]] = original

    metrics.to_csv(metrics_out, index=False)
    return metrics_out, clusters_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/background_event_qc"), type=Path)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--cluster-csv", type=Path)
    parser.add_argument("--plane", action="append", dest="planes")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    metrics_out, clusters_out = build_figure_metrics(
        args.session_dir.expanduser().resolve(),
        args.output_dir,
        metrics_csv=args.metrics_csv,
        cluster_csv=args.cluster_csv,
        planes=args.planes,
        max_frames=args.max_frames,
    )
    print(metrics_out)
    print(clusters_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
