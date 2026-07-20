#!/usr/bin/env python3
"""Generate 3D ROI event-composition plots with long clusters separated."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mesoscope_snr import (  # noqa: E402
    event_cluster_amplitude_table,
    event_composition_kmeans,
    roi_event_composition_from_cluster_table,
)


def _load_cached_plane(
    session_dir: Path,
    plane: str,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plane_dir = session_dir / plane
    dff = np.load(plane_dir / "dff.npy", mmap_mode="r")
    events = np.load(plane_dir / "events.npy", mmap_mode="r")
    timestamps = np.load(plane_dir / "dff_timestamps.npy", mmap_mode="r")
    dff = dff if dff.shape[0] >= dff.shape[1] else dff.T
    events = events if events.shape[0] >= events.shape[1] else events.T
    n_frames = dff.shape[0] if max_frames is None else min(int(max_frames), dff.shape[0])
    return (
        np.asarray(dff[:n_frames], dtype=np.float32),
        np.asarray(events[:n_frames, : dff.shape[1]], dtype=np.float32),
        np.asarray(timestamps[:n_frames], dtype=float),
    )


def _planes_from_session(session_dir: Path, requested: list[str] | None) -> list[str]:
    available = sorted(
        p.name for p in session_dir.iterdir() if p.is_dir() and (p / "dff.npy").exists()
    )
    if not requested:
        return available
    return [plane for plane in requested if plane in available]


def _cluster_adjusted_composition(
    composition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rename_for_kmeans = {
        "nonlong_event_fraction_lt_2sd": "background_event_fraction_lt_2sd",
        "nonlong_event_fraction_2_4sd": "background_event_fraction_2_4sd",
        "nonlong_event_fraction_gt_4sd": "background_event_fraction_gt_4sd",
    }
    work = composition.rename(columns=rename_for_kmeans)
    clustered, centroids = event_composition_kmeans(work, n_clusters=3)
    composition = composition.copy()
    composition["nonlong_event_composition_cluster"] = clustered[
        "event_composition_cluster"
    ]
    centroids = centroids.rename(
        columns={
            "event_composition_cluster": "nonlong_event_composition_cluster",
            "centroid_fraction_lt_2sd": "centroid_nonlong_fraction_lt_2sd",
            "centroid_fraction_2_4sd": "centroid_nonlong_fraction_2_4sd",
            "centroid_fraction_gt_4sd": "centroid_nonlong_fraction_gt_4sd",
        }
    )
    return composition, centroids


def _plot_3d(composition: pd.DataFrame, output_path: Path) -> None:
    plot_df = composition.dropna(
        subset=[
            "nonlong_event_fraction_lt_2sd",
            "nonlong_event_fraction_2_4sd",
            "nonlong_event_fraction_gt_4sd",
            "nonlong_event_composition_cluster",
        ]
    ).copy()
    colors = {0: "#4575b4", 1: "#6a994e", 2: "#d73027"}
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    for cluster, group in plot_df.groupby("nonlong_event_composition_cluster"):
        cluster_id = int(cluster)
        edgecolors = np.where(group["has_long_gt_3s_event_cluster"], "#111827", "none")
        linewidths = np.where(group["has_long_gt_3s_event_cluster"], 0.8, 0.0)
        ax.scatter(
            group["nonlong_event_fraction_lt_2sd"],
            group["nonlong_event_fraction_2_4sd"],
            group["nonlong_event_fraction_gt_4sd"],
            s=28,
            alpha=0.72,
            color=colors.get(cluster_id, "#64748b"),
            edgecolors=edgecolors,
            linewidths=linewidths,
            label=f"cluster {cluster_id} (n={len(group)})",
        )

    long_n = int(plot_df["has_long_gt_3s_event_cluster"].sum())
    ax.set_xlabel("<2 SD fraction, non-long events")
    ax.set_ylabel("2-4 SD fraction, non-long events")
    ax.set_zlabel(">=4 SD fraction, non-long events")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=24, azim=42)
    ax.set_title(
        "ROI event composition excluding >=3 s event clusters\n"
        f"black outline: ROI has >=1 long cluster (n={long_n})"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(
    session_dir: Path,
    metrics_csv: Path,
    output_dir: Path,
    *,
    planes: list[str] | None = None,
    max_frames: int | None = None,
    long_span_s: float = 3.0,
) -> dict[str, Path]:
    metrics = pd.read_csv(metrics_csv)
    plane_names = _planes_from_session(session_dir, planes)
    cluster_tables = []
    for plane in plane_names:
        print(f"[INFO] classifying clusters for {plane}", flush=True)
        dff, events, timestamps = _load_cached_plane(session_dir, plane, max_frames=max_frames)
        plane_metrics = metrics.loc[metrics["plane"].astype(str) == str(plane)]
        table = event_cluster_amplitude_table(
            dff,
            events,
            timestamps,
            roi_metrics=plane_metrics,
            plane=plane,
            long_span_s=long_span_s,
        )
        table.insert(0, "session", session_dir.name)
        cluster_tables.append(table)

    cluster_table = pd.concat(cluster_tables, ignore_index=True) if cluster_tables else pd.DataFrame()
    composition = roi_event_composition_from_cluster_table(
        cluster_table,
        group_cols=("plane", "roi_index"),
    )
    composition.insert(0, "session", session_dir.name)
    composition, centroids = _cluster_adjusted_composition(composition)
    annotated = metrics.merge(
        composition,
        on=["session", "plane", "roi_index"],
        how="left",
        suffixes=("", "_cluster_adjusted"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = session_dir.name
    cluster_csv = output_dir / f"{stem}_event_clusters_with_amplitude_classes.csv"
    composition_csv = output_dir / f"{stem}_nonlong_event_composition_by_roi.csv"
    annotated_csv = output_dir / f"{stem}_roi_metrics_with_nonlong_event_composition.csv"
    centroids_csv = output_dir / f"{stem}_nonlong_event_composition_centroids.csv"
    png = output_dir / f"{stem}_nonlong_event_composition_3d.png"

    cluster_table.to_csv(cluster_csv, index=False)
    composition.to_csv(composition_csv, index=False)
    annotated.to_csv(annotated_csv, index=False)
    centroids.to_csv(centroids_csv, index=False)
    _plot_3d(composition, png)
    return {
        "cluster_csv": cluster_csv,
        "composition_csv": composition_csv,
        "annotated_csv": annotated_csv,
        "centroids_csv": centroids_csv,
        "png": png,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--output-dir", default="outputs/background_event_qc", type=Path)
    parser.add_argument("--plane", action="append", dest="planes")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--long-span-s", type=float, default=3.0)
    args = parser.parse_args()
    outputs = run(
        args.session_dir.expanduser().resolve(),
        args.metrics_csv.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        planes=args.planes,
        max_frames=args.max_frames,
        long_span_s=args.long_span_s,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
