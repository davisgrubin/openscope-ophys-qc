"""Batch ROI SNR summaries across sessions, planes, and days."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import mesoscope_qc_pipeline as qc
import mesoscope_qc_reports as qcr
from mesoscope_snr import calculate_roi_snr_metrics


DEFAULT_DOWNLOAD_DIR = Path(
    "/storage/scratch1/3/grubin6/openscope_ophys_qc_dandi_downloads"
)


@dataclass
class SnrBatchConfig:
    """Configuration for compact daily ROI metric extraction."""

    day: str
    planes: str | list[str] = "all"
    max_frames: int | None = None
    gaussian_sigma: float = 3.0
    consecutive_samples: int = 5
    exceptional_robust_std: bool = False
    event_threshold: float = 0.0
    baseline_bins: int = 10
    kernel_pre_s: float = 0.5
    kernel_post_s: float = 2.0
    max_kernel_events: int = 500
    min_events_for_distribution_fit: int = 20


def safe_name(value: Any) -> str:
    """Return a filesystem-safe name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return cleaned or "unknown"


def discover_local_sessions_for_day(
    day: str,
    search_dirs: Iterable[str | Path] = (DEFAULT_DOWNLOAD_DIR,),
) -> list[Path]:
    """Find local NWB files whose filename contains the requested YYYY-MM-DD day."""
    matches: list[Path] = []
    for search_dir in search_dirs:
        root = Path(search_dir).expanduser()
        if not root.exists():
            continue
        matches.extend(sorted(root.glob(f"*{day}*.nwb")))
    return sorted(dict.fromkeys(path.resolve() for path in matches))


def read_session_list(path: str | Path) -> list[str]:
    """Read session sources from a text file, one source per non-comment line."""
    sources = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            sources.append(stripped)
    return sources


def session_label(session_source: str | Path, nwb: Any) -> str:
    """Return a stable session label for saved tables."""
    session_id = getattr(nwb, "session_id", None)
    if session_id:
        return str(session_id)
    return Path(str(session_source).rstrip("/")).stem or str(session_source)


def session_date(nwb: Any) -> str:
    """Return the session date as YYYY-MM-DD when possible."""
    start = getattr(nwb, "session_start_time", None)
    if start is None:
        return "unknown"
    try:
        return pd.to_datetime(start).date().isoformat()
    except Exception:
        return str(start).split("T")[0]


def select_planes(available_planes: Iterable[str], requested: str | Iterable[str]) -> list[str]:
    """Resolve requested plane names against the session's available planes."""
    available = list(available_planes)
    if requested == "all":
        return available
    requested_list = list(requested)
    return [plane for plane in requested_list if plane in available]


def load_timeseries_column(
    series: Any,
    roi_index: int,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one ROI column and timestamps from an NWB TimeSeries-like object."""
    data = series.data
    n_time = data.shape[0]
    stop = min(n_time, max_frames) if max_frames is not None else n_time
    trace = np.asarray(data[:stop, int(roi_index)], dtype=np.float32)
    try:
        timestamps = np.asarray(series.timestamps[:stop], dtype=float)
    except Exception:
        timestamps = np.arange(stop, dtype=float)
    return trace, timestamps


def retrieve_roi_trace(
    session_source: str | Path,
    plane: str,
    roi_index: int,
    *,
    max_frames: int | None = None,
    include_events: bool = True,
) -> dict[str, np.ndarray | str | int | None]:
    """
    Retrieve full dF/F for a saved ROI identity without storing traces in summaries.

    Parameters identify the session source, plane, and ROI index saved in the
    compact ROI metrics table. The function reads only the requested ROI column.
    """
    session = qcr.open_session(session_source)
    try:
        nwb = session["nwb"]
        proc = nwb.processing[str(plane)]
        dff_series = qc.get_timeseries_from_proc(
            proc,
            [("dff_timeseries", "dff_timeseries"), ("dff_timeseries",)],
        )
        if dff_series is None:
            raise KeyError(f"No dF/F timeseries found for {plane}")
        dff, timestamps = load_timeseries_column(dff_series, roi_index, max_frames=max_frames)

        events = None
        if include_events:
            event_series = qc.get_timeseries_from_proc(
                proc,
                [("event_timeseries",), ("events", "event_timeseries")],
            )
            if event_series is not None:
                events, _ = load_timeseries_column(event_series, roi_index, max_frames=max_frames)

        return {
            "session_source": str(session_source),
            "session_id": session_label(session_source, nwb),
            "plane": str(plane),
            "roi_index": int(roi_index),
            "timestamps": timestamps,
            "dff": dff,
            "events": events,
        }
    finally:
        qcr.close_session(session)


def retrieve_roi_trace_from_stats(
    stats_csv: str | Path,
    *,
    session_id: str,
    plane: str,
    roi_index: int,
    max_frames: int | None = None,
    include_events: bool = True,
) -> dict[str, np.ndarray | str | int | None]:
    """Retrieve dF/F for an ROI selected from a saved compact stats CSV."""
    stats = pd.read_csv(stats_csv)
    match = stats.loc[
        (stats["session_id"].astype(str) == str(session_id))
        & (stats["plane"].astype(str) == str(plane))
        & (stats["roi_index"].astype(int) == int(roi_index))
    ]
    if match.empty:
        raise KeyError(f"No ROI row found for {session_id} {plane} ROI {roi_index}")
    session_source = match.iloc[0]["session_source"]
    return retrieve_roi_trace(
        session_source,
        plane,
        roi_index,
        max_frames=max_frames,
        include_events=include_events,
    )


def compute_session_metrics(
    session_source: str | Path,
    config: SnrBatchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Compute compact ROI metrics for one session without saving traces."""
    session_start = time.perf_counter()
    session = qcr.open_session(session_source)
    rows: list[pd.DataFrame] = []
    plane_inventory_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    try:
        nwb = session["nwb"]
        sid = session_label(session_source, nwb)
        date = session_date(nwb)
        plane_meta = qc.get_plane_metadata(nwb)
        available_planes = list(session.get("planes", qc.get_plane_names(nwb)))
        planes_to_run = select_planes(available_planes, config.planes)

        for plane_name in planes_to_run:
            plane_start = time.perf_counter()
            proc = nwb.processing[plane_name]
            dff_series = qc.get_timeseries_from_proc(
                proc,
                [("dff_timeseries", "dff_timeseries"), ("dff_timeseries",)],
            )
            if dff_series is None:
                timing_rows.append(
                    {
                        "session_id": sid,
                        "plane": plane_name,
                        "status": "missing_dff",
                        "elapsed_s": time.perf_counter() - plane_start,
                    }
                )
                continue

            dff, timestamps = qc.load_timeseries_matrix(
                dff_series,
                max_frames=config.max_frames,
            )
            event_series = qc.get_timeseries_from_proc(
                proc,
                [("event_timeseries",), ("events", "event_timeseries")],
            )
            events = None
            if event_series is not None:
                events, _ = qc.load_timeseries_matrix(event_series, max_frames=config.max_frames)
                if events.shape != dff.shape:
                    events = None

            roi_metadata = qc.get_roi_metadata_for_plane(
                nwb,
                plane_name,
                load_masks=False,
            ).reset_index(drop=True)
            metrics = calculate_roi_snr_metrics(
                dff,
                timestamps=timestamps,
                events=events,
                gaussian_sigma=config.gaussian_sigma,
                consecutive_samples=config.consecutive_samples,
                exceptional_robust_std=config.exceptional_robust_std,
                event_threshold=config.event_threshold,
                baseline_bins=config.baseline_bins,
                kernel_pre_s=config.kernel_pre_s,
                kernel_post_s=config.kernel_post_s,
                max_kernel_events=config.max_kernel_events,
                min_events_for_distribution_fit=config.min_events_for_distribution_fit,
            )
            plane_df = roi_metadata.merge(metrics, on="roi_index", how="right")
            plane_df.insert(0, "session_source", str(session_source))
            plane_df.insert(1, "session_id", sid)
            plane_df.insert(2, "session_date", date)
            if not plane_meta.empty:
                meta_cols = [c for c in ["plane", "structure", "depth_um", "location"] if c in plane_meta]
                plane_df = plane_df.merge(plane_meta[meta_cols], on="plane", how="left")
            rows.append(plane_df)

            duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else np.nan
            plane_inventory_rows.append(
                {
                    "session_source": str(session_source),
                    "session_id": sid,
                    "session_date": date,
                    "plane": plane_name,
                    "n_rois": int(dff.shape[1]),
                    "n_timepoints": int(dff.shape[0]),
                    "duration_s": duration_s,
                    "has_events": events is not None,
                }
            )
            timing_rows.append(
                {
                    "session_id": sid,
                    "plane": plane_name,
                    "status": "ok",
                    "elapsed_s": time.perf_counter() - plane_start,
                    "n_rois": int(dff.shape[1]),
                    "n_timepoints": int(dff.shape[0]),
                }
            )

        session_elapsed = time.perf_counter() - session_start
        timing_rows.append(
            {
                "session_id": sid,
                "plane": "__session_total__",
                "status": "ok",
                "elapsed_s": session_elapsed,
                "n_rois": int(sum(row.get("n_rois", 0) for row in plane_inventory_rows)),
            }
        )
    finally:
        qcr.close_session(session)

    roi_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    plane_inventory = pd.DataFrame(plane_inventory_rows)
    return roi_df, plane_inventory, timing_rows


def output_file_sizes(output_dir: str | Path) -> pd.DataFrame:
    """Return file sizes for saved summary artifacts."""
    rows = []
    for path in sorted(Path(output_dir).glob("*")):
        if path.is_file():
            rows.append({"path": str(path), "size_bytes": path.stat().st_size})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["size_mb"] = out["size_bytes"] / (1024**2)
    return out


def run_daily_summary(
    day: str,
    *,
    session_sources: Iterable[str | Path] | None = None,
    search_dirs: Iterable[str | Path] = (DEFAULT_DOWNLOAD_DIR,),
    output_root: str | Path = Path("outputs") / "snr_metrics_by_day",
    config: SnrBatchConfig | None = None,
) -> dict[str, Any]:
    """Compute and save compact ROI SNR statistics for a day's sessions."""
    config = config or SnrBatchConfig(day=day)
    sources = list(session_sources or [])
    if not sources:
        sources = discover_local_sessions_for_day(day, search_dirs=search_dirs)
    sources = [str(source) for source in sources]

    output_dir = Path(output_root) / safe_name(day)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    roi_tables = []
    plane_tables = []
    timing_rows: list[dict[str, Any]] = []
    for source in sources:
        source_start = time.perf_counter()
        try:
            roi_df, plane_df, timings = compute_session_metrics(source, config)
            if not roi_df.empty:
                roi_tables.append(roi_df)
            if not plane_df.empty:
                plane_tables.append(plane_df)
            timing_rows.extend(timings)
        except Exception as exc:
            timing_rows.append(
                {
                    "session_source": str(source),
                    "session_id": Path(str(source)).stem,
                    "plane": "__session_total__",
                    "status": "error",
                    "error": repr(exc),
                    "elapsed_s": time.perf_counter() - source_start,
                }
            )

    roi_metrics = pd.concat(roi_tables, ignore_index=True) if roi_tables else pd.DataFrame()
    plane_inventory = pd.concat(plane_tables, ignore_index=True) if plane_tables else pd.DataFrame()
    timing = pd.DataFrame(timing_rows)
    session_inventory = (
        plane_inventory.groupby(["session_source", "session_id", "session_date"], dropna=False)
        .agg(
            n_planes=("plane", "nunique"),
            n_rois=("n_rois", "sum"),
            n_timepoints_min=("n_timepoints", "min"),
            n_timepoints_max=("n_timepoints", "max"),
            duration_s_max=("duration_s", "max"),
            has_events_all=("has_events", "all"),
        )
        .reset_index()
        if not plane_inventory.empty
        else pd.DataFrame()
    )

    roi_path = output_dir / "roi_metrics.csv"
    plane_path = output_dir / "plane_inventory.csv"
    session_path = output_dir / "session_inventory.csv"
    timing_path = output_dir / "timing.csv"
    roi_metrics.to_csv(roi_path, index=False)
    plane_inventory.to_csv(plane_path, index=False)
    session_inventory.to_csv(session_path, index=False)
    timing.to_csv(timing_path, index=False)

    elapsed_s = time.perf_counter() - start
    file_sizes = output_file_sizes(output_dir)
    size_path = output_dir / "storage_usage.csv"
    file_sizes.to_csv(size_path, index=False)
    total_size_bytes = int(file_sizes["size_bytes"].sum()) if not file_sizes.empty else 0
    manifest = {
        "day": day,
        "session_sources": sources,
        "config": asdict(config),
        "elapsed_s": elapsed_s,
        "n_sessions": int(session_inventory["session_id"].nunique()) if not session_inventory.empty else 0,
        "n_planes": int(len(plane_inventory)),
        "n_roi_rows": int(len(roi_metrics)),
        "output_dir": str(output_dir),
        "total_output_size_bytes": total_size_bytes,
        "total_output_size_mb": total_size_bytes / (1024**2),
        "artifacts": {
            "roi_metrics": str(roi_path),
            "plane_inventory": str(plane_path),
            "session_inventory": str(session_path),
            "timing": str(timing_path),
            "storage_usage": str(size_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "roi_metrics": roi_metrics,
        "plane_inventory": plane_inventory,
        "session_inventory": session_inventory,
        "timing": timing,
        "storage_usage": output_file_sizes(output_dir),
    }
