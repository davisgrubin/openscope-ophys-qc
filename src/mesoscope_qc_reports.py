"""Reusable QC report plots and PDF writers for OpenScope mesoscope sessions."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pynwb import NWBHDF5IO

import mesoscope_qc_pipeline as qc
import session_loader as sl


DEFAULT_SESSION_SOURCE = Path(
    os.environ.get(
        "OPENSCOPE_QC_DEFAULT_SESSION_SOURCE",
        "/storage/scratch1/3/grubin6/openscope_ophys_qc_dandi_downloads/"
        "sub-839909_ses-multiplane-ophys-839909-2026-02-20-12-53-27_ophys.nwb",
    )
)
DEFAULT_MAX_FRAMES = int(os.environ.get("OPENSCOPE_QC_MAX_FRAMES", "10000"))
DEFAULT_NEURONS_PER_PLANE = int(os.environ.get("OPENSCOPE_QC_NEURONS_PER_PLANE", "1"))
DEFAULT_QC_STIM_TABLE = os.environ.get("OPENSCOPE_QC_STIM_TABLE", "trial_intervals")
DEFAULT_SPIKE_WINDOW_S = (-1.0, 2.0)
DEFAULT_STIM_WINDOW_S = (-1.0, 2.0)
DEFAULT_STIM_AUC_WINDOW_S = (0.0, 0.5)
DEFAULT_MASK_LIMIT = None


@dataclass
class PlaneReportData:
    plane_name: str
    plane_df: pd.DataFrame
    mean_projection: np.ndarray | None
    mean_projection_label: str | None
    max_projection: np.ndarray | None
    max_projection_label: str | None
    roi_indices: list[int]
    pixel_masks: list[np.ndarray] | None
    mask_shape: tuple[int, int] | None
    dff: np.ndarray
    timestamps: np.ndarray
    events: np.ndarray | None


@dataclass
class SessionReportContext:
    session_source: str
    nwb: Any
    plane_names: list[str]
    plane_meta: pd.DataFrame
    roi_qc: pd.DataFrame
    planes: dict[str, PlaneReportData]
    stimulus_times: np.ndarray
    stimulus_table: str | None


def resolve_session_source(session_source: str | Path | None = None) -> Path | str:
    source = DEFAULT_SESSION_SOURCE if session_source is None else session_source
    if isinstance(source, Path):
        return source.expanduser().resolve()
    source_path = Path(str(source)).expanduser()
    return source_path.resolve() if source_path.exists() else str(source)


def open_session(session_source: str | Path | None = None) -> dict[str, Any]:
    source = resolve_session_source(session_source)
    if isinstance(source, Path) and source.exists():
        io = NWBHDF5IO(path=str(source), mode="r", load_namespaces=True)
        nwb = io.read()
        meta = {
            "session_id": getattr(nwb, "session_id", None) or source.stem,
            "session_description": getattr(nwb, "session_description", None),
            "session_start_time": str(getattr(nwb, "session_start_time", "")),
            "source_path": str(source),
        }
        return {
            "path": str(source),
            "nwb": nwb,
            "meta": meta,
            "plane_meta": qc.get_plane_metadata(nwb),
            "planes": qc.get_plane_names(nwb),
            "io": io,
        }
    return sl.load_session(str(source))


def close_session(session: dict[str, Any]) -> None:
    try:
        sl.close_session(session)
    except Exception:
        try:
            session["io"].close()
        except Exception:
            pass


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return cleaned or "unknown"


def _roi_row(plane_df: pd.DataFrame, roi_index: int) -> pd.Series:
    matches = plane_df.loc[plane_df["roi_index"] == roi_index]
    if len(matches):
        return matches.iloc[0]
    return pd.Series(dtype=float)


def _image_mask_to_pixel_mask(image_mask: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    mask = np.asarray(image_mask, dtype=np.float32)
    y, x = np.nonzero(mask > threshold)
    if y.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.column_stack([x, y, mask[y, x]]).astype(np.float32)


def _plane_proc(nwb: Any, plane_name: str) -> Any:
    return nwb.processing[plane_name]


def get_plane_projections(nwb: Any, plane_name: str) -> tuple[np.ndarray | None, str | None, np.ndarray | None, str | None]:
    proc = _plane_proc(nwb, plane_name)
    try:
        image_mod = proc["images"]
    except Exception:
        return None, None, None, None
    mean_projection = mean_label = max_projection = max_label = None
    for image_name, label, slot in [
        ("average_projection", "Average projection", "mean"),
        ("max_projection", "Max projection", "max"),
    ]:
        try:
            if image_name in image_mod.images:
                arr = np.asarray(image_mod[image_name].data, dtype=np.float32)
                if slot == "mean":
                    mean_projection, mean_label = arr, label
                else:
                    max_projection, max_label = arr, label
                continue
        except Exception:
            pass
        try:
            arr = np.asarray(image_mod[image_name].data, dtype=np.float32)
            if slot == "mean":
                mean_projection, mean_label = arr, label
            else:
                max_projection, max_label = arr, label
        except Exception:
            pass
    if mean_projection is None and max_projection is not None:
        mean_projection, mean_label = max_projection, "Max projection"
    if max_projection is None and mean_projection is not None:
        max_projection, max_label = mean_projection, "Average projection"
    return mean_projection, mean_label, max_projection, max_label


def _load_plane_masks(nwb: Any, plane_name: str, max_rois: int | None = None) -> tuple[list[np.ndarray], list[int], tuple[int, int] | None]:
    roi_table = _plane_proc(nwb, plane_name)["image_segmentation"]["roi_table"]
    n_rois = len(roi_table)
    limit = n_rois if max_rois is None else min(n_rois, int(max_rois))
    colnames = list(getattr(roi_table, "colnames", []))
    roi_indices = list(range(limit))

    pixel_masks: list[np.ndarray] = []
    shape: tuple[int, int] | None = None

    if "pixel_mask" in colnames:
        for idx in roi_indices:
            try:
                pix = np.asarray(roi_table["pixel_mask"][idx], dtype=np.float32).reshape(-1, 3)
            except Exception:
                pix = np.empty((0, 3), dtype=np.float32)
            pixel_masks.append(pix)
            if shape is None and pix.size:
                shape = (int(np.nanmax(pix[:, 1])) + 1, int(np.nanmax(pix[:, 0])) + 1)
    elif "image_mask" in colnames:
        masks_data = roi_table["image_mask"].data
        if limit and shape is None:
            try:
                shape = np.asarray(masks_data[0], dtype=np.float32).shape
            except Exception:
                shape = None
        for idx in roi_indices:
            try:
                pixel_masks.append(_image_mask_to_pixel_mask(masks_data[idx]))
            except Exception:
                pixel_masks.append(np.empty((0, 3), dtype=np.float32))
    else:
        return [], roi_indices, shape

    return pixel_masks, roi_indices, shape


def _shape_metrics(pixel_masks: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for roi_index, pix in enumerate(pixel_masks):
        x = pix[:, 0] if pix.size else np.empty(0)
        y = pix[:, 1] if pix.size else np.empty(0)
        area = float(x.size)
        if area > 0:
            coords = np.column_stack([x, y]).astype(float)
            cov = np.cov(coords.T) if len(coords) > 1 else np.eye(2)
            eigvals = np.sort(np.linalg.eigvalsh(cov))
            elongation = float(np.sqrt((eigvals[-1] + 1e-9) / (eigvals[0] + 1e-9)))
            perimeter = float(max(1.0, np.sqrt(area) * 4.0))
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        else:
            elongation = np.nan
            circularity = np.nan
        rows.append(
            {
                "roi_index": roi_index,
                "roi_area_pix": area,
                "roi_elongation": elongation,
                "roi_circularity": circularity,
            }
        )
    return pd.DataFrame(rows)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask)
    positive = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary & positive


def _combined_boundary(pixel_masks: list[np.ndarray], shape: tuple[int, int] | None, selected_roi: int | None = None) -> np.ndarray | None:
    if not pixel_masks or shape is None:
        return None
    overlay = np.zeros(shape, dtype=float)
    for i, pix in enumerate(pixel_masks):
        if selected_roi is not None and i == selected_roi:
            continue
        if not len(pix):
            continue
        dense = np.zeros(shape, dtype=np.float32)
        x = np.clip(pix[:, 0].astype(int), 0, shape[1] - 1)
        y = np.clip(pix[:, 1].astype(int), 0, shape[0] - 1)
        dense[y, x] = 1.0
        overlay[_mask_boundary(dense)] = 1.0
    return np.ma.masked_where(overlay == 0, overlay)


def _selected_boundary(pixel_masks: list[np.ndarray], shape: tuple[int, int] | None, selected_roi: int) -> np.ndarray | None:
    if not pixel_masks or shape is None or selected_roi >= len(pixel_masks):
        return None
    pix = pixel_masks[selected_roi]
    if not len(pix):
        return None
    dense = np.zeros(shape, dtype=np.float32)
    x = np.clip(pix[:, 0].astype(int), 0, shape[1] - 1)
    y = np.clip(pix[:, 1].astype(int), 0, shape[0] - 1)
    dense[y, x] = 1.0
    boundary = _mask_boundary(dense)
    return np.ma.masked_where(~boundary, boundary)


def _roi_binary_mask(pixel_masks: list[np.ndarray], shape: tuple[int, int] | None, roi_index: int) -> np.ndarray | None:
    if not pixel_masks or shape is None or roi_index >= len(pixel_masks):
        return None
    pix = pixel_masks[roi_index]
    if not len(pix):
        return None
    dense = np.zeros(shape, dtype=np.uint8)
    x = np.clip(pix[:, 0].astype(int), 0, shape[1] - 1)
    y = np.clip(pix[:, 1].astype(int), 0, shape[0] - 1)
    dense[y, x] = 1
    return dense


def _mask_label_image(
    pixel_masks: list[np.ndarray],
    shape: tuple[int, int] | None,
    *,
    selected_roi: int | None = None,
) -> np.ndarray | None:
    if not pixel_masks or shape is None:
        return None
    labels = np.zeros(shape, dtype=np.int32)
    selected_label = len(pixel_masks) + 1
    for idx, pix in enumerate(pixel_masks):
        if selected_roi is not None and idx == selected_roi:
            continue
        if not len(pix):
            continue
        x = np.clip(pix[:, 0].astype(int), 0, shape[1] - 1)
        y = np.clip(pix[:, 1].astype(int), 0, shape[0] - 1)
        labels[y, x] = np.maximum(labels[y, x], idx + 1)
    if selected_roi is not None and selected_roi < len(pixel_masks):
        pix = pixel_masks[selected_roi]
        if len(pix):
            x = np.clip(pix[:, 0].astype(int), 0, shape[1] - 1)
            y = np.clip(pix[:, 1].astype(int), 0, shape[0] - 1)
            labels[y, x] = selected_label
    return labels


def _plane_background(projection: np.ndarray | None) -> np.ndarray | None:
    if projection is None:
        return None
    arr = np.asarray(projection, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99.8])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _plot_projection_panel(
    ax: plt.Axes,
    projection: np.ndarray | None,
    *,
    title: str,
    plane: PlaneReportData,
    selected_roi: int | None = None,
) -> None:
    bg = _plane_background(projection)
    if bg is not None:
        ax.imshow(bg, cmap="gray", interpolation="nearest")
    else:
        ax.text(0.5, 0.5, "projection not available", ha="center", va="center", transform=ax.transAxes)
    if plane.pixel_masks and plane.mask_shape is not None:
        if selected_roi is None:
            labels = _mask_label_image(plane.pixel_masks, plane.mask_shape)
            if labels is not None:
                cmap = plt.get_cmap("turbo").copy()
                cmap.set_bad((0, 0, 0, 0))
                masked = np.ma.masked_where(labels <= 0, labels)
                ax.imshow(masked, cmap=cmap, interpolation="nearest", alpha=0.70, vmin=0, vmax=max(1, int(labels.max())))
                overlay = _combined_boundary(plane.pixel_masks, plane.mask_shape)
                if overlay is not None:
                    ax.imshow(overlay, cmap="gray", alpha=0.45)
        else:
            selected_mask = _roi_binary_mask(plane.pixel_masks, plane.mask_shape, selected_roi)
            if selected_mask is not None:
                rgba = np.zeros((*selected_mask.shape, 4), dtype=np.float32)
                rgba[selected_mask > 0] = np.array([1.0, 0.0, 0.0, 0.98], dtype=np.float32)
                ax.imshow(rgba, interpolation="nearest")
                selected_boundary = _selected_boundary(plane.pixel_masks, plane.mask_shape, selected_roi)
                if selected_boundary is not None:
                    ax.imshow(selected_boundary, cmap="gray", alpha=1.0)
            other = _combined_boundary(plane.pixel_masks, plane.mask_shape, selected_roi=selected_roi)
            if other is not None:
                ax.imshow(other, cmap="gray", alpha=0.35)
    ax.set_title(title)
    ax.set_axis_off()


def _load_timeseries(nwb: Any, plane_name: str, max_frames: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    proc = _plane_proc(nwb, plane_name)
    dff_series = qc.get_timeseries_from_proc(proc, [("dff_timeseries", "dff_timeseries"), ("dff_timeseries",)])
    if dff_series is None:
        raise KeyError(f"No dF/F timeseries found for plane {plane_name}")
    dff, ts = qc.load_timeseries_matrix(dff_series, max_frames=max_frames)
    event_series = qc.get_timeseries_from_proc(proc, [("event_timeseries",), ("events", "event_timeseries")])
    events = None
    if event_series is not None:
        events, _ = qc.load_timeseries_matrix(event_series, max_frames=max_frames)
    return dff, ts, events


def load_plane_report_data(
    nwb: Any,
    plane_name: str,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    max_rois: int | None = DEFAULT_MASK_LIMIT,
) -> PlaneReportData:
    plane_df = qc.analyze_plane(nwb, plane_name, load_masks=False, max_frames=max_frames)
    mean_projection, mean_projection_label, max_projection, max_projection_label = get_plane_projections(nwb, plane_name)
    pixel_masks, roi_indices, shape = _load_plane_masks(nwb, plane_name, max_rois=max_rois)
    if pixel_masks:
        shape_df = _shape_metrics(pixel_masks)
        plane_df = plane_df.drop(columns=[c for c in shape_df.columns if c in plane_df.columns and c != "roi_index"], errors="ignore")
        plane_df = plane_df.merge(shape_df, on="roi_index", how="left")
    dff, timestamps, events = _load_timeseries(nwb, plane_name, max_frames=max_frames)
    return PlaneReportData(
        plane_name=plane_name,
        plane_df=plane_df,
        mean_projection=mean_projection,
        mean_projection_label=mean_projection_label,
        max_projection=max_projection,
        max_projection_label=max_projection_label,
        roi_indices=roi_indices,
        pixel_masks=pixel_masks if pixel_masks else None,
        mask_shape=shape,
        dff=dff,
        timestamps=timestamps,
        events=events,
    )


def build_session_report_context(
    session_source: str | Path | None = None,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    max_rois: int | None = DEFAULT_MASK_LIMIT,
    plane_names: list[str] | None = None,
) -> SessionReportContext:
    session = open_session(session_source)
    try:
        nwb = session["nwb"]
        planes = plane_names or session["planes"]
        stimulus_times, stimulus_table = first_stimulus_times(nwb, table_name=None, max_trials=None)
        plane_data = {}
        roi_frames = []
        for plane_name in planes:
            plane_data[plane_name] = load_plane_report_data(nwb, plane_name, max_frames=max_frames, max_rois=max_rois)
            roi_frames.append(plane_data[plane_name].plane_df)
        roi_qc = pd.concat(roi_frames, ignore_index=True) if roi_frames else pd.DataFrame()
        if "plane_meta" in session and session["plane_meta"] is not None:
            plane_meta = session["plane_meta"]
            if not isinstance(plane_meta, pd.DataFrame):
                plane_meta = pd.DataFrame(plane_meta)
        else:
            plane_meta = qc.get_plane_metadata(nwb)
        return SessionReportContext(
            session_source=str(session.get("path", session_source or DEFAULT_SESSION_SOURCE)),
            nwb=nwb,
            plane_names=list(planes),
            plane_meta=plane_meta,
            roi_qc=roi_qc,
            planes=plane_data,
            stimulus_times=stimulus_times,
            stimulus_table=stimulus_table,
        )
    finally:
        close_session(session)


def plot_plane_fov_overlay(ctx: SessionReportContext, plane_name: str) -> plt.Figure:
    plane = ctx.planes[plane_name]
    plane_df = plane.plane_df
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f"{plane_name} FOV summary ({len(plane_df)} ROIs)", fontsize=14, fontweight="bold")

    _plot_projection_panel(fig.add_subplot(gs[0, 0]), plane.max_projection, title="Functional max projection", plane=plane)
    _plot_projection_panel(fig.add_subplot(gs[0, 1]), plane.mean_projection, title="Functional mean projection", plane=plane)
    _plot_projection_panel(fig.add_subplot(gs[1, 0]), plane.max_projection, title="Functional max + ROI outlines", plane=plane)
    _plot_projection_panel(fig.add_subplot(gs[1, 1]), plane.mean_projection, title="Functional mean + ROI outlines", plane=plane)
    return fig


def plot_plane_roi_size_distribution(ctx: SessionReportContext, plane_name: str) -> plt.Figure:
    plane_df = ctx.planes[plane_name].plane_df
    fig, ax = plt.subplots(figsize=(6, 4))
    area = pd.to_numeric(plane_df.get("roi_area_pix"), errors="coerce").dropna()
    ax.hist(area, bins=40, color="steelblue")
    ax.set_title("Distribution of ROI sizes")
    ax.set_xlabel("area (pixels)")
    ax.set_ylabel("ROI count")
    plt.tight_layout()
    return fig


def plot_plane_roi_probability_distribution(ctx: SessionReportContext, plane_name: str) -> plt.Figure:
    plane_df = ctx.planes[plane_name].plane_df
    fig, ax = plt.subplots(figsize=(6, 4))
    prob_cols = [c for c in ["soma_probability", "dendrite_probability"] if c in plane_df]
    for col in prob_cols:
        vals = pd.to_numeric(plane_df[col], errors="coerce").dropna()
        ax.hist(vals, bins=np.linspace(0, 1, 31), alpha=0.65, label=col.replace("_", " "))
    ax.set_title("Distribution of soma/dendrite probability")
    ax.set_xlabel("probability")
    ax.set_ylabel("ROI count")
    if prob_cols:
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_plane_roi_shape_distribution(ctx: SessionReportContext, plane_name: str) -> plt.Figure:
    plane_df = ctx.planes[plane_name].plane_df
    fig, ax = plt.subplots(figsize=(6, 4))
    elong = pd.to_numeric(plane_df.get("roi_elongation"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    circ = pd.to_numeric(plane_df.get("roi_circularity"), errors="coerce").dropna()
    if len(elong):
        ax.hist(elong, bins=40, alpha=0.65, label="elongation")
    ax2 = ax.twiny()
    if len(circ):
        ax2.hist(circ, bins=np.linspace(0, 1, 31), alpha=0.35, color="darkorange", label="circularity")
        ax2.set_xlabel("circularity")
    ax.set_title("ROI shape: roundness / elongation")
    ax.set_xlabel("elongation ratio")
    ax.set_ylabel("ROI count")
    plt.tight_layout()
    return fig


def plot_plane_qc_overview(ctx: SessionReportContext, plane_name: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plane_df = ctx.planes[plane_name].plane_df
    plane = ctx.planes[plane_name]
    fig.suptitle(f"{plane_name}: {len(plane_df)} ROIs", fontsize=14, fontweight="bold")

    _plot_projection_panel(
        axes[0, 0],
        plane.max_projection,
        title=f"Functional max + white outlines\nNumber of ROIs: {len(plane_df)}",
        plane=plane,
    )
    _plot_projection_panel(
        axes[0, 1],
        plane.mean_projection,
        title="Functional mean + white outlines",
        plane=plane,
    )

    area = pd.to_numeric(plane_df.get("roi_area_pix"), errors="coerce").dropna()
    axes[1, 0].hist(area, bins=40, color="steelblue")
    axes[1, 0].set_title("Distribution of ROI sizes")
    axes[1, 0].set_xlabel("area (pixels)")
    axes[1, 0].set_ylabel("ROI count")

    prob_cols = [c for c in ["soma_probability", "dendrite_probability"] if c in plane_df]
    for col in prob_cols:
        vals = pd.to_numeric(plane_df[col], errors="coerce").dropna()
        axes[1, 1].hist(vals, bins=np.linspace(0, 1, 31), alpha=0.65, label=col.replace("_", " "))
    axes[1, 1].set_title("Distribution of soma/dendrite probability")
    axes[1, 1].set_xlabel("probability")
    axes[1, 1].set_ylabel("ROI count")
    if prob_cols:
        axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def gaussian_fit_r2(values: np.ndarray, bins: int = 50) -> tuple[float | np.nan, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10 or np.nanstd(vals) == 0:
        return np.nan, None, None, None
    hist, edges = np.histogram(vals, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.nanmean(vals)
    sigma = np.nanstd(vals)
    pred = np.exp(-0.5 * ((centers - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ss_res = np.sum((hist - pred) ** 2)
    ss_tot = np.sum((hist - np.mean(hist)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), centers, hist, pred


def rolling_baseline(trace: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(np.asarray(trace, dtype=float))
    return s.rolling(window=window, center=True, min_periods=max(3, window // 5)).median().bfill().ffill().to_numpy()


def event_indices_from_oasis(events: np.ndarray | None, trace: np.ndarray) -> tuple[np.ndarray, str]:
    if events is not None:
        vals = np.asarray(events, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            threshold = max(0.0, np.nanpercentile(finite, 95))
            idx = np.flatnonzero(vals > threshold)
            if idx.size:
                return idx, "event_timeseries/OASIS-like events"
    tr = np.asarray(trace, dtype=float)
    robust_sigma = 1.4826 * np.nanmedian(np.abs(tr - np.nanmedian(tr)))
    threshold = np.nanmedian(tr) + 3 * robust_sigma
    candidates = np.flatnonzero(tr > threshold)
    if candidates.size == 0:
        return candidates, "thresholded dF/F fallback"
    keep = np.r_[True, np.diff(candidates) > 1]
    return candidates[keep], "thresholded dF/F fallback"


def windowed_average(trace: np.ndarray, timestamps: np.ndarray, centers: np.ndarray, window_s: tuple[float, float]) -> tuple[np.ndarray | None, np.ndarray | None]:
    trace = np.asarray(trace, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if len(timestamps) < 2 or len(centers) == 0:
        return None, None
    dt = float(np.nanmedian(np.diff(timestamps)))
    offsets = np.arange(int(np.floor(window_s[0] / dt)), int(np.ceil(window_s[1] / dt)) + 1)
    rel_t = offsets * dt
    snippets = []
    for center in centers:
        base = int(np.searchsorted(timestamps, center)) if np.issubdtype(np.asarray(centers).dtype, np.floating) else int(center)
        idx = base + offsets
        valid = (idx >= 0) & (idx < trace.size)
        if valid.mean() < 0.8:
            continue
        snip = np.full(offsets.size, np.nan)
        snip[valid] = trace[idx[valid]]
        snippets.append(snip)
    if not snippets:
        return rel_t, None
    return rel_t, np.nanmean(np.vstack(snippets), axis=0)


def first_stimulus_times(nwb: Any, table_name: str | None = None, max_trials: int | None = None) -> tuple[np.ndarray, str | None]:
    if not hasattr(nwb, "intervals") or nwb.intervals is None:
        return np.array([]), None
    if table_name:
        names = [table_name]
    else:
        names = [name for name in nwb.intervals.keys() if "presentation" in name.lower()]
        names += [name for name in nwb.intervals.keys() if name not in names]
    for name in names:
        if name not in nwb.intervals:
            continue
        try:
            df = nwb.intervals[name].to_dataframe()
        except Exception:
            continue
        if "start_time" in df.columns:
            vals = pd.to_numeric(df["start_time"], errors="coerce").dropna().to_numpy()
        elif df.index.name == "start_time" or np.issubdtype(df.index.dtype, np.number):
            vals = pd.to_numeric(pd.Series(df.index), errors="coerce").dropna().to_numpy()
        else:
            continue
        return vals[:max_trials] if max_trials is not None else vals, name
    return np.array([]), None


def plot_neuron_fov_marked(ctx: SessionReportContext, plane_name: str, roi_index: int) -> plt.Figure:
    plane = ctx.planes[plane_name]
    row = _roi_row(plane.plane_df, roi_index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    _plot_projection_panel(axes[0], plane.max_projection, title="Functional max + selected ROI", plane=plane, selected_roi=roi_index)
    _plot_projection_panel(axes[1], plane.mean_projection, title="Functional mean + selected ROI", plane=plane, selected_roi=roi_index)
    roi_size = row.get("roi_area_pix", np.nan)
    soma_prob = row.get("soma_probability", np.nan)
    dend_prob = row.get("dendrite_probability", np.nan)
    fig.suptitle(
        f"ROI {roi_index} | size={roi_size:.0f} pix | soma p={soma_prob:.3f} | dendrite p={dend_prob:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    return fig


def plot_neuron_dff_trace(ctx: SessionReportContext, plane_name: str, roi_index: int) -> tuple[plt.Figure, dict[str, float]]:
    plane = ctx.planes[plane_name]
    trace = np.asarray(plane.dff[:, roi_index], dtype=float)
    timestamps = plane.timestamps
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestamps, trace, lw=0.8, color="black")
    n = trace.size
    k = max(1, int(0.1 * n))
    early = float(np.nanmedian(trace[:k]))
    late = float(np.nanmedian(trace[-k:]))
    drift = late - early
    ax.axhline(early, color="steelblue", ls="--", lw=1, label="early median")
    ax.axhline(late, color="darkorange", ls="--", lw=1, label="late median")
    ax.set_title(f"DFF drift, early vs late median DFF: {drift:.4g}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, {"early_median": early, "late_median": late, "drift": drift}


def plot_neuron_noise_distribution(ctx: SessionReportContext, plane_name: str, roi_index: int) -> tuple[plt.Figure, dict[str, float]]:
    plane = ctx.planes[plane_name]
    trace = np.asarray(plane.dff[:, roi_index], dtype=float)
    dt = np.nanmedian(np.diff(plane.timestamps)) if len(plane.timestamps) > 2 else 1.0
    baseline = rolling_baseline(trace, max(15, int(round(30 / max(dt, 1e-6)))))
    noise = trace - baseline
    r2, centers, hist, pred = gaussian_fit_r2(noise)
    fig, ax = plt.subplots(figsize=(8, 4))
    if centers is not None:
        ax.bar(centers, hist, width=np.nanmedian(np.diff(centers)), alpha=0.6, color="slategray", label="residual")
        ax.plot(centers, pred, color="crimson", lw=2, label="Gaussian fit")
    ax.set_title(f"OASIS noise distribution: Gaussian R^2={r2:.3f}")
    ax.set_xlabel("dF/F residual")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, {"gaussian_r2": float(r2) if np.isfinite(r2) else np.nan}


def plot_neuron_average_transient(ctx: SessionReportContext, plane_name: str, roi_index: int) -> tuple[plt.Figure, dict[str, Any]]:
    plane = ctx.planes[plane_name]
    trace = np.asarray(plane.dff[:, roi_index], dtype=float)
    event_trace = None if plane.events is None else np.asarray(plane.events[:, roi_index], dtype=float)
    spike_idx, spike_source = event_indices_from_oasis(event_trace, trace)
    rel_t, avg = windowed_average(trace, plane.timestamps, spike_idx, DEFAULT_SPIKE_WINDOW_S)
    fig, ax = plt.subplots(figsize=(8, 4))
    if avg is not None:
        ax.plot(rel_t, avg, color="darkgreen", lw=2)
        ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_title(f"Average calcium transient across spikes\n{len(spike_idx)} events from {spike_source}")
    ax.set_xlabel("time from spike/event (s)")
    ax.set_ylabel("mean dF/F")
    plt.tight_layout()
    return fig, {"spike_indices": spike_idx, "spike_source": spike_source, "avg_time": rel_t, "avg_trace": avg}


def plot_neuron_stimulus_aligned_dff(ctx: SessionReportContext, plane_name: str, roi_index: int, table_name: str | None = None) -> tuple[plt.Figure, dict[str, Any]]:
    plane = ctx.planes[plane_name]
    trace = np.asarray(plane.dff[:, roi_index], dtype=float)
    if table_name is not None:
        stim_times, stim_name = first_stimulus_times(ctx.nwb, table_name=table_name, max_trials=None)
    else:
        stim_times = np.asarray(ctx.stimulus_times, dtype=float)
        stim_name = ctx.stimulus_table
    rel_t, stim_avg = windowed_average(trace, plane.timestamps, stim_times, DEFAULT_STIM_WINDOW_S)
    auc_500ms = np.nan
    fig, ax = plt.subplots(figsize=(8, 4))
    if stim_avg is not None:
        ax.plot(rel_t, stim_avg, color="purple", lw=2)
        ax.axvline(0, color="black", lw=1, ls="--")
        mask = (rel_t >= DEFAULT_STIM_AUC_WINDOW_S[0]) & (rel_t <= DEFAULT_STIM_AUC_WINDOW_S[1])
        if np.any(mask):
            auc_500ms = float(np.trapz(stim_avg[mask], rel_t[mask]))
    else:
        ax.text(0.5, 0.5, "No stimulus interval table found", ha="center", va="center", transform=ax.transAxes)
    title = "Stimulus-aligned DFF average across trials"
    if np.isfinite(auc_500ms):
        title = f"{title}\nAUC in 500ms after stimulus: {auc_500ms:.4g}"
    elif stim_name is not None:
        title = f"{title}\n{len(stim_times)} trials from {stim_name}"
    ax.set_title(title)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("mean dF/F")
    plt.tight_layout()
    return fig, {"stimulus_times": stim_times, "stimulus_table": stim_name, "auc_500ms": auc_500ms, "avg_time": rel_t, "avg_trace": stim_avg}


def plot_neuron_qc_detail(ctx: SessionReportContext, plane_name: str, roi_index: int) -> plt.Figure:
    plane = ctx.planes[plane_name]
    row = _roi_row(plane.plane_df, roi_index)
    roi_size = row.get("roi_area_pix", np.nan)
    soma_prob = row.get("soma_probability", np.nan)
    dend_prob = row.get("dendrite_probability", np.nan)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        f"{plane_name} ROI {roi_index}: size={roi_size:.0f} pix, soma p={soma_prob:.3f}, dendrite p={dend_prob:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    _plot_projection_panel(axes[0, 0], plane.max_projection, title="Functional max + selected ROI", plane=plane, selected_roi=roi_index)
    _plot_projection_panel(axes[0, 1], plane.mean_projection, title="Functional mean + selected ROI", plane=plane, selected_roi=roi_index)

    trace = np.asarray(plane.dff[:, roi_index], dtype=float)
    timestamps = plane.timestamps
    event_trace = None if plane.events is None else np.asarray(plane.events[:, roi_index], dtype=float)

    axes[1, 0].plot(timestamps, trace, lw=0.8, color="black")
    n = trace.size
    k = max(1, int(0.1 * n))
    early = float(np.nanmedian(trace[:k]))
    late = float(np.nanmedian(trace[-k:]))
    drift = late - early
    axes[1, 0].axhline(early, color="steelblue", ls="--", lw=1, label="early median")
    axes[1, 0].axhline(late, color="darkorange", ls="--", lw=1, label="late median")
    axes[1, 0].set_title(f"DFF drift, early vs late median DFF: {drift:.4g}")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("dF/F")
    axes[1, 0].legend(fontsize=8)

    dt = np.nanmedian(np.diff(timestamps)) if len(timestamps) > 2 else 1.0
    baseline = rolling_baseline(trace, max(15, int(round(30 / max(dt, 1e-6)))))
    noise = trace - baseline
    r2, centers, hist, pred = gaussian_fit_r2(noise)
    if centers is not None:
        axes[1, 1].bar(centers, hist, width=np.nanmedian(np.diff(centers)), alpha=0.6, color="slategray", label="residual")
        axes[1, 1].plot(centers, pred, color="crimson", lw=2, label="Gaussian fit")
    axes[1, 1].set_title(f"OASIS noise distribution: Gaussian R^2={r2:.3f}")
    axes[1, 1].set_xlabel("dF/F residual")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend(fontsize=8)

    spike_idx, spike_source = event_indices_from_oasis(event_trace, trace)
    rel_t, avg = windowed_average(trace, timestamps, spike_idx, DEFAULT_SPIKE_WINDOW_S)
    if avg is not None:
        axes[2, 0].plot(rel_t, avg, color="darkgreen", lw=2)
        axes[2, 0].axvline(0, color="black", lw=1, ls="--")
    axes[2, 0].set_title(f"Average calcium transient across spikes\n{len(spike_idx)} events from {spike_source}")
    axes[2, 0].set_xlabel("time from spike/event (s)")
    axes[2, 0].set_ylabel("mean dF/F")

    stim_times = np.asarray(ctx.stimulus_times, dtype=float)
    stim_name = ctx.stimulus_table
    rel_t, stim_avg = windowed_average(trace, timestamps, stim_times, DEFAULT_STIM_WINDOW_S)
    auc_500ms = np.nan
    if stim_avg is not None:
        axes[2, 1].plot(rel_t, stim_avg, color="purple", lw=2)
        axes[2, 1].axvline(0, color="black", lw=1, ls="--")
        mask = (rel_t >= DEFAULT_STIM_AUC_WINDOW_S[0]) & (rel_t <= DEFAULT_STIM_AUC_WINDOW_S[1])
        if np.any(mask):
            auc_500ms = float(np.trapz(stim_avg[mask], rel_t[mask]))
        title = f"Stimulus-aligned DFF average across trials\nAUC in 500ms after stimulus: {auc_500ms:.4g}"
    else:
        axes[2, 1].text(0.5, 0.5, "No stimulus interval table found", ha="center", va="center", transform=axes[2, 1].transAxes)
        title = "Stimulus-aligned DFF average across trials"
    if stim_name is not None:
        title = f"{title}\n{len(stim_times)} trials from {stim_name}"
    axes[2, 1].set_title(title)
    axes[2, 1].set_xlabel("time from stimulus (s)")
    axes[2, 1].set_ylabel("mean dF/F")

    return fig


def _title_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.94)
    ax = fig.add_axes([0.10, 0.08, 0.80, 0.78])
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def _session_label(ctx: SessionReportContext) -> str:
    session_id = getattr(ctx.nwb, "session_id", None)
    return str(session_id or Path(ctx.session_source).stem)


def _source_label(ctx: SessionReportContext) -> str:
    source = str(ctx.session_source)
    if source.startswith("s3://"):
        return "AIND S3 session"
    return "local session file"


def _default_plot_names(kind: str) -> list[str]:
    if kind == "plane":
        return ["fov", "size", "probability", "shape"]
    if kind == "neuron":
        return ["fov", "dff", "noise", "spike", "stimulus"]
    return []


def render_plane_plots(ctx: SessionReportContext, plane_name: str, plots: list[str] | None = None) -> dict[str, plt.Figure]:
    wanted = plots or _default_plot_names("plane")
    out: dict[str, plt.Figure] = {}
    if "fov" in wanted:
        out["fov"] = plot_plane_qc_overview(ctx, plane_name)
    if "size" in wanted:
        out["size"] = plot_plane_roi_size_distribution(ctx, plane_name)
    if "probability" in wanted:
        out["probability"] = plot_plane_roi_probability_distribution(ctx, plane_name)
    if "shape" in wanted:
        out["shape"] = plot_plane_roi_shape_distribution(ctx, plane_name)
    return out


def render_neuron_plots(ctx: SessionReportContext, plane_name: str, roi_index: int, plots: list[str] | None = None) -> dict[str, plt.Figure]:
    wanted = plots or _default_plot_names("neuron")
    out: dict[str, plt.Figure] = {}
    if "fov" in wanted:
        out["fov"] = plot_neuron_fov_marked(ctx, plane_name, roi_index)
    if "dff" in wanted:
        out["dff"] = plot_neuron_dff_trace(ctx, plane_name, roi_index)[0]
    if "noise" in wanted:
        out["noise"] = plot_neuron_noise_distribution(ctx, plane_name, roi_index)[0]
    if "spike" in wanted:
        out["spike"] = plot_neuron_average_transient(ctx, plane_name, roi_index)[0]
    if "stimulus" in wanted:
        out["stimulus"] = plot_neuron_stimulus_aligned_dff(ctx, plane_name, roi_index)[0]
    return out


def write_session_qc_summary_pdf(
    session_source: str | Path | None = None,
    output_path: str | Path = "session_qc_summary.pdf",
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    max_rois: int | None = DEFAULT_MASK_LIMIT,
    plane_names: list[str] | None = None,
) -> Path:
    ctx = build_session_report_context(session_source=session_source, max_frames=max_frames, max_rois=max_rois, plane_names=plane_names)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _title_page(
            pdf,
            "Session QC Summary",
            [
                f"Session: {_session_label(ctx)}",
                f"Source: {_source_label(ctx)}",
                f"Planes: {', '.join(ctx.plane_names)}",
                f"Max frames loaded per plane: {max_frames}",
                f"Mask cap per plane: {max_rois}",
            ],
        )
        for plane_name in ctx.plane_names:
            fig = plot_plane_qc_overview(ctx, plane_name)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return output_path


def write_roi_qc_summary_pdf(
    session_source: str | Path | None = None,
    output_path: str | Path = "roi_qc_summary.pdf",
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    max_rois: int | None = DEFAULT_MASK_LIMIT,
    neurons_per_plane: int | None = DEFAULT_NEURONS_PER_PLANE,
    plane_names: list[str] | None = None,
) -> Path:
    ctx = build_session_report_context(session_source=session_source, max_frames=max_frames, max_rois=max_rois, plane_names=plane_names)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _title_page(
            pdf,
            "ROI QC Summary",
            [
                f"Session: {_session_label(ctx)}",
                f"Source: {_source_label(ctx)}",
                f"Planes: {', '.join(ctx.plane_names)}",
                f"Max frames loaded per plane: {max_frames}",
                f"Mask cap per plane: {max_rois}",
                f"Neurons per plane: {neurons_per_plane}",
            ],
        )
        for plane_name in ctx.plane_names:
            plane_df = ctx.planes[plane_name].plane_df
            roi_ids = plane_df["roi_index"].dropna().astype(int).tolist()
            if neurons_per_plane is not None:
                roi_ids = roi_ids[: int(neurons_per_plane)]
            for roi_index in roi_ids:
                fig = plot_neuron_qc_detail(ctx, plane_name, roi_index)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return output_path


def write_qc_summaries(
    session_source: str | Path | None = None,
    output_dir: str | Path = ".",
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    max_rois: int | None = DEFAULT_MASK_LIMIT,
    neurons_per_plane: int | None = DEFAULT_NEURONS_PER_PLANE,
    plane_names: list[str] | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_pdf = write_session_qc_summary_pdf(
        session_source=session_source,
        output_path=output_dir / "session_qc_summary.pdf",
        max_frames=max_frames,
        max_rois=max_rois,
        plane_names=plane_names,
    )
    roi_pdf = write_roi_qc_summary_pdf(
        session_source=session_source,
        output_path=output_dir / "roi_qc_summary.pdf",
        max_frames=max_frames,
        max_rois=max_rois,
        neurons_per_plane=neurons_per_plane,
        plane_names=plane_names,
    )
    return session_pdf, roi_pdf
