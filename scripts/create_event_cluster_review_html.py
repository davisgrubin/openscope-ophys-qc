#!/usr/bin/env python3
"""Create a standalone HTML viewer for event-cluster ROI QC."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_mesoscope_plane_html import (  # noqa: E402
    _build_roi_payload,
    _float32_b64,
    _green_png_data_uri,
    _load_plane_arrays,
)


PREFERRED_METRICS = [
    "background_event_median_amp_noise_units",
    "background_event_fraction_gt_4sd",
    "background_event_fraction_2_4sd",
    "background_event_fraction_lt_2sd",
    "background_event_count_ge_2sd",
    "background_event_rate_ge_2sd_hz",
    "trace_event_median_amp_noise_units",
    "trace_event_fraction_gt_4sd",
    "trace_event_fraction_2_4sd",
    "trace_event_fraction_lt_2sd",
    "trace_event_count_ge_2sd",
    "trace_event_rate_ge_2sd_hz",
    "event_exp_gauss_fit_score",
    "event_exponential_ks_stat",
    "event_model_residual_gaussian_ks_stat",
    "roi_area_pixels",
    "soma_probability",
    "dendrite_probability",
    "roi_classifier_confidence",
    "nonlong_event_fraction_gt_4sd",
    "nonlong_event_fraction_2_4sd",
    "nonlong_event_fraction_lt_2sd",
    "long_gt_3s_fraction_all_clusters",
    "max_event_cluster_span_s",
    "p95_event_cluster_span_s",
    "max_raw_onsets_in_cluster",
    "n_warning_long_event_clusters",
    "n_severe_long_event_clusters",
    "n_extreme_long_event_clusters",
    "n_lt_2sd_event_clusters",
    "n_2_4sd_event_clusters",
    "n_gt_4sd_event_clusters",
    "n_long_gt_3s_event_clusters",
]


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _metric_options(metrics: pd.DataFrame) -> list[str]:
    out = []
    for col in PREFERRED_METRICS:
        if col in metrics.columns and col not in out:
            vals = pd.to_numeric(metrics[col], errors="coerce")
            if vals.notna().sum() >= 3:
                out.append(col)
    return out


def _region_from_plane(plane: str) -> str:
    match = re.match(r"^(.+?)_\d+$", str(plane))
    return match.group(1) if match else str(plane)


def _load_plane_metadata(session_dir: Path) -> dict[str, dict]:
    meta_path = session_dir / "plane_metadata.csv"
    if not meta_path.exists():
        return {}
    meta = pd.read_csv(meta_path)
    out = {}
    for _, row in meta.iterrows():
        plane = str(row.get("plane", ""))
        if not plane:
            continue
        structure = row.get("structure", _region_from_plane(plane))
        depth_um = pd.to_numeric(row.get("depth_um", np.nan), errors="coerce")
        out[plane] = {
            "structure": None if pd.isna(structure) else str(structure),
            "depthUm": float(depth_um) if np.isfinite(depth_um) else None,
        }
    return out


def _plane_payload(
    session_dir: Path,
    plane: str,
    metrics: pd.DataFrame,
    plane_info: dict,
    max_frames: int | None,
) -> dict:
    fallback_reason = None
    try:
        arrays = _load_plane_arrays(session_dir, plane, max_frames=max_frames)
        projection = arrays["projection"]
        shape = tuple(arrays["shape"])
        rois, _ = _build_roi_payload(arrays["roi_indices"], arrays["pixel_masks"], shape)
        dff = np.asarray(arrays["dff"], dtype=np.float32)
        events = None if arrays["events"] is None else np.asarray(arrays["events"], dtype=np.float32)
        frame_rate = float(arrays["frame_rate"])
    except FileNotFoundError as exc:
        fallback_reason = str(exc)
        plane_dir = session_dir / plane
        dff_raw = np.load(plane_dir / "dff.npy", mmap_mode="r")
        events_path = plane_dir / "events.npy"
        events_raw = np.load(events_path, mmap_mode="r") if events_path.exists() else None
        if dff_raw.shape[0] >= dff_raw.shape[1]:
            dff_frames_rois = dff_raw
        else:
            dff_frames_rois = dff_raw.T
        n_frames_raw = dff_frames_rois.shape[0]
        n_frames = n_frames_raw if max_frames is None else min(int(max_frames), n_frames_raw)
        dff = np.asarray(dff_frames_rois[:n_frames].T, dtype=np.float32)
        events = None
        if events_raw is not None:
            event_frames_rois = events_raw if events_raw.shape[0] >= events_raw.shape[1] else events_raw.T
            events = np.asarray(event_frames_rois[:n_frames, : dff.shape[0]].T, dtype=np.float32)
        ts_path = plane_dir / "dff_timestamps.npy"
        frame_rate = 1.0
        if ts_path.exists():
            timestamps = np.load(ts_path, mmap_mode="r")[:n_frames]
            diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                frame_rate = float(1.0 / np.median(diffs))
        shape = (512, 512)
        projection = np.zeros(shape, dtype=np.float32)
        rois = []
    n_rois, n_frames = dff.shape

    plane_metrics = metrics.loc[metrics["plane"].astype(str) == str(plane)].copy()
    plane_metrics["roi_index"] = pd.to_numeric(
        plane_metrics["roi_index"],
        errors="coerce",
    ).astype("Int64")
    plane_metrics = plane_metrics.sort_values("roi_index")
    metric_rows = []
    for _, row in plane_metrics.iterrows():
        if pd.isna(row["roi_index"]):
            continue
        roi = int(row["roi_index"])
        if roi < 0 or roi >= n_rois:
            continue
        metric_rows.append({col: _json_safe(row[col]) for col in plane_metrics.columns})
    if not rois and {"roi_centroid_x", "roi_centroid_y"}.issubset(plane_metrics.columns):
        for _, row in plane_metrics.iterrows():
            if pd.isna(row["roi_index"]):
                continue
            x = pd.to_numeric(row.get("roi_centroid_x"), errors="coerce")
            y = pd.to_numeric(row.get("roi_centroid_y"), errors="coerce")
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            area = pd.to_numeric(row.get("roi_area_pixels"), errors="coerce")
            radius = float(np.sqrt(area / np.pi)) if np.isfinite(area) and area > 0 else 4.0
            radius = float(np.clip(radius, 2.0, 14.0))
            rois.append(
                {
                    "roi": int(row["roi_index"]),
                    "path": (
                        f"M{x - radius:.2f} {y:.2f}"
                        f"a{radius:.2f} {radius:.2f} 0 1 0 {2 * radius:.2f} 0"
                        f"a{radius:.2f} {radius:.2f} 0 1 0 {-2 * radius:.2f} 0"
                    ),
                    "cx": float(x),
                    "cy": float(y),
                }
            )

    return {
        "frameRate": frame_rate,
        "nRois": int(n_rois),
        "nFrames": int(n_frames),
        "imageWidth": int(shape[1]),
        "imageHeight": int(shape[0]),
        "structure": plane_info.get("structure") or _region_from_plane(plane),
        "depthUm": plane_info.get("depthUm"),
        "depthSource": "plane_metadata.csv depth_um" if plane_info.get("depthUm") is not None else "plane name order fallback",
        "projection": _green_png_data_uri(projection),
        "rois": rois[:n_rois],
        "metrics": metric_rows,
        "dff": _float32_b64(dff),
        "events": _float32_b64(events) if events is not None else None,
        "eventsAvailable": events is not None,
        "fallbackReason": fallback_reason,
    }


def _top_long_clusters(clusters_csv: Path | None, max_clusters: int) -> list[dict]:
    if clusters_csv is None or not clusters_csv.exists() or max_clusters <= 0:
        return []
    clusters = pd.read_csv(clusters_csv)
    required = {"plane", "roi_index", "midpoint_s", "cluster_span_s"}
    if not required.issubset(clusters.columns):
        return []
    clusters = clusters.sort_values("cluster_span_s", ascending=False).head(max_clusters)
    keep = [
        "plane",
        "roi_index",
        "first_onset_s",
        "last_onset_s",
        "midpoint_s",
        "cluster_span_s",
        "evaluation_start_s",
        "evaluation_end_s",
        "raw_onsets_in_cluster",
    ]
    keep = [col for col in keep if col in clusters.columns]
    return [
        {col: _json_safe(row[col]) for col in keep}
        for _, row in clusters[keep].iterrows()
    ]


def _cluster_payload(clusters_csv: Path | None, max_clusters: int) -> list[dict]:
    if clusters_csv is None or not clusters_csv.exists() or max_clusters <= 0:
        return []
    clusters = pd.read_csv(clusters_csv)
    required = {"plane", "roi_index", "midpoint_s", "evaluation_start_s", "evaluation_end_s"}
    if not required.issubset(clusters.columns):
        return []
    keep = [
        "plane",
        "roi_index",
        "first_onset_s",
        "last_onset_s",
        "midpoint_s",
        "cluster_span_s",
        "evaluation_start_s",
        "evaluation_end_s",
        "raw_onsets_in_cluster",
        "event_amplitude_noise_units",
        "event_type",
    ]
    keep = [col for col in keep if col in clusters.columns]
    sort_cols = [col for col in ["plane", "roi_index", "midpoint_s"] if col in clusters.columns]
    clusters = clusters.sort_values(sort_cols).head(max_clusters)
    return [
        {col: _json_safe(row[col]) for col in keep}
        for _, row in clusters[keep].iterrows()
    ]


def create_event_cluster_review_html(
    session_dir: Path,
    metrics_csv: Path,
    output_path: Path,
    *,
    clusters_csv: Path | None = None,
    planes: list[str] | None = None,
    max_frames: int | None = None,
    max_long_clusters: int = 500,
    max_clusters: int = 200000,
) -> Path:
    metrics = pd.read_csv(metrics_csv)
    if "plane" not in metrics or "roi_index" not in metrics:
        raise ValueError("metrics CSV must include plane and roi_index columns")
    plane_metadata = _load_plane_metadata(session_dir)
    plane_names = planes or sorted(metrics["plane"].dropna().astype(str).unique())
    plane_names = [plane for plane in plane_names if (session_dir / plane / "dff.npy").exists()]
    if not plane_names:
        raise ValueError(f"No planes with dff.npy found in {session_dir}")
    plane_names = sorted(
        plane_names,
        key=lambda p: (
            plane_metadata.get(p, {}).get("structure") or _region_from_plane(p),
            (
                plane_metadata.get(p, {}).get("depthUm")
                if plane_metadata.get(p, {}).get("depthUm") is not None
                else float("inf")
            ),
            p,
        ),
    )

    payload = {
        "session": session_dir.name,
        "metricOptions": _metric_options(metrics),
        "planes": plane_names,
        "regions": sorted(
            {
                (plane_metadata.get(plane, {}).get("structure") or _region_from_plane(plane))
                for plane in plane_names
            }
        ),
        "planeData": {},
        "longClusters": _top_long_clusters(clusters_csv, max_long_clusters),
        "eventClusters": _cluster_payload(clusters_csv, max_clusters),
        "settings": {
            "mergeWithinS": 0.5,
            "baselinePreS": 0.5,
            "peakPostS": 2.0,
            "warningSpanS": 3.0,
            "severeSpanS": 5.0,
            "extremeSpanS": 8.0,
        },
    }
    for plane in plane_names:
        print(f"[INFO] embedding {plane}", flush=True)
        payload["planeData"][plane] = _plane_payload(
            session_dir,
            plane,
            metrics,
            plane_metadata.get(plane, {}),
            max_frames=max_frames,
        )

    html = HTML_TEMPLATE.replace("__TITLE__", f"{session_dir.name} event-cluster review")
    html = html.replace(
        "__PLANE_OPTIONS__",
        "\n".join(f'<option value="{p}">{p}</option>' for p in plane_names),
    )
    html = html.replace(
        "__METRIC_OPTIONS__",
        "\n".join(f'<option value="{m}">{m}</option>' for m in payload["metricOptions"]),
    )
    html = html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f7f7f5; --panel:#fff; --ink:#1f2933; --muted:#667085; --line:#d0d5dd; --accent:#c2410c; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Arial, Helvetica, sans-serif; }
.page { width:min(1760px, calc(100vw - 28px)); margin:16px auto 28px; }
.header { display:flex; justify-content:space-between; gap:16px; align-items:end; margin-bottom:12px; }
h1 { margin:0; font-size:22px; letter-spacing:0; }
.meta { color:var(--muted); font-size:13px; text-align:right; line-height:1.35; }
.controls { display:grid; grid-template-columns:repeat(6, auto) 1fr; gap:9px; align-items:end; margin-bottom:10px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:10px; box-sizing:border-box; }
.panel-title { font-size:14px; font-weight:700; margin-bottom:8px; }
label { font-size:12px; color:#475467; display:grid; gap:3px; }
select, input, button { font:inherit; border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#fff; box-sizing:border-box; }
select { width:220px; } #planeSelect { width:120px; } input { width:90px; }
button { cursor:pointer; white-space:nowrap; }
.overview-grid { display:grid; grid-template-columns:minmax(320px, .8fr) minmax(520px, 1.2fr); gap:10px; align-items:start; }
.viewer-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; align-items:start; margin-top:10px; }
.image-wrap { position:relative; width:100%; aspect-ratio:1/1; background:#111; overflow:hidden; }
.image-wrap img, .image-wrap svg { position:absolute; inset:0; width:100%; height:100%; }
.image-wrap img { object-fit:contain; image-rendering:pixelated; }
.roi { fill:transparent; stroke:rgba(255,255,255,.86); stroke-width:.65; cursor:pointer; vector-effect:non-scaling-stroke; pointer-events:all; }
.roi:hover { fill:rgba(6,182,212,.22); stroke:#06b6d4; stroke-width:1.5; }
.roi.selected { fill:rgba(194,65,12,.24); stroke:var(--accent); stroke-width:2; }
canvas { width:100%; display:block; background:#fff; border:1px solid var(--line); box-sizing:border-box; }
#roiTraceCanvas { height:340px; cursor:grab; }
#compositionCanvas { height:416px; cursor:grab; }
.region-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:10px; }
.region-canvas { height:416px; cursor:grab; }
#histCanvas { height:210px; }
#fitCanvas { height:260px; }
.long-list { max-height:154px; overflow:auto; border:1px solid var(--line); margin-top:8px; }
.trace-grid { display:grid; grid-template-columns:1fr; gap:10px; margin-top:10px; }
.drawer-backdrop { position:fixed; inset:0; background:rgba(15,23,42,.28); opacity:0; pointer-events:none; transition:opacity .16s ease; z-index:20; }
.drawer { position:fixed; top:0; right:0; width:min(760px, calc(100vw - 30px)); height:100vh; background:#fff; border-left:1px solid var(--line); box-shadow:-8px 0 28px rgba(15,23,42,.18); transform:translateX(100%); transition:transform .18s ease; z-index:21; padding:12px; box-sizing:border-box; display:grid; grid-template-rows:auto 1fr; gap:8px; }
.drawer.open { transform:translateX(0); }
.drawer-backdrop.open { opacity:1; pointer-events:auto; }
.drawer-head { display:flex; justify-content:space-between; gap:10px; align-items:center; }
.drawer .table-wrap { max-height:none; height:100%; }
.table-wrap { max-height:416px; overflow:auto; border:1px solid var(--line); }
table { width:100%; border-collapse:collapse; font-size:12px; }
th, td { border-bottom:1px solid #e5e7eb; padding:5px 6px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
tr { cursor:pointer; }
tr:hover { background:#f8fafc; }
tr.selected { background:#fff7ed; }
.note { color:var(--muted); font-size:12px; margin-top:6px; }
.pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 7px; margin:0 4px 4px 0; font-size:12px; }
.long-list button { width:100%; border:0; border-bottom:1px solid #e5e7eb; border-radius:0; text-align:left; font-size:12px; }
@media (max-width:1180px) { .overview-grid, .viewer-grid, .controls { grid-template-columns:1fr; } .header { display:block; } .meta { text-align:left; } }
</style>
</head>
<body>
<div class="page">
  <div class="header"><h1>__TITLE__</h1><div class="meta" id="sessionMeta"></div></div>
  <div class="controls panel">
    <label>Plane<select id="planeSelect">__PLANE_OPTIONS__</select></label>
    <label>Sort metric<select id="metricSelect">__METRIC_OPTIONS__</select></label>
    <label>Direction<select id="sortDir"><option value="desc">descending</option><option value="asc">ascending</option></select></label>
    <label>Composition<select id="labelFilter"><option value="all">all</option></select></label>
    <label>Long flags<select id="longFilter"><option value="all">all</option><option value="warning">warning >=3s</option><option value="severe">severe >=5s</option><option value="extreme">extreme >=8s</option></select></label>
    <div><button id="resetView">Reset</button> <button id="openMetricsDrawer">ROI rows</button> <button id="openLongDrawer">Long events</button></div>
  </div>
  <div class="overview-grid">
    <div class="panel"><div class="panel-title">Functional projection</div><div class="image-wrap"><img id="projection"><svg id="overlay" preserveAspectRatio="xMidYMid meet"></svg></div><div class="note" id="roiReadout"></div></div>
    <div class="panel">
      <div class="panel-title">Metric distribution</div><canvas id="histCanvas"></canvas>
      <div class="panel-title" style="margin-top:10px;">Fit diagnostics for selected ROI and cluster</div><canvas id="fitCanvas"></canvas><div class="note" id="fitReadout"></div>
    </div>
  </div>
  <div class="viewer-grid">
    <div class="panel">
      <div class="panel-title">3D event-size composition</div><canvas id="compositionCanvas"></canvas><div class="note">Drag to rotate. Wheel to zoom. Click an ROI point to select it. Left/right arrows move through the current sort order.</div>
    </div>
    <div class="panel">
      <div class="panel-title">3D ROI locations by region</div><div class="region-grid" id="spatialRegionGrid"></div><div class="note">Each region is shown separately. Axes are image x, image y, and official depth in microns from plane metadata when available. Click a point to switch to that ROI and plane.</div>
    </div>
  </div>
  <div class="trace-grid">
    <div class="panel"><div class="panel-title">Selected ROI dF/F and detected events</div><canvas id="roiTraceCanvas"></canvas><div class="note">Full-width trace view. Wheel to zoom time. Drag to pan. Double-click to reset. Left/right arrows move through the current sort order.</div></div>
  </div>
</div>
<div class="drawer-backdrop" id="metricsBackdrop"></div>
<aside class="drawer" id="metricsDrawer">
  <div class="drawer-head"><div class="panel-title" style="margin:0;">ROI rows in current sort order</div><button id="closeMetricsDrawer">Close</button></div>
  <div class="table-wrap"><table id="roiTable"></table></div>
</aside>
<aside class="drawer" id="longDrawer">
  <div class="drawer-head"><div class="panel-title" style="margin:0;">Longest event clusters for selected plane</div><button id="closeLongDrawer">Close</button></div>
  <div class="long-list" id="longList"></div>
</aside>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const payload = JSON.parse(document.getElementById("payload").textContent);
const decoded = {};
let plane = payload.planes[0], data = null, selected = 0, metric = payload.metricOptions[0];
let viewStart = 0, viewEnd = 1, dragState = null;
let compYaw = 0.72, compPitch = 0.36, compZoom = 1.0, compDrag = null;
const spatialStates = {};
let spatialDrag = null;
const colors = {sub:"#4575b4", mid:"#6a994e", high:"#d73027", long:"#f97316"};
const clusterColors = {0:"#4575b4", 1:"#6a994e", 2:"#d73027"};
function b64f32(base64) {
  const binary = atob(base64), bytes = new Uint8Array(binary.length);
  const chunk = 1024*1024;
  for (let start=0; start<binary.length; start+=chunk) {
    const end = Math.min(binary.length, start+chunk);
    for (let i=start; i<end; i++) bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}
function getPlane(name) {
  const p = payload.planeData[name];
  if (!decoded[name]) decoded[name] = {dff:b64f32(p.dff), events:p.events ? b64f32(p.events) : null};
  return p;
}
function arrays() { return decoded[plane]; }
function fit(canvas) { const r=window.devicePixelRatio||1, box=canvas.getBoundingClientRect(); canvas.width=Math.max(1,Math.round(box.width*r)); canvas.height=Math.max(1,Math.round(box.height*r)); }
function clear(ctx,c) { ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle="#fff"; ctx.fillRect(0,0,c.width,c.height); }
function fmt(v) { return Number.isFinite(v) ? Number(v).toPrecision(4) : ""; }
function metricValue(row, col) { const v = row ? row[col] : null; return typeof v === "number" && Number.isFinite(v) ? v : NaN; }
function selectedRow() { return data.metrics.find(r=>r.roi_index===selected) || null; }
function traceFor(roi) { return arrays().dff.subarray(roi*data.nFrames, (roi+1)*data.nFrames); }
function eventsFor(roi) { return arrays().events ? arrays().events.subarray(roi*data.nFrames, (roi+1)*data.nFrames) : null; }
function duration() { return (data.nFrames-1)/data.frameRate; }
function median(values) { const v=values.filter(Number.isFinite).sort((a,b)=>a-b); if(!v.length) return NaN; const m=Math.floor(v.length/2); return v.length%2 ? v[m] : (v[m-1]+v[m])/2; }
function percentile(arr,p) { const v=arr.filter(Number.isFinite).sort((a,b)=>a-b); if(!v.length) return NaN; return v[Math.max(0,Math.min(v.length-1,Math.floor((p/100)*(v.length-1))))]; }
function setView(a,b) {
  const dur=duration(), minSpan=Math.max(0.05, 1/data.frameRate);
  a=Math.max(0,Math.min(dur,a)); b=Math.max(0,Math.min(dur,b)); if(b<a) [a,b]=[b,a];
  if(b-a<minSpan) { const mid=(a+b)/2; a=mid-minSpan/2; b=mid+minSpan/2; }
  if(a<0) { b-=a; a=0; } if(b>dur) { a-=b-dur; b=dur; }
  viewStart=Math.max(0,a); viewEnd=Math.min(dur,b);
}
function eventOnsets(ev) {
  if (!ev) return [];
  const out=[]; let prev=false;
  for (let i=0;i<ev.length;i++) { const pos=Number.isFinite(ev[i]) && ev[i]>0; if(pos && !prev) out.push(i); prev=pos; }
  return out;
}
function filteredRows() {
  const label=document.getElementById("labelFilter").value, long=document.getElementById("longFilter").value;
  let rows=data.metrics.slice();
  if(label !== "all") rows=rows.filter(r=>String(r.event_composition_label||"")===label);
  if(long !== "all") rows=rows.filter(r=>r[`has_${long}_long_event_cluster`] === true);
  const dir=document.getElementById("sortDir").value === "asc" ? 1 : -1;
  rows.sort((a,b)=> {
    const av=metricValue(a,metric), bv=metricValue(b,metric);
    if(!Number.isFinite(av) && !Number.isFinite(bv)) return 0;
    if(!Number.isFinite(av)) return 1;
    if(!Number.isFinite(bv)) return -1;
    return dir*(av-bv);
  });
  return rows;
}
function compositionRows() {
  return filteredRows().filter(r =>
    Number.isFinite(metricValue(r,"nonlong_event_fraction_lt_2sd")) &&
    Number.isFinite(metricValue(r,"nonlong_event_fraction_2_4sd")) &&
    Number.isFinite(metricValue(r,"nonlong_event_fraction_gt_4sd"))
  );
}
function allMetricRows() {
  return payload.planes.flatMap(p => (payload.planeData[p].metrics || []));
}
function clusterName(cluster) {
  const rows=allMetricRows().filter(r=>Number(r.nonlong_event_composition_cluster)===Number(cluster));
  const counts={};
  for(const row of rows) {
    const label=String(row.event_composition_label || "").trim();
    if(label) counts[label]=(counts[label]||0)+1;
  }
  const labels=Object.keys(counts).sort((a,b)=>counts[b]-counts[a] || a.localeCompare(b));
  return labels.length ? labels[0].replaceAll("_"," ") : `cluster ${cluster}`;
}
function activeClusterLegend() {
  const clusters=Array.from(new Set(allMetricRows().map(r=>Number(r.nonlong_event_composition_cluster)).filter(Number.isFinite))).sort((a,b)=>a-b);
  return clusters.map(cluster=>({cluster:cluster, label:clusterName(cluster)}));
}
function rowFor(planeName, roi) {
  const pd=payload.planeData[planeName];
  return pd ? pd.metrics.find(r=>r.roi_index===roi) || null : null;
}
function loadPlane(name) {
  plane=name; data=getPlane(plane); selected=0; setView(0, duration());
  document.getElementById("projection").src=data.projection;
  document.getElementById("sessionMeta").textContent=`${data.nRois} ROIs | ${data.nFrames.toLocaleString()} frames | ${data.frameRate.toFixed(3)} Hz`;
  makeOverlay(); fillLabelFilter(); drawAll();
}
function makeOverlay() {
  const svg=document.getElementById("overlay"); svg.replaceChildren(); svg.setAttribute("viewBox", `0 0 ${data.imageWidth} ${data.imageHeight}`);
  for(const r of data.rois) { const path=document.createElementNS("http://www.w3.org/2000/svg","path"); path.setAttribute("d",r.path); path.dataset.roi=r.roi; path.classList.add("roi"); path.addEventListener("click",()=>selectRoi(r.roi)); svg.appendChild(path); }
}
function fillLabelFilter() {
  const select=document.getElementById("labelFilter"), current=select.value;
  const labels=Array.from(new Set(data.metrics.map(r=>r.event_composition_label).filter(Boolean))).sort();
  select.replaceChildren(Object.assign(document.createElement("option"), {value:"all", textContent:"all"}));
  for (const label of labels) select.appendChild(Object.assign(document.createElement("option"), {value:label, textContent:label}));
  select.value = labels.includes(current) ? current : "all";
}
function selectRoi(roi) {
  selected=Math.max(0,Math.min(data.nRois-1,Math.round(roi)));
  setView(0, duration());
  drawAll();
}
function selectPlaneRoi(planeName, roi) {
  if(planeName !== plane) {
    plane=planeName; data=getPlane(plane); selected=Math.max(0,Math.min(data.nRois-1,Math.round(roi)));
    setView(0, duration());
    document.getElementById("planeSelect").value=plane;
    document.getElementById("projection").src=data.projection;
    document.getElementById("sessionMeta").textContent=`${data.nRois} ROIs | ${data.nFrames.toLocaleString()} frames | ${data.frameRate.toFixed(3)} Hz`;
    makeOverlay(); fillLabelFilter();
  } else {
    selected=Math.max(0,Math.min(data.nRois-1,Math.round(roi)));
  }
  setView(0, duration());
  drawAll();
}
function selectAdjacentSorted(delta) {
  const rows=filteredRows();
  if(!rows.length) return;
  let idx=rows.findIndex(r=>r.roi_index===selected);
  if(idx < 0) idx = delta > 0 ? -1 : 0;
  const next=rows[Math.max(0, Math.min(rows.length-1, idx+delta))];
  if(next) selectRoi(next.roi_index);
}
function currentSortPosition() {
  const rows=filteredRows();
  const idx=rows.findIndex(r=>r.roi_index===selected);
  return {index:idx, total:rows.length};
}
function selectedRowTime() {
  const matches=payload.longClusters.filter(c=>c.plane===plane && c.roi_index===selected);
  return matches.length ? matches[0].midpoint_s : NaN;
}
function rotateCompositionPoint(x,y,z) {
  x -= 1/3; y -= 1/3; z -= 1/3;
  const cy=Math.cos(compYaw), sy=Math.sin(compYaw), cp=Math.cos(compPitch), sp=Math.sin(compPitch);
  const x1=cy*x-sy*z, z1=sy*x+cy*z, y1=cp*y-sp*z1, z2=sp*y+cp*z1;
  return [x1,y1,z2];
}
function projectCompositionRow(row,c) {
  const x=metricValue(row,"nonlong_event_fraction_lt_2sd");
  const y=metricValue(row,"nonlong_event_fraction_2_4sd");
  const z=metricValue(row,"nonlong_event_fraction_gt_4sd");
  const [rx,ry,rz]=rotateCompositionPoint(x,y,z);
  const scale=Math.min(c.width,c.height)*0.76*compZoom, perspective=1.7/(1.7-rz);
  return {x:c.width/2+rx*scale*perspective, y:c.height/2-ry*scale*perspective, z:rz};
}
function drawCompositionAxis(ctx,c, from, to, label) {
  const fake = (p) => ({nonlong_event_fraction_lt_2sd:p[0], nonlong_event_fraction_2_4sd:p[1], nonlong_event_fraction_gt_4sd:p[2]});
  const p0=projectCompositionRow(fake(from), c), p1=projectCompositionRow(fake(to), c);
  ctx.strokeStyle="#94a3b8"; ctx.lineWidth=1.2; ctx.beginPath(); ctx.moveTo(p0.x,p0.y); ctx.lineTo(p1.x,p1.y); ctx.stroke();
  ctx.fillStyle="#475467"; ctx.font="12px Arial"; ctx.textAlign="center"; ctx.textBaseline="bottom"; ctx.fillText(label,p1.x,p1.y-7);
  ctx.font="10px Arial"; ctx.textBaseline="top"; ctx.fillText("fraction of non-long event clusters",p1.x,p1.y+5);
}
function drawComposition3d() {
  const c=document.getElementById("compositionCanvas"); fit(c); const ctx=c.getContext("2d"); clear(ctx,c);
  const rows=compositionRows(), size=6;
  drawCompositionAxis(ctx,c,[0,0,0],[1,0,0],"<2 SD");
  drawCompositionAxis(ctx,c,[0,0,0],[0,1,0],"2-4 SD");
  drawCompositionAxis(ctx,c,[0,0,0],[0,0,1],">=4 SD");
  const projected=rows.map(r=>({row:r,...projectCompositionRow(r,c)})).sort((a,b)=>a.z-b.z);
  for(const item of projected) {
    const row=item.row, cluster=Number(row.nonlong_event_composition_cluster);
    const isSel=row.roi_index===selected, hasLong=row.has_long_gt_3s_event_cluster === true;
    ctx.beginPath(); ctx.arc(item.x,item.y,isSel ? size*1.9 : size,0,Math.PI*2);
    ctx.fillStyle=clusterColors[cluster] || "#64748b"; ctx.globalAlpha=isSel ? 1 : 0.72; ctx.fill(); ctx.globalAlpha=1;
    if(hasLong || isSel) {
      ctx.strokeStyle=isSel ? "#f97316" : "#111827"; ctx.lineWidth=isSel ? 3 : 1.2; ctx.stroke();
    }
  }
  ctx.fillStyle="#1f2933"; ctx.font="13px Arial"; ctx.textAlign="left"; ctx.textBaseline="top";
  ctx.fillText(`${rows.length} ROIs in current filter`, 12, 14);
  ctx.fillText("Axes: event-size category fractions per ROI", 12, c.height-24);
  let lx=12;
  for(const item of activeClusterLegend()) {
    ctx.fillStyle=clusterColors[item.cluster] || "#64748b";
    ctx.fillText(item.label, lx, 36);
    lx += Math.max(90, item.label.length * 7 + 18);
  }
  ctx.strokeStyle="#111827"; ctx.lineWidth=1.2; ctx.strokeRect(lx, 29, 12, 12); ctx.fillStyle="#1f2933"; ctx.fillText("has >=3s event cluster", lx+18, 36);
}
function nearestCompositionRoi(event) {
  const c=document.getElementById("compositionCanvas"), box=c.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  const x=(event.clientX-box.left)*dpr, y=(event.clientY-box.top)*dpr, rows=compositionRows();
  let best=null;
  for(const row of rows) {
    const p=projectCompositionRow(row,c), d=(p.x-x)**2+(p.y-y)**2;
    if(!best || d<best.d) best={row,d};
  }
  return best && best.d < 500 ? best.row.roi_index : null;
}
function spatialState(region) {
  if(!spatialStates[region]) spatialStates[region] = {yaw:0.78, pitch:0.46, zoom:1.0};
  return spatialStates[region];
}
function rotateSpatialPoint(region,x,y,z) {
  const state=spatialState(region);
  x -= 0.5; y -= 0.5; z -= 0.5;
  const cy=Math.cos(state.yaw), sy=Math.sin(state.yaw), cp=Math.cos(state.pitch), sp=Math.sin(state.pitch);
  const x1=cy*x-sy*z, z1=sy*x+cy*z, y1=cp*y-sp*z1, z2=sp*y+cp*z1;
  return [x1,y1,z2];
}
function regionPlanes(region) {
  return payload.planes
    .filter(p => (payload.planeData[p].structure || p.replace(/_\d+$/,"")) === region)
    .sort((a,b) => {
      const da=payload.planeData[a].depthUm, db=payload.planeData[b].depthUm;
      const af=Number.isFinite(da), bf=Number.isFinite(db);
      if(af && bf && da !== db) return da-db;
      if(af !== bf) return af ? -1 : 1;
      return String(a).localeCompare(String(b));
    });
}
function spatialPoints(region) {
  const out=[], planes=regionPlanes(region);
  const denom=Math.max(1, planes.length-1);
  for(let zi=0; zi<planes.length; zi++) {
    const planeName=planes[zi], pd=payload.planeData[planeName];
    if(!pd || !pd.rois) continue;
    const z=zi/denom;
    for(const roi of pd.rois) {
      if(!Number.isFinite(roi.cx) || !Number.isFinite(roi.cy)) continue;
      const row=rowFor(planeName, roi.roi);
      out.push({
        plane:planeName,
        roi:roi.roi,
        row:row,
        x:roi.cx/Math.max(1,pd.imageWidth),
        y:roi.cy/Math.max(1,pd.imageHeight),
        z:z,
        depthUm:pd.depthUm,
        depthSource:pd.depthSource,
        depthRank:zi,
        planeIndex:zi,
      });
    }
  }
  return out;
}
function projectSpatialPoint(region, point,c) {
  const state=spatialState(region);
  const [rx,ry,rz]=rotateSpatialPoint(region, point.x, point.y, point.z);
  const scale=Math.min(c.width,c.height)*0.74*state.zoom, perspective=1.8/(1.8-rz);
  return {x:c.width/2+rx*scale*perspective, y:c.height/2-ry*scale*perspective, z:rz};
}
function drawSpatialAxis(ctx,c, region, from, to, label) {
  const p0=projectSpatialPoint(region, {x:from[0], y:from[1], z:from[2]}, c);
  const p1=projectSpatialPoint(region, {x:to[0], y:to[1], z:to[2]}, c);
  ctx.strokeStyle="#94a3b8"; ctx.lineWidth=1.2; ctx.beginPath(); ctx.moveTo(p0.x,p0.y); ctx.lineTo(p1.x,p1.y); ctx.stroke();
  ctx.fillStyle="#475467"; ctx.font="12px Arial"; ctx.textAlign="center"; ctx.textBaseline="bottom"; ctx.fillText(label,p1.x,p1.y-7);
}
function drawSpatial3d(region) {
  const c=document.querySelector(`canvas[data-region="${region}"]`);
  if(!c) return;
  fit(c); const ctx=c.getContext("2d"); clear(ctx,c);
  const pts=spatialPoints(region), size=4;
  drawSpatialAxis(ctx,c,region,[0,0,0],[1,0,0],"image x");
  drawSpatialAxis(ctx,c,region,[0,0,0],[0,1,0],"image y");
  drawSpatialAxis(ctx,c,region,[0,0,0],[0,0,1],"depth order");
  const projected=pts.map(p=>({point:p,...projectSpatialPoint(region,p,c)})).sort((a,b)=>a.z-b.z);
  for(const item of projected) {
    const p=item.point, row=p.row || {}, isCurrentPlane=p.plane===plane, isSel=isCurrentPlane && p.roi===selected;
    const cluster=Number(row.nonlong_event_composition_cluster);
    ctx.beginPath(); ctx.arc(item.x,item.y,isSel ? size*2.3 : (isCurrentPlane ? size*1.35 : size),0,Math.PI*2);
    ctx.fillStyle=clusterColors[cluster] || (isCurrentPlane ? "#64748b" : "#cbd5e1");
    ctx.globalAlpha=isSel ? 1 : (isCurrentPlane ? 0.78 : 0.38); ctx.fill(); ctx.globalAlpha=1;
    if(row.has_long_gt_3s_event_cluster === true || isSel) {
      ctx.strokeStyle=isSel ? "#f97316" : "#111827"; ctx.lineWidth=isSel ? 3 : 1; ctx.stroke();
    }
  }
  ctx.fillStyle="#1f2933"; ctx.font="13px Arial"; ctx.textAlign="left"; ctx.textBaseline="top";
  const planes=regionPlanes(region);
  const depths=planes.map(p=>payload.planeData[p].depthUm).filter(Number.isFinite);
  const depthLabel=planes.map(p=> {
    const d=payload.planeData[p].depthUm;
    return `${p}:${Number.isFinite(d) ? d.toFixed(0)+" um" : "order"}`;
  }).join("  ");
  ctx.fillText(`${region}: ${pts.length} ROIs, ${planes.length} planes`, 12, 14);
  ctx.fillText(`official depths: ${depthLabel}`, 12, 34);
  ctx.fillText(`selected: ${plane} ROI ${selected}`, 12, 54);
  let lx=12;
  for(const item of activeClusterLegend()) {
    ctx.fillStyle=clusterColors[item.cluster] || "#64748b";
    ctx.fillText(item.label, lx, c.height-24);
    lx += Math.max(90, item.label.length * 7 + 18);
  }
}
function drawSpatialRegions() {
  for(const region of payload.regions) drawSpatial3d(region);
}
function nearestSpatialPoint(event, region, c) {
  const box=c.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  const x=(event.clientX-box.left)*dpr, y=(event.clientY-box.top)*dpr;
  let best=null;
  for(const point of spatialPoints(region)) {
    const p=projectSpatialPoint(region,point,c), d=(p.x-x)**2+(p.y-y)**2;
    if(!best || d<best.d) best={point,d};
  }
  return best && best.d < 360 ? best.point : null;
}
function drawTable() {
  const table=document.getElementById("roiTable"), rows=filteredRows().slice(0,220);
  const cols=["roi_index","event_composition_label","event_composition_cluster",metric,"roi_area_pixels","soma_probability","max_event_cluster_span_s","n_warning_long_event_clusters"];
  table.innerHTML="<thead><tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead><tbody></tbody>";
  const tbody=table.querySelector("tbody");
  for(const row of rows) {
    const tr=document.createElement("tr"); tr.classList.toggle("selected", row.roi_index===selected);
    tr.addEventListener("click",()=>selectRoi(row.roi_index));
    tr.innerHTML=cols.map(c=>`<td>${c==="roi_index" ? row[c] : (typeof row[c] === "string" ? row[c] : fmt(metricValue(row,c)))}</td>`).join("");
    tbody.appendChild(tr);
  }
}
function drawHist() {
  const c=document.getElementById("histCanvas"); fit(c); const ctx=c.getContext("2d"); clear(ctx,c);
  const vals=data.metrics.map(r=>metricValue(r,metric)).filter(Number.isFinite); if(!vals.length) return;
  const min=Math.min(...vals), max=Math.max(...vals), pad=(max-min||1)*.05, x0=min-pad, x1=max+pad, bins=50, counts=Array(bins).fill(0);
  for(const v of vals) counts[Math.max(0,Math.min(bins-1,Math.floor((v-x0)/(x1-x0)*bins)))]++;
  const l=58,t=18,w=c.width-78,h=c.height-56,ymax=Math.max(...counts,1); ctx.fillStyle="#64748b";
  counts.forEach((n,i)=>{ const x=l+i*w/bins, bh=n/ymax*h; ctx.fillRect(x,t+h-bh,w/bins-1,bh); });
  const row=data.metrics.find(r=>r.roi_index===selected), sv=metricValue(row,metric);
  if(Number.isFinite(sv)) { const x=l+(sv-x0)/(x1-x0)*w; ctx.strokeStyle="#c2410c"; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(x,t); ctx.lineTo(x,t+h); ctx.stroke(); }
  ctx.strokeStyle="#d0d5dd"; ctx.strokeRect(l,t,w,h);
  ctx.fillStyle="#475467"; ctx.font="12px Arial"; ctx.textAlign="center"; ctx.textBaseline="top"; ctx.fillText(metric,l+w/2,c.height-18);
  ctx.textBaseline="top";
  for(let tick=0; tick<=4; tick++) {
    const value=x0+(tick/4)*(x1-x0), x=l+(tick/4)*w;
    ctx.strokeStyle="#d0d5dd"; ctx.beginPath(); ctx.moveTo(x,t+h); ctx.lineTo(x,t+h+4); ctx.stroke();
    ctx.fillStyle="#475467"; ctx.fillText(fmt(value), x, t+h+7);
  }
  ctx.save(); ctx.translate(14,t+h/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText("ROI count",0,0); ctx.restore();
  ctx.textAlign="right"; ctx.textBaseline="middle"; ctx.fillText("0",l-6,t+h);
  ctx.fillText(String(ymax),l-6,t);
}
function selectedClusterRows() {
  const row=selectedRow() || {};
  const cluster=Number(row.nonlong_event_composition_cluster);
  if(!Number.isFinite(cluster)) return [];
  return data.metrics.filter(r=>Number(r.nonlong_event_composition_cluster)===cluster);
}
function clustersForRoi(planeName, roi) {
  return (payload.eventClusters || []).filter(c=>c.plane===planeName && c.roi_index===roi);
}
function finiteMetric(rows, col) {
  return rows.map(r=>metricValue(r,col)).filter(Number.isFinite);
}
function mean(values) {
  return values.length ? values.reduce((a,b)=>a+b,0)/values.length : NaN;
}
function normalPdf(x, mu, sd) {
  return Math.exp(-0.5*((x-mu)/sd)**2)/(sd*Math.sqrt(2*Math.PI));
}
function expPdf(x, scale) {
  return x >= 0 && scale > 0 ? Math.exp(-x/scale)/scale : 0;
}
function eventValuesForRoi(roi) {
  const ev=eventsFor(roi);
  if(!ev) return [];
  const vals=[];
  for(let i=0;i<ev.length;i++) if(Number.isFinite(ev[i]) && ev[i]>0) vals.push(ev[i]);
  vals.sort((a,b)=>a-b);
  return vals;
}
function expResiduals(events, scale) {
  const residuals=[], n=events.length;
  if(!n || !Number.isFinite(scale) || scale<=0) return residuals;
  for(let i=0;i<n;i++) {
    const p=(i+0.5)/n;
    const expected=-scale*Math.log(Math.max(1e-12, 1-p));
    residuals.push(events[i]-expected);
  }
  return residuals;
}
function drawDensityPanel(ctx, bounds, values, pdfFn, title, xLabel, fillColor, lineColor) {
  const [l,t,w,h]=bounds;
  const finite=values.filter(Number.isFinite);
  ctx.strokeStyle="#d0d5dd"; ctx.strokeRect(l,t,w,h);
  ctx.fillStyle="#101828"; ctx.font="12px Arial"; ctx.textAlign="left"; ctx.textBaseline="top"; ctx.fillText(title,l,t-16);
  if(finite.length < 3) {
    ctx.fillStyle="#667085"; ctx.fillText("not enough samples", l+8, t+8);
    return;
  }
  let x0=Math.min(...finite), x1=Math.max(...finite);
  if(x0===x1) { x0-=1; x1+=1; }
  const pad=(x1-x0)*0.06; x0-=pad; x1+=pad;
  const bins=36, counts=Array(bins).fill(0), dx=(x1-x0)/bins;
  for(const v of finite) counts[Math.max(0,Math.min(bins-1,Math.floor((v-x0)/(x1-x0)*bins)))]++;
  const densities=counts.map(n=>n/(finite.length*dx));
  let ymax=Math.max(...densities, 1e-12);
  const curve=[];
  for(let i=0;i<=160;i++) {
    const x=x0+(i/160)*(x1-x0), y=pdfFn(x);
    if(Number.isFinite(y)) { curve.push([x,y]); if(y>ymax) ymax=y; }
  }
  const xOf=x=>l+((x-x0)/(x1-x0))*w, yOf=y=>t+h-(y/ymax)*h;
  ctx.fillStyle=fillColor;
  densities.forEach((d,i)=>{ const x=l+i*w/bins, bh=(d/ymax)*h; ctx.fillRect(x,t+h-bh,w/bins-1,bh); });
  ctx.strokeStyle=lineColor; ctx.lineWidth=2; ctx.beginPath();
  curve.forEach(([x,y],i)=>{ const px=xOf(x), py=yOf(y); if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py); });
  ctx.stroke();
  ctx.fillStyle="#475467"; ctx.font="11px Arial"; ctx.textAlign="center"; ctx.textBaseline="top";
  for(let tick=0;tick<=4;tick++) {
    const x=x0+(tick/4)*(x1-x0), px=xOf(x);
    ctx.strokeStyle="#d0d5dd"; ctx.beginPath(); ctx.moveTo(px,t+h); ctx.lineTo(px,t+h+4); ctx.stroke();
    ctx.fillText(fmt(x), px, t+h+6);
  }
  ctx.fillText(xLabel, l+w/2, t+h+22);
  ctx.save(); ctx.translate(l-38,t+h/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText("density",0,0); ctx.restore();
}
function drawFitDiagnostics() {
  const c=document.getElementById("fitCanvas"); fit(c); const ctx=c.getContext("2d"); clear(ctx,c);
  const row=selectedRow() || {}, clusterRows=selectedClusterRows();
  const ev=eventValuesForRoi(selected);
  const scale=mean(ev);
  const residuals=expResiduals(ev, scale);
  const mu=mean(residuals), sd=Math.sqrt(mean(residuals.map(v=>(v-mu)**2)));
  const l=62, top=30, gap=42, panelH=Math.max(62, (c.height-78-gap)/2), w=c.width-88;
  drawDensityPanel(ctx, [l,top,w,panelH], ev, x=>expPdf(x, scale), "Selected ROI event amplitudes vs exponential", "event amplitude", "rgba(29,78,216,.45)", "#dc2626");
  drawDensityPanel(ctx, [l,top+panelH+gap,w,panelH], residuals, x=>normalPdf(x, mu, sd), "Exponential-quantile residuals vs Gaussian", "residual amplitude", "rgba(100,116,139,.62)", "#dc2626");
  const itemCols=["event_exponential_ks_stat","event_model_residual_gaussian_ks_stat","event_exp_gauss_fit_score"];
  const cluster=Number(row.nonlong_event_composition_cluster);
  const label=row.event_composition_label || "";
  const clusterText=Number.isFinite(cluster) ? clusterName(cluster) : "no cluster";
  document.getElementById("fitReadout").innerHTML =
    `<div class="pill">${plane}</div><div class="pill">ROI ${selected}</div><div class="pill">${clusterText}</div><div class="pill">${label}</div>` +
    `<div class="pill">event samples: ${ev.length}</div><div class="pill">exp scale: ${fmt(scale)}</div><div class="pill">residual mu/sd: ${fmt(mu)} / ${fmt(sd)}</div>` +
    itemCols.map(col=>`<div class="pill">${col}: ${fmt(metricValue(row,col))}</div>`).join("") +
    `<div class="pill">selected-plane cluster n: ${clusterRows.length}</div>` +
    `<div class="pill">cluster median exp KS: ${fmt(median(finiteMetric(clusterRows,"event_exponential_ks_stat")))}</div>` +
    `<div class="pill">cluster median Gaussian residual KS: ${fmt(median(finiteMetric(clusterRows,"event_model_residual_gaussian_ks_stat")))}</div>`;
}
function drawLongList() {
  const div=document.getElementById("longList"); div.replaceChildren();
  for(const c of payload.longClusters.filter(c=>c.plane===plane).slice(0,80)) {
    const btn=document.createElement("button");
    btn.textContent=`ROI ${c.roi_index} | span ${fmt(c.cluster_span_s)}s | midpoint ${fmt(c.midpoint_s)}s | onsets ${c.raw_onsets_in_cluster ?? ""}`;
    btn.addEventListener("click",()=>{ selected=c.roi_index; setView(0, duration()); drawAll(); });
    div.appendChild(btn);
  }
}
function drawRoiTrace() {
  const c=document.getElementById("roiTraceCanvas"); fit(c); const ctx=c.getContext("2d"); clear(ctx,c);
  const tr=traceFor(selected), ev=eventsFor(selected), f0=Math.max(0,Math.floor(viewStart*data.frameRate)), f1=Math.min(data.nFrames-1,Math.ceil(viewEnd*data.frameRate));
  const l=72,t=18,w=c.width-92,h=c.height-66, visible=tr.subarray(f0,f1+1);
  let ymin=Infinity, ymax=-Infinity;
  for(let i=0;i<visible.length;i++) {
    const v=visible[i];
    if(Number.isFinite(v)) { if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
  }
  if(!Number.isFinite(ymin) || !Number.isFinite(ymax) || ymin===ymax) { ymin=-1; ymax=1; }
  const padY=(ymax-ymin)*0.08 || 1;
  ymin-=padY; ymax+=padY;
  const xOf=i=>l+((i/data.frameRate-viewStart)/(viewEnd-viewStart))*w, yOf=v=>t+(1-((v-ymin)/(ymax-ymin)))*h;
  ctx.strokeStyle="#d0d5dd"; ctx.beginPath(); ctx.moveTo(l,t); ctx.lineTo(l,t+h); ctx.lineTo(l+w,t+h); ctx.stroke();
  const windowS=viewEnd-viewStart, useMs=windowS <= 2.0;
  ctx.fillStyle="#475467"; ctx.font="12px Arial"; ctx.textAlign="right"; ctx.textBaseline="middle";
  for(let tick=0; tick<=4; tick++) {
    const val=ymin+(tick/4)*(ymax-ymin), y=yOf(val);
    ctx.strokeStyle="#eef0f2"; ctx.beginPath(); ctx.moveTo(l,y); ctx.lineTo(l+w,y); ctx.stroke();
    ctx.fillStyle="#475467"; ctx.fillText(val.toFixed(2), l-8, y);
  }
  ctx.textAlign="center"; ctx.textBaseline="top";
  for(let tick=0; tick<=5; tick++) {
    const timeS=viewStart+(tick/5)*windowS, x=l+(tick/5)*w;
    ctx.strokeStyle="#eef0f2"; ctx.beginPath(); ctx.moveTo(x,t); ctx.lineTo(x,t+h); ctx.stroke();
    ctx.fillStyle="#475467";
    const label=useMs ? `${((timeS-viewStart)*1000).toFixed(0)} ms` : `${timeS.toFixed(windowS < 20 ? 1 : 0)} s`;
    ctx.fillText(label, x, t+h+8);
  }
  ctx.fillText(useMs ? "time from window start (ms)" : "session time (s)", l+w/2, c.height-18);
  ctx.save(); ctx.translate(18,t+h/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText("dF/F",0,0); ctx.restore();
  const clusters=clustersForRoi(plane, selected);
  for(const cl of clusters) {
    const a=Number(cl.evaluation_start_s), b=Number(cl.evaluation_end_s), mid=Number(cl.midpoint_s);
    if(!Number.isFinite(a) || !Number.isFinite(b) || b < viewStart || a > viewEnd) continue;
    const xa=l+((Math.max(a,viewStart)-viewStart)/(viewEnd-viewStart))*w;
    const xb=l+((Math.min(b,viewEnd)-viewStart)/(viewEnd-viewStart))*w;
    ctx.fillStyle=String(cl.event_type||"").includes("long") ? "rgba(249,115,22,.20)" : "rgba(251,146,60,.13)";
    ctx.fillRect(xa,t,Math.max(1,xb-xa),h);
    if(Number.isFinite(mid) && mid >= viewStart && mid <= viewEnd) {
      const xm=l+((mid-viewStart)/(viewEnd-viewStart))*w;
      ctx.strokeStyle="#f97316"; ctx.lineWidth=1.3; ctx.beginPath(); ctx.moveTo(xm,t); ctx.lineTo(xm,t+h); ctx.stroke();
    }
  }
  if(ev) { ctx.strokeStyle="rgba(239,68,68,.45)"; ctx.lineWidth=1; for(let i=f0+1;i<=f1;i++) if(ev[i]>0 && !(ev[i-1]>0)) { const x=xOf(i); ctx.beginPath(); ctx.moveTo(x,t); ctx.lineTo(x,t+h); ctx.stroke(); } }
  ctx.strokeStyle="#1d4ed8"; ctx.lineWidth=1.5; ctx.beginPath();
  const spanFrames=Math.max(2, f1-f0+1), columns=Math.max(1,Math.floor(w)), framesPerPixel=spanFrames/columns;
  if(framesPerPixel <= 1.5) {
    let first=true;
    for(let i=f0;i<=f1;i++) {
      const x=xOf(i), y=yOf(tr[i]);
      if(first) { ctx.moveTo(x,y); first=false; } else ctx.lineTo(x,y);
    }
  } else {
    for(let col=0; col<columns; col++) {
      const a=Math.floor(f0+col*framesPerPixel), b=Math.min(f1,Math.floor(f0+(col+1)*framesPerPixel));
      let minV=Infinity, maxV=-Infinity;
      for(let frame=a; frame<=b; frame++) {
        const v=tr[frame];
        if(Number.isFinite(v)) { if(v<minV) minV=v; if(v>maxV) maxV=v; }
      }
      if(Number.isFinite(minV) && Number.isFinite(maxV)) {
        const x=l+col;
        ctx.moveTo(x,yOf(minV)); ctx.lineTo(x,yOf(maxV));
      }
    }
  }
  const pos=currentSortPosition(), posText=pos.index >= 0 ? ` | sorted ${pos.index+1}/${pos.total}` : "";
  ctx.stroke(); ctx.fillStyle="#101828"; ctx.font="12px Arial"; ctx.textAlign="left"; ctx.textBaseline="top"; ctx.fillText(`ROI ${selected}${posText} | ${viewStart.toFixed(1)}-${viewEnd.toFixed(1)}s`, l+4, 6);
  ctx.fillStyle="#475467"; ctx.textAlign="right";
  ctx.fillText("red: raw event onset | orange: merged event window/midpoint", l+w-4, 6);
}
function drawReadout() {
  const row=selectedRow() || {};
  document.querySelectorAll(".roi").forEach(p=>p.classList.toggle("selected", Number(p.dataset.roi)===selected));
  const fields=["event_composition_label","background_event_median_amp_noise_units","background_event_fraction_gt_4sd","trace_event_fraction_gt_4sd","roi_area_pixels","soma_probability","event_exp_gauss_fit_score","max_event_cluster_span_s"];
  const pos=currentSortPosition(), posText=pos.index >= 0 ? `${pos.index+1}/${pos.total}` : "not in current filter";
  document.getElementById("roiReadout").innerHTML=`<div class="pill">ROI ${selected}</div><div class="pill">sort position: ${posText}</div>`+fields.map(f=>`<div class="pill">${f}: ${typeof row[f]==="string" ? row[f] : fmt(metricValue(row,f))}</div>`).join("");
}
function drawAll() { drawComposition3d(); drawSpatialRegions(); drawTable(); drawHist(); drawFitDiagnostics(); drawLongList(); drawRoiTrace(); drawReadout(); }
function installTraceInteractions(canvas) {
  canvas.addEventListener("wheel", e=>{ e.preventDefault(); const rect=canvas.getBoundingClientRect(), frac=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)), center=viewStart+frac*(viewEnd-viewStart), scale=e.deltaY<0?.78:1.28, span=Math.max(.5,Math.min(duration(),(viewEnd-viewStart)*scale)); setView(center-frac*span, center+(1-frac)*span); drawAll(); }, {passive:false});
  canvas.addEventListener("mousedown", e=>{ dragState={x:e.clientX,a:viewStart,b:viewEnd}; canvas.style.cursor="grabbing"; });
  window.addEventListener("mousemove", e=>{ if(!dragState) return; const rect=canvas.getBoundingClientRect(), shift=-(e.clientX-dragState.x)/rect.width*(dragState.b-dragState.a); setView(dragState.a+shift, dragState.b+shift); drawAll(); });
  window.addEventListener("mouseup", ()=>{ dragState=null; canvas.style.cursor="grab"; });
  canvas.addEventListener("dblclick", ()=>{ setView(0,duration()); drawAll(); });
}
document.getElementById("planeSelect").addEventListener("change", e=>loadPlane(e.target.value));
document.getElementById("metricSelect").addEventListener("change", e=>{ metric=e.target.value; drawAll(); });
["sortDir","labelFilter","longFilter"].forEach(id=>document.getElementById(id).addEventListener("change", drawAll));
document.getElementById("resetView").addEventListener("click",()=>{ setView(0,duration()); drawAll(); });
window.addEventListener("resize", drawAll);
installTraceInteractions(document.getElementById("roiTraceCanvas"));
function installSpatialRegionCanvases() {
  const grid=document.getElementById("spatialRegionGrid");
  grid.replaceChildren();
  for(const region of payload.regions) {
    spatialState(region);
    const wrap=document.createElement("div");
    const title=document.createElement("div");
    title.className="panel-title";
    title.style.marginBottom="4px";
    title.textContent=region;
    const canvas=document.createElement("canvas");
    canvas.className="region-canvas";
    canvas.dataset.region=region;
    canvas.addEventListener("mousedown", e=>{ spatialDrag={region:region,canvas:canvas,x:e.clientX,y:e.clientY,moved:false}; canvas.style.cursor="grabbing"; });
    canvas.addEventListener("wheel", e=>{ e.preventDefault(); const state=spatialState(region); state.zoom=Math.max(.45,Math.min(2.6,state.zoom*(e.deltaY<0?1.12:.88))); drawSpatial3d(region); }, {passive:false});
    wrap.appendChild(title);
    wrap.appendChild(canvas);
    grid.appendChild(wrap);
  }
}
const compCanvas=document.getElementById("compositionCanvas");
compCanvas.addEventListener("mousedown", e=>{ compDrag={x:e.clientX,y:e.clientY,moved:false}; compCanvas.style.cursor="grabbing"; });
window.addEventListener("mousemove", e=>{ if(!compDrag) return; const dx=e.clientX-compDrag.x, dy=e.clientY-compDrag.y; if(Math.abs(dx)+Math.abs(dy)>2) compDrag.moved=true; compYaw+=dx*.008; compPitch=Math.max(-1.35,Math.min(1.35,compPitch+dy*.008)); compDrag.x=e.clientX; compDrag.y=e.clientY; drawComposition3d(); });
window.addEventListener("mouseup", e=>{ if(compDrag && !compDrag.moved) { const roi=nearestCompositionRoi(e); if(roi !== null) selectRoi(roi); } compDrag=null; compCanvas.style.cursor="grab"; });
compCanvas.addEventListener("wheel", e=>{ e.preventDefault(); compZoom=Math.max(.45,Math.min(2.6,compZoom*(e.deltaY<0?1.12:.88))); drawComposition3d(); }, {passive:false});
window.addEventListener("mousemove", e=>{ if(!spatialDrag) return; const state=spatialState(spatialDrag.region), dx=e.clientX-spatialDrag.x, dy=e.clientY-spatialDrag.y; if(Math.abs(dx)+Math.abs(dy)>2) spatialDrag.moved=true; state.yaw+=dx*.008; state.pitch=Math.max(-1.35,Math.min(1.35,state.pitch+dy*.008)); spatialDrag.x=e.clientX; spatialDrag.y=e.clientY; drawSpatial3d(spatialDrag.region); });
window.addEventListener("mouseup", e=>{ if(spatialDrag && !spatialDrag.moved) { const p=nearestSpatialPoint(e, spatialDrag.region, spatialDrag.canvas); if(p !== null) selectPlaneRoi(p.plane, p.roi); } document.querySelectorAll(".region-canvas").forEach(c=>c.style.cursor="grab"); spatialDrag=null; });
document.addEventListener("keydown", e=>{
  if(e.target && ["INPUT","TEXTAREA"].includes(e.target.tagName)) return;
  if(e.key === "ArrowRight") { e.preventDefault(); selectAdjacentSorted(1); }
  if(e.key === "ArrowLeft") { e.preventDefault(); selectAdjacentSorted(-1); }
});
function setDrawer(open) {
  document.getElementById("metricsDrawer").classList.toggle("open", open);
  document.getElementById("metricsBackdrop").classList.toggle("open", open);
}
function setLongDrawer(open) {
  document.getElementById("longDrawer").classList.toggle("open", open);
  document.getElementById("metricsBackdrop").classList.toggle("open", open);
}
document.getElementById("openMetricsDrawer").addEventListener("click",()=>setDrawer(true));
document.getElementById("closeMetricsDrawer").addEventListener("click",()=>setDrawer(false));
document.getElementById("openLongDrawer").addEventListener("click",()=>setLongDrawer(true));
document.getElementById("closeLongDrawer").addEventListener("click",()=>setLongDrawer(false));
document.getElementById("metricsBackdrop").addEventListener("click",()=>{ setDrawer(false); setLongDrawer(false); });
installSpatialRegionCanvases();
loadPlane(plane);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--clusters-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plane", action="append", dest="planes")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-long-clusters", type=int, default=500)
    parser.add_argument("--max-clusters", type=int, default=200000)
    args = parser.parse_args()
    out = create_event_cluster_review_html(
        args.session_dir.expanduser().resolve(),
        args.metrics_csv.expanduser().resolve(),
        args.output.expanduser().resolve(),
        clusters_csv=args.clusters_csv.expanduser().resolve() if args.clusters_csv else None,
        planes=args.planes,
        max_frames=args.max_frames,
        max_long_clusters=args.max_long_clusters,
        max_clusters=args.max_clusters,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
