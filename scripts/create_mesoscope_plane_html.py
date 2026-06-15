#!/usr/bin/env python3
"""Create a standalone interactive HTML viewer for one materialized mesoscope plane."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return cleaned or "unknown"


def _normalize_image(image: np.ndarray, p_low: float = 1, p_high: float = 99.8) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = np.percentile(finite, [p_low, p_high])
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _green_png_data_uri(image: np.ndarray) -> str:
    gray = _normalize_image(image)
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    rgb[..., 1] = gray
    buffer = io.BytesIO()
    plt.imsave(buffer, rgb, format="png")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mask_png_data_uri(mask: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    masked = np.ma.masked_where(mask <= 0, mask)
    ax.imshow(masked, cmap="turbo", interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, transparent=True)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pixel_runs_path(xpix: np.ndarray, ypix: np.ndarray) -> str:
    segments: list[str] = []
    if not xpix.size:
        return ""
    for y in np.unique(ypix):
        row_x = np.sort(np.unique(xpix[ypix == y]))
        start = int(row_x[0])
        previous = start
        for x in row_x[1:]:
            x = int(x)
            if x != previous + 1:
                width = previous - start + 1
                segments.append(f"M{start} {int(y)}h{width}v1h-{width}z")
                start = x
            previous = x
        width = previous - start + 1
        segments.append(f"M{start} {int(y)}h{width}v1h-{width}z")
    return "".join(segments)


def _float32_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes()).decode("ascii")


def _load_sparse_masks(mask_path: Path) -> tuple[np.ndarray, tuple[int, int], list[np.ndarray]]:
    with np.load(mask_path) as data:
        roi_indices = data["roi_indices"].astype(np.int32)
        shape = tuple(int(v) for v in data["shape"])
        indptr = data["indptr"].astype(np.int64)
        xyw = data["xyw"].astype(np.float32)
    masks = [xyw[indptr[i] : indptr[i + 1]] for i in range(len(indptr) - 1)]
    return roi_indices, shape, masks


def _load_plane_arrays(session_dir: Path, plane: str, max_frames: int | None) -> dict:
    plane_safe = _safe_name(plane)
    plane_dir = session_dir / plane_safe
    projection_path = plane_dir / "projection.npy"
    mask_path = session_dir / f"{plane_safe}_pixel_masks.npz"
    dff_path = plane_dir / "dff.npy"
    event_path = plane_dir / "events.npy"
    ts_path = plane_dir / "dff_timestamps.npy"

    if not projection_path.exists():
        raise FileNotFoundError(f"Missing projection: {projection_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing sparse masks: {mask_path}")
    if not dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F: {dff_path}")

    projection = np.load(projection_path).astype(np.float32, copy=False)
    roi_indices, shape, pixel_masks = _load_sparse_masks(mask_path)
    dff = np.load(dff_path, mmap_mode="r")
    events = np.load(event_path, mmap_mode="r") if event_path.exists() else None

    if dff.shape[0] >= dff.shape[1]:
        dff_frames_rois = dff
    else:
        dff_frames_rois = dff.T
    n_frames = dff_frames_rois.shape[0] if max_frames is None else min(int(max_frames), dff_frames_rois.shape[0])
    dff_roi_frames = np.asarray(dff_frames_rois[:n_frames, : len(pixel_masks)].T, dtype=np.float32)

    event_roi_frames = None
    if events is not None:
        event_frames_rois = events if events.shape[0] >= events.shape[1] else events.T
        event_roi_frames = np.asarray(event_frames_rois[:n_frames, : len(pixel_masks)].T, dtype=np.float32)

    frame_rate = 1.0
    if ts_path.exists():
        timestamps = np.load(ts_path, mmap_mode="r")[:n_frames]
        diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            frame_rate = float(1.0 / np.median(diffs))

    return {
        "projection": projection,
        "roi_indices": roi_indices,
        "shape": shape,
        "pixel_masks": pixel_masks,
        "dff": dff_roi_frames,
        "events": event_roi_frames,
        "frame_rate": frame_rate,
    }


def _build_roi_payload(roi_indices: np.ndarray, pixel_masks: list[np.ndarray], shape: tuple[int, int]) -> tuple[list[dict], np.ndarray]:
    label_mask = np.zeros(shape, dtype=np.int32)
    rois = []
    for out_i, (roi_id, pix) in enumerate(zip(roi_indices, pixel_masks)):
        if pix.size:
            x = np.clip(pix[:, 0].astype(int), 0, shape[1] - 1)
            y = np.clip(pix[:, 1].astype(int), 0, shape[0] - 1)
            w = pix[:, 2].astype(np.float32)
            label_mask[y, x] = out_i + 1
            cx = float(np.average(x, weights=np.maximum(w, 1e-6))) if x.size else np.nan
            cy = float(np.average(y, weights=np.maximum(w, 1e-6))) if y.size else np.nan
        else:
            x = np.empty(0, dtype=int)
            y = np.empty(0, dtype=int)
            cx = cy = np.nan
        rois.append(
            {
                "roi": int(out_i),
                "sourceRoi": int(roi_id),
                "path": _pixel_runs_path(x, y),
                "npix": int(x.size),
                "cx": cx,
                "cy": cy,
            }
        )
    return rois, label_mask


def create_plane_html(session_dir: Path, plane: str, output_path: Path, max_frames: int | None = None) -> Path:
    arrays = _load_plane_arrays(session_dir, plane, max_frames=max_frames)
    projection = arrays["projection"]
    shape = tuple(projection.shape[:2])
    mask_shape = tuple(arrays["shape"])
    if mask_shape != shape:
        shape = mask_shape
    rois, label_mask = _build_roi_payload(arrays["roi_indices"], arrays["pixel_masks"], shape)
    dff = arrays["dff"]
    events = arrays["events"]
    n_rois, n_frames = dff.shape
    image_height, image_width = shape
    payload = {
        "session": session_dir.name,
        "plane": plane,
        "frameRate": float(arrays["frame_rate"]),
        "nRois": int(n_rois),
        "nFrames": int(n_frames),
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "projection": _green_png_data_uri(projection),
        "mask": _mask_png_data_uri(label_mask),
        "rois": rois[:n_rois],
        "dff": _float32_b64(dff),
        "events": _float32_b64(events) if events is not None else None,
        "eventsAvailable": events is not None,
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{session_dir.name} {plane} ROI viewer</title>
<style>
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #202124; background: #f6f7f8; }}
.page {{ width: min(1680px, calc(100vw - 28px)); margin: 16px auto 26px; }}
.head {{ display: flex; justify-content: space-between; gap: 14px; align-items: end; margin-bottom: 12px; }}
h1 {{ margin: 0; font-size: 21px; letter-spacing: 0; }}
.meta {{ color: #667085; font-size: 13px; text-align: right; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
.panel {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 7px; padding: 10px; box-sizing: border-box; }}
.title {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; }}
.imagewrap {{ position: relative; width: 100%; aspect-ratio: 1/1; background: #111; overflow: hidden; }}
.imagewrap img, .imagewrap svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.imagewrap img {{ object-fit: contain; image-rendering: pixelated; }}
.roi {{ fill: transparent; stroke: rgba(255,255,255,.88); stroke-width: .7; cursor: pointer; vector-effect: non-scaling-stroke; pointer-events: all; }}
.roi:hover {{ fill: rgba(6,182,212,.22); stroke: #06b6d4; stroke-width: 1.6; }}
.roi.selected {{ fill: rgba(220,38,38,.22); stroke: #dc2626; stroke-width: 1.8; }}
.controls {{ display: grid; grid-template-columns: 1fr auto auto auto auto auto auto; gap: 9px; align-items: center; margin-top: 10px; }}
button, input, select {{ font: inherit; }}
button {{ border: 1px solid #d0d5dd; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
input, select {{ border: 1px solid #d0d5dd; border-radius: 6px; padding: 7px 8px; width: 86px; }}
select {{ width: 112px; }}
canvas {{ width: 100%; display: block; background: #fff; border: 1px solid #d0d5dd; box-sizing: border-box; }}
#stackCanvas {{ height: 560px; cursor: crosshair; }}
#traceCanvas {{ height: 250px; cursor: grab; }}
#traceCanvas.dragging {{ cursor: grabbing; }}
.plots {{ display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 10px; }}
.note {{ margin-top: 6px; color: #667085; font-size: 12px; }}
@media (max-width: 980px) {{ .grid, .controls {{ grid-template-columns: 1fr; }} .head {{ display: block; }} .meta {{ text-align: left; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="head"><h1>{session_dir.name} | {plane}</h1><div class="meta" id="meta"></div></div>
  <div class="grid">
    <div class="panel"><div class="title">Functional projection</div><div class="imagewrap"><img id="projection"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
    <div class="panel"><div class="title">ROI masks</div><div class="imagewrap"><img id="mask"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
  </div>
  <div class="panel controls">
    <div id="readout"></div>
    <label>Trace <select id="traceMode"><option value="dff">dF/F</option><option value="events">Events</option></select></label>
    <label>ROI <input id="roiInput" type="number" min="0" value="0"></label>
    <label>Start s <input id="timeStart" type="number" min="0" step="0.001" value="0"></label>
    <label>End s <input id="timeEnd" type="number" min="0" step="0.001" value="0"></label>
    <label>First ROI <input id="yStart" type="number" min="0" value="0"></label>
    <label>Last ROI <input id="yEnd" type="number" min="0" value="0"></label>
    <button id="reset">Reset zoom</button>
  </div>
  <div class="plots">
    <div class="panel"><div class="title" id="traceTitle">Selected ROI trace</div><canvas id="traceCanvas"></canvas><div class="note">Wheel or drag to zoom/pan time. Double-click to reset.</div></div>
    <div class="panel"><div class="title" id="stackTitle">Stacked ROI traces</div><canvas id="stackCanvas"></canvas><div class="note">Wheel to zoom time. Use First/Last ROI to choose displayed rows. Click a row to select an ROI.</div></div>
  </div>
</div>
<script id="payload" type="application/json">{json.dumps(payload, separators=(",", ":"))}</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("payload").textContent);
document.getElementById("projection").src = data.projection;
document.getElementById("mask").src = data.mask;
document.getElementById("meta").textContent = `${{data.nRois}} ROIs | ${{data.nFrames.toLocaleString()}} frames | ${{data.frameRate.toFixed(3)}} Hz | ${{data.imageWidth}} x ${{data.imageHeight}}`;
document.querySelectorAll(".overlay").forEach(svg => svg.setAttribute("viewBox", `0 0 ${{data.imageWidth}} ${{data.imageHeight}}`));
document.getElementById("roiInput").max = data.nRois - 1;
document.getElementById("yStart").max = data.nRois - 1;
document.getElementById("yEnd").max = data.nRois - 1;
document.getElementById("yEnd").value = Math.min(19, data.nRois - 1);
if (!data.eventsAvailable) document.querySelector('#traceMode option[value="events"]').disabled = true;
const sessionDurationSec = (data.nFrames - 1) / data.frameRate;
document.getElementById("timeStart").max = sessionDurationSec.toFixed(3);
document.getElementById("timeEnd").max = sessionDurationSec.toFixed(3);
document.getElementById("timeEnd").value = sessionDurationSec.toFixed(3);
function b64f32(base64) {{
  if (!base64) return null;
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}
const traces = {{dff: b64f32(data.dff), events: b64f32(data.events)}};
let traceMode = "dff", selected = 0, x0 = 0, x1 = data.nFrames - 1, y0 = 0, y1 = Math.min(19, data.nRois - 1);
function active() {{ return traces[traceMode] || traces.dff; }}
function label() {{ return traceMode === "events" ? "events" : "dF/F"; }}
function fit(canvas) {{
  const r = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * r));
  canvas.height = Math.max(1, Math.round(box.height * r));
}}
function trace(roi) {{ const a = active(); return a.subarray(roi * data.nFrames, (roi + 1) * data.nFrames); }}
function syncTimeInputs() {{
  document.getElementById("timeStart").value = (x0 / data.frameRate).toFixed(3);
  document.getElementById("timeEnd").value = (x1 / data.frameRate).toFixed(3);
}}
function setFrameWindow(startFrame, endFrame) {{
  x0 = Math.max(0, Math.min(data.nFrames - 1, startFrame));
  x1 = Math.max(0, Math.min(data.nFrames - 1, endFrame));
  if (x1 <= x0) x1 = Math.min(data.nFrames - 1, x0 + 1);
  syncTimeInputs();
  draw();
}}
function setTimeWindow(startSec, endSec) {{
  setFrameWindow(Math.round(Number(startSec) * data.frameRate), Math.round(Number(endSec) * data.frameRate));
}}
function setSelected(roi) {{
  selected = Math.max(0, Math.min(data.nRois - 1, Math.round(roi)));
  const r = data.rois[selected] || {{}};
  document.getElementById("roiInput").value = selected;
  document.getElementById("readout").textContent = `Selected ROI ${{selected}} | source ROI ${{r.sourceRoi ?? selected}} | ${{r.npix ?? 0}} px`;
  document.getElementById("traceTitle").textContent = `Selected ROI ${{selected}} ${{label()}}`;
  document.getElementById("stackTitle").textContent = `${{label()}}, stacked ROIs`;
  document.querySelectorAll(".roi").forEach(c => c.classList.toggle("selected", Number(c.dataset.roi) === selected));
  draw();
}}
function makeOverlays() {{
  document.querySelectorAll(".overlay").forEach(svg => {{
    data.rois.forEach(r => {{
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", r.path);
      path.dataset.roi = r.roi;
      path.classList.add("roi");
      path.addEventListener("click", () => setSelected(r.roi));
      svg.appendChild(path);
    }});
  }});
}}
function drawAxes(ctx, w, h, l, t, pw, ph, xLabel, yLabel) {{
  ctx.strokeStyle = "#d0d5dd"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(l,t); ctx.lineTo(l,t+ph); ctx.lineTo(l+pw,t+ph); ctx.stroke();
  ctx.fillStyle = "#475467"; ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"; ctx.fillText(xLabel, l + pw / 2, h - 8);
  ctx.save(); ctx.translate(14, t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
}}
function tickStep(seconds) {{
  const options = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,30,60,120,300,600];
  for (const step of options) if (seconds / step <= 8) return step;
  return 1200;
}}
function timeLabel(seconds, majorStep) {{
  if (majorStep < 1) return `${{Math.round(seconds * 1000)}} ms`;
  if (seconds < 60) return `${{seconds.toFixed(majorStep < 2 ? 1 : 0)}} s`;
  return `${{(seconds / 60).toFixed(1)}} min`;
}}
function drawTimeGrid(ctx, l, t, pw, ph) {{
  const startSec = x0 / data.frameRate, endSec = x1 / data.frameRate, spanSec = Math.max(1e-9, endSec - startSec);
  const major = tickStep(spanSec), minor = major >= 1 ? major / 10 : major / 5;
  function xOfSec(sec) {{ return l + (sec - startSec) / spanSec * pw; }}
  ctx.save(); ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
  if (spanSec <= 20) {{
    ctx.strokeStyle = "#f1f5f9"; ctx.lineWidth = 1;
    for (let sec = Math.ceil(startSec / minor) * minor; sec <= endSec; sec += minor) {{
      const x = xOfSec(sec); ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
    }}
  }}
  ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 1;
  const labelTicks = [];
  for (let sec = Math.ceil(startSec / major) * major; sec <= endSec; sec += major) {{
    const x = xOfSec(sec); ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke(); labelTicks.push([x, timeLabel(sec, major)]);
  }}
  ctx.restore();
  ctx.fillStyle = "#475467"; ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.textBaseline = "top";
  labelTicks.forEach(([x, text]) => ctx.fillText(text, x, t + ph + 8));
}}
function colorForRoi(roi) {{
  const palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#17becf","#bcbd22","#7f7f7f","#2563eb","#dc2626","#059669","#ca8a04","#7c3aed","#0891b2","#be123c","#4d7c0f","#c2410c","#4338ca"];
  return palette[roi % palette.length];
}}
function drawStack() {{
  const canvas = document.getElementById("stackCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 62, r = 16, t = 14, b = 56, pw = w-l-r, ph = h-t-b;
  ctx.clearRect(0,0,w,h); ctx.fillStyle = "#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)","ROI index"); drawTimeGrid(ctx,l,t,pw,ph);
  const ys = Math.max(0, Math.floor(y0)), ye = Math.min(data.nRois - 1, Math.ceil(y1)), xs = Math.max(0, Math.floor(x0)), xe = Math.min(data.nFrames - 1, Math.ceil(x1));
  const count = Math.max(1, ye - ys + 1), rowH = ph / count, pixelCount = Math.max(1, Math.floor(pw));
  let amplitudes = [];
  for (let roi = ys; roi <= ye; roi++) {{
    const tr = trace(roi); let sum = 0, n = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ sum += v; n++; }} }}
    const center = n ? sum / n : 0; let dev = 0, dn = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ dev += Math.abs(v - center); dn++; }} }}
    amplitudes.push(Math.max(dn ? dev / dn : 0, traceMode === "events" ? 0.02 : 0.2));
  }}
  const sortedAmp = amplitudes.slice().sort((a,b) => a-b), typicalAmp = Math.max(sortedAmp[Math.floor(sortedAmp.length / 2)] || 1, traceMode === "events" ? 0.02 : 0.2);
  const scale = (rowH * 0.24) / typicalAmp;
  ctx.save(); ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
  for (let roi = ys; roi <= ye; roi++) {{
    const tr = trace(roi), row = roi - ys, baseline = t + rowH * (row + 0.5), color = colorForRoi(roi);
    ctx.strokeStyle = roi === selected ? "#111827" : color; ctx.lineWidth = roi === selected ? Math.max(1.8, 1.8 * (window.devicePixelRatio || 1)) : Math.max(0.8, window.devicePixelRatio || 1);
    let sum = 0, n = 0; for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ sum += v; n++; }} }}
    const center = n ? sum / n : 0, framesPerPixel = (xe - xs + 1) / pixelCount;
    ctx.beginPath();
    if (framesPerPixel <= 1.2) {{
      let first = true;
      for (let f = xs; f <= xe; f++) {{ const x = l + (f - x0) / (x1 - x0) * pw, y = baseline - (tr[f] - center) * scale; if (first) {{ ctx.moveTo(x, y); first = false; }} else ctx.lineTo(x, y); }}
    }} else {{
      for (let px = 0; px < pixelCount; px++) {{
        const f0 = Math.max(xs, Math.floor(xs + px * framesPerPixel)), f1 = Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel));
        let minV = Infinity, maxV = -Infinity;
        for (let f = f0; f <= f1; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ minV = Math.min(minV, v); maxV = Math.max(maxV, v); }} }}
        if (!Number.isFinite(minV)) continue; const x = l + px; ctx.moveTo(x, baseline - (minV - center) * scale); ctx.lineTo(x, baseline - (maxV - center) * scale);
      }}
    }}
    ctx.stroke();
    if (count <= 80) {{ ctx.fillStyle = color; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(roi), l - 8, baseline); }}
  }}
  ctx.restore();
}}
function drawTrace() {{
  const canvas = document.getElementById("traceCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w=canvas.width,h=canvas.height,l=62,r=16,t=14,b=56,pw=w-l-r,ph=h-t-b,tr=trace(selected);
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)",label()); drawTimeGrid(ctx,l,t,pw,ph);
  const xs=Math.max(0,Math.floor(x0)), xe=Math.min(data.nFrames-1,Math.ceil(x1)); let lo=Infinity,hi=-Infinity;
  for (let f=xs; f<=xe; f++) {{ const v=tr[f]; if (Number.isFinite(v)) {{ lo=Math.min(lo,v); hi=Math.max(hi,v); }} }}
  if (!Number.isFinite(lo) || hi<=lo) {{ lo=-1; hi=1; }} const pad=(hi-lo)*.08||1; lo-=pad; hi+=pad;
  const yOf=v=>t+(1-(v-lo)/(hi-lo))*ph, pixelCount=Math.max(1,Math.floor(pw)), framesPerPixel=(xe-xs+1)/pixelCount;
  ctx.strokeStyle="#1d4ed8"; ctx.lineWidth=Math.max(1,window.devicePixelRatio||1); ctx.beginPath();
  if (framesPerPixel <= 1.2) {{
    let first=true; for (let f=xs; f<=xe; f++) {{ const x=l+(f-x0)/(x1-x0)*pw, y=yOf(tr[f]); if (first) {{ ctx.moveTo(x,y); first=false; }} else ctx.lineTo(x,y); }}
  }} else {{
    for (let px=0; px<pixelCount; px++) {{
      const f0=Math.max(xs, Math.floor(xs + px * framesPerPixel)), f1=Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel)); let minV=Infinity, maxV=-Infinity;
      for (let f=f0; f<=f1; f++) {{ const v=tr[f]; if (Number.isFinite(v)) {{ minV=Math.min(minV,v); maxV=Math.max(maxV,v); }} }}
      if (!Number.isFinite(minV)) continue; const x=l+px; ctx.moveTo(x, yOf(minV)); ctx.lineTo(x, yOf(maxV));
    }}
  }}
  ctx.stroke();
  ctx.fillStyle="#475467"; ctx.textAlign="right"; ctx.textBaseline="middle"; for (let i=0; i<=4; i++) {{ const v=lo+i/4*(hi-lo); ctx.fillText(v.toFixed(2), l-8, yOf(v)); }}
}}
function draw() {{ drawTrace(); drawStack(); }}
function reset() {{ x0=0; x1=data.nFrames-1; y0=0; y1=Math.min(19, data.nRois-1); document.getElementById("yStart").value=0; document.getElementById("yEnd").value=y1; syncTimeInputs(); draw(); }}
document.getElementById("roiInput").addEventListener("change", e => setSelected(Number(e.target.value)));
document.getElementById("traceMode").addEventListener("change", e => {{ traceMode = e.target.value; setSelected(selected); }});
document.getElementById("timeStart").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("timeEnd").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("yStart").addEventListener("change", e => {{ y0=Number(e.target.value); draw(); }});
document.getElementById("yEnd").addEventListener("change", e => {{ y1=Number(e.target.value); draw(); }});
document.getElementById("reset").addEventListener("click", reset);
document.getElementById("stackCanvas").addEventListener("click", e => {{ const rect=e.target.getBoundingClientRect(), frac=(e.clientY-rect.top)/rect.height; setSelected(y0 + frac * (y1-y0+1)); }});
for (const id of ["stackCanvas", "traceCanvas"]) document.getElementById(id).addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
let dragging=false, sx=0, start0=0, start1=0;
document.getElementById("traceCanvas").addEventListener("mousedown", e => {{ dragging=true; sx=e.clientX; start0=x0; start1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!dragging) return; const rect=document.getElementById("traceCanvas").getBoundingClientRect(), shift=-(e.clientX-sx)/rect.width*(start1-start0); setFrameWindow(start0+shift, start1+shift); }});
window.addEventListener("mouseup", () => {{ dragging=false; document.getElementById("traceCanvas").classList.remove("dragging"); }});
document.getElementById("traceCanvas").addEventListener("dblclick", reset);
window.addEventListener("resize", draw);
makeOverlays(); syncTimeInputs(); setSelected(0);
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--plane", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    out = create_plane_html(
        args.session_dir.expanduser().resolve(),
        args.plane,
        args.output.expanduser().resolve(),
        max_frames=args.max_frames,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
