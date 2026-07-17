#!/usr/bin/env python3
"""Create a standalone session-level ROI metric/transient review HTML."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from create_mesoscope_plane_html import (
    _build_roi_payload,
    _float32_b64,
    _green_png_data_uri,
    _load_plane_arrays,
    _mask_png_data_uri,
)


DEFAULT_EVENT_COLUMNS = [
    "event_exp_gauss_fit_score",
    "event_triggered_dff_snr",
    "event_onset_rate_hz",
    "calcium_kernel_peak_dff",
    "calcium_kernel_decay_r2",
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
    preferred = [
        "event_triggered_dff_snr",
        "event_exp_gauss_fit_score",
        "calcium_kernel_peak_dff",
        "calcium_kernel_decay_r2",
        "robust_event_snr",
        "exceptional_event_score",
        "snr_pair__signal_p99_p50__noise_lower_mode_rms",
        "snr_pair__signal_p99_p50__noise_lower_mode_mad",
        "snr_pair__signal_p95_p50__noise_lower_mode_rms",
    ]
    numeric = []
    for col in metrics.columns:
        vals = pd.to_numeric(metrics[col], errors="coerce")
        if vals.notna().sum() >= 3:
            numeric.append(col)
    out = []
    for col in preferred + [c for c in numeric if c.startswith("snr_pair__")] + numeric:
        if col in numeric and col not in out:
            out.append(col)
    return out


def _plane_payload(session_dir: Path, plane: str, metrics: pd.DataFrame, max_frames: int | None) -> dict:
    arrays = _load_plane_arrays(session_dir, plane, max_frames=max_frames)
    projection = arrays["projection"]
    shape = tuple(arrays["shape"])
    image_height, image_width = shape
    rois, label_mask = _build_roi_payload(arrays["roi_indices"], arrays["pixel_masks"], shape)
    dff = np.asarray(arrays["dff"], dtype=np.float32)
    events = None if arrays["events"] is None else np.asarray(arrays["events"], dtype=np.float32)
    n_rois, n_frames = dff.shape

    plane_metrics = metrics.loc[metrics["plane"].astype(str) == str(plane)].copy()
    plane_metrics["roi_index"] = pd.to_numeric(plane_metrics["roi_index"], errors="coerce").astype("Int64")
    plane_metrics = plane_metrics.sort_values("roi_index")
    metric_rows = []
    for _, row in plane_metrics.iterrows():
        if pd.isna(row["roi_index"]):
            continue
        roi = int(row["roi_index"])
        if roi < 0 or roi >= n_rois:
            continue
        metric_rows.append({col: _json_safe(row[col]) for col in plane_metrics.columns})

    return {
        "frameRate": float(arrays["frame_rate"]),
        "nRois": int(n_rois),
        "nFrames": int(n_frames),
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "projection": _green_png_data_uri(projection),
        "mask": _mask_png_data_uri(label_mask),
        "rois": rois[:n_rois],
        "metrics": metric_rows,
        "dff": _float32_b64(dff),
        "events": _float32_b64(events) if events is not None else None,
        "eventsAvailable": events is not None,
    }


def create_metric_review_html(
    session_dir: Path,
    metrics_csv: Path,
    output_path: Path,
    planes: list[str] | None = None,
    max_frames: int | None = None,
) -> Path:
    metrics = pd.read_csv(metrics_csv)
    if "plane" not in metrics or "roi_index" not in metrics:
        raise ValueError("metrics CSV must include plane and roi_index columns")
    plane_names = planes or sorted(metrics["plane"].dropna().astype(str).unique())
    plane_names = [plane for plane in plane_names if (session_dir / plane / "dff.npy").exists()]
    if not plane_names:
        raise ValueError(f"No planes with dff.npy found in {session_dir}")

    payload = {
        "session": session_dir.name,
        "metricOptions": _metric_options(metrics),
        "eventColumns": [c for c in DEFAULT_EVENT_COLUMNS if c in metrics.columns],
        "planes": plane_names,
        "planeData": {},
    }
    for plane in plane_names:
        print(f"[INFO] embedding {plane}", flush=True)
        payload["planeData"][plane] = _plane_payload(session_dir, plane, metrics, max_frames=max_frames)

    plane_options = "\n".join(f'<option value="{p}">{p}</option>' for p in plane_names)
    metric_options = "\n".join(f'<option value="{m}">{m}</option>' for m in payload["metricOptions"])
    html = HTML_TEMPLATE.replace("__TITLE__", f"{session_dir.name} metric review")
    html = html.replace("__PLANE_OPTIONS__", plane_options)
    html = html.replace("__METRIC_OPTIONS__", metric_options)
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
body { margin:0; font-family:Arial, Helvetica, sans-serif; background:#f6f7f8; color:#202124; }
.page { width:min(1760px, calc(100vw - 28px)); margin:14px auto 28px; }
.head { display:flex; justify-content:space-between; gap:12px; align-items:end; margin-bottom:10px; }
h1 { margin:0; font-size:21px; }
.meta { color:#667085; font-size:13px; text-align:right; }
.controls { display:grid; grid-template-columns:repeat(8, auto) 1fr; gap:9px; align-items:end; background:#fff; border:1px solid #d0d5dd; border-radius:7px; padding:10px; margin-bottom:10px; }
label { font-size:12px; color:#475467; display:grid; gap:3px; }
select, input, button { font:inherit; border:1px solid #d0d5dd; border-radius:6px; padding:7px 8px; background:#fff; box-sizing:border-box; }
select { width:220px; } #planeSelect { width:120px; } input { width:92px; }
button { cursor:pointer; white-space:nowrap; }
.grid { display:grid; grid-template-columns:340px 1fr; gap:10px; }
.panel { background:#fff; border:1px solid #d0d5dd; border-radius:7px; padding:10px; box-sizing:border-box; }
.title { font-size:14px; font-weight:700; margin-bottom:8px; }
.imagewrap { position:relative; width:100%; aspect-ratio:1/1; background:#111; overflow:hidden; }
.imagewrap img, .imagewrap svg { position:absolute; inset:0; width:100%; height:100%; }
.imagewrap img { object-fit:contain; image-rendering:pixelated; }
.roi { fill:transparent; stroke:rgba(255,255,255,.85); stroke-width:.65; cursor:pointer; vector-effect:non-scaling-stroke; pointer-events:all; }
.roi:hover { fill:rgba(6,182,212,.22); stroke:#06b6d4; stroke-width:1.5; }
.roi.selected { fill:rgba(220,38,38,.24); stroke:#dc2626; stroke-width:1.9; }
.metricTable { width:100%; border-collapse:collapse; font-size:12px; }
.metricTable th, .metricTable td { border-bottom:1px solid #e5e7eb; padding:4px 5px; text-align:right; }
.metricTable th:first-child, .metricTable td:first-child { text-align:left; }
canvas { width:100%; display:block; background:#fff; border:1px solid #d0d5dd; box-sizing:border-box; }
#histCanvas { height:210px; }
#binCanvas { height:500px; }
#eventCanvas { height:240px; }
#roiCanvas { height:240px; cursor:grab; }
.rightGrid { display:grid; grid-template-columns:1fr; gap:10px; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.note { color:#667085; font-size:12px; margin-top:5px; }
@media (max-width:1100px) { .grid, .two, .controls { grid-template-columns:1fr; } .head { display:block; } .meta { text-align:left; } }
</style>
</head>
<body>
<div class="page">
  <div class="head"><h1>__TITLE__</h1><div class="meta" id="meta"></div></div>
  <div class="controls">
    <label>Plane<select id="planeSelect">__PLANE_OPTIONS__</select></label>
    <label>Metric<select id="metricSelect">__METRIC_OPTIONS__</select></label>
    <label>Low cut<input id="lowCut" type="number" step="any"></label>
    <label>High cut<input id="highCut" type="number" step="any"></label>
    <label>Start s<input id="timeStart" type="number" min="0" step="0.001" value="0"></label>
    <label>End s<input id="timeEnd" type="number" min="0" step="0.001" value="0"></label>
    <label>Traces/bin<input id="tracesPerBin" type="number" min="1" max="80" value="20"></label>
    <label>ROI<input id="roiInput" type="number" min="0" value="0"></label>
    <div><button id="optimize">Optimize high bin by event fit</button> <button id="reset">Reset</button></div>
  </div>
  <div class="grid">
    <div>
      <div class="panel"><div class="title">Functional projection</div><div class="imagewrap"><img id="projection"><svg id="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
      <div class="panel" style="margin-top:10px;"><div class="title">Selected ROI metrics</div><div id="roiMetrics"></div></div>
    </div>
    <div class="rightGrid">
      <div class="two">
        <div class="panel"><div class="title">Metric distribution and bin boundaries</div><canvas id="histCanvas"></canvas><div class="note" id="binReadout"></div></div>
        <div class="panel"><div class="title">Event-score summary by bin</div><div id="binSummary"></div></div>
      </div>
      <div class="panel"><div class="title">Low / medium / high ROI dF/F traces</div><canvas id="binCanvas"></canvas><div class="note">Traces are median-centered. Adjust Start/End seconds to zoom into time.</div></div>
      <div class="two">
        <div class="panel"><div class="title">Average event-triggered dF/F by bin</div><canvas id="eventCanvas"></canvas></div>
        <div class="panel"><div class="title">Selected ROI dF/F with detected events</div><canvas id="roiCanvas"></canvas><div class="note">Click ROI masks or change ROI input.</div></div>
      </div>
    </div>
  </div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const payload = JSON.parse(document.getElementById("payload").textContent);
const decoded = {};
const binLabels = ["low","medium","high"];
const binColors = {low:"#4575b4", medium:"#6a994e", high:"#d73027"};
let plane = payload.planes[0], data = null, metric = payload.metricOptions[0], selected = 0;
let lowCut = 0, highCut = 1, timeStart = 0, timeEnd = 1;
function b64f32(base64) {
  const binary = atob(base64), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
function getPlane(name) {
  const p = payload.planeData[name];
  if (!decoded[name]) decoded[name] = {dff:b64f32(p.dff), events:p.events ? b64f32(p.events) : null};
  return p;
}
function arrays() { return decoded[plane]; }
function fit(canvas) {
  const r = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width*r)); canvas.height = Math.max(1, Math.round(box.height*r));
}
function metricValue(row, name) {
  const v = row ? row[name] : null;
  return typeof v === "number" && Number.isFinite(v) ? v : NaN;
}
function rowsWithMetric() { return data.metrics.filter(r => Number.isFinite(metricValue(r, metric))); }
function duration() { return (data.nFrames - 1) / data.frameRate; }
function frameRange() {
  const a = Math.max(0, Math.min(data.nFrames-1, Math.round(timeStart*data.frameRate)));
  const b = Math.max(a+1, Math.min(data.nFrames, Math.round(timeEnd*data.frameRate)));
  return [a,b];
}
function traceFor(roi) { return arrays().dff.subarray(roi*data.nFrames, (roi+1)*data.nFrames); }
function eventsFor(roi) { return arrays().events ? arrays().events.subarray(roi*data.nFrames, (roi+1)*data.nFrames) : null; }
function assignBin(row) {
  const v = metricValue(row, metric);
  if (!Number.isFinite(v)) return null;
  if (v <= lowCut) return "low";
  if (v >= highCut) return "high";
  return "medium";
}
function currentBins() {
  const out = {low:[], medium:[], high:[]};
  for (const row of rowsWithMetric()) out[assignBin(row)].push(row);
  return out;
}
function setDefaultCuts() {
  const vals = rowsWithMetric().map(r => metricValue(r, metric)).sort((a,b)=>a-b);
  if (!vals.length) { lowCut=0; highCut=1; return; }
  lowCut = vals[Math.floor(vals.length/3)];
  highCut = vals[Math.floor(vals.length*2/3)];
  document.getElementById("lowCut").value = Number(lowCut.toPrecision(6));
  document.getElementById("highCut").value = Number(highCut.toPrecision(6));
}
function loadPlane(name) {
  plane = name; data = getPlane(name); metric = document.getElementById("metricSelect").value;
  document.getElementById("projection").src = data.projection;
  document.getElementById("meta").textContent = `${data.nRois} ROIs | ${data.nFrames.toLocaleString()} frames | ${data.frameRate.toFixed(3)} Hz`;
  document.getElementById("timeEnd").value = Math.min(180, duration()).toFixed(3);
  document.getElementById("timeStart").value = "0";
  timeStart = 0; timeEnd = Math.min(180, duration());
  selected = 0; document.getElementById("roiInput").max = data.nRois-1; document.getElementById("roiInput").value = selected;
  makeOverlay(); setDefaultCuts(); redraw();
}
function makeOverlay() {
  const svg = document.getElementById("overlay"); svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${data.imageWidth} ${data.imageHeight}`);
  for (const r of data.rois) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", r.path); path.dataset.roi = r.roi; path.classList.add("roi");
    path.addEventListener("click", () => { selected = r.roi; document.getElementById("roiInput").value = selected; redraw(); });
    svg.appendChild(path);
  }
}
function canvasContext(id) { const c=document.getElementById(id); fit(c); return [c, c.getContext("2d")]; }
function clear(ctx, c) { ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle="#fff"; ctx.fillRect(0,0,c.width,c.height); }
function drawHist() {
  const [c,ctx]=canvasContext("histCanvas"); clear(ctx,c);
  const vals = rowsWithMetric().map(r => metricValue(r, metric));
  if (!vals.length) return;
  const min = Math.min(...vals), max = Math.max(...vals), pad = (max-min || 1)*0.05;
  const xMin=min-pad, xMax=max+pad, bins=60, counts=Array(bins).fill(0);
  for (const v of vals) counts[Math.max(0, Math.min(bins-1, Math.floor((v-xMin)/(xMax-xMin)*bins)))]++;
  const l=55,t=20,w=c.width-75,h=c.height-55, ymax=Math.max(...counts,1);
  ctx.fillStyle="#94a3b8";
  counts.forEach((n,i)=>{ const x=l+i*w/bins, bh=n/ymax*h; ctx.fillRect(x,t+h-bh,w/bins-1,bh); });
  for (const [cut,col] of [[lowCut,binColors.low],[highCut,binColors.high]]) {
    const x=l+(cut-xMin)/(xMax-xMin)*w; ctx.strokeStyle=col; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(x,t); ctx.lineTo(x,t+h); ctx.stroke();
  }
  ctx.strokeStyle="#d0d5dd"; ctx.strokeRect(l,t,w,h);
  ctx.fillStyle="#475467"; ctx.font="12px Arial"; ctx.textAlign="center"; ctx.fillText(metric, l+w/2, c.height-10);
}
function median(values) {
  const v = values.filter(Number.isFinite).sort((a,b)=>a-b);
  if (!v.length) return NaN;
  const mid = Math.floor(v.length/2);
  return v.length%2 ? v[mid] : (v[mid-1]+v[mid])/2;
}
function drawSummary() {
  const bins=currentBins();
  const cols = payload.eventColumns;
  let html = '<table class="metricTable"><thead><tr><th>bin</th><th>n</th>' + cols.map(c=>`<th>${c}</th>`).join("") + '</tr></thead><tbody>';
  for (const b of binLabels) {
    html += `<tr><td>${b}</td><td>${bins[b].length}</td>`;
    for (const col of cols) html += `<td>${fmt(median(bins[b].map(r=>metricValue(r,col))))}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById("binSummary").innerHTML = html;
  document.getElementById("binReadout").textContent = binLabels.map(b=>`${b}: ${bins[b].length}`).join(" | ");
}
function fmt(v) { return Number.isFinite(v) ? v.toPrecision(4) : ""; }
function drawBinTraces() {
  const [c,ctx]=canvasContext("binCanvas"); clear(ctx,c);
  const bins=currentBins(), [f0,f1]=frameRange(), nShow=Math.max(1, Number(document.getElementById("tracesPerBin").value)||20);
  const colW=c.width/3, top=28, bottom=28, h=c.height-top-bottom;
  for (let bi=0; bi<3; bi++) {
    const b=binLabels[bi], rows=bins[b], xBase=bi*colW, sample=rows.slice().sort((a,b)=>metricValue(a,metric)-metricValue(b,metric)).filter((_,i)=> i % Math.max(1, Math.floor(rows.length/nShow)) === 0).slice(0,nShow);
    ctx.fillStyle=binColors[b]; ctx.font="13px Arial"; ctx.textAlign="left"; ctx.fillText(`${b} n=${rows.length}`, xBase+8, 18);
    let all=[]; for (const row of sample) { const tr=traceFor(row.roi_index).subarray(f0,f1); const med=median(Array.from(tr)); for (const y of tr) if (Number.isFinite(y)) all.push(y-med); }
    const lo = percentile(all,1), hi = percentile(all,99), span=(hi-lo)||1;
    for (const row of sample) {
      const tr=traceFor(row.roi_index).subarray(f0,f1), med=median(Array.from(tr));
      ctx.strokeStyle="rgba(0,0,0,.32)"; ctx.lineWidth=1; ctx.beginPath();
      for (let i=0;i<tr.length;i++) {
        const x=xBase+8+i/(tr.length-1)*(colW-16), y=top+h-(tr[i]-med-lo)/span*h;
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }
  }
}
function percentile(arr, p) { const v=arr.filter(Number.isFinite).sort((a,b)=>a-b); if(!v.length) return 0; return v[Math.max(0,Math.min(v.length-1,Math.floor((p/100)*(v.length-1))))]; }
function eventTriggered(row) {
  const ev=eventsFor(row.roi_index), tr=traceFor(row.roi_index); if (!ev) return null;
  const pre=Math.max(1,Math.round(0.5*data.frameRate)), post=Math.max(2,Math.round(2*data.frameRate)), wins=[];
  for (let i=1;i<ev.length-post;i++) if (ev[i]>0 && !(ev[i-1]>0) && i>=pre) {
    const w=Array.from(tr.subarray(i-pre,i+post+1)); const base=median(w.slice(0,pre)); wins.push(w.map(x=>x-base)); if (wins.length>=120) break;
  }
  if (!wins.length) return null;
  return wins[0].map((_,i)=>median(wins.map(w=>w[i])));
}
function drawEventAvg() {
  const [c,ctx]=canvasContext("eventCanvas"); clear(ctx,c);
  const bins=currentBins(), pre=Math.max(1,Math.round(0.5*data.frameRate)), post=Math.max(2,Math.round(2*data.frameRate));
  const traces={};
  for (const b of binLabels) {
    traces[b]=[];
    for (const row of bins[b].slice(0,80)) { const e=eventTriggered(row); if(e) traces[b].push(e); }
  }
  let all=[]; for (const b of binLabels) for (const tr of traces[b]) all.push(...tr);
  const lo=percentile(all,1), hi=percentile(all,99), span=(hi-lo)||1, l=45,t=15,w=c.width-65,h=c.height-45;
  for (const b of binLabels) {
    if (!traces[b].length) continue;
    const avg=traces[b][0].map((_,i)=>median(traces[b].map(t=>t[i])));
    ctx.strokeStyle=binColors[b]; ctx.lineWidth=2; ctx.beginPath();
    for (let i=0;i<avg.length;i++) { const x=l+i/(avg.length-1)*w, y=t+h-(avg[i]-lo)/span*h; if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y); }
    ctx.stroke(); ctx.fillStyle=binColors[b]; ctx.fillText(`${b} n=${traces[b].length}`, l+8, t+16+16*binLabels.indexOf(b));
  }
  const zx=l+pre/(pre+post)*w; ctx.strokeStyle="#64748b"; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(zx,t); ctx.lineTo(zx,t+h); ctx.stroke(); ctx.setLineDash([]);
}
function drawRoi() {
  const [c,ctx]=canvasContext("roiCanvas"); clear(ctx,c);
  const [f0,f1]=frameRange(), tr=traceFor(selected).subarray(f0,f1), ev=eventsFor(selected)?.subarray(f0,f1);
  const med=median(Array.from(tr)), yvals=Array.from(tr).map(v=>v-med), lo=percentile(yvals,1), hi=percentile(yvals,99), span=(hi-lo)||1;
  const l=45,t=15,w=c.width-65,h=c.height-45;
  ctx.strokeStyle="#111827"; ctx.lineWidth=1.5; ctx.beginPath();
  for (let i=0;i<tr.length;i++) { const x=l+i/(tr.length-1)*w, y=t+h-(tr[i]-med-lo)/span*h; if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y); }
  ctx.stroke();
  if (ev) { ctx.strokeStyle="#ef4444"; ctx.lineWidth=1; for (let i=1;i<ev.length;i++) if(ev[i]>0 && !(ev[i-1]>0)) { const x=l+i/(ev.length-1)*w; ctx.beginPath(); ctx.moveTo(x,t); ctx.lineTo(x,t+h); ctx.stroke(); } }
}
function drawRoiTable() {
  const row=data.metrics.find(r=>r.roi_index===selected) || {};
  document.querySelectorAll(".roi").forEach(p => p.classList.toggle("selected", Number(p.dataset.roi)===selected));
  const cols=[metric, ...payload.eventColumns, "robust_event_snr", "exceptional_event_score"].filter((v,i,a)=>v in row && a.indexOf(v)===i);
  let html='<table class="metricTable"><tbody>';
  for (const col of cols) html += `<tr><td>${col}</td><td>${fmt(metricValue(row,col))}</td></tr>`;
  html += '</tbody></table>'; document.getElementById("roiMetrics").innerHTML=html;
}
function redraw() { drawHist(); drawSummary(); drawBinTraces(); drawEventAvg(); drawRoi(); drawRoiTable(); }
function optimizeHighBin() {
  const rows=rowsWithMetric().filter(r=>Number.isFinite(metricValue(r,"event_exp_gauss_fit_score"))).sort((a,b)=>metricValue(a,metric)-metricValue(b,metric));
  if (rows.length < 30) return;
  let best=null, minN=Math.max(5, Math.ceil(rows.length*0.1));
  for (let qi=50; qi<=95; qi++) {
    const cut=metricValue(rows[Math.floor(rows.length*qi/100)], metric);
    const high=rows.filter(r=>metricValue(r,metric)>=cut), rest=rows.filter(r=>metricValue(r,metric)<cut);
    if (high.length<minN || rest.length<minN) continue;
    const gain=median(high.map(r=>metricValue(r,"event_exp_gauss_fit_score"))) - median(rest.map(r=>metricValue(r,"event_exp_gauss_fit_score")));
    if (!best || gain>best.gain) best={cut,gain};
  }
  if (best) { highCut=best.cut; const below=rows.filter(r=>metricValue(r,metric)<highCut).map(r=>metricValue(r,metric)).sort((a,b)=>a-b); lowCut=below[Math.floor(below.length/2)] ?? lowCut; document.getElementById("lowCut").value=Number(lowCut.toPrecision(6)); document.getElementById("highCut").value=Number(highCut.toPrecision(6)); redraw(); }
}
document.getElementById("planeSelect").addEventListener("change", e=>loadPlane(e.target.value));
document.getElementById("metricSelect").addEventListener("change", e=>{ metric=e.target.value; setDefaultCuts(); redraw(); });
document.getElementById("lowCut").addEventListener("change", e=>{ lowCut=Number(e.target.value); redraw(); });
document.getElementById("highCut").addEventListener("change", e=>{ highCut=Number(e.target.value); redraw(); });
document.getElementById("timeStart").addEventListener("change", e=>{ timeStart=Number(e.target.value); redraw(); });
document.getElementById("timeEnd").addEventListener("change", e=>{ timeEnd=Number(e.target.value); redraw(); });
document.getElementById("tracesPerBin").addEventListener("change", redraw);
document.getElementById("roiInput").addEventListener("change", e=>{ selected=Math.max(0,Math.min(data.nRois-1,Math.round(Number(e.target.value)))); redraw(); });
document.getElementById("optimize").addEventListener("click", optimizeHighBin);
document.getElementById("reset").addEventListener("click", ()=>loadPlane(plane));
loadPlane(plane);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plane", action="append", dest="planes")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    out = create_metric_review_html(
        args.session_dir.expanduser().resolve(),
        args.metrics_csv.expanduser().resolve(),
        args.output.expanduser().resolve(),
        planes=args.planes,
        max_frames=args.max_frames,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
