#!/usr/bin/env python3
"""Create a standalone interactive HTML viewer containing multiple mesoscope planes."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from create_mesoscope_plane_html import (
    _build_roi_payload,
    _float32_b64,
    _green_png_data_uri,
    _load_plane_arrays,
    _mask_png_data_uri,
    _safe_name,
)


def _available_planes(session_dir: Path) -> list[str]:
    planes = []
    for path in sorted(session_dir.glob("*_pixel_masks.npz")):
        plane = path.name[: -len("_pixel_masks.npz")]
        if (session_dir / plane / "projection.npy").exists() and (session_dir / plane / "dff.npy").exists():
            planes.append(plane)
    return planes


def _make_plane_payload(session_dir: Path, plane: str, max_frames: int | None) -> dict:
    arrays = _load_plane_arrays(session_dir, plane, max_frames=max_frames)
    projection = arrays["projection"]
    shape = tuple(arrays["shape"])
    if shape != tuple(projection.shape[:2]):
        image_height, image_width = shape
    else:
        image_height, image_width = projection.shape[:2]
    rois, label_mask = _build_roi_payload(arrays["roi_indices"], arrays["pixel_masks"], shape)
    dff = np.asarray(arrays["dff"], dtype=np.float32)
    events = None if arrays["events"] is None else np.asarray(arrays["events"], dtype=np.float32)
    n_rois, n_frames = dff.shape
    return {
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


def create_session_html(
    session_dir: Path,
    output_path: Path,
    planes: list[str] | None = None,
    max_frames: int | None = None,
) -> Path:
    plane_names = planes or _available_planes(session_dir)
    if not plane_names:
        raise ValueError(f"No materialized planes found in {session_dir}")
    payload = {
        "session": session_dir.name,
        "planes": plane_names,
        "planeData": {},
    }
    for plane in plane_names:
        print(f"[INFO] embedding {plane}", flush=True)
        payload["planeData"][plane] = _make_plane_payload(session_dir, plane, max_frames=max_frames)

    plane_options = "\n".join(f'<option value="{plane}">{plane}</option>' for plane in plane_names)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{session_dir.name} ROI viewer</title>
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
.controls {{ display: grid; grid-template-columns: 1fr auto auto auto auto auto auto auto; gap: 9px; align-items: center; margin-top: 10px; }}
button, input, select {{ font: inherit; }}
button {{ border: 1px solid #d0d5dd; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
input, select {{ border: 1px solid #d0d5dd; border-radius: 6px; padding: 7px 8px; width: 86px; box-sizing: border-box; }}
select {{ width: 112px; }}
#planeSelect {{ width: 120px; }}
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
  <div class="head"><h1>{session_dir.name} | <span id="planeTitle"></span></h1><div class="meta" id="meta"></div></div>
  <div class="grid">
    <div class="panel"><div class="title">Functional projection</div><div class="imagewrap"><img id="projection"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
    <div class="panel"><div class="title">ROI masks</div><div class="imagewrap"><img id="mask"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
  </div>
  <div class="panel controls">
    <div id="readout"></div>
    <label>Plane <select id="planeSelect">{plane_options}</select></label>
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
const payload = JSON.parse(document.getElementById("payload").textContent);
const decoded = {{}};
let data = null, plane = payload.planes[0], traceMode = "dff", selected = 0, x0 = 0, x1 = 1, y0 = 0, y1 = 0;
function b64f32(base64) {{
  if (!base64) return null;
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}
function getPlane(name) {{
  const p = payload.planeData[name];
  if (!decoded[name]) decoded[name] = {{dff: b64f32(p.dff), events: b64f32(p.events)}};
  return p;
}}
function active() {{ return (decoded[plane] && decoded[plane][traceMode]) || decoded[plane].dff; }}
function label() {{ return traceMode === "events" ? "events" : "dF/F"; }}
function sessionDurationSec() {{ return (data.nFrames - 1) / data.frameRate; }}
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
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${{data.imageWidth}} ${{data.imageHeight}}`);
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
function loadPlane(name) {{
  plane = name;
  data = getPlane(name);
  document.getElementById("planeTitle").textContent = name;
  document.getElementById("projection").src = data.projection;
  document.getElementById("mask").src = data.mask;
  document.getElementById("meta").textContent = `${{data.nRois}} ROIs | ${{data.nFrames.toLocaleString()}} frames | ${{data.frameRate.toFixed(3)}} Hz | ${{data.imageWidth}} x ${{data.imageHeight}}`;
  document.getElementById("roiInput").max = data.nRois - 1;
  document.getElementById("yStart").max = data.nRois - 1;
  document.getElementById("yEnd").max = data.nRois - 1;
  document.getElementById("timeStart").max = sessionDurationSec().toFixed(3);
  document.getElementById("timeEnd").max = sessionDurationSec().toFixed(3);
  document.querySelector('#traceMode option[value="events"]').disabled = !data.eventsAvailable;
  if (traceMode === "events" && !data.eventsAvailable) {{
    traceMode = "dff";
    document.getElementById("traceMode").value = "dff";
  }}
  selected = 0; x0 = 0; x1 = data.nFrames - 1; y0 = 0; y1 = Math.min(19, data.nRois - 1);
  document.getElementById("yStart").value = y0;
  document.getElementById("yEnd").value = y1;
  document.getElementById("timeEnd").value = sessionDurationSec().toFixed(3);
  makeOverlays();
  syncTimeInputs();
  setSelected(0);
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
  if (!data) return;
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
  if (!data) return;
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
document.getElementById("planeSelect").addEventListener("change", e => loadPlane(e.target.value));
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
loadPlane(plane);
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planes", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    planes = args.planes
    if planes and len(planes) == 1 and "," in planes[0]:
        planes = [p.strip() for p in planes[0].split(",") if p.strip()]
    out = create_session_html(
        args.session_dir.expanduser().resolve(),
        args.output.expanduser().resolve(),
        planes=planes,
        max_frames=args.max_frames,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
