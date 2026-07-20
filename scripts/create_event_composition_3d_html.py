#!/usr/bin/env python3
"""Create an interactive 3D ROI event-composition HTML viewer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_mesoscope_plane_html import _float32_b64, _load_plane_arrays  # noqa: E402


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


def _plane_payload(session_dir: Path, plane: str, max_frames: int | None) -> dict:
    arrays = _load_plane_arrays(session_dir, plane, max_frames=max_frames)
    dff = np.asarray(arrays["dff"], dtype=np.float32)
    events = None if arrays["events"] is None else np.asarray(arrays["events"], dtype=np.float32)
    return {
        "frameRate": float(arrays["frame_rate"]),
        "nRois": int(dff.shape[0]),
        "nFrames": int(dff.shape[1]),
        "dff": _float32_b64(dff),
        "events": _float32_b64(events) if events is not None else None,
    }


def create_event_composition_3d_html(
    session_dir: Path,
    composition_csv: Path,
    metrics_csv: Path,
    output_path: Path,
    *,
    planes: list[str] | None = None,
    max_frames: int | None = None,
) -> Path:
    composition = pd.read_csv(composition_csv)
    metrics = pd.read_csv(metrics_csv)
    rows = composition.merge(
        metrics,
        on=["session", "plane", "roi_index"],
        how="left",
        suffixes=("", "_metric"),
    )
    plane_names = planes or sorted(rows["plane"].dropna().astype(str).unique())
    plane_names = [plane for plane in plane_names if (session_dir / plane / "dff.npy").exists()]
    rows = rows.loc[rows["plane"].astype(str).isin(plane_names)].copy()

    point_cols = [
        "session",
        "plane",
        "roi_index",
        "nonlong_event_fraction_lt_2sd",
        "nonlong_event_fraction_2_4sd",
        "nonlong_event_fraction_gt_4sd",
        "long_gt_3s_fraction_all_clusters",
        "has_long_gt_3s_event_cluster",
        "nonlong_event_composition_cluster",
        "n_event_clusters_total",
        "n_event_clusters_nonlong",
        "n_long_gt_3s_event_clusters",
        "event_composition_label",
        "event_exp_gauss_fit_score",
        "background_event_p95_amp_noise_units",
        "max_event_cluster_span_s",
    ]
    point_cols = [col for col in point_cols if col in rows.columns]
    points = [
        {col: _json_safe(row[col]) for col in point_cols}
        for _, row in rows[point_cols].iterrows()
    ]

    payload = {
        "session": session_dir.name,
        "planes": plane_names,
        "points": points,
        "planeData": {},
    }
    for plane in plane_names:
        print(f"[INFO] embedding {plane}", flush=True)
        payload["planeData"][plane] = _plane_payload(session_dir, plane, max_frames=max_frames)

    html = HTML_TEMPLATE.replace("__TITLE__", f"{session_dir.name} 3D event composition")
    html = html.replace(
        "__PLANE_OPTIONS__",
        "\n".join(f'<option value="{p}">{p}</option>' for p in ["all", *plane_names]),
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
:root { --bg:#f7f7f5; --panel:#fff; --ink:#1f2933; --muted:#667085; --line:#d0d5dd; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Arial, Helvetica, sans-serif; }
.page { width:min(1760px, calc(100vw - 28px)); margin:16px auto 28px; }
.header { display:flex; justify-content:space-between; gap:16px; align-items:end; margin-bottom:12px; }
h1 { margin:0; font-size:22px; }
.meta { color:var(--muted); font-size:13px; text-align:right; line-height:1.35; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:10px; box-sizing:border-box; }
.panel-title { font-size:14px; font-weight:700; margin-bottom:8px; }
.controls { display:grid; grid-template-columns:repeat(4, auto) 1fr; gap:9px; align-items:end; margin-bottom:10px; }
label { font-size:12px; color:#475467; display:grid; gap:3px; }
select, input, button { font:inherit; border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#fff; box-sizing:border-box; }
select { width:160px; } input { width:90px; } button { cursor:pointer; white-space:nowrap; }
.grid { display:grid; grid-template-columns:minmax(620px, 1.15fr) minmax(520px, .85fr); gap:10px; align-items:start; }
canvas { width:100%; display:block; background:#fff; border:1px solid var(--line); box-sizing:border-box; }
#scatterCanvas { height:720px; cursor:grab; }
#traceCanvas { height:310px; cursor:grab; }
.readout { font-size:13px; line-height:1.45; }
.pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 7px; margin:0 4px 5px 0; font-size:12px; }
.note { color:var(--muted); font-size:12px; margin-top:6px; }
@media (max-width:1180px) { .grid, .controls { grid-template-columns:1fr; } .header { display:block; } .meta { text-align:left; } #scatterCanvas { height:620px; } }
</style>
</head>
<body>
<div class="page">
  <div class="header"><h1>__TITLE__</h1><div class="meta" id="meta"></div></div>
  <div class="controls panel">
    <label>Plane<select id="planeSelect">__PLANE_OPTIONS__</select></label>
    <label>Long clusters<select id="longFilter"><option value="all">all</option><option value="long">has >=3s cluster</option><option value="nonlong">no >=3s cluster</option></select></label>
    <label>Point size<input id="pointSize" type="number" min="2" max="16" value="6"></label>
    <label>Trace window s<input id="traceWindow" type="number" min="1" max="120" value="30"></label>
    <div><button id="reset3d">Reset 3D</button></div>
  </div>
  <div class="grid">
    <div class="panel">
      <div class="panel-title">3D ROI Event Composition</div>
      <canvas id="scatterCanvas"></canvas>
      <div class="note">Drag to rotate. Wheel to zoom. Click a point to select that ROI. Axes are non-long event fractions; long >=3s clusters are excluded from those fractions and shown with a black outline.</div>
    </div>
    <div>
      <div class="panel">
        <div class="panel-title">Selected ROI</div>
        <div class="readout" id="readout"></div>
      </div>
      <div class="panel" style="margin-top:10px;">
        <div class="panel-title">Selected ROI dF/F</div>
        <canvas id="traceCanvas"></canvas>
        <div class="note">Wheel to zoom time. Drag to pan. Double-click resets around the selected ROI.</div>
      </div>
    </div>
  </div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const payload = JSON.parse(document.getElementById("payload").textContent);
const decoded = {};
const colors = {0:"#4575b4", 1:"#6a994e", 2:"#d73027"};
let yaw = 0.72, pitch = 0.36, zoom3d = 1.0, selectedIndex = 0, dragging3d = null;
let traceView = null, traceDrag = null;
function b64f32(base64) {
  const binary = atob(base64), bytes = new Uint8Array(binary.length), chunk=1024*1024;
  for (let start=0; start<binary.length; start+=chunk) {
    const end=Math.min(binary.length,start+chunk);
    for (let i=start;i<end;i++) bytes[i]=binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}
function getPlaneData(plane) {
  const p = payload.planeData[plane];
  if (!decoded[plane]) decoded[plane] = {dff:b64f32(p.dff), events:p.events ? b64f32(p.events) : null};
  return p;
}
function arrays(plane) { getPlaneData(plane); return decoded[plane]; }
function fit(canvas) { const r=window.devicePixelRatio||1, box=canvas.getBoundingClientRect(); canvas.width=Math.max(1,Math.round(box.width*r)); canvas.height=Math.max(1,Math.round(box.height*r)); }
function clear(ctx,c) { ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle="#fff"; ctx.fillRect(0,0,c.width,c.height); }
function fmt(v) { return Number.isFinite(v) ? Number(v).toPrecision(4) : ""; }
function activePoints() {
  const plane=document.getElementById("planeSelect").value, lf=document.getElementById("longFilter").value;
  return payload.points.filter(p => {
    if (plane !== "all" && p.plane !== plane) return false;
    const hasLong = p.has_long_gt_3s_event_cluster === true;
    if (lf === "long" && !hasLong) return false;
    if (lf === "nonlong" && hasLong) return false;
    return Number.isFinite(p.nonlong_event_fraction_lt_2sd) && Number.isFinite(p.nonlong_event_fraction_2_4sd) && Number.isFinite(p.nonlong_event_fraction_gt_4sd);
  });
}
function rotatePoint(x,y,z) {
  x -= 1/3; y -= 1/3; z -= 1/3;
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const x1=cy*x-sy*z, z1=sy*x+cy*z, y1=cp*y-sp*z1, z2=sp*y+cp*z1;
  return [x1,y1,z2];
}
function projectPoint(p,c) {
  const [rx,ry,rz]=rotatePoint(p.nonlong_event_fraction_lt_2sd, p.nonlong_event_fraction_2_4sd, p.nonlong_event_fraction_gt_4sd);
  const scale=Math.min(c.width,c.height)*0.82*zoom3d, perspective=1.7/(1.7-rz);
  return {x:c.width/2+rx*scale*perspective, y:c.height/2-ry*scale*perspective, z:rz};
}
function drawAxis(ctx,c, from, to, label) {
  const p0=projectPoint({nonlong_event_fraction_lt_2sd:from[0], nonlong_event_fraction_2_4sd:from[1], nonlong_event_fraction_gt_4sd:from[2]}, c);
  const p1=projectPoint({nonlong_event_fraction_lt_2sd:to[0], nonlong_event_fraction_2_4sd:to[1], nonlong_event_fraction_gt_4sd:to[2]}, c);
  ctx.strokeStyle="#94a3b8"; ctx.lineWidth=1.2; ctx.beginPath(); ctx.moveTo(p0.x,p0.y); ctx.lineTo(p1.x,p1.y); ctx.stroke();
  ctx.fillStyle="#475467"; ctx.font="12px Arial"; ctx.textAlign="center"; ctx.fillText(label,p1.x,p1.y-8);
}
function drawScatter() {
  const c=document.getElementById("scatterCanvas"); fit(c); const ctx=c.getContext("2d"); clear(ctx,c);
  const pts=activePoints(), size=Number(document.getElementById("pointSize").value)||6;
  drawAxis(ctx,c,[0,0,0],[1,0,0],"<2 SD");
  drawAxis(ctx,c,[0,0,0],[0,1,0],"2-4 SD");
  drawAxis(ctx,c,[0,0,0],[0,0,1],">=4 SD");
  const projected=pts.map((p,i)=>({p,i,...projectPoint(p,c)})).sort((a,b)=>a.z-b.z);
  for (const item of projected) {
    const p=item.p, cluster=Number(p.nonlong_event_composition_cluster), isSel=payload.points[selectedIndex]===p;
    ctx.beginPath(); ctx.arc(item.x,item.y,isSel ? size*1.8 : size,0,Math.PI*2);
    ctx.fillStyle=colors[cluster] || "#64748b"; ctx.globalAlpha=isSel ? 1 : 0.72; ctx.fill(); ctx.globalAlpha=1;
    if (p.has_long_gt_3s_event_cluster || isSel) {
      ctx.strokeStyle=isSel ? "#f97316" : "#111827"; ctx.lineWidth=isSel ? 3 : 1.2; ctx.stroke();
    }
  }
  ctx.fillStyle="#1f2933"; ctx.font="13px Arial"; ctx.textAlign="left";
  ctx.fillText(`${pts.length} ROIs shown`, 12, 20);
  ctx.fillStyle="#4575b4"; ctx.fillText("cluster 0", 12, 42);
  ctx.fillStyle="#6a994e"; ctx.fillText("cluster 1", 88, 42);
  ctx.fillStyle="#d73027"; ctx.fillText("cluster 2", 164, 42);
  ctx.strokeStyle="#111827"; ctx.lineWidth=1.2; ctx.strokeRect(242, 33, 12, 12); ctx.fillStyle="#1f2933"; ctx.fillText("has >=3s event cluster", 260, 42);
}
function nearestPoint(event) {
  const c=document.getElementById("scatterCanvas"), box=c.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  const x=(event.clientX-box.left)*dpr, y=(event.clientY-box.top)*dpr, pts=activePoints();
  let best=null;
  for (const p of pts) {
    const proj=projectPoint(p,c), d=(proj.x-x)**2+(proj.y-y)**2;
    if (!best || d<best.d) best={p,d};
  }
  if (best && best.d < 400) return payload.points.indexOf(best.p);
  return null;
}
function selectedPoint() { return payload.points[selectedIndex] || activePoints()[0]; }
function traceFor(p) {
  const pd=getPlaneData(p.plane), arr=arrays(p.plane).dff, n=pd.nFrames, roi=Number(p.roi_index);
  return arr.subarray(roi*n, (roi+1)*n);
}
function eventsFor(p) {
  const pd=getPlaneData(p.plane), ev=arrays(p.plane).events, n=pd.nFrames, roi=Number(p.roi_index);
  return ev ? ev.subarray(roi*n, (roi+1)*n) : null;
}
function setTraceAroundSelected() {
  const p=selectedPoint(), pd=getPlaneData(p.plane), dur=(pd.nFrames-1)/pd.frameRate, win=Number(document.getElementById("traceWindow").value)||30;
  traceView={start:0, end:Math.min(dur, win)};
}
function setTraceView(a,b) {
  const p=selectedPoint(), pd=getPlaneData(p.plane), dur=(pd.nFrames-1)/pd.frameRate, minSpan=Math.max(.05,1/pd.frameRate);
  a=Math.max(0,Math.min(dur,a)); b=Math.max(0,Math.min(dur,b)); if(b<a) [a,b]=[b,a];
  if(b-a<minSpan) { const mid=(a+b)/2; a=mid-minSpan/2; b=mid+minSpan/2; }
  if(a<0) { b-=a; a=0; } if(b>dur) { a-=b-dur; b=dur; }
  traceView={start:Math.max(0,a), end:Math.min(dur,b)};
}
function drawTrace() {
  const p=selectedPoint(), pd=getPlaneData(p.plane), tr=traceFor(p), ev=eventsFor(p), c=document.getElementById("traceCanvas"); fit(c); const ctx=c.getContext("2d"); clear(ctx,c);
  if(!traceView) setTraceAroundSelected();
  const f0=Math.max(0,Math.floor(traceView.start*pd.frameRate)), f1=Math.min(pd.nFrames-1,Math.ceil(traceView.end*pd.frameRate));
  const l=62,t=14,w=c.width-78,h=c.height-54, visible=tr.subarray(f0,f1+1);
  let ymin=Infinity,ymax=-Infinity;
  for(let i=0;i<visible.length;i++) { const v=visible[i]; if(Number.isFinite(v)) { if(v<ymin)ymin=v; if(v>ymax)ymax=v; } }
  if(!Number.isFinite(ymin)||!Number.isFinite(ymax)||ymin===ymax) { ymin=-1; ymax=1; }
  const padY=(ymax-ymin)*.08||1; ymin-=padY; ymax+=padY;
  const xOf=i=>l+((i/pd.frameRate-traceView.start)/(traceView.end-traceView.start))*w, yOf=v=>t+(1-((v-ymin)/(ymax-ymin)))*h;
  ctx.strokeStyle="#d0d5dd"; ctx.beginPath(); ctx.moveTo(l,t); ctx.lineTo(l,t+h); ctx.lineTo(l+w,t+h); ctx.stroke();
  ctx.font="12px Arial"; ctx.fillStyle="#475467"; ctx.textAlign="right"; ctx.textBaseline="middle";
  for(let tick=0;tick<=4;tick++) { const val=ymin+(tick/4)*(ymax-ymin), y=yOf(val); ctx.strokeStyle="#eef0f2"; ctx.beginPath(); ctx.moveTo(l,y); ctx.lineTo(l+w,y); ctx.stroke(); ctx.fillText(val.toFixed(2),l-8,y); }
  const windowS=traceView.end-traceView.start, useMs=windowS<=2.0; ctx.textAlign="center"; ctx.textBaseline="top";
  for(let tick=0;tick<=5;tick++) { const timeS=traceView.start+(tick/5)*windowS, x=l+(tick/5)*w; ctx.strokeStyle="#eef0f2"; ctx.beginPath(); ctx.moveTo(x,t); ctx.lineTo(x,t+h); ctx.stroke(); ctx.fillStyle="#475467"; ctx.fillText(useMs ? `${((timeS-traceView.start)*1000).toFixed(0)} ms` : `${timeS.toFixed(windowS<20?1:0)} s`, x, t+h+8); }
  ctx.fillText(useMs ? "time from window start" : "session time", l+w/2, c.height-14);
  if(ev) { ctx.strokeStyle="rgba(239,68,68,.45)"; for(let i=f0+1;i<=f1;i++) if(ev[i]>0 && !(ev[i-1]>0)) { const x=xOf(i); ctx.beginPath(); ctx.moveTo(x,t); ctx.lineTo(x,t+h); ctx.stroke(); } }
  ctx.strokeStyle="#1d4ed8"; ctx.lineWidth=1.5; ctx.beginPath();
  const spanFrames=Math.max(2,f1-f0+1), columns=Math.max(1,Math.floor(w)), framesPerPixel=spanFrames/columns;
  if(framesPerPixel<=1.5) { let first=true; for(let i=f0;i<=f1;i++) { const x=xOf(i), y=yOf(tr[i]); if(first){ctx.moveTo(x,y); first=false;} else ctx.lineTo(x,y); } }
  else { for(let col=0;col<columns;col++) { const a=Math.floor(f0+col*framesPerPixel), b=Math.min(f1,Math.floor(f0+(col+1)*framesPerPixel)); let mn=Infinity,mx=-Infinity; for(let f=a;f<=b;f++){const v=tr[f]; if(Number.isFinite(v)){if(v<mn)mn=v;if(v>mx)mx=v;}} if(Number.isFinite(mn)&&Number.isFinite(mx)){const x=l+col; ctx.moveTo(x,yOf(mn)); ctx.lineTo(x,yOf(mx));} } }
  ctx.stroke(); ctx.fillStyle="#101828"; ctx.textAlign="left"; ctx.textBaseline="top"; ctx.fillText(`${p.plane} ROI ${p.roi_index} | ${traceView.start.toFixed(1)}-${traceView.end.toFixed(1)}s`, l+4, 6);
}
function drawReadout() {
  const p=selectedPoint(); if(!p) return;
  const fields=[
    ["plane",p.plane],["ROI",p.roi_index],["cluster",p.nonlong_event_composition_cluster],
    ["<2 SD",fmt(p.nonlong_event_fraction_lt_2sd)],["2-4 SD",fmt(p.nonlong_event_fraction_2_4sd)],[">=4 SD",fmt(p.nonlong_event_fraction_gt_4sd)],
    ["long fraction",fmt(p.long_gt_3s_fraction_all_clusters)],["has >=3s",p.has_long_gt_3s_event_cluster],
    ["event fit",fmt(p.event_exp_gauss_fit_score)],["p95 amp SD",fmt(p.background_event_p95_amp_noise_units)]
  ];
  document.getElementById("readout").innerHTML=fields.map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join("");
}
function drawAll() { drawScatter(); drawReadout(); drawTrace(); }
const scatter=document.getElementById("scatterCanvas");
scatter.addEventListener("mousedown", e=>{ dragging3d={x:e.clientX,y:e.clientY,moved:false}; scatter.style.cursor="grabbing"; });
window.addEventListener("mousemove", e=>{ if(!dragging3d) return; const dx=e.clientX-dragging3d.x, dy=e.clientY-dragging3d.y; if(Math.abs(dx)+Math.abs(dy)>2) dragging3d.moved=true; yaw+=dx*.008; pitch=Math.max(-1.35,Math.min(1.35,pitch+dy*.008)); dragging3d.x=e.clientX; dragging3d.y=e.clientY; drawScatter(); });
window.addEventListener("mouseup", e=>{ if(dragging3d && !dragging3d.moved) { const idx=nearestPoint(e); if(idx !== null) { selectedIndex=idx; setTraceAroundSelected(); drawAll(); } } dragging3d=null; scatter.style.cursor="grab"; });
scatter.addEventListener("wheel", e=>{ e.preventDefault(); zoom3d=Math.max(.45,Math.min(2.6,zoom3d*(e.deltaY<0?1.12:.88))); drawScatter(); }, {passive:false});
document.getElementById("reset3d").addEventListener("click",()=>{ yaw=.72; pitch=.36; zoom3d=1; drawScatter(); });
["planeSelect","longFilter","pointSize"].forEach(id=>document.getElementById(id).addEventListener("change",()=>{ const pts=activePoints(); if(pts.length) selectedIndex=payload.points.indexOf(pts[0]); setTraceAroundSelected(); drawAll(); }));
document.getElementById("traceWindow").addEventListener("change",()=>{ setTraceAroundSelected(); drawAll(); });
const traceCanvas=document.getElementById("traceCanvas");
traceCanvas.addEventListener("wheel", e=>{ e.preventDefault(); const rect=traceCanvas.getBoundingClientRect(), frac=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)), center=traceView.start+frac*(traceView.end-traceView.start), span=Math.max(.05,(traceView.end-traceView.start)*(e.deltaY<0?.78:1.28)); setTraceView(center-frac*span, center+(1-frac)*span); drawAll(); }, {passive:false});
traceCanvas.addEventListener("mousedown", e=>{ traceDrag={x:e.clientX,a:traceView.start,b:traceView.end}; traceCanvas.style.cursor="grabbing"; });
window.addEventListener("mousemove", e=>{ if(!traceDrag) return; const rect=traceCanvas.getBoundingClientRect(), shift=-(e.clientX-traceDrag.x)/rect.width*(traceDrag.b-traceDrag.a); setTraceView(traceDrag.a+shift, traceDrag.b+shift); drawAll(); });
window.addEventListener("mouseup", ()=>{ traceDrag=null; traceCanvas.style.cursor="grab"; });
traceCanvas.addEventListener("dblclick",()=>{ setTraceAroundSelected(); drawAll(); });
window.addEventListener("resize", drawAll);
document.getElementById("meta").textContent=`${payload.points.length} ROIs | ${payload.planes.length} planes | 3D fractions exclude >=3s clusters`;
selectedIndex=0; setTraceAroundSelected(); drawAll();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--composition-csv", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plane", action="append", dest="planes")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    out = create_event_composition_3d_html(
        args.session_dir.expanduser().resolve(),
        args.composition_csv.expanduser().resolve(),
        args.metrics_csv.expanduser().resolve(),
        args.output.expanduser().resolve(),
        planes=args.planes,
        max_frames=args.max_frames,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
