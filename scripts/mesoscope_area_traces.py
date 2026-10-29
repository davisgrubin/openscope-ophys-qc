"""Second-pass extraction for the mesoscope SNR panel.

Three things the first pass left out:

A. ROI mask geometry (area, centroid, bounding box) for every session. The
   first pass ran with --no-masks because reading the dense image_mask stack
   one ROI at a time costs ~1.1 s/ROI over S3. The masks are stored as a
   (n_roi, 512, 512) float32 zarr array chunked (44, 64, 128), so reading
   chunk-aligned slabs of ROIs instead of single ROIs costs ~0.03 s/ROI --
   a 33x saving that turns the whole sweep from hours into minutes.

B. Per-stimulus-context metrics for one deep-dive session. The ophys NWB
   carries no interval tables, so the context boundaries are read from the
   asset's session.json stimulus_epochs and converted to seconds relative to
   the NWB session start time.

C. dF/F and event traces for a requested list of ROIs, so the ROIs whose
   metric ranks disagree most can be shown as traces.

Every stage writes per-session output and skips work already on disk, so a
preempted job can be resubmitted without repeating completed sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import boto3
import botocore
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import mesoscope_qc_pipeline as qc  # noqa: E402
import snr_three_metrics as M  # noqa: E402

# Copied verbatim from the first-pass driver (mesoscope_snr_compare.py); these
# are the container/key spellings the OpenScope processed NWBs actually use.
DFF_PATHS = [
    ("dff_timeseries", "dff_timeseries"),
    ("dff_timeseries",),
    ("dff", "dff"),
]
EVENT_PATHS = [
    ("event_timeseries", "event_timeseries"),
    ("event_timeseries",),
    ("events", "events"),
]
BUCKET = "aind-open-data"
SLAB = 44  # matches the image_mask chunk size along the ROI axis

# Definitional names for the metric columns the library emits.
CANONICAL_COLUMNS = {
    "maedeh_snr": "event_amplitude_snr",
    "maedeh_noise_floor": "noise_floor",
    "maedeh_baseline": "baseline_dff",
    "maedeh_peak_rate_hz": "peak_rate_hz",
    "davis_robust_event_snr": "percentile_range_snr",
    "davis_noise_fast_mad": "noise_fast_mad",
    "davis_signal_p95_p50": "signal_p95_minus_p50",
    "ido_frac_gt4sd": "frac_events_gt4sd",
    "ido_frac_lt2sd": "frac_events_lt2sd",
    "ido_frac_2_4sd": "frac_events_2_4sd",
    "ido_median_event_sd": "median_event_amp_sd",
    "ido_p90_event_sd": "p90_event_amp_sd",
    "ido_event_rate_hz": "event_rate_hz",
    "ido_n_events": "n_events",
    "ido_noise_sd_raw": "noise_sd_raw",
}


def s3_client():
    return boto3.client("s3", config=botocore.config.Config(
        signature_version=botocore.UNSIGNED))


def session_label(s3_path: str) -> str:
    for part in s3_path.split("/"):
        if part.startswith("multiplane-ophys_"):
            bits = part.split("_")
            return f"{bits[1]}_{bits[2]}"
    return s3_path.rstrip("/").split("/")[-1]


def asset_prefix(s3_path: str) -> str:
    for part in s3_path.split("/"):
        if part.startswith("multiplane-ophys_"):
            return part + "/"
    raise ValueError(s3_path)


def dt_from_timestamps(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=float)
    return float(np.median(np.diff(ts))) if ts.size > 1 else np.nan


def plane_segmentation(nwb, plane_name: str):
    """The PlaneSegmentation table for one plane, wherever it is filed."""
    proc = nwb.processing[plane_name]
    for key in list(proc.data_interfaces):
        di = proc[key]
        if hasattr(di, "plane_segmentations"):
            segs = di.plane_segmentations
            return segs[list(segs)[0]]
    return None


# ---------------------------------------------------------------------------
# A. mask geometry
# ---------------------------------------------------------------------------

def mask_geometry_slab(seg, slab: int = SLAB) -> pd.DataFrame:
    """Area, weighted centroid and bounding box per ROI, read in slabs.

    Reading `slab` ROIs per request rather than one keeps each read aligned to
    the zarr chunk along the ROI axis, which is where the 33x speed-up over
    per-ROI access comes from.
    """
    data = seg["image_mask"].data
    n = int(data.shape[0])
    rec = {k: np.full(n, np.nan) for k in
           ("roi_area_pix", "roi_centroid_x_pix", "roi_centroid_y_pix",
            "roi_bbox_w_pix", "roi_bbox_h_pix", "roi_mask_weight_sum")}
    for start in range(0, n, slab):
        stop = min(start + slab, n)
        block = np.asarray(data[start:stop])
        for j in range(block.shape[0]):
            mask = block[j]
            yy, xx = np.nonzero(mask > 0)
            i = start + j
            rec["roi_area_pix"][i] = float(yy.size)
            if yy.size == 0:
                continue
            w = mask[yy, xx].astype(float)
            tot = float(w.sum())
            rec["roi_mask_weight_sum"][i] = tot
            if tot > 0:
                rec["roi_centroid_x_pix"][i] = float((xx * w).sum() / tot)
                rec["roi_centroid_y_pix"][i] = float((yy * w).sum() / tot)
            else:
                rec["roi_centroid_x_pix"][i] = float(xx.mean())
                rec["roi_centroid_y_pix"][i] = float(yy.mean())
            rec["roi_bbox_w_pix"][i] = float(xx.max() - xx.min() + 1)
            rec["roi_bbox_h_pix"][i] = float(yy.max() - yy.min() + 1)
    out = pd.DataFrame(rec)
    out.insert(0, "roi_index", np.arange(n))
    return out


def stage_mask_geometry(s3_path: str, out_dir: Path) -> None:
    label = session_label(s3_path)
    dest = out_dir / f"mask_geom_{label}.csv"
    if dest.exists():
        print(f"[{label}] mask geometry already present -- skipped", flush=True)
        return
    t0 = time.time()
    opened = qc.open_nwb_from_s3(s3_path)
    nwb = opened.nwb
    try:
        frames = []
        for plane_name in qc.get_plane_names(nwb):
            tp = time.time()
            seg = plane_segmentation(nwb, plane_name)
            if seg is None or "image_mask" not in list(seg.colnames):
                print(f"    [{plane_name}] no image_mask -- skipped", flush=True)
                continue
            g = mask_geometry_slab(seg)
            g.insert(0, "plane", plane_name)
            frames.append(g)
            print(f"    [{plane_name}] {len(g)} ROIs in {time.time()-tp:.1f}s",
                  flush=True)
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        df.insert(0, "session_label", label)
        df.to_csv(dest, index=False)
        print(f"[{label}] mask geometry {len(df)} ROIs in {time.time()-t0:.1f}s"
              f" -> {dest.name}", flush=True)
    finally:
        try:
            opened.io.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# B/C. per-context metrics and example traces for one session
# ---------------------------------------------------------------------------

def read_epochs(s3_path: str, session_start) -> pd.DataFrame:
    """Stimulus epochs as seconds relative to the NWB session start time."""
    key = asset_prefix(s3_path) + "session.json"
    body = s3_client().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    eps = json.loads(body).get("stimulus_epochs") or []
    rows = []
    for k, e in enumerate(eps):
        t0 = pd.Timestamp(e["stimulus_start_time"])
        t1 = pd.Timestamp(e["stimulus_end_time"])
        rows.append({"epoch_index": k,
                     "epoch_name": e.get("stimulus_name") or f"epoch {k}",
                     "t_start_s": (t0 - session_start).total_seconds(),
                     "t_end_s": (t1 - session_start).total_seconds()})
    df = pd.DataFrame(rows)
    # Two blocks can carry the same name (a control block is often repeated
    # before and after the manipulation); make the label unique by occurrence.
    dup = df.epoch_name.duplicated(keep=False)
    seen: dict[str, int] = {}
    labels = []
    for name, is_dup in zip(df.epoch_name, dup):
        if is_dup:
            seen[name] = seen.get(name, 0) + 1
            labels.append(f"{name} ({seen[name]})")
        else:
            labels.append(name)
    df["epoch_label"] = labels
    return df


def align_timestamps(ts: np.ndarray, session_start) -> np.ndarray:
    """Trace timestamps as seconds from session start, whatever the origin."""
    ts = np.asarray(ts, dtype=float)
    if ts.size and ts[0] > 1e6:  # stored as unix epoch seconds
        ts = ts - session_start.timestamp()
    return ts


def stage_context_and_traces(s3_path: str, out_dir: Path,
                             picks: pd.DataFrame,
                             label_planes: list[str],
                             min_frames: int = 300) -> None:
    label = session_label(s3_path)
    ctx_dest = out_dir / f"context_metrics_{label}.csv"
    tr_dest = out_dir / f"traces_{label}.npz"
    lbl_dest = out_dir / f"roi_label_images_{label}.npz"
    if ctx_dest.exists() and tr_dest.exists() and lbl_dest.exists():
        print(f"[{label}] context/traces already present -- skipped", flush=True)
        return

    opened = qc.open_nwb_from_s3(s3_path)
    nwb = opened.nwb
    try:
        session_start = pd.Timestamp(nwb.session_start_time)
        epochs = read_epochs(s3_path, session_start)
        epochs.to_csv(out_dir / f"epochs_{label}.csv", index=False)
        print(epochs.to_string(index=False), flush=True)

        ctx_rows, trace_store, label_images = [], {}, {}
        for plane_name in qc.get_plane_names(nwb):
            tp = time.time()
            dff_series = qc.get_timeseries_from_proc(nwb.processing[plane_name],
                                                     DFF_PATHS)
            if dff_series is None:
                print(f"    [{plane_name}] no dF/F timeseries at any of"
                      f" {DFF_PATHS} -- skipped", flush=True)
                continue
            dff, ts = qc.load_timeseries_matrix(dff_series)
            ts = align_timestamps(ts, session_start)
            dt = dt_from_timestamps(ts)
            ev_series = qc.get_timeseries_from_proc(nwb.processing[plane_name],
                                                    EVENT_PATHS)
            events = None
            if ev_series is not None:
                events, _ = qc.load_timeseries_matrix(ev_series)
                if events.shape != dff.shape:
                    events = None

            # C. traces for the requested ROIs of this plane
            want = picks.loc[picks.plane_label == plane_name, "roi_index"].astype(int)
            for ri in want:
                if ri >= dff.shape[1]:
                    continue
                trace_store[f"{plane_name}|{ri}|dff"] = dff[:, ri].astype(np.float32)
                if events is not None:
                    trace_store[f"{plane_name}|{ri}|ev"] = events[:, ri].astype(np.float32)
            trace_store[f"{plane_name}|ts"] = ts.astype(np.float32)

            # B. metrics recomputed inside each stimulus context
            for _, ep in epochs.iterrows():
                sel = (ts >= ep.t_start_s) & (ts < ep.t_end_s)
                if int(sel.sum()) < min_frames:
                    continue
                idx = np.flatnonzero(sel)
                for i in range(dff.shape[1]):
                    seg_trace = dff[idx, i]
                    seg_ev = events[idx, i] if events is not None else None
                    row = M.all_metrics(seg_trace, dt, event_trace=seg_ev)
                    row.update({"plane": plane_name, "roi_index": int(i),
                                "epoch_index": int(ep.epoch_index),
                                "epoch_label": ep.epoch_label,
                                "epoch_name": ep.epoch_name,
                                "n_frames": int(idx.size),
                                "duration_s": float(idx.size * dt)})
                    ctx_rows.append(row)
            print(f"    [{plane_name}] context metrics for {dff.shape[1]} ROIs"
                  f" x {len(epochs)} epochs in {time.time()-tp:.1f}s", flush=True)

            # ROI map background for the layout figure
            if plane_name in label_planes:
                seg = plane_segmentation(nwb, plane_name)
                if seg is not None and "image_mask" in list(seg.colnames):
                    data = seg["image_mask"].data
                    n = int(data.shape[0])
                    best = np.zeros(data.shape[1:], dtype=np.float32)
                    lbl = np.zeros(data.shape[1:], dtype=np.int32)
                    for start in range(0, n, SLAB):
                        stop = min(start + SLAB, n)
                        block = np.asarray(data[start:stop])
                        for j in range(block.shape[0]):
                            m = block[j]
                            hit = m > best
                            best[hit] = m[hit]
                            lbl[hit] = start + j + 1  # 0 = background
                    label_images[plane_name] = lbl.astype(np.int32)
                    print(f"    [{plane_name}] label image {lbl.shape}", flush=True)

        ctx = pd.DataFrame(ctx_rows).rename(columns=CANONICAL_COLUMNS)
        ctx.insert(0, "session_label", label)
        ctx.to_csv(ctx_dest, index=False)
        np.savez_compressed(tr_dest, **trace_store)
        np.savez_compressed(lbl_dest, **label_images)

        # A compact bundle for the figures. The full traces stay on the cluster;
        # what comes back is a fixed display window, a per-ROI most-active
        # window, and the event peak amplitudes, which is all the trace and
        # histogram rows of the layout figure need.
        bundle_compact(out_dir, label, trace_store, label_images, ctx)
        print(f"[{label}] context rows {len(ctx)}; traces {len(trace_store)};"
              f" label images {list(label_images)}", flush=True)
    finally:
        try:
            opened.io.close()
        except Exception:
            pass


def bundle_compact(out_dir: Path, label: str, trace_store: dict,
                   label_images: dict, ctx: pd.DataFrame,
                   fixed_window_s: float = 300.0,
                   active_window_s: float = 120.0) -> None:
    """Small npz + gzipped CSVs sized for transfer off the cluster."""
    store: dict[str, np.ndarray] = {}
    for key, arr in trace_store.items():
        if not key.endswith("|dff"):
            continue
        plane, ri, _ = key.split("|")
        ts = trace_store[f"{plane}|ts"]
        dt = float(np.median(np.diff(ts))) if ts.size > 1 else np.nan
        ev = trace_store.get(f"{plane}|{ri}|ev")

        n_fix = int(fixed_window_s / dt)
        store[f"{plane}|{ri}|fixed_dff"] = arr[:n_fix].astype(np.float32)
        if ev is not None:
            store[f"{plane}|{ri}|fixed_ev"] = ev[:n_fix].astype(np.float32)
        store[f"{plane}|{ri}|fixed_t"] = ts[:n_fix].astype(np.float32)

        # window with the largest summed event activity, so a sparse ROI is
        # shown where its few events actually are
        n_act = int(active_window_s / dt)
        drive = ev if ev is not None else np.abs(arr)
        if drive.size > n_act:
            csum = np.concatenate([[0.0], np.cumsum(drive, dtype=float)])
            start = int(np.argmax(csum[n_act:] - csum[:-n_act]))
        else:
            start = 0
        sl = slice(start, start + n_act)
        store[f"{plane}|{ri}|active_dff"] = arr[sl].astype(np.float32)
        if ev is not None:
            store[f"{plane}|{ri}|active_ev"] = ev[sl].astype(np.float32)
        store[f"{plane}|{ri}|active_t"] = ts[sl].astype(np.float32)

        # full-session event peak amplitudes in units of the trace's own
        # baseline noise -- the x axis of the event-category histogram
        try:
            idx = (M.event_indices_from_allen(ev, max(1, int(round(0.5 / dt))))
                   if ev is not None else np.empty(0, dtype=int))
            noise, _ = M.baseline_noise_lower_half(arr.astype(float))
            amps = M.event_amplitudes_raw(arr.astype(float), idx, dt, noise)
            store[f"{plane}|{ri}|event_amp_sd"] = np.asarray(amps, dtype=np.float32)
        except Exception:
            print(f"    event amplitudes failed for {plane} ROI {ri}:\n"
                  f"{traceback.format_exc()}", flush=True)
    for plane, img in label_images.items():
        store[f"{plane}|label_image"] = img.astype(np.int16)

    np.savez_compressed(out_dir / f"figure_bundle_{label}.npz", **store)

    # The metrics library still emits columns prefixed with the name of whoever
    # proposed each scheme; every table that leaves this script carries the
    # definitional name instead. Provenance is recorded in the methods note.
    ctx = ctx.rename(columns=CANONICAL_COLUMNS)

    keep = [c for c in ("session_label", "plane", "roi_index", "epoch_index",
                        "epoch_label", "epoch_name", "n_frames", "duration_s",
                        "event_amplitude_snr", "percentile_range_snr",
                        "frac_events_gt4sd", "frac_events_lt2sd",
                        "frac_events_2_4sd", "median_event_amp_sd",
                        "event_rate_hz", "n_events", "noise_floor",
                        "noise_fast_mad", "noise_sd_raw",
                        "signal_p95_minus_p50") if c in ctx.columns]
    ctx[keep].to_csv(out_dir / f"context_slim_{label}.csv.gz", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-list", required=True,
                    help="text file of s3 paths, one per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--picks", default=None,
                    help="CSV with plane_label,roi_index for trace extraction")
    ap.add_argument("--deep-session", default=None,
                    help="session label to run the context/trace stage on")
    ap.add_argument("--label-planes", nargs="*", default=["VISp_0", "VISl_4"])
    ap.add_argument("--stages", default="AB")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = [ln.strip() for ln in open(args.session_list)
                if ln.strip() and not ln.startswith("#")]
    print(f"{len(sessions)} sessions; stages {args.stages}", flush=True)

    if "A" in args.stages:
        for s3_path in sessions:
            try:
                stage_mask_geometry(s3_path, out_dir)
            except Exception:
                print(f"[{session_label(s3_path)}] mask stage FAILED:\n"
                      f"{traceback.format_exc()}", flush=True)

    if "B" in args.stages and args.deep_session:
        picks = (pd.read_csv(args.picks) if args.picks
                 else pd.DataFrame(columns=["plane_label", "roi_index"]))
        for s3_path in sessions:
            if session_label(s3_path) != args.deep_session:
                continue
            try:
                stage_context_and_traces(s3_path, out_dir, picks,
                                         args.label_planes)
            except Exception:
                print(f"[{args.deep_session}] context stage FAILED:\n"
                      f"{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
