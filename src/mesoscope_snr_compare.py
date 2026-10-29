#!/usr/bin/env python3
"""Run the three candidate ROI SNR metrics over one or more mesoscope sessions.

For each processed session on S3 this opens the NWB, walks every imaging plane,
and computes all three metric families (see snr_three_metrics.py) for every ROI
in a single pass over its dF/F trace. ROI anatomy travels with the metrics --
soma/dendrite class and probability, mask pixel area and centroid, plane and
targeted structure -- so the "is this metric just reporting ROI size?" question
can be answered from the same table.

Ido's scheme needs a detector. The mesoscope ships the Allen pipeline's OASIS
`event_timeseries`, which is what discussion #156 specifies, so that is used
directly; the matched-filter fallback in snr_three_metrics only engages for a
plane whose event trace is missing or all-zero.

Output per session:
  outputs/snr3/roi_metrics_<label>.csv      one row per ROI, all three metrics
  outputs/snr3/plane_summary_<label>.csv    per-plane medians and class shares

Usage
-----
    python src/mesoscope_snr_compare.py --sessions s3://... [s3://... ...]
    python src/mesoscope_snr_compare.py --session-list outputs/picks.csv
    python src/mesoscope_snr_compare.py --sessions s3://... --max-rois 40   # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mesoscope_qc_pipeline as qc  # noqa: E402
import snr_three_metrics as M  # noqa: E402

# Candidate NWB paths for each timeseries, most specific first. The processing
# module layout is documented in documentation/filetype_reference.md.
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


def session_label(s3_path: str) -> str:
    """.../multiplane-ophys_839909_2026-03-19_12-45-42_processed_.../pophys.nwb.zarr
    -> 839909_2026-03-19. The path may point at the asset folder or at the zarr
    inside it, so find the asset component rather than taking the basename."""
    for part in reversed(s3_path.rstrip("/").split("/")):
        if part.startswith("multiplane-ophys_"):
            fields = part.split("_processed_")[0].split("_")
            if len(fields) >= 3:
                return f"{fields[1]}_{fields[2]}"
    return s3_path.rstrip("/").split("/")[-1]


def dt_from_timestamps(ts: np.ndarray) -> float:
    """Median sample interval. Timestamps are explicit in these NWBs."""
    if ts.size < 3:
        return 1.0 / M.FRAME_RATE_HZ_NOMINAL
    dt = float(np.nanmedian(np.diff(ts)))
    if not np.isfinite(dt) or dt <= 0:
        return 1.0 / M.FRAME_RATE_HZ_NOMINAL
    return dt


def process_plane(nwb, plane_name: str, max_rois: int | None = None,
                  max_frames: int | None = None, load_masks: bool = True) -> pd.DataFrame | None:
    """Per-ROI metrics for one imaging plane, with ROI anatomy joined on."""
    proc = nwb.processing[plane_name]

    dff_series = qc.get_timeseries_from_proc(proc, DFF_PATHS)
    if dff_series is None:
        print(f"    [{plane_name}] no dF/F timeseries -- skipped", flush=True)
        return None
    t_read = time.time()
    dff, ts = qc.load_timeseries_matrix(dff_series, max_frames=max_frames)
    dt = dt_from_timestamps(ts)
    t_dff = time.time() - t_read

    t_read = time.time()
    ev_series = qc.get_timeseries_from_proc(proc, EVENT_PATHS)
    events = None
    if ev_series is not None:
        events, _ = qc.load_timeseries_matrix(ev_series, max_frames=max_frames)
        if events.shape != dff.shape:
            print(f"    [{plane_name}] event shape {events.shape} != dff {dff.shape}"
                  " -- falling back to matched filter", flush=True)
            events = None

    # load_masks=True is what gives roi_area_pix, the confound we need to test.
    # It reads the dense image_mask stack (~100 MB/plane), so the multi-session
    # sweep turns it off and only the deep-dive session pays for it.
    t_ev = time.time() - t_read
    t_read = time.time()
    meta = qc.get_roi_metadata_for_plane(nwb, plane_name, load_masks=load_masks)
    t_meta = time.time() - t_read

    n_roi = dff.shape[1]
    if len(meta) != n_roi:
        print(f"    [{plane_name}] roi_table {len(meta)} rows vs {n_roi} traces"
              " -- truncating to the shorter", flush=True)
        n_roi = min(n_roi, len(meta))

    order = range(n_roi) if max_rois is None else range(min(n_roi, max_rois))

    plane = nwb.imaging_planes.get(plane_name) if hasattr(nwb, "imaging_planes") else None
    structure, depth_um = qc.parse_structure_depth(qc.safe_getattr(plane, "location"))
    # The plane label itself carries the targeted area (VISp_0, VISl_4, ...) and
    # is populated even where imaging_planes.location is a placeholder string,
    # which filetype_reference.md documents as the common case here.
    label_structure = plane_name.split("_")[0] if "_" in plane_name else plane_name
    plane_index = plane_name.split("_")[-1] if "_" in plane_name else ""

    t_read = time.time()
    rows = []
    for i in order:
        trace = dff[:, i]
        ev = events[:, i] if events is not None else None
        row = M.all_metrics(trace, dt, event_trace=ev)
        row.update({
            "plane": plane_name,
            "roi_index": int(i),
            "structure": structure or label_structure,
            "plane_index": plane_index,
            "depth_um": depth_um,
            "dt_s": dt,
            "n_frames": int(dff.shape[0]),
            "duration_s": float(dff.shape[0] * dt),
            "dff_mean": float(np.nanmean(trace)),
            "dff_std": float(np.nanstd(trace)),
            "dff_p99": float(np.nanpercentile(trace, 99)),
            "frac_nan": float(np.mean(~np.isfinite(trace))),
        })
        rows.append(row)

    t_metric = time.time() - t_read
    print(f"    [{plane_name}] timing: dff {t_dff:.1f}s  events {t_ev:.1f}s"
          f"  roi_meta {t_meta:.1f}s  metrics {t_metric:.1f}s"
          f"  ({dff.shape[0]} frames x {dff.shape[1]} ROIs)", flush=True)

    df = pd.DataFrame(rows)
    keep = [c for c in ["roi_index", "is_soma", "is_dendrite", "soma_probability",
                        "dendrite_probability", "roi_area_pix",
                        "roi_centroid_x_pix", "roi_centroid_y_pix"] if c in meta.columns]
    return df.merge(meta[keep], on="roi_index", how="left")


def process_session(s3_path: str, out_dir: Path, max_rois: int | None = None,
                    max_frames: int | None = None, load_masks: bool = True,
                    only_planes: list[str] | None = None) -> pd.DataFrame | None:
    label = session_label(s3_path)
    print(f"[{label}] opening {s3_path}", flush=True)
    t0 = time.time()
    opened = qc.open_nwb_from_s3(s3_path)
    nwb = opened.nwb
    try:
        planes = qc.get_plane_names(nwb)
        if only_planes:
            planes = [p for p in planes if p in only_planes]
        print(f"[{label}] {len(planes)} planes: {planes}", flush=True)
        frames = []
        for plane_name in planes:
            tp = time.time()
            try:
                df = process_plane(nwb, plane_name, max_rois=max_rois,
                                   max_frames=max_frames, load_masks=load_masks)
            except Exception:
                print(f"    [{plane_name}] FAILED:\n{traceback.format_exc()}", flush=True)
                continue
            if df is not None and len(df):
                frames.append(df)
                print(f"    [{plane_name}] {len(df)} ROIs in {time.time()-tp:.1f}s"
                      f" (detector: {df['ido_detector'].mode().iat[0]})", flush=True)
        if not frames:
            print(f"[{label}] no planes yielded metrics", flush=True)
            return None

        session = pd.concat(frames, ignore_index=True)
        session.insert(0, "session_label", label)
        session.insert(1, "s3_path", s3_path)
        session.insert(2, "subject_id", label.split("_")[0])
        session.insert(3, "acq_date", label.split("_")[1] if "_" in label else "")

        # Quality classes are assigned per session: k-means is fitted within a
        # session so the three classes describe that session's own spread, which
        # is how #156 and PR #171 both do it.
        session = M.assign_ido_classes(session)

        out_dir.mkdir(parents=True, exist_ok=True)
        roi_csv = out_dir / f"roi_metrics_{label}.csv"
        session.to_csv(roi_csv, index=False)

        summary = summarize(session)
        summary.to_csv(out_dir / f"plane_summary_{label}.csv", index=False)

        print(f"[{label}] {len(session)} ROIs total in {time.time()-t0:.1f}s"
              f" -> {roi_csv.name}", flush=True)
        return session
    finally:
        try:
            opened.io.close()
        except Exception:
            pass


def summarize(session: pd.DataFrame) -> pd.DataFrame:
    """Per-plane medians of each metric plus Ido's class shares."""
    metric_cols = ["ido_frac_gt4sd", "ido_median_event_sd", "ido_event_rate_hz",
                   "maedeh_snr", "maedeh_snr_trace_p95", "maedeh_noise_floor",
                   "davis_robust_event_snr", "davis_exceptional_event_score",
                   "roi_area_pix", "soma_probability"]
    metric_cols = [c for c in metric_cols if c in session.columns]
    g = session.groupby(["session_label", "plane", "structure"], dropna=False)
    out = g[metric_cols].median().reset_index()
    out["n_roi"] = g.size().values
    if "ido_quality_class" in session.columns:
        shares = (session.groupby(["session_label", "plane", "structure"],
                                  dropna=False)["ido_quality_class"]
                  .value_counts(normalize=True).unstack(fill_value=0.0).reset_index())
        out = out.merge(shares, on=["session_label", "plane", "structure"], how="left")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=[])
    ap.add_argument("--session-list", default=None,
                    help="CSV with an s3_path column")
    ap.add_argument("--out-dir", default="outputs/snr3")
    ap.add_argument("--max-rois", type=int, default=None,
                    help="smoke mode: only this many ROIs per plane")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--planes", nargs="*", default=None,
                    help="restrict to these imaging planes")
    ap.add_argument("--no-masks", action="store_true",
                    help="skip the dense image_mask read (no roi_area_pix)")
    args = ap.parse_args()

    paths = list(args.sessions)
    if args.session_list:
        paths += pd.read_csv(args.session_list)["s3_path"].tolist()
    if not paths:
        ap.error("no sessions given")

    out_dir = Path(args.out_dir)
    done = []
    for p in paths:
        try:
            df = process_session(p, out_dir, max_rois=args.max_rois,
                                 max_frames=args.max_frames,
                                 load_masks=not args.no_masks,
                                 only_planes=args.planes)
            if df is not None:
                done.append(df)
        except Exception:
            print(f"SESSION FAILED {p}:\n{traceback.format_exc()}", flush=True)

    if len(done) > 1:
        combined = pd.concat(done, ignore_index=True)
        combined.to_csv(out_dir / "roi_metrics_multisession.csv", index=False)
        pd.concat([summarize(d) for d in done], ignore_index=True).to_csv(
            out_dir / "session_summary_multisession.csv", index=False)
        print(f"combined {len(combined)} ROIs from {len(done)} sessions", flush=True)


if __name__ == "__main__":
    main()
