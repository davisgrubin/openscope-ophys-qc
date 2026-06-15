#!/usr/bin/env python3
"""Materialize one OpenScope ophys session's notebook QC inputs to local storage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import session_loader as sl  # noqa: E402
import mesoscope_qc_pipeline as qc  # noqa: E402


def safe_name(value: str) -> str:
    text = str(value).strip().rstrip("/")
    text = text.split("/")[-1] if "/" in text else text
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned[:180] or "unknown"


def image_mask_to_pixel_mask(image_mask: np.ndarray, threshold: float = 0) -> np.ndarray:
    mask = np.asarray(image_mask, dtype=np.float32)
    y, x = np.nonzero(mask > threshold)
    if y.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.column_stack([x, y, mask[y, x]]).astype(np.float32)


def pack_pixel_masks(pixel_masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(np.asarray(pix)) for pix in pixel_masks], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if counts.sum() == 0:
        xyw = np.empty((0, 3), dtype=np.float32)
    else:
        xyw = np.concatenate([np.asarray(pix, dtype=np.float32).reshape(-1, 3) for pix in pixel_masks], axis=0)
    return indptr, xyw


def get_plane_projection(nwb, plane_name: str) -> tuple[np.ndarray | None, str | None]:
    proc = nwb.processing[plane_name]
    try:
        image_mod = proc["images"]
    except Exception:
        return None, None
    for image_name, label in [("average_projection", "Average projection"), ("max_projection", "Max projection")]:
        try:
            if image_name in image_mod.images:
                return np.asarray(image_mod[image_name].data, dtype=np.float32), label
        except Exception:
            pass
        try:
            return np.asarray(image_mod[image_name].data, dtype=np.float32), label
        except Exception:
            pass
    return None, None


def write_plane_masks(proc, out_path: Path, shape: tuple[int, int] | None) -> dict:
    table = proc["image_segmentation"]["roi_table"]
    colnames = list(getattr(table, "colnames", []))
    roi_indices = np.arange(len(table), dtype=np.int32)
    pixel_masks: list[np.ndarray] = []
    source = None

    if "pixel_mask" in colnames:
        source = "pixel_mask"
        for i in roi_indices:
            try:
                pixel_masks.append(np.asarray(table["pixel_mask"][int(i)], dtype=np.float32).reshape(-1, 3))
            except Exception:
                pixel_masks.append(np.empty((0, 3), dtype=np.float32))
    elif "image_mask" in colnames:
        source = "image_mask_converted"
        masks_data = table["image_mask"].data
        if shape is None and len(table):
            shape = np.asarray(masks_data[0], dtype=np.float32).shape
        for i in roi_indices:
            try:
                pixel_masks.append(image_mask_to_pixel_mask(masks_data[int(i)]))
            except Exception:
                pixel_masks.append(np.empty((0, 3), dtype=np.float32))
    else:
        return {"n_rois": int(len(table)), "source": "none", "path": None}

    indptr, xyw = pack_pixel_masks(pixel_masks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        roi_indices=roi_indices,
        shape=np.asarray(shape if shape is not None else (0, 0), dtype=np.int32),
        indptr=indptr,
        xyw=xyw,
    )
    return {
        "n_rois": int(len(table)),
        "source": source,
        "path": str(out_path),
        "n_sparse_pixels": int(xyw.shape[0]),
    }


def get_timeseries(proc, paths):
    return qc.get_timeseries_from_proc(proc, paths)


def open_local_nwb(path: Path) -> dict:
    io = NWBHDF5IO(path=str(path), mode="r", load_namespaces=True)
    nwb = io.read()
    meta = {
        "session_id": getattr(nwb, "session_id", None) or path.stem,
        "session_description": getattr(nwb, "session_description", None),
        "session_start_time": str(getattr(nwb, "session_start_time", "")),
        "source_path": str(path),
    }
    subject = getattr(nwb, "subject", None)
    if subject is not None:
        meta["subject"] = {
            "subject_id": getattr(subject, "subject_id", None),
            "species": getattr(subject, "species", None),
            "sex": getattr(subject, "sex", None),
            "age": getattr(subject, "age", None),
            "genotype": getattr(subject, "genotype", None),
        }
    return {
        "path": str(path),
        "nwb": nwb,
        "meta": meta,
        "plane_meta": qc.get_plane_metadata(nwb),
        "planes": qc.get_plane_names(nwb),
        "io": io,
    }


def open_session_source(session_source: str) -> dict:
    source_path = Path(session_source).expanduser()
    if source_path.exists():
        return open_local_nwb(source_path.resolve())
    return sl.load_session(session_source)


def close_opened_session(session: dict) -> None:
    try:
        sl.close_session(session)
    except Exception:
        try:
            session["io"].close()
        except Exception:
            pass


def materialize(session_source: str, out_root: Path, max_frames: int | None, include_timeseries: bool) -> dict:
    session = open_session_source(session_source)
    try:
        nwb = session["nwb"]
        session_key = safe_name(session["meta"].get("session_id") or session_source)
        out_dir = out_root / session_key
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "session_source": session_source,
            "session_key": session_key,
            "out_dir": str(out_dir),
            "max_frames": max_frames,
            "planes": {},
        }
        (out_dir / "session_metadata.json").write_text(json.dumps(session["meta"], indent=2), encoding="utf-8")
        plane_meta = session["plane_meta"]
        if not isinstance(plane_meta, pd.DataFrame):
            plane_meta = pd.DataFrame(plane_meta)
        plane_meta.to_csv(out_dir / "plane_metadata.csv", index=False)

        for plane_name in session["planes"]:
            print(f"[INFO] materializing {plane_name}", flush=True)
            proc = nwb.processing[plane_name]
            plane_dir = out_dir / safe_name(plane_name)
            plane_dir.mkdir(parents=True, exist_ok=True)

            projection, projection_label = get_plane_projection(nwb, plane_name)
            if projection is not None:
                np.save(plane_dir / "projection.npy", projection.astype(np.float32, copy=False))

            mask_info = write_plane_masks(
                proc,
                out_dir / f"{safe_name(plane_name)}_pixel_masks.npz",
                None if projection is None else projection.shape[:2],
            )

            plane_info = {
                "projection": str(plane_dir / "projection.npy") if projection is not None else None,
                "projection_label": projection_label,
                "masks": mask_info,
            }

            if include_timeseries:
                dff_series = get_timeseries(proc, [("dff_timeseries", "dff_timeseries"), ("dff_timeseries",)])
                if dff_series is not None:
                    dff, ts = qc.load_timeseries_matrix(dff_series, max_frames=max_frames)
                    np.save(plane_dir / "dff.npy", dff.astype(np.float32, copy=False))
                    np.save(plane_dir / "dff_timestamps.npy", ts.astype(np.float64, copy=False))
                    plane_info["dff"] = str(plane_dir / "dff.npy")
                    plane_info["dff_shape"] = list(dff.shape)

                event_series = get_timeseries(proc, [("event_timeseries",), ("events", "event_timeseries")])
                if event_series is not None:
                    events, event_ts = qc.load_timeseries_matrix(event_series, max_frames=max_frames)
                    np.save(plane_dir / "events.npy", events.astype(np.float32, copy=False))
                    np.save(plane_dir / "event_timestamps.npy", event_ts.astype(np.float64, copy=False))
                    plane_info["events"] = str(plane_dir / "events.npy")
                    plane_info["events_shape"] = list(events.shape)

            summary["planes"][plane_name] = plane_info

        (out_dir / "materialization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        close_opened_session(session)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-source", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--full-timeseries", action="store_true")
    parser.add_argument("--skip-timeseries", action="store_true")
    args = parser.parse_args()

    max_frames = None if args.full_timeseries else args.max_frames
    summary = materialize(
        args.session_source,
        args.out_root.expanduser().resolve(),
        max_frames=max_frames,
        include_timeseries=not args.skip_timeseries,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
