#!/usr/bin/env python3
"""Build the OpenScope Predictive Processing session inventory from S3.

Two corrections over session_loader.discover_sessions():

1. The `multiplane-ophys_` prefix in aind-open-data is shared with other Allen
   projects (most of it is "Learning mFISH-V1omFISH", Garrett/Arkhipov). Only
   assets whose data_description.json has project_name == "OpenScope" belong to
   this data release, so the project name is read per session and used as the
   filter.

2. Each acquisition has several `_processed_` assets from different pipeline
   runs -- a behaviour/stimulus NWB, a combined NWB, and the ophys one. Only the
   ophys asset contains `pophys.nwb.zarr`, and it is often NOT the newest, so
   keeping the latest processed timestamp (what _keep_latest does) can select an
   asset with no imaging data in it at all. This resolves each acquisition to
   its newest asset that actually contains pophys.nwb.zarr, and records the
   direct zarr path so downstream openers are unambiguous.

Writes outputs/session_inventory_openscope.csv with one row per acquisition.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mesoscope_qc_pipeline as qc  # noqa: E402

BUCKET = "aind-open-data"
PREFIX = "multiplane-ophys_"
ASSET_RE = re.compile(
    r"^multiplane-ophys_(?P<subject>\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})"
    r"(?:_(?P<kind>processed|nwb)_(?P<proc>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}))?$"
)


def list_assets() -> list[str]:
    """Every top-level asset folder under the multiplane-ophys prefix."""
    cl = qc.get_s3_client()
    pag = cl.get_paginator("list_objects_v2")
    out = []
    for page in pag.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            out.append(cp["Prefix"].rstrip("/"))
    return sorted(out)


def parse_asset(asset: str) -> dict | None:
    m = ASSET_RE.match(asset)
    if not m:
        return None
    d = m.groupdict()
    return {
        "asset": asset,
        "subject_id": d["subject"],
        "acq_date": d["date"],
        "acq_time": d["time"],
        "session_key": f"multiplane-ophys_{d['subject']}_{d['date']}_{d['time']}",
        "kind": d["kind"] or "raw",
        "processed_ts": d["proc"] or "",
    }


def read_project(asset: str) -> tuple[str, str, str]:
    """(project_name, investigators, modality) from an asset's data_description."""
    cl = qc.get_s3_client()
    try:
        obj = cl.get_object(Bucket=BUCKET, Key=f"{asset}/data_description.json")
        d = json.loads(obj["Body"].read())
        return (
            d.get("project_name") or "",
            ";".join(i.get("name", "") for i in (d.get("investigators") or [])),
            ";".join(m.get("abbreviation", "") for m in (d.get("modality") or [])),
        )
    except Exception as exc:
        return (f"ERR:{type(exc).__name__}", "", "")


def has_pophys(asset: str) -> bool:
    """Cheap existence check for the ophys zarr inside an asset."""
    cl = qc.get_s3_client()
    try:
        resp = cl.list_objects_v2(
            Bucket=BUCKET, Prefix=f"{asset}/pophys.nwb.zarr/", MaxKeys=1
        )
        return resp.get("KeyCount", 0) > 0
    except Exception:
        return False


def read_stim(asset: str) -> tuple[str, str]:
    """(session_type, stimulus script names) from an asset's session.json."""
    cl = qc.get_s3_client()
    try:
        obj = cl.get_object(Bucket=BUCKET, Key=f"{asset}/session.json")
        d = json.loads(obj["Body"].read())
        eps = d.get("stimulus_epochs") or []
        scripts = sorted({(e.get("script") or {}).get("name", "") for e in eps})
        return (d.get("session_type") or "", ";".join(s for s in scripts if s))
    except Exception:
        return ("", "")


def main() -> None:
    assets = list_assets()
    parsed = [p for p in (parse_asset(a) for a in assets) if p]
    df = pd.DataFrame(parsed)
    print(f"assets: {len(assets)} | parsed: {len(df)} | "
          f"acquisitions: {df.session_key.nunique()}", flush=True)

    # Project name is a property of the acquisition, so read it once per
    # acquisition from its newest asset rather than once per asset.
    rep = (df.sort_values("processed_ts").groupby("session_key")
             .tail(1)[["session_key", "asset"]])
    with ThreadPoolExecutor(max_workers=48) as ex:
        proj = list(ex.map(read_project, rep.asset.tolist()))
    rep = rep.assign(project_name=[p[0] for p in proj],
                     investigators=[p[1] for p in proj],
                     modality=[p[2] for p in proj]).drop(columns="asset")
    df = df.merge(rep, on="session_key", how="left")
    print("\nproject_name counts (per acquisition):", flush=True)
    print(rep.project_name.value_counts(dropna=False).to_string(), flush=True)

    os_assets = df[(df.project_name == "OpenScope") & (df.kind == "processed")].copy()
    print(f"\nOpenScope processed assets: {len(os_assets)} across "
          f"{os_assets.session_key.nunique()} acquisitions", flush=True)

    with ThreadPoolExecutor(max_workers=48) as ex:
        os_assets["has_pophys"] = list(ex.map(has_pophys, os_assets.asset.tolist()))
    print(f"assets carrying pophys.nwb.zarr: {int(os_assets.has_pophys.sum())}", flush=True)

    keep = (os_assets[os_assets.has_pophys]
            .sort_values("processed_ts")
            .groupby("session_key").tail(1)
            .sort_values(["acq_date", "acq_time"])
            .reset_index(drop=True))
    keep["s3_path"] = "s3://" + BUCKET + "/" + keep.asset + "/pophys.nwb.zarr"

    with ThreadPoolExecutor(max_workers=48) as ex:
        stim = list(ex.map(read_stim, keep.asset.tolist()))
    keep["session_type"] = [s[0] for s in stim]
    keep["stim_script"] = [s[1] for s in stim]

    out = Path("outputs/session_inventory_openscope.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out, index=False)

    print(f"\n=== OpenScope sessions with ophys data: {len(keep)} ===", flush=True)
    print(f"subjects: {keep.subject_id.nunique()} | "
          f"dates: {keep.acq_date.min()} -> {keep.acq_date.max()}", flush=True)
    print("\nper-month:", flush=True)
    print(keep.groupby(keep.acq_date.str[:7]).size().to_string(), flush=True)
    print("\nstim_script:", flush=True)
    print(keep.stim_script.value_counts().head(20).to_string(), flush=True)
    print("\nsession_type:", flush=True)
    print(keep.session_type.value_counts().head(20).to_string(), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
