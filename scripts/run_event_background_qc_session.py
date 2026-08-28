#!/usr/bin/env python3
"""Materialize one mesoscope session and compute compact event/background QC."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_event_background_figure_qc import build_figure_metrics  # noqa: E402
from materialize_session_cache import materialize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-source", required=True)
    parser.add_argument("--materialized-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--full-timeseries", action="store_true")
    parser.add_argument("--skip-timeseries", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    max_frames = None if args.full_timeseries else args.max_frames
    summary = materialize(
        args.session_source,
        args.materialized_root.expanduser().resolve(),
        max_frames=max_frames,
        include_timeseries=not args.skip_timeseries,
    )
    session_dir = Path(summary["out_dir"])
    output_dir = args.output_root.expanduser().resolve() / session_dir.name
    nwb_source = Path(args.session_source).expanduser()
    nwb_source = nwb_source.resolve() if nwb_source.exists() else None
    metrics_csv, clusters_csv = build_figure_metrics(
        session_dir,
        output_dir,
        nwb_source=nwb_source,
        max_frames=max_frames,
    )
    elapsed_s = time.perf_counter() - start
    manifest = {
        "session_source": args.session_source,
        "session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_csv),
        "clusters_csv": str(clusters_csv),
        "max_frames": max_frames,
        "elapsed_s": elapsed_s,
        "elapsed_min": elapsed_s / 60,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "event_background_qc_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
