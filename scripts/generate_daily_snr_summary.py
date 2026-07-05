#!/usr/bin/env python
"""Generate compact ROI SNR summaries for sessions from one day."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mesoscope_daily_snr import (  # noqa: E402
    DEFAULT_DOWNLOAD_DIR,
    SnrBatchConfig,
    read_session_list,
    run_daily_summary,
)


def _parse_planes(value: str):
    if value == "all":
        return "all"
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True, help="Session day, YYYY-MM-DD.")
    parser.add_argument(
        "--session-source",
        action="append",
        default=[],
        help="Session source path/S3 URI. Repeat for multiple sessions.",
    )
    parser.add_argument(
        "--session-list",
        help="Text file with one session source path/S3 URI per line.",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Local directory to search for *DAY*.nwb when no session sources are provided.",
    )
    parser.add_argument("--output-root", default="outputs/snr_metrics_by_day")
    parser.add_argument("--planes", default="all", help="'all' or comma-separated plane names.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Frame limit. Omit for full sessions.",
    )
    parser.add_argument("--gaussian-sigma", type=float, default=3.0)
    parser.add_argument("--consecutive-samples", type=int, default=5)
    parser.add_argument("--event-threshold", type=float, default=0.0)
    parser.add_argument("--baseline-bins", type=int, default=10)
    parser.add_argument("--kernel-pre-s", type=float, default=0.5)
    parser.add_argument("--kernel-post-s", type=float, default=2.0)
    parser.add_argument("--max-kernel-events", type=int, default=500)
    parser.add_argument("--min-events-for-distribution-fit", type=int, default=20)
    args = parser.parse_args()

    session_sources = list(args.session_source)
    if args.session_list:
        session_sources.extend(read_session_list(args.session_list))

    config = SnrBatchConfig(
        day=args.day,
        planes=_parse_planes(args.planes),
        max_frames=args.max_frames,
        gaussian_sigma=args.gaussian_sigma,
        consecutive_samples=args.consecutive_samples,
        event_threshold=args.event_threshold,
        baseline_bins=args.baseline_bins,
        kernel_pre_s=args.kernel_pre_s,
        kernel_post_s=args.kernel_post_s,
        max_kernel_events=args.max_kernel_events,
        min_events_for_distribution_fit=args.min_events_for_distribution_fit,
    )
    result = run_daily_summary(
        args.day,
        session_sources=session_sources or None,
        search_dirs=args.search_dir or (DEFAULT_DOWNLOAD_DIR,),
        output_root=args.output_root,
        config=config,
    )
    print(json.dumps(result["manifest"], indent=2))


if __name__ == "__main__":
    main()
