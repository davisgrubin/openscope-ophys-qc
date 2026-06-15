#!/usr/bin/env python3
"""Generate a one-plane FOV comparison PDF across all sessions from a given day."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import session_loader as sl
from mesoscope_qc_reports import DEFAULT_MAX_FRAMES, DEFAULT_MASK_LIMIT, build_session_report_context, plot_plane_fov_overlay


def _session_date_matches(session_path: str, date_text: str) -> bool:
    return f"_{date_text}_" in session_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Session acquisition date in YYYY-MM-DD format.")
    parser.add_argument("--plane", required=True, help="Plane name to compare, e.g. VISp_0.")
    parser.add_argument("--output", default="outputs/qc_summaries/daily_plane_fov_comparison.pdf", type=Path)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--max-rois", type=int, default=DEFAULT_MASK_LIMIT)
    parser.add_argument("--bucket", default="aind-open-data")
    parser.add_argument("--prefix", default="multiplane-ophys_")
    parser.add_argument("--deduplicate", action="store_true", default=True)
    parser.add_argument("--no-deduplicate", action="store_false", dest="deduplicate")
    args = parser.parse_args()

    sessions = sl.discover_sessions(bucket=args.bucket, prefix=args.prefix, deduplicate=args.deduplicate)
    matches = [path for path in sessions if _session_date_matches(path, args.date)]
    if not matches:
        raise SystemExit(f"No sessions found for {args.date}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        title = plt.figure(figsize=(8.5, 11))
        title.suptitle(f"Daily plane FOV comparison: {args.date} | {args.plane}", fontsize=18, fontweight="bold", y=0.94)
        ax = title.add_axes([0.10, 0.08, 0.80, 0.78])
        ax.axis("off")
        ax.text(
            0.0,
            1.0,
            "\n".join([
                f"Matching sessions: {len(matches)}",
                f"Plane: {args.plane}",
                f"Max frames per session: {args.max_frames}",
                f"Mask cap per session: {args.max_rois}",
            ]),
            va="top",
            ha="left",
            fontsize=11,
        )
        pdf.savefig(title)
        plt.close(title)

        for session_source in matches:
            try:
                ctx = build_session_report_context(
                    session_source=session_source,
                    max_frames=args.max_frames,
                    max_rois=args.max_rois,
                    plane_names=[args.plane],
                )
            except KeyError:
                continue
            if args.plane not in ctx.planes:
                continue
            fig = plot_plane_fov_overlay(ctx, args.plane)
            fig.suptitle(f"{args.date} | {str(session_source).split('/')[-1]} | {args.plane}", fontsize=13, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
