#!/usr/bin/env python3
"""Generate session- and ROI-level QC summary PDFs for a mesoscope session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mesoscope_qc_reports import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MASK_LIMIT,
    DEFAULT_NEURONS_PER_PLANE,
    DEFAULT_SESSION_SOURCE,
    write_qc_summaries,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-source", default=str(DEFAULT_SESSION_SOURCE))
    parser.add_argument("--output-dir", default="outputs/qc_summaries", type=Path)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--max-rois", type=int, default=DEFAULT_MASK_LIMIT)
    parser.add_argument("--neurons-per-plane", type=int, default=DEFAULT_NEURONS_PER_PLANE)
    parser.add_argument("--planes", nargs="*", default=None)
    args = parser.parse_args()

    session_pdf, roi_pdf = write_qc_summaries(
        session_source=args.session_source,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        max_rois=args.max_rois,
        neurons_per_plane=args.neurons_per_plane,
        plane_names=args.planes,
    )
    print(session_pdf)
    print(roi_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
