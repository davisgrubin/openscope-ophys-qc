#!/usr/bin/env python3
"""Create chained Slurm jobs for compact event/background QC across sessions."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PYTHON = Path("/storage/project/r-fnajafi3-0/grubin6/.conda/envs/mesoscope_qc/bin/python")
DEFAULT_DOWNLOAD_DIR = Path("/storage/scratch1/3/grubin6/openscope_ophys_qc_dandi_downloads")
DEFAULT_MATERIALIZED_ROOT = Path("/storage/scratch1/3/grubin6/openscope_ophys_qc_materialized_batch")
DEFAULT_OUTPUT_ROOT = Path("/storage/scratch1/3/grubin6/openscope_ophys_qc_event_background_batch")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120]


def read_session_sources(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_sessions(day: str | None, search_dir: Path) -> list[str]:
    pattern = "*.nwb" if day is None else f"*{day}*.nwb"
    return [str(path.resolve()) for path in sorted(search_dir.expanduser().glob(pattern))]


def job_text(
    *,
    session_source: str,
    repo_root: Path,
    python: Path,
    materialized_root: Path,
    output_root: Path,
    max_frames: int,
    time_limit: str,
    qos: str,
    cpus: int,
    mem: str,
    logs_dir: Path,
) -> str:
    stem = safe_name(Path(session_source.rstrip("/")).stem or session_source)
    return f"""#!/bin/bash
#SBATCH --job-name=evbg_{stem[:40]}
#SBATCH --qos={qos}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --output={logs_dir / (stem + ".%j.out")}
#SBATCH --error={logs_dir / (stem + ".%j.err")}

set -euo pipefail
cd {repo_root}

{python} scripts/run_event_background_qc_session.py \\
  --session-source {session_source!r} \\
  --materialized-root {str(materialized_root)!r} \\
  --output-root {str(output_root)!r} \\
  --max-frames {max_frames}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-list", type=Path)
    parser.add_argument("--day", help="Discover local NWBs containing this YYYY-MM-DD date.")
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--jobs-dir", type=Path, default=Path("outputs/slurm_event_background_qc"))
    parser.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--time", default="00:55:00")
    parser.add_argument("--qos", default="embers")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="32G")
    args = parser.parse_args()

    if args.session_list:
        sessions = read_session_sources(args.session_list)
    else:
        sessions = discover_sessions(args.day, args.search_dir)
    if not sessions:
        raise SystemExit("No session sources found. Pass --session-list or --day.")

    repo_root = Path(__file__).resolve().parents[1]
    jobs_dir = args.jobs_dir.expanduser()
    logs_dir = jobs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    job_paths = []
    for index, session_source in enumerate(sessions, start=1):
        stem = safe_name(f"{index:03d}_{Path(session_source.rstrip('/')).stem or session_source}")
        path = jobs_dir / f"{stem}.sh"
        path.write_text(
            job_text(
                session_source=session_source,
                repo_root=repo_root,
                python=args.python,
                materialized_root=args.materialized_root,
                output_root=args.output_root,
                max_frames=args.max_frames,
                time_limit=args.time,
                qos=args.qos,
                cpus=args.cpus,
                mem=args.mem,
                logs_dir=logs_dir,
            ),
            encoding="utf-8",
            newline="\n",
        )
        path.chmod(0o755)
        job_paths.append(path)

    submit = jobs_dir / "submit_chained_jobs.sh"
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "# This submits one session at a time. Each later job starts only after the",
        "# previous session completed successfully. Review the generated job scripts",
        "# before running this file.",
        "previous_job_id=\"\"",
    ]
    for path in job_paths:
        rel = path.relative_to(jobs_dir)
        lines.extend(
            [
                "if [[ -z \"${previous_job_id}\" ]]; then",
                f"  previous_job_id=$(sbatch --parsable {rel})",
                "else",
                f"  previous_job_id=$(sbatch --parsable --dependency=afterok:${{previous_job_id}} {rel})",
                "fi",
                f"echo \"submitted {rel} as ${{previous_job_id}}\"",
            ]
        )
    submit.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    submit.chmod(0o755)

    print(f"Wrote {len(job_paths)} job scripts under {jobs_dir}")
    print(f"Review, then submit with: bash {submit}")
    print("No jobs were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
