"""
zdrift_plots.py
---------------
Displacement overviews and cross-plane correlation matrices for the sessions
at the extremes of the z-drift distribution.

Selects the N sessions with the largest and smallest axial drift from plane_df,
loads their raw motion-correction CSVs from S3, and produces two figures per
session: a three-panel displacement overview and a cross-plane correlation matrix.

Typical usage
-------------
    import zdrift_plots

    zdrift_plots.run(plane_df, all_paths, n=5,
                     save_dir="outputs/stage2/figures/zdrift_examples")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from motion_loader import displacement_matrix, load_motion_data
from s3_utils import parse_session_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGING_RATE = 9.48

plt.rcParams.update({
    "figure.dpi"        : 130,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "font.size"         : 10,
})


# ---------------------------------------------------------------------------
# Session selection
# ---------------------------------------------------------------------------

def select_sessions(
    plane_df: pd.DataFrame,
    n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the top-n and bottom-n sessions ranked by maximum z_drift_um
    across planes.

    Parameters
    ----------
    plane_df : from motion_qc.run()
    n        : number of sessions at each extreme

    Returns
    -------
    (worst_df, best_df) — each has columns session_id, max_z_drift_um
    """
    ok = (plane_df[plane_df["error"].isna()].copy()
          if "error" in plane_df.columns else plane_df.copy())
    ok = ok.dropna(subset=["z_drift_um"])

    per_session = (
        ok.groupby("session_id")["z_drift_um"]
        .max()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"z_drift_um": "max_z_drift_um"})
    )

    worst = per_session.head(n).copy()
    best  = per_session.tail(n).copy()

    print(f"Worst {n} z-drift sessions:")
    print(worst.to_string(index=False))
    print(f"\nBest {n} z-drift sessions:")
    print(best.to_string(index=False))

    return worst, best


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def _plot_displacement_overview(
    motion_data: dict[str, Any],
    z_drift_um: float,
    tag: str,
) -> plt.Figure:
    """Three-panel displacement overview annotated with z-drift and tag."""
    disp_mat, planes = displacement_matrix(motion_data)
    sid = motion_data.get("session_id", "")

    T         = disp_mat.shape[1]
    t         = np.arange(T) / IMAGING_RATE
    consensus = np.median(disp_mat, axis=0)
    n_planes  = len(planes)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"[{tag.upper()}]  {sid}  —  max z-drift = {z_drift_um:.1f} µm",
        fontsize=12, fontweight="bold",
    )

    for i, plane in enumerate(planes):
        axes[0].plot(t, disp_mat[i], lw=0.4, alpha=0.6, label=plane)
    axes[0].plot(t, consensus, lw=1.2, color="black", label="consensus", zorder=10)
    axes[0].set_ylabel("Displacement (µm)")
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title("Per-plane displacement + consensus")

    im = axes[1].imshow(
        disp_mat, aspect="auto", interpolation="none",
        extent=[0, t[-1], 0, n_planes], cmap="hot",
        vmin=0, vmax=np.percentile(disp_mat, 99),
    )
    axes[1].set_yticks(np.arange(n_planes) + 0.5)
    axes[1].set_yticklabels(planes, fontsize=7)
    axes[1].set_ylabel("Plane")
    axes[1].set_title("Displacement heatmap (hot = large motion)")
    plt.colorbar(im, ax=axes[1], label="µm")

    axes[2].plot(t, consensus, lw=0.6, color="steelblue")
    axes[2].set_ylabel("Consensus displacement (µm)")
    axes[2].set_title("Session-level motion proxy (median across planes)")

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.xaxis.set_tick_params(labelbottom=True)

    plt.tight_layout()
    return fig


def _plot_cross_plane_corr(
    motion_data: dict[str, Any],
    z_drift_um: float,
    tag: str,
) -> plt.Figure:
    """Cross-plane displacement correlation matrix for one case-study session."""
    disp_mat, planes = displacement_matrix(motion_data)
    sid = motion_data.get("session_id", "")
    n   = len(planes)

    corr = np.corrcoef(disp_mat)
    off  = corr[np.triu_indices(n, k=1)].mean() if n > 1 else np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        f"[{tag.upper()}]  {sid}  —  max z-drift = {z_drift_um:.1f} µm",
        fontsize=11, fontweight="bold",
    )
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(planes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(planes, fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Cross-plane displacement correlation\n"
                 f"mean off-diagonal = {off:.3f}")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-session runner
# ---------------------------------------------------------------------------

def plot_session(
    s3_path: str,
    z_drift_um: float,
    tag: str,
    save_dir: Path,
) -> dict[str, plt.Figure]:
    """
    Load motion data and produce both figures for one case-study session.

    Parameters
    ----------
    s3_path    : s3:// URL for the session asset
    z_drift_um : max z-drift for title annotation
    tag        : "worst" or "best"
    save_dir   : directory for PNG outputs

    Returns
    -------
    {"displacement": fig, "cross_plane_corr": fig}
    """
    session_id = parse_session_id(s3_path)
    safe_sid   = session_id.replace("/", "_")

    print(f"  loading {session_id} ...", end=" ", flush=True)
    motion_data = load_motion_data(s3_path, verbose=False)

    if not motion_data["plane_names"]:
        print("✗  (no planes loaded)")
        return {}

    print(f"✓  ({len(motion_data['plane_names'])} planes, "
          f"{len(list(motion_data['planes'].values())[0])} frames)")

    save_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, plt.Figure] = {}

    fig1 = _plot_displacement_overview(motion_data, z_drift_um, tag)
    path1 = save_dir / f"{safe_sid}_displacement.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    figs["displacement"] = fig1
    plt.close(fig1)

    fig2 = _plot_cross_plane_corr(motion_data, z_drift_um, tag)
    path2 = save_dir / f"{safe_sid}_cross_plane_corr.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    figs["cross_plane_corr"] = fig2
    plt.close(fig2)

    return figs


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run(
    plane_df: pd.DataFrame,
    all_paths: list[str],
    n: int = 5,
    save_dir: str | Path = "outputs/stage2/figures/zdrift_examples",
) -> dict[str, dict[str, plt.Figure]]:
    """
    Select worst and best z-drift sessions and produce case-study figures.

    Parameters
    ----------
    plane_df  : from motion_qc.run()
    all_paths : full list of s3:// session paths
    n         : number of sessions at each extreme
    save_dir  : root output directory

    Returns
    -------
    Nested dict: {session_id: {"displacement": fig, "cross_plane_corr": fig}}
    """
    save_dir = Path(save_dir)
    worst_df, best_df = select_sessions(plane_df, n=n)

    path_map  = {parse_session_id(p): p for p in all_paths}
    all_figs: dict[str, dict[str, plt.Figure]] = {}

    for tag, df in [("worst", worst_df), ("best", best_df)]:
        sub_dir = save_dir / tag
        print(f"\n── {tag.upper()} z-drift sessions ──────────────────────")
        for _, row in df.iterrows():
            sid     = row["session_id"]
            drift   = row["max_z_drift_um"]
            s3_path = path_map.get(sid)
            if s3_path is None:
                print(f"  {sid}: path not found")
                continue
            figs = plot_session(s3_path, drift, tag=tag, save_dir=sub_dir)
            all_figs[sid] = figs

    print(f"\nDone. Figures saved under {save_dir}/worst/ and {save_dir}/best/")
    return all_figs
