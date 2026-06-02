"""
motion_plots.py
---------------
Visualization for motion QC data, covering both single-session and
multi-session outputs.

Single-session functions accept the dict returned by motion_loader.load_motion_data().
Multi-session functions accept the DataFrames returned by motion_qc.run().

Every function produces one independent figure and returns it.
Call plt.show() or plt.close() in the notebook as appropriate.

Typical usage
-------------
    import motion_plots as mp

    # Single session
    mp.plot_displacement_traces(motion_data)
    mp.plot_cross_plane_corr(motion_data)

    # Multi-session
    figs = mp.plot_all(plane_df, session_df,
                       save_dir="outputs/stage2/figures/motion")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from motion_loader import displacement_matrix

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#FF5722", "#607D8B",
]

PLANE_ORDER = [
    "VISl_4", "VISl_5", "VISl_6", "VISl_7",
    "VISp_0", "VISp_1", "VISp_2", "VISp_3",
]

AREA_COLOR = {"VISl": "#2196F3", "VISp": "#F44336"}

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
# Internal helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _subject_colors(df: pd.DataFrame) -> dict[str, str]:
    subjects = sorted(df["subject_id"].dropna().unique())
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(subjects)}


def _parse_date(session_id: str) -> pd.Timestamp | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(session_id))
    try:
        return pd.Timestamp(m.group(1)) if m else None
    except Exception:
        return None


def _ok_plane(plane_df: pd.DataFrame) -> pd.DataFrame:
    return (
        plane_df[plane_df["error"].isna()].copy()
        if "error" in plane_df.columns
        else plane_df.copy()
    )


def _ok_session(session_df: pd.DataFrame) -> pd.DataFrame:
    return (
        session_df[session_df["error"].isna()].copy()
        if "error" in session_df.columns
        else session_df.copy()
    )


def _present_planes(ok: pd.DataFrame) -> list[str]:
    return [p for p in PLANE_ORDER if p in ok["plane"].values]


def _boxplot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    plane_col: str = "plane",
    hline: float | None = None,
    hline_label: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Core boxplot: one box per plane, jittered session points overlaid.
    Grand median (dashed black) and mean (dotted grey) drawn as reference lines.
    Per-plane mean shown as a diamond marker.
    Blue boxes = VISl, red boxes = VISp.
    """
    planes = [p for p in PLANE_ORDER if p in df[plane_col].values]
    if not planes:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return fig

    data   = [df.loc[df[plane_col] == p, metric].dropna().values for p in planes]
    colors = [AREA_COLOR.get(p.split("_")[0], "#999") for p in planes]

    fig, ax = plt.subplots(figsize=(max(8, len(planes) * 1.1), 5))

    bp = ax.boxplot(
        data, patch_artist=True,
        medianprops=dict(color="black", lw=0),
        whiskerprops=dict(lw=1.3, color="#444"),
        capprops=dict(lw=1.3, color="#444"),
        flierprops=dict(marker="o", markersize=3, alpha=0.35, color="#888"),
        widths=0.52, zorder=2,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for xi, (vals, c) in enumerate(zip(data, colors), start=1):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.20, 0.20, size=len(vals))
        ax.scatter(xi + jitter, vals, color=c, alpha=0.65, s=22,
                   edgecolors="white", linewidths=0.4, zorder=4)

    all_vals     = np.concatenate([v for v in data if len(v) > 0])
    grand_median = float(np.median(all_vals))
    grand_mean   = float(np.mean(all_vals))

    ax.axhline(grand_median, color="black", lw=1.4, linestyle="--",
               label=f"grand median = {grand_median:.3g}", zorder=5)
    ax.axhline(grand_mean, color="#555", lw=1.4, linestyle=":",
               label=f"grand mean   = {grand_mean:.3g}", zorder=5)

    for xi, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.scatter(xi, float(np.mean(vals)), marker="D", color="black",
                   s=28, zorder=6, edgecolors="white", linewidths=0.6)

    if hline is not None:
        ax.axhline(hline, color="#E53935", lw=1.4, linestyle="-.",
                   label=hline_label or f"threshold = {hline:.3g}", zorder=5)

    n_visl = sum(1 for p in planes if "VISl" in p)
    if 0 < n_visl < len(planes):
        ax.axvline(n_visl + 0.5, color="#aaa", lw=1.0, linestyle=":", zorder=1)
        ax.text((n_visl * 0.5 + 0.5) / len(planes), 1.01, "VISl",
                ha="center", va="bottom", fontsize=9,
                color=AREA_COLOR["VISl"], fontweight="bold",
                transform=ax.transAxes)
        ax.text((n_visl + (len(planes) - n_visl) * 0.5 + 0.5) / len(planes), 1.01,
                "VISp", ha="center", va="bottom", fontsize=9,
                color=AREA_COLOR["VISp"], fontweight="bold",
                transform=ax.transAxes)

    ax.set_xticks(range(1, len(planes) + 1))
    ax.set_xticklabels(planes, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=32)
    ax.scatter([], [], marker="D", color="black", s=28, label="per-plane mean")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        _savefig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Single-session figures
# ---------------------------------------------------------------------------

def plot_displacement_traces(
    motion_data: dict[str, Any],
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """
    Three-panel displacement overview for one session:
    per-plane traces, displacement heatmap, and consensus trace.
    """
    disp_mat, planes = displacement_matrix(motion_data)
    if disp_mat.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No motion data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    T         = disp_mat.shape[1]
    t         = np.arange(T) / IMAGING_RATE
    consensus = np.median(disp_mat, axis=0)
    n_planes  = len(planes)
    sid       = motion_data.get("session_id", "")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f"{sid} — displacement overview", fontsize=12, fontweight="bold")

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
    if save_dir:
        _savefig(fig, Path(save_dir) / "s1_displacement_traces.png")
    return fig


def plot_cross_plane_corr(
    motion_data: dict[str, Any],
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """Cross-plane displacement correlation matrix for one session."""
    disp_mat, planes = displacement_matrix(motion_data)
    if disp_mat.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No motion data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n    = len(planes)
    corr = np.corrcoef(disp_mat)
    off  = corr[np.triu_indices(n, k=1)].mean() if n > 1 else np.nan
    sid  = motion_data.get("session_id", "")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(planes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(planes, fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(
        f"{sid}\nCross-plane displacement correlation  "
        f"(mean off-diagonal = {off:.3f})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "s1_cross_plane_corr.png")
    return fig


# ---------------------------------------------------------------------------
# Multi-session: session-level figures
# ---------------------------------------------------------------------------

def plot_median_vs_p95(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok  = _ok_session(session_df)
    col = _subject_colors(session_df)

    fig, ax = plt.subplots(figsize=(7, 5))
    for subj, grp in ok.groupby("subject_id"):
        ax.scatter(
            grp["session_median_disp_um_median"],
            grp["session_p95_disp_um_median"],
            color=col.get(str(subj), "#999"),
            label=str(subj), alpha=0.8, s=60,
            edgecolors="white", linewidths=0.5,
        )
    lim = max(
        ok["session_p95_disp_um_median"].max(),
        ok["session_median_disp_um_median"].max(),
    ) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="y = x")
    ax.set_xlabel("Median displacement (µm)")
    ax.set_ylabel("95th percentile displacement (µm)")
    ax.set_title("Median vs P95 displacement", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, title="Subject", title_fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "01_median_vs_p95.png")
    return fig


def plot_bad_frame_frac_hist(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok   = _ok_session(session_df)
    vals = ok["session_bad_frame_frac_median"].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=20, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.axvline(vals.median(), color="#F44336", lw=1.5, linestyle="--",
               label=f"median = {vals.median():.3f}")
    ax.set_xlabel("Bad frame fraction (displacement > 5 µm)")
    ax.set_ylabel("Sessions")
    ax.set_title("Bad frame fraction distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "02_bad_frame_frac_hist.png")
    return fig


def plot_displacement_slope_bar(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok   = _ok_session(session_df)
    vals = ok["session_displacement_slope_um_per_min_median"].dropna().sort_values()
    pos  = (vals > 0).sum()
    neg  = (vals <= 0).sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(vals)), vals.values,
           color=["#F44336" if v > 0 else "#4CAF50" for v in vals],
           width=0.8, edgecolor="none")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Session (sorted by slope)")
    ax.set_ylabel("Displacement slope (µm / min)")
    ax.set_title(f"Displacement trend  ↑{pos} worsening  ↓{neg} improving",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "03_displacement_slope_bar.png")
    return fig


def plot_settling_time_vs_early(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok  = _ok_session(session_df)
    col = _subject_colors(session_df)

    fig, ax = plt.subplots(figsize=(7, 5))
    for subj, grp in ok.groupby("subject_id"):
        ax.scatter(
            grp["session_first_5min_median_um_median"],
            grp["session_settling_time_s_median"],
            color=col.get(str(subj), "#999"),
            label=str(subj), alpha=0.8, s=60,
            edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel("First 5-min median displacement (µm)")
    ax.set_ylabel("Settling time (s)")
    ax.set_title("Early displacement vs settling time",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, title="Subject", title_fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "04_settling_time_vs_early.png")
    return fig


def plot_early_vs_late(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok  = _ok_session(session_df)
    col = _subject_colors(session_df)
    ok2 = ok.dropna(subset=["session_first_5min_median_um_median",
                             "session_last_5min_median_um_median"])

    fig, ax = plt.subplots(figsize=(6, 6))
    for subj, grp in ok2.groupby("subject_id"):
        ax.scatter(
            grp["session_first_5min_median_um_median"],
            grp["session_last_5min_median_um_median"],
            color=col.get(str(subj), "#999"),
            label=str(subj), alpha=0.8, s=60,
            edgecolors="white", linewidths=0.5,
        )
    lim = max(
        ok2["session_first_5min_median_um_median"].max(),
        ok2["session_last_5min_median_um_median"].max(),
    ) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="no change")
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color="#F44336")
    ax.fill_between([0, lim], [0, 0],   [0, lim],   alpha=0.04, color="#4CAF50")
    ax.text(lim * 0.05, lim * 0.92, "Got worse ↑", color="#F44336", fontsize=9)
    ax.text(lim * 0.55, lim * 0.05, "Got better ↓", color="#4CAF50", fontsize=9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Early session median displacement (µm)")
    ax.set_ylabel("Late session median displacement (µm)")
    ax.set_title("Session stability: early vs late",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, title="Subject", title_fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "05_early_vs_late.png")
    return fig


def plot_invalid_frame_frac_per_session(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok   = _ok_session(session_df)
    vals = ok["session_invalid_frame_frac_median"].dropna().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(vals)), vals.values, color="#FF9800", edgecolor="none")
    ax.set_xlabel("Session (sorted)")
    ax.set_ylabel("Median invalid frame fraction")
    ax.set_title("Invalid frames per session  (suite2p flag)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "06_invalid_frame_frac_per_session.png")
    return fig


def plot_cross_plane_corr_bar(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok       = _ok_session(session_df)
    col      = _subject_colors(session_df)
    ok_sorted = ok.dropna(subset=["mean_cross_plane_corr"]).sort_values(
        "mean_cross_plane_corr"
    )
    bar_colors = [col.get(str(s), "#999") for s in ok_sorted["subject_id"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(ok_sorted)), ok_sorted["mean_cross_plane_corr"].values,
           color=bar_colors, edgecolor="none")
    ax.axhline(0.3, color="#F44336", lw=1.5, linestyle="--",
               label="outlier threshold (0.3)")
    ax.set_xlabel("Session (sorted by coherence)")
    ax.set_ylabel("Mean cross-plane correlation")
    ax.set_title("Cross-plane displacement coherence\n(high = motion is global)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=c, markersize=8, label=s)
        for s, c in col.items()
    ]
    handles.append(plt.Line2D([0], [0], color="#F44336", lw=1.5,
                               linestyle="--", label="outlier threshold"))
    ax.legend(handles=handles, fontsize=7, title="Subject", title_fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "07_cross_plane_corr_bar.png")
    return fig


def plot_outlier_plane_count(
    session_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    ok     = _ok_session(session_df)
    counts = ok["outlier_plane_count"].dropna().value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index.astype(int), counts.values,
           color="#E91E63", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of outlier planes per session")
    ax.set_ylabel("Sessions")
    ax.set_title("Outlier plane count\n(cross-plane corr < 0.3)",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "08_outlier_plane_count.png")
    return fig


# ---------------------------------------------------------------------------
# Multi-session: plane-level boxplot figures
# ---------------------------------------------------------------------------

def plot_displacement_by_plane(plane_df, save_dir=None):
    fig = _boxplot_metric(_ok_plane(plane_df), "median_disp_um",
                          "Median displacement (µm)",
                          "Median displacement by plane  (blue=VISl, red=VISp)",
                          save_path=Path(save_dir) / "09_displacement_by_plane.png" if save_dir else None)
    return fig

def plot_displacement_by_area(plane_df, save_dir=None):
    ok      = _ok_plane(plane_df)
    structs = sorted(ok["structure"].dropna().unique())
    data    = [ok.loc[ok["structure"] == s, "median_disp_um"].dropna().values for s in structs]
    colors  = ["#2196F3" if "VISl" in s else "#F44336" for s in structs]
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(structs) + 1)); ax.set_xticklabels(structs)
    ax.set_ylabel("Median displacement (µm)")
    ax.set_title("Displacement by brain area", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "10_displacement_by_area.png")
    return fig

def plot_bad_frame_frac_by_plane(plane_df, save_dir=None):
    return _boxplot_metric(_ok_plane(plane_df), "bad_frame_frac",
                           "Bad frame fraction (displacement > 5 µm)",
                           "Bad frame fraction by plane",
                           hline=0.05, hline_label="5% threshold",
                           save_path=Path(save_dir) / "11_bad_frame_frac_by_plane.png" if save_dir else None)

def plot_zdrift_by_plane(plane_df, save_dir=None):
    return _boxplot_metric(_ok_plane(plane_df), "z_drift_um",
                           "Z-drift (µm)",
                           "Z-drift distribution by plane",
                           hline=0.0, hline_label="no drift",
                           save_path=Path(save_dir) / "12_zdrift_by_plane.png" if save_dir else None)

def plot_zdrift_vs_lateral(plane_df, save_dir=None):
    ok = _ok_plane(plane_df).dropna(subset=["z_drift_um", "median_disp_um"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for struct, grp in ok.groupby("structure"):
        c = "#2196F3" if "VISl" in str(struct) else "#F44336"
        ax.scatter(grp["z_drift_um"], grp["median_disp_um"],
                   color=c, alpha=0.6, s=40, label=str(struct))
    ax.set_xlabel("Z-drift (µm)"); ax.set_ylabel("Median lateral displacement (µm)")
    ax.set_title("Z-drift vs lateral motion", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "13_zdrift_vs_lateral.png")
    return fig

def plot_reg_corr_by_plane(plane_df, save_dir=None):
    return _boxplot_metric(_ok_plane(plane_df), "mean_reg_corr",
                           "Mean registration correlation",
                           "Registration correlation by plane",
                           save_path=Path(save_dir) / "14_reg_corr_by_plane.png" if save_dir else None)

def plot_reg_corr_vs_displacement(plane_df, save_dir=None):
    ok = _ok_plane(plane_df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for struct, grp in ok.groupby("structure"):
        c = "#2196F3" if "VISl" in str(struct) else "#F44336"
        ax.scatter(grp["mean_reg_corr"], grp["median_disp_um"],
                   color=c, alpha=0.5, s=30, label=str(struct))
    ax.set_xlabel("Mean registration correlation")
    ax.set_ylabel("Median displacement (µm)")
    ax.set_title("Registration quality vs motion", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "15_reg_corr_vs_displacement.png")
    return fig

def plot_bursts_vs_bad_frame_frac(plane_df, save_dir=None):
    ok = _ok_plane(plane_df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for struct, grp in ok.groupby("structure"):
        c = "#2196F3" if "VISl" in str(struct) else "#F44336"
        ax.scatter(grp["n_motion_bursts"], grp["bad_frame_frac"],
                   color=c, alpha=0.5, s=30, label=str(struct))
    ax.set_xlabel("Number of motion bursts")
    ax.set_ylabel("Bad frame fraction")
    ax.set_title("Burst count vs contamination\n(few large vs many small events)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "16_bursts_vs_bad_frame_frac.png")
    return fig

def plot_longest_clean_run(plane_df, save_dir=None):
    ok   = _ok_plane(plane_df)
    vals = ok["longest_clean_run_s"].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals / 60, bins=25, color="#4CAF50", edgecolor="white", linewidth=0.5)
    ax.axvline(vals.median() / 60, color="#F44336", lw=1.5, linestyle="--",
               label=f"median = {vals.median()/60:.1f} min")
    ax.set_xlabel("Longest clean epoch (minutes)")
    ax.set_ylabel("Planes")
    ax.set_title("Longest continuous clean run\n(displacement ≤ 5 µm)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "17_longest_clean_run.png")
    return fig

def plot_bad_frame_frac_vs_reg_corr(plane_df, save_dir=None):
    ok = _ok_plane(plane_df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for struct, grp in ok.groupby("structure"):
        c = "#2196F3" if "VISl" in str(struct) else "#F44336"
        ax.scatter(grp["bad_frame_frac"], grp["mean_reg_corr"],
                   color=c, alpha=0.5, s=30, label=str(struct))
    ax.set_xlabel("Bad frame fraction")
    ax.set_ylabel("Mean registration correlation")
    ax.set_title("Motion contamination vs registration quality",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "18_bad_frame_frac_vs_reg_corr.png")
    return fig

def plot_session_plane_heatmap(plane_df, save_dir=None):
    ok    = _ok_plane(plane_df)
    pivot = ok.pivot_table(index="session_id", columns="plane",
                           values="median_disp_um", aggfunc="median")
    cols  = [p for p in PLANE_ORDER if p in pivot.columns]
    pivot = pivot[cols].loc[pivot.median(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="none",
                   vmin=0, vmax=np.nanpercentile(pivot.values, 95))
    plt.colorbar(im, ax=ax, label="Median displacement (µm)", shrink=0.6)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)

    def _short(sid):
        parts = str(sid).split("_")
        return f"{parts[1]}  {parts[2]}" if len(parts) >= 4 else str(sid)

    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([_short(s) for s in pivot.index], fontsize=7)
    ax.set_title("Session × plane displacement heatmap\n(sorted by median displacement)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        _savefig(fig, Path(save_dir) / "19_session_plane_heatmap.png")
    return fig

def plot_longitudinal_per_subject(plane_df, save_dir=None):
    ok = _ok_plane(plane_df)
    sess_med = (ok.groupby(["subject_id", "session_id"])["median_disp_um"]
                .median().reset_index())
    sess_med["date"] = sess_med["session_id"].apply(_parse_date)
    sess_med = sess_med.dropna(subset=["date"]).sort_values(["subject_id", "date"])
    col      = _subject_colors(sess_med)
    figures  = {}
    for subj in sorted(sess_med["subject_id"].dropna().unique()):
        grp = sess_med[sess_med["subject_id"] == subj].sort_values("date")
        c   = col.get(str(subj), "#2196F3")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(grp["date"], grp["median_disp_um"], marker="o", color=c, lw=1.5, markersize=5)
        ax.fill_between(grp["date"], grp["median_disp_um"], alpha=0.15, color=c)
        ax.set_title(f"Longitudinal motion stability — Subject {subj}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Median displacement (µm)")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        if save_dir:
            _savefig(fig, Path(save_dir) / f"20_longitudinal_{subj}.png")
        figures[str(subj)] = fig
    return figures


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def plot_all(
    plane_df: pd.DataFrame,
    session_df: pd.DataFrame,
    save_dir: str | Path = "outputs/stage2/figures/motion",
) -> dict[str, plt.Figure]:
    """
    Run all multi-session motion figures and save PNGs.

    Returns {name: figure}.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving motion QC figures to {save_dir}/\n")

    steps = [
        ("median_vs_p95",              lambda: plot_median_vs_p95(session_df, save_dir)),
        ("bad_frame_frac_hist",        lambda: plot_bad_frame_frac_hist(session_df, save_dir)),
        ("displacement_slope_bar",     lambda: plot_displacement_slope_bar(session_df, save_dir)),
        ("settling_time_vs_early",     lambda: plot_settling_time_vs_early(session_df, save_dir)),
        ("early_vs_late",              lambda: plot_early_vs_late(session_df, save_dir)),
        ("invalid_frame_frac",         lambda: plot_invalid_frame_frac_per_session(session_df, save_dir)),
        ("cross_plane_corr_bar",       lambda: plot_cross_plane_corr_bar(session_df, save_dir)),
        ("outlier_plane_count",        lambda: plot_outlier_plane_count(session_df, save_dir)),
        ("displacement_by_plane",      lambda: plot_displacement_by_plane(plane_df, save_dir)),
        ("displacement_by_area",       lambda: plot_displacement_by_area(plane_df, save_dir)),
        ("bad_frame_frac_by_plane",    lambda: plot_bad_frame_frac_by_plane(plane_df, save_dir)),
        ("zdrift_by_plane",            lambda: plot_zdrift_by_plane(plane_df, save_dir)),
        ("zdrift_vs_lateral",          lambda: plot_zdrift_vs_lateral(plane_df, save_dir)),
        ("reg_corr_by_plane",          lambda: plot_reg_corr_by_plane(plane_df, save_dir)),
        ("reg_corr_vs_displacement",   lambda: plot_reg_corr_vs_displacement(plane_df, save_dir)),
        ("bursts_vs_bad_frame_frac",   lambda: plot_bursts_vs_bad_frame_frac(plane_df, save_dir)),
        ("longest_clean_run",          lambda: plot_longest_clean_run(plane_df, save_dir)),
        ("bad_frame_frac_vs_reg_corr", lambda: plot_bad_frame_frac_vs_reg_corr(plane_df, save_dir)),
        ("session_plane_heatmap",      lambda: plot_session_plane_heatmap(plane_df, save_dir)),
    ]

    figures: dict[str, plt.Figure] = {}
    for name, fn in steps:
        print(f"  {name} ...", end=" ", flush=True)
        try:
            fig = fn()
            figures[name] = fig
            plt.close(fig)
            print("✓")
        except Exception as e:
            print(f"✗  ({e})")

    print(f"  longitudinal ...", end=" ", flush=True)
    try:
        long_figs = plot_longitudinal_per_subject(plane_df, save_dir)
        for subj, fig in long_figs.items():
            figures[f"longitudinal_{subj}"] = fig
            plt.close(fig)
        print(f"✓ ({len(long_figs)} subjects)")
    except Exception as e:
        print(f"✗  ({e})")

    print(f"\nDone. {len(figures)} figures saved.")
    return figures
