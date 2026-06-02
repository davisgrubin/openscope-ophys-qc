"""
scatter_plots.py
----------------
Scatter plots relating every displacement and registration quality metric
to ROI classification targets (mean soma probability, mean dendrite probability).

One figure per classification target; each figure shows all displacement metrics
in a grid. Points are colored by brain area (VISl=blue, VISp=red). Pearson r
and p-value are annotated in the corner of each panel.

Typical usage
-------------
    import scatter_plots

    merged_df = scatter_plots.merge(plane_df, summary_df)
    figs = scatter_plots.plot_all(merged_df,
                                  save_dir="outputs/stage2/figures/scatter")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AREA_COLOR = {"VISl": "#2196F3", "VISp": "#F44336"}

# Motion metrics to place on the x-axis of each panel
DISPLACEMENT_METRICS: list[tuple[str, str]] = [
    ("mean_disp_um",                   "Mean displacement (µm)"),
    ("p95_disp_um",                    "P95 displacement (µm)"),
    ("p99_disp_um",                    "P99 displacement (µm)"),
    ("displacement_cv",                "Displacement CV"),
    ("bad_frame_frac",                 "Bad frame fraction"),
    ("invalid_frame_frac",             "Invalid frame fraction"),
    ("mean_reg_corr",                  "Mean reg. correlation"),
    ("median_reg_corr",                "Median reg. correlation"),
    ("displacement_slope_um_per_min",  "Displacement slope (µm/min)"),
    ("n_motion_bursts",                "Motion burst count"),
    ("z_drift_um",                     "Z-drift (µm)"),
    ("intensity_stability_pct",        "Intensity stability (%)"),
]

# ROI targets to place on the y-axis
ROI_TARGETS: list[tuple[str, str, str]] = [
    ("mean_soma_prob",     "Mean soma probability",     "soma_prob"),
    ("mean_dendrite_prob", "Mean dendrite probability", "dendrite_prob"),
]

plt.rcParams.update({
    "figure.dpi"        : 130,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "font.size"         : 9,
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def merge(
    motion_df: pd.DataFrame,
    roi_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the motion QC plane DataFrame with the ROI plane summary on
    session_id × plane, producing one row per session × plane with both
    motion metrics and ROI classification scores present.

    Parameters
    ----------
    motion_df      : from motion_qc.run()
    roi_summary_df : from roi_classifier.run() — the summary CSV

    Returns
    -------
    Merged DataFrame.
    """
    roi_cols = [
        "session_id", "plane",
        "n_rois",
        "soma_frac", "dendrite_frac",
        "mean_soma_prob", "median_soma_prob",
        "mean_dendrite_prob", "median_dendrite_prob",
    ]
    roi_cols_present = [c for c in roi_cols if c in roi_summary_df.columns]

    # Drop any pre-existing ROI columns from motion_df to avoid conflicts
    drop = [c for c in roi_cols_present if c not in ("session_id", "plane")
            and c in motion_df.columns]
    clean_motion = motion_df.drop(columns=drop, errors="ignore")

    return clean_motion.merge(
        roi_summary_df[roi_cols_present],
        on=["session_id", "plane"],
        how="left",
    )


# ---------------------------------------------------------------------------
# Panel helper
# ---------------------------------------------------------------------------

def _scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
) -> None:
    sub = df[[x_col, y_col, "structure"]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    sub["structure"] = sub["structure"].fillna("unknown")

    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#aaa")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        return

    for struct, grp in sub.groupby("structure"):
        c = AREA_COLOR.get(str(struct), "#888")
        ax.scatter(grp[x_col], grp[y_col],
                   color=c, alpha=0.65, s=22,
                   edgecolors="white", linewidths=0.4,
                   label=str(struct), zorder=3)

    can_corr = (
        len(sub) >= 2
        and sub[x_col].nunique() >= 2
        and sub[y_col].nunique() >= 2
    )

    if can_corr:
        r, p    = stats.pearsonr(sub[x_col], sub[y_col])
        p_str   = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        color_r = "#C62828" if p < 0.05 else "#555"

        ax.text(0.97, 0.97, f"r={r:+.2f}\n{p_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=color_r,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))

        if len(sub) >= 3:
            x_arr = sub[x_col].to_numpy(float)
            y_arr = sub[y_col].to_numpy(float)
            if np.isfinite(x_arr).all() and np.isfinite(y_arr).all():
                m, b   = np.polyfit(x_arr, y_arr, 1)
                x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
                lc     = "#C62828" if p < 0.05 else "#aaa"
                ax.plot(x_line, m * x_line + b, color=lc,
                        lw=1.2, linestyle="--", zorder=2, alpha=0.8)
    else:
        ax.text(0.97, 0.97, "r=n/a",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="#777",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)


# ---------------------------------------------------------------------------
# Per-target figure
# ---------------------------------------------------------------------------

def plot_target(
    plane_df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    One figure: grid of scatter plots for all displacement metrics vs one
    ROI classification target.

    Parameters
    ----------
    plane_df  : merged DataFrame from scatter_plots.merge()
    y_col     : ROI classification column name
    ylabel    : y-axis label
    save_path : optional PNG output path
    """
    ok      = (plane_df[plane_df["error"].isna()].copy()
               if "error" in plane_df.columns else plane_df.copy())
    metrics = [(col, lbl) for col, lbl in DISPLACEMENT_METRICS if col in ok.columns]

    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.2, nrows * 3.8),
                              squeeze=False)
    fig.suptitle(f"Displacement metrics  vs  {ylabel}",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, (x_col, xlabel) in enumerate(metrics):
        _scatter_panel(axes[idx // ncols][idx % ncols],
                       ok, x_col, y_col, xlabel, ylabel)

    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=7, label=area)
        for area, c in AREA_COLOR.items()
    ]
    fig.legend(handles=handles, title="Area", title_fontsize=9, fontsize=9,
               loc="upper right", bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def plot_all(
    plane_df: pd.DataFrame,
    save_dir: str | Path = "outputs/stage2/figures/scatter",
) -> dict[str, plt.Figure]:
    """
    Produce one figure per ROI classification target and save PNGs.

    Parameters
    ----------
    plane_df : merged DataFrame from scatter_plots.merge()
    save_dir : directory for PNG outputs

    Returns
    -------
    dict of {target_col: figure}
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    missing = [col for col, _, _ in ROI_TARGETS if col not in plane_df.columns]
    if missing:
        print(
            f"Missing ROI columns: {missing}\n"
            f"Call scatter_plots.merge(plane_df, summary_df) first."
        )
        return {}

    figures: dict[str, plt.Figure] = {}
    print(f"Saving scatter figures to {save_dir}/\n")

    for y_col, ylabel, stem in ROI_TARGETS:
        print(f"  {y_col} ...", end=" ", flush=True)
        try:
            fname = save_dir / f"scatter_{stem}.png"
            fig   = plot_target(plane_df, y_col, ylabel, save_path=fname)
            figures[y_col] = fig
            plt.close(fig)
            print("✓")
        except Exception as e:
            print(f"✗  ({e})")

    print(f"\nDone. {len(figures)}/{len(ROI_TARGETS)} figures saved.")
    return figures
