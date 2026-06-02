"""
roi_plots.py
------------
Boxplots of ROI classification metrics from the roi_plane_summary DataFrame.

One figure per metric. x-axis = imaging plane, y-axis = metric value,
each box = distribution across sessions for that plane.

Typical usage
-------------
    import roi_plots

    figs = roi_plots.plot_all(summary_df,
                              save_dir="outputs/stage2/figures/roi")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLANE_ORDER = [
    "VISl_4", "VISl_5", "VISl_6", "VISl_7",
    "VISp_0", "VISp_1", "VISp_2", "VISp_3",
]

AREA_COLOR = {"VISl": "#2196F3", "VISp": "#F44336"}

plt.rcParams.update({
    "figure.dpi"        : 130,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "font.size"         : 11,
})

# (metric_col, ylabel, title, hline, hline_label)
METRICS: list[tuple] = [
    ("n_rois",               "Number of ROIs",             "ROI yield per plane",                     None, None),
    ("soma_frac",            "Soma fraction",               "Fraction of ROIs classified as soma",     0.5,  "50%"),
    ("dendrite_frac",        "Dendrite fraction",           "Fraction of ROIs classified as dendrite", None, None),
    ("mean_soma_prob",       "Mean soma probability",       "Mean soma probability per plane",          None, None),
    ("median_soma_prob",     "Median soma probability",     "Median soma probability per plane",        None, None),
    ("mean_dendrite_prob",   "Mean dendrite probability",   "Mean dendrite probability per plane",      None, None),
    ("median_dendrite_prob", "Median dendrite probability", "Median dendrite probability per plane",    None, None),
]


# ---------------------------------------------------------------------------
# Core boxplot
# ---------------------------------------------------------------------------

def boxplot_metric(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    hline: float | None = None,
    hline_label: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    One figure, one boxplot.

    Parameters
    ----------
    summary_df  : roi_plane_summary DataFrame (one row per session × plane)
    metric      : column name
    ylabel      : y-axis label
    title       : figure title
    hline       : optional reference line
    hline_label : label for that line
    save_path   : if given, save PNG here
    """
    planes = [p for p in PLANE_ORDER if p in summary_df["plane"].values]

    if not planes:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return fig

    data   = [summary_df.loc[summary_df["plane"] == p, metric].dropna().values
              for p in planes]
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
        ax.text(n_visl * 0.5 / len(planes) + 0.5 / len(planes), 1.01,
                "VISl", ha="center", va="bottom", fontsize=9,
                color=AREA_COLOR["VISl"], fontweight="bold",
                transform=ax.transAxes)
        ax.text((n_visl + (len(planes) - n_visl) * 0.5) / len(planes) +
                0.5 / len(planes), 1.01,
                "VISp", ha="center", va="bottom", fontsize=9,
                color=AREA_COLOR["VISp"], fontweight="bold",
                transform=ax.transAxes)

    ax.set_xticks(range(1, len(planes) + 1))
    ax.set_xticklabels(planes, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=32)
    ax.scatter([], [], marker="D", color="black", s=28, label="per-plane mean")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.82, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def plot_all(
    summary_df: pd.DataFrame,
    save_dir: str | Path = "outputs/stage2/figures/roi",
) -> dict[str, plt.Figure]:
    """
    Produce one figure per ROI classification metric and save PNGs.

    Parameters
    ----------
    summary_df : roi_plane_summary DataFrame
    save_dir   : output directory

    Returns
    -------
    dict of {metric_col: figure}
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ROI classification figures to {save_dir}/\n")

    figures: dict[str, plt.Figure] = {}

    for i, (col, ylabel, title, hline, hline_label) in enumerate(METRICS):
        if col not in summary_df.columns:
            print(f"  skip {col!r}  (column not present)")
            continue
        if summary_df[col].dropna().empty:
            print(f"  skip {col!r}  (all NaN)")
            continue

        fname = save_dir / f"{i+1:02d}_{col}.png"
        print(f"  [{i+1:02d}] {col} ...", end=" ", flush=True)
        try:
            fig = boxplot_metric(
                summary_df, col,
                ylabel=ylabel, title=title,
                hline=hline, hline_label=hline_label,
                save_path=fname,
            )
            figures[col] = fig
            plt.close(fig)
            print("✓")
        except Exception as e:
            print(f"✗  ({e})")

    print(f"\nDone. {len(figures)}/{len(METRICS)} figures saved.")
    return figures
