# OpenScope Ophys QC

Quality control pipeline for the [OpenScope Community Predictive Processing](https://alleninstitute.org/division/neural-dynamics/) planar mesoscope dataset, hosted on AWS S3 at `s3://aind-open-data`.

The pipeline loads processed NWB/Zarr sessions, computes per-ROI and per-plane QC metrics (dF/F distributions, event rates, soma probability, drift), and produces summary figures and CSVs for a single session or a sample of the full dataset.

---

## Repository structure

```
OpenScope_Ophys_QC/
├── README.md
├── .gitignore
├── requirements_mesoscope_qc.txt
│
├── src/
│   ├── mesoscope_qc_pipeline.py   # Core QC functions (loading, metrics, plotting)
│   └── session_loader.py          # Session discovery, loading, and visualization helpers
│
├── notebooks/
│   └── OpenScope_Ophys_QC_Validation.ipynb  # Main analysis notebook
│
└── outputs/                       # Generated locally, not committed
    ├── figures/
    └── stage1/
```

---

## Setup
**Requirements:** Python 3.10+

```bash
git clone https://github.com/<your-org>/OpenScope_Ophys_QC.git
cd OpenScope_Ophys_QC

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements_mesoscope_qc.txt
```
---

## Running the notebook
Open `DataLoader.ipynb`. The notebook is organized into two stages:

| Stage | Scope | Goal |
|-------|-------|------|
| **1** | Single session | Inspect metrics and establish reference distributions |
| **2** | Multiple sessions | Load and compare a sample of the full dataset |

In Stage 1, set `SESSION_SOURCE` to any processed session path from `s3://aind-open-data`. In Stage 2, `sl.discover_sessions()` will automatically find and deduplicate all available sessions.
---
## Data

Data comes from the [Allen Institute for Neural Dynamics open data bucket](https://registry.opendata.aws/aind-open-data/).

Sessions follow the AIND naming convention:
```
# Raw asset
<modality>_<subject-id>_<acquisition-date>_<acquisition-time>

# Derived (processed) asset
<source-asset-name>_<label>_<processing-date>_<processing-time>
```

Example:
```
multiplane-ophys_837568_2026-03-05_14-14-51_processed_2026-03-06_11-31-22
└─ source asset ───────────────────────────┘ └─label─┘ └─ processing ts ─┘
```

When multiple processing runs exist for the same raw session, `discover_sessions()` keeps the latest by default.

---
## Extending the pipeline
- **New QC metrics** — add them to `mesoscope_qc_pipeline.py`. The `analyze_plane()` function returns a per-ROI DataFrame; new columns added there will propagate automatically to `summarize_planes()` and the output CSVs.
- **New visualizations** — add functions to `session_loader.py`. Follow the existing pattern: accept a DataFrame and an optional `save_path`, return a `plt.Figure`.
- **New notebook stages** — the two-stage structure in the notebook is intentional. Add stages sequentially; each stage should be self-contained and save its outputs to a dedicated `outputs/stageN/` folder.