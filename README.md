# OpenScope Ophys QC

Quality control pipeline for the [OpenScope Community Predictive Processing](https://alleninstitute.org/division/neural-dynamics/) planar mesoscope dataset, hosted on AWS S3 at `s3://aind-open-data`.

The pipeline loads processed NWB/Zarr sessions, computes per-ROI and per-plane QC metrics (dF/F distributions, event rates, soma probability, drift), and produces summary figures and CSVs for a single session or a sample of the full dataset.

> 📄 **For a full reference on session organization, file types, and NWB structure see** [File Type Reference](documentation/filetype_reference.md)

---

## Repository Structure

```
openscope-ophys-qc/
├── README.md
├── .gitignore
├── requirements_mesoscope_qc.txt
│
├── documentation/
│   └── filetype_reference.md      # Session structure, file types, NWB schema
│
├── src/                           # Importable pipeline modules
│   ├── s3_utils.py                # Shared S3 helpers
│   ├── session_loader.py          # Session discovery and loading
│   ├── motion_loader.py           # Single-session motion data loading
│   ├── motion_qc.py               # Multi-session motion metric computation
│   ├── motion_plots.py            # Motion QC visualizations
│   ├── roi_classifier.py          # ROI classification extraction from NWB
│   ├── roi_plots.py               # ROI classification visualizations
│   ├── scatter_plots.py           # Cross-analysis scatter figures
│   └── zdrift_plots.py            # Z-drift case study figures
│
├── notebooks/
│   ├── data_loader_example.ipynb        # How to load sessions
│   ├── motion_qc_analysis.ipynb         # Motion QC figures and metrics
│   ├── guided_tour_of_ophys_session.ipynb  # Guided tour of a single session
│   └── openscope_qc.ipynb               # Full QC pipeline (single + multi-session)
│
└── outputs/                       # Generated locally, not committed
    ├── figures/
    ├── stage1/
    └── stage2/
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Dedalus9/openscope-ophys-qc.git
cd openscope-ophys-qc

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements_mesoscope_qc.txt
```

---
## Notebooks

### [`data_loader_example.ipynb`](notebooks/data_loader_example.ipynb)
Demonstrates how to load single and multiple sessions directly from S3:
- No `src/` modules required — fully self-contained and runnable out of the box
- Shows how to load a single session and inspect its metadata, planes, and NWB contents
- Shows how to discover and load a sample of sessions across the full dataset
- Intended as a starting point for new users before moving to the analysis notebooks

### [`guided_tour_of_ophys_session.ipynb`](notebooks/guided_tour_of_ophys_session.ipynb)
A systematic walkthrough of every file and data type in a single processed session:
- Covers session metadata, subject information, processing provenance, and QC evaluations
- Loads and visualizes motion correction offsets, registration quality images, and per-plane projections
- Reads the NWB to inspect ROI segmentation, fluorescence traces, deconvolved events, and population activity
- Documents known gaps and caveats in the dataset that affect downstream analysis

### [`openscope_motion_qc.ipynb`](notebooks/openscope_motion_qc.ipynb)
The full two-stage QC pipeline, making use of all `src/` modules:

**Stage 1 — Single Session**
- Loads per-frame motion data for every imaging plane from S3
- Produces a three-panel displacement overview (per-plane traces, heatmap, consensus) and a cross-plane correlation matrix
- Computes a per-plane summary statistics table covering displacement, registration correlation, bad-frame fraction, and invalid-frame fraction
- Figures saved to `notebooks/outputs/stage1/`

**Stage 2 — Multi-Session**
- Discovers all available sessions in the S3 bucket and samples a configurable subset
- Computes the full set of motion QC metrics (displacement statistics, registration quality, z-drift, burst count, settling time, intensity stability) for every session × plane
- Extracts ROI classification scores (soma probability, dendrite probability, ROI yield) from the NWB Zarr for every session × plane
- Joins motion and ROI data to test whether image quality predicts classification outcomes
- Generates the complete figure set: session- and plane-level motion boxplots, ROI classification boxplots, cross-analysis scatter plots, and z-drift case studies at the extremes of the quality distribution
- Figures saved to `notebooks/outputs/stage2/figures/motion/`, `notebooks/outputs/stage2/figures/roi/`, `notebooks/outputs/stage2/figures/scatter/`, and `notebooks/outputs/stage2/figures/zdrift_examples/`
- Summary CSVs saved to `notebooks/outputs/stage2/`

---

## Extending the Pipeline
- **New QC metrics** — add them to the relevant `src/` module; new columns added to the plane-level DataFrame will propagate automatically to the session-level aggregation and output CSVs
- **New visualizations** — add functions to `motion_plots.py`, `roi_plots.py`, or `scatter_plots.py`; follow the existing pattern of accepting a DataFrame and an optional `save_dir`, returning a `plt.Figure`
- **New notebook stages** — each stage should be self-contained and save its outputs to a dedicated `outputs/stageN/` folder