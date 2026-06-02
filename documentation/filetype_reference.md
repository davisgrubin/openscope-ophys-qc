# OpenScope Community Predictive Processing — File Type Reference

Asset root: `s3://aind-open-data/multiplane-ophys_{subject}_{acq-date}_{acq-time}_processed_{proc-date}_{proc-time}/`

---

## Session-level files

| File | Format | Contents |
|------|--------|----------|
| `quality_control.json` | JSON | Human-reviewed QC evaluations per plane: z-drift (µm), intensity stability (%), pass/fail status, reviewer name and timestamp |
| `pophys.nwb.zarr/` | Zarr (NWB) | Main data file. Contains all ROI traces, segmentation masks, metadata, and imaging plane info for all 8 planes |
| `data_description.json` | JSON | Session-level provenance: subject ID, acquisition date, modality, institution |
| `subject.json` | JSON | Subject metadata: species, sex, age, genotype, date of birth |
| `procedures.json` | JSON | Surgical and experimental procedures |
| `rig.json` | JSON | Hardware configuration: microscope model, objectives, scan parameters |
| `session.json` | JSON | Session metadata: experimenter, stimulus protocol, notes |
| `processing.json` | JSON | Pipeline provenance: which processing modules ran and in what order |

---

## Per-plane files

One set per plane folder: `VISl_4/`, `VISl_5/`, `VISl_6/`, `VISl_7/`, `VISp_0/`, `VISp_1/`, `VISp_2/`, `VISp_3/`

### Motion correction

| File | Format | Contents |
|------|--------|----------|
| `{plane}_motion_transform.csv` | CSV | Per-frame registration offsets: `x`, `y` (rigid translation in pixels), `x_pre_clip`, `y_pre_clip` (raw estimates before clipping to maxregshift), `correlation` (peak cross-correlation between frame and template), `is_valid` (suite2p flag), `nonrigid_x`, `nonrigid_y`, `nonrigid_corr` (per-block non-rigid corrections, stored as arrays). 41,307 rows × 10 columns. |
| `{plane}_registered.h5` | HDF5 | Full motion-corrected pixel movie after registration |
| `{plane}_motion_correction_data_process.json` | JSON | Full processing provenance: pipeline name ("Video motion correction"), software version (8.0), start/end timestamps, input/output paths, and all suite2p parameters (nonrigid=True, block_size=[128,128], maxregshift=0.1, smooth_sigma=1.15, snr_thresh=1.2, etc.). Also contains `suite2p_args` sub-object with the complete suite2p parameter set passed to the registration step. |
| `{plane}_registration_summary_metric.json` | JSON | Dropdown-style QC evaluation record: pass/fail options, reviewer status history, and reference to the registration summary PNG. Note: the `value.value` field is empty until a human reviewer selects an option — check `status_history` for actual review state. |
| `{plane}_fov_quality_metric.json` | JSON | Similar dropdown QC record for FOV quality (timeseries shuffled, incorrect area/depth, crosstalk). Same structure as registration_summary_metric.json. |
| `{plane}_motion_preview.webm` | WebM video | Preview video of registered movie for visual QC (2s frame bins, 10× playback) |
| `{plane}_registration_summary.png` | PNG | Suite2p-style registration summary plot |
| `{plane}_registration_summary_PC0high.png` | PNG | Frames with highest PC0 projection (high-motion frames) |
| `{plane}_registration_summary_PC0low.png` | PNG | Frames with lowest PC0 projection |
| `{plane}_registration_summary_PC0rof.png` | PNG | Residual-of-fit for PC0 |
| `{plane}_registration_summary_PC3high.png` | PNG | Frames with highest PC3 projection |
| `{plane}_registration_summary_PC3low.png` | PNG | Frames with lowest PC3 projection |
| `{plane}_registration_summary_PC3rof.png` | PNG | Residual-of-fit for PC3 |
| `{plane}_registration_summary_nonrigid.png` | PNG | Non-rigid deformation field summary plot |
| `{plane}_average_projection.png` | PNG | Average intensity projection of registered movie (template image) |
| `{plane}_maximum_projection.png` | PNG | Maximum intensity projection of registered movie |
| `{plane}_combined_projection.png` | PNG | Combined average + maximum projection overlay |

### Segmentation and traces (in NWB)

These are accessed via the NWB file rather than as standalone files, under `nwb.processing[plane_name]`.

| NWB path | Format | Contents |
|----------|--------|----------|
| `image_segmentation/roi_table` | NWB DynamicTable | Per-ROI metadata: `image_mask` (H×W dense float), `is_soma`, `soma_probability`, `is_dendrite`, `dendrite_probability` |
| `raw_timeseries/ROI_fluorescence_timeseries` | TimeSeries | Raw fluorescence F, shape (T, n_rois). Timestamps stored explicitly (`rate=None`); read timestamps array separately. |
| `neuropil_fluorescence_timeseries` | TimeSeries | Neuropil fluorescence, shape (T, n_rois). Timestamps explicit. |
| `neuropil_corrected_timeseries` | TimeSeries | Neuropil-corrected fluorescence, shape (T, n_rois). Timestamps explicit. |
| `dff_timeseries/dff_timeseries` | TimeSeries | ΔF/F traces, shape (T, n_rois). Timestamps explicit. |
| `event_timeseries` | TimeSeries | Deconvolved event amplitudes (OASIS), shape (T, n_rois). Timestamps explicit. |
| `images/average_projection` | GrayscaleImage | 512×512 float32 average projection, **normalized to [0, 1]** |
| `images/max_projection` | GrayscaleImage | 512×512 float32 maximum projection, **normalized to [0, 1]** |
| `images/segmentation_mask_image` | GrayscaleImage | 512×512 uint16 ROI label image (pixel value = ROI index; max value = n_rois for that plane) |

---

## Dataset constants

| Property | Value |
|----------|-------|
| Frame rate | 9.48 Hz |
| Frames per session | 41,307 (nominal; varies across sessions) |
| Pixel size | 0.78 µm/pixel |
| Frame size | 512 × 512 pixels |
| Planes per session | 8 (VISl_4–7, VISp_0–3) |
| Indicator | jGCaMP8s |
| Segmentation | suite2p-cellpose anatomical mode |
| Registration | suite2p 0.14.4, nonrigid, block_size=[128,128] |
| Sessions in dataset | 364 (deduplicated to latest processing run) |

### ROI counts — reference session (837568, 2026-03-05)

| Plane | ROIs |
|-------|------|
| VISl_4 | 353 |
| VISl_5 | 249 |
| VISl_6 | 130 |
| VISl_7 | 248 |
| VISp_0 | 386 |
| VISp_1 | 458 |
| VISp_2 | 130 |
| VISp_3 | 263 |
| **Total** | **2,217** |

---

## Known gaps and caveats

| Issue | Detail |
|-------|--------|
| `registration_summary_metric.json` reviewer status | The `status_history` for some planes shows `"Pending review"` rather than a completed human review. The `value.value` field is an empty list `[]` until a reviewer selects an option — cannot be used as a programmatic pass/fail without checking `status_history[-1].status`. |
| Depth (µm) not in NWB | `nwb.imaging_planes[plane].location` returns `"Two-photon imaging plane a"` for all planes — not a numeric depth. Actual imaging depths are not stored in the NWB for this dataset. May be recoverable from `procedures.json` or `session.json`. |
| Stimulus intervals absent from NWB | `nwb.intervals` is empty. Stimulus timing for the predictive processing paradigm is not present in the planar-ophys NWB. It likely lives in a separate behavioral NWB or timing file not yet identified. |
| `rate=None` on all timeseries | Sampling rate is not stored as a scalar; timestamps are stored as an explicit array. Always read the `.timestamps` array rather than assuming a fixed rate. |
| `is_valid` stored as string | The `is_valid` column in `motion_transform.csv` is stored as `"True"`/`"False"` strings, not booleans. Parse with `.astype(str).str.lower() == "true"`. |

---

## File access patterns

| Goal | Files needed |
|------|-------------|
| Motion QC metrics | `{plane}_motion_transform.csv`, `quality_control.json` |
| Visual registration check | `{plane}_registration_summary.png`, `{plane}_motion_preview.webm` |
| ROI-level signal QC | `pophys.nwb.zarr` (dff, events, roi_table) |
| Session metadata | `data_description.json`, `subject.json`, `session.json` |
| Full pixel data | `{plane}_registered.h5` |
| Processing provenance | `{plane}_motion_correction_data_process.json`, `processing.json` |