"""Signal-to-noise metrics for mesoscope dF/F traces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import log_ndtr


def half_sample_mode(values: np.ndarray) -> float:
    """Estimate the mode using the recursive half-sample mode."""
    data = np.sort(np.asarray(values, dtype=float))
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan
    if data.size == 1:
        return float(data[0])
    if data.size == 2:
        return float(np.mean(data))
    if data.size == 3:
        left = data[1] - data[0]
        right = data[2] - data[1]
        if left < right:
            return float(np.mean(data[:2]))
        if right < left:
            return float(np.mean(data[1:]))
        return float(data[1])

    width = (data.size + 1) // 2
    starts = np.arange(data.size - width + 1)
    spans = data[starts + width - 1] - data[starts]
    start = int(starts[np.argmin(spans)])
    return half_sample_mode(data[start : start + width])


def exceptional_event_metric(
    trace: np.ndarray,
    consecutive_samples: int = 5,
    robust_std: bool = False,
) -> dict[str, float]:
    """
    Adapt the exceptional-event fitness from evaluate_components.py.

    The original fitness is a log tail probability. More negative values indicate
    a less probable and therefore stronger run of positive-going events.
    """
    values = np.asarray(trace, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "exceptional_event_fitness": np.nan,
            "exceptional_event_score": np.nan,
            "exceptional_event_mode": np.nan,
            "exceptional_event_noise_sd": np.nan,
        }

    mode = half_sample_mode(values)
    lower_deviation = mode - values[values < mode]
    lower_deviation = lower_deviation[np.isfinite(lower_deviation) & (lower_deviation > 0)]
    if lower_deviation.size == 0:
        noise_sd = np.nan
    elif robust_std:
        half_iqr = float(np.nanmedian(lower_deviation))
        noise_sd = 2.0 * half_iqr / 1.349
    else:
        noise_sd = float(np.sqrt(np.nanmean(lower_deviation**2)))

    if not np.isfinite(noise_sd) or noise_sd <= 0:
        fitness = np.nan
    else:
        z = (values - mode) / (3.0 * noise_sd)
        log_tail_probability = log_ndtr(-z)
        window = max(1, int(consecutive_samples))
        moving_log_probability = np.convolve(
            log_tail_probability,
            np.ones(window, dtype=float),
            mode="full",
        )[: values.size]
        fitness = float(np.nanmin(moving_log_probability))

    return {
        "exceptional_event_fitness": fitness,
        "exceptional_event_score": -fitness if np.isfinite(fitness) else np.nan,
        "exceptional_event_mode": float(mode),
        "exceptional_event_noise_sd": noise_sd,
    }


def robust_event_snr(trace: np.ndarray, sigma: float = 3.0) -> dict[str, float]:
    """Compute robust event amplitude divided by robust fast-residual noise."""
    values = np.asarray(trace, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "robust_event_snr": np.nan,
            "robust_event_signal_amp": np.nan,
            "robust_event_noise_sd": np.nan,
        }

    signal_amp = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 50))
    smooth = gaussian_filter1d(values, sigma=float(sigma))
    residual = values - smooth
    residual_median = np.nanmedian(residual)
    noise_sd = float(1.4826 * np.nanmedian(np.abs(residual - residual_median)))
    snr = signal_amp / noise_sd if noise_sd > 0 else np.nan
    return {
        "robust_event_snr": float(snr),
        "robust_event_signal_amp": signal_amp,
        "robust_event_noise_sd": noise_sd,
    }


def roi_mask_metrics(mask: np.ndarray) -> dict[str, float]:
    """Compute area and shape metrics from one dense ROI image mask."""
    binary = np.asarray(mask) > 0
    yy, xx = np.nonzero(binary)
    area = int(binary.sum())
    if area == 0:
        return {
            "roi_area_pix": 0,
            "roi_perimeter_pix": np.nan,
            "roi_circularity": np.nan,
            "roi_elongation": np.nan,
        }

    padded = np.pad(binary.astype(np.int8), 1)
    perimeter = int(
        np.abs(np.diff(padded, axis=0)).sum()
        + np.abs(np.diff(padded, axis=1)).sum()
    )
    circularity = 4.0 * np.pi * area / perimeter**2 if perimeter else np.nan

    elongation = np.nan
    if area >= 3:
        coordinates = np.column_stack([xx, yy]).astype(float)
        eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
        if eigenvalues[0] > 0:
            elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))

    return {
        "roi_area_pix": area,
        "roi_perimeter_pix": perimeter,
        "roi_circularity": float(circularity),
        "roi_elongation": elongation,
    }


def calculate_roi_extraction_metrics(roi_table) -> pd.DataFrame:
    """
    Calculate morphology and classifier-confidence metrics from an NWB ROI table.

    Masks are read one ROI at a time. Sparse ``pixel_mask`` data are preferred
    when available, avoiding a bulk read of the dense image-mask array.
    """
    n_rois = len(roi_table)
    colnames = list(getattr(roi_table, "colnames", []))
    soma_probability = (
        np.asarray(roi_table["soma_probability"].data[:], dtype=float)
        if "soma_probability" in colnames
        else np.full(n_rois, np.nan)
    )
    dendrite_probability = (
        np.asarray(roi_table["dendrite_probability"].data[:], dtype=float)
        if "dendrite_probability" in colnames
        else np.full(n_rois, np.nan)
    )

    rows = []
    for roi_index in range(n_rois):
        if "pixel_mask" in colnames:
            pixel_mask = np.asarray(roi_table["pixel_mask"][roi_index], dtype=float).reshape(-1, 3)
            if len(pixel_mask):
                width = int(np.nanmax(pixel_mask[:, 0])) + 1
                height = int(np.nanmax(pixel_mask[:, 1])) + 1
                mask = np.zeros((height, width), dtype=bool)
                x = pixel_mask[:, 0].astype(int)
                y = pixel_mask[:, 1].astype(int)
                mask[y, x] = pixel_mask[:, 2] > 0
            else:
                mask = np.zeros((0, 0), dtype=bool)
        elif "image_mask" in colnames:
            mask = np.asarray(roi_table["image_mask"].data[roi_index])
        else:
            mask = np.zeros((0, 0), dtype=bool)

        row = {"roi_index": roi_index}
        row.update(roi_mask_metrics(mask))
        soma_prob = soma_probability[roi_index]
        dendrite_prob = dendrite_probability[roi_index]
        row["roi_classifier_confidence"] = (
            float(np.nanmax([soma_prob, dendrite_prob]))
            if np.any(np.isfinite([soma_prob, dendrite_prob]))
            else np.nan
        )
        row["roi_classifier_margin"] = (
            float(abs(soma_prob - dendrite_prob))
            if np.isfinite(soma_prob) and np.isfinite(dendrite_prob)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_stability_metrics(
    trace: np.ndarray,
    noise_sd: float,
    n_bins: int = 10,
) -> dict[str, float]:
    """Measure slow baseline movement across a dF/F trace."""
    values = np.asarray(trace, dtype=float)
    n_values = len(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "dff_early_median": np.nan,
            "dff_late_median": np.nan,
            "dff_drift_delta": np.nan,
            "dff_abs_drift_noise_units": np.nan,
            "dff_baseline_bin_median_sd": np.nan,
            "dff_baseline_bin_range": np.nan,
            "dff_baseline_range_noise_units": np.nan,
        }

    edge = max(1, n_values // 10)
    early = float(np.nanmedian(values[:edge]))
    late = float(np.nanmedian(values[-edge:]))
    drift = late - early

    bin_medians = np.array(
        [np.nanmedian(chunk) for chunk in np.array_split(values, max(2, int(n_bins)))],
        dtype=float,
    )
    bin_medians = bin_medians[np.isfinite(bin_medians)]
    baseline_sd = float(np.nanstd(bin_medians)) if len(bin_medians) else np.nan
    baseline_range = (
        float(np.nanmax(bin_medians) - np.nanmin(bin_medians))
        if len(bin_medians)
        else np.nan
    )
    valid_noise = np.isfinite(noise_sd) and noise_sd > 0
    return {
        "dff_early_median": early,
        "dff_late_median": late,
        "dff_drift_delta": drift,
        "dff_abs_drift_noise_units": abs(drift) / noise_sd if valid_noise else np.nan,
        "dff_baseline_bin_median_sd": baseline_sd,
        "dff_baseline_bin_range": baseline_range,
        "dff_baseline_range_noise_units": baseline_range / noise_sd if valid_noise else np.nan,
    }


def _event_onsets(event_trace: np.ndarray, threshold: float) -> np.ndarray:
    positive = np.isfinite(event_trace) & (event_trace > threshold)
    return np.flatnonzero(positive & ~np.r_[False, positive[:-1]])


def _event_triggered_average(
    trace: np.ndarray,
    event_indices: np.ndarray,
    sample_rate_hz: float,
    pre_s: float,
    post_s: float,
    max_events: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    pre_frames = max(1, int(round(pre_s * sample_rate_hz)))
    post_frames = max(2, int(round(post_s * sample_rate_hz)))
    valid = event_indices[
        (event_indices >= pre_frames) & (event_indices + post_frames < len(trace))
    ]
    if len(valid) == 0:
        return None, None
    if len(valid) > max_events:
        keep = np.linspace(0, len(valid) - 1, max_events).astype(int)
        valid = valid[keep]

    windows = np.stack(
        [trace[index - pre_frames : index + post_frames + 1] for index in valid],
        axis=0,
    ).astype(float)
    baseline = np.nanmedian(windows[:, :pre_frames], axis=1, keepdims=True)
    windows -= baseline
    time = np.arange(-pre_frames, post_frames + 1, dtype=float) / sample_rate_hz
    return time, np.nanmedian(windows, axis=0)


def _fit_calcium_decay(
    time: np.ndarray | None,
    transient: np.ndarray | None,
) -> dict[str, float]:
    if time is None or transient is None:
        return {
            "calcium_kernel_peak_dff": np.nan,
            "calcium_kernel_tau_s": np.nan,
            "calcium_kernel_decay_r2": np.nan,
        }

    post = np.flatnonzero(time >= 0)
    if len(post) < 4:
        return {
            "calcium_kernel_peak_dff": np.nan,
            "calcium_kernel_tau_s": np.nan,
            "calcium_kernel_decay_r2": np.nan,
        }
    peak_index = int(post[np.nanargmax(transient[post])])
    peak = float(transient[peak_index])
    decay_time = time[peak_index:] - time[peak_index]
    decay = transient[peak_index:]
    threshold = max(peak * 0.1, 0)
    below_threshold = np.flatnonzero(
        np.isfinite(decay[1:]) & (decay[1:] <= threshold)
    )
    stop = int(below_threshold[0] + 1) if len(below_threshold) else len(decay)
    decay_time = decay_time[:stop]
    decay = decay[:stop]
    valid = np.isfinite(decay) & (decay > threshold)
    if peak <= 0 or np.sum(valid) < 4:
        return {
            "calcium_kernel_peak_dff": peak,
            "calcium_kernel_tau_s": np.nan,
            "calcium_kernel_decay_r2": np.nan,
        }

    x = decay_time[valid]
    y = np.log(decay[valid])
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    denominator = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum((y - prediction) ** 2) / denominator if denominator > 0 else np.nan
    tau = -1.0 / slope if slope < 0 else np.nan
    return {
        "calcium_kernel_peak_dff": peak,
        "calcium_kernel_tau_s": float(tau),
        "calcium_kernel_decay_r2": float(r2),
    }


def event_extraction_metrics(
    trace: np.ndarray,
    event_trace: np.ndarray,
    timestamps: np.ndarray,
    noise_sd: float,
    *,
    threshold: float = 0.0,
    kernel_pre_s: float = 0.5,
    kernel_post_s: float = 2.0,
    max_kernel_events: int = 500,
) -> dict[str, float]:
    """Measure extracted-event frequency, amplitude, SNR, and calcium decay."""
    event_values = np.asarray(event_trace, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    duration_s = (
        float(timestamps[-1] - timestamps[0])
        if len(timestamps) > 1
        else np.nan
    )
    positive = event_values[np.isfinite(event_values) & (event_values > threshold)]
    onsets = _event_onsets(event_values, threshold)
    sample_rate_hz = (
        float(1.0 / np.nanmedian(np.diff(timestamps)))
        if len(timestamps) > 2
        else np.nan
    )
    valid_duration = np.isfinite(duration_s) and duration_s > 0

    time, transient = (None, None)
    if np.isfinite(sample_rate_hz) and sample_rate_hz > 0:
        time, transient = _event_triggered_average(
            np.asarray(trace, dtype=float),
            onsets,
            sample_rate_hz,
            kernel_pre_s,
            kernel_post_s,
            max_kernel_events,
        )
    decay_metrics = _fit_calcium_decay(time, transient)
    peak = decay_metrics["calcium_kernel_peak_dff"]
    valid_noise = np.isfinite(noise_sd) and noise_sd > 0

    return {
        "event_positive_sample_count": int(len(positive)),
        "event_positive_sample_rate_hz": len(positive) / duration_s if valid_duration else np.nan,
        "event_onset_count": int(len(onsets)),
        "event_onset_rate_hz": len(onsets) / duration_s if valid_duration else np.nan,
        "event_amplitude_median": float(np.nanmedian(positive)) if len(positive) else np.nan,
        "event_amplitude_p95": float(np.nanpercentile(positive, 95)) if len(positive) else np.nan,
        "event_triggered_dff_snr": peak / noise_sd if valid_noise else np.nan,
        **decay_metrics,
    }


def calculate_roi_snr_metrics(
    dff: np.ndarray,
    timestamps: np.ndarray | None = None,
    events: np.ndarray | None = None,
    *,
    gaussian_sigma: float = 3.0,
    consecutive_samples: int = 5,
    exceptional_robust_std: bool = False,
    event_threshold: float = 0.0,
    baseline_bins: int = 10,
    kernel_pre_s: float = 0.5,
    kernel_post_s: float = 2.0,
    max_kernel_events: int = 500,
) -> pd.DataFrame:
    """Calculate dF/F and optional extracted-event QC metrics for every ROI."""
    matrix = np.asarray(dff, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"dff must be a 2D time-by-ROI matrix, got shape {matrix.shape}")
    if timestamps is None:
        timestamps = np.arange(matrix.shape[0], dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if len(timestamps) != matrix.shape[0]:
        raise ValueError("timestamps must have one value per dF/F timepoint")
    event_matrix = None if events is None else np.asarray(events, dtype=float)
    if event_matrix is not None and event_matrix.shape != matrix.shape:
        raise ValueError(
            f"events must match dff shape {matrix.shape}, got {event_matrix.shape}"
        )

    rows = []
    for roi_index in range(matrix.shape[1]):
        trace = matrix[:, roi_index]
        finite = np.isfinite(trace)
        row = {
            "roi_index": roi_index,
            "n_timepoints": int(trace.size),
            "n_finite_timepoints": int(np.sum(finite)),
            "fraction_nan": float(1.0 - np.mean(finite)),
        }
        snr_metrics = robust_event_snr(trace, sigma=gaussian_sigma)
        row.update(snr_metrics)
        row.update(
            baseline_stability_metrics(
                trace,
                snr_metrics["robust_event_noise_sd"],
                n_bins=baseline_bins,
            )
        )
        row.update(
            exceptional_event_metric(
                trace,
                consecutive_samples=consecutive_samples,
                robust_std=exceptional_robust_std,
            )
        )
        if event_matrix is not None:
            row.update(
                event_extraction_metrics(
                    trace,
                    event_matrix[:, roi_index],
                    timestamps,
                    snr_metrics["robust_event_noise_sd"],
                    threshold=event_threshold,
                    kernel_pre_s=kernel_pre_s,
                    kernel_post_s=kernel_post_s,
                    max_kernel_events=max_kernel_events,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)
