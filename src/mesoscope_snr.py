"""Signal-to-noise metrics for mesoscope dF/F traces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import log_ndtr
from scipy.stats import expon, kstest, norm


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


def _mad_sd(values: np.ndarray) -> float:
    """Robust SD estimate from median absolute deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    return float(1.4826 * mad)


def signal_noise_component_metrics(
    trace: np.ndarray,
    *,
    sigma: float = 3.0,
    baseline_bins: int = 10,
) -> dict[str, float]:
    """
    Calculate separable signal and noise estimates plus every signal/noise pair.

    These are intentionally component-style metrics. They let the notebook ask
    whether a ranking is driven by high signal, low noise, or a particular
    signal/noise pairing.
    """
    values = np.asarray(trace, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    median = float(np.nanpercentile(values, 50))
    mode = half_sample_mode(values)
    lower_deviation = mode - values[values < mode]
    lower_deviation = lower_deviation[
        np.isfinite(lower_deviation) & (lower_deviation > 0)
    ]
    upper_from_median = values[values > median] - median
    upper_from_median = upper_from_median[
        np.isfinite(upper_from_median) & (upper_from_median > 0)
    ]

    smooth = gaussian_filter1d(values, sigma=float(sigma))
    residual = values - smooth
    diff = np.diff(values)
    bin_medians = np.array(
        [np.nanmedian(chunk) for chunk in np.array_split(values, max(2, int(baseline_bins)))],
        dtype=float,
    )
    bin_medians = bin_medians[np.isfinite(bin_medians)]

    signal_metrics = {
        "signal_p95_p50": float(np.nanpercentile(values, 95) - median),
        "signal_p99_p50": float(np.nanpercentile(values, 99) - median),
        "signal_p95_mode": float(np.nanpercentile(values, 95) - mode),
        "signal_upper_mad": _mad_sd(upper_from_median),
        "signal_positive_auc": float(np.nanmean(np.clip(values - median, 0, None))),
    }
    noise_metrics = {
        "noise_fast_mad": _mad_sd(residual),
        "noise_firstdiff_mad": _mad_sd(diff) / np.sqrt(2.0) if diff.size else np.nan,
        "noise_lower_mode_rms": (
            float(np.sqrt(np.nanmean(lower_deviation**2)))
            if lower_deviation.size
            else np.nan
        ),
        "noise_lower_mode_mad": (
            float(2.0 * np.nanmedian(lower_deviation) / 1.349)
            if lower_deviation.size
            else np.nan
        ),
        "noise_baseline_bin_sd": (
            float(np.nanstd(bin_medians)) if bin_medians.size else np.nan
        ),
    }

    out = {
        "signal_noise_baseline_median": median,
        "signal_noise_baseline_mode": float(mode),
        **signal_metrics,
        **noise_metrics,
    }
    for signal_name, signal_value in signal_metrics.items():
        for noise_name, noise_value in noise_metrics.items():
            out[f"snr_pair__{signal_name}__{noise_name}"] = (
                signal_value / noise_value
                if np.isfinite(signal_value)
                and np.isfinite(noise_value)
                and noise_value > 0
                else np.nan
            )
    return out


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


def _sample_rate_hz_from_timestamps(timestamps: np.ndarray) -> float:
    """Estimate sample rate from timestamps."""
    timestamps = np.asarray(timestamps, dtype=float)
    if len(timestamps) <= 2:
        return np.nan
    dt = np.nanmedian(np.diff(timestamps))
    return float(1.0 / dt) if np.isfinite(dt) and dt > 0 else np.nan


def _merged_event_clusters(
    event_indices: np.ndarray,
    sample_rate_hz: float,
    merge_within_s: float,
) -> list[tuple[int, int, int]]:
    """Merge event onsets that are close enough to share one calcium transient."""
    event_indices = np.asarray(event_indices, dtype=int)
    if event_indices.size == 0:
        return []
    merge_frames = max(1, int(round(float(merge_within_s) * sample_rate_hz)))
    clusters: list[tuple[int, int, int]] = []
    start = int(event_indices[0])
    last = int(event_indices[0])
    count = 1
    for index in event_indices[1:]:
        index = int(index)
        if index - last <= merge_frames:
            last = index
            count += 1
        else:
            clusters.append((start, last, count))
            start = last = index
            count = 1
    clusters.append((start, last, count))
    return clusters


def background_event_amplitude_metrics(
    trace: np.ndarray,
    event_trace: np.ndarray,
    timestamps: np.ndarray,
    *,
    threshold: float = 0.0,
    background_exclude_pre_s: float = 0.5,
    background_exclude_post_s: float = 2.0,
    merge_within_s: float = 0.5,
    local_baseline_pre_s: float = 0.5,
    peak_search_post_s: float = 2.0,
    low_sd: float = 2.0,
    high_sd: float = 4.0,
) -> dict[str, float]:
    """
    Compare detected calcium events with the ROI's non-event background.

    The background is the dF/F trace after removing windows around detected
    event onsets. Closely spaced event onsets are merged into a cluster, and
    the cluster midpoint is used as the event time for amplitude measurement.
    """
    values = np.asarray(trace, dtype=float)
    event_values = np.asarray(event_trace, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if values.shape != event_values.shape:
        raise ValueError("trace and event_trace must have the same shape")

    sample_rate_hz = _sample_rate_hz_from_timestamps(timestamps)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return {
            "background_sample_fraction": np.nan,
            "background_median_dff": np.nan,
            "background_noise_sd": np.nan,
            "background_event_raw_onset_count": np.nan,
            "background_event_cluster_count": np.nan,
            "background_event_mean_raw_onsets_per_cluster": np.nan,
            "background_event_median_amp_dff": np.nan,
            "background_event_p95_amp_dff": np.nan,
            "background_event_median_amp_noise_units": np.nan,
            "background_event_p95_amp_noise_units": np.nan,
            "background_event_fraction_lt_2sd": np.nan,
            "background_event_fraction_2_4sd": np.nan,
            "background_event_fraction_gt_4sd": np.nan,
            "background_event_count_ge_2sd": np.nan,
            "background_event_rate_ge_2sd_hz": np.nan,
            "background_event_max_cluster_span_s": np.nan,
            "background_event_max_raw_onsets_per_cluster": np.nan,
        }

    onsets = _event_onsets(event_values, threshold)
    duration_s = (
        float(timestamps[-1] - timestamps[0])
        if len(timestamps) > 1
        else np.nan
    )
    exclude = np.zeros(values.shape[0], dtype=bool)
    pre_frames = max(0, int(round(background_exclude_pre_s * sample_rate_hz)))
    post_frames = max(1, int(round(background_exclude_post_s * sample_rate_hz)))
    for onset in onsets:
        start = max(0, int(onset) - pre_frames)
        stop = min(len(values), int(onset) + post_frames + 1)
        exclude[start:stop] = True
    background = values[(~exclude) & np.isfinite(values)]
    background_median = float(np.nanmedian(background)) if background.size else np.nan
    noise_sd = _mad_sd(background)

    clusters = _merged_event_clusters(onsets, sample_rate_hz, merge_within_s)
    local_pre_frames = max(1, int(round(local_baseline_pre_s * sample_rate_hz)))
    peak_post_frames = max(1, int(round(peak_search_post_s * sample_rate_hz)))
    amplitudes = []
    cluster_counts = []
    cluster_spans_s = []
    for start, last, count in clusters:
        midpoint = int(round((start + last) / 2.0))
        baseline_start = max(0, midpoint - local_pre_frames)
        baseline_window = values[baseline_start:midpoint]
        local_baseline = (
            float(np.nanmedian(baseline_window[np.isfinite(baseline_window)]))
            if np.any(np.isfinite(baseline_window))
            else background_median
        )
        response_stop = min(len(values), midpoint + peak_post_frames + 1)
        response = values[midpoint:response_stop]
        if not np.any(np.isfinite(response)) or not np.isfinite(local_baseline):
            continue
        amplitudes.append(float(np.nanmax(response) - local_baseline))
        cluster_counts.append(count)
        cluster_spans_s.append(float((last - start) / sample_rate_hz))

    amplitudes = np.asarray(amplitudes, dtype=float)
    valid_noise = np.isfinite(noise_sd) and noise_sd > 0
    amp_z = amplitudes / noise_sd if valid_noise else np.full(amplitudes.shape, np.nan)
    valid_z = amp_z[np.isfinite(amp_z)]
    valid_duration = np.isfinite(duration_s) and duration_s > 0
    count_ge_low = int(np.sum(valid_z >= low_sd)) if valid_z.size else 0

    return {
        "background_sample_fraction": float(background.size / values.size) if values.size else np.nan,
        "background_median_dff": background_median,
        "background_noise_sd": noise_sd,
        "background_event_raw_onset_count": int(len(onsets)),
        "background_event_cluster_count": int(len(clusters)),
        "background_event_mean_raw_onsets_per_cluster": (
            float(np.nanmean(cluster_counts)) if cluster_counts else np.nan
        ),
        "background_event_median_amp_dff": (
            float(np.nanmedian(amplitudes)) if amplitudes.size else np.nan
        ),
        "background_event_p95_amp_dff": (
            float(np.nanpercentile(amplitudes, 95)) if amplitudes.size else np.nan
        ),
        "background_event_median_amp_noise_units": (
            float(np.nanmedian(valid_z)) if valid_z.size else np.nan
        ),
        "background_event_p95_amp_noise_units": (
            float(np.nanpercentile(valid_z, 95)) if valid_z.size else np.nan
        ),
        "background_event_fraction_lt_2sd": (
            float(np.mean(valid_z < low_sd)) if valid_z.size else np.nan
        ),
        "background_event_fraction_2_4sd": (
            float(np.mean((valid_z >= low_sd) & (valid_z < high_sd)))
            if valid_z.size
            else np.nan
        ),
        "background_event_fraction_gt_4sd": (
            float(np.mean(valid_z >= high_sd)) if valid_z.size else np.nan
        ),
        "background_event_count_ge_2sd": count_ge_low,
        "background_event_rate_ge_2sd_hz": (
            count_ge_low / duration_s if valid_duration else np.nan
        ),
        "background_event_max_cluster_span_s": (
            float(np.nanmax(cluster_spans_s)) if cluster_spans_s else np.nan
        ),
        "background_event_max_raw_onsets_per_cluster": (
            int(np.nanmax(cluster_counts)) if cluster_counts else 0
        ),
    }


def trace_threshold_event_metrics(
    trace: np.ndarray,
    timestamps: np.ndarray,
    *,
    baseline_percentile: float = 10.0,
    noise_percentile: float = 50.0,
    low_sd: float = 2.0,
    high_sd: float = 4.0,
    merge_within_s: float = 0.5,
) -> dict[str, float]:
    """
    Detect dF/F excursions after estimating background from the trace itself.

    This is a background-first complement to ``background_event_amplitude_metrics``.
    It does not use the Allen event trace to define candidate events. Instead it
    estimates a low-percentile baseline, estimates robust noise from values near
    the lower half of the trace, identifies contiguous excursions above
    ``baseline + low_sd * noise_sd``, and merges close excursions into one
    trace-defined event.
    """
    values = np.asarray(trace, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if values.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if len(timestamps) != len(values):
        raise ValueError("timestamps must have one value per trace sample")

    finite = values[np.isfinite(values)]
    sample_rate_hz = _sample_rate_hz_from_timestamps(timestamps)
    duration_s = (
        float(timestamps[-1] - timestamps[0])
        if len(timestamps) > 1
        else np.nan
    )
    if finite.size == 0 or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return {
            "trace_event_baseline_p10_dff": np.nan,
            "trace_event_noise_sd": np.nan,
            "trace_event_cluster_count": np.nan,
            "trace_event_count_ge_2sd": np.nan,
            "trace_event_rate_ge_2sd_hz": np.nan,
            "trace_event_median_amp_noise_units": np.nan,
            "trace_event_p95_amp_noise_units": np.nan,
            "trace_event_fraction_lt_2sd": np.nan,
            "trace_event_fraction_2_4sd": np.nan,
            "trace_event_fraction_gt_4sd": np.nan,
            "trace_event_max_cluster_span_s": np.nan,
        }

    baseline = float(np.nanpercentile(finite, baseline_percentile))
    noise_cutoff = float(np.nanpercentile(finite, noise_percentile))
    noise_samples = finite[finite <= noise_cutoff]
    noise_sd = _mad_sd(noise_samples)
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        noise_sd = _mad_sd(finite)

    valid_noise = np.isfinite(noise_sd) and noise_sd > 0
    if not valid_noise:
        return {
            "trace_event_baseline_p10_dff": baseline,
            "trace_event_noise_sd": noise_sd,
            "trace_event_cluster_count": 0,
            "trace_event_count_ge_2sd": 0,
            "trace_event_rate_ge_2sd_hz": np.nan,
            "trace_event_median_amp_noise_units": np.nan,
            "trace_event_p95_amp_noise_units": np.nan,
            "trace_event_fraction_lt_2sd": np.nan,
            "trace_event_fraction_2_4sd": np.nan,
            "trace_event_fraction_gt_4sd": np.nan,
            "trace_event_max_cluster_span_s": np.nan,
        }

    threshold = baseline + low_sd * noise_sd
    above = np.isfinite(values) & (values >= threshold)
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    stops = np.flatnonzero(above & ~np.r_[above[1:], False])
    merge_frames = max(1, int(round(merge_within_s * sample_rate_hz)))
    clusters: list[tuple[int, int]] = []
    if len(starts):
        current_start = int(starts[0])
        current_stop = int(stops[0])
        for start, stop in zip(starts[1:], stops[1:]):
            start = int(start)
            stop = int(stop)
            if start - current_stop <= merge_frames:
                current_stop = stop
            else:
                clusters.append((current_start, current_stop))
                current_start, current_stop = start, stop
        clusters.append((current_start, current_stop))

    amplitudes = []
    spans_s = []
    for start, stop in clusters:
        response = values[start : stop + 1]
        if not np.any(np.isfinite(response)):
            continue
        amplitudes.append(float(np.nanmax(response) - baseline))
        spans_s.append(float((stop - start) / sample_rate_hz))

    amplitudes = np.asarray(amplitudes, dtype=float)
    amp_z = amplitudes / noise_sd if amplitudes.size else np.array([], dtype=float)
    valid_z = amp_z[np.isfinite(amp_z)]
    count_ge_low = int(np.sum(valid_z >= low_sd)) if valid_z.size else 0
    valid_duration = np.isfinite(duration_s) and duration_s > 0

    return {
        "trace_event_baseline_p10_dff": baseline,
        "trace_event_noise_sd": noise_sd,
        "trace_event_cluster_count": int(len(clusters)),
        "trace_event_count_ge_2sd": count_ge_low,
        "trace_event_rate_ge_2sd_hz": (
            count_ge_low / duration_s if valid_duration else np.nan
        ),
        "trace_event_median_amp_noise_units": (
            float(np.nanmedian(valid_z)) if valid_z.size else np.nan
        ),
        "trace_event_p95_amp_noise_units": (
            float(np.nanpercentile(valid_z, 95)) if valid_z.size else np.nan
        ),
        "trace_event_fraction_lt_2sd": (
            float(np.mean(valid_z < low_sd)) if valid_z.size else np.nan
        ),
        "trace_event_fraction_2_4sd": (
            float(np.mean((valid_z >= low_sd) & (valid_z < high_sd)))
            if valid_z.size
            else np.nan
        ),
        "trace_event_fraction_gt_4sd": (
            float(np.mean(valid_z >= high_sd)) if valid_z.size else np.nan
        ),
        "trace_event_max_cluster_span_s": (
            float(np.nanmax(spans_s)) if spans_s else np.nan
        ),
    }


def event_cluster_amplitude_table(
    dff: np.ndarray,
    events: np.ndarray,
    timestamps: np.ndarray,
    *,
    roi_metrics: pd.DataFrame | None = None,
    plane: str | None = None,
    event_threshold: float = 0.0,
    merge_within_s: float = 0.5,
    local_baseline_pre_s: float = 0.5,
    peak_search_post_s: float = 2.0,
    low_sd: float = 2.0,
    high_sd: float = 4.0,
    long_span_s: float = 3.0,
) -> pd.DataFrame:
    """
    Return one row per merged event cluster with amplitude and class labels.

    Clusters with first-to-last onset span greater than or equal to
    ``long_span_s`` are labeled as their own event type and should be excluded
    from the three-way amplitude composition used for ROI clustering.
    """
    matrix = np.asarray(dff, dtype=float)
    event_matrix = np.asarray(events, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"dff must be time-by-ROI, got shape {matrix.shape}")
    if event_matrix.shape != matrix.shape:
        raise ValueError(f"events must match dff shape {matrix.shape}, got {event_matrix.shape}")
    if len(timestamps) != matrix.shape[0]:
        raise ValueError("timestamps must have one value per dF/F timepoint")

    sample_rate_hz = _sample_rate_hz_from_timestamps(timestamps)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return pd.DataFrame()

    noise_by_roi: dict[int, float] = {}
    if roi_metrics is not None and "background_noise_sd" in roi_metrics.columns:
        for _, row in roi_metrics.iterrows():
            if pd.isna(row.get("roi_index")):
                continue
            noise_by_roi[int(row["roi_index"])] = _jsonable_float(row["background_noise_sd"])

    baseline_pre_frames = max(1, int(round(local_baseline_pre_s * sample_rate_hz)))
    peak_post_frames = max(1, int(round(peak_search_post_s * sample_rate_hz)))
    rows = []
    for roi_index in range(matrix.shape[1]):
        trace = matrix[:, roi_index]
        event_trace = event_matrix[:, roi_index]
        noise_sd = noise_by_roi.get(roi_index)
        if noise_sd is None or not np.isfinite(noise_sd) or noise_sd <= 0:
            noise_sd = robust_event_snr(trace)["robust_event_noise_sd"]

        onsets = _event_onsets(event_trace, event_threshold)
        clusters = _merged_event_clusters(onsets, sample_rate_hz, merge_within_s)
        for cluster_index, (first, last, raw_count) in enumerate(clusters):
            midpoint = int(round((first + last) / 2.0))
            baseline_start = max(0, midpoint - baseline_pre_frames)
            response_stop = min(matrix.shape[0], midpoint + peak_post_frames + 1)
            baseline_window = trace[baseline_start:midpoint]
            response = trace[midpoint:response_stop]
            local_baseline = (
                float(np.nanmedian(baseline_window[np.isfinite(baseline_window)]))
                if np.any(np.isfinite(baseline_window))
                else np.nan
            )
            peak = (
                float(np.nanmax(response))
                if np.any(np.isfinite(response))
                else np.nan
            )
            amplitude = peak - local_baseline if np.isfinite(local_baseline) else np.nan
            amplitude_noise_units = (
                amplitude / noise_sd
                if np.isfinite(amplitude) and np.isfinite(noise_sd) and noise_sd > 0
                else np.nan
            )
            cluster_span_s = float((last - first) / sample_rate_hz)
            if cluster_span_s >= long_span_s:
                event_type = "long_gt_3s"
            elif not np.isfinite(amplitude_noise_units):
                event_type = "missing_amplitude"
            elif amplitude_noise_units < low_sd:
                event_type = "lt_2sd"
            elif amplitude_noise_units < high_sd:
                event_type = "2_4sd"
            else:
                event_type = "gt_4sd"

            row = {
                "roi_index": int(roi_index),
                "cluster_index": int(cluster_index),
                "first_onset_frame": int(first),
                "last_onset_frame": int(last),
                "midpoint_frame": int(midpoint),
                "first_onset_s": float(timestamps[first]),
                "last_onset_s": float(timestamps[last]),
                "midpoint_s": float(timestamps[midpoint]),
                "cluster_span_s": cluster_span_s,
                "raw_onsets_in_cluster": int(raw_count),
                "local_baseline_dff": local_baseline,
                "peak_dff": peak,
                "event_amplitude_dff": amplitude,
                "background_noise_sd": float(noise_sd) if np.isfinite(noise_sd) else np.nan,
                "event_amplitude_noise_units": amplitude_noise_units,
                "event_type": event_type,
                "is_long_gt_3s": bool(cluster_span_s >= long_span_s),
            }
            if plane is not None:
                row["plane"] = str(plane)
            rows.append(row)
    return pd.DataFrame(rows)


def _jsonable_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def roi_event_composition_from_cluster_table(
    cluster_table: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("plane", "roi_index"),
) -> pd.DataFrame:
    """
    Calculate ROI event-type fractions, excluding long clusters from 3-way mix.

    The output includes a separate long-cluster fraction across all clusters.
    The ``nonlong_event_fraction_*`` columns sum to one over non-long clusters
    only, so long sustained windows do not influence the three-way ROI cluster.
    """
    if cluster_table.empty:
        return pd.DataFrame(columns=[*group_cols])
    required = set(group_cols).union({"event_type"})
    missing = sorted(required.difference(cluster_table.columns))
    if missing:
        raise KeyError(f"Missing cluster table columns: {missing}")

    type_counts = (
        cluster_table.groupby([*group_cols, "event_type"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["lt_2sd", "2_4sd", "gt_4sd", "long_gt_3s", "missing_amplitude"]:
        if col not in type_counts.columns:
            type_counts[col] = 0

    type_counts["n_event_clusters_total"] = (
        type_counts["lt_2sd"]
        + type_counts["2_4sd"]
        + type_counts["gt_4sd"]
        + type_counts["long_gt_3s"]
        + type_counts["missing_amplitude"]
    )
    type_counts["n_event_clusters_nonlong"] = (
        type_counts["lt_2sd"] + type_counts["2_4sd"] + type_counts["gt_4sd"]
    )
    denom = type_counts["n_event_clusters_nonlong"].replace(0, np.nan)
    type_counts["nonlong_event_fraction_lt_2sd"] = type_counts["lt_2sd"] / denom
    type_counts["nonlong_event_fraction_2_4sd"] = type_counts["2_4sd"] / denom
    type_counts["nonlong_event_fraction_gt_4sd"] = type_counts["gt_4sd"] / denom
    type_counts["long_gt_3s_fraction_all_clusters"] = (
        type_counts["long_gt_3s"]
        / type_counts["n_event_clusters_total"].replace(0, np.nan)
    )
    type_counts["has_long_gt_3s_event_cluster"] = type_counts["long_gt_3s"] > 0

    rename_counts = {
        "lt_2sd": "n_lt_2sd_event_clusters",
        "2_4sd": "n_2_4sd_event_clusters",
        "gt_4sd": "n_gt_4sd_event_clusters",
        "long_gt_3s": "n_long_gt_3s_event_clusters",
        "missing_amplitude": "n_missing_amplitude_event_clusters",
    }
    return type_counts.rename(columns=rename_counts)


EVENT_COMPOSITION_COLUMNS = [
    "background_event_fraction_lt_2sd",
    "background_event_fraction_2_4sd",
    "background_event_fraction_gt_4sd",
]


def event_composition_labels(
    metrics: pd.DataFrame,
    *,
    dominance_threshold: float = 0.55,
    ambiguity_margin: float = 0.15,
) -> pd.DataFrame:
    """
    Label ROIs by which event-amplitude category dominates their events.

    Fractions are expected to represent events below 2 background SD, between
    2-4 background SD, and above 4 background SD. ROIs are labeled ambiguous
    when no fraction is dominant enough, or when the top two fractions are too
    close to interpret as primarily one event type.
    """
    missing = [col for col in EVENT_COMPOSITION_COLUMNS if col not in metrics.columns]
    if missing:
        raise KeyError(f"Missing event composition columns: {missing}")

    out = metrics.copy()
    values = out[EVENT_COMPOSITION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    arr = values.to_numpy(dtype=float)
    sums = np.nansum(arr, axis=1, keepdims=True)
    valid = np.isfinite(arr).all(axis=1) & (sums[:, 0] > 0)
    normalized = np.full_like(arr, np.nan, dtype=float)
    normalized[valid] = arr[valid] / sums[valid]

    order = np.argsort(np.nan_to_num(normalized, nan=-np.inf), axis=1)
    top_idx = order[:, -1]
    second_idx = order[:, -2]
    top = normalized[np.arange(len(normalized)), top_idx]
    second = normalized[np.arange(len(normalized)), second_idx]
    margin = top - second
    names = np.array(["mostly_sub_2sd", "mostly_2_4sd", "mostly_gt_4sd"], dtype=object)
    labels = np.full(len(out), "no_events_or_missing", dtype=object)
    dominant = valid & (top >= dominance_threshold) & (margin >= ambiguity_margin)
    labels[valid & ~dominant] = "mixed_or_ambiguous"
    labels[dominant] = names[top_idx[dominant]]

    entropy = np.full(len(out), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy[valid] = -np.nansum(
            normalized[valid] * np.log2(np.clip(normalized[valid], 1e-12, 1.0)),
            axis=1,
        ) / np.log2(3)

    out["event_composition_label"] = labels
    out["event_composition_dominant_fraction"] = top
    out["event_composition_second_fraction"] = second
    out["event_composition_margin"] = margin
    out["event_composition_entropy"] = entropy
    return out


def event_composition_kmeans(
    metrics: pd.DataFrame,
    *,
    n_clusters: int = 3,
    n_iter: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster ROIs by event-amplitude composition using deterministic k-means.

    This intentionally avoids adding a scikit-learn dependency. Initial centers
    are chosen from quantiles of the high-amplitude event fraction.
    """
    missing = [col for col in EVENT_COMPOSITION_COLUMNS if col not in metrics.columns]
    if missing:
        raise KeyError(f"Missing event composition columns: {missing}")

    out = metrics.copy()
    values = out[EVENT_COMPOSITION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    arr = values.to_numpy(dtype=float)
    sums = np.nansum(arr, axis=1, keepdims=True)
    valid = np.isfinite(arr).all(axis=1) & (sums[:, 0] > 0)
    x = arr[valid] / sums[valid]
    if x.shape[0] < n_clusters:
        out["event_composition_cluster"] = np.nan
        return out, pd.DataFrame()

    order = np.argsort(x[:, 2])
    quantiles = np.linspace(0, len(order) - 1, n_clusters).round().astype(int)
    centers = x[order[quantiles]].copy()
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(max(1, int(n_iter))):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster in range(n_clusters):
            members = x[new_labels == cluster]
            if len(members):
                new_centers[cluster] = np.mean(members, axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels = new_labels
        centers = new_centers

    center_order = np.argsort(centers[:, 2])
    remap = {old: new for new, old in enumerate(center_order)}
    ordered_labels = np.array([remap[label] for label in labels], dtype=int)
    ordered_centers = centers[center_order]

    full_labels = np.full(len(out), np.nan)
    full_labels[valid] = ordered_labels
    out["event_composition_cluster"] = full_labels

    centroid_rows = []
    for cluster, center in enumerate(ordered_centers):
        members = x[ordered_labels == cluster]
        distances = (
            np.linalg.norm(members - center[None, :], axis=1)
            if len(members)
            else np.array([], dtype=float)
        )
        centroid_rows.append(
            {
                "event_composition_cluster": cluster,
                "n_rois": int(len(members)),
                "centroid_fraction_lt_2sd": float(center[0]),
                "centroid_fraction_2_4sd": float(center[1]),
                "centroid_fraction_gt_4sd": float(center[2]),
                "mean_distance_to_centroid": (
                    float(np.mean(distances)) if len(distances) else np.nan
                ),
            }
        )
    return out, pd.DataFrame(centroid_rows)


def long_event_window_flags(
    clusters: pd.DataFrame,
    *,
    warning_span_s: float = 3.0,
    severe_span_s: float = 5.0,
    extreme_span_s: float = 8.0,
) -> pd.DataFrame:
    """Summarize long merged-event clusters per ROI for trace inspection."""
    required = {"plane", "roi_index", "cluster_span_s", "raw_onsets_in_cluster"}
    missing = sorted(required.difference(clusters.columns))
    if missing:
        raise KeyError(f"Missing cluster columns: {missing}")
    grouped = clusters.groupby(["plane", "roi_index"], dropna=False)
    out = grouped.agg(
        event_cluster_count=("cluster_span_s", "size"),
        max_event_cluster_span_s=("cluster_span_s", "max"),
        p95_event_cluster_span_s=("cluster_span_s", lambda x: float(np.nanpercentile(x, 95))),
        max_raw_onsets_in_cluster=("raw_onsets_in_cluster", "max"),
    ).reset_index()
    for threshold, name in [
        (warning_span_s, "warning"),
        (severe_span_s, "severe"),
        (extreme_span_s, "extreme"),
    ]:
        counts = (
            clusters.loc[clusters["cluster_span_s"] >= threshold]
            .groupby(["plane", "roi_index"], dropna=False)
            .size()
            .rename(f"n_{name}_long_event_clusters")
            .reset_index()
        )
        out = out.merge(counts, on=["plane", "roi_index"], how="left")
        out[f"n_{name}_long_event_clusters"] = (
            out[f"n_{name}_long_event_clusters"].fillna(0).astype(int)
        )
        out[f"has_{name}_long_event_cluster"] = (
            out[f"n_{name}_long_event_clusters"] > 0
        )
    return out


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
    sample_rate_hz = _sample_rate_hz_from_timestamps(timestamps)
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


def event_exponential_gaussian_fit_metric(
    event_trace: np.ndarray,
    *,
    threshold: float = 0.0,
    min_events: int = 20,
) -> dict[str, float]:
    """
    Score whether extracted events look exponential and model residuals Gaussian.

    Positive event amplitudes are compared with a fitted exponential
    distribution. The sorted-event residuals relative to the expected
    exponential quantiles are then compared with a fitted Gaussian. The combined
    score is high when both Kolmogorov-Smirnov statistics are low.
    """
    values = np.asarray(event_trace, dtype=float)
    events = values[np.isfinite(values) & (values > threshold)]
    if events.size < int(min_events):
        return {
            "event_exp_gauss_fit_score": np.nan,
            "event_exponential_scale": np.nan,
            "event_exponential_ks_stat": np.nan,
            "event_exponential_ks_pvalue": np.nan,
            "event_model_residual_gaussian_ks_stat": np.nan,
            "event_model_residual_gaussian_ks_pvalue": np.nan,
        }

    events = np.sort(events)
    scale = float(np.nanmean(events))
    if not np.isfinite(scale) or scale <= 0:
        return {
            "event_exp_gauss_fit_score": np.nan,
            "event_exponential_scale": np.nan,
            "event_exponential_ks_stat": np.nan,
            "event_exponential_ks_pvalue": np.nan,
            "event_model_residual_gaussian_ks_stat": np.nan,
            "event_model_residual_gaussian_ks_pvalue": np.nan,
        }

    exp_ks = kstest(events, "expon", args=(0.0, scale))
    probabilities = (np.arange(events.size, dtype=float) + 0.5) / events.size
    expected = expon.ppf(probabilities, loc=0.0, scale=scale)
    residuals = events - expected
    residual_sd = float(np.nanstd(residuals))
    if np.isfinite(residual_sd) and residual_sd > 0:
        residual_mu = float(np.nanmean(residuals))
        gauss_ks = kstest(residuals, "norm", args=(residual_mu, residual_sd))
        gauss_stat = float(gauss_ks.statistic)
        gauss_pvalue = float(gauss_ks.pvalue)
    else:
        gauss_stat = np.nan
        gauss_pvalue = np.nan

    stats = [float(exp_ks.statistic), gauss_stat]
    finite_stats = [stat for stat in stats if np.isfinite(stat)]
    score = 1.0 - float(np.mean(finite_stats)) if finite_stats else np.nan
    return {
        "event_exp_gauss_fit_score": score,
        "event_exponential_scale": scale,
        "event_exponential_ks_stat": float(exp_ks.statistic),
        "event_exponential_ks_pvalue": float(exp_ks.pvalue),
        "event_model_residual_gaussian_ks_stat": gauss_stat,
        "event_model_residual_gaussian_ks_pvalue": gauss_pvalue,
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
    min_events_for_distribution_fit: int = 20,
    background_event_merge_within_s: float = 0.5,
    background_event_local_baseline_pre_s: float | None = None,
    background_event_peak_search_post_s: float | None = None,
    background_event_low_sd: float = 2.0,
    background_event_high_sd: float = 4.0,
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
            signal_noise_component_metrics(
                trace,
                sigma=gaussian_sigma,
                baseline_bins=baseline_bins,
            )
        )
        row.update(
            baseline_stability_metrics(
                trace,
                snr_metrics["robust_event_noise_sd"],
                n_bins=baseline_bins,
            )
        )
        row.update(
            trace_threshold_event_metrics(
                trace,
                timestamps,
                low_sd=background_event_low_sd,
                high_sd=background_event_high_sd,
                merge_within_s=background_event_merge_within_s,
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
            row.update(
                background_event_amplitude_metrics(
                    trace,
                    event_matrix[:, roi_index],
                    timestamps,
                    threshold=event_threshold,
                    background_exclude_pre_s=kernel_pre_s,
                    background_exclude_post_s=kernel_post_s,
                    merge_within_s=background_event_merge_within_s,
                    local_baseline_pre_s=(
                        kernel_pre_s
                        if background_event_local_baseline_pre_s is None
                        else background_event_local_baseline_pre_s
                    ),
                    peak_search_post_s=(
                        kernel_post_s
                        if background_event_peak_search_post_s is None
                        else background_event_peak_search_post_s
                    ),
                    low_sd=background_event_low_sd,
                    high_sd=background_event_high_sd,
                )
            )
            row.update(
                event_exponential_gaussian_fit_metric(
                    event_matrix[:, roi_index],
                    threshold=event_threshold,
                    min_events=min_events_for_distribution_fit,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)
