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


def calculate_roi_snr_metrics(
    dff: np.ndarray,
    *,
    gaussian_sigma: float = 3.0,
    consecutive_samples: int = 5,
    exceptional_robust_std: bool = False,
) -> pd.DataFrame:
    """Calculate both SNR metric families for a time-by-ROI dF/F matrix."""
    matrix = np.asarray(dff, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"dff must be a 2D time-by-ROI matrix, got shape {matrix.shape}")

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
        row.update(robust_event_snr(trace, sigma=gaussian_sigma))
        row.update(
            exceptional_event_metric(
                trace,
                consecutive_samples=consecutive_samples,
                robust_std=exceptional_robust_std,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)
