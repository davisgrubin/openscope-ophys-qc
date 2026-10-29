#!/usr/bin/env python3
"""Three candidate ROI SNR metrics for the mesoscope column of Figure 7.

The three schemes being compared, each named for who proposed it:

M_ido    Event-amplitude composition (PR #171 / discussion #156).
         Events are detected, every event amplitude is expressed in SDs above a
         non-event baseline of the *raw* trace, amplitudes are binned
         < 2 / 2-4 / > 4 SD, and each ROI is described by the three fractions.
         k-means (k = 3) on the standardised fractions gives Low / Intermediate /
         High SNR classes, ordered by the centroid's > 4 SD fraction.
         On the mesoscope the detector is the Allen pipeline's own OASIS event
         trace where it exists (that is what #156 specifies), with a matched
         filter as fallback for sessions that ship no events.

M_maedeh Amplitude SNR (2026-08-25 meeting).
         Noise floor from the SD of successive differences -- std(diff(f))/sqrt(2),
         the "std of the dF" estimator she named -- and signal as the 95th
         percentile of *peak* amplitudes above the baseline. SNR = signal / noise.
         The 95th-percentile-of-the-whole-trace variant is reported alongside it
         because her wording admits both readings.

M_davis  Robust event SNR (SessionPlaneROISNR.ipynb, metric 2).
         (P95 - P50) of dF/F over the MAD SD of the residual left after
         Gaussian-smoothing the trace, so the noise term is explicitly the fast
         component. `exceptional_event_score` (metric 1 of the same notebook) is
         carried as a secondary column.

Every function takes one 1-D dF/F trace and returns a flat dict, so the three
schemes can be run in a single pass over an ROI and compared per ROI.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import log_ndtr

# Indicator decay time constant. This dataset is jGCaMP8s (see
# documentation/filetype_reference.md), whose single-AP decay half-time is
# ~0.30 s, i.e. tau = t_half / ln 2 ~ 0.43 s. At the 9.48 Hz frame rate of the
# planar mesoscope that is ~4 frames, so the amplitude windows below span a
# handful of samples rather than one.
#
# tau enters in two places: the fallback matched filter (used only when a
# session ships no OASIS event trace) and the local-baseline / peak windows of
# the event-amplitude measurement, which both schemes' amplitudes depend on.
TAU_JGCAMP8S_S = 0.43

# Retained under the old name so nothing that imported it breaks; the value is
# the jGCaMP8s constant, not GCaMP6f.
TAU_GCAMP6F_S = TAU_JGCAMP8S_S

# Nominal planar-mesoscope frame rate. Timestamps are always read from the NWB
# rather than assumed -- this is only a fallback and a sanity check.
FRAME_RATE_HZ_NOMINAL = 9.48

SD_DETECT_MESOSCOPE = 2.0   # detection floor of the mesoscope scheme (#156)
SD_LARGE = 4.0              # large-event boundary, identical across columns


# --------------------------------------------------------------------------
# shared noise / baseline estimators
# --------------------------------------------------------------------------

def mad_sd(values: np.ndarray) -> float:
    """Robust SD from the median absolute deviation."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def successive_difference_sd(values: np.ndarray) -> float:
    """Maedeh's noise floor: SD of frame-to-frame differences / sqrt(2).

    Differencing removes any signal slower than one frame, so what is left is
    dominated by shot/read noise. The sqrt(2) undoes the variance doubling that
    differencing two independent samples introduces, putting the estimate back
    on the per-sample scale.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan
    return float(np.std(np.diff(v)) / np.sqrt(2.0))


def half_sample_mode(values: np.ndarray) -> float:
    """Recursive half-sample mode -- the centre of the densest half."""
    data = np.sort(np.asarray(values, dtype=float))
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan
    if data.size == 1:
        return float(data[0])
    if data.size == 2:
        return float(np.mean(data))
    if data.size == 3:
        left, right = data[1] - data[0], data[2] - data[1]
        if left < right:
            return float(np.mean(data[:2]))
        if right < left:
            return float(np.mean(data[1:]))
        return float(data[1])
    width = (data.size + 1) // 2
    starts = np.arange(data.size - width + 1)
    spans = data[starts + width - 1] - data[starts]
    start = int(starts[np.argmin(spans)])
    return half_sample_mode(data[start:start + width])


def _local_maxima(x: np.ndarray, thr: float, refractory: int) -> np.ndarray:
    """Indices of local maxima above `thr`, thinned by a refractory period."""
    if x.size < 3:
        return np.empty(0, dtype=int)
    cand = np.flatnonzero((x[1:-1] >= thr) & (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:])) + 1
    keep: list[int] = []
    last = -np.inf
    for i in cand:
        if i - last >= refractory:
            keep.append(int(i))
            last = i
        elif keep and x[i] > x[keep[-1]]:
            keep[-1] = int(i)
            last = i
    return np.asarray(keep, dtype=int)


# --------------------------------------------------------------------------
# M_ido -- event-amplitude composition
# --------------------------------------------------------------------------

def matched_filter_kernel(dt: float, tau: float = TAU_JGCAMP8S_S) -> np.ndarray:
    """Unit-energy causal exponential kernel matched to the indicator decay."""
    n = max(3, int(np.ceil(5.0 * tau / dt)))
    k = np.exp(-np.arange(n) * dt / tau)
    return (k / np.linalg.norm(k)).astype(float)


def baseline_noise_lower_half(x: np.ndarray) -> tuple[float, float]:
    """Noise SD from the below-median half, which transients cannot inflate.

    Calcium transients are strictly positive-going, so the lower half of the
    distribution carries noise only. Returns (sd, median).
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 100:
        return np.nan, np.nan
    med = float(np.median(v))
    lower = v[v < med]
    if lower.size < 50:
        return np.nan, med
    sd = float(1.4826 * np.median(med - lower))
    scale = float(np.percentile(v, 95) - med)
    if sd <= 0 or (scale > 0 and sd < 1e-4 * scale):
        return np.nan, med
    return sd, med


def event_indices_from_allen(event_trace: np.ndarray, refractory: int) -> np.ndarray:
    """Event onsets from an Allen/OASIS deconvolved event trace.

    OASIS returns a sparse non-negative amplitude series: every strictly
    positive sample is an inferred event. Contiguous positive runs are collapsed
    to the frame of their maximum so one transient yields one event.
    """
    e = np.asarray(event_trace, dtype=float)
    e = np.where(np.isfinite(e), e, 0.0)
    pos = e > 0
    if not pos.any():
        return np.empty(0, dtype=int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], pos.view(np.int8), [0]))))
    starts, stops = edges[0::2], edges[1::2]
    return np.array([s + int(np.argmax(e[s:t])) for s, t in zip(starts, stops)], dtype=int)


def event_amplitudes_raw(trace: np.ndarray, idx: np.ndarray, dt: float,
                         raw_noise: float, tau: float = TAU_JGCAMP8S_S) -> np.ndarray:
    """Event amplitude on the raw dF/F, in units of raw-trace noise SD.

    Local baseline is the median over [-3tau, -1tau] before the event; the peak
    is the maximum over [0, +2tau] after it. Detection and amplitude measurement
    are deliberately separate, which is what lets a detected event legitimately
    land below 2 SD and makes the three-way composition available at all.
    """
    if idx.size == 0 or not np.isfinite(raw_noise) or raw_noise <= 0:
        return np.full(idx.size, np.nan)
    n = trace.size
    pre_a = max(1, int(round(3 * tau / dt)))
    pre_b = max(1, int(round(tau / dt)))
    post = max(1, int(round(2 * tau / dt)))
    amps = np.full(idx.size, np.nan, dtype=float)
    for k, i in enumerate(idx):
        b0, b1 = max(0, i - pre_a), max(1, i - pre_b)
        base_win, peak_win = trace[b0:b1], trace[i:min(n, i + post + 1)]
        if base_win.size == 0 or peak_win.size == 0:
            continue
        base, peak = np.nanmedian(base_win), np.nanmax(peak_win)
        if np.isfinite(base) and np.isfinite(peak):
            amps[k] = (peak - base) / raw_noise
    return amps


def metric_ido(trace: np.ndarray, dt: float, event_trace: np.ndarray | None = None,
               tau: float = TAU_JGCAMP8S_S) -> dict[str, float]:
    """Event-amplitude composition metric (#156 mesoscope scheme, PR #171)."""
    x = np.asarray(trace, dtype=float)
    finite = np.isfinite(x)
    out: dict[str, float] = {
        "ido_n_events": np.nan, "ido_event_rate_hz": np.nan,
        "ido_noise_sd_raw": np.nan, "ido_frac_lt2sd": np.nan,
        "ido_frac_2_4sd": np.nan, "ido_frac_gt4sd": np.nan,
        "ido_median_event_sd": np.nan, "ido_p90_event_sd": np.nan,
        "ido_false_pos_frac": np.nan, "ido_detector": "none",
    }
    if finite.sum() < 100:
        out["ido_qc_flag"] = "too_short"
        return out

    raw_noise, _ = baseline_noise_lower_half(x)
    out["ido_noise_sd_raw"] = raw_noise
    if not np.isfinite(raw_noise):
        out["ido_qc_flag"] = "no_baseline_noise"
        return out

    refractory = max(1, int(round(tau / dt)))

    if event_trace is not None and np.isfinite(np.asarray(event_trace, float)).any() \
            and np.nansum(np.asarray(event_trace, float)) > 0:
        idx = event_indices_from_allen(event_trace, refractory)
        out["ido_detector"] = "allen_oasis"
        n_false = 0
    else:
        # Fallback: matched-filter detection at the mesoscope 2 SD floor, with
        # negative-going peaks of the same filtered trace as the empirical
        # false-positive null.
        filled = x.copy()
        if not finite.all():
            filled[~finite] = np.interp(np.flatnonzero(~finite),
                                        np.flatnonzero(finite), x[finite])
        filt = np.convolve(filled, matched_filter_kernel(dt, tau), mode="same")
        filt[~finite] = np.nan
        sd_f, med_f = baseline_noise_lower_half(filt)
        if not np.isfinite(sd_f):
            out["ido_qc_flag"] = "no_baseline_noise"
            return out
        z = (filt - med_f) / sd_f
        z_pos = np.where(np.isfinite(z), z, -np.inf)
        z_neg = np.where(np.isfinite(z), -z, -np.inf)
        idx = _local_maxima(z_pos, SD_DETECT_MESOSCOPE, refractory)
        n_false = _local_maxima(z_neg, SD_DETECT_MESOSCOPE, refractory).size
        out["ido_detector"] = "matched_filter"

    out["ido_n_events"] = float(idx.size)
    duration_s = float(finite.sum()) * dt
    out["ido_event_rate_hz"] = idx.size / duration_s if duration_s > 0 else np.nan
    out["ido_false_pos_frac"] = (n_false / idx.size) if idx.size else np.nan

    amps = event_amplitudes_raw(x, idx, dt, raw_noise, tau)
    fin = amps[np.isfinite(amps)]
    if fin.size == 0:
        out["ido_qc_flag"] = "no_events"
        return out
    out["ido_frac_lt2sd"] = float(np.mean(fin < 2.0))
    out["ido_frac_2_4sd"] = float(np.mean((fin >= 2.0) & (fin < SD_LARGE)))
    out["ido_frac_gt4sd"] = float(np.mean(fin >= SD_LARGE))
    out["ido_median_event_sd"] = float(np.median(fin))
    out["ido_p90_event_sd"] = float(np.percentile(fin, 90))
    out["ido_qc_flag"] = "ok"
    return out


CLASS_FEATURES = ["ido_frac_lt2sd", "ido_frac_2_4sd", "ido_frac_gt4sd"]
CLASS_NAMES = ["Low SNR", "Intermediate", "High SNR"]


def assign_ido_classes(metrics, seed: int = 0, features=None, prefix="ido"):
    """k-means (k=3) on the standardised event-amplitude fractions.

    Clusters are named by their centroid's > 4 SD fraction so the labels mean
    the same thing across runs and sessions.
    """
    from scipy.cluster.vq import kmeans2
    import pandas as pd

    feature_cols = features or CLASS_FEATURES
    df = metrics.copy()
    col = f"{prefix}_quality_class"
    usable = df.get(f"{prefix}_qc_flag", pd.Series("ok", index=df.index)) == "ok"
    for f in feature_cols:
        usable &= df[f].notna()
    df[col] = "Excluded (no baseline noise)"
    if usable.sum() < 3:
        return df
    feats = df.loc[usable, feature_cols].to_numpy(dtype=float)
    mu, sigma = feats.mean(0), feats.std(0)
    sigma[sigma == 0] = 1.0
    centroids, labels = kmeans2((feats - mu) / sigma, 3, minit="++", seed=seed)
    order = np.argsort(centroids[:, -1])
    remap = {int(old): new for new, old in enumerate(order)}
    df.loc[usable, col] = [CLASS_NAMES[remap[int(l)]] for l in labels]
    return df


# --------------------------------------------------------------------------
# M_maedeh -- amplitude SNR
# --------------------------------------------------------------------------

def metric_maedeh(trace: np.ndarray, dt: float, tau: float = TAU_JGCAMP8S_S,
                  peak_threshold_sd: float = 2.0) -> dict[str, float]:
    """Amplitude SNR: 95th percentile of peak amplitudes over the noise floor.

    The noise floor is the SD of successive differences, which is the estimator
    named in the meeting ("you differentiate the activity ... the standard
    deviation of that fluctuation"). Peaks are local maxima of the trace at or
    above `peak_threshold_sd` noise SDs over the baseline, measured against the
    trace median; the 95th percentile of their amplitudes is the signal term, so
    a single artefactual excursion cannot set the SNR on its own.

    The whole-trace variant (P95 of the trace itself, not of its peaks) is
    returned alongside, because the meeting wording admits both readings and
    they behave differently for sparsely active ROIs.
    """
    x = np.asarray(trace, dtype=float)
    xf = x[np.isfinite(x)]
    out: dict[str, float] = {
        "maedeh_noise_floor": np.nan, "maedeh_baseline": np.nan,
        "maedeh_signal_p95_peaks": np.nan, "maedeh_snr": np.nan,
        "maedeh_snr_trace_p95": np.nan, "maedeh_n_peaks": np.nan,
        "maedeh_peak_rate_hz": np.nan,
    }
    if xf.size < 10:
        return out

    noise = successive_difference_sd(xf)
    baseline = float(np.median(xf))
    out["maedeh_noise_floor"] = noise
    out["maedeh_baseline"] = baseline
    if not np.isfinite(noise) or noise <= 0:
        return out

    # whole-trace reading
    out["maedeh_snr_trace_p95"] = float((np.percentile(xf, 95) - baseline) / noise)

    # peak reading
    refractory = max(1, int(round(tau / dt)))
    thr = baseline + peak_threshold_sd * noise
    xp = np.where(np.isfinite(x), x, -np.inf)
    idx = _local_maxima(xp, thr, refractory)
    out["maedeh_n_peaks"] = float(idx.size)
    duration_s = float(np.isfinite(x).sum()) * dt
    out["maedeh_peak_rate_hz"] = idx.size / duration_s if duration_s > 0 else np.nan
    if idx.size == 0:
        # No peak clears the floor: the ROI has no resolvable signal. Reporting
        # 0 rather than NaN keeps it in the distribution, which is the honest
        # place for an ROI with nothing above noise.
        out["maedeh_signal_p95_peaks"] = 0.0
        out["maedeh_snr"] = 0.0
        return out
    amps = x[idx] - baseline
    sig = float(np.percentile(amps, 95))
    out["maedeh_signal_p95_peaks"] = sig
    out["maedeh_snr"] = sig / noise
    return out


# --------------------------------------------------------------------------
# M_davis -- robust event SNR
# --------------------------------------------------------------------------

def metric_davis(trace: np.ndarray, sigma: float = 3.0,
                 consecutive_samples: int = 5) -> dict[str, float]:
    """Robust event SNR plus the sustained exceptional-event score.

    robust_event_snr = (P95 - P50) / MAD-SD of the residual after Gaussian
    smoothing, so the numerator is transient amplitude and the denominator is
    explicitly the fast noise component.
    """
    v = np.asarray(trace, dtype=float)
    v = v[np.isfinite(v)]
    out: dict[str, float] = {
        "davis_robust_event_snr": np.nan, "davis_signal_p95_p50": np.nan,
        "davis_noise_fast_mad": np.nan, "davis_exceptional_event_score": np.nan,
    }
    if v.size < 10:
        return out

    signal_amp = float(np.percentile(v, 95) - np.percentile(v, 50))
    residual = v - gaussian_filter1d(v, sigma=float(sigma))
    noise_sd = mad_sd(residual)
    out["davis_signal_p95_p50"] = signal_amp
    out["davis_noise_fast_mad"] = noise_sd
    if np.isfinite(noise_sd) and noise_sd > 0:
        out["davis_robust_event_snr"] = signal_amp / noise_sd

    # sustained exceptional-event score
    mode = half_sample_mode(v)
    lower = mode - v[v < mode]
    lower = lower[np.isfinite(lower) & (lower > 0)]
    if lower.size:
        ex_noise = float(np.sqrt(np.mean(lower ** 2)))
        if np.isfinite(ex_noise) and ex_noise > 0:
            z = (v - mode) / (3.0 * ex_noise)
            logp = log_ndtr(-z)
            w = max(1, int(consecutive_samples))
            moving = np.convolve(logp, np.ones(w), mode="full")[:v.size]
            out["davis_exceptional_event_score"] = float(-np.nanmin(moving))
    return out


def all_metrics(trace: np.ndarray, dt: float,
                event_trace: np.ndarray | None = None) -> dict[str, float]:
    """Run all three schemes on one ROI trace and merge into one flat row."""
    row: dict[str, float] = {}
    row.update(metric_ido(trace, dt, event_trace))
    row.update(metric_maedeh(trace, dt))
    row.update(metric_davis(trace))
    return row
