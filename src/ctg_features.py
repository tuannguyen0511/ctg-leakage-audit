"""Preregistered CTG cleaning, stage-1 features, and 1 Hz DL inputs."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch


FEATURE_NAMES = (
    "baseline", "stv_ms", "ltv", "pct_min_var", "n_accel", "n_decel",
    "decel_area", "n_prolonged_decel", "n_contractions", "lf_power",
    "hf_power", "n_valid_min",
)


def clean_signals(fhr, uc, cfg):
    fhr = np.asarray(fhr, dtype=np.float64).copy()
    uc = np.asarray(uc, dtype=np.float64).copy()
    fhr[(fhr == 0) | (fhr < cfg["fhr_min_bpm"]) | (fhr > cfg["fhr_max_bpm"])] = np.nan
    step = np.abs(np.diff(fhr))
    fhr[1:][step > cfg["fhr_max_step_bpm"]] = np.nan
    uc[uc == 0] = np.nan
    return fhr, uc


def _right_aligned_minutes(fhr, fs):
    width = int(fs * 60)
    left_pad = (-len(fhr)) % width
    padded = np.pad(fhr, (left_pad, 0), constant_values=np.nan)
    return padded.reshape(-1, width)


def _runs(condition):
    condition = np.asarray(condition, dtype=bool)
    edges = np.diff(np.r_[False, condition, False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _stv_ms(fhr, fs, cfg):
    epoch = int(round(cfg["stv_epoch_seconds"] * fs))
    left_pad = (-len(fhr)) % epoch
    rr = np.full(len(fhr), np.nan)
    valid = np.isfinite(fhr)
    rr[valid] = 60000.0 / fhr[valid]
    rr = np.pad(rr, (left_pad, 0), constant_values=np.nan).reshape(-1, epoch)
    epoch_mean = np.full(len(rr), np.nan)
    for i, values in enumerate(rr):
        valid_values = values[np.isfinite(values)]
        if len(valid_values) >= cfg["stv_min_valid_fraction"] * epoch:
            epoch_mean[i] = float(np.mean(valid_values))
    adjacent_valid = np.isfinite(epoch_mean[:-1]) & np.isfinite(epoch_mean[1:])
    return float(np.mean(np.abs(np.diff(epoch_mean)[adjacent_valid]))) if np.any(adjacent_valid) else np.nan


def _spectral_power(fhr, fs, cfg):
    nperseg = int(cfg["welch_nperseg"])
    if len(fhr) < nperseg or np.isfinite(fhr).sum() == 0:
        return np.nan, np.nan
    valid = np.isfinite(fhr)
    interpolated = np.interp(np.arange(len(fhr)), np.flatnonzero(valid), fhr[valid])
    interpolated -= interpolated.mean()
    frequencies, density = welch(
        interpolated, fs=fs, window="hann", nperseg=nperseg,
        noverlap=int(cfg["welch_noverlap"]), detrend="constant", scaling="density",
    )
    powers = []
    for low, high in (cfg["welch_lf_hz"], cfg["welch_hf_hz"]):
        band = (frequencies >= low) & (frequencies <= high)
        powers.append(float(np.trapezoid(density[band], frequencies[band])) if band.sum() >= 2 else np.nan)
    return tuple(powers)


def compute_features(fhr, uc, cfg):
    """Compute from actual samples only. The right edge is the stage-1 endpoint."""
    fs = int(cfg["fs_hz"])
    features = {name: np.nan for name in FEATURE_NAMES}
    minutes = _right_aligned_minutes(fhr, fs)
    ranges = []
    for minute in minutes:
        values = minute[np.isfinite(minute)]
        if len(values):
            ranges.append(float(values.max() - values.min()))
    features["n_valid_min"] = len(ranges)
    if not np.isfinite(fhr).any():
        return features

    rolling_median = pd.Series(fhr).rolling(
        int(cfg["baseline_rolling_minutes"] * 60 * fs), center=True, min_periods=1
    ).median().to_numpy()
    accepted = np.isfinite(fhr) & (np.abs(fhr - rolling_median) <= cfg["baseline_reject_bpm"])
    baseline = float(np.median(fhr[accepted])) if accepted.any() else np.nan
    features["baseline"] = baseline
    features["stv_ms"] = _stv_ms(fhr, fs, cfg)
    features["ltv"] = float(np.mean(ranges)) if ranges else np.nan
    features["pct_min_var"] = float(100 * np.mean(np.asarray(ranges) < cfg["minute_low_variability_bpm"])) if ranges else np.nan

    minimum_samples = int(np.ceil(cfg["event_min_seconds"] * fs))
    for label, condition in (
        ("accel", fhr >= baseline + cfg["accel_threshold_bpm"]),
        ("decel", fhr <= baseline - cfg["decel_threshold_bpm"]),
    ):
        starts, ends = _runs(condition & np.isfinite(fhr))
        qualifying = [(start, end) for start, end in zip(starts, ends) if end - start >= minimum_samples]
        features[f"n_{label}"] = len(qualifying)
        if label == "decel":
            features["decel_area"] = float(sum(np.maximum(baseline - fhr[start:end], 0).sum() / fs for start, end in qualifying))
            features["n_prolonged_decel"] = sum(end - start >= cfg["prolonged_decel_seconds"] * fs for start, end in qualifying)

    if len(uc) == 0 or 100 * np.mean(~np.isfinite(uc)) > cfg["uc_nan_limit_pct"]:
        features["n_contractions"] = np.nan
    else:
        smooth = pd.Series(uc).rolling(
            int(cfg["contraction_smoothing_seconds"] * fs), center=True, min_periods=1
        ).mean().to_numpy(copy=True)
        smooth[~np.isfinite(smooth)] = -np.inf
        peaks, _ = find_peaks(
            smooth,
            height=float(np.nanmedian(uc)) + cfg["contraction_height_above_median"],
            distance=int(cfg["contraction_min_distance_seconds"] * fs),
        )
        features["n_contractions"] = len(peaks)

    features["lf_power"], features["hf_power"] = _spectral_power(fhr, fs, cfg)
    return features


def _mean_1hz(signal, target_seconds, fs):
    padded = np.full(target_seconds * fs, np.nan)
    if len(signal):
        padded[-len(signal):] = signal
    groups = padded.reshape(target_seconds, fs)
    count = np.isfinite(groups).sum(axis=1)
    values = np.divide(np.nansum(groups, axis=1), count, out=np.full(target_seconds, np.nan), where=count > 0)
    return values


def _fill_seconds(values):
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(len(values), dtype=np.float32)
    return np.interp(np.arange(len(values)), np.flatnonzero(valid), values[valid]).astype(np.float32)


def build_dl_1hz(fhr, uc, minutes, cfg):
    fs = int(cfg["fs_hz"])
    seconds = int(minutes * 60)
    fhr_second = _mean_1hz(fhr, seconds, fs)
    mask = (~np.isfinite(fhr_second)).astype(np.float32)
    if mask.all():
        return np.stack((np.zeros(seconds, np.float32), mask, np.zeros(seconds, np.float32)))
    uc_second = _mean_1hz(uc, seconds, fs)
    return np.stack((_fill_seconds(fhr_second), mask, _fill_seconds(uc_second))).astype(np.float32)
