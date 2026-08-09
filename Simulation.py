from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter
from scipy.stats import binomtest, rankdata, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score


SEED = 20260718
HOLDOUT_SEED_OFFSET = 100_003
DIFFICULTY_SEED_OFFSET = 200_003
DEFAULT_WINDOWS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
DEFAULT_NOISE_LEVELS = (0.00, 0.02, 0.05, 0.10, 0.15)
PROTOTYPES = np.array([-1.0, 0.0, 1.0])
MAIN_NOISE = 0.05
MAIN_FRACTION = 0.30
CONFIDENCE_MARGIN = 0.15


@dataclass(frozen=True)
class TrialParameters:
    trial_id: int
    initial_side: int
    outcome: int
    q0: float
    distance_from_separatrix: float
    damage_fraction_requested: float
    damage_fraction_realized: float
    correction_rate: float
    coupling: float
    memory_rate: float
    memory_noise: float
    process_noise: float
    correction_heterogeneity_cv: float
    correction_map_mean: float
    correction_map_sd: float
    correction_map_min: float
    correction_map_max: float
    target_separation_scale: float
    target_overlap_sd: float
    target_overlap_offset: float
    outcome_target_selector: float


@dataclass(frozen=True)
class ParameterRanges:

    log10_delta: tuple[float, float]
    damage_fraction: tuple[float, float]
    correction_rate: tuple[float, float]
    coupling: tuple[float, float]
    memory_rate: tuple[float, float]
    memory_noise: tuple[float, float]
    process_noise: tuple[float, float]
    correction_heterogeneity_cv: tuple[float, float]


BASELINE_RANGES = ParameterRanges(
    log10_delta=(math.log10(0.005), math.log10(0.35)),
    damage_fraction=(0.18, 0.55),
    correction_rate=(0.60, 1.15),
    coupling=(0.06, 0.24),
    memory_rate=(1.5, 3.0),
    memory_noise=(0.15, 0.30),
    process_noise=(0.003, 0.012),
    correction_heterogeneity_cv=(0.08, 0.22),
)


SHIFTED_HOLDOUT_RANGES = ParameterRanges(
    log10_delta=(math.log10(0.004), math.log10(0.30)),
    damage_fraction=(0.22, 0.60),
    correction_rate=(0.50, 1.25),
    coupling=(0.04, 0.28),
    memory_rate=(1.3, 3.2),
    memory_noise=(0.12, 0.34),
    process_noise=(0.004, 0.014),
    correction_heterogeneity_cv=(0.10, 0.26),
)


def masked_laplacian(field: np.ndarray, mask: np.ndarray) -> np.ndarray:

    field = np.asarray(field)
    out = np.zeros_like(field, dtype=float)
    degree = np.zeros(mask.shape, dtype=float)

    vertical = mask[:-1, :] & mask[1:, :]
    dv = field[..., :-1, :] - field[..., 1:, :]
    out[..., 1:, :] += dv * vertical
    out[..., :-1, :] -= dv * vertical
    degree[1:, :] += vertical
    degree[:-1, :] += vertical

    horizontal = mask[:, :-1] & mask[:, 1:]
    dh = field[..., :, :-1] - field[..., :, 1:]
    out[..., :, 1:] += dh * horizontal
    out[..., :, :-1] -= dh * horizontal
    degree[:, 1:] += horizontal
    degree[:, :-1] += horizontal

    safe_degree = np.where(degree > 0, degree, 1.0)
    out = out / safe_degree
    out[..., ~mask] = 0.0
    return out


def anisotropic_masked_laplacian(
    field: np.ndarray,
    mask: np.ndarray,
    horizontal_weight: float = 1.5,
    vertical_weight: float = 0.5,
) -> np.ndarray:

    field = np.asarray(field)
    out = np.zeros_like(field, dtype=float)
    degree = np.zeros(mask.shape, dtype=float)

    vertical = mask[:-1, :] & mask[1:, :]
    dv = field[..., :-1, :] - field[..., 1:, :]
    out[..., 1:, :] += vertical_weight * dv * vertical
    out[..., :-1, :] -= vertical_weight * dv * vertical
    degree[1:, :] += vertical
    degree[:-1, :] += vertical

    horizontal = mask[:, :-1] & mask[:, 1:]
    dh = field[..., :, :-1] - field[..., :, 1:]
    out[..., :, 1:] += horizontal_weight * dh * horizontal
    out[..., :, :-1] -= horizontal_weight * dh * horizontal
    degree[:, 1:] += horizontal
    degree[:, :-1] += horizontal

    safe_degree = np.where(degree > 0, degree, 1.0)
    out = out / safe_degree
    out[..., ~mask] = 0.0
    return out


def make_geometry(height: int = 22, width: int = 44) -> tuple[np.ndarray, ...]:

    x_axis = np.linspace(-1.0, 1.0, width)
    y_axis = np.linspace(-1.0, 1.0, height)
    x, y = np.meshgrid(x_axis, y_axis)
    mask = (x / 1.0) ** 2 + (y / 0.72) ** 2 <= 1.0

    original_raw = np.zeros_like(x)
    original_raw[x <= -0.42] = 1.0
    original_raw[x >= 0.42] = -1.0

    alternate_raw = np.zeros_like(x)
    alternate_raw[np.abs(x) >= 0.42] = 1.0

    original_raw[~mask] = 0.0
    alternate_raw[~mask] = 0.0

    original = original_raw.copy()
    alternate = alternate_raw.copy()
    for _ in range(18):
        original += 0.20 * masked_laplacian(original, mask)
        alternate += 0.20 * masked_laplacian(alternate, mask)
        original[~mask] = 0.0
        alternate[~mask] = 0.0

    original_labels = decode_anatomy(original, mask)
    alternate_labels = decode_anatomy(alternate, mask)
    return mask, x, y, original, alternate, original_labels, alternate_labels


def make_overlap_geometry(
    separation_scale: float,
    height: int = 22,
    width: int = 44,
) -> tuple[np.ndarray, ...]:

    if not 0.0 < separation_scale <= 1.0:
        raise ValueError("separation_scale must be in (0, 1].")
    mask, x, y, original, alternate, original_labels, _ = make_geometry(
        height=height, width=width
    )
    midpoint = 0.5 * (original + alternate)
    scaled_original = midpoint + separation_scale * (original - midpoint)
    scaled_alternate = midpoint + separation_scale * (alternate - midpoint)
    scaled_original[~mask] = 0.0
    scaled_alternate[~mask] = 0.0
    scaled_original_labels = decode_anatomy(scaled_original, mask)
    scaled_alternate_labels = decode_anatomy(scaled_alternate, mask)
    return (
        mask,
        x,
        y,
        scaled_original,
        scaled_alternate,
        scaled_original_labels,
        scaled_alternate_labels,
    )


def decode_anatomy(target: np.ndarray, mask: np.ndarray) -> np.ndarray:

    distances = np.abs(target[..., None] - PROTOTYPES)
    labels = np.argmin(distances, axis=-1).astype(int)
    labels[~mask] = -1
    return labels


def make_damage_mask(
    x: np.ndarray, mask: np.ndarray, requested_fraction: float
) -> np.ndarray:

    active_x = x[mask]
    threshold = np.quantile(active_x, 1.0 - requested_fraction)
    damage = mask & (x >= threshold)
    return damage


def simulate_trial(
    rng: np.random.Generator,
    trial_id: int,
    initial_side: int,
    mask: np.ndarray,
    x: np.ndarray,
    original: np.ndarray,
    alternate: np.ndarray,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, TrialParameters, np.ndarray]:

    log_delta = rng.uniform(np.log10(0.005), np.log10(0.35))
    delta = float(10**log_delta)
    q0 = 0.5 + delta if initial_side == 1 else 0.5 - delta

    requested_damage = float(rng.uniform(0.18, 0.55))
    damage = make_damage_mask(x, mask, requested_damage)
    realized_damage = float(damage.sum() / mask.sum())

    correction_rate = float(rng.uniform(0.60, 1.15))
    coupling = float(rng.uniform(0.06, 0.24))
    memory_rate = float(rng.uniform(1.5, 3.0))
    memory_noise = float(rng.uniform(0.15, 0.30))
    process_noise = float(rng.uniform(0.003, 0.012))
    correction_heterogeneity_cv = float(rng.uniform(0.08, 0.22))
    correction_map = correction_rate * (
        1.0 + rng.normal(0.0, correction_heterogeneity_cv, size=mask.shape)
    )
    correction_map = np.clip(correction_map, 0.25 * correction_rate, 2.0 * correction_rate)
    correction_map[~mask] = 0.0

    voltage = original.copy()
    voltage[damage] = rng.normal(0.0, 0.12, size=int(damage.sum()))
    voltage[mask] += rng.normal(0.0, 0.025, size=int(mask.sum()))
    voltage[~mask] = 0.0

    movie = np.empty((n_steps + 1, *mask.shape), dtype=np.float32)
    q_trace = np.empty(n_steps + 1, dtype=np.float64)
    movie[0] = voltage
    q_trace[0] = q0
    q = q0

    for step in range(1, n_steps + 1):
        instantaneous_target = (1.0 - q) * original + q * alternate
        drift = correction_map * (instantaneous_target - voltage)
        drift += coupling * masked_laplacian(voltage, mask)
        stochastic = process_noise * math.sqrt(dt) * rng.normal(size=voltage.shape)
        voltage = voltage + dt * drift + stochastic
        voltage[~mask] = 0.0
        voltage[mask] = np.clip(voltage[mask], -1.5, 1.5)

        q += dt * memory_rate * q * (1.0 - q) * (q - 0.5)
        q += memory_noise * math.sqrt(dt) * q * (1.0 - q) * rng.normal()
        q = float(np.clip(q, 0.0, 1.0))

        movie[step] = voltage
        q_trace[step] = q

    final_outcome = int(q_trace[-1] >= 0.5)
    params = TrialParameters(
        trial_id=trial_id,
        initial_side=initial_side,
        outcome=final_outcome,
        q0=float(q0),
        distance_from_separatrix=delta,
        damage_fraction_requested=requested_damage,
        damage_fraction_realized=realized_damage,
        correction_rate=correction_rate,
        coupling=coupling,
        memory_rate=memory_rate,
        memory_noise=memory_noise,
        process_noise=process_noise,
        correction_heterogeneity_cv=correction_heterogeneity_cv,
        correction_map_mean=float(correction_map[mask].mean()),
        correction_map_sd=float(correction_map[mask].std(ddof=0)),
        correction_map_min=float(correction_map[mask].min()),
        correction_map_max=float(correction_map[mask].max()),
        target_separation_scale=1.0,
        target_overlap_sd=0.0,
        target_overlap_offset=0.0,
        outcome_target_selector=float(final_outcome),
    )
    return movie, q_trace, params, damage


def simulate_trial_scenario(
    rng: np.random.Generator,
    trial_id: int,
    initial_side: int,
    mask: np.ndarray,
    x: np.ndarray,
    original: np.ndarray,
    alternate: np.ndarray,
    dt: float,
    n_steps: int,
    ranges: ParameterRanges,
    forward_model: str = "linear_isotropic",
    target_separation_scale: float = 1.0,
    target_overlap_sd: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, TrialParameters, np.ndarray]:

    allowed = {
        "linear_isotropic",
        "nonlinear_correction",
        "anisotropic_coupling",
        "time_varying_correction",
    }
    if forward_model not in allowed:
        raise ValueError(f"Unknown forward_model: {forward_model}")
    if not 0.0 < target_separation_scale <= 1.0:
        raise ValueError("target_separation_scale must be in (0, 1].")
    if target_overlap_sd < 0:
        raise ValueError("target_overlap_sd must be nonnegative.")

    log_delta = rng.uniform(*ranges.log10_delta)
    delta = float(10**log_delta)
    q0 = 0.5 + delta if initial_side == 1 else 0.5 - delta

    requested_damage = float(rng.uniform(*ranges.damage_fraction))
    damage = make_damage_mask(x, mask, requested_damage)
    realized_damage = float(damage.sum() / mask.sum())

    correction_rate = float(rng.uniform(*ranges.correction_rate))
    coupling = float(rng.uniform(*ranges.coupling))
    memory_rate = float(rng.uniform(*ranges.memory_rate))
    memory_noise = float(rng.uniform(*ranges.memory_noise))
    process_noise = float(rng.uniform(*ranges.process_noise))
    correction_heterogeneity_cv = float(
        rng.uniform(*ranges.correction_heterogeneity_cv)
    )
    correction_map = correction_rate * (
        1.0 + rng.normal(0.0, correction_heterogeneity_cv, size=mask.shape)
    )
    correction_map = np.clip(
        correction_map, 0.25 * correction_rate, 2.0 * correction_rate
    )
    correction_map[~mask] = 0.0
    target_overlap_offset = float(rng.normal(0.0, target_overlap_sd))

    voltage = original.copy()
    voltage[damage] = rng.normal(0.0, 0.12, size=int(damage.sum()))
    voltage[mask] += rng.normal(0.0, 0.025, size=int(mask.sum()))
    voltage[~mask] = 0.0

    movie = np.empty((n_steps + 1, *mask.shape), dtype=np.float32)
    q_trace = np.empty(n_steps + 1, dtype=np.float64)
    movie[0] = voltage
    q_trace[0] = q0
    q = q0

    for step in range(1, n_steps + 1):
        target_selector = float(
            np.clip(
                0.5
                + target_separation_scale * (q - 0.5)
                + target_overlap_offset,
                0.0,
                1.0,
            )
        )
        instantaneous_target = (
            (1.0 - target_selector) * original + target_selector * alternate
        )
        correction_difference = instantaneous_target - voltage
        if forward_model == "nonlinear_correction":
            correction_component = np.tanh(correction_difference)
        else:
            correction_component = correction_difference
        time_multiplier = (
            1.0 + 0.35 * math.sin(2.0 * math.pi * step / n_steps)
            if forward_model == "time_varying_correction"
            else 1.0
        )
        drift = time_multiplier * correction_map * correction_component
        if forward_model == "anisotropic_coupling":
            drift += coupling * anisotropic_masked_laplacian(voltage, mask)
        else:
            drift += coupling * masked_laplacian(voltage, mask)
        stochastic = process_noise * math.sqrt(dt) * rng.normal(size=voltage.shape)
        voltage = voltage + dt * drift + stochastic
        voltage[~mask] = 0.0
        voltage[mask] = np.clip(voltage[mask], -1.5, 1.5)

        q += dt * memory_rate * q * (1.0 - q) * (q - 0.5)
        q += memory_noise * math.sqrt(dt) * q * (1.0 - q) * rng.normal()
        q = float(np.clip(q, 0.0, 1.0))

        movie[step] = voltage
        q_trace[step] = q

    final_outcome = int(q_trace[-1] >= 0.5)
    outcome_target_selector = float(
        np.clip(
            0.5
            + target_separation_scale * (final_outcome - 0.5)
            + target_overlap_offset,
            0.0,
            1.0,
        )
    )
    params = TrialParameters(
        trial_id=trial_id,
        initial_side=initial_side,
        outcome=final_outcome,
        q0=float(q0),
        distance_from_separatrix=delta,
        damage_fraction_requested=requested_damage,
        damage_fraction_realized=realized_damage,
        correction_rate=correction_rate,
        coupling=coupling,
        memory_rate=memory_rate,
        memory_noise=memory_noise,
        process_noise=process_noise,
        correction_heterogeneity_cv=correction_heterogeneity_cv,
        correction_map_mean=float(correction_map[mask].mean()),
        correction_map_sd=float(correction_map[mask].std(ddof=0)),
        correction_map_min=float(correction_map[mask].min()),
        correction_map_max=float(correction_map[mask].max()),
        target_separation_scale=target_separation_scale,
        target_overlap_sd=target_overlap_sd,
        target_overlap_offset=target_overlap_offset,
        outcome_target_selector=outcome_target_selector,
    )
    return movie, q_trace, params, damage


def _odd_window(n_times: int) -> int:
    candidate = min(21, n_times if n_times % 2 else n_times - 1)
    return max(5, candidate)


def _validated_odd_window(requested: int, n_times: int) -> int:

    candidate = min(int(requested), n_times if n_times % 2 else n_times - 1)
    if candidate % 2 == 0:
        candidate -= 1
    return max(5, candidate)


def infer_target_from_movie(
    observed: np.ndarray,
    dt: float,
    mask: np.ndarray,
    *,
    smoothing: str = "savgol",
    derivative_window: int | None = None,
    coefficient_bounds: tuple[tuple[float, float], tuple[float, float]] | None = (
        (0.08, 2.0),
        (0.0, 0.60),
    ),
    temporal_aggregation: str = "median",
) -> dict[str, np.ndarray | float | bool]:

    n_times = observed.shape[0]
    if n_times < 6:
        raise ValueError("At least six observations are required for inference.")

    if smoothing not in {"savgol", "none"}:
        raise ValueError("smoothing must be 'savgol' or 'none'.")
    if temporal_aggregation not in {"median", "mean"}:
        raise ValueError("temporal_aggregation must be 'median' or 'mean'.")

    window = (
        _odd_window(n_times)
        if derivative_window is None
        else _validated_odd_window(derivative_window, n_times)
    )
    if smoothing == "savgol":
        smooth = savgol_filter(
            observed, window_length=window, polyorder=2, axis=0, mode="interp"
        )
        derivative = savgol_filter(
            observed,
            window_length=window,
            polyorder=2,
            deriv=1,
            delta=dt,
            axis=0,
            mode="interp",
        )
    else:
        smooth = np.asarray(observed, dtype=float)
        derivative = np.gradient(smooth, dt, axis=0, edge_order=2)
    lap = masked_laplacian(smooth, mask)

    edge = max(1, window // 5) if smoothing == "savgol" else 1
    if n_times - 2 * edge < 4:
        edge = 0
    use = slice(edge, n_times - edge if edge else None)

    v = smooth[use][:, mask]
    y = derivative[use][:, mask]
    spatial = lap[use][:, mask]

    v_centered = v - v.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    spatial_centered = spatial - spatial.mean(axis=0, keepdims=True)

    design = np.column_stack(
        [(-v_centered).ravel(), spatial_centered.ravel()]
    )
    response = y_centered.ravel()
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    correction_raw = float(coefficients[0])
    coupling_raw = float(coefficients[1])
    if coefficient_bounds is None:
        correction_hat = correction_raw
        coupling_hat = coupling_raw
        correction_hit_lower = False
        correction_hit_upper = False
        coupling_hit_lower = False
        coupling_hit_upper = False
    else:
        correction_limits, coupling_limits = coefficient_bounds
        correction_hat = float(np.clip(correction_raw, *correction_limits))
        coupling_hat = float(np.clip(coupling_raw, *coupling_limits))
        correction_hit_lower = bool(correction_raw <= correction_limits[0])
        correction_hit_upper = bool(correction_raw >= correction_limits[1])
        coupling_hit_lower = bool(coupling_raw <= coupling_limits[0])
        coupling_hit_upper = bool(coupling_raw >= coupling_limits[1])

    correction_for_target = correction_hat
    if abs(correction_for_target) < 1e-8:
        correction_for_target = math.copysign(1e-8, correction_for_target or 1.0)

    target_instantaneous = (
        y - coupling_hat * spatial + correction_hat * v
    ) / correction_for_target
    if temporal_aggregation == "median":
        target_values = np.median(target_instantaneous, axis=0)
    else:
        target_values = np.mean(target_instantaneous, axis=0)
    target_hat = np.zeros(mask.shape, dtype=float)
    target_hat[mask] = np.clip(target_values, -1.5, 1.5)

    return {
        "target": target_hat,
        "correction_rate": correction_hat,
        "coupling": coupling_hat,
        "raw_correction_rate": correction_raw,
        "raw_coupling": coupling_raw,
        "correction_hit_lower": correction_hit_lower,
        "correction_hit_upper": correction_hit_upper,
        "coupling_hit_lower": coupling_hit_lower,
        "coupling_hit_upper": coupling_hit_upper,
        "smoothing": smoothing,
        "derivative_window": int(window),
        "temporal_aggregation": temporal_aggregation,
    }


def project_q(
    field: np.ndarray, original: np.ndarray, alternate: np.ndarray, mask: np.ndarray
) -> float:

    difference = (alternate - original)[mask]
    numerator = float(np.dot((field - original)[mask], difference))
    denominator = float(np.dot(difference, difference))
    return numerator / denominator


def mean_class_dice(
    predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> float:
    scores: list[float] = []
    for label in range(len(PROTOTYPES)):
        p = (predicted == label) & mask
        t = (truth == label) & mask
        denominator = int(p.sum() + t.sum())
        scores.append(1.0 if denominator == 0 else 2.0 * int((p & t).sum()) / denominator)
    return float(np.mean(scores))


def mean_class_dice_nonempty(
    predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> float:

    scores: list[float] = []
    for label in range(len(PROTOTYPES)):
        predicted_class = (predicted == label) & mask
        truth_class = (truth == label) & mask
        denominator = int(predicted_class.sum() + truth_class.sum())
        if denominator > 0:
            scores.append(
                2.0
                * int((predicted_class & truth_class).sum())
                / denominator
            )
    return float(np.mean(scores)) if scores else 1.0


def field_metrics(
    target_hat: np.ndarray,
    truth_target: np.ndarray,
    truth_labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    residual = target_hat[mask] - truth_target[mask]
    rmse = float(np.sqrt(np.mean(residual**2)))
    nrmse = rmse / float(PROTOTYPES.max() - PROTOTYPES.min())
    if np.std(target_hat[mask]) == 0 or np.std(truth_target[mask]) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(target_hat[mask], truth_target[mask])[0, 1])
    predicted_labels = decode_anatomy(target_hat, mask)
    pixel_accuracy = float(np.mean(predicted_labels[mask] == truth_labels[mask]))
    dice = mean_class_dice(predicted_labels, truth_labels, mask)
    dice_nonempty = mean_class_dice_nonempty(
        predicted_labels, truth_labels, mask
    )
    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": correlation,
        "pixel_accuracy": pixel_accuracy,
        "macro_dice": dice,
        "macro_dice_nonempty_classes": dice_nonempty,
    }


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    statistic=np.mean,
    n_boot: int = 2000,
) -> tuple[float, float]:
    values = np.asarray(values)
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[i] = statistic(sample)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> tuple[float, float]:
    estimates: list[float] = []
    indices = np.arange(len(labels))
    while len(estimates) < n_boot:
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        estimates.append(fast_auc(labels[sample], scores[sample]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires both outcome classes.")
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def paired_mean_ci(
    first: np.ndarray,
    second: np.ndarray,
    seed: int,
    n_boot: int = 4000,
) -> tuple[float, float, float]:

    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if differences.ndim != 1 or len(differences) == 0:
        raise ValueError("Paired arrays must be nonempty one-dimensional arrays.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    estimates = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def paired_auc_difference_ci(
    labels: np.ndarray,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    seed: int,
    n_boot: int = 4000,
) -> tuple[float, float, float]:

    labels = np.asarray(labels, dtype=int)
    first_scores = np.asarray(first_scores, dtype=float)
    second_scores = np.asarray(second_scores, dtype=float)
    if not (
        labels.ndim == first_scores.ndim == second_scores.ndim == 1
        and len(labels) == len(first_scores) == len(second_scores)
        and len(labels) > 0
    ):
        raise ValueError("Labels and paired score arrays must be aligned vectors.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Paired AUROC comparison requires two outcome classes.")
    observed = fast_auc(labels, first_scores) - fast_auc(labels, second_scores)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    estimates: list[float] = []
    while len(estimates) < n_boot:
        sample = rng.choice(indices, size=len(indices), replace=True)
        sample_labels = labels[sample]
        if len(np.unique(sample_labels)) != 2:
            continue
        estimates.append(
            fast_auc(sample_labels, first_scores[sample])
            - fast_auc(sample_labels, second_scores[sample])
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(observed), float(low), float(high)


def paired_auc_permutation_p(
    labels: np.ndarray,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    seed: int,
    n_permutations: int = 4000,
) -> float:

    labels = np.asarray(labels, dtype=int)
    first_scores = np.asarray(first_scores, dtype=float)
    second_scores = np.asarray(second_scores, dtype=float)
    observed = abs(
        fast_auc(labels, first_scores) - fast_auc(labels, second_scores)
    )
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        swap = rng.random(len(labels)) < 0.5
        permuted_first = np.where(swap, second_scores, first_scores)
        permuted_second = np.where(swap, first_scores, second_scores)
        difference = abs(
            fast_auc(labels, permuted_first)
            - fast_auc(labels, permuted_second)
        )
        if difference >= observed - 1e-15:
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def exact_mcnemar_p(first_correct: np.ndarray, second_correct: np.ndarray) -> float:

    first_correct = np.asarray(first_correct, dtype=bool)
    second_correct = np.asarray(second_correct, dtype=bool)
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    return float(
        binomtest(
            min(first_only, second_only),
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def paired_wilcoxon_p(first: np.ndarray, second: np.ndarray) -> float:

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.allclose(first, second, rtol=0.0, atol=1e-15):
        return 1.0
    return float(wilcoxon(first, second, alternative="two-sided").pvalue)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:

    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p) - rank) * p[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(p), dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted


def quantile_summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    values = pd.Series(np.asarray(values, dtype=float))
    quantiles = values.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {str(index): float(value) for index, value in quantiles.items()}


def correlation_summary(first: pd.Series, second: pd.Series) -> dict[str, float]:

    first_values = first.to_numpy(dtype=float)
    second_values = second.to_numpy(dtype=float)
    pearson = float(np.corrcoef(first_values, second_values)[0, 1])
    first_ranks = rankdata(first_values, method="average")
    second_ranks = rankdata(second_values, method="average")
    spearman = float(np.corrcoef(first_ranks, second_ranks)[0, 1])
    return {"pearson_r": pearson, "spearman_rho": spearman}


def summarize_metrics(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 901)
    sensitivity_rng = np.random.default_rng(seed + 1901)
    same_time_auc_rng = np.random.default_rng(seed + 2901)
    rows: list[dict[str, float]] = []
    for (noise, fraction), group in frame.groupby(
        ["measurement_noise", "observation_fraction"], sort=True
    ):
        labels = group["outcome"].to_numpy(dtype=int)
        predictions = group["prediction"].to_numpy(dtype=int)
        scores = group["q_hat"].to_numpy(dtype=float)
        same_time_scores = group["same_time_snapshot_q"].to_numpy(dtype=float)
        correct = (labels == predictions).astype(float)
        same_time_predictions = group["same_time_snapshot_prediction"].to_numpy(
            dtype=int
        )
        same_time_correct = (labels == same_time_predictions).astype(float)
        acc_low, acc_high = bootstrap_ci(correct, rng)
        same_time_low, same_time_high = bootstrap_ci(same_time_correct, rng)
        auc = fast_auc(labels, scores)
        auc_low, auc_high = bootstrap_auc_ci(labels, scores, rng)
        same_time_auc = fast_auc(labels, same_time_scores)
        same_time_auc_low, same_time_auc_high = bootstrap_auc_ci(
            labels, same_time_scores, same_time_auc_rng
        )
        dice_low, dice_high = bootstrap_ci(group["macro_dice"].to_numpy(), rng)
        dice_nonempty_low, dice_nonempty_high = bootstrap_ci(
            group["macro_dice_nonempty_classes"].to_numpy(), sensitivity_rng
        )
        nrmse_low, nrmse_high = bootstrap_ci(group["nrmse"].to_numpy(), rng)
        snapshot_dice_low, snapshot_dice_high = bootstrap_ci(
            group["same_time_snapshot_macro_dice"].to_numpy(), rng
        )
        snapshot_dice_nonempty_low, snapshot_dice_nonempty_high = bootstrap_ci(
            group[
                "same_time_snapshot_macro_dice_nonempty_classes"
            ].to_numpy(),
            sensitivity_rng,
        )
        snapshot_nrmse_low, snapshot_nrmse_high = bootstrap_ci(
            group["same_time_snapshot_nrmse"].to_numpy(), rng
        )
        rows.append(
            {
                "measurement_noise": float(noise),
                "observation_fraction": float(fraction),
                "n": int(len(group)),
                "accuracy": float(accuracy_score(labels, predictions)),
                "accuracy_ci_low": acc_low,
                "accuracy_ci_high": acc_high,
                "same_time_snapshot_accuracy": float(same_time_correct.mean()),
                "same_time_snapshot_ci_low": same_time_low,
                "same_time_snapshot_ci_high": same_time_high,
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predictions)
                ),
                "auroc": auc,
                "auroc_ci_low": auc_low,
                "auroc_ci_high": auc_high,
                "same_time_snapshot_auroc": same_time_auc,
                "same_time_snapshot_auroc_ci_low": same_time_auc_low,
                "same_time_snapshot_auroc_ci_high": same_time_auc_high,
                "mean_macro_dice": float(group["macro_dice"].mean()),
                "macro_dice_ci_low": dice_low,
                "macro_dice_ci_high": dice_high,
                "mean_macro_dice_nonempty_classes": float(
                    group["macro_dice_nonempty_classes"].mean()
                ),
                "macro_dice_nonempty_classes_ci_low": dice_nonempty_low,
                "macro_dice_nonempty_classes_ci_high": dice_nonempty_high,
                "mean_nrmse": float(group["nrmse"].mean()),
                "nrmse_ci_low": nrmse_low,
                "nrmse_ci_high": nrmse_high,
                "mean_same_time_snapshot_macro_dice": float(
                    group["same_time_snapshot_macro_dice"].mean()
                ),
                "same_time_snapshot_macro_dice_ci_low": snapshot_dice_low,
                "same_time_snapshot_macro_dice_ci_high": snapshot_dice_high,
                "mean_same_time_snapshot_macro_dice_nonempty_classes": float(
                    group[
                        "same_time_snapshot_macro_dice_nonempty_classes"
                    ].mean()
                ),
                "same_time_snapshot_macro_dice_nonempty_classes_ci_low": snapshot_dice_nonempty_low,
                "same_time_snapshot_macro_dice_nonempty_classes_ci_high": snapshot_dice_nonempty_high,
                "mean_same_time_snapshot_nrmse": float(
                    group["same_time_snapshot_nrmse"].mean()
                ),
                "same_time_snapshot_nrmse_ci_low": snapshot_nrmse_low,
                "same_time_snapshot_nrmse_ci_high": snapshot_nrmse_high,
                "mean_pixel_accuracy": float(group["pixel_accuracy"].mean()),
                "median_correlation": float(group["correlation"].median()),
                "median_correction_relative_error": float(
                    group["correction_relative_error"].median()
                ),
                "median_coupling_absolute_error": float(
                    group["coupling_absolute_error"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def earliest_stable_decision(
    frame: pd.DataFrame, confidence_margin: float = 0.15
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (trial_id, noise), group in frame.groupby(
        ["trial_id", "measurement_noise"], sort=True
    ):
        group = group.sort_values("observation_fraction")
        correct = group["prediction"].to_numpy() == group["outcome"].to_numpy()
        confident = np.abs(group["q_hat"].to_numpy(dtype=float) - 0.5) >= confidence_margin
        reliable = correct & confident
        fractions = group["observation_fraction"].to_numpy(dtype=float)
        stable_fraction = np.nan
        for index in range(len(reliable)):
            if bool(np.all(reliable[index:])):
                stable_fraction = float(fractions[index])
                break
        first = group.iloc[0]
        rows.append(
            {
                "trial_id": int(trial_id),
                "measurement_noise": float(noise),
                "outcome": int(first["outcome"]),
                "distance_from_separatrix": float(
                    first["distance_from_separatrix"]
                ),
                "earliest_stable_fraction": stable_fraction,
                "confidence_margin": confidence_margin,
            }
        )
    return pd.DataFrame(rows)


def prospective_decision_rule(
    frame: pd.DataFrame,
    confidence_margin: float = CONFIDENCE_MARGIN,
) -> pd.DataFrame:

    rows: list[dict[str, float | int | bool]] = []
    for trial_id, group in frame.groupby("trial_id", sort=True):
        group = group.sort_values("observation_fraction")
        predictions = group["prediction"].to_numpy(dtype=int)
        margins = np.abs(group["q_hat"].to_numpy(dtype=float) - 0.5)
        fractions = group["observation_fraction"].to_numpy(dtype=float)
        decision: float = np.nan
        decision_fraction: float = np.nan
        for index in range(1, len(group)):
            if (
                predictions[index] == predictions[index - 1]
                and margins[index] >= confidence_margin
                and margins[index - 1] >= confidence_margin
            ):
                decision = float(predictions[index])
                decision_fraction = float(fractions[index])
                break
        outcome = int(group.iloc[0]["outcome"])
        rows.append(
            {
                "trial_id": int(trial_id),
                "outcome": outcome,
                "prospective_decision": decision,
                "decision_fraction": decision_fraction,
                "confidence_margin": float(confidence_margin),
                "decision_made": bool(np.isfinite(decision)),
                "correct_if_decided": (
                    bool(int(decision) == outcome) if np.isfinite(decision) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def template_separation_table(scales: Iterable[float]) -> pd.DataFrame:

    rows: list[dict[str, float | int | str]] = []
    for scale in scales:
        (
            mask,
            _,
            _,
            original,
            alternate,
            original_labels,
            alternate_labels,
        ) = make_overlap_geometry(float(scale))
        difference = (alternate - original)[mask]
        rmse = float(np.sqrt(np.mean(difference**2)))
        rows.append(
            {
                "difficulty": (
                    "original" if np.isclose(scale, 1.0) else f"scale_{scale:.2f}"
                ),
                "separation_scale": float(scale),
                "active_tissue_pixels": int(mask.sum()),
                "euclidean_distance": float(np.linalg.norm(difference)),
                "rmse": rmse,
                "nrmse_full_voltage_range": rmse / 2.0,
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "fraction_voltage_values_different": float(
                    np.mean(np.abs(difference) > 1e-12)
                ),
                "fraction_anatomical_labels_different": float(
                    np.mean(original_labels[mask] != alternate_labels[mask])
                ),
                "macro_dice_between_anatomical_templates": mean_class_dice(
                    original_labels, alternate_labels, mask
                ),
                "macro_dice_between_anatomical_templates_nonempty_classes": (
                    mean_class_dice_nonempty(
                        original_labels, alternate_labels, mask
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def make_ambiguity_outputs(
    main: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:

    ordered = main.sort_values("trial_id").copy()
    ordered["correct"] = ordered["prediction"] == ordered["outcome"]
    ordered["q_final_distance_from_boundary"] = np.abs(ordered["q_final"] - 0.5)
    ordered["q_hat_distance_from_boundary"] = np.abs(ordered["q_hat"] - 0.5)
    ordered["ambiguous_score"] = (
        ordered["q_hat_distance_from_boundary"] < CONFIDENCE_MARGIN
    )
    export = ordered[
        [
            "trial_id",
            "outcome",
            "q0",
            "q_final",
            "q_final_distance_from_boundary",
            "q_hat",
            "q_hat_distance_from_boundary",
            "ambiguous_score",
            "prediction",
            "correct",
        ]
    ].copy()
    unambiguous = ordered.loc[~ordered["ambiguous_score"]]
    summary: dict[str, object] = {
        "benchmark": {
            "measurement_noise": MAIN_NOISE,
            "observation_fraction": MAIN_FRACTION,
            "n": int(len(ordered)),
        },
        "final_q_quantiles": quantile_summary(ordered["q_final"]),
        "final_q_distance_quantiles": quantile_summary(
            ordered["q_final_distance_from_boundary"]
        ),
        "final_q_counts_within_boundary_margin": {
            str(margin): int(
                (ordered["q_final_distance_from_boundary"] <= margin).sum()
            )
            for margin in (0.01, 0.025, 0.05, 0.10, 0.15)
        },
        "initial_side_differs_from_final_outcome": int(
            (parameters["initial_side"] != parameters["outcome"]).sum()
        ),
        "score_exclusion_analysis": {
            "margin": CONFIDENCE_MARGIN,
            "ambiguous_n": int(ordered["ambiguous_score"].sum()),
            "ambiguous_accuracy": float(
                ordered.loc[ordered["ambiguous_score"], "correct"].mean()
            ),
            "retained_n": int(len(unambiguous)),
            "coverage": float(len(unambiguous) / len(ordered)),
            "retained_accuracy": float(unambiguous["correct"].mean()),
            "all_case_accuracy": float(ordered["correct"].mean()),
            "errors_in_ambiguous_group": int(
                ((~ordered["correct"]) & ordered["ambiguous_score"]).sum()
            ),
        },
        "interpretation": (
            "q_hat is an uncalibrated projection score. The exclusion analysis "
            "does not interpret q_hat as a probability."
        ),
    }
    return export, summary


def make_identifiability_outputs(
    main: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:

    frame = main.sort_values("trial_id").copy()
    frame["correct"] = frame["prediction"] == frame["outcome"]
    frame["sampled_k_error"] = (
        frame["estimated_correction_rate"] - frame["true_correction_rate"]
    )
    frame["realized_mean_k_error"] = (
        frame["estimated_correction_rate"] - frame["realized_correction_map_mean"]
    )
    frame["coupling_error"] = frame["estimated_coupling"] - frame["true_coupling"]
    frame["any_coefficient_boundary_hit"] = frame[
        [
            "correction_hit_lower",
            "correction_hit_upper",
            "coupling_hit_lower",
            "coupling_hit_upper",
        ]
    ].any(axis=1)
    large_k_error = frame["correction_relative_error"] > 0.50
    large_d_error = frame["coupling_absolute_error"] > 0.20
    boundary = frame["any_coefficient_boundary_hit"]

    summary: dict[str, object] = {
        "benchmark": {
            "measurement_noise": MAIN_NOISE,
            "observation_fraction": MAIN_FRACTION,
            "n": int(len(frame)),
        },
        "true_and_estimated_distributions": {
            "sampled_base_k": quantile_summary(frame["true_correction_rate"]),
            "realized_mean_k_i": quantile_summary(
                frame["realized_correction_map_mean"]
            ),
            "estimated_k": quantile_summary(frame["estimated_correction_rate"]),
            "raw_estimated_k": quantile_summary(frame["raw_correction_rate"]),
            "true_D": quantile_summary(frame["true_coupling"]),
            "estimated_D": quantile_summary(frame["estimated_coupling"]),
            "raw_estimated_D": quantile_summary(frame["raw_coupling"]),
        },
        "boundary_hits": {
            "k_lower_0.08": int(frame["correction_hit_lower"].sum()),
            "k_upper_2.0": int(frame["correction_hit_upper"].sum()),
            "D_lower_0.0": int(frame["coupling_hit_lower"].sum()),
            "D_upper_0.60": int(frame["coupling_hit_upper"].sum()),
            "any_boundary": int(boundary.sum()),
        },
        "correlations": {
            "sampled_base_k_vs_estimated_k": correlation_summary(
                frame["true_correction_rate"], frame["estimated_correction_rate"]
            ),
            "realized_mean_k_i_vs_estimated_k": correlation_summary(
                frame["realized_correction_map_mean"],
                frame["estimated_correction_rate"],
            ),
            "true_D_vs_estimated_D": correlation_summary(
                frame["true_coupling"], frame["estimated_coupling"]
            ),
            "estimated_k_vs_estimated_D": correlation_summary(
                frame["estimated_correction_rate"], frame["estimated_coupling"]
            ),
            "signed_k_error_vs_signed_D_error": correlation_summary(
                frame["sampled_k_error"], frame["coupling_error"]
            ),
        },
        "prediction_despite_parameter_error": {
            "k_relative_error_over_0.50": {
                "n": int(large_k_error.sum()),
                "accuracy": float(frame.loc[large_k_error, "correct"].mean()),
            },
            "D_absolute_error_over_0.20": {
                "n": int(large_d_error.sum()),
                "accuracy": float(frame.loc[large_d_error, "correct"].mean()),
            },
            "any_boundary_hit": {
                "n": int(boundary.sum()),
                "accuracy": float(frame.loc[boundary, "correct"].mean()),
            },
        },
        "interpretation": (
            "Accurate target prediction is evaluated separately from recovery "
            "of the forward dynamical coefficients."
        ),
    }
    export_columns = [
        "trial_id",
        "outcome",
        "correct",
        "true_correction_rate",
        "realized_correction_map_mean",
        "realized_correction_map_sd",
        "estimated_correction_rate",
        "raw_correction_rate",
        "sampled_k_error",
        "realized_mean_k_error",
        "true_coupling",
        "estimated_coupling",
        "raw_coupling",
        "coupling_error",
        "correction_hit_lower",
        "correction_hit_upper",
        "coupling_hit_lower",
        "coupling_hit_upper",
        "any_coefficient_boundary_hit",
        "macro_dice",
        "macro_dice_nonempty_classes",
        "nrmse",
    ]
    return frame[export_columns].copy(), summary


def make_full_grid_effects(frame: pd.DataFrame, seed: int) -> pd.DataFrame:

    rows: list[dict[str, float | int | str]] = []
    counter = 0
    for (noise, fraction), group in frame.groupby(
        ["measurement_noise", "observation_fraction"], sort=True
    ):
        group = group.sort_values("trial_id")
        labels = group["outcome"].to_numpy(dtype=int)
        movie_correct = (group["prediction"] == group["outcome"]).to_numpy(float)
        snapshot_correct = (
            group["same_time_snapshot_prediction"] == group["outcome"]
        ).to_numpy(float)
        metric_pairs = [
            (
                "accuracy",
                movie_correct,
                snapshot_correct,
                exact_mcnemar_p(movie_correct.astype(bool), snapshot_correct.astype(bool)),
            ),
            (
                "macro_dice",
                group["macro_dice"].to_numpy(float),
                group["same_time_snapshot_macro_dice"].to_numpy(float),
                paired_wilcoxon_p(
                    group["macro_dice"].to_numpy(float),
                    group["same_time_snapshot_macro_dice"].to_numpy(float),
                ),
            ),
            (
                "nrmse",
                group["nrmse"].to_numpy(float),
                group["same_time_snapshot_nrmse"].to_numpy(float),
                paired_wilcoxon_p(
                    group["nrmse"].to_numpy(float),
                    group["same_time_snapshot_nrmse"].to_numpy(float),
                ),
            ),
        ]
        for metric, movie, snapshot, p_value in metric_pairs:
            effect, low, high = paired_mean_ci(
                movie, snapshot, seed + 20_000 + counter
            )
            rows.append(
                {
                    "measurement_noise": float(noise),
                    "observation_fraction": float(fraction),
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "movie_mean": float(np.mean(movie)),
                    "same_time_snapshot_mean": float(np.mean(snapshot)),
                    "paired_effect_movie_minus_snapshot": effect,
                    "paired_effect_ci_low": low,
                    "paired_effect_ci_high": high,
                    "two_sided_p": p_value,
                }
            )
            counter += 1
        movie_scores = group["q_hat"].to_numpy(dtype=float)
        snapshot_scores = group["same_time_snapshot_q"].to_numpy(dtype=float)
        auc_effect, auc_low, auc_high = paired_auc_difference_ci(
            labels,
            movie_scores,
            snapshot_scores,
            seed + 21_000 + counter,
        )
        auc_p = paired_auc_permutation_p(
            labels,
            movie_scores,
            snapshot_scores,
            seed + 22_000 + counter,
        )
        rows.append(
            {
                "measurement_noise": float(noise),
                "observation_fraction": float(fraction),
                "metric": "auroc",
                "n_pairs": int(len(group)),
                "movie_mean": float(fast_auc(labels, movie_scores)),
                "same_time_snapshot_mean": float(
                    fast_auc(labels, snapshot_scores)
                ),
                "paired_effect_movie_minus_snapshot": auc_effect,
                "paired_effect_ci_low": auc_low,
                "paired_effect_ci_high": auc_high,
                "two_sided_p": auc_p,
            }
        )
        counter += 1
    effects = pd.DataFrame(rows)
    effects["holm_p_within_metric_30_cells"] = np.nan
    for metric, indices in effects.groupby("metric").groups.items():
        effects.loc[indices, "holm_p_within_metric_30_cells"] = holm_adjust(
            effects.loc[indices, "two_sided_p"]
        )
    return effects


def summarize_ablations(rows: pd.DataFrame, seed: int) -> pd.DataFrame:

    default = rows[rows["ablation"] == "default"].sort_values("trial_id")
    summaries: list[dict[str, float | int | str]] = []
    for index, (name, group) in enumerate(rows.groupby("ablation", sort=False)):
        group = group.sort_values("trial_id")
        if not np.array_equal(group["trial_id"], default["trial_id"]):
            raise RuntimeError(f"Ablation {name} is not paired to the default rows.")
        correct = (group["prediction"] == group["outcome"]).to_numpy(float)
        default_correct = (
            default["prediction"] == default["outcome"]
        ).to_numpy(float)
        accuracy_effect, accuracy_low, accuracy_high = paired_mean_ci(
            correct, default_correct, seed + 30_000 + index * 3
        )
        dice_effect, dice_low, dice_high = paired_mean_ci(
            group["macro_dice"].to_numpy(float),
            default["macro_dice"].to_numpy(float),
            seed + 30_001 + index * 3,
        )
        nrmse_effect, nrmse_low, nrmse_high = paired_mean_ci(
            group["nrmse"].to_numpy(float),
            default["nrmse"].to_numpy(float),
            seed + 30_002 + index * 3,
        )
        summaries.append(
            {
                "ablation": name,
                "n_pairs": int(len(group)),
                "accuracy": float(correct.mean()),
                "mean_macro_dice": float(group["macro_dice"].mean()),
                "mean_macro_dice_nonempty_classes": float(
                    group["macro_dice_nonempty_classes"].mean()
                ),
                "mean_nrmse": float(group["nrmse"].mean()),
                "accuracy_minus_default": accuracy_effect,
                "accuracy_effect_ci_low": accuracy_low,
                "accuracy_effect_ci_high": accuracy_high,
                "accuracy_two_sided_p": exact_mcnemar_p(
                    correct.astype(bool), default_correct.astype(bool)
                ),
                "macro_dice_minus_default": dice_effect,
                "macro_dice_effect_ci_low": dice_low,
                "macro_dice_effect_ci_high": dice_high,
                "macro_dice_two_sided_p": paired_wilcoxon_p(
                    group["macro_dice"], default["macro_dice"]
                ),
                "nrmse_minus_default": nrmse_effect,
                "nrmse_effect_ci_low": nrmse_low,
                "nrmse_effect_ci_high": nrmse_high,
                "nrmse_two_sided_p": paired_wilcoxon_p(
                    group["nrmse"], default["nrmse"]
                ),
                "fraction_any_coefficient_boundary": float(
                    group["any_coefficient_boundary_hit"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    for metric in ("accuracy", "macro_dice", "nrmse"):
        column = f"{metric}_two_sided_p"
        adjusted_column = f"{metric}_holm_p_across_ablation_variants"
        summary[adjusted_column] = 1.0
        nondefault = summary["ablation"] != "default"
        summary.loc[nondefault, adjusted_column] = holm_adjust(
            summary.loc[nondefault, column]
        )
    return summary


def evaluate_revision_cohort(
    *,
    scenario: str,
    seed: int,
    n_trials: int,
    ranges: ParameterRanges,
    forward_model: str,
    separation_scale: float,
    target_overlap_sd: float,
    observation_fractions: tuple[float, ...],
    measurement_noise: float = MAIN_NOISE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if n_trials % 2:
        raise ValueError("n_trials must be even.")
    rng = np.random.default_rng(seed)
    dt = 0.05
    n_steps = 220
    (
        mask,
        x,
        _,
        original,
        alternate,
        original_labels,
        alternate_labels,
    ) = make_geometry()
    initial_sides = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))
    rng.shuffle(initial_sides)
    rows: list[dict[str, float | int | str | bool]] = []
    parameter_rows: list[dict[str, float | int | str]] = []

    for trial_id, initial_side_value in enumerate(initial_sides):
        movie, q_trace, params, _ = simulate_trial_scenario(
            rng,
            trial_id,
            int(initial_side_value),
            mask,
            x,
            original,
            alternate,
            dt,
            n_steps,
            ranges,
            forward_model=forward_model,
            target_separation_scale=separation_scale,
            target_overlap_sd=target_overlap_sd,
        )
        parameter_row = asdict(params)
        parameter_row.update(
            {
                "scenario": scenario,
                "scenario_seed": seed,
                "forward_model": forward_model,
                "separation_scale": separation_scale,
                "target_overlap_sd": target_overlap_sd,
            }
        )
        parameter_rows.append(parameter_row)

        observed = movie.astype(float).copy()
        if measurement_noise > 0:
            measurement = rng.normal(0.0, measurement_noise, size=observed.shape)
            measurement[:, ~mask] = 0.0
            observed += measurement
        observed[:, ~mask] = 0.0
        outcome = params.outcome
        truth_target = (
            (1.0 - params.outcome_target_selector) * original
            + params.outcome_target_selector * alternate
        )
        truth_labels = decode_anatomy(truth_target, mask)

        for fraction in observation_fractions:
            end = max(6, int(round(fraction * n_steps)) + 1)
            inverse = infer_target_from_movie(observed[:end], dt, mask)
            target_hat = np.asarray(inverse["target"])
            q_hat = project_q(target_hat, original, alternate, mask)
            prediction = int(q_hat >= 0.5)
            snapshot = observed[end - 1]
            snapshot_q = project_q(snapshot, original, alternate, mask)
            metrics = field_metrics(target_hat, truth_target, truth_labels, mask)
            snapshot_metrics = field_metrics(
                snapshot, truth_target, truth_labels, mask
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_seed": seed,
                    "trial_id": trial_id,
                    "measurement_noise": measurement_noise,
                    "observation_fraction": fraction,
                    "observations_used": end,
                    "forward_model": forward_model,
                    "separation_scale": separation_scale,
                    "target_overlap_sd": target_overlap_sd,
                    "target_overlap_offset": params.target_overlap_offset,
                    "outcome_target_selector": params.outcome_target_selector,
                    "outcome": outcome,
                    "q0": params.q0,
                    "q_final": float(q_trace[-1]),
                    "q_hat": q_hat,
                    "prediction": prediction,
                    "same_time_snapshot_q": snapshot_q,
                    "same_time_snapshot_prediction": int(snapshot_q >= 0.5),
                    "true_correction_rate": params.correction_rate,
                    "realized_correction_map_mean": params.correction_map_mean,
                    "true_coupling": params.coupling,
                    "estimated_correction_rate": float(inverse["correction_rate"]),
                    "estimated_coupling": float(inverse["coupling"]),
                    "raw_correction_rate": float(inverse["raw_correction_rate"]),
                    "raw_coupling": float(inverse["raw_coupling"]),
                    "correction_hit_lower": bool(inverse["correction_hit_lower"]),
                    "correction_hit_upper": bool(inverse["correction_hit_upper"]),
                    "coupling_hit_lower": bool(inverse["coupling_hit_lower"]),
                    "coupling_hit_upper": bool(inverse["coupling_hit_upper"]),
                    **{
                        f"same_time_snapshot_{key}": value
                        for key, value in snapshot_metrics.items()
                    },
                    **metrics,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows)


def summarize_revision_scenarios(frame: pd.DataFrame, seed: int) -> pd.DataFrame:

    rows: list[dict[str, float | int | str]] = []
    for index, (scenario, group) in enumerate(frame.groupby("scenario", sort=False)):
        main = group[np.isclose(group["observation_fraction"], MAIN_FRACTION)].copy()
        main = main.sort_values("trial_id")
        labels = main["outcome"].to_numpy(int)
        predictions = main["prediction"].to_numpy(int)
        correct = (labels == predictions).astype(float)
        snapshot_correct = (
            labels == main["same_time_snapshot_prediction"].to_numpy(int)
        ).astype(float)
        rng = np.random.default_rng(seed + 40_000 + index)
        accuracy_low, accuracy_high = bootstrap_ci(correct, rng)
        dice_low, dice_high = bootstrap_ci(main["macro_dice"].to_numpy(), rng)
        dice_nonempty_low, dice_nonempty_high = bootstrap_ci(
            main["macro_dice_nonempty_classes"].to_numpy(), rng
        )
        nrmse_low, nrmse_high = bootstrap_ci(main["nrmse"].to_numpy(), rng)
        accuracy_effect, effect_low, effect_high = paired_mean_ci(
            correct, snapshot_correct, seed + 41_000 + index
        )
        rows.append(
            {
                "scenario": scenario,
                "scenario_seed": int(main.iloc[0]["scenario_seed"]),
                "forward_model": str(main.iloc[0]["forward_model"]),
                "separation_scale": float(main.iloc[0]["separation_scale"]),
                "target_overlap_sd": float(main.iloc[0]["target_overlap_sd"]),
                "measurement_noise": MAIN_NOISE,
                "observation_fraction": MAIN_FRACTION,
                "n": int(len(main)),
                "original_outcomes": int((labels == 0).sum()),
                "alternate_outcomes": int((labels == 1).sum()),
                "target_selector_wrong_side_fraction": float(
                    np.mean(
                        (main["outcome_target_selector"].to_numpy(float) >= 0.5)
                        != labels.astype(bool)
                    )
                ),
                "target_selector_within_0.15_of_boundary_fraction": float(
                    np.mean(
                        np.abs(
                            main["outcome_target_selector"].to_numpy(float) - 0.5
                        )
                        < CONFIDENCE_MARGIN
                    )
                ),
                "accuracy": float(correct.mean()),
                "accuracy_ci_low": accuracy_low,
                "accuracy_ci_high": accuracy_high,
                "auroc": fast_auc(labels, main["q_hat"].to_numpy(float)),
                "mean_macro_dice": float(main["macro_dice"].mean()),
                "macro_dice_ci_low": dice_low,
                "macro_dice_ci_high": dice_high,
                "mean_macro_dice_nonempty_classes": float(
                    main["macro_dice_nonempty_classes"].mean()
                ),
                "macro_dice_nonempty_classes_ci_low": dice_nonempty_low,
                "macro_dice_nonempty_classes_ci_high": dice_nonempty_high,
                "mean_nrmse": float(main["nrmse"].mean()),
                "nrmse_ci_low": nrmse_low,
                "nrmse_ci_high": nrmse_high,
                "same_time_snapshot_accuracy": float(snapshot_correct.mean()),
                "accuracy_movie_minus_same_time_snapshot": accuracy_effect,
                "accuracy_effect_ci_low": effect_low,
                "accuracy_effect_ci_high": effect_high,
                "accuracy_mcnemar_two_sided_p": exact_mcnemar_p(
                    correct.astype(bool), snapshot_correct.astype(bool)
                ),
            }
        )
    return pd.DataFrame(rows)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.11,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def plot_model_overview(
    output: Path,
    mask: np.ndarray,
    original: np.ndarray,
    alternate: np.ndarray,
    original_labels: np.ndarray,
    alternate_labels: np.ndarray,
    example: dict[str, np.ndarray | TrialParameters],
) -> None:
    cmap_v = "coolwarm"
    cmap_a = ListedColormap(["#3957A5", "#EEEEEE", "#D64B40"])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)

    for ax, field, title in zip(
        axes[0, :2],
        [original, alternate],
        ["Original target voltage", "Alternate target voltage"],
    ):
        shown = np.where(mask, field, np.nan)
        im = ax.imshow(shown, cmap=cmap_v, vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[0, :2], shrink=0.72, pad=0.02)
    cbar.set_label("Normalized membrane voltage")

    damaged = np.asarray(example["movie"])[0]
    axes[0, 2].imshow(np.where(mask, damaged, np.nan), cmap=cmap_v, vmin=-1, vmax=1)
    axes[0, 2].set_title("Post-damage starting state")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(np.where(mask, original_labels, np.nan), cmap=cmap_a, vmin=0, vmax=2)
    axes[1, 0].set_title("Decoded original anatomy")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(np.where(mask, alternate_labels, np.nan), cmap=cmap_a, vmin=0, vmax=2)
    axes[1, 1].set_title("Decoded alternate anatomy")
    axes[1, 1].axis("off")

    q_trace = np.asarray(example["q_trace"])
    axes[1, 2].plot(np.linspace(0, 1, len(q_trace)), q_trace, color="#5B2A86", lw=2)
    axes[1, 2].axhline(0.5, color="black", ls="--", lw=1, label="Separatrix")
    axes[1, 2].set(
        xlabel="Fraction of simulated time",
        ylabel="Hidden memory state, q",
        ylim=(-0.02, 1.02),
        title="Bistable target selection",
    )
    axes[1, 2].legend(frameon=False, fontsize=8)

    for label, ax in zip("ABCDEF", axes.ravel()):
        _panel_label(ax, label)
    fig.suptitle("Forward model and explicit voltage-to-anatomy decoder", fontsize=13)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_example_reconstruction(
    output: Path,
    mask: np.ndarray,
    original: np.ndarray,
    alternate: np.ndarray,
    example: dict[str, object],
    dt: float,
    windows: Iterable[float],
) -> None:
    movie = np.asarray(example["observed"])
    params = example["params"]
    assert isinstance(params, TrialParameters)
    truth = alternate if params.outcome else original
    selected = [0.10, 0.30, 0.50]
    selected = [value for value in selected if value in windows]
    fig, axes = plt.subplots(2, len(selected) + 1, figsize=(11.4, 5.2), constrained_layout=True)

    axes[0, 0].imshow(np.where(mask, truth, np.nan), cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 0].set_title("Final hidden target")
    axes[1, 0].imshow(
        np.where(mask, decode_anatomy(truth, mask), np.nan),
        cmap=ListedColormap(["#3957A5", "#EEEEEE", "#D64B40"]),
        vmin=0,
        vmax=2,
    )
    axes[1, 0].set_title("Final target anatomy")

    for column, fraction in enumerate(selected, start=1):
        end = max(6, int(round(fraction * (len(movie) - 1))) + 1)
        inferred = infer_target_from_movie(movie[:end], dt, mask)["target"]
        inferred = np.asarray(inferred)
        axes[0, column].imshow(
            np.where(mask, inferred, np.nan), cmap="coolwarm", vmin=-1, vmax=1
        )
        axes[0, column].set_title(f"Inferred target\n{fraction:.0%} observed")
        axes[1, column].imshow(
            np.where(mask, decode_anatomy(inferred, mask), np.nan),
            cmap=ListedColormap(["#3957A5", "#EEEEEE", "#D64B40"]),
            vmin=0,
            vmax=2,
        )
        axes[1, column].set_title("Decoded anatomy")

    for label, ax in zip("ABCDEFGH", axes.ravel()):
        ax.axis("off")
        _panel_label(ax, label)
    fig.suptitle(
        f"Example inverse reconstruction (alternate outcome; noise={example['noise']:.2f})",
        fontsize=13,
    )
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_performance(output: Path, summary: pd.DataFrame, main_noise: float = 0.05) -> None:
    data = summary[np.isclose(summary["measurement_noise"], main_noise)].copy()
    x = data["observation_fraction"].to_numpy() * 100
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)

    specs = [
        ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "Outcome accuracy", (0.45, 1.02)),
        ("auroc", "auroc_ci_low", "auroc_ci_high", "Outcome AUROC", (0.45, 1.02)),
        (
            "mean_macro_dice",
            "macro_dice_ci_low",
            "macro_dice_ci_high",
            "Anatomical macro-Dice",
            (0.0, 1.02),
        ),
        ("mean_nrmse", "nrmse_ci_low", "nrmse_ci_high", "Target NRMSE (lower is better)", (0.0, None)),
    ]
    colors = ["#235789", "#7A5195", "#2A9D8F", "#D95F02"]
    for label, ax, spec, color in zip("ABCD", axes.ravel(), specs, colors):
        metric, low, high, ylabel, ylim = spec
        y = data[metric].to_numpy()
        lo = data[low].to_numpy()
        hi = data[high].to_numpy()
        dynamic_label = (
            "Inverse movie"
            if metric in {"accuracy", "auroc", "mean_macro_dice", "mean_nrmse"}
            else None
        )
        ax.plot(x, y, marker="o", color=color, lw=2, label=dynamic_label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        if metric == "accuracy":
            snapshot_y = data["same_time_snapshot_accuracy"].to_numpy()
            snapshot_lo = data["same_time_snapshot_ci_low"].to_numpy()
            snapshot_hi = data["same_time_snapshot_ci_high"].to_numpy()
            ax.plot(
                x,
                snapshot_y,
                marker="s",
                color="#666666",
                ls="--",
                lw=1.5,
                label="Same-time snapshot",
            )
            ax.fill_between(
                x, snapshot_lo, snapshot_hi, color="#666666", alpha=0.10
            )
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        elif metric == "auroc":
            snapshot_y = data["same_time_snapshot_auroc"].to_numpy()
            snapshot_lo = data["same_time_snapshot_auroc_ci_low"].to_numpy()
            snapshot_hi = data["same_time_snapshot_auroc_ci_high"].to_numpy()
            ax.plot(
                x,
                snapshot_y,
                marker="s",
                color="#666666",
                ls="--",
                lw=1.5,
                label="Same-time snapshot",
            )
            ax.fill_between(
                x, snapshot_lo, snapshot_hi, color="#666666", alpha=0.10
            )
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        elif metric == "mean_macro_dice":
            snapshot_y = data["mean_same_time_snapshot_macro_dice"].to_numpy()
            ax.plot(
                x,
                snapshot_y,
                marker="s",
                color="#666666",
                ls="--",
                lw=1.5,
                label="Same-time snapshot",
            )
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        elif metric == "mean_nrmse":
            snapshot_y = data["mean_same_time_snapshot_nrmse"].to_numpy()
            ax.plot(
                x,
                snapshot_y,
                marker="s",
                color="#666666",
                ls="--",
                lw=1.5,
                label="Same-time snapshot",
            )
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.set(xlabel="Voltage movie observed (%)", ylabel=ylabel, ylim=ylim)
        ax.grid(alpha=0.22)
        _panel_label(ax, label)
    fig.suptitle("Inverse-model performance at 5% measurement noise", fontsize=13)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(
    output: Path,
    effects: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    template_separation: pd.DataFrame,
    identifiability: pd.DataFrame,
) -> None:

    fractions = sorted(effects["observation_fraction"].unique())
    noises = sorted(effects["measurement_noise"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    specifications = [
        ("accuracy", "Accuracy difference", 1.0),
        ("macro_dice", "Macro-Dice difference", 1.0),
        ("nrmse", "NRMSE improvement", -1.0),
    ]
    for label, ax, (metric, title, direction) in zip(
        "ABC", axes[0], specifications
    ):
        selected = effects[effects["metric"] == metric].copy()
        selected["display_effect"] = (
            direction * selected["paired_effect_movie_minus_snapshot"]
        )
        pivot = selected.pivot(
            index="measurement_noise",
            columns="observation_fraction",
            values="display_effect",
        ).reindex(index=noises, columns=fractions)
        limit = max(0.02, float(np.nanmax(np.abs(pivot.to_numpy()))))
        im = ax.imshow(
            pivot.to_numpy(),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_xticks(range(len(fractions)), [f"{f:.0%}" for f in fractions])
        ax.set_yticks(range(len(noises)), [f"{n:.2f}" for n in noises])
        ax.set(
            xlabel="Movie observed",
            ylabel="Measurement-noise SD",
            title=title,
        )
        for row in range(len(noises)):
            for column in range(len(fractions)):
                value = float(pivot.iloc[row, column])
                ax.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="white" if abs(value) > 0.60 * limit else "black",
                )
        colorbar = fig.colorbar(im, ax=ax, shrink=0.84)
        colorbar.set_label("Positive values favor the movie")
        _panel_label(ax, label)

    robustness_names = [
        "shifted_holdout",
        "nonlinear_correction",
        "time_varying_correction",
        "anisotropic_coupling",
    ]
    robust = scenario_summary[
        scenario_summary["scenario"].isin(robustness_names)
    ].set_index("scenario").reindex(robustness_names)
    robust_labels = ["Shifted", "Nonlinear", "Time-varying", "Anisotropic"]
    error = np.vstack(
        [
            robust["accuracy"] - robust["accuracy_ci_low"],
            robust["accuracy_ci_high"] - robust["accuracy"],
        ]
    )
    axes[1, 0].bar(
        robust_labels,
        robust["accuracy"],
        yerr=error,
        capsize=3,
        color=["#235789", "#7A5195", "#E9C46A", "#2A9D8F"],
    )
    axes[1, 0].set(
        ylim=(0.45, 1.02),
        ylabel="Outcome accuracy",
        title="Shifted holdout and misspecification",
    )
    axes[1, 0].tick_params(axis="x", rotation=24)
    axes[1, 0].grid(axis="y", alpha=0.22)
    _panel_label(axes[1, 0], "D")

    difficulty = scenario_summary[
        scenario_summary["scenario"].str.startswith("target_scale_")
    ].copy()
    difficulty = difficulty.merge(
        template_separation[["separation_scale", "nrmse_full_voltage_range"]],
        on="separation_scale",
        how="left",
    ).sort_values("nrmse_full_voltage_range")
    axes[1, 1].plot(
        difficulty["nrmse_full_voltage_range"],
        difficulty["accuracy"],
        marker="o",
        color="#D95F02",
        lw=2,
    )
    axes[1, 1].set(
        xlabel="Template separation NRMSE",
        ylabel="Outcome accuracy",
        ylim=(0.45, 1.02),
        title="Graded target difficulty",
    )
    axes[1, 1].grid(alpha=0.22)
    _panel_label(axes[1, 1], "E")

    axes[1, 2].scatter(
        identifiability["estimated_correction_rate"],
        identifiability["estimated_coupling"],
        c=identifiability["correct"].astype(int),
        cmap=ListedColormap(["#A61C3C", "#2A9D8F"]),
        s=22,
        alpha=0.72,
    )
    axes[1, 2].scatter(
        [], [], color="#2A9D8F", s=22, label="Correct outcome"
    )
    axes[1, 2].scatter(
        [], [], color="#A61C3C", s=22, label="Incorrect outcome"
    )
    estimate_r = float(
        np.corrcoef(
            identifiability["estimated_correction_rate"],
            identifiability["estimated_coupling"],
        )[0, 1]
    )
    boundary_n = int(identifiability["any_coefficient_boundary_hit"].sum())
    axes[1, 2].text(
        0.98,
        0.96,
        f"Pearson r = {estimate_r:.3f}\nBoundary hit: {boundary_n}/{len(identifiability)}",
        transform=axes[1, 2].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    axes[1, 2].set(
        xlabel="Estimated correction rate",
        ylabel="Estimated coupling",
        title="Compensating coefficient estimates",
    )
    axes[1, 2].grid(alpha=0.22)
    axes[1, 2].legend(frameon=False, fontsize=7, loc="lower left")
    _panel_label(axes[1, 2], "F")
    fig.suptitle(
        "Robustness, target difficulty, and parameter identifiability",
        fontsize=13,
    )
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_decision_time_original(output: Path, decisions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    main = decisions[np.isclose(decisions["measurement_noise"], 0.05)].dropna()
    axes[0].scatter(
        main["distance_from_separatrix"],
        main["earliest_stable_fraction"] * 100,
        c=main["outcome"],
        cmap=ListedColormap(["#235789", "#D95F02"]),
        s=25,
        alpha=0.72,
        edgecolors="none",
    )
    axes[0].set_xscale("log")
    axes[0].set(
        xlabel="Initial distance from bistable boundary |q₀ − 0.5|",
        ylabel="Earliest reliable decision (% of movie)",
        title="Ambiguity near the separatrix",
    )
    axes[0].grid(alpha=0.22)

    noises = sorted(decisions["measurement_noise"].unique())
    values = [
        decisions[np.isclose(decisions["measurement_noise"], noise)][
            "earliest_stable_fraction"
        ].dropna().to_numpy()
        * 100
        for noise in noises
    ]
    axes[1].boxplot(values, tick_labels=[f"{n:.2f}" for n in noises], showfliers=False)
    axes[1].set(
        xlabel="Measurement-noise SD",
        ylabel="Earliest reliable decision (% of movie)",
        title="Decision time under measurement noise",
    )
    axes[1].grid(axis="y", alpha=0.22)
    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")
    fig.suptitle("When the final morphology became predictable", fontsize=13)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_decision_time(
    output: Path,
    decisions: pd.DataFrame,
    prospective: pd.DataFrame,
) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    main_all = decisions[np.isclose(decisions["measurement_noise"], MAIN_NOISE)].copy()
    main = main_all.dropna(subset=["earliest_stable_fraction"])
    axes[0].scatter(
        main["distance_from_separatrix"],
        main["earliest_stable_fraction"] * 100,
        c=main["outcome"],
        cmap=ListedColormap(["#235789", "#D95F02"]),
        s=25,
        alpha=0.72,
        edgecolors="none",
    )
    missing = main_all[main_all["earliest_stable_fraction"].isna()]
    axes[0].scatter(
        missing["distance_from_separatrix"],
        np.full(len(missing), 55.0),
        marker="x",
        color="#A61C3C",
        s=35,
        label=f"No stable decision (n={len(missing)})",
    )
    axes[0].set_xscale("log")
    axes[0].set(
        xlabel=r"Initial distance from bistable boundary $|q_0 - 0.5|$",
        ylabel="Earliest reliable decision (% of movie)",
        title="Retrospective decision time",
    )
    axes[0].set_yticks(
        [5, 10, 20, 30, 50, 55],
        ["5", "10", "20", "30", "50", "No decision"],
    )
    axes[0].text(
        0.98,
        0.98,
        f"No stable decision: n={len(missing)}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    axes[0].grid(alpha=0.22)

    noises = sorted(decisions["measurement_noise"].unique())
    values = [
        decisions[np.isclose(decisions["measurement_noise"], noise)][
            "earliest_stable_fraction"
        ].dropna().to_numpy()
        * 100
        for noise in noises
    ]
    axes[1].boxplot(
        values, tick_labels=[f"{noise:.2f}" for noise in noises], showfliers=False
    )
    axes[1].set(
        xlabel="Measurement-noise SD",
        ylabel="Earliest reliable decision (% of movie)",
        title="Decision time under measurement noise",
    )
    no_decisions = [
        int(
            decisions[np.isclose(decisions["measurement_noise"], noise)][
                "earliest_stable_fraction"
            ].isna().sum()
        )
        for noise in noises
    ]
    for position, count in enumerate(no_decisions, start=1):
        axes[1].text(
            position, 53.0, f"ND={count}", ha="center", va="bottom", fontsize=7
        )
    axes[1].set_ylim(0, 58)
    axes[1].grid(axis="y", alpha=0.22)

    fraction_order = list(DEFAULT_WINDOWS[1:])
    counts = [
        int(np.isclose(prospective["decision_fraction"], fraction).sum())
        for fraction in fraction_order
    ]
    no_decision_count = int((~prospective["decision_made"]).sum())
    categories = [f"{fraction:.0%}" for fraction in fraction_order] + ["No decision"]
    bars = axes[2].bar(
        categories,
        counts + [no_decision_count],
        color=["#2A9D8F"] * len(counts) + ["#A61C3C"],
    )
    axes[2].bar_label(bars, fontsize=8)
    axes[2].set(
        xlabel="Prospective stopping window",
        ylabel="Holdout trajectories",
        title="Prospective rule on shifted holdout",
    )
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(axis="y", alpha=0.22)
    for label, axis in zip("ABC", axes):
        _panel_label(axis, label)
    fig.suptitle(
        "Exploratory retrospective and prospective decision timing", fontsize=13
    )
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_revision_validation(
    output: Path,
    scenario_summary: pd.DataFrame,
    template_separation: pd.DataFrame,
    identifiability: pd.DataFrame,
) -> None:

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)

    robustness_names = [
        "shifted_holdout",
        "nonlinear_correction",
        "anisotropic_coupling",
    ]
    robust = scenario_summary[
        scenario_summary["scenario"].isin(robustness_names)
    ].set_index("scenario").reindex(robustness_names)
    labels = [
        "Shifted\nholdout",
        "Nonlinear\ncorrection",
        "Anisotropic\ncoupling",
    ]
    accuracy_error = np.vstack(
        [
            robust["accuracy"] - robust["accuracy_ci_low"],
            robust["accuracy_ci_high"] - robust["accuracy"],
        ]
    )
    axes[0, 0].bar(
        labels,
        robust["accuracy"],
        yerr=accuracy_error,
        capsize=3,
        color=["#235789", "#7A5195", "#2A9D8F"],
    )
    axes[0, 0].set(
        ylim=(0.45, 1.02),
        ylabel="Outcome accuracy",
        title="Independent challenge cohorts",
    )
    axes[0, 0].grid(axis="y", alpha=0.22)

    difficulty = scenario_summary[
        scenario_summary["scenario"].str.startswith("target_scale_")
    ].copy()
    difficulty = difficulty.merge(
        template_separation[["separation_scale", "nrmse_full_voltage_range"]],
        on="separation_scale",
        how="left",
    ).sort_values("nrmse_full_voltage_range")
    axes[0, 1].plot(
        difficulty["nrmse_full_voltage_range"],
        difficulty["accuracy"],
        marker="o",
        color="#D95F02",
        lw=2,
    )
    axes[0, 1].set(
        xlabel="Template separation NRMSE",
        ylabel="Outcome accuracy",
        ylim=(0.45, 1.02),
        title="Graded target difficulty",
    )
    axes[0, 1].grid(alpha=0.22)

    axes[0, 2].hist(
        identifiability["raw_correction_rate"],
        bins=20,
        color="#7A5195",
        alpha=0.82,
    )
    axes[0, 2].axvline(
        0.08, color="black", ls="--", lw=1, label="Fitted lower bound"
    )
    axes[0, 2].set(
        xlabel="Raw estimated correction rate",
        ylabel="Trajectories",
        title="Unbounded correction estimates",
    )
    axes[0, 2].legend(frameon=False, fontsize=7)

    axes[1, 0].scatter(
        identifiability["realized_correction_map_mean"],
        identifiability["estimated_correction_rate"],
        s=18,
        alpha=0.65,
        color="#235789",
    )
    axes[1, 0].plot([0, 1.3], [0, 1.3], color="black", ls="--", lw=1)
    axes[1, 0].set(
        xlabel="Realized mean correction rate",
        ylabel="Estimated global correction rate",
        title="Correction-rate recovery",
    )
    axes[1, 0].grid(alpha=0.22)

    axes[1, 1].scatter(
        identifiability["true_coupling"],
        identifiability["estimated_coupling"],
        s=18,
        alpha=0.65,
        color="#2A9D8F",
    )
    axes[1, 1].plot([0, 0.65], [0, 0.65], color="black", ls="--", lw=1)
    axes[1, 1].set(
        xlabel="True isotropic coupling",
        ylabel="Estimated coupling",
        title="Coupling recovery",
    )
    axes[1, 1].grid(alpha=0.22)

    axes[1, 2].scatter(
        identifiability["estimated_correction_rate"],
        identifiability["estimated_coupling"],
        c=identifiability["correct"].astype(int),
        cmap=ListedColormap(["#A61C3C", "#2A9D8F"]),
        s=22,
        alpha=0.72,
    )
    axes[1, 2].scatter(
        [], [], color="#2A9D8F", s=22, label="Correct outcome"
    )
    axes[1, 2].scatter(
        [], [], color="#A61C3C", s=22, label="Incorrect outcome"
    )
    axes[1, 2].set(
        xlabel="Estimated correction rate",
        ylabel="Estimated coupling",
        title="Compensating coefficient estimates",
    )
    axes[1, 2].grid(alpha=0.22)
    axes[1, 2].legend(frameon=False, fontsize=7, loc="lower left")

    for label, axis in zip("ABCDEF", axes.ravel()):
        _panel_label(axis, label)
    fig.suptitle(
        "Reviewer-requested robustness and identifiability analyses", fontsize=13
    )
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def run_study(
    output_dir: Path,
    n_base_trials: int,
    seed: int,
    windows: tuple[float, ...],
    noise_levels: tuple[float, ...],
) -> None:
    run_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    dt = 0.05
    n_steps = 220
    mask, x, _, original, alternate, original_labels, alternate_labels = make_geometry()

    if n_base_trials % 2:
        raise ValueError("n_base_trials must be even so the two initial sides are balanced.")

    rows: list[dict[str, float | int]] = []
    trial_parameter_rows: list[dict[str, float | int]] = []
    baseline_rows: list[dict[str, float | int]] = []
    ablation_rows: list[dict[str, float | int | str | bool]] = []
    ablation_specifications: list[tuple[str, dict[str, object]]] = [
        ("default", {}),
        ("no_smoothing", {"smoothing": "none"}),
        ("derivative_window_11", {"derivative_window": 11}),
        ("derivative_window_31", {"derivative_window": 31}),
        (
            "wider_coefficient_bounds",
            {"coefficient_bounds": ((0.01, 4.0), (0.0, 1.20))},
        ),
        ("temporal_mean", {"temporal_aggregation": "mean"}),
    ]
    example_for_model: dict[str, object] | None = None
    example_reconstruction: dict[str, object] | None = None

    initial_sides = np.array([0] * (n_base_trials // 2) + [1] * (n_base_trials // 2))
    rng.shuffle(initial_sides)

    for trial_id, initial_side_value in enumerate(initial_sides):
        initial_side = int(initial_side_value)
        movie, q_trace, params, damage = simulate_trial(
            rng,
            trial_id,
            initial_side,
            mask,
            x,
            original,
            alternate,
            dt,
            n_steps,
        )
        outcome = params.outcome
        trial_parameter_rows.append(asdict(params))
        if example_for_model is None and outcome == 1 and params.distance_from_separatrix > 0.08:
            example_for_model = {
                "movie": movie.copy(),
                "q_trace": q_trace.copy(),
                "params": params,
                "damage": damage.copy(),
            }

        truth_target = alternate if outcome else original
        truth_labels = alternate_labels if outcome else original_labels

        for noise in noise_levels:
            observed = movie.astype(float).copy()
            if noise > 0:
                measurement = rng.normal(0.0, noise, size=observed.shape)
                measurement[:, ~mask] = 0.0
                observed += measurement
            observed[:, ~mask] = 0.0

            snapshot_q = project_q(observed[0], original, alternate, mask)
            baseline_rows.append(
                {
                    "trial_id": trial_id,
                    "measurement_noise": noise,
                    "outcome": outcome,
                    "snapshot_q": snapshot_q,
                    "snapshot_prediction": int(snapshot_q >= 0.5),
                }
            )

            if (
                example_reconstruction is None
                and outcome == 1
                and np.isclose(noise, 0.05)
                and 0.04 < params.distance_from_separatrix < 0.16
            ):
                example_reconstruction = {
                    "observed": observed.copy(),
                    "params": params,
                    "noise": noise,
                }

            for fraction in windows:
                end = max(6, int(round(fraction * n_steps)) + 1)
                inverse = infer_target_from_movie(observed[:end], dt, mask)
                target_hat = np.asarray(inverse["target"])
                q_hat = project_q(target_hat, original, alternate, mask)
                prediction = int(q_hat >= 0.5)
                same_time_snapshot_q = project_q(
                    observed[end - 1], original, alternate, mask
                )
                same_time_snapshot_prediction = int(same_time_snapshot_q >= 0.5)
                metrics = field_metrics(
                    target_hat, truth_target, truth_labels, mask
                )
                same_time_snapshot_metrics = field_metrics(
                    observed[end - 1], truth_target, truth_labels, mask
                )
                rows.append(
                    {
                        "trial_id": trial_id,
                        "measurement_noise": noise,
                        "observation_fraction": fraction,
                        "observations_used": end,
                        "outcome": outcome,
                        "prediction": prediction,
                        "same_time_snapshot_q": same_time_snapshot_q,
                        "same_time_snapshot_prediction": same_time_snapshot_prediction,
                        "q0": params.q0,
                        "q_final": float(q_trace[-1]),
                        "q_hat": q_hat,
                        "distance_from_separatrix": params.distance_from_separatrix,
                        "damage_fraction": params.damage_fraction_realized,
                        "true_correction_rate": params.correction_rate,
                        "realized_correction_map_mean": params.correction_map_mean,
                        "realized_correction_map_sd": params.correction_map_sd,
                        "realized_correction_map_min": params.correction_map_min,
                        "realized_correction_map_max": params.correction_map_max,
                        "estimated_correction_rate": float(
                            inverse["correction_rate"]
                        ),
                        "raw_correction_rate": float(
                            inverse["raw_correction_rate"]
                        ),
                        "correction_hit_lower": bool(
                            inverse["correction_hit_lower"]
                        ),
                        "correction_hit_upper": bool(
                            inverse["correction_hit_upper"]
                        ),
                        "correction_relative_error": abs(
                            float(inverse["correction_rate"])
                            - params.correction_rate
                        )
                        / params.correction_rate,
                        "true_coupling": params.coupling,
                        "estimated_coupling": float(inverse["coupling"]),
                        "raw_coupling": float(inverse["raw_coupling"]),
                        "coupling_hit_lower": bool(inverse["coupling_hit_lower"]),
                        "coupling_hit_upper": bool(inverse["coupling_hit_upper"]),
                        "coupling_absolute_error": abs(
                            float(inverse["coupling"]) - params.coupling
                        ),
                        **{
                            f"same_time_snapshot_{key}": value
                            for key, value in same_time_snapshot_metrics.items()
                        },
                        **metrics,
                    }
                )

                if np.isclose(noise, MAIN_NOISE) and np.isclose(
                    fraction, MAIN_FRACTION
                ):
                    for ablation_name, options in ablation_specifications:
                        ablation_inverse = (
                            inverse
                            if ablation_name == "default"
                            else infer_target_from_movie(
                                observed[:end], dt, mask, **options
                            )
                        )
                        ablation_target = np.asarray(ablation_inverse["target"])
                        ablation_q = project_q(
                            ablation_target, original, alternate, mask
                        )
                        ablation_metrics = field_metrics(
                            ablation_target, truth_target, truth_labels, mask
                        )
                        ablation_rows.append(
                            {
                                "trial_id": trial_id,
                                "outcome": outcome,
                                "ablation": ablation_name,
                                "q_hat": ablation_q,
                                "prediction": int(ablation_q >= 0.5),
                                "estimated_correction_rate": float(
                                    ablation_inverse["correction_rate"]
                                ),
                                "estimated_coupling": float(
                                    ablation_inverse["coupling"]
                                ),
                                "raw_correction_rate": float(
                                    ablation_inverse["raw_correction_rate"]
                                ),
                                "raw_coupling": float(
                                    ablation_inverse["raw_coupling"]
                                ),
                                "correction_hit_lower": bool(
                                    ablation_inverse["correction_hit_lower"]
                                ),
                                "correction_hit_upper": bool(
                                    ablation_inverse["correction_hit_upper"]
                                ),
                                "coupling_hit_lower": bool(
                                    ablation_inverse["coupling_hit_lower"]
                                ),
                                "coupling_hit_upper": bool(
                                    ablation_inverse["coupling_hit_upper"]
                                ),
                                "any_coefficient_boundary_hit": bool(
                                    ablation_inverse["correction_hit_lower"]
                                    or ablation_inverse["correction_hit_upper"]
                                    or ablation_inverse["coupling_hit_lower"]
                                    or ablation_inverse["coupling_hit_upper"]
                                ),
                                "derivative_window": int(
                                    ablation_inverse["derivative_window"]
                                ),
                                "smoothing": str(ablation_inverse["smoothing"]),
                                "temporal_aggregation": str(
                                    ablation_inverse["temporal_aggregation"]
                                ),
                                **ablation_metrics,
                            }
                        )

    metrics_frame = pd.DataFrame(rows)
    parameters_frame = pd.DataFrame(trial_parameter_rows)
    baseline_frame = pd.DataFrame(baseline_rows)
    ablation_frame = pd.DataFrame(ablation_rows)
    summary = summarize_metrics(metrics_frame, seed)
    decisions = earliest_stable_decision(metrics_frame)
    full_grid_effects = make_full_grid_effects(metrics_frame, seed)
    ablation_summary = summarize_ablations(ablation_frame, seed)

    metrics_frame.to_csv(results_dir / "trial_window_metrics.csv", index=False)
    metrics_frame[
        [
            "trial_id",
            "measurement_noise",
            "observation_fraction",
            "outcome",
            "same_time_snapshot_q",
            "same_time_snapshot_prediction",
        ]
    ].to_csv(results_dir / "same_time_snapshot_control.csv", index=False)
    parameters_frame.to_csv(results_dir / "trial_parameters.csv", index=False)
    baseline_frame.to_csv(results_dir / "snapshot_baseline.csv", index=False)
    summary.to_csv(results_dir / "summary_by_noise_and_window.csv", index=False)
    decisions.to_csv(results_dir / "decision_times.csv", index=False)
    full_grid_effects.to_csv(
        results_dir / "paired_movie_vs_snapshot_full_grid.csv", index=False
    )
    ablation_frame.to_csv(results_dir / "ablation_trial_metrics.csv", index=False)
    ablation_summary.to_csv(results_dir / "ablation_summary.csv", index=False)

    main_noise = MAIN_NOISE
    main_fraction = MAIN_FRACTION
    main = metrics_frame[
        np.isclose(metrics_frame["measurement_noise"], main_noise)
        & np.isclose(metrics_frame["observation_fraction"], main_fraction)
    ].sort_values("trial_id")
    baseline = baseline_frame[
        np.isclose(baseline_frame["measurement_noise"], main_noise)
    ].sort_values("trial_id")

    dynamic_correct = (
        main["prediction"].to_numpy() == main["outcome"].to_numpy()
    )
    baseline_correct = (
        baseline["snapshot_prediction"].to_numpy()
        == baseline["outcome"].to_numpy()
    )
    same_time_correct = (
        main["same_time_snapshot_prediction"].to_numpy()
        == main["outcome"].to_numpy()
    )
    dynamic_only = int(np.sum(dynamic_correct & ~baseline_correct))
    baseline_only = int(np.sum(~dynamic_correct & baseline_correct))
    discordant = dynamic_only + baseline_only
    mcnemar_p = (
        float(
            binomtest(
                min(dynamic_only, baseline_only), discordant, p=0.5, alternative="two-sided"
            ).pvalue
        )
        if discordant
        else 1.0
    )
    primary_effect, primary_effect_low, primary_effect_high = paired_mean_ci(
        dynamic_correct.astype(float),
        baseline_correct.astype(float),
        seed + 50_000,
    )

    dynamic_only_same_time = int(np.sum(dynamic_correct & ~same_time_correct))
    same_time_only = int(np.sum(~dynamic_correct & same_time_correct))
    same_time_discordant = dynamic_only_same_time + same_time_only
    same_time_mcnemar_p = (
        float(
            binomtest(
                min(dynamic_only_same_time, same_time_only),
                same_time_discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if same_time_discordant
        else 1.0
    )
    same_time_accuracy_effect, same_time_accuracy_low, same_time_accuracy_high = (
        paired_mean_ci(
            dynamic_correct.astype(float),
            same_time_correct.astype(float),
            seed + 50_001,
        )
    )
    dice_effect, dice_effect_low, dice_effect_high = paired_mean_ci(
        main["macro_dice"].to_numpy(),
        main["same_time_snapshot_macro_dice"].to_numpy(),
        seed + 50_002,
    )
    dice_nonempty_effect, dice_nonempty_low, dice_nonempty_high = paired_mean_ci(
        main["macro_dice_nonempty_classes"].to_numpy(),
        main[
            "same_time_snapshot_macro_dice_nonempty_classes"
        ].to_numpy(),
        seed + 50_005,
    )
    nrmse_effect, nrmse_effect_low, nrmse_effect_high = paired_mean_ci(
        main["nrmse"].to_numpy(),
        main["same_time_snapshot_nrmse"].to_numpy(),
        seed + 50_003,
    )
    main_labels = main["outcome"].to_numpy(dtype=int)
    baseline_labels = baseline["outcome"].to_numpy(dtype=int)
    if not np.array_equal(main_labels, baseline_labels):
        raise RuntimeError("Immediate and movie AUROC controls are not aligned.")
    movie_scores = main["q_hat"].to_numpy(dtype=float)
    same_time_scores = main["same_time_snapshot_q"].to_numpy(dtype=float)
    immediate_scores = baseline["snapshot_q"].to_numpy(dtype=float)
    immediate_auc_effect, immediate_auc_low, immediate_auc_high = (
        paired_auc_difference_ci(
            main_labels,
            movie_scores,
            immediate_scores,
            seed + 50_006,
        )
    )
    immediate_auc_p = paired_auc_permutation_p(
        main_labels,
        movie_scores,
        immediate_scores,
        seed + 50_007,
    )
    main_auc_grid = full_grid_effects[
        (full_grid_effects["metric"] == "auroc")
        & np.isclose(full_grid_effects["measurement_noise"], main_noise)
        & np.isclose(full_grid_effects["observation_fraction"], main_fraction)
    ].iloc[0]
    target_dice_p = paired_wilcoxon_p(
        main["macro_dice"], main["same_time_snapshot_macro_dice"]
    )
    target_nrmse_p = paired_wilcoxon_p(
        main["nrmse"], main["same_time_snapshot_nrmse"]
    )
    target_dice_nonempty_p = paired_wilcoxon_p(
        main["macro_dice_nonempty_classes"],
        main["same_time_snapshot_macro_dice_nonempty_classes"],
    )
    key_secondary_holm = holm_adjust(
        [same_time_mcnemar_p, target_dice_p, target_nrmse_p]
    )

    early = metrics_frame[
        np.isclose(metrics_frame["measurement_noise"], main_noise)
        & np.isclose(metrics_frame["observation_fraction"], 0.10)
    ].sort_values("trial_id")
    late = metrics_frame[
        np.isclose(metrics_frame["measurement_noise"], main_noise)
        & np.isclose(metrics_frame["observation_fraction"], 0.50)
    ].sort_values("trial_id")
    wilcoxon_result = wilcoxon(
        early["nrmse"].to_numpy(),
        late["nrmse"].to_numpy(),
        alternative="two-sided",
    )
    temporal_effect, temporal_effect_low, temporal_effect_high = paired_mean_ci(
        early["nrmse"].to_numpy(),
        late["nrmse"].to_numpy(),
        seed + 50_004,
    )

    main_summary = summary[
        np.isclose(summary["measurement_noise"], main_noise)
        & np.isclose(summary["observation_fraction"], main_fraction)
    ].iloc[0]
    main_decisions = decisions[
        np.isclose(decisions["measurement_noise"], main_noise)
    ]
    decision_rate = float(main_decisions["earliest_stable_fraction"].notna().mean())
    median_decision = float(main_decisions["earliest_stable_fraction"].median())

    ambiguity_frame, ambiguity_summary = make_ambiguity_outputs(
        main, parameters_frame
    )
    identifiability_frame, identifiability_summary = make_identifiability_outputs(
        main
    )
    template_separation = template_separation_table((1.0, 0.65, 0.35, 0.15))

    scenario_frames: list[pd.DataFrame] = []
    scenario_parameter_frames: list[pd.DataFrame] = []
    holdout_seed = seed + HOLDOUT_SEED_OFFSET
    difficulty_seed = seed + DIFFICULTY_SEED_OFFSET
    scenario_specifications = [
        (
            "shifted_holdout",
            holdout_seed,
            SHIFTED_HOLDOUT_RANGES,
            "linear_isotropic",
            1.0,
            0.0,
            windows,
        ),
        (
            "nonlinear_correction",
            holdout_seed,
            SHIFTED_HOLDOUT_RANGES,
            "nonlinear_correction",
            1.0,
            0.0,
            (MAIN_FRACTION,),
        ),
        (
            "time_varying_correction",
            holdout_seed,
            SHIFTED_HOLDOUT_RANGES,
            "time_varying_correction",
            1.0,
            0.0,
            (MAIN_FRACTION,),
        ),
        (
            "anisotropic_coupling",
            holdout_seed,
            SHIFTED_HOLDOUT_RANGES,
            "anisotropic_coupling",
            1.0,
            0.0,
            (MAIN_FRACTION,),
        ),
        (
            "target_scale_1.00",
            difficulty_seed,
            BASELINE_RANGES,
            "linear_isotropic",
            1.0,
            0.0,
            (MAIN_FRACTION,),
        ),
        (
            "target_scale_0.65",
            difficulty_seed,
            BASELINE_RANGES,
            "linear_isotropic",
            0.65,
            0.18,
            (MAIN_FRACTION,),
        ),
        (
            "target_scale_0.35",
            difficulty_seed,
            BASELINE_RANGES,
            "linear_isotropic",
            0.35,
            0.18,
            (MAIN_FRACTION,),
        ),
        (
            "target_scale_0.15",
            difficulty_seed,
            BASELINE_RANGES,
            "linear_isotropic",
            0.15,
            0.18,
            (MAIN_FRACTION,),
        ),
    ]
    for (
        scenario_name,
        scenario_seed,
        scenario_ranges,
        forward_model,
        separation_scale,
        target_overlap_sd,
        scenario_windows,
    ) in scenario_specifications:
        scenario_frame, scenario_parameters = evaluate_revision_cohort(
            scenario=scenario_name,
            seed=scenario_seed,
            n_trials=n_base_trials,
            ranges=scenario_ranges,
            forward_model=forward_model,
            separation_scale=separation_scale,
            target_overlap_sd=target_overlap_sd,
            observation_fractions=tuple(scenario_windows),
            measurement_noise=MAIN_NOISE,
        )
        scenario_frames.append(scenario_frame)
        scenario_parameter_frames.append(scenario_parameters)

    revision_scenarios = pd.concat(scenario_frames, ignore_index=True)
    revision_parameters = pd.concat(scenario_parameter_frames, ignore_index=True)
    scenario_summary = summarize_revision_scenarios(revision_scenarios, seed)
    prospective_input = revision_scenarios[
        revision_scenarios["scenario"] == "shifted_holdout"
    ].copy()
    prospective = prospective_decision_rule(prospective_input)
    decided = prospective[prospective["decision_made"]]
    prospective_summary = {
        "rule": (
            "Stop at the second of two consecutive tested windows when both "
            "predictions agree and both absolute q_hat margins from 0.5 are "
            f"at least {CONFIDENCE_MARGIN:.2f}."
        ),
        "evaluation_cohort": "shifted_holdout",
        "cohort_seed": holdout_seed,
        "n": int(len(prospective)),
        "decisions_made": int(prospective["decision_made"].sum()),
        "coverage": float(prospective["decision_made"].mean()),
        "accuracy_if_decided": float(decided["correct_if_decided"].mean()),
        "median_decision_fraction": float(decided["decision_fraction"].median()),
        "no_decision_n": int((~prospective["decision_made"]).sum()),
    }

    ambiguity_frame.to_csv(results_dir / "ambiguity_main_benchmark.csv", index=False)
    identifiability_frame.to_csv(
        results_dir / "parameter_identifiability_main.csv", index=False
    )
    template_separation.to_csv(results_dir / "template_separation.csv", index=False)
    revision_scenarios.to_csv(
        results_dir / "revision_scenario_trial_metrics.csv", index=False
    )
    revision_parameters.to_csv(
        results_dir / "revision_scenario_parameters.csv", index=False
    )
    scenario_summary.to_csv(
        results_dir / "revision_scenario_summary.csv", index=False
    )
    prospective.to_csv(
        results_dir / "prospective_holdout_decisions.csv", index=False
    )
    with (results_dir / "ambiguity_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(ambiguity_summary, handle, indent=2)
    with (results_dir / "parameter_identifiability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(identifiability_summary, handle, indent=2)

    exact_summary = {
        "statistical_plan": {
            "primary_hypothesis": (
                "At 5% measurement noise and 30% observation, inverse-movie "
                "outcome accuracy exceeds immediate post-damage snapshot accuracy."
            ),
            "primary_endpoint": "Paired difference in binary outcome accuracy",
            "primary_test": "Two-sided exact McNemar test",
            "key_secondary_family": (
                "Inverse movie versus same-time snapshot accuracy, macro-Dice, "
                "and NRMSE with Holm correction across the three tests."
            ),
            "exploratory_auroc_comparison": (
                "Paired trajectory-bootstrap AUROC differences and two-sided "
                "paired score-swap permutation tests, with Holm correction "
                "across the 30 full-grid AUROC comparisons."
            ),
            "exploratory_analyses": (
                "Other noise levels, observation windows, robustness scenarios, "
                "target difficulty, ablations, identifiability, and decision timing."
            ),
        },
        "study_design": {
            "random_seed": seed,
            "base_trajectories": n_base_trials,
            "outcome_counts": {
                "original": int((parameters_frame["outcome"] == 0).sum()),
                "alternate": int((parameters_frame["outcome"] == 1).sum()),
            },
            "measurement_noise_levels": list(noise_levels),
            "observation_fractions": list(windows),
            "grid_height": int(mask.shape[0]),
            "grid_width": int(mask.shape[1]),
            "active_tissue_pixels": int(mask.sum()),
            "time_step": dt,
            "integration_steps": n_steps,
            "simulated_time": dt * n_steps,
        },
        "primary_benchmark": {
            "measurement_noise": main_noise,
            "observation_fraction": main_fraction,
            "accuracy": float(main_summary["accuracy"]),
            "accuracy_ci_95": [
                float(main_summary["accuracy_ci_low"]),
                float(main_summary["accuracy_ci_high"]),
            ],
            "auroc": float(main_summary["auroc"]),
            "auroc_ci_95": [
                float(main_summary["auroc_ci_low"]),
                float(main_summary["auroc_ci_high"]),
            ],
            "mean_macro_dice": float(main_summary["mean_macro_dice"]),
            "macro_dice_ci_95": [
                float(main_summary["macro_dice_ci_low"]),
                float(main_summary["macro_dice_ci_high"]),
            ],
            "mean_macro_dice_nonempty_classes_sensitivity": float(
                main_summary["mean_macro_dice_nonempty_classes"]
            ),
            "macro_dice_nonempty_classes_sensitivity_ci_95": [
                float(main_summary["macro_dice_nonempty_classes_ci_low"]),
                float(main_summary["macro_dice_nonempty_classes_ci_high"]),
            ],
            "mean_nrmse": float(main_summary["mean_nrmse"]),
            "nrmse_ci_95": [
                float(main_summary["nrmse_ci_low"]),
                float(main_summary["nrmse_ci_high"]),
            ],
            "snapshot_accuracy": float(baseline_correct.mean()),
            "snapshot_auroc": float(fast_auc(main_labels, immediate_scores)),
            "dynamic_minus_snapshot_auroc": immediate_auc_effect,
            "dynamic_minus_snapshot_auroc_ci_95": [
                immediate_auc_low,
                immediate_auc_high,
            ],
            "dynamic_vs_snapshot_auroc_permutation_two_sided_p": immediate_auc_p,
            "dynamic_minus_snapshot_accuracy": primary_effect,
            "dynamic_minus_snapshot_accuracy_ci_95": [
                primary_effect_low,
                primary_effect_high,
            ],
            "mcnemar_exact_two_sided_p": mcnemar_p,
            "dynamic_only_correct": dynamic_only,
            "snapshot_only_correct": baseline_only,
            "same_time_snapshot_accuracy": float(same_time_correct.mean()),
            "same_time_snapshot_auroc": float(
                fast_auc(main_labels, same_time_scores)
            ),
            "dynamic_minus_same_time_snapshot_auroc": float(
                main_auc_grid["paired_effect_movie_minus_snapshot"]
            ),
            "dynamic_minus_same_time_snapshot_auroc_ci_95": [
                float(main_auc_grid["paired_effect_ci_low"]),
                float(main_auc_grid["paired_effect_ci_high"]),
            ],
            "dynamic_vs_same_time_snapshot_auroc_permutation_two_sided_p": float(
                main_auc_grid["two_sided_p"]
            ),
            "dynamic_vs_same_time_snapshot_auroc_holm_p_across_30_cells": float(
                main_auc_grid["holm_p_within_metric_30_cells"]
            ),
            "dynamic_minus_same_time_snapshot_accuracy": same_time_accuracy_effect,
            "dynamic_minus_same_time_snapshot_accuracy_ci_95": [
                same_time_accuracy_low,
                same_time_accuracy_high,
            ],
            "dynamic_vs_same_time_snapshot_mcnemar_exact_two_sided_p": same_time_mcnemar_p,
            "dynamic_vs_same_time_snapshot_mcnemar_holm_p": float(
                key_secondary_holm[0]
            ),
            "dynamic_only_correct_vs_same_time": dynamic_only_same_time,
            "same_time_only_correct": same_time_only,
        },
        "temporal_reconstruction_test": {
            "mean_nrmse_at_10_percent": float(early["nrmse"].mean()),
            "mean_nrmse_at_50_percent": float(late["nrmse"].mean()),
            "paired_mean_difference_10_minus_50_percent": temporal_effect,
            "paired_mean_difference_ci_95": [
                temporal_effect_low,
                temporal_effect_high,
            ],
            "paired_wilcoxon_statistic": float(wilcoxon_result.statistic),
            "paired_wilcoxon_two_sided_p": float(wilcoxon_result.pvalue),
        },
        "same_time_target_control": {
            "macro_dice_definition": (
                "The published three-class macro-Dice assigns 1 when a class "
                "is absent from both maps. The nonempty-class sensitivity "
                "excludes classes absent from both maps."
            ),
            "mean_dynamic_macro_dice": float(main["macro_dice"].mean()),
            "mean_snapshot_macro_dice": float(
                main["same_time_snapshot_macro_dice"].mean()
            ),
            "dynamic_minus_snapshot_macro_dice": dice_effect,
            "dynamic_minus_snapshot_macro_dice_ci_95": [
                dice_effect_low,
                dice_effect_high,
            ],
            "macro_dice_wilcoxon_two_sided_p": target_dice_p,
            "macro_dice_wilcoxon_holm_p": float(key_secondary_holm[1]),
            "mean_dynamic_macro_dice_nonempty_classes_sensitivity": float(
                main["macro_dice_nonempty_classes"].mean()
            ),
            "mean_snapshot_macro_dice_nonempty_classes_sensitivity": float(
                main[
                    "same_time_snapshot_macro_dice_nonempty_classes"
                ].mean()
            ),
            "dynamic_minus_snapshot_macro_dice_nonempty_classes_sensitivity": dice_nonempty_effect,
            "dynamic_minus_snapshot_macro_dice_nonempty_classes_sensitivity_ci_95": [
                dice_nonempty_low,
                dice_nonempty_high,
            ],
            "macro_dice_nonempty_classes_sensitivity_two_sided_p": target_dice_nonempty_p,
            "mean_dynamic_nrmse": float(main["nrmse"].mean()),
            "mean_snapshot_nrmse": float(main["same_time_snapshot_nrmse"].mean()),
            "dynamic_minus_snapshot_nrmse": nrmse_effect,
            "dynamic_minus_snapshot_nrmse_ci_95": [
                nrmse_effect_low,
                nrmse_effect_high,
            ],
            "nrmse_wilcoxon_two_sided_p": target_nrmse_p,
            "nrmse_wilcoxon_holm_p": float(key_secondary_holm[2]),
        },
        "decision_time": {
            "required_q_margin_from_boundary": 0.15,
            "fraction_with_stable_decision_by_last_tested_window": decision_rate,
            "median_earliest_stable_fraction": median_decision,
            "no_stable_decision_by_noise": {
                str(float(noise)): int(
                    decisions[
                        np.isclose(decisions["measurement_noise"], noise)
                    ]["earliest_stable_fraction"].isna().sum()
                )
                for noise in sorted(decisions["measurement_noise"].unique())
            },
            "retrospective_status": (
                "Exploratory. The rule uses the known final outcome and later "
                "observations and is not a prospective stopping criterion."
            ),
            "prospective_holdout_rule": prospective_summary,
        },
    }

    expected_headline = {
        "original_outcomes": 78,
        "alternate_outcomes": 82,
        "accuracy": 0.95625,
        "auroc": 0.9954659161976235,
        "mean_macro_dice": 0.8610585038378847,
        "mean_nrmse": 0.1452286619783303,
    }
    actual_headline = {
        "original_outcomes": int((parameters_frame["outcome"] == 0).sum()),
        "alternate_outcomes": int((parameters_frame["outcome"] == 1).sum()),
        "accuracy": float(main_summary["accuracy"]),
        "auroc": float(main_summary["auroc"]),
        "mean_macro_dice": float(main_summary["mean_macro_dice"]),
        "mean_nrmse": float(main_summary["mean_nrmse"]),
    }
    standard_design = (
        seed == SEED
        and n_base_trials == 160
        and tuple(windows) == DEFAULT_WINDOWS
        and tuple(noise_levels) == DEFAULT_NOISE_LEVELS
    )
    headline_checks = {
        key: (
            bool(actual_headline[key] == expected)
            if isinstance(expected, int)
            else bool(math.isclose(actual_headline[key], expected, abs_tol=1e-12))
        )
        for key, expected in expected_headline.items()
    }
    reproduction_check = {
        "standard_design": standard_design,
        "expected_original_headline": expected_headline,
        "actual_revised_script_headline": actual_headline,
        "checks": headline_checks,
        "all_checks_passed": bool(all(headline_checks.values())),
        "interpretation": (
            "The reviewer-requested additions did not alter the original "
            "160-trajectory benchmark pathway."
        ),
    }
    if standard_design and not reproduction_check["all_checks_passed"]:
        raise RuntimeError("The locked baseline reproduction check failed.")

    accuracy_effects = full_grid_effects[
        full_grid_effects["metric"] == "accuracy"
    ]
    auroc_effects = full_grid_effects[
        full_grid_effects["metric"] == "auroc"
    ]
    dice_effects = full_grid_effects[
        full_grid_effects["metric"] == "macro_dice"
    ]
    nrmse_effects = full_grid_effects[
        full_grid_effects["metric"] == "nrmse"
    ]
    revision_summary: dict[str, object] = {
        "baseline_reproduction": reproduction_check,
        "sampling_and_scenarios": {
            "baseline_ranges": asdict(BASELINE_RANGES),
            "shifted_holdout_ranges": asdict(SHIFTED_HOLDOUT_RANGES),
            "holdout_seed": holdout_seed,
            "target_difficulty_seed": difficulty_seed,
            "parameter_dependencies": (
                "All listed scalar parameters were sampled independently. "
                "Initial sides were balanced. The correction map was sampled "
                "once per trajectory and remained fixed over time except in "
                "the declared time-varying-rate challenge."
            ),
            "clipping": {
                "local_correction_rate": "0.25k to 2k",
                "voltage": "-1.5 to 1.5",
                "hidden_q": "0 to 1",
                "default_estimated_k": "0.08 to 2.0",
                "default_estimated_D": "0 to 0.60",
            },
            "noise_reuse": (
                "Each underlying baseline trajectory and parameter set was "
                "reused across all noise and observation conditions. A new "
                "measurement-noise realization was generated for each noise "
                "level."
            ),
            "spatial_boundary_operator": (
                "Degree-normalized four-neighbour graph Laplacian with edges "
                "only between active tissue positions and no coupling outside "
                "the mask. It is not claimed to be a generally mass-conserving "
                "physical flux discretization at variable-degree boundaries."
            ),
            "misspecification_definitions": {
                "nonlinear_correction": "k_i tanh(U - V)",
                "time_varying_correction": (
                    "k_i[1 + 0.35 sin(2 pi t/T)](U - V)"
                ),
                "anisotropic_coupling": (
                    "Horizontal and vertical coupling weights of 1.5 and 0.5"
                ),
                "inverse_retuning": False,
            },
            "target_difficulty_definition": (
                "Nominal class-template separation was scaled to 1.00, 0.65, "
                "0.35, or 0.15. The three reduced-separation conditions also "
                "used a trajectory-level target-selector offset drawn once "
                "from N(0, 0.18) and held fixed over time, creating partially "
                "overlapping class-conditional target distributions."
            ),
        },
        "primary_and_revised_statistics": exact_summary,
        "template_separation": json.loads(
            template_separation.to_json(orient="records")
        ),
        "ambiguity": ambiguity_summary,
        "identifiability": identifiability_summary,
        "scenario_summary": json.loads(scenario_summary.to_json(orient="records")),
        "ablation_summary": {
            "spatial_regularization_used": False,
            "rows": json.loads(ablation_summary.to_json(orient="records")),
        },
        "full_grid_movie_vs_snapshot": {
            "conditions": 30,
            "accuracy_movie_better": int(
                (accuracy_effects["paired_effect_movie_minus_snapshot"] > 0).sum()
            ),
            "accuracy_tied": int(
                np.isclose(
                    accuracy_effects["paired_effect_movie_minus_snapshot"], 0.0
                ).sum()
            ),
            "accuracy_movie_worse": int(
                (accuracy_effects["paired_effect_movie_minus_snapshot"] < 0).sum()
            ),
            "auroc_movie_better": int(
                (auroc_effects["paired_effect_movie_minus_snapshot"] > 0).sum()
            ),
            "auroc_tied": int(
                np.isclose(
                    auroc_effects["paired_effect_movie_minus_snapshot"], 0.0
                ).sum()
            ),
            "auroc_movie_worse": int(
                (auroc_effects["paired_effect_movie_minus_snapshot"] < 0).sum()
            ),
            "macro_dice_movie_better": int(
                (dice_effects["paired_effect_movie_minus_snapshot"] > 0).sum()
            ),
            "macro_dice_movie_worse": int(
                (dice_effects["paired_effect_movie_minus_snapshot"] < 0).sum()
            ),
            "nrmse_movie_better": int(
                (nrmse_effects["paired_effect_movie_minus_snapshot"] < 0).sum()
            ),
            "nrmse_movie_worse": int(
                (nrmse_effects["paired_effect_movie_minus_snapshot"] > 0).sum()
            ),
            "multiplicity": (
                "Two-sided P values are Holm adjusted within each metric's "
                "30-cell exploratory family."
            ),
        },
        "decision_timing": {
            "retrospective": exact_summary["decision_time"],
            "prospective_holdout": prospective_summary,
        },
        "scientific_caveats": [
            "The model is dimensionless and is not calibrated to a specific organism.",
            "q_hat is an uncalibrated geometric projection score, not a probability.",
            "Accurate target reconstruction does not imply recovery of k or D.",
            "The prospective rule was evaluated once on the shifted holdout cohort but requires external validation.",
            "The full-grid movie advantage is exploratory and depends on metric and observation condition.",
            "At target-separation scales of 0.35 and 0.15, the fixed three-level decoder maps both nominal targets to the same anatomical labels. In that regime, classification accuracy and AUROC quantify outcome difficulty, while macro-Dice is not outcome-discriminating and the smaller NRMSE partly reflects reduced target amplitude.",
        ],
    }
    with (results_dir / "exact_statistical_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(exact_summary, handle, indent=2)
    with (results_dir / "baseline_reproduction_check.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(reproduction_check, handle, indent=2)

    if example_for_model is None or example_reconstruction is None:
        raise RuntimeError("The fixed selection criteria did not yield example trials.")
    plot_model_overview(
        figures_dir / "figure1_model_overview.png",
        mask,
        original,
        alternate,
        original_labels,
        alternate_labels,
        example_for_model,
    )
    plot_example_reconstruction(
        figures_dir / "figure2_example_reconstruction.png",
        mask,
        original,
        alternate,
        example_reconstruction,
        dt,
        windows,
    )
    plot_performance(figures_dir / "figure3_prediction_performance.png", summary)
    plot_robustness(
        figures_dir / "figure4_noise_robustness.png",
        full_grid_effects,
        scenario_summary,
        template_separation,
        identifiability_frame,
    )
    plot_decision_time(
        figures_dir / "figure5_decision_time.png", decisions, prospective
    )

    runtime_seconds = float(time.perf_counter() - run_started)
    revision_summary["runtime_seconds"] = runtime_seconds
    revision_summary["generated_result_files"] = sorted(
        {
            *(path.name for path in results_dir.iterdir() if path.is_file()),
            "revision_statistical_summary.json",
        }
    )
    revision_summary["generated_figure_files"] = sorted(
        path.name for path in figures_dir.iterdir() if path.is_file()
    )
    with (results_dir / "revision_statistical_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(revision_summary, handle, indent=2)

    print(json.dumps(revision_summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that will receive figures and result tables.",
    )
    parser.add_argument(
        "--n-base-trials",
        type=int,
        default=160,
        help="Number of unique simulated trajectories (must be even).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_study(
        output_dir=arguments.output_dir,
        n_base_trials=arguments.n_base_trials,
        seed=arguments.seed,
        windows=DEFAULT_WINDOWS,
        noise_levels=DEFAULT_NOISE_LEVELS,
    )
