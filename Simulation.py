from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_WINDOWS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
DEFAULT_NOISE_LEVELS = (0.00, 0.02, 0.05, 0.10, 0.15)
PROTOTYPES = np.array([-1.0, 0.0, 1.0])  # posterior, trunk, anterior


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


def masked_laplacian(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a degree-normalized four-neighbour Laplacian with no-flux edges.

    ``field`` may be shaped ``(H, W)`` or ``(..., H, W)``. Only connections
    between two active tissue pixels contribute. Normalizing by each pixel's
    active degree avoids an artificial edge sink.
    """

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


def make_geometry(height: int = 22, width: int = 44) -> tuple[np.ndarray, ...]:
    """Create the tissue mask and two target-voltage/anatomy patterns."""

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


def decode_anatomy(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Decode voltage into posterior/trunk/anterior identity by nearest level."""

    distances = np.abs(target[..., None] - PROTOTYPES)
    labels = np.argmin(distances, axis=-1).astype(int)
    labels[~mask] = -1
    return labels


def make_damage_mask(
    x: np.ndarray, mask: np.ndarray, requested_fraction: float
) -> np.ndarray:
    """Reset a posterior strip containing approximately the requested fraction."""

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
    """Simulate one damaged tissue and its bioelectric correction movie."""

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
    )
    return movie, q_trace, params, damage


def _odd_window(n_times: int) -> int:
    candidate = min(21, n_times if n_times % 2 else n_times - 1)
    return max(5, candidate)


def infer_target_from_movie(
    observed: np.ndarray, dt: float, mask: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Estimate correction, coupling, and the latent target from voltage alone.

    For dV/dt = k(U - V) + D*L(V), subtracting each pixel's time mean
    removes the unknown, approximately constant target term. Least squares then
    estimates k and D from centered voltage dynamics. U is reconstructed by
    rearranging the fitted equation and taking a time median.
    """

    n_times = observed.shape[0]
    if n_times < 6:
        raise ValueError("At least six observations are required for inference.")

    window = _odd_window(n_times)
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
    lap = masked_laplacian(smooth, mask)

    edge = max(1, window // 5)
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
    correction_hat = float(np.clip(coefficients[0], 0.08, 2.0))
    coupling_hat = float(np.clip(coefficients[1], 0.0, 0.60))

    target_instantaneous = (
        y - coupling_hat * spatial + correction_hat * v
    ) / correction_hat
    target_values = np.median(target_instantaneous, axis=0)
    target_hat = np.zeros(mask.shape, dtype=float)
    target_hat[mask] = np.clip(target_values, -1.5, 1.5)

    return {
        "target": target_hat,
        "correction_rate": correction_hat,
        "coupling": coupling_hat,
    }


def project_q(
    field: np.ndarray, original: np.ndarray, alternate: np.ndarray, mask: np.ndarray
) -> float:
    """Project a field onto the line connecting the two candidate attractors."""

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
    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": correlation,
        "pixel_accuracy": pixel_accuracy,
        "macro_dice": dice,
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
    """Compute binary AUROC from ranks without estimator-validation overhead."""

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


def summarize_metrics(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 901)
    rows: list[dict[str, float]] = []
    for (noise, fraction), group in frame.groupby(
        ["measurement_noise", "observation_fraction"], sort=True
    ):
        labels = group["outcome"].to_numpy(dtype=int)
        predictions = group["prediction"].to_numpy(dtype=int)
        scores = group["q_hat"].to_numpy(dtype=float)
        correct = (labels == predictions).astype(float)
        same_time_predictions = group["same_time_snapshot_prediction"].to_numpy(
            dtype=int
        )
        same_time_correct = (labels == same_time_predictions).astype(float)
        acc_low, acc_high = bootstrap_ci(correct, rng)
        same_time_low, same_time_high = bootstrap_ci(same_time_correct, rng)
        auc = fast_auc(labels, scores)
        auc_low, auc_high = bootstrap_auc_ci(labels, scores, rng)
        dice_low, dice_high = bootstrap_ci(group["macro_dice"].to_numpy(), rng)
        nrmse_low, nrmse_high = bootstrap_ci(group["nrmse"].to_numpy(), rng)
        snapshot_dice_low, snapshot_dice_high = bootstrap_ci(
            group["same_time_snapshot_macro_dice"].to_numpy(), rng
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
                "mean_macro_dice": float(group["macro_dice"].mean()),
                "macro_dice_ci_low": dice_low,
                "macro_dice_ci_high": dice_high,
                "mean_nrmse": float(group["nrmse"].mean()),
                "nrmse_ci_low": nrmse_low,
                "nrmse_ci_high": nrmse_high,
                "mean_same_time_snapshot_macro_dice": float(
                    group["same_time_snapshot_macro_dice"].mean()
                ),
                "same_time_snapshot_macro_dice_ci_low": snapshot_dice_low,
                "same_time_snapshot_macro_dice_ci_high": snapshot_dice_high,
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
            if metric in {"accuracy", "mean_macro_dice", "mean_nrmse"}
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


def plot_robustness(output: Path, summary: pd.DataFrame) -> None:
    fractions = sorted(summary["observation_fraction"].unique())
    noises = sorted(summary["measurement_noise"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for label, ax, metric, title, vmin, vmax, cmap in [
        ("A", axes[0], "accuracy", "Outcome accuracy", 0.5, 1.0, "viridis"),
        ("B", axes[1], "mean_macro_dice", "Anatomical macro-Dice", 0.0, 1.0, "magma"),
    ]:
        pivot = summary.pivot(
            index="measurement_noise", columns="observation_fraction", values=metric
        ).reindex(index=noises, columns=fractions)
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(fractions)), [f"{f:.0%}" for f in fractions])
        ax.set_yticks(range(len(noises)), [f"{n:.2f}" for n in noises])
        ax.set(xlabel="Movie observed", ylabel="Measurement-noise SD", title=title)
        for row in range(len(noises)):
            for column in range(len(fractions)):
                value = pivot.iloc[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (metric == "accuracy" and value < 0.72) or (metric != "accuracy" and value < 0.55) else "black")
        fig.colorbar(im, ax=ax, shrink=0.84)
        _panel_label(ax, label)
    fig.suptitle("Robustness to observation time and voltage-measurement noise", fontsize=13)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_decision_time(output: Path, decisions: pd.DataFrame) -> None:
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


def run_study(
    output_dir: Path,
    n_base_trials: int,
    seed: int,
    windows: tuple[float, ...],
    noise_levels: tuple[float, ...],
) -> None:
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
                        "estimated_correction_rate": float(
                            inverse["correction_rate"]
                        ),
                        "correction_relative_error": abs(
                            float(inverse["correction_rate"])
                            - params.correction_rate
                        )
                        / params.correction_rate,
                        "true_coupling": params.coupling,
                        "estimated_coupling": float(inverse["coupling"]),
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

    metrics_frame = pd.DataFrame(rows)
    parameters_frame = pd.DataFrame(trial_parameter_rows)
    baseline_frame = pd.DataFrame(baseline_rows)
    summary = summarize_metrics(metrics_frame, seed)
    decisions = earliest_stable_decision(metrics_frame)

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

    main_noise = 0.05
    main_fraction = 0.30
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

    target_dice_test = wilcoxon(
        main["macro_dice"].to_numpy(),
        main["same_time_snapshot_macro_dice"].to_numpy(),
        alternative="greater",
    )
    target_nrmse_test = wilcoxon(
        main["nrmse"].to_numpy(),
        main["same_time_snapshot_nrmse"].to_numpy(),
        alternative="less",
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
        early["nrmse"].to_numpy(), late["nrmse"].to_numpy(), alternative="greater"
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

    exact_summary = {
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
            "mean_nrmse": float(main_summary["mean_nrmse"]),
            "nrmse_ci_95": [
                float(main_summary["nrmse_ci_low"]),
                float(main_summary["nrmse_ci_high"]),
            ],
            "snapshot_accuracy": float(baseline_correct.mean()),
            "dynamic_minus_snapshot_accuracy": float(
                dynamic_correct.mean() - baseline_correct.mean()
            ),
            "mcnemar_exact_p": mcnemar_p,
            "dynamic_only_correct": dynamic_only,
            "snapshot_only_correct": baseline_only,
            "same_time_snapshot_accuracy": float(same_time_correct.mean()),
            "dynamic_minus_same_time_snapshot_accuracy": float(
                dynamic_correct.mean() - same_time_correct.mean()
            ),
            "dynamic_vs_same_time_snapshot_mcnemar_exact_p": same_time_mcnemar_p,
            "dynamic_only_correct_vs_same_time": dynamic_only_same_time,
            "same_time_only_correct": same_time_only,
        },
        "temporal_reconstruction_test": {
            "mean_nrmse_at_10_percent": float(early["nrmse"].mean()),
            "mean_nrmse_at_50_percent": float(late["nrmse"].mean()),
            "paired_wilcoxon_statistic": float(wilcoxon_result.statistic),
            "paired_wilcoxon_one_sided_p": float(wilcoxon_result.pvalue),
        },
        "same_time_target_control": {
            "mean_dynamic_macro_dice": float(main["macro_dice"].mean()),
            "mean_snapshot_macro_dice": float(
                main["same_time_snapshot_macro_dice"].mean()
            ),
            "dynamic_minus_snapshot_macro_dice": float(
                main["macro_dice"].mean()
                - main["same_time_snapshot_macro_dice"].mean()
            ),
            "macro_dice_wilcoxon_one_sided_p": float(target_dice_test.pvalue),
            "mean_dynamic_nrmse": float(main["nrmse"].mean()),
            "mean_snapshot_nrmse": float(main["same_time_snapshot_nrmse"].mean()),
            "dynamic_minus_snapshot_nrmse": float(
                main["nrmse"].mean() - main["same_time_snapshot_nrmse"].mean()
            ),
            "nrmse_wilcoxon_one_sided_p": float(target_nrmse_test.pvalue),
        },
        "decision_time": {
            "required_q_margin_from_boundary": 0.15,
            "fraction_with_stable_decision_by_last_tested_window": decision_rate,
            "median_earliest_stable_fraction": median_decision,
        },
    }
    with (results_dir / "exact_statistical_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(exact_summary, handle, indent=2)

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
    plot_robustness(figures_dir / "figure4_noise_robustness.png", summary)
    plot_decision_time(figures_dir / "figure5_decision_time.png", decisions)

    print(json.dumps(exact_summary, indent=2))


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
