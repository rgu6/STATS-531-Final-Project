from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "cache" / "computedraft" / "v1" / "rl2_masked_v4"
REPORT_EXPORTS = CACHE_ROOT / "report_exports"
PLOTS_DIR = REPORT_EXPORTS / "plots"


def _load_paths() -> tuple[pd.DataFrame, dict[str, object]]:
    parameter_summary = pd.read_csv(REPORT_EXPORTS / "parameter_summary.csv")
    resolved_config = json.loads((REPORT_EXPORTS / "resolved_config.json").read_text(encoding="utf-8"))
    return parameter_summary, resolved_config


def _participant_log_k(frame: pd.DataFrame, pooling_mode: str) -> np.ndarray:
    subset = frame.loc[
        (frame["model_family"].astype(str) == "ar1")
        & (frame["pooling_mode"].astype(str) == str(pooling_mode))
    ].copy()
    if subset.empty:
        return np.array([], dtype=float)

    if "log_k_fitbit" in subset.columns and subset["log_k_fitbit"].notna().any():
        values = pd.to_numeric(subset["log_k_fitbit"], errors="coerce")
    else:
        values = np.log(pd.to_numeric(subset["k_fitbit"], errors="coerce"))
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values.to_numpy(dtype=float)


def _bootstrap_empirical_cloud(
    log_k_values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if log_k_values.size < 2:
        return pd.DataFrame(columns=["empirical_mu_log_k_fitbit", "empirical_log_tau_log_k_fitbit"])

    rng = np.random.default_rng(int(seed))
    sample_size = int(log_k_values.size)
    draws = []
    for _ in range(int(n_boot)):
        sample = rng.choice(log_k_values, size=sample_size, replace=True)
        sample_sd = float(np.std(sample, ddof=1))
        if not np.isfinite(sample_sd) or sample_sd <= 0.0:
            continue
        draws.append(
            {
                "empirical_mu_log_k_fitbit": float(np.mean(sample)),
                "empirical_log_tau_log_k_fitbit": float(np.log(sample_sd)),
            }
        )
    return pd.DataFrame(draws)


def _full_sample_point(log_k_values: np.ndarray) -> tuple[float, float]:
    sample_sd = float(np.std(log_k_values, ddof=1))
    return float(np.mean(log_k_values)), float(np.log(sample_sd))


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    parameter_summary, resolved_config = _load_paths()

    personal_log_k = _participant_log_k(parameter_summary, "unit_specific")
    partial_log_k = _participant_log_k(parameter_summary, "partial_pooled_k")

    personal_boot = _bootstrap_empirical_cloud(personal_log_k, n_boot=4000, seed=531)
    partial_boot = _bootstrap_empirical_cloud(partial_log_k, n_boot=4000, seed=532)

    personal_mu, personal_log_tau = _full_sample_point(personal_log_k)
    partial_mu, partial_log_tau = _full_sample_point(partial_log_k)

    x_low = float(resolved_config.get("mu_log_k_fitbit_window_low", 0.2))
    x_high = float(resolved_config.get("mu_log_k_fitbit_window_high", 2.0))
    y_low = float(resolved_config.get("log_tau_log_k_fitbit_window_low", -1.7))
    y_high = float(resolved_config.get("log_tau_log_k_fitbit_window_high", 1.0))

    fig, ax = plt.subplots(figsize=(11.2, 8.2))

    if not personal_boot.empty:
        ax.scatter(
            personal_boot["empirical_mu_log_k_fitbit"],
            personal_boot["empirical_log_tau_log_k_fitbit"],
            s=22,
            alpha=0.15,
            color="#2b6cb0",
            edgecolors="none",
            label="Personal bootstrap cloud",
        )
    if not partial_boot.empty:
        ax.scatter(
            partial_boot["empirical_mu_log_k_fitbit"],
            partial_boot["empirical_log_tau_log_k_fitbit"],
            s=22,
            alpha=0.15,
            color="#d97706",
            edgecolors="none",
            label="Partial bootstrap cloud",
        )

    ax.scatter(
        [personal_mu],
        [personal_log_tau],
        s=180,
        color="#1d4ed8",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="Personal full-sample empirical",
    )
    ax.scatter(
        [partial_mu],
        [partial_log_tau],
        s=180,
        color="#b91c1c",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="Partial full-sample empirical",
    )

    window_rect = patches.Rectangle(
        (x_low, y_low),
        x_high - x_low,
        y_high - y_low,
        fill=False,
        linewidth=2.4,
        linestyle="--",
        edgecolor="#16a34a",
        label="RL=2 global window",
    )
    ax.add_patch(window_rect)

    x_min = min(
        float(np.nanmin(personal_boot["empirical_mu_log_k_fitbit"])) if not personal_boot.empty else personal_mu,
        float(np.nanmin(partial_boot["empirical_mu_log_k_fitbit"])) if not partial_boot.empty else partial_mu,
        x_low,
    )
    x_max = max(
        float(np.nanmax(personal_boot["empirical_mu_log_k_fitbit"])) if not personal_boot.empty else personal_mu,
        float(np.nanmax(partial_boot["empirical_mu_log_k_fitbit"])) if not partial_boot.empty else partial_mu,
        x_high,
    )
    y_min = min(
        float(np.nanmin(personal_boot["empirical_log_tau_log_k_fitbit"])) if not personal_boot.empty else personal_log_tau,
        float(np.nanmin(partial_boot["empirical_log_tau_log_k_fitbit"])) if not partial_boot.empty else partial_log_tau,
        y_low,
    )
    y_max = max(
        float(np.nanmax(personal_boot["empirical_log_tau_log_k_fitbit"])) if not personal_boot.empty else personal_log_tau,
        float(np.nanmax(partial_boot["empirical_log_tau_log_k_fitbit"])) if not partial_boot.empty else partial_log_tau,
        y_high,
    )

    ax.set_xlim(x_min - 0.08, x_max + 0.08)
    ax.set_ylim(y_min - 0.08, y_max + 0.08)
    ax.set_xlabel("empirical mu_log_k_fitbit", fontsize=21)
    ax.set_ylabel("empirical log_tau_log_k_fitbit", fontsize=21)
    ax.set_title(
        "Empirical (mu_log_k, log tau_log_k) cloud for personal and partial pooling",
        fontsize=24,
    )
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=16)

    fig.tight_layout()

    png_path = PLOTS_DIR / "empirical_mu_logk_vs_log_tau_personal_partial.png"
    csv_path = REPORT_EXPORTS / "empirical_mu_logk_vs_log_tau_personal_partial_bootstrap.csv"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    export_boot = pd.concat(
        [
            personal_boot.assign(pooling_mode="unit_specific"),
            partial_boot.assign(pooling_mode="partial_pooled_k"),
        ],
        ignore_index=True,
    )
    export_boot.to_csv(csv_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "pooling_mode": "unit_specific",
                "empirical_mu_log_k_fitbit": personal_mu,
                "empirical_log_tau_log_k_fitbit": personal_log_tau,
            },
            {
                "pooling_mode": "partial_pooled_k",
                "empirical_mu_log_k_fitbit": partial_mu,
                "empirical_log_tau_log_k_fitbit": partial_log_tau,
            },
        ]
    )
    summary.to_csv(REPORT_EXPORTS / "empirical_mu_logk_vs_log_tau_personal_partial_summary.csv", index=False)


if __name__ == "__main__":
    main()
