from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _coerce_boolean_mask(values: pd.Series | np.ndarray | list[object] | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(False, index=index, dtype=bool)
    series = pd.Series(values, index=index) if not isinstance(values, pd.Series) else values.reindex(index)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).astype(float).gt(0.0)
    return series.fillna(False).astype(bool)


def _comparable_numeric_frame(truth: pd.Series, estimate: pd.Series) -> pd.DataFrame:
    comparable = pd.DataFrame(
        {
            "truth": pd.to_numeric(truth, errors="coerce"),
            "estimate": pd.to_numeric(estimate, errors="coerce"),
        }
    ).dropna()
    if comparable.empty:
        return comparable
    finite_mask = np.isfinite(comparable["truth"]) & np.isfinite(comparable["estimate"])
    return comparable.loc[finite_mask].reset_index(drop=True)


def _safe_rmse(truth: pd.Series, estimate: pd.Series) -> float:
    comparable = _comparable_numeric_frame(truth, estimate)
    if comparable.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(comparable["estimate"] - comparable["truth"]))))


def _safe_mae(truth: pd.Series, estimate: pd.Series) -> float:
    comparable = _comparable_numeric_frame(truth, estimate)
    if comparable.empty:
        return float("nan")
    return float(np.mean(np.abs(comparable["estimate"] - comparable["truth"])))


def _safe_corr(truth: pd.Series, estimate: pd.Series) -> float:
    comparable = _comparable_numeric_frame(truth, estimate)
    if len(comparable) < 2:
        return float("nan")
    return float(comparable["truth"].corr(comparable["estimate"]))


def build_masked_hourly_benchmark_frame(
    reconstructed_frame: pd.DataFrame,
    *,
    truth_column: str = "fitbit_truth_for_eval",
    mask_flag_column: str = "fitbit_masked_for_eval",
    positive_truth_only: bool = True,
    estimate_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    required = {truth_column, mask_flag_column}
    missing = required - set(reconstructed_frame.columns)
    if missing:
        raise ValueError(f"Reconstruction frame missing required columns: {sorted(missing)}")

    truth = pd.to_numeric(reconstructed_frame[truth_column], errors="coerce")
    masked = _coerce_boolean_mask(reconstructed_frame[mask_flag_column], reconstructed_frame.index)
    benchmark_mask = masked & truth.notna()
    if positive_truth_only:
        benchmark_mask = benchmark_mask & truth.gt(0.0)

    benchmark = reconstructed_frame.loc[benchmark_mask].copy()
    benchmark["benchmark_truth_for_eval"] = truth.loc[benchmark_mask].to_numpy(dtype=float)
    benchmark["benchmark_positive_truth_only"] = bool(positive_truth_only)

    for column in estimate_columns or ():
        if column not in benchmark.columns:
            continue
        estimate = pd.to_numeric(benchmark[column], errors="coerce")
        benchmark[f"{column}_benchmark_error"] = estimate - benchmark["benchmark_truth_for_eval"]
    return benchmark


def build_heldout_subtotal_benchmark_frame(
    reconstructed_frame: pd.DataFrame,
    reward_window_frame: pd.DataFrame,
    *,
    truth_column: str = "fitbit_truth_for_eval",
    estimate_column: str = "latent_fitbit_scale_steps",
    mask_flag_column: str = "fitbit_masked_for_eval",
) -> pd.DataFrame:
    required_hourly = {truth_column, estimate_column, mask_flag_column}
    missing_hourly = required_hourly - set(reconstructed_frame.columns)
    if missing_hourly:
        raise ValueError(f"Reconstruction frame missing required columns: {sorted(missing_hourly)}")

    required_window_columns = {"decision_id", "window_start_position", "window_end_position"}
    missing_window_columns = required_window_columns - set(reward_window_frame.columns)
    if missing_window_columns:
        raise ValueError(f"Reward window frame missing required columns: {sorted(missing_window_columns)}")

    truth = pd.to_numeric(reconstructed_frame[truth_column], errors="coerce")
    estimate = pd.to_numeric(reconstructed_frame[estimate_column], errors="coerce")
    masked = _coerce_boolean_mask(reconstructed_frame[mask_flag_column], reconstructed_frame.index)

    rows: list[dict[str, object]] = []
    for reward_row in reward_window_frame.itertuples(index=False):
        start_pos = int(getattr(reward_row, "window_start_position"))
        end_pos = int(getattr(reward_row, "window_end_position"))
        window_truth = truth.iloc[start_pos : end_pos + 1]
        window_estimate = estimate.iloc[start_pos : end_pos + 1]
        window_masked = masked.iloc[start_pos : end_pos + 1]

        heldout_mask = window_masked & window_truth.notna()
        comparable = pd.DataFrame(
            {
                "truth": window_truth.loc[heldout_mask],
                "estimate": window_estimate.loc[heldout_mask],
            }
        ).dropna()
        heldout_hours = int(len(comparable))
        if heldout_hours <= 0:
            continue

        subtotal_error = float((comparable["estimate"] - comparable["truth"]).sum())
        rows.append(
            {
                "decision_id": getattr(reward_row, "decision_id"),
                "decision_time_utc": getattr(reward_row, "decision_time_utc", pd.NaT),
                "decision_local_time": getattr(reward_row, "decision_local_time", pd.NaT),
                "decision_local_date": getattr(reward_row, "decision_local_date", pd.NaT),
                "heldout_masked_hours": heldout_hours,
                "heldout_true_subtotal": float(comparable["truth"].sum()),
                "heldout_predicted_subtotal": float(comparable["estimate"].sum()),
                "heldout_subtotal_error": subtotal_error,
                "heldout_subtotal_error_sqrtm": float(subtotal_error / np.sqrt(float(heldout_hours))),
            }
        )
    return pd.DataFrame(rows)


def summarize_hourly_metrics_by_participant(
    hourly_frame: pd.DataFrame,
    *,
    estimate_column: str = "latent_fitbit_scale_steps",
) -> pd.DataFrame:
    if hourly_frame.empty or "participant_id" not in hourly_frame.columns:
        return pd.DataFrame(columns=["participant_id", "hourly_rmse", "hourly_mae", "hourly_correlation", "hourly_benchmark_hours"])
    benchmark = build_masked_hourly_benchmark_frame(
        hourly_frame,
        truth_column="fitbit_truth_for_eval",
        mask_flag_column="fitbit_masked_for_eval",
        positive_truth_only=True,
        estimate_columns=[estimate_column],
    )
    if benchmark.empty:
        return pd.DataFrame(columns=["participant_id", "hourly_rmse", "hourly_mae", "hourly_correlation", "hourly_benchmark_hours"])
    rows: list[dict[str, object]] = []
    for participant_id, part in benchmark.groupby("participant_id", dropna=False):
        truth = pd.to_numeric(part["benchmark_truth_for_eval"], errors="coerce")
        estimate = pd.to_numeric(part[estimate_column], errors="coerce")
        rows.append(
            {
                "participant_id": str(participant_id),
                "hourly_rmse": _safe_rmse(truth, estimate),
                "hourly_mae": _safe_mae(truth, estimate),
                "hourly_correlation": _safe_corr(truth, estimate),
                "hourly_benchmark_hours": int(_comparable_numeric_frame(truth, estimate).shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_reward_metrics_by_participant(reward_frame: pd.DataFrame) -> pd.DataFrame:
    if reward_frame.empty or "participant_id" not in reward_frame.columns:
        return pd.DataFrame(columns=["participant_id", "reward_rmse", "daily_correlation", "reward_windows_compared"])
    rows: list[dict[str, object]] = []
    for participant_id, part in reward_frame.groupby("participant_id", dropna=False):
        truth = pd.to_numeric(part["observed_subtotal"], errors="coerce")
        estimate = pd.to_numeric(part["predicted_full_24h_reward"], errors="coerce")
        comparable = _comparable_numeric_frame(truth, estimate)
        rows.append(
            {
                "participant_id": str(participant_id),
                "reward_rmse": _safe_rmse(truth, estimate),
                "daily_correlation": _safe_corr(truth, estimate),
                "reward_windows_compared": int(len(comparable)),
            }
        )
    return pd.DataFrame(rows)


def summarize_daily_heldout_metrics_by_participant(
    hourly_frame: pd.DataFrame,
    reward_frame: pd.DataFrame,
    *,
    estimate_column: str = "latent_fitbit_scale_steps",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["participant_id", "daily_rmse_sqrtm", "daily_rmse", "heldout_daily_correlation", "daily_windows_used"]
    if (
        hourly_frame.empty
        or reward_frame.empty
        or "participant_id" not in hourly_frame.columns
        or "participant_id" not in reward_frame.columns
    ):
        return pd.DataFrame(columns=columns), pd.DataFrame()

    rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for participant_id in sorted(set(reward_frame["participant_id"].astype(str))):
        hourly_part = hourly_frame.loc[hourly_frame["participant_id"].astype(str) == str(participant_id)].reset_index(drop=True)
        reward_part = reward_frame.loc[reward_frame["participant_id"].astype(str) == str(participant_id)].reset_index(drop=True)
        detail = build_heldout_subtotal_benchmark_frame(
            hourly_part,
            reward_part,
            truth_column="fitbit_truth_for_eval",
            estimate_column=estimate_column,
            mask_flag_column="fitbit_masked_for_eval",
        )
        if detail.empty:
            rows.append(
                {
                    "participant_id": str(participant_id),
                    "daily_rmse_sqrtm": np.nan,
                    "daily_rmse": np.nan,
                    "heldout_daily_correlation": np.nan,
                    "daily_windows_used": 0,
                }
            )
            continue
        detail = detail.copy()
        detail["participant_id"] = str(participant_id)
        detail_frames.append(detail)
        truth = pd.to_numeric(detail["heldout_true_subtotal"], errors="coerce")
        estimate = pd.to_numeric(detail["heldout_predicted_subtotal"], errors="coerce")
        normalized_error = pd.to_numeric(detail["heldout_subtotal_error_sqrtm"], errors="coerce")
        rows.append(
            {
                "participant_id": str(participant_id),
                "daily_rmse_sqrtm": float(np.sqrt(np.nanmean(np.square(normalized_error)))) if normalized_error.notna().any() else np.nan,
                "daily_rmse": _safe_rmse(truth, estimate),
                "heldout_daily_correlation": _safe_corr(truth, estimate),
                "daily_windows_used": int(_comparable_numeric_frame(truth, estimate).shape[0]),
            }
        )
    detail_frame = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return pd.DataFrame(rows), detail_frame


def build_parameter_summary(masked_fits: dict[tuple[str, str, str], dict[str, object]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (model_family, pooling_mode, mask_id), artifact in sorted(masked_fits.items()):
        participant_estimates = artifact.get("participant_estimates", pd.DataFrame()).copy()
        if participant_estimates.empty:
            continue
        if "participantidentifier" in participant_estimates.columns and "participant_id" not in participant_estimates.columns:
            participant_estimates["participant_id"] = participant_estimates["participantidentifier"].astype(str)
        participant_estimates["model_family"] = str(model_family)
        participant_estimates["pooling_mode"] = str(pooling_mode)
        participant_estimates["mask_id"] = str(mask_id)
        participant_estimates["estimate_source"] = "derived_masked_fit_recovery"
        rows.append(participant_estimates)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_summary_frames(
    masked_fits: dict[tuple[str, str, str], dict[str, object]],
    parameter_summary: pd.DataFrame,
    *,
    benchmark_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark_rows: list[dict[str, object]] = []
    participant_metric_frames: list[pd.DataFrame] = []
    heldout_detail_frames: list[pd.DataFrame] = []

    for (model_family, pooling_mode, mask_id), artifact in sorted(masked_fits.items()):
        hourly_frame = artifact.get("hourly_prediction_frame", pd.DataFrame()).copy()
        reward_frame = artifact.get("reward_window_frame", pd.DataFrame()).copy()
        if hourly_frame.empty or reward_frame.empty:
            continue

        hourly_metrics = summarize_hourly_metrics_by_participant(hourly_frame)
        reward_metrics = summarize_reward_metrics_by_participant(reward_frame)
        daily_metrics, daily_detail = summarize_daily_heldout_metrics_by_participant(hourly_frame, reward_frame)
        combined = hourly_metrics.merge(reward_metrics, how="outer", on="participant_id").merge(
            daily_metrics,
            how="outer",
            on="participant_id",
        )
        if not combined.empty:
            combined["model_family"] = str(model_family)
            combined["pooling_mode"] = str(pooling_mode)
            combined["mask_id"] = str(mask_id)
            combined["estimate_type"] = "filter"
            participant_metric_frames.append(combined)

        pooled_detail = build_heldout_subtotal_benchmark_frame(hourly_frame, reward_frame)
        if not pooled_detail.empty:
            pooled_detail = pooled_detail.copy()
            pooled_detail["model_family"] = str(model_family)
            pooled_detail["pooling_mode"] = str(pooling_mode)
            pooled_detail["mask_id"] = str(mask_id)
            pooled_detail["estimate_type"] = "filter"
            heldout_detail_frames.append(pooled_detail)

        mean_loglik = np.nan
        if not parameter_summary.empty:
            part = parameter_summary.loc[
                (parameter_summary["model_family"].astype(str) == str(model_family))
                & (parameter_summary["pooling_mode"].astype(str) == str(pooling_mode))
            ].copy()
            if not part.empty and "panel_unit_loglik" in part.columns:
                mean_loglik = float(pd.to_numeric(part["panel_unit_loglik"], errors="coerce").mean())

        benchmark_rows.append(
            {
                "benchmark_key": "core_masking",
                "benchmark_label": "Masked hourly + held-out subtotal",
                "benchmark_version": str(benchmark_version),
                "model_family": str(model_family),
                "pooling_mode": str(pooling_mode),
                "mask_id": str(mask_id),
                "estimate_type": "filter",
                "hourly_masked_rmse": float(pd.to_numeric(hourly_metrics.get("hourly_rmse"), errors="coerce").mean()) if not hourly_metrics.empty else np.nan,
                "hourly_masked_mae": float(pd.to_numeric(hourly_metrics.get("hourly_mae"), errors="coerce").mean()) if not hourly_metrics.empty else np.nan,
                "hourly_masked_correlation": float(pd.to_numeric(hourly_metrics.get("hourly_correlation"), errors="coerce").mean()) if not hourly_metrics.empty else np.nan,
                "daily_correlation": float(pd.to_numeric(reward_metrics.get("daily_correlation"), errors="coerce").mean()) if not reward_metrics.empty else np.nan,
                "heldout_subtotal_rmse_sqrtm": float(pd.to_numeric(daily_metrics.get("daily_rmse_sqrtm"), errors="coerce").mean()) if not daily_metrics.empty else np.nan,
                "pooled_window_subtotal_rmse_sqrtm": float(
                    np.sqrt(
                        np.nanmean(
                            np.square(pd.to_numeric(pooled_detail.get("heldout_subtotal_error_sqrtm"), errors="coerce"))
                        )
                    )
                )
                if not pooled_detail.empty
                else np.nan,
                "mean_signed_subtotal_error": float(pd.to_numeric(pooled_detail.get("heldout_subtotal_error"), errors="coerce").mean()) if not pooled_detail.empty else np.nan,
                "mean_signed_normalized_subtotal_error": float(pd.to_numeric(pooled_detail.get("heldout_subtotal_error_sqrtm"), errors="coerce").mean()) if not pooled_detail.empty else np.nan,
                "n_windows_used": int(len(pooled_detail)) if not pooled_detail.empty else 0,
                "n_participants_used": int(pd.to_numeric(daily_metrics.get("daily_windows_used"), errors="coerce").fillna(0).gt(0).sum()) if not daily_metrics.empty else 0,
                "mean_windows_per_participant": float(
                    pd.to_numeric(daily_metrics.get("daily_windows_used"), errors="coerce").replace(0, np.nan).mean()
                )
                if not daily_metrics.empty
                else np.nan,
                "mean_masked_hours": float(pd.to_numeric(pooled_detail.get("heldout_masked_hours"), errors="coerce").mean()) if not pooled_detail.empty else np.nan,
                "mean_panel_unit_loglik": mean_loglik,
                "subtotal_metric_variant": "participant_mean_daily_rmse_sqrtm",
            }
        )

    benchmark_summary = pd.DataFrame(benchmark_rows)
    model_comparison = benchmark_summary.loc[
        :,
        [
            "benchmark_key",
            "benchmark_label",
            "model_family",
            "pooling_mode",
            "estimate_type",
            "hourly_masked_rmse",
            "hourly_masked_mae",
            "hourly_masked_correlation",
            "daily_correlation",
            "heldout_subtotal_rmse_sqrtm",
            "pooled_window_subtotal_rmse_sqrtm",
            "mean_signed_subtotal_error",
            "mean_signed_normalized_subtotal_error",
            "n_windows_used",
            "n_participants_used",
            "mean_windows_per_participant",
            "mean_masked_hours",
            "mean_panel_unit_loglik",
            "subtotal_metric_variant",
        ],
    ].copy() if not benchmark_summary.empty else pd.DataFrame()

    participant_metric_summary = (
        pd.concat(participant_metric_frames, ignore_index=True) if participant_metric_frames else pd.DataFrame()
    )
    heldout_detail_summary = (
        pd.concat(heldout_detail_frames, ignore_index=True) if heldout_detail_frames else pd.DataFrame()
    )
    return benchmark_summary, model_comparison, participant_metric_summary, heldout_detail_summary


def build_task_manifest(benchmark_summary: pd.DataFrame) -> pd.DataFrame:
    if benchmark_summary.empty:
        return pd.DataFrame(columns=["task_id", "artifact_type", "model_family", "pooling_mode", "mask_id", "status", "runtime_seconds"])
    rows = []
    for row in benchmark_summary.itertuples(index=False):
        rows.append(
            {
                "task_id": f"benchmark::core_masking::{row.model_family}::{row.pooling_mode}::{row.mask_id}",
                "artifact_type": "benchmark",
                "model_family": str(row.model_family),
                "pooling_mode": str(row.pooling_mode),
                "mask_id": str(row.mask_id),
                "status": "recovered_from_derived_artifact",
                "runtime_seconds": np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_plot_files(model_comparison: pd.DataFrame, parameter_summary: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not model_comparison.empty:
        subtotal_plot_df = model_comparison.assign(
            label=lambda df: df["model_family"].astype(str) + " / " + df["pooling_mode"].astype(str) + " / " + df["estimate_type"].astype(str)
        )
        plt.figure(figsize=(8.5, 4.5))
        plt.bar(
            subtotal_plot_df["label"],
            pd.to_numeric(subtotal_plot_df["heldout_subtotal_rmse_sqrtm"], errors="coerce"),
            color="#d4a72c",
            edgecolor="#7a5f12",
        )
        plt.ylabel("Participant-averaged RMSE Subtotal")
        plt.xlabel("Model / pooling / estimate")
        plt.title("Held-out subtotal benchmark comparison")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / "heldout_subtotal_benchmark_comparison.png", dpi=180, bbox_inches="tight")
        plt.close()

    scatter_source = parameter_summary.copy()
    if not scatter_source.empty:
        scatter_source = scatter_source.loc[scatter_source["model_family"].astype(str) == "ar1"].copy()
    required_columns = {"phi", "sigma", "k_fitbit", "pooling_mode"}
    if scatter_source.empty or not required_columns.issubset(scatter_source.columns):
        return

    scatter_source["pooling_mode"] = scatter_source["pooling_mode"].astype(str)
    color_map = {
        "unit_specific": "#1f77b4",
        "global_shared_k": "#ff7f0e",
        "partial_pooled_k": "#2ca02c",
        "not_applicable": "#7f7f7f",
    }
    scatter_specs = [
        ("phi", "sigma", "phi vs sigma", plot_dir / "ar1_phi_vs_sigma.png"),
        ("sigma", "k_fitbit", "sigma vs k", plot_dir / "ar1_sigma_vs_k.png"),
        ("phi", "k_fitbit", "phi vs k", plot_dir / "ar1_phi_vs_k.png"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (x_col, y_col, title, single_path) in zip(axes, scatter_specs):
        plotted_labels = set()
        single_fig, single_ax = plt.subplots(figsize=(6, 5))
        for pooling_mode, part in scatter_source.groupby("pooling_mode", dropna=False):
            valid = pd.DataFrame(
                {
                    "x": pd.to_numeric(part[x_col], errors="coerce"),
                    "y": pd.to_numeric(part[y_col], errors="coerce"),
                }
            ).dropna()
            if valid.empty:
                continue
            ax.scatter(
                valid["x"],
                valid["y"],
                s=28,
                alpha=0.80,
                color=color_map.get(str(pooling_mode), "#444444"),
                label=str(pooling_mode) if str(pooling_mode) not in plotted_labels else None,
            )
            single_ax.scatter(
                valid["x"],
                valid["y"],
                s=28,
                alpha=0.80,
                color=color_map.get(str(pooling_mode), "#444444"),
                label=str(pooling_mode) if str(pooling_mode) not in plotted_labels else None,
            )
            plotted_labels.add(str(pooling_mode))
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
        single_ax.set_title(title)
        single_ax.set_xlabel(x_col)
        single_ax.set_ylabel(y_col)
        single_ax.grid(True, alpha=0.25)
        single_handles, single_labels = single_ax.get_legend_handles_labels()
        if single_handles:
            single_ax.legend(single_handles, single_labels, title="pooling", fontsize=9)
        single_fig.tight_layout()
        single_fig.savefig(single_path, dpi=180, bbox_inches="tight")
        plt.close(single_fig)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, title="pooling", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "ar1_parameter_scatterplots.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild RL=2 report exports from the surviving derived artifact.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-version", default="v1")
    parser.add_argument("--artifact-tag", default="rl2_masked_v4")
    parser.add_argument("--derived-id", default="dd9d4bd38203ff8e")
    parser.add_argument("--benchmark-version", default="v2")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cache_root = project_root / "cache" / "computedraft" / str(args.cache_version) / str(args.artifact_tag)
    derived_artifact_path = cache_root / "derived" / str(args.derived_id) / "artifact.pkl"
    export_root = cache_root / "report_exports"
    plot_root = export_root / "plots"
    export_root.mkdir(parents=True, exist_ok=True)

    with derived_artifact_path.open("rb") as handle:
        derived_artifact = pickle.load(handle)

    masked_fits = dict(derived_artifact.get("masked_fits", {}))
    parameter_summary = build_parameter_summary(masked_fits)
    benchmark_summary, model_comparison, participant_metric_summary, heldout_detail_summary = build_summary_frames(
        masked_fits,
        parameter_summary,
        benchmark_version=str(args.benchmark_version),
    )

    resolved_config_path = export_root / "resolved_config.json"
    resolved_config: dict[str, object] = {}
    if resolved_config_path.exists():
        resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    resolved_config.update(
        {
            "cache_version": str(args.cache_version),
            "artifact_tag": str(args.artifact_tag),
            "benchmark_version": str(args.benchmark_version),
            "report_export_source": "derived_masked_fit_recovery",
            "subtotal_metric_variant": "participant_mean_daily_rmse_sqrtm",
            "derived_artifact_id": str(args.derived_id),
        }
    )
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True), encoding="utf-8")

    benchmark_summary.to_csv(export_root / "benchmark_summary.csv", index=False)
    model_comparison.to_csv(export_root / "model_comparison.csv", index=False)
    parameter_summary.to_csv(export_root / "parameter_summary.csv", index=False)

    timing_summary = derived_artifact.get("timing_summary", pd.DataFrame())
    if not isinstance(timing_summary, pd.DataFrame):
        timing_summary = pd.DataFrame(timing_summary)
    timing_summary.to_csv(export_root / "timing_summary.csv", index=False)

    build_task_manifest(benchmark_summary).to_csv(export_root / "task_manifest.csv", index=False)

    cohort_summary_path = export_root / "cohort_summary.json"
    cohort_summary: dict[str, object] = {}
    if cohort_summary_path.exists():
        cohort_summary = json.loads(cohort_summary_path.read_text(encoding="utf-8"))
    cohort_summary.update(
        {
            "selected_participants": int(len(derived_artifact.get("participant_ids", []))),
            "full_set_participants": int(cohort_summary.get("full_set_participants", len(derived_artifact.get("participant_ids", [])))),
            "small_set_participants": int(cohort_summary.get("small_set_participants", 0)),
        }
    )
    cohort_summary_path.write_text(json.dumps(cohort_summary, indent=2, sort_keys=True), encoding="utf-8")

    participant_metric_summary.to_csv(export_root / "participant_metric_summary.csv", index=False)
    heldout_detail_summary.to_csv(export_root / "heldout_subtotal_detail.csv", index=False)
    write_plot_files(model_comparison, parameter_summary, plot_root)

    print(f"Wrote report exports to {export_root}")
    print(f"benchmark_summary rows: {len(benchmark_summary)}")
    print(f"parameter_summary rows: {len(parameter_summary)}")


if __name__ == "__main__":
    main()
