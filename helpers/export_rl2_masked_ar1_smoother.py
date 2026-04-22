from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDA_ROOT = PROJECT_ROOT.parent / "eda"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EDA_ROOT) not in sys.path:
    sys.path.insert(0, str(EDA_ROOT))

from helpers.computedraft_pipeline import (
    _compute_masked_hour_counts,
    _load_csv_slice,
    _prediction_table_from_frames,
    _reward_table_from_frames,
    build_csv_hourly_model_data,
    make_masked_cohort_frame,
)
from helpers.rebuild_rl2_report_exports_from_derived import (
    summarize_daily_heldout_metrics_by_participant,
    summarize_hourly_metrics_by_participant,
    summarize_reward_metrics_by_participant,
)
from src.ihs_eda.step_pomp import (
    build_heldout_subtotal_benchmark_frame,
    build_masked_hourly_benchmark_frame,
    summarize_heldout_subtotal_rmse_sqrtm,
    summarize_masked_hourly_reconstruction,
)

from helpers.rl2_masked_jax_smoother import run_batched_masked_ar1_smoother


ANALYSIS_START_UTC = pd.Timestamp("2024-07-01T00:00:00+00:00")
ANALYSIS_END_UTC = pd.Timestamp("2024-09-30T23:00:00+00:00")
ASSUMED_TIMEZONE = "America/New_York"
MASK_REGIME = "random_hours"
MISSING_FRACTION = 0.20
MASK_SEED = 531
MASK_REPLICATE_INDEX = 0
MASK_ID = "mask_rep_01"
MODEL_FAMILY = "ar1"
POOLING_MODES = ["unit_specific", "global_shared_k", "partial_pooled_k"]


def _load_selected_participant_ids(cohort_summary_path: Path) -> list[str]:
    payload = json.loads(cohort_summary_path.read_text(encoding="utf-8"))
    values = payload.get("selected_participant_ids", [])
    return [str(value) for value in values]


def _build_data_by_participant(masked_cohort_frame: pd.DataFrame, participant_ids: list[str]) -> dict[str, object]:
    data_by_participant: dict[str, object] = {}
    for participant_id in participant_ids:
        participant_frame = masked_cohort_frame.loc[
            masked_cohort_frame["PARTICIPANTIDENTIFIER"].astype(str) == str(participant_id)
        ].copy()
        data_by_participant[str(participant_id)] = build_csv_hourly_model_data(
            participant_frame=participant_frame,
            participant_id=str(participant_id),
            start_utc=ANALYSIS_START_UTC,
            end_utc=ANALYSIS_END_UTC,
            assumed_timezone=ASSUMED_TIMEZONE,
            covariate_mode="none",
            architecture_key="ar1_empirical",
        )
    return data_by_participant


def _build_smoothed_hourly_frame(
    hourly_frame: pd.DataFrame,
    *,
    smoothed_x_mean: np.ndarray,
    smoothed_x_q10: np.ndarray,
    smoothed_x_q90: np.ndarray,
    smoothed_n_mean: np.ndarray,
    smoothed_n_q10: np.ndarray,
    smoothed_n_q90: np.ndarray,
) -> pd.DataFrame:
    frame = hourly_frame.copy().reset_index(drop=True)
    frame["x_backward_smoothed_mean"] = smoothed_x_mean
    frame["x_backward_smoothed_q10"] = smoothed_x_q10
    frame["x_backward_smoothed_q90"] = smoothed_x_q90
    frame["latent_fitbit_scale_steps_backward"] = smoothed_n_mean
    frame["latent_fitbit_scale_steps_backward_q10"] = smoothed_n_q10
    frame["latent_fitbit_scale_steps_backward_q90"] = smoothed_n_q90
    frame["fitbit_steps_obs_fitted_backward"] = smoothed_n_mean
    return frame


def _build_smoothed_reward_frame(data, smoothed_hourly_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, reward_row in data.decision_window_frame.iterrows():
        start_pos = int(reward_row["window_start_position"])
        end_pos = int(reward_row["window_end_position"])
        window = smoothed_hourly_frame.iloc[start_pos : end_pos + 1].copy()
        rows.append(
            {
                **reward_row.to_dict(),
                "latent_reward_24h": float(
                    pd.to_numeric(window["latent_fitbit_scale_steps_backward"], errors="coerce").sum(min_count=1)
                ),
                "fitbit_observed_reward_24h": float(
                    pd.to_numeric(window.get("fitbit_steps_obs"), errors="coerce").sum(min_count=1)
                ),
                "fitbit_fitted_reward_24h": float(
                    pd.to_numeric(window.get("fitbit_steps_obs_fitted_backward"), errors="coerce").sum(min_count=1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_smoothed_metrics(
    *,
    pooling_mode: str,
    hourly_frame: pd.DataFrame,
    reward_frame: pd.DataFrame,
    avg_loglik: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly_summary = summarize_masked_hourly_reconstruction(
        hourly_frame,
        truth_column="fitbit_truth_for_eval",
        estimate_column="latent_fitbit_scale_steps_backward",
        mask_flag_column="fitbit_masked_for_eval",
        positive_truth_only=True,
    )
    subtotal_summary = summarize_heldout_subtotal_rmse_sqrtm(
        hourly_frame,
        reward_frame,
        truth_column="fitbit_truth_for_eval",
        estimate_column="latent_fitbit_scale_steps_backward",
        mask_flag_column="fitbit_masked_for_eval",
    )
    participant_hourly = summarize_hourly_metrics_by_participant(
        hourly_frame,
        estimate_column="latent_fitbit_scale_steps_backward",
    )
    participant_daily, daily_detail = summarize_daily_heldout_metrics_by_participant(
        hourly_frame,
        reward_frame,
        estimate_column="latent_fitbit_scale_steps_backward",
    )
    participant_reward = summarize_reward_metrics_by_participant(reward_frame)
    participant_metrics = participant_hourly.merge(participant_reward, how="outer", on="participant_id").merge(
        participant_daily,
        how="outer",
        on="participant_id",
    )
    participant_metrics["model_family"] = MODEL_FAMILY
    participant_metrics["pooling_mode"] = str(pooling_mode)
    participant_metrics["mask_id"] = MASK_ID
    participant_metrics["estimate_type"] = "smoothed"

    daily_correlation_avg = float(
        pd.to_numeric(participant_metrics["heldout_daily_correlation"], errors="coerce").dropna().mean()
    )
    hourly_correlation_avg = float(
        pd.to_numeric(participant_metrics["hourly_correlation"], errors="coerce").dropna().mean()
    )
    hourly_rmse_avg = float(
        pd.to_numeric(participant_metrics["hourly_rmse"], errors="coerce").dropna().mean()
    )
    subtotal_rmse_avg = float(
        pd.to_numeric(participant_metrics["daily_rmse_sqrtm"], errors="coerce").dropna().mean()
    )

    summary_row = pd.DataFrame(
        [
            {
                "model_family": MODEL_FAMILY,
                "pooling_mode": str(pooling_mode),
                "mask_id": MASK_ID,
                "estimate_type": "smoothed",
                "avg_log_likelihood": float(avg_loglik),
                "hourly_rmse": float(pd.to_numeric(hourly_summary["rmse"], errors="coerce").iloc[0]),
                "hourly_rmse_participant_mean": hourly_rmse_avg,
                "avg_hourly_correlation": hourly_correlation_avg,
                "avg_daily_correlation": daily_correlation_avg,
                "rmse_subtotal_participant_mean": subtotal_rmse_avg,
                "rmse_subtotal_pooled_windows": float(pd.to_numeric(subtotal_summary["rmse_sqrtm"], errors="coerce").iloc[0]),
                "mean_signed_subtotal_error": float(
                    pd.to_numeric(subtotal_summary["mean_signed_subtotal_error"], errors="coerce").iloc[0]
                ),
                "mean_signed_normalized_subtotal_error": float(
                    pd.to_numeric(subtotal_summary["mean_signed_normalized_subtotal_error"], errors="coerce").iloc[0]
                ),
                "n_participants_used": int(
                    pd.to_numeric(participant_metrics["participant_id"].notna(), errors="coerce").fillna(0).sum()
                ),
                "n_windows_used": int(
                    pd.to_numeric(subtotal_summary["n_windows_used"], errors="coerce").fillna(0).iloc[0]
                ),
            }
        ]
    )
    daily_detail["model_family"] = MODEL_FAMILY
    daily_detail["pooling_mode"] = str(pooling_mode)
    daily_detail["mask_id"] = MASK_ID
    daily_detail["estimate_type"] = "smoothed"
    return summary_row, participant_metrics, daily_detail


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RL=2 masked AR(1) smoother results.")
    parser.add_argument("--artifact-tag", default="rl2_masked_v4")
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=13)
    parser.add_argument("--seed", type=int, default=40531)
    parser.add_argument("--export-folder", default="smoother_report_exports")
    args = parser.parse_args()

    cache_root = PROJECT_ROOT / "cache" / "computedraft" / "v1" / str(args.artifact_tag)
    report_exports_root = cache_root / "report_exports"
    export_root = cache_root / str(args.export_folder) / "jax_masked_ar1_v1"
    export_root.mkdir(parents=True, exist_ok=True)

    participant_ids = _load_selected_participant_ids(report_exports_root / "cohort_summary.json")
    parameter_summary = pd.read_csv(report_exports_root / "parameter_summary.csv")
    benchmark_summary = pd.read_csv(report_exports_root / "benchmark_summary.csv")

    load_start = perf_counter()
    cohort_frame = _load_csv_slice(
        PROJECT_ROOT / "FitbitHourly.csv",
        participant_ids=participant_ids,
        start_utc=ANALYSIS_START_UTC,
        end_utc=ANALYSIS_END_UTC,
    )
    masked_cohort_frame, mask_summary = make_masked_cohort_frame(
        cohort_frame,
        participant_ids=participant_ids,
        mask_regime=MASK_REGIME,
        missing_fraction=MISSING_FRACTION,
        mask_seed=MASK_SEED,
        replicate_index=MASK_REPLICATE_INDEX,
    )
    data_by_participant = _build_data_by_participant(masked_cohort_frame, participant_ids)
    load_seconds = float(perf_counter() - load_start)

    summary_frames: list[pd.DataFrame] = []
    participant_metric_frames: list[pd.DataFrame] = []
    daily_detail_frames: list[pd.DataFrame] = []
    hourly_frames: list[pd.DataFrame] = []
    reward_frames: list[pd.DataFrame] = []
    smoother_status_rows: list[dict[str, object]] = []

    for mode_index, pooling_mode in enumerate(POOLING_MODES):
        mode_params = parameter_summary.loc[
            (parameter_summary["model_family"].astype(str) == MODEL_FAMILY)
            & (parameter_summary["pooling_mode"].astype(str) == str(pooling_mode))
        ].copy()
        mode_params = mode_params.sort_values("participantidentifier").reset_index(drop=True)
        avg_loglik = float(pd.to_numeric(mode_params["panel_unit_loglik"], errors="coerce").mean())

        mode_hourly_frames: list[pd.DataFrame] = []
        mode_reward_frames: list[pd.DataFrame] = []
        mode_timing_rows: list[dict[str, object]] = []

        for chunk_start in range(0, len(participant_ids), int(args.chunk_size)):
            chunk_ids = participant_ids[chunk_start : chunk_start + int(args.chunk_size)]
            chunk_params = (
                mode_params.loc[mode_params["participantidentifier"].astype(str).isin(chunk_ids)]
                .sort_values("participantidentifier")
                .reset_index(drop=True)
            )
            chunk_hourly_frames = [data_by_participant[pid].hourly_frame for pid in chunk_ids]
            chunk_phi = pd.to_numeric(chunk_params["phi"], errors="coerce").to_numpy(dtype=np.float32)
            chunk_sigma = pd.to_numeric(chunk_params["sigma"], errors="coerce").to_numpy(dtype=np.float32)
            chunk_k = pd.to_numeric(
                chunk_params["k_fitbit"].fillna(np.exp(pd.to_numeric(chunk_params["log_k_fitbit"], errors="coerce"))),
                errors="coerce",
            ).to_numpy(dtype=np.float32)
            chunk_seed = int(args.seed) + 10_000 * mode_index + chunk_start
            run_start = perf_counter()
            chunk_result = run_batched_masked_ar1_smoother(
                participant_ids=chunk_ids,
                hourly_frames=chunk_hourly_frames,
                phi=chunk_phi,
                sigma=chunk_sigma,
                k_fitbit=chunk_k,
                n_particles=int(args.particles),
                random_seed=chunk_seed,
            )
            chunk_seconds = float(perf_counter() - run_start)

            for position, participant_id in enumerate(chunk_ids):
                data = data_by_participant[participant_id]
                smoothed_hourly = _build_smoothed_hourly_frame(
                    data.hourly_frame,
                    smoothed_x_mean=chunk_result.smoothed_x_mean[position],
                    smoothed_x_q10=chunk_result.smoothed_x_q10[position],
                    smoothed_x_q90=chunk_result.smoothed_x_q90[position],
                    smoothed_n_mean=chunk_result.smoothed_n_mean[position],
                    smoothed_n_q10=chunk_result.smoothed_n_q10[position],
                    smoothed_n_q90=chunk_result.smoothed_n_q90[position],
                )
                smoothed_reward = _build_smoothed_reward_frame(data, smoothed_hourly)
                reward_counts = _compute_masked_hour_counts(smoothed_hourly, smoothed_reward)
                reward_table = _reward_table_from_frames(
                    smoothed_reward,
                    participant_id=str(participant_id),
                    model_family=MODEL_FAMILY,
                    pooling_mode=str(pooling_mode),
                    mask_id=MASK_ID,
                    predicted_latent_reward_column="latent_reward_24h",
                ).merge(reward_counts, how="left", on="decision_id")
                reward_table["artificially_masked_hours"] = pd.to_numeric(
                    reward_table.get("artificially_masked_hours_y"),
                    errors="coerce",
                ).fillna(
                    pd.to_numeric(reward_table.get("artificially_masked_hours_x"), errors="coerce")
                ).fillna(0).astype(int)
                hourly_table = _prediction_table_from_frames(
                    smoothed_hourly,
                    smoothed_reward,
                    participant_id=str(participant_id),
                    model_family=MODEL_FAMILY,
                    pooling_mode=str(pooling_mode),
                    covariate_mode="none",
                    mask_id=MASK_ID,
                    predicted_observed_column="fitbit_steps_obs_fitted_backward",
                    predicted_latent_column="latent_fitbit_scale_steps_backward",
                )
                mode_hourly_frames.append(hourly_table)
                mode_reward_frames.append(reward_table)
                mode_timing_rows.append(
                    {
                        "participant_id": str(participant_id),
                        "model_family": MODEL_FAMILY,
                        "pooling_mode": str(pooling_mode),
                        "mask_id": MASK_ID,
                        "particles": int(args.particles),
                        "chunk_seed": chunk_seed,
                        "chunk_runtime_seconds": chunk_seconds,
                        "participant_loglik_estimate": float(chunk_result.loglik_by_participant[position]),
                        "mean_ess": float(chunk_result.mean_ess_by_participant[position]),
                        "min_ess": float(chunk_result.min_ess_by_participant[position]),
                    }
                )

        mode_hourly = pd.concat(mode_hourly_frames, ignore_index=True)
        mode_reward = pd.concat(mode_reward_frames, ignore_index=True)
        mode_summary, mode_participant_metrics, mode_daily_detail = _summarize_smoothed_metrics(
            pooling_mode=str(pooling_mode),
            hourly_frame=mode_hourly,
            reward_frame=mode_reward,
            avg_loglik=avg_loglik,
        )
        mode_summary["particles"] = int(args.particles)
        mode_summary["smoother_method"] = "jax_batched_backward_marginal"
        mode_summary["load_reconstruction_seconds"] = load_seconds

        mode_root = export_root / str(pooling_mode)
        mode_root.mkdir(parents=True, exist_ok=True)
        mode_summary.to_csv(mode_root / "summary.csv", index=False)
        mode_participant_metrics.to_csv(mode_root / "participant_metrics.csv", index=False)
        mode_daily_detail.to_csv(mode_root / "heldout_subtotal_detail.csv", index=False)
        mode_hourly.to_parquet(mode_root / "hourly_smoothed.parquet", index=False)
        mode_reward.to_parquet(mode_root / "reward_smoothed.parquet", index=False)
        pd.DataFrame(mode_timing_rows).to_csv(mode_root / "smoother_diagnostics.csv", index=False)

        filter_row = benchmark_summary.loc[
            (benchmark_summary["model_family"].astype(str) == MODEL_FAMILY)
            & (benchmark_summary["pooling_mode"].astype(str) == str(pooling_mode))
            & (benchmark_summary["estimate_type"].astype(str) == "filter")
        ].copy()
        if not filter_row.empty:
            filter_row = filter_row.rename(
                columns={
                    "hourly_masked_rmse": "hourly_rmse",
                    "hourly_masked_correlation": "avg_hourly_correlation",
                    "daily_correlation": "avg_daily_correlation",
                    "heldout_subtotal_rmse_sqrtm": "rmse_subtotal_participant_mean",
                    "mean_panel_unit_loglik": "avg_log_likelihood",
                }
            )
            filter_row["estimate_type"] = "filter_cached"
            filter_row = filter_row[
                [
                    "model_family",
                    "pooling_mode",
                    "mask_id",
                    "estimate_type",
                    "avg_log_likelihood",
                    "hourly_rmse",
                    "avg_hourly_correlation",
                    "avg_daily_correlation",
                    "rmse_subtotal_participant_mean",
                ]
            ]
            summary_frames.append(filter_row)

        summary_frames.append(mode_summary)
        participant_metric_frames.append(mode_participant_metrics)
        daily_detail_frames.append(mode_daily_detail)
        hourly_frames.append(mode_hourly)
        reward_frames.append(mode_reward)
        smoother_status_rows.append(
            {
                "model_family": MODEL_FAMILY,
                "pooling_mode": str(pooling_mode),
                "mask_id": MASK_ID,
                "status": "completed",
                "particles": int(args.particles),
                "n_participants": int(mode_participant_metrics["participant_id"].nunique()),
            }
        )

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_participant_metrics = pd.concat(participant_metric_frames, ignore_index=True)
    combined_daily_detail = pd.concat(daily_detail_frames, ignore_index=True)
    combined_hourly = pd.concat(hourly_frames, ignore_index=True)
    combined_reward = pd.concat(reward_frames, ignore_index=True)
    combined_summary.to_csv(export_root / "combined_summary.csv", index=False)
    combined_participant_metrics.to_csv(export_root / "combined_participant_metrics.csv", index=False)
    combined_daily_detail.to_csv(export_root / "combined_heldout_subtotal_detail.csv", index=False)
    combined_hourly.to_parquet(export_root / "combined_hourly_smoothed.parquet", index=False)
    combined_reward.to_parquet(export_root / "combined_reward_smoothed.parquet", index=False)
    mask_summary.to_csv(export_root / "mask_summary.csv", index=False)
    pd.DataFrame(smoother_status_rows).to_csv(export_root / "run_status.csv", index=False)

    _write_json(
        export_root / "run_manifest.json",
        {
            "artifact_tag": str(args.artifact_tag),
            "model_family": MODEL_FAMILY,
            "pooling_modes": POOLING_MODES,
            "mask_id": MASK_ID,
            "analysis_start_utc": ANALYSIS_START_UTC.isoformat(),
            "analysis_end_utc": ANALYSIS_END_UTC.isoformat(),
            "assumed_timezone": ASSUMED_TIMEZONE,
            "mask_regime": MASK_REGIME,
            "missing_fraction": MISSING_FRACTION,
            "mask_seed": MASK_SEED,
            "particles": int(args.particles),
            "chunk_size": int(args.chunk_size),
            "seed": int(args.seed),
            "n_participants": len(participant_ids),
            "parameter_source": str(report_exports_root / "parameter_summary.csv"),
            "benchmark_source": str(report_exports_root / "benchmark_summary.csv"),
            "cohort_source": str(report_exports_root / "cohort_summary.json"),
        },
    )


if __name__ == "__main__":
    main()
