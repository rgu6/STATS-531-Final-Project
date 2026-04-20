from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import jax
import numpy as np
import pandas as pd

from helpers.computedraft_pipeline import (
    BENCHMARK_SPECS,
    POMP_FAMILY_TO_ARCHITECTURE,
    benchmark_from_fit_artifact,
    build_csv_hourly_model_data,
    ensure_base_artifact,
    make_masked_cohort_frame,
    run_arma_model,
    run_panel_pomp_model,
)
from src.ihs_eda.step_pomp import GlobalSearchStageConfig, MODEL_FAMILY_LABELS, POOLING_MODE_LABELS
from src.ihs_eda.step_pomp.time_estimator import (
    ALL_MODEL_FAMILIES,
    POOLING_MODE_OPTIONS,
    TIME_ESTIMATOR_VERSION,
    RuntimeEstimateRequest,
    benchmark_root_from_project,
    build_runtime_calibration_frame,
    expand_model_pooling_pairs,
    flatten_stage_config_features,
    runtime_request_from_dict,
)


@dataclass(frozen=True, slots=True)
class TimingScenario:
    scenario_id: str
    stage_profile: str
    model_family: str
    pooling_mode: str
    fit_mode: str
    participant_count: int
    panel_batch_size: int
    missing_fraction: float
    mask_replicate_index: int = 0
    request_source: str | None = None


def default_stage_profiles() -> dict[str, list[GlobalSearchStageConfig]]:
    return {
        "tiny": [
            GlobalSearchStageConfig(
                name="Stage 1",
                n_starts=1,
                keep_top=1,
                particles=20,
                mif_iterations=1,
                cooling_fraction=0.50,
                rw_sd_scale=0.02,
                evaluation_particles=20,
                evaluation_pfilter_reps=1,
                n_fresh_random_starts=0,
                n_jittered_restarts=0,
            ),
            GlobalSearchStageConfig(
                name="Stage 2",
                n_starts=1,
                keep_top=1,
                particles=20,
                mif_iterations=1,
                cooling_fraction=0.50,
                rw_sd_scale=0.02,
                evaluation_particles=20,
                evaluation_pfilter_reps=1,
                n_fresh_random_starts=0,
                n_jittered_restarts=0,
            ),
        ],
        "moderate": [
            GlobalSearchStageConfig(
                name="Stage 1",
                n_starts=2,
                keep_top=1,
                particles=40,
                mif_iterations=2,
                cooling_fraction=0.50,
                rw_sd_scale=0.02,
                evaluation_particles=60,
                evaluation_pfilter_reps=1,
                n_fresh_random_starts=1,
                n_jittered_restarts=0,
            ),
            GlobalSearchStageConfig(
                name="Stage 2",
                n_starts=1,
                keep_top=1,
                particles=80,
                mif_iterations=3,
                cooling_fraction=0.50,
                rw_sd_scale=0.02,
                evaluation_particles=100,
                evaluation_pfilter_reps=1,
                n_fresh_random_starts=0,
                n_jittered_restarts=0,
            ),
        ],
    }


def build_benchmark_scenarios(scenario_set: str, *, missing_fraction: float) -> list[TimingScenario]:
    missing = float(missing_fraction)
    scenarios: list[TimingScenario] = []

    def add(
        scenario_id: str,
        *,
        stage_profile: str,
        model_family: str,
        pooling_mode: str,
        fit_mode: str,
        participant_count: int,
        panel_batch_size: int,
    ) -> None:
        scenarios.append(
            TimingScenario(
                scenario_id=str(scenario_id),
                stage_profile=str(stage_profile),
                model_family=str(model_family),
                pooling_mode=str(pooling_mode),
                fit_mode=str(fit_mode),
                participant_count=int(participant_count),
                panel_batch_size=int(panel_batch_size),
                missing_fraction=missing,
            )
        )

    mode = str(scenario_set).strip().lower()

    # Core batching/scaling probes.
    for stage_profile in ["tiny", "moderate"]:
        for fit_mode in ["unmasked", "masked"]:
            add(
                f"ar1_unit_{stage_profile}_p1_b1_{fit_mode}",
                stage_profile=stage_profile,
                model_family="ar1",
                pooling_mode="unit_specific",
                fit_mode=fit_mode,
                participant_count=1,
                panel_batch_size=1,
            )
        for participant_count, batch_size in [(4, 1), (4, 4)]:
            add(
                f"ar1_unit_{stage_profile}_p{participant_count}_b{batch_size}_unmasked",
                stage_profile=stage_profile,
                model_family="ar1",
                pooling_mode="unit_specific",
                fit_mode="unmasked",
                participant_count=participant_count,
                panel_batch_size=batch_size,
            )

    # Pooling-mode probes on a common light AR(1) baseline.
    for pooling_mode in POOLING_MODE_OPTIONS:
        for fit_mode in ["unmasked", "masked"]:
            add(
                f"ar1_{pooling_mode}_tiny_p1_b1_{fit_mode}",
                stage_profile="tiny",
                model_family="ar1",
                pooling_mode=pooling_mode,
                fit_mode=fit_mode,
                participant_count=1,
                panel_batch_size=1,
            )

    # Additional family probes.
    family_probe_models = [
        "ar2",
        "wear_observation",
        "wear_latent",
        "ar2_wear_observation",
        "ar2_wear_latent",
    ]
    for model_family in family_probe_models:
        add(
            f"{model_family}_unit_tiny_p1_b1_unmasked",
            stage_profile="tiny",
            model_family=model_family,
            pooling_mode="unit_specific",
            fit_mode="unmasked",
            participant_count=1,
            panel_batch_size=1,
        )
    for model_family in ["ar2", "wear_latent"]:
        add(
            f"{model_family}_unit_tiny_p1_b1_masked",
            stage_profile="tiny",
            model_family=model_family,
            pooling_mode="unit_specific",
            fit_mode="masked",
            participant_count=1,
            panel_batch_size=1,
        )

    # ARMA timing probes.
    for participant_count in [1, 4]:
        for fit_mode in ["unmasked", "masked"]:
            add(
                f"arma_tiny_p{participant_count}_{fit_mode}",
                stage_profile="tiny",
                model_family="arma_aic",
                pooling_mode="not_applicable",
                fit_mode=fit_mode,
                participant_count=participant_count,
                panel_batch_size=1,
            )

    if mode == "extended":
        for model_family in ["ar2", "wear_observation", "wear_latent", "ar2_wear_observation", "ar2_wear_latent"]:
            for pooling_mode in ["global_shared_k", "partial_pooled_k"]:
                add(
                    f"{model_family}_{pooling_mode}_tiny_p1_b1_unmasked",
                    stage_profile="tiny",
                    model_family=model_family,
                    pooling_mode=pooling_mode,
                    fit_mode="unmasked",
                    participant_count=1,
                    panel_batch_size=1,
                )
        for fit_mode in ["unmasked", "masked"]:
            add(
                f"ar2_unit_moderate_p4_b1_{fit_mode}",
                stage_profile="moderate",
                model_family="ar2",
                pooling_mode="unit_specific",
                fit_mode=fit_mode,
                participant_count=4,
                panel_batch_size=1,
            )
            add(
                f"ar2_unit_moderate_p4_b4_{fit_mode}",
                stage_profile="moderate",
                model_family="ar2",
                pooling_mode="unit_specific",
                fit_mode=fit_mode,
                participant_count=4,
                panel_batch_size=4,
            )
    elif mode != "quick":
        raise ValueError(f"Unknown scenario_set: {scenario_set}")

    return scenarios


def _fit_modes_for_scope(fit_scope: str) -> list[str]:
    scope = str(fit_scope).strip().lower()
    fit_modes: list[str] = []
    if scope in {"unmasked_only", "both"}:
        fit_modes.append("unmasked")
    if scope in {"masked_only", "both"}:
        fit_modes.append("masked")
    return fit_modes


def build_request_validation_scenarios(
    request: RuntimeEstimateRequest,
    *,
    request_hash: str,
) -> tuple[list[TimingScenario], dict[str, list[GlobalSearchStageConfig]]]:
    stage_profile_key = f"imported::{request_hash}"
    scenarios: list[TimingScenario] = []
    for model_family, pooling_mode in expand_model_pooling_pairs(request.model_families, request.pooling_modes):
        for fit_mode in _fit_modes_for_scope(request.fit_scope):
            replicate_count = int(request.masking_replicates) if fit_mode == "masked" else 1
            for replicate_index in range(int(max(replicate_count, 1))):
                scenario_id = (
                    f"imported_{request_hash}_{model_family}_{pooling_mode}_{fit_mode}"
                    + (f"_rep_{replicate_index + 1:02d}" if fit_mode == "masked" else "")
                )
                scenarios.append(
                    TimingScenario(
                        scenario_id=scenario_id,
                        stage_profile=stage_profile_key,
                        model_family=str(model_family),
                        pooling_mode=str(pooling_mode),
                        fit_mode=str(fit_mode),
                        participant_count=int(request.participant_count),
                        panel_batch_size=int(request.participants_per_batch if model_family != "arma_aic" else 1),
                        missing_fraction=float(request.missing_fraction),
                        mask_replicate_index=int(replicate_index),
                        request_source=str(request_hash),
                    )
                )
    return scenarios, {stage_profile_key: list(request.stage_configs)}


def _resolved_base_config(
    *,
    cache_version: str,
    artifact_tag: str,
    analysis_start_utc_iso: str,
    analysis_end_utc_iso: str,
    assumed_local_timezone: str,
) -> dict[str, object]:
    return {
        "cache_version": str(cache_version),
        "artifact_tag": str(artifact_tag),
        "participant_set": "full_set",
        "analysis_start_utc_iso": str(analysis_start_utc_iso),
        "analysis_end_utc_iso": str(analysis_end_utc_iso),
        "assumed_local_timezone": str(assumed_local_timezone),
    }


def _chunk_participants(participant_ids: list[str], batch_size: int) -> list[list[str]]:
    size = max(int(batch_size), 1)
    return [participant_ids[index : index + size] for index in range(0, len(participant_ids), size)]


def _combine_fit_artifacts(artifacts: list[dict[str, object]]) -> dict[str, object]:
    if not artifacts:
        return {
            "participant_estimates": pd.DataFrame(),
            "estimate_summary": pd.DataFrame(),
            "shared_estimates": {},
            "candidate_frame": pd.DataFrame(),
            "stage_summary": pd.DataFrame(),
            "best_loglik": np.nan,
            "hourly_prediction_frame": pd.DataFrame(),
            "reward_window_frame": pd.DataFrame(),
            "timing_frame": pd.DataFrame(),
        }
    participant_estimates = pd.concat(
        [artifact.get("participant_estimates", pd.DataFrame()) for artifact in artifacts if artifact.get("participant_estimates") is not None],
        ignore_index=True,
    )
    candidate_frame = pd.concat(
        [artifact.get("candidate_frame", pd.DataFrame()) for artifact in artifacts if artifact.get("candidate_frame") is not None],
        ignore_index=True,
    )
    stage_summary = pd.concat(
        [artifact.get("stage_summary", pd.DataFrame()) for artifact in artifacts if artifact.get("stage_summary") is not None],
        ignore_index=True,
    )
    hourly_prediction_frame = pd.concat(
        [artifact.get("hourly_prediction_frame", pd.DataFrame()) for artifact in artifacts if artifact.get("hourly_prediction_frame") is not None],
        ignore_index=True,
    )
    reward_window_frame = pd.concat(
        [artifact.get("reward_window_frame", pd.DataFrame()) for artifact in artifacts if artifact.get("reward_window_frame") is not None],
        ignore_index=True,
    )
    timing_frame = pd.concat(
        [artifact.get("timing_frame", pd.DataFrame()) for artifact in artifacts if artifact.get("timing_frame") is not None],
        ignore_index=True,
    )

    if participant_estimates.empty:
        estimate_summary = pd.DataFrame()
    else:
        numeric_cols = [
            column
            for column in participant_estimates.columns
            if column not in {"participantidentifier", "k_fitbit_pooling_mode"}
            and pd.api.types.is_numeric_dtype(pd.to_numeric(participant_estimates[column], errors="coerce"))
        ]
        if numeric_cols:
            estimate_summary = (
                participant_estimates[numeric_cols]
                .agg(["mean", "std", "median", "min", "max"])
                .transpose()
                .reset_index()
                .rename(columns={"index": "parameter"})
            )
        else:
            estimate_summary = pd.DataFrame()
    return {
        "participant_estimates": participant_estimates,
        "estimate_summary": estimate_summary,
        "shared_estimates": {},
        "candidate_frame": candidate_frame,
        "stage_summary": stage_summary,
        "best_loglik": float(pd.to_numeric(participant_estimates.get("panel_unit_loglik"), errors="coerce").sum())
        if not participant_estimates.empty and "panel_unit_loglik" in participant_estimates.columns
        else np.nan,
        "hourly_prediction_frame": hourly_prediction_frame,
        "reward_window_frame": reward_window_frame,
        "timing_frame": timing_frame,
    }


def _timing_step_sum(timing_frame: pd.DataFrame, step_name: str) -> float:
    if timing_frame.empty or "step" not in timing_frame.columns:
        return 0.0
    return float(
        pd.to_numeric(
            timing_frame.loc[timing_frame["step"].astype(str) == str(step_name), "seconds"],
            errors="coerce",
        ).sum()
    )


def _fit_timing_components(timing_frame: pd.DataFrame) -> dict[str, float]:
    return {
        "panel_global_search_seconds": _timing_step_sum(timing_frame, "panel_global_search"),
        "postfit_filter_total_seconds": _timing_step_sum(timing_frame, "panel_postfit_filter"),
        "postfit_fit_seconds": _timing_step_sum(timing_frame, "panel_postfit_fit_total"),
        "postfit_simulate_seconds": _timing_step_sum(timing_frame, "panel_postfit_simulate"),
        "postfit_postprocess_seconds": _timing_step_sum(timing_frame, "panel_postfit_postprocess"),
        "postfit_backward_smoother_seconds": _timing_step_sum(timing_frame, "panel_postfit_backward_smoother"),
        "postfit_run_total_seconds": _timing_step_sum(timing_frame, "panel_postfit_run_total"),
        "postfit_host_overhead_seconds": _timing_step_sum(timing_frame, "panel_postfit_host_overhead"),
    }


def _run_scenario_batches(
    *,
    cohort_frame: pd.DataFrame,
    selected_participants: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    assumed_timezone: str,
    model_family: str,
    pooling_mode: str,
    stage_configs: list[GlobalSearchStageConfig],
    search_windows: dict[str, tuple[float, float]],
    global_seed: int,
    panel_batch_size: int,
    covariate_mode: str,
    mask_id: str | None,
    warm_start_estimates: pd.DataFrame | None,
    inference_only: bool,
) -> dict[str, object]:
    data_prep_seconds = 0.0
    fit_total_seconds = 0.0
    panel_global_search_seconds = 0.0
    postfit_filter_total_seconds = 0.0
    postfit_fit_seconds = 0.0
    postfit_simulate_seconds = 0.0
    postfit_postprocess_seconds = 0.0
    postfit_backward_smoother_seconds = 0.0
    postfit_run_total_seconds = 0.0
    postfit_host_overhead_seconds = 0.0
    artifacts: list[dict[str, object]] = []
    batch_sizes: list[int] = []
    participant_hours: list[int] = []

    for batch_index, participant_chunk in enumerate(_chunk_participants(selected_participants, panel_batch_size), start=1):
        batch_sizes.append(int(len(participant_chunk)))
        data_map: dict[str, object] = {}
        data_prep_start = perf_counter()
        architecture_key = POMP_FAMILY_TO_ARCHITECTURE.get(model_family, "ar1_empirical")
        for participant_id in participant_chunk:
            participant_frame = cohort_frame.loc[
                cohort_frame["PARTICIPANTIDENTIFIER"].astype(str) == str(participant_id)
            ].copy()
            model_data = build_csv_hourly_model_data(
                participant_frame,
                participant_id=str(participant_id),
                start_utc=start_utc,
                end_utc=end_utc,
                assumed_timezone=assumed_timezone,
                covariate_mode=str(covariate_mode),
                architecture_key=str(architecture_key),
            )
            data_map[str(participant_id)] = model_data
            participant_hours.append(int(len(model_data.hourly_frame)))
        data_prep_seconds += float(perf_counter() - data_prep_start)

        batch_warm_start = None
        if warm_start_estimates is not None and not warm_start_estimates.empty:
            batch_warm_start = warm_start_estimates.loc[
                warm_start_estimates["participantidentifier"].astype(str).isin([str(pid) for pid in participant_chunk])
            ].copy()

        fit_start = perf_counter()
        if model_family == "arma_aic":
            artifact = run_arma_model(
                data_map,
                rl=1,
                model_family=str(model_family),
                pooling_mode=str(pooling_mode),
                covariate_mode=str(covariate_mode),
                mask_id=mask_id,
                include_prediction_artifacts=not inference_only,
            )
        else:
            artifact = run_panel_pomp_model(
                data_map,
                architecture_key=str(architecture_key),
                pooling_mode=str(pooling_mode),
                search_windows=search_windows,
                stage_configs=stage_configs,
                global_seed=int(global_seed) + batch_index * 1_000,
                model_family=str(model_family),
                covariate_mode=str(covariate_mode),
                mask_id=mask_id,
                warm_start_estimates=batch_warm_start,
                include_postfit_reconstruction=not inference_only,
            )
        fit_total_seconds += float(perf_counter() - fit_start)
        timing_components = _fit_timing_components(artifact.get("timing_frame", pd.DataFrame()))
        panel_global_search_seconds += float(timing_components["panel_global_search_seconds"])
        postfit_filter_total_seconds += float(timing_components["postfit_filter_total_seconds"])
        postfit_fit_seconds += float(timing_components["postfit_fit_seconds"])
        postfit_simulate_seconds += float(timing_components["postfit_simulate_seconds"])
        postfit_postprocess_seconds += float(timing_components["postfit_postprocess_seconds"])
        postfit_backward_smoother_seconds += float(timing_components["postfit_backward_smoother_seconds"])
        postfit_run_total_seconds += float(timing_components["postfit_run_total_seconds"])
        postfit_host_overhead_seconds += float(timing_components["postfit_host_overhead_seconds"])
        artifacts.append(artifact)

    combined_artifact = _combine_fit_artifacts(artifacts)
    participant_hours_mean = float(np.mean(participant_hours)) if participant_hours else np.nan
    return {
        "artifact": combined_artifact,
        "data_prep_seconds": float(data_prep_seconds),
        "fit_total_seconds": float(fit_total_seconds),
        "panel_global_search_seconds": float(panel_global_search_seconds),
        "postfit_filter_total_seconds": float(postfit_filter_total_seconds),
        "postfit_fit_seconds": float(postfit_fit_seconds),
        "postfit_simulate_seconds": float(postfit_simulate_seconds),
        "postfit_postprocess_seconds": float(postfit_postprocess_seconds),
        "postfit_backward_smoother_seconds": float(postfit_backward_smoother_seconds),
        "postfit_run_total_seconds": float(postfit_run_total_seconds),
        "postfit_host_overhead_seconds": float(postfit_host_overhead_seconds),
        "fit_unattributed_seconds": float(
            max(
                fit_total_seconds - panel_global_search_seconds - postfit_filter_total_seconds,
                0.0,
            )
        ),
        "batch_sizes": batch_sizes,
        "batch_count": int(len(batch_sizes)),
        "participants_in_batch_mean": float(np.mean(batch_sizes)) if batch_sizes else np.nan,
        "participants_in_batch_max": int(max(batch_sizes)) if batch_sizes else 0,
        "hours_per_participant": participant_hours_mean,
    }


def _search_windows() -> dict[str, tuple[float, float]]:
    return {
        "phi": (-0.95, 0.95),
        "phi1": (-1.50, 1.50),
        "phi2": (-0.90, 0.90),
        "sigma": (0.05, 0.80),
        "k_fitbit": (5.0, 300.0),
        "log_k_fitbit": (float(np.log(5.0)), float(np.log(300.0))),
        "mu_log_k_fitbit": (float(np.log(5.0)), float(np.log(300.0))),
        "log_tau_log_k_fitbit": (float(np.log(0.05)), float(np.log(2.0))),
        "eta_wear_lag1": (-2.0, 3.0),
        "wear_phi": (-0.90, 0.95),
        "wear_sigma": (0.05, 0.80),
    }


def _scenario_record(
    *,
    scenario: TimingScenario,
    benchmark_tag: str,
    scenario_set: str,
    device: str,
    jax_backend: str,
    fit_payload: dict[str, object],
    mask_build_seconds: float,
    benchmark_build_seconds: float,
    warm_start_used: bool,
    search_units: dict[str, float],
) -> dict[str, object]:
    batch_sizes = list(fit_payload["batch_sizes"])
    inference_fit_seconds = float(
        max(
            float(fit_payload["fit_total_seconds"]) - float(fit_payload["postfit_filter_total_seconds"]),
            0.0,
        )
    )
    full_wall_seconds = float(
        float(fit_payload["fit_total_seconds"])
        + float(fit_payload["data_prep_seconds"])
        + float(mask_build_seconds)
        + float(benchmark_build_seconds)
    )
    auxiliary_artifact_seconds = float(max(full_wall_seconds - inference_fit_seconds, 0.0))
    return {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "benchmark_version": TIME_ESTIMATOR_VERSION,
        "benchmark_tag": str(benchmark_tag),
        "scenario_set": str(scenario_set),
        "scenario_id": str(scenario.scenario_id),
        "stage_profile": str(scenario.stage_profile),
        "model_family": str(scenario.model_family),
        "pooling_mode": str(scenario.pooling_mode),
        "fit_mode": str(scenario.fit_mode),
        "participant_count": int(scenario.participant_count),
        "panel_batch_size": int(scenario.panel_batch_size),
        "mask_replicate_index": int(scenario.mask_replicate_index),
        "request_source": str(scenario.request_source) if scenario.request_source is not None else None,
        "participants_in_batch": float(fit_payload["participants_in_batch_mean"]),
        "max_batch_size": int(fit_payload["participants_in_batch_max"]),
        "batch_count": int(fit_payload["batch_count"]),
        "batch_sizes_json": json.dumps(batch_sizes),
        "hours_per_participant": float(fit_payload["hours_per_participant"]),
        "device": str(device),
        "jax_backend": str(jax_backend),
        "mask_build_seconds": float(mask_build_seconds),
        "benchmark_build_seconds": float(benchmark_build_seconds),
        "data_prep_seconds": float(fit_payload["data_prep_seconds"]),
        "inference_fit_seconds": inference_fit_seconds,
        "full_wall_seconds": full_wall_seconds,
        "auxiliary_artifact_seconds": auxiliary_artifact_seconds,
        "panel_global_search_seconds": float(fit_payload["panel_global_search_seconds"]),
        "postfit_filter_total_seconds": float(fit_payload["postfit_filter_total_seconds"]),
        "postfit_fit_seconds": float(fit_payload["postfit_fit_seconds"]),
        "postfit_simulate_seconds": float(fit_payload["postfit_simulate_seconds"]),
        "postfit_postprocess_seconds": float(fit_payload["postfit_postprocess_seconds"]),
        "postfit_backward_smoother_seconds": float(fit_payload["postfit_backward_smoother_seconds"]),
        "postfit_run_total_seconds": float(fit_payload["postfit_run_total_seconds"]),
        "postfit_host_overhead_seconds": float(fit_payload["postfit_host_overhead_seconds"]),
        "fit_unattributed_seconds": float(fit_payload["fit_unattributed_seconds"]),
        "fit_total_seconds": float(fit_payload["fit_total_seconds"]),
        "warm_start_used": bool(warm_start_used),
        **search_units,
    }


def run_timing_benchmark_suite(
    csv_path: Path,
    *,
    project_root: Path,
    benchmark_tag: str,
    scenario_set: str,
    device: str,
    cache_version: str = TIME_ESTIMATOR_VERSION,
    analysis_start_utc_iso: str = "2024-07-01T00:00:00+00:00",
    analysis_end_utc_iso: str = "2024-09-30T23:00:00+00:00",
    assumed_local_timezone: str = "America/New_York",
    missing_fraction: float = 0.20,
    mask_seed: int = 531,
    global_seed: int = 531,
    covariate_mode: str = "none",
    force_recompute: bool = False,
    request_payload: dict[str, object] | None = None,
    request_mode: str = "append",
    inference_only: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if progress_callback is None:
        progress_callback = lambda _message: None
    benchmark_root = benchmark_root_from_project(Path(project_root))
    run_timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = benchmark_root / str(cache_version) / str(benchmark_tag) / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = _resolved_base_config(
        cache_version=str(cache_version),
        artifact_tag=f"computetimetest::{benchmark_tag}",
        analysis_start_utc_iso=str(analysis_start_utc_iso),
        analysis_end_utc_iso=str(analysis_end_utc_iso),
        assumed_local_timezone=str(assumed_local_timezone),
    )
    base_artifact, _, _ = ensure_base_artifact(
        Path(csv_path),
        cache_root=Path(project_root) / "cache" / "computedraft",
        resolved_config=base_config,
        force_recompute=bool(force_recompute),
    )
    available_ids = list(base_artifact["full_set_ids"])
    if not available_ids:
        raise ValueError("No eligible participants were available in the CSV-based full_set.")

    cohort_frame = base_artifact["cohort_frame"].copy()
    start_utc = pd.Timestamp(str(analysis_start_utc_iso)).tz_convert("UTC")
    end_utc = pd.Timestamp(str(analysis_end_utc_iso)).tz_convert("UTC")
    search_windows = _search_windows()
    stage_profiles = default_stage_profiles()
    scenarios = build_benchmark_scenarios(str(scenario_set), missing_fraction=float(missing_fraction))
    imported_request = None
    if request_payload:
        imported_request = runtime_request_from_dict(dict(request_payload.get("request", {})))
        imported_request_hash = str(request_payload.get("request_hash") or "request")
        imported_scenarios, imported_stage_profiles = build_request_validation_scenarios(
            imported_request,
            request_hash=imported_request_hash,
        )
        stage_profiles.update(imported_stage_profiles)
        mode = str(request_mode).strip().lower()
        if mode == "request_only":
            scenarios = imported_scenarios
        else:
            scenarios.extend(imported_scenarios)

    raw_rows: list[dict[str, object]] = []
    scenario_summary_rows: list[dict[str, object]] = []
    cached_unmasked_results: dict[tuple[str, str, str, int, int], dict[str, object]] = {}
    raw_timings_partial_path = run_dir / "raw_timings.partial.csv"
    scenario_summary_partial_path = run_dir / "scenario_summary.partial.csv"

    def _write_partial_outputs() -> None:
        pd.DataFrame(raw_rows).to_csv(raw_timings_partial_path, index=False)
        pd.DataFrame(scenario_summary_rows).to_csv(scenario_summary_partial_path, index=False)

    for scenario_index, scenario in enumerate(scenarios, start=1):
        progress_callback(
            f"[{scenario_index:,}/{len(scenarios):,}] {scenario.model_family} / {scenario.pooling_mode} / "
            f"{scenario.fit_mode} / p={scenario.participant_count} / b={scenario.panel_batch_size} / {scenario.stage_profile}"
        )
        participant_ids = available_ids[: int(scenario.participant_count)]
        stage_configs = stage_profiles[str(scenario.stage_profile)]
        search_units = flatten_stage_config_features(stage_configs)
        mask_build_seconds = 0.0
        benchmark_build_seconds = 0.0
        warm_start_used = False

        source_frame = cohort_frame
        if scenario.fit_mode == "masked":
            mask_start = perf_counter()
            masked_frame, _ = make_masked_cohort_frame(
                cohort_frame,
                participant_ids=participant_ids,
                mask_regime="random_hours",
                missing_fraction=float(scenario.missing_fraction),
                mask_seed=int(mask_seed),
                replicate_index=int(scenario.mask_replicate_index),
            )
            mask_build_seconds = float(perf_counter() - mask_start)
            source_frame = masked_frame

        warm_start_estimates = None
        warm_key = (
            str(scenario.model_family),
            str(scenario.pooling_mode),
            str(scenario.stage_profile),
            int(scenario.participant_count),
            int(scenario.panel_batch_size),
        )
        if scenario.fit_mode == "masked" and warm_key in cached_unmasked_results:
            warm_start_estimates = cached_unmasked_results[warm_key].get("participant_estimates")
            warm_start_used = warm_start_estimates is not None and not warm_start_estimates.empty

        fit_payload = _run_scenario_batches(
            cohort_frame=source_frame,
            selected_participants=participant_ids,
            start_utc=start_utc,
            end_utc=end_utc,
            assumed_timezone=str(assumed_local_timezone),
            model_family=str(scenario.model_family),
            pooling_mode=str(scenario.pooling_mode),
            stage_configs=stage_configs,
            search_windows=search_windows,
            global_seed=int(global_seed),
            panel_batch_size=int(scenario.panel_batch_size),
            covariate_mode=str(covariate_mode),
            mask_id="mask_rep_01" if scenario.fit_mode == "masked" else None,
            warm_start_estimates=warm_start_estimates,
            inference_only=bool(inference_only),
        )

        artifact = fit_payload["artifact"]
        if scenario.fit_mode == "unmasked":
            cached_unmasked_results[warm_key] = artifact
        if scenario.fit_mode == "masked" and not inference_only:
            benchmark_start = perf_counter()
            benchmark_artifact = benchmark_from_fit_artifact(
                artifact,
                model_family=str(scenario.model_family),
                pooling_mode=str(scenario.pooling_mode),
                mask_id="mask_rep_01",
                benchmark_spec=BENCHMARK_SPECS["core_masking"],
            )
            benchmark_build_seconds = float(perf_counter() - benchmark_start)
            artifact["benchmark_summary"] = benchmark_artifact.get("summary_frame", pd.DataFrame())

        row = _scenario_record(
            scenario=scenario,
            benchmark_tag=str(benchmark_tag),
            scenario_set=str(scenario_set),
            device=str(device),
            jax_backend=str(jax.default_backend()),
            fit_payload=fit_payload,
            mask_build_seconds=mask_build_seconds,
            benchmark_build_seconds=benchmark_build_seconds,
            warm_start_used=warm_start_used,
            search_units=search_units,
        )
        raw_rows.append(row)
        scenario_summary_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "model_family": scenario.model_family,
                "pooling_mode": scenario.pooling_mode,
                "fit_mode": scenario.fit_mode,
                "participant_count": scenario.participant_count,
                "panel_batch_size": scenario.panel_batch_size,
                "stage_profile": scenario.stage_profile,
                "fit_total_seconds": row["fit_total_seconds"],
                "inference_fit_seconds": row["inference_fit_seconds"],
                "full_wall_seconds": row["full_wall_seconds"],
                "auxiliary_artifact_seconds": row["auxiliary_artifact_seconds"],
                "panel_global_search_seconds": row["panel_global_search_seconds"],
                "postfit_filter_total_seconds": row["postfit_filter_total_seconds"],
                "postfit_fit_seconds": row["postfit_fit_seconds"],
                "postfit_simulate_seconds": row["postfit_simulate_seconds"],
                "postfit_postprocess_seconds": row["postfit_postprocess_seconds"],
                "postfit_host_overhead_seconds": row["postfit_host_overhead_seconds"],
                "fit_unattributed_seconds": row["fit_unattributed_seconds"],
                "data_prep_seconds": row["data_prep_seconds"],
                "mask_build_seconds": row["mask_build_seconds"],
                "benchmark_build_seconds": row["benchmark_build_seconds"],
                "warm_start_used": row["warm_start_used"],
            }
        )
        _write_partial_outputs()

    raw_timings = pd.DataFrame(raw_rows)
    calibration = build_runtime_calibration_frame(raw_timings)
    scenario_summary = pd.DataFrame(scenario_summary_rows)
    request_validation = pd.DataFrame()
    if request_payload and not raw_timings.empty:
        imported_rows = raw_timings.loc[raw_timings["scenario_id"].astype(str).str.startswith(f"imported_{request_payload.get('request_hash', 'request')}")].copy()
        if not imported_rows.empty:
            actual_inference_seconds = float(
                pd.to_numeric(imported_rows["inference_fit_seconds"], errors="coerce").sum()
            )
            actual_benchmark_wall_seconds = float(
                pd.to_numeric(imported_rows["full_wall_seconds"], errors="coerce").sum()
            )
            estimate_summary_rows = request_payload.get("estimate_summary", [])
            predicted_inference_seconds = float(estimate_summary_rows[0].get("estimate_seconds", np.nan)) if estimate_summary_rows else np.nan
            predicted_inference_low_seconds = float(estimate_summary_rows[0].get("estimate_low_seconds", np.nan)) if estimate_summary_rows else np.nan
            predicted_inference_high_seconds = float(estimate_summary_rows[0].get("estimate_high_seconds", np.nan)) if estimate_summary_rows else np.nan
            predicted_full_seconds = float(estimate_summary_rows[0].get("full_estimate_seconds", np.nan)) if estimate_summary_rows else np.nan
            predicted_full_low_seconds = float(estimate_summary_rows[0].get("full_estimate_low_seconds", np.nan)) if estimate_summary_rows else np.nan
            predicted_full_high_seconds = float(estimate_summary_rows[0].get("full_estimate_high_seconds", np.nan)) if estimate_summary_rows else np.nan
            inference_ratio = (
                actual_inference_seconds / predicted_inference_seconds
                if np.isfinite(predicted_inference_seconds) and predicted_inference_seconds > 0
                else np.nan
            )
            request_validation = pd.DataFrame(
                [
                    {
                        "request_hash": request_payload.get("request_hash"),
                        "label": request_payload.get("label"),
                        "estimate_target": request_payload.get("estimate_target", "inference_only"),
                        "measured_scope": "inference_only" if inference_only else "full_artifact_path",
                        "predicted_inference_low_seconds": predicted_inference_low_seconds,
                        "predicted_inference_seconds": predicted_inference_seconds,
                        "predicted_inference_high_seconds": predicted_inference_high_seconds,
                        "actual_inference_seconds": actual_inference_seconds,
                        "actual_over_predicted_inference_ratio": inference_ratio,
                        "predicted_full_low_seconds": predicted_full_low_seconds,
                        "predicted_full_seconds": predicted_full_seconds,
                        "predicted_full_high_seconds": predicted_full_high_seconds,
                        "actual_benchmark_wall_seconds": actual_benchmark_wall_seconds,
                        "actual_full_seconds": actual_benchmark_wall_seconds if not inference_only else np.nan,
                        "actual_over_predicted_full_ratio": (
                            actual_benchmark_wall_seconds / predicted_full_seconds
                            if (not inference_only) and np.isfinite(predicted_full_seconds) and predicted_full_seconds > 0
                            else np.nan
                        ),
                        "n_imported_rows": int(len(imported_rows)),
                    }
                ]
            )

    raw_timings_path = run_dir / "raw_timings.csv"
    calibration_path = run_dir / "calibration.csv"
    scenario_summary_path = run_dir / "scenario_summary.csv"
    request_validation_path = run_dir / "request_validation.csv"
    manifest_path = run_dir / "manifest.json"
    raw_timings.to_csv(raw_timings_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    scenario_summary.to_csv(scenario_summary_path, index=False)
    if not request_validation.empty:
        request_validation.to_csv(request_validation_path, index=False)
    manifest = {
        "benchmark_version": TIME_ESTIMATOR_VERSION,
        "benchmark_tag": str(benchmark_tag),
        "scenario_set": str(scenario_set),
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "device": str(device),
        "jax_backend": str(jax.default_backend()),
        "csv_path": str(Path(csv_path).resolve()),
        "project_root": str(Path(project_root).resolve()),
        "raw_timings_path": str(raw_timings_path),
        "calibration_path": str(calibration_path),
        "scenario_summary_path": str(scenario_summary_path),
        "request_validation_path": str(request_validation_path) if not request_validation.empty else None,
        "n_rows": int(len(raw_timings)),
        "n_scenarios": int(len(scenarios)),
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "stage_profiles": {name: [asdict(stage) for stage in configs] for name, configs in stage_profiles.items()},
        "model_labels": MODEL_FAMILY_LABELS,
        "pooling_labels": POOLING_MODE_LABELS,
        "all_model_families": list(ALL_MODEL_FAMILIES),
        "pooling_mode_options": list(POOLING_MODE_OPTIONS),
        "request_mode": str(request_mode),
        "inference_only": bool(inference_only),
        "imported_request_hash": request_payload.get("request_hash") if request_payload else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    raw_timings_partial_path.unlink(missing_ok=True)
    scenario_summary_partial_path.unlink(missing_ok=True)

    return {
        "run_dir": run_dir,
        "raw_timings": raw_timings,
        "calibration": calibration,
        "scenario_summary": scenario_summary,
        "request_validation": request_validation,
        "manifest": manifest,
    }
