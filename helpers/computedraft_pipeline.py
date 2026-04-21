from __future__ import annotations

import json
import math
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.ihs_eda.step_pomp import (
    ARCHITECTURE_SPEC_LOOKUP,
    COVARIATE_SPECS,
    HourlyStepModelData,
    GlobalSearchStageConfig,
    build_baseline_components,
    build_heldout_subtotal_benchmark_frame,
    build_masked_hourly_benchmark_frame,
    build_step_pomp_default_params,
    mask_fitbit_hours,
    run_multistage_panel_step_pomp_search,
    run_step_pomp_if2,
    summarize_heldout_subtotal_rmse_sqrtm,
    summarize_masked_hourly_reconstruction,
)
from src.ihs_eda.step_pomp.data import build_wear_logit_baseline_components


POMP_FAMILY_TO_ARCHITECTURE = {
    "ar1": "ar1_empirical",
    "ar2": "ar2_empirical",
    "wear_observation": "wear_observation_model",
    "wear_latent": "wear_latent_model",
    "ar2_wear_observation": "ar2_wear_observation_model",
    "ar2_wear_latent": "ar2_wear_latent_model",
    "ar1_seasonal24": "ar1_empirical_seasonal24",
    "ar2_seasonal24": "ar2_empirical_seasonal24",
}

AGGREGATE_TASK_CACHE_SCHEMA_VERSION = "v2"
SMOOTHER_TASK_CACHE_SCHEMA_VERSION = "v1"

COVARIATE_MODE_MAP = {
    "none": [],
    "base": [
        "trail_steps_24h",
        "trail_sleep_minutes_24h",
        "trail_hr_mean_24h",
        "send_lag_0_3h",
        "send_lag_4_11h",
        "send_lag_12_23h",
    ],
}

def _optional_numeric(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column_name], errors="coerce")


@dataclass(slots=True)
class AggregatePipelineResult:
    base_artifact: dict[str, object]
    task_manifest: pd.DataFrame
    derived_artifact: dict[str, object]
    timing_summary: pd.DataFrame


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    key: str
    label: str
    version: str


BENCHMARK_SPECS = {
    "core_masking": BenchmarkSpec(
        key="core_masking",
        label="Masked hourly + held-out subtotal",
        version="v1",
    ),
}

SUPPORTED_SMOOTHER_SCOPE = {
    "none",
    "from_cache_only",
    "compute_missing_only",
}
SUPPORTED_SMOOTHER_METHODS = {"backward_particle"}


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(serialized)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")


def _write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _read_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha_identity(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()[:16]


def _fit_identity_context(
    *,
    resolved_config: dict[str, object],
    participant_ids: list[str],
    search_windows: dict[str, tuple[float, float]],
) -> dict[str, object]:
    # Keep per-fit cache identity local to the fit inputs, not the whole document config.
    # This lets tomorrow's render reuse tonight's fit artifacts even if model_set/model_family changes.
    return {
        "schema_version": AGGREGATE_TASK_CACHE_SCHEMA_VERSION,
        "analysis_start_utc_iso": resolved_config["analysis_start_utc_iso"],
        "analysis_end_utc_iso": resolved_config["analysis_end_utc_iso"],
        "assumed_local_timezone": resolved_config["assumed_local_timezone"],
        "participant_ids": list(map(str, participant_ids)),
        "panel_chunk_size": (
            None if resolved_config.get("panel_chunk_size") is None else int(resolved_config["panel_chunk_size"])
        ),
        "covariate_mode": resolved_config["covariate_mode"],
        "search_stage_settings": resolved_config["search_stage_settings"],
        "search_windows": search_windows,
        "global_seed": int(resolved_config["global_seed"]),
    }


def _unmasked_fit_identity(
    *,
    resolved_config: dict[str, object],
    participant_ids: list[str],
    search_windows: dict[str, tuple[float, float]],
    model_family: str,
    pooling_mode: str,
) -> str:
    return _sha_identity(
        {
            "kind": "unmasked_fit",
            "model_family": str(model_family),
            "pooling_mode": str(pooling_mode),
            "arma_prediction_schema": "stable_v1" if str(model_family) == "arma_aic" else None,
            **_fit_identity_context(
                resolved_config=resolved_config,
                participant_ids=participant_ids,
                search_windows=search_windows,
            ),
        }
    )


def _masked_fit_identity(
    *,
    resolved_config: dict[str, object],
    participant_ids: list[str],
    search_windows: dict[str, tuple[float, float]],
    model_family: str,
    pooling_mode: str,
    mask_id: str,
) -> str:
    return _sha_identity(
        {
            "kind": "masked_fit",
            "model_family": str(model_family),
            "pooling_mode": str(pooling_mode),
            "arma_prediction_schema": "stable_v1" if str(model_family) == "arma_aic" else None,
            "mask_id": str(mask_id),
            "mask_regime": resolved_config["mask_regime"],
            "missing_fraction": float(resolved_config["missing_fraction"]),
            "mask_seed": int(resolved_config["mask_seed"]),
            **_fit_identity_context(
                resolved_config=resolved_config,
                participant_ids=participant_ids,
                search_windows=search_windows,
            ),
        }
    )


def _benchmark_identity(
    *,
    masked_fit_identity: str,
    model_family: str,
    pooling_mode: str,
    mask_id: str,
    benchmark_key: str,
    benchmark_version: str,
    smoother_identity: str | None = None,
) -> str:
    return _sha_identity(
        {
            "kind": "benchmark",
            "schema_version": AGGREGATE_TASK_CACHE_SCHEMA_VERSION,
            "benchmark_key": str(benchmark_key),
            "benchmark_version": str(benchmark_version),
            "model_family": str(model_family),
            "pooling_mode": str(pooling_mode),
            "mask_id": str(mask_id),
            "masked_fit_identity": str(masked_fit_identity),
            "smoother_identity": None if smoother_identity is None else str(smoother_identity),
            "positive_truth_only": True,
        }
    )


def _smoother_identity(
    *,
    source_fit_identity: str,
    model_family: str,
    pooling_mode: str,
    mask_id: str,
    smoother_method: str,
    backward_smoother_particles: int,
    backward_smoother_trajectories: int,
    backward_smoother_seed: int,
) -> str:
    # Smoother cache identity is downstream of the fit identity so smoother settings
    # can change without invalidating the expensive fit artifacts.
    return _sha_identity(
        {
            "kind": "smoother",
            "schema_version": SMOOTHER_TASK_CACHE_SCHEMA_VERSION,
            "source_fit_identity": str(source_fit_identity),
            "model_family": str(model_family),
            "pooling_mode": str(pooling_mode),
            "mask_id": str(mask_id),
            "smoother_method": str(smoother_method),
            "backward_smoother_particles": int(backward_smoother_particles),
            "backward_smoother_trajectories": int(backward_smoother_trajectories),
            "backward_smoother_seed": int(backward_smoother_seed),
        }
    )


def _normalize_smoother_scope(raw_value: object) -> str:
    normalized = str("none" if raw_value is None else raw_value).strip().lower()
    if normalized not in SUPPORTED_SMOOTHER_SCOPE:
        raise ValueError(
            f"Unknown smoother_scope: {raw_value!r}. Supported values: {sorted(SUPPORTED_SMOOTHER_SCOPE)}"
        )
    return normalized


def _normalize_smoother_method(raw_value: object) -> str:
    normalized = str("backward_particle" if raw_value is None else raw_value).strip().lower()
    if normalized not in SUPPORTED_SMOOTHER_METHODS:
        raise ValueError(
            f"Unknown smoother_method: {raw_value!r}. Supported values: {sorted(SUPPORTED_SMOOTHER_METHODS)}"
        )
    return normalized


def _smoother_supported_for_configuration(
    *,
    model_family: str,
    pooling_mode: str,
    architecture_key: str | None,
) -> tuple[bool, str]:
    if str(model_family) != "ar1":
        return False, "unsupported_model_family"
    if str(pooling_mode) not in {"unit_specific", "global_shared_k", "partial_pooled_k"}:
        return False, "unsupported_pooling_mode"
    if architecture_key is None:
        return False, "missing_architecture"
    architecture = ARCHITECTURE_SPEC_LOOKUP[architecture_key]
    if int(getattr(architecture, "ar_order", 1)) != 1:
        return False, "unsupported_ar_order"
    if int(getattr(architecture, "seasonal_lag_hours", 0)) > 0:
        return False, "unsupported_seasonal_model"
    if str(getattr(architecture, "missingness_mode", "standard")).strip().lower() != "standard":
        return False, "unsupported_missingness_mode"
    return True, "supported"


def _cache_hit(path: Path, force_recompute: bool) -> bool:
    return (not force_recompute) and path.exists()


def _load_csv_slice(
    csv_path: Path,
    *,
    participant_ids: list[str] | None,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    usecols = [
        "PARTICIPANTIDENTIFIER",
        "STUDY_USER_ID",
        "time_utc",
        "date_utc",
        "hour_of_day_utc",
        "appleSteps",
        "appleStepsMissing",
        "appleHRAvg",
        "appleSleepMinutes",
        "fitbitSteps",
        "fitbitStepsMissing",
        "fitbitHRAvg",
        "fitbitSleepMinutes",
        "garminSteps",
        "garminStepsMissing",
        "moodScore",
        "messageSent",
        "messageStrategy",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=250_000, dtype={"messageStrategy": "string"}):
        chunk["time_utc"] = pd.to_datetime(chunk["time_utc"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["time_utc"])
        chunk = chunk.loc[chunk["time_utc"].between(start_utc, end_utc, inclusive="both")]
        if participant_ids is not None:
            chunk = chunk.loc[chunk["PARTICIPANTIDENTIFIER"].astype(str).isin(participant_ids)]
        if chunk.empty:
            continue
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True).sort_values(["PARTICIPANTIDENTIFIER", "time_utc"]).reset_index(drop=True)


def compute_participant_summary(
    csv_path: Path,
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    usecols = ["PARTICIPANTIDENTIFIER", "time_utc", "fitbitSteps", "fitbitStepsMissing"]
    summaries: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=250_000):
        chunk["time_utc"] = pd.to_datetime(chunk["time_utc"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["time_utc"])
        chunk = chunk.loc[chunk["time_utc"].between(start_utc, end_utc, inclusive="both")]
        if chunk.empty:
            continue
        chunk["participantidentifier"] = chunk["PARTICIPANTIDENTIFIER"].astype(str)
        chunk["fitbit_observed_notna"] = pd.to_numeric(chunk["fitbitSteps"], errors="coerce").notna().astype(int)
        if "fitbitStepsMissing" in chunk.columns:
            chunk["fitbit_missing_flag"] = pd.to_numeric(chunk["fitbitStepsMissing"], errors="coerce").fillna(1.0)
        else:
            chunk["fitbit_missing_flag"] = 1.0 - chunk["fitbit_observed_notna"]
        summaries.append(
            chunk.groupby("participantidentifier", as_index=False).agg(
                hours_in_selected_range=("participantidentifier", "size"),
                observed_fitbit_hours=("fitbit_observed_notna", "sum"),
                missing_fitbit_hours_from_notna=("fitbit_observed_notna", lambda s: int(len(s) - int(s.sum()))),
                fitbit_missing_fraction_from_notna=("fitbit_observed_notna", lambda s: float(1.0 - float(np.mean(s)))),
                fitbit_missing_fraction_from_flag=("fitbit_missing_flag", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            )
        )
    if not summaries:
        return pd.DataFrame(
            columns=[
                "participantidentifier",
                "hours_in_selected_range",
                "observed_fitbit_hours",
                "missing_fitbit_hours_from_notna",
                "fitbit_missing_fraction_from_notna",
                "fitbit_missing_fraction_from_flag",
            ]
        )
    combined = pd.concat(summaries, ignore_index=True)
    result = (
        combined.groupby("participantidentifier", as_index=False)
        .agg(
            hours_in_selected_range=("hours_in_selected_range", "sum"),
            observed_fitbit_hours=("observed_fitbit_hours", "sum"),
            missing_fitbit_hours_from_notna=("missing_fitbit_hours_from_notna", "sum"),
            fitbit_missing_fraction_from_notna=("fitbit_missing_fraction_from_notna", "mean"),
            fitbit_missing_fraction_from_flag=("fitbit_missing_fraction_from_flag", "mean"),
        )
        .sort_values(["fitbit_missing_fraction_from_flag", "participantidentifier"])
        .reset_index(drop=True)
    )
    result["missingness_flag_vs_notna_absdiff"] = (
        pd.to_numeric(result["fitbit_missing_fraction_from_flag"], errors="coerce")
        - pd.to_numeric(result["fitbit_missing_fraction_from_notna"], errors="coerce")
    ).abs()
    return result


def _decision_windows(frame: pd.DataFrame) -> pd.DataFrame:
    decision_rows = frame.loc[frame["local_hour"] == 15].copy()
    if decision_rows.empty:
        return pd.DataFrame(
            columns=[
                "decision_id",
                "decision_time_utc",
                "decision_local_time",
                "decision_local_date",
                "hours_in_window",
                "complete_window",
                "fitbit_observed_hours",
                "fitbit_coverage_fraction",
                "window_start_position",
                "window_end_position",
            ]
        )
    rows: list[dict[str, object]] = []
    for decision_id, (_, row) in enumerate(decision_rows.iterrows(), start=1):
        start_pos = int(row.name)
        window = frame.iloc[start_pos : start_pos + 24].copy()
        hours_in_window = int(len(window))
        fitbit_observed = int(_optional_numeric(window, "fitbit_observed").fillna(0).sum())
        rows.append(
            {
                "decision_id": decision_id,
                "decision_time_utc": row["time_utc"],
                "decision_local_time": row["local_time"],
                "decision_local_date": row["local_date"],
                "hours_in_window": hours_in_window,
                "complete_window": bool(hours_in_window == 24),
                "fitbit_observed_hours": fitbit_observed,
                "fitbit_coverage_fraction": float(fitbit_observed / hours_in_window) if hours_in_window > 0 else np.nan,
                "window_start_position": start_pos,
                "window_end_position": start_pos + hours_in_window - 1,
            }
        )
    return pd.DataFrame(rows)


def _prepare_covariates(
    frame: pd.DataFrame,
    *,
    covariate_keys: list[str],
    baseline_mode: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, float]], list[str], dict[str, object]]:
    result = frame.copy()
    fitbit_steps = _optional_numeric(result, "fitbitSteps")
    truth_for_eval = _optional_numeric(result, "fitbit_truth_for_eval")
    natural_observed = _optional_numeric(result, "fitbit_observed_natural")
    if natural_observed.notna().any():
        natural_fitbit_wear = natural_observed.fillna(0.0).clip(lower=0.0, upper=1.0)
    else:
        natural_fitbit_wear = (fitbit_steps.notna() | truth_for_eval.notna()).astype(float)

    step_proxy = fitbit_steps.clip(lower=0.0)
    baseline_components = build_baseline_components(
        step_proxy,
        result["local_hour"],
        result["local_dayofweek"],
        baseline_mode=baseline_mode,
    )
    result["baseline_log_mean"] = baseline_components["baseline_log_mean"]
    wear_components = build_wear_logit_baseline_components(
        natural_fitbit_wear,
        result["local_hour"],
        result["local_dayofweek"],
    )
    result["fitbit_wear_observed_natural"] = natural_fitbit_wear
    result["fitbit_wear_lag1"] = natural_fitbit_wear.shift(1).fillna(float(wear_components["global_probability"]))
    result["fitbit_wear_logit_baseline"] = wear_components["baseline_logit"]
    result["fitbit_wear_probability_baseline"] = wear_components["baseline_probability"]

    sleep_minutes = _optional_numeric(result, "fitbitSleepMinutes")
    hr_avg = _optional_numeric(result, "fitbitHRAvg")
    mood_score = _optional_numeric(result, "moodScore")
    hr_fill = float(hr_avg.dropna().median()) if hr_avg.notna().any() else 0.0
    mood_fill = float(mood_score.dropna().mean()) if mood_score.notna().any() else 0.0

    result["cov_trail_steps_24h"] = step_proxy.fillna(0.0).shift(1).rolling(24, min_periods=1).sum()
    result["cov_trail_sleep_minutes_24h"] = sleep_minutes.fillna(0.0).shift(1).rolling(24, min_periods=1).sum()
    result["cov_trail_hr_mean_24h"] = hr_avg.fillna(hr_fill).shift(1).rolling(24, min_periods=1).mean()
    result["cov_trail_mood_mean_24h"] = mood_score.fillna(mood_fill).shift(1).rolling(24, min_periods=1).mean()

    send_series = _optional_numeric(result, "messageSent").fillna(0.0).clip(lower=0.0, upper=1.0)
    hours_since_send = np.full(len(result), np.inf, dtype=float)
    last_send_index: int | None = None
    for index, sent_flag in enumerate(send_series.to_numpy(dtype=float)):
        if sent_flag >= 0.5:
            last_send_index = index
            hours_since_send[index] = 0.0
        elif last_send_index is not None:
            hours_since_send[index] = float(index - last_send_index)
    result["cov_send_lag_0_3h"] = ((hours_since_send >= 0) & (hours_since_send <= 3)).astype(float)
    result["cov_send_lag_4_11h"] = ((hours_since_send >= 4) & (hours_since_send <= 11)).astype(float)
    result["cov_send_lag_12_23h"] = ((hours_since_send >= 12) & (hours_since_send <= 23)).astype(float)

    covariate_params: dict[str, str] = {}
    covariate_standardization: dict[str, dict[str, float]] = {}
    active_covariates: list[str] = []
    dropped_covariates: list[str] = []

    for key in covariate_keys:
        if key not in COVARIATE_SPECS:
            continue
        spec = COVARIATE_SPECS[key]
        values = _optional_numeric(result, spec.source_col)
        if spec.standardize:
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=0)) if values.notna().any() else 0.0
            if np.isclose(std, 0.0):
                standardized = values.fillna(mean) - mean
            else:
                standardized = (values.fillna(mean) - mean) / std
            if float(np.nanmax(np.abs(standardized.to_numpy(dtype=float)))) <= 1e-12:
                dropped_covariates.append(key)
                continue
            result[spec.source_col] = standardized.astype(float)
            covariate_standardization[key] = {"mean": mean, "std": std}
        else:
            binary_values = values.fillna(0.0).astype(float)
            if float(binary_values.max()) <= 0.0:
                dropped_covariates.append(key)
                continue
            result[spec.source_col] = binary_values
            covariate_standardization[key] = {"mean": 0.0, "std": 1.0}
        covariate_params[key] = spec.beta_param
        active_covariates.append(key)

    mean_structure_summary = {
        "baseline_mode": str(baseline_mode),
        "baseline_mean": float(pd.to_numeric(result["baseline_log_mean"], errors="coerce").mean()),
        "wear_probability_mean": float(pd.to_numeric(result["fitbit_wear_probability_baseline"], errors="coerce").mean()),
    }
    return result, covariate_params, covariate_standardization, dropped_covariates, mean_structure_summary


def build_csv_hourly_model_data(
    participant_frame: pd.DataFrame,
    *,
    participant_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    assumed_timezone: str,
    covariate_mode: str,
    architecture_key: str,
) -> HourlyStepModelData:
    architecture = ARCHITECTURE_SPEC_LOOKUP[architecture_key]
    working = participant_frame.copy()
    working["time_utc"] = pd.to_datetime(working["time_utc"], utc=True, errors="coerce")
    working = working.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    if working.empty:
        raise ValueError(f"No hourly rows available for {participant_id}")

    study_user_id = None
    if "STUDY_USER_ID" in working.columns:
        study_values = working["STUDY_USER_ID"].dropna().astype(str)
        study_user_id = study_values.iloc[0] if not study_values.empty else None

    working["local_time"] = working["time_utc"].dt.tz_convert(assumed_timezone)
    working["local_date"] = working["local_time"].dt.date
    working["local_hour"] = working["local_time"].dt.hour.astype(int)
    working["local_dayofweek"] = working["local_time"].dt.dayofweek.astype(int)

    covariate_keys = list(COVARIATE_MODE_MAP.get(str(covariate_mode), []))
    working, covariate_params, covariate_standardization, dropped_covariates, mean_summary = _prepare_covariates(
        working,
        covariate_keys=covariate_keys,
        baseline_mode=str(architecture.baseline_mode),
    )

    selected = working.loc[(working["time_utc"] >= start_utc) & (working["time_utc"] <= end_utc)].copy()
    if selected.empty:
        raise ValueError(f"No hourly rows inside the selected range for {participant_id}")

    selected["fitbit_steps_obs"] = _optional_numeric(selected, "fitbitSteps")
    selected["fitbit_observed"] = selected["fitbit_steps_obs"].notna().astype(int)
    selected["time_index"] = np.arange(1, len(selected) + 1, dtype=float)
    decision_window_frame = _decision_windows(selected)
    calibration_defaults: dict[str, float] = {}

    return HourlyStepModelData(
        participant_id=str(participant_id),
        study_user_id=study_user_id,
        hourly_frame=selected.reset_index(drop=True),
        measurement_keys=["fitbit"],
        covariate_keys=[key for key in covariate_keys if key not in dropped_covariates],
        covariate_params=covariate_params,
        covariate_standardization=covariate_standardization,
        calibration_defaults=calibration_defaults,
        mean_structure_summary=mean_summary,
        decision_window_frame=decision_window_frame,
        dropped_measurements=[],
        dropped_covariates=dropped_covariates,
        start_utc=start_utc,
        end_utc=end_utc,
        timezone_strategy=f"csv_assumed_timezone::{assumed_timezone}",
    )


def _pooling_mode_for_panel(pooling_mode: str) -> str:
    mapping = {
        "unit_specific": "unit_specific",
        "global_shared_k": "global_shared",
        "partial_pooled_k": "partial_pooled",
    }
    if pooling_mode not in mapping:
        raise ValueError(f"Unknown pooling_mode: {pooling_mode}")
    return mapping[pooling_mode]


def _pooling_modes_for_model_family(model_family: str, active_pooling_modes: list[str]) -> list[str]:
    if str(model_family) == "arma_aic":
        return ["not_applicable"]
    modes = [str(mode) for mode in active_pooling_modes]
    if not modes:
        raise ValueError("At least one active pooling mode is required.")
    return modes


def _initial_params_for_pooling(
    hourly_data: HourlyStepModelData,
    *,
    architecture_key: str,
    pooling_mode: str,
) -> dict[str, float]:
    architecture = ARCHITECTURE_SPEC_LOOKUP[architecture_key]
    defaults = build_step_pomp_default_params(
        hourly_data,
        ["fitbit"],
        hourly_data.covariate_keys,
        ar_order=int(architecture.ar_order),
        seasonal_lag_hours=int(getattr(architecture, "seasonal_lag_hours", 0)),
        missingness_mode=str(getattr(architecture, "missingness_mode", "standard")),
    )
    if pooling_mode != "partial_pooled_k":
        return defaults
    base_k = float(defaults.pop("k_fitbit"))
    defaults["log_k_fitbit"] = float(np.log(max(base_k, 1e-6)))
    defaults["mu_log_k_fitbit"] = float(np.log(max(base_k, 1e-6)))
    defaults["log_tau_log_k_fitbit"] = float(np.log(0.50))
    return defaults


def _start_box_for_initial_params(initial_params: dict[str, float], search_windows: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for name, value in initial_params.items():
        if name in search_windows:
            bounds[name] = tuple(search_windows[name])
        else:
            width = max(abs(float(value)) * 0.25, 0.10)
            bounds[name] = (float(value) - width, float(value) + width)
    return bounds


def _attach_reward_window_ids(hourly_frame: pd.DataFrame, reward_window_frame: pd.DataFrame) -> pd.DataFrame:
    frame = hourly_frame.copy()
    frame["reward_window_id"] = pd.NA
    frame["decision_date"] = pd.Series([pd.NaT] * len(frame), index=frame.index, dtype="object")
    for row in reward_window_frame.itertuples(index=False):
        start_pos = int(getattr(row, "window_start_position"))
        end_pos = int(getattr(row, "window_end_position"))
        frame.loc[start_pos : end_pos, "reward_window_id"] = getattr(row, "decision_id")
        frame.loc[start_pos : end_pos, "decision_date"] = getattr(row, "decision_local_date", pd.NaT)
    return frame


def _chunk_participant_ids(participant_ids: list[str], panel_chunk_size: int | None) -> list[list[str]]:
    if not participant_ids:
        return []
    requested_size = len(participant_ids) if panel_chunk_size is None else int(panel_chunk_size)
    chunk_size = max(1, min(requested_size, len(participant_ids)))
    return [participant_ids[index : index + chunk_size] for index in range(0, len(participant_ids), chunk_size)]


def _prediction_table_from_frames(
    hourly_frame: pd.DataFrame,
    reward_window_frame: pd.DataFrame,
    *,
    participant_id: str,
    model_family: str,
    pooling_mode: str,
    covariate_mode: str,
    mask_id: str | None,
    predicted_observed_column: str,
    predicted_latent_column: str,
) -> pd.DataFrame:
    frame = hourly_frame.copy()
    frame = _attach_reward_window_ids(frame, reward_window_frame)
    frame["participant_id"] = str(participant_id)
    frame["model_family"] = str(model_family)
    frame["pooling_mode"] = str(pooling_mode)
    frame["covariate_mode"] = str(covariate_mode)
    frame["mask_id"] = mask_id
    frame["timestamp"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    frame["is_natural_missing"] = 1.0 - _optional_numeric(frame, "fitbit_observed_natural").fillna(
        _optional_numeric(frame, "fitbit_observed").fillna(0.0)
    )
    frame["is_artificially_masked"] = _optional_numeric(frame, "fitbit_masked_for_eval").fillna(0.0)
    frame["observed_hourly_value"] = _optional_numeric(frame, "fitbit_steps_obs")
    frame["predicted_observed_scale_hourly_value"] = _optional_numeric(frame, predicted_observed_column)
    frame["predicted_latent_hourly_value"] = _optional_numeric(frame, predicted_latent_column)
    return frame


def _prediction_table_from_run(
    run_result,
    *,
    participant_id: str,
    model_family: str,
    pooling_mode: str,
    covariate_mode: str,
    mask_id: str | None,
) -> pd.DataFrame:
    return _prediction_table_from_frames(
        run_result.plot_frame,
        run_result.reward_frame,
        participant_id=participant_id,
        model_family=model_family,
        pooling_mode=pooling_mode,
        covariate_mode=covariate_mode,
        mask_id=mask_id,
        predicted_observed_column="fitbit_steps_obs_fitted",
        predicted_latent_column="latent_fitbit_scale_steps",
    )


def _reward_table_from_frames(
    reward_frame: pd.DataFrame,
    *,
    participant_id: str,
    model_family: str,
    pooling_mode: str,
    mask_id: str | None,
    predicted_latent_reward_column: str,
) -> pd.DataFrame:
    frame = reward_frame.copy()
    frame["participant_id"] = str(participant_id)
    frame["model_family"] = str(model_family)
    frame["pooling_mode"] = str(pooling_mode)
    frame["mask_id"] = mask_id
    frame["predicted_full_24h_reward"] = _optional_numeric(frame, predicted_latent_reward_column)
    frame["observed_subtotal"] = _optional_numeric(frame, "fitbit_observed_reward_24h")
    frame["artificially_masked_hours"] = 0
    frame["naturally_missing_hours"] = (
        24 - _optional_numeric(frame, "fitbit_observed_hours").fillna(0).astype(int)
    )
    return frame


def _reward_table_from_run(
    run_result,
    *,
    participant_id: str,
    model_family: str,
    pooling_mode: str,
    mask_id: str | None,
) -> pd.DataFrame:
    return _reward_table_from_frames(
        run_result.reward_frame,
        participant_id=participant_id,
        model_family=model_family,
        pooling_mode=pooling_mode,
        mask_id=mask_id,
        predicted_latent_reward_column="latent_reward_24h",
    )


def _compute_masked_hour_counts(hourly_frame: pd.DataFrame, reward_window_frame: pd.DataFrame) -> pd.DataFrame:
    if reward_window_frame.empty:
        return pd.DataFrame(columns=["decision_id", "artificially_masked_hours"])
    masked = _optional_numeric(hourly_frame, "fitbit_masked_for_eval").fillna(0.0)
    rows = []
    for row in reward_window_frame.itertuples(index=False):
        start_pos = int(getattr(row, "window_start_position"))
        end_pos = int(getattr(row, "window_end_position"))
        rows.append(
            {
                "decision_id": getattr(row, "decision_id"),
                "artificially_masked_hours": int(masked.iloc[start_pos : end_pos + 1].sum()),
            }
        )
    return pd.DataFrame(rows)


def _mask_contiguous_block(
    hourly_frame: pd.DataFrame,
    *,
    mask_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    result = hourly_frame.copy()
    observed_index = result.index[_optional_numeric(result, "fitbitSteps").notna()].to_numpy()
    result["fitbit_masked_for_eval"] = False
    result["fitbit_truth_for_eval"] = np.nan
    result["masked_reward_window"] = False
    result["masked_reward_window_id"] = pd.NA
    if observed_index.size == 0:
        return result
    n_mask = max(1, int(np.floor(float(mask_fraction) * observed_index.size)))
    n_mask = min(n_mask, int(observed_index.size))
    rng = np.random.default_rng(int(random_seed))
    observed_sorted = np.sort(observed_index)
    if n_mask >= len(observed_sorted):
        selected = observed_sorted
    else:
        start_at = int(rng.integers(0, len(observed_sorted) - n_mask + 1))
        selected = observed_sorted[start_at : start_at + n_mask]
    result.loc[selected, "fitbit_masked_for_eval"] = True
    result.loc[selected, "fitbit_truth_for_eval"] = pd.to_numeric(result.loc[selected, "fitbitSteps"], errors="coerce")
    result.loc[selected, "fitbitSteps"] = np.nan
    return result


def make_masked_cohort_frame(
    cohort_frame: pd.DataFrame,
    *,
    participant_ids: list[str],
    mask_regime: str,
    missing_fraction: float,
    mask_seed: int,
    replicate_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    mask_rows: list[dict[str, object]] = []
    for offset, participant_id in enumerate(participant_ids):
        raw = cohort_frame.loc[cohort_frame["PARTICIPANTIDENTIFIER"].astype(str) == str(participant_id)].copy()
        participant_seed = int(mask_seed) + int(replicate_index) * 10_000 + offset
        raw["fitbit_observed_natural"] = _optional_numeric(raw, "fitbitSteps").notna().astype(float)
        if mask_regime == "random_hours":
            masked = mask_fitbit_hours(
                raw,
                fitbit_column="fitbitSteps",
                mask_fraction=float(missing_fraction),
                random_seed=participant_seed,
            )
        elif mask_regime == "contiguous_block":
            masked = _mask_contiguous_block(
                raw,
                mask_fraction=float(missing_fraction),
                random_seed=participant_seed,
            )
        else:
            raise ValueError(f"Unknown mask_regime: {mask_regime}")
        masked["participantidentifier"] = str(participant_id)
        frames.append(masked)
        mask_rows.append(
            {
                "participant_id": str(participant_id),
                "mask_regime": str(mask_regime),
                "missing_fraction": float(missing_fraction),
                "mask_seed": int(mask_seed),
                "mask_replicate": int(replicate_index),
                "masked_hours": int(_optional_numeric(masked, "fitbit_masked_for_eval").fillna(0.0).sum()),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(mask_rows)


def _arma_grid(rl: int) -> list[tuple[int, int]]:
    if int(rl) <= 0:
        values = [0, 1]
    else:
        values = [0, 1, 2]
    return [(p, q) for p in values for q in values if not (p == 0 and q == 0)]


def _stable_steps_from_predicted_log(predicted_log: pd.Series, observed_steps: pd.Series) -> pd.Series:
    finite_observed = pd.to_numeric(observed_steps, errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed_max = float(finite_observed.max()) if not finite_observed.empty else np.nan
    if not np.isfinite(observed_max) or observed_max <= 0:
        prediction_cap = 100_000.0
    else:
        prediction_cap = max(1.0, observed_max * 10.0)
    clipped_log = pd.to_numeric(predicted_log, errors="coerce").clip(upper=np.log1p(prediction_cap))
    return pd.Series(np.expm1(clipped_log), index=predicted_log.index, dtype="float64").clip(lower=0.0)


def _run_arma_participant(
    participant_data: HourlyStepModelData,
    *,
    rl: int,
) -> dict[str, object]:
    frame = participant_data.hourly_frame.copy()
    series = np.log1p(_optional_numeric(frame, "fitbit_steps_obs"))
    best_result = None
    best_order = None
    best_aic = np.inf
    for p, q in _arma_grid(rl):
        try:
            model = ARIMA(
                series,
                order=(int(p), 0, int(q)),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit()
        except Exception:
            continue
        if float(fitted.aic) < float(best_aic):
            best_aic = float(fitted.aic)
            best_order = (int(p), int(q))
            best_result = fitted
    if best_result is None or best_order is None:
        raise ValueError(f"No ARMA model converged for participant {participant_data.participant_id}")

    predicted_log = pd.Series(best_result.predict(start=0, end=len(frame) - 1), index=frame.index, dtype="float64")
    predicted_steps = _stable_steps_from_predicted_log(predicted_log, _optional_numeric(frame, "fitbit_steps_obs"))
    plot_frame = frame.copy()
    plot_frame["latent_fitbit_scale_steps"] = predicted_steps
    plot_frame["fitbit_steps_obs_fitted"] = predicted_steps
    plot_frame = _attach_reward_window_ids(plot_frame, participant_data.decision_window_frame)

    reward_rows = []
    for _, reward_row in participant_data.decision_window_frame.iterrows():
        start_pos = int(reward_row["window_start_position"])
        end_pos = int(reward_row["window_end_position"])
        window = plot_frame.iloc[start_pos : end_pos + 1].copy()
        reward_rows.append(
            {
                **reward_row.to_dict(),
                "latent_reward_24h": float(pd.to_numeric(window["latent_fitbit_scale_steps"], errors="coerce").sum(min_count=1)),
                "fitbit_observed_reward_24h": float(pd.to_numeric(window["fitbit_steps_obs"], errors="coerce").sum(min_count=1)),
                "fitbit_fitted_reward_24h": float(pd.to_numeric(window["fitbit_steps_obs_fitted"], errors="coerce").sum(min_count=1)),
            }
        )
    reward_frame = pd.DataFrame(reward_rows)

    return {
        "participant_id": str(participant_data.participant_id),
        "order_p": int(best_order[0]),
        "order_q": int(best_order[1]),
        "aic": float(best_result.aic),
        "loglik": float(best_result.llf),
        "plot_frame": plot_frame,
        "reward_frame": reward_frame,
    }


def run_arma_model(
    data_by_participant: dict[str, HourlyStepModelData],
    *,
    rl: int,
    model_family: str,
    pooling_mode: str,
    covariate_mode: str,
    mask_id: str | None,
    include_prediction_artifacts: bool = True,
) -> dict[str, object]:
    rows = []
    hourly_frames = []
    reward_frames = []
    timing_rows = []
    for participant_id in sorted(data_by_participant):
        start = perf_counter()
        result = _run_arma_participant(data_by_participant[participant_id], rl=rl)
        runtime = float(perf_counter() - start)
        rows.append(
            {
                "participantidentifier": str(participant_id),
                "phi": np.nan,
                "sigma": np.nan,
                "k_fitbit": np.nan,
                "arma_p": int(result["order_p"]),
                "arma_q": int(result["order_q"]),
                "aic": float(result["aic"]),
                "panel_unit_loglik": float(result["loglik"]),
            }
        )
        if include_prediction_artifacts:
            hourly = result["plot_frame"].copy()
            hourly["participant_id"] = str(participant_id)
            hourly["model_family"] = str(model_family)
            hourly["pooling_mode"] = str(pooling_mode)
            hourly["covariate_mode"] = str(covariate_mode)
            hourly["mask_id"] = mask_id
            hourly["timestamp"] = pd.to_datetime(hourly["time_utc"], utc=True, errors="coerce")
            hourly["is_natural_missing"] = 1.0 - _optional_numeric(hourly, "fitbit_observed_natural").fillna(
                _optional_numeric(hourly, "fitbit_observed").fillna(0.0)
            )
            hourly["is_artificially_masked"] = _optional_numeric(hourly, "fitbit_masked_for_eval").fillna(0.0)
            hourly["observed_hourly_value"] = _optional_numeric(hourly, "fitbit_steps_obs")
            hourly["predicted_observed_scale_hourly_value"] = _optional_numeric(hourly, "fitbit_steps_obs_fitted")
            hourly["predicted_latent_hourly_value"] = _optional_numeric(hourly, "latent_fitbit_scale_steps")
            hourly_frames.append(hourly)
            reward = result["reward_frame"].copy()
            reward["participant_id"] = str(participant_id)
            reward["model_family"] = str(model_family)
            reward["pooling_mode"] = str(pooling_mode)
            reward["mask_id"] = mask_id
            reward["predicted_full_24h_reward"] = _optional_numeric(reward, "latent_reward_24h")
            reward["observed_subtotal"] = _optional_numeric(reward, "fitbit_observed_reward_24h")
            reward["artificially_masked_hours"] = 0
            reward["naturally_missing_hours"] = 24 - _optional_numeric(reward, "fitbit_observed_hours").fillna(0).astype(int)
            reward_frames.append(reward)
        timing_rows.append({"participant_id": str(participant_id), "step": "arma_fit", "seconds": runtime})
    participant_estimates = pd.DataFrame(rows)
    estimate_summary = (
        participant_estimates[["arma_p", "arma_q", "aic", "panel_unit_loglik"]]
        .agg(["mean", "std", "median", "min", "max"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "parameter"})
    )
    return {
        "kind": "arma_aic",
        "participant_estimates": participant_estimates,
        "estimate_summary": estimate_summary,
        "shared_estimates": {},
        "candidate_frame": pd.DataFrame(),
        "stage_summary": pd.DataFrame(),
        "best_loglik": float(pd.to_numeric(participant_estimates["panel_unit_loglik"], errors="coerce").sum()),
        "hourly_prediction_frame": pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame(),
        "reward_window_frame": pd.concat(reward_frames, ignore_index=True) if reward_frames else pd.DataFrame(),
        "timing_frame": pd.DataFrame(timing_rows),
    }


def run_panel_pomp_model(
    data_by_participant: dict[str, HourlyStepModelData],
    *,
    architecture_key: str,
    pooling_mode: str,
    search_windows: dict[str, tuple[float, float]],
    stage_configs: list[GlobalSearchStageConfig],
    global_seed: int,
    model_family: str,
    covariate_mode: str,
    mask_id: str | None,
    warm_start_estimates: pd.DataFrame | None,
    panel_chunk_size: int | None = None,
    include_postfit_reconstruction: bool = True,
) -> dict[str, object]:
    architecture = ARCHITECTURE_SPEC_LOOKUP[architecture_key]
    if int(getattr(architecture, "seasonal_lag_hours", 0)) > 0:
        raise ValueError(
            f"Aggregate panel fitting in computedraft does not currently support seasonal architecture {architecture_key!r}."
        )
    participant_ids = sorted(str(participant_id) for participant_id in data_by_participant)
    participant_chunks = _chunk_participant_ids(participant_ids, panel_chunk_size)

    participant_estimate_frames: list[pd.DataFrame] = []
    estimate_summary_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    stage_summary_frames: list[pd.DataFrame] = []
    hourly_frames: list[pd.DataFrame] = []
    reward_frames: list[pd.DataFrame] = []
    timing_rows: list[dict[str, object]] = []
    total_best_loglik = 0.0
    shared_estimates: dict[str, float] = {}

    for chunk_index, participant_chunk in enumerate(participant_chunks, start=1):
        chunk_data_by_participant = {participant_id: data_by_participant[participant_id] for participant_id in participant_chunk}
        chunk_warm_start = None
        if warm_start_estimates is not None and not warm_start_estimates.empty:
            chunk_warm_start = warm_start_estimates.loc[
                warm_start_estimates["participantidentifier"].astype(str).isin(list(map(str, participant_chunk)))
            ].copy()

        initial_params_by_participant: dict[str, dict[str, float]] = {}
        for participant_id, hourly_data in chunk_data_by_participant.items():
            params = _initial_params_for_pooling(
                hourly_data,
                architecture_key=architecture_key,
                pooling_mode=pooling_mode,
            )
            if chunk_warm_start is not None and not chunk_warm_start.empty:
                match = chunk_warm_start.loc[chunk_warm_start["participantidentifier"].astype(str) == str(participant_id)]
                if not match.empty:
                    for name in list(params):
                        if name in match.columns and pd.notna(match.iloc[0][name]):
                            params[name] = float(match.iloc[0][name])
            initial_params_by_participant[str(participant_id)] = params

        sample_params = initial_params_by_participant[next(iter(initial_params_by_participant))]
        start_box = _start_box_for_initial_params(sample_params, search_windows)
        panel_result = run_multistage_panel_step_pomp_search(
            chunk_data_by_participant,
            initial_params_by_participant=initial_params_by_participant,
            free_params=list(sample_params.keys()),
            start_box=start_box,
            stage_configs=stage_configs,
            random_seed=int(global_seed) + (chunk_index - 1) * 10_000,
            ar_order=int(architecture.ar_order),
            missingness_mode=str(getattr(architecture, "missingness_mode", "standard")),
            k_fitbit_pooling_mode=_pooling_mode_for_panel(pooling_mode),
            include_current_theta_as_start=True,
            jitter_scale=0.10,
            box_shrink_factor_per_stage=1.00,
        )

        chunk_participant_estimates = panel_result.participant_estimates.copy()
        chunk_participant_estimates["panel_chunk_index"] = int(chunk_index)
        chunk_participant_estimates["panel_chunk_size_requested"] = int(
            len(participant_ids) if panel_chunk_size is None else panel_chunk_size
        )
        chunk_participant_estimates["panel_chunk_n_participants"] = int(len(participant_chunk))
        participant_estimate_frames.append(chunk_participant_estimates)

        chunk_estimate_summary = panel_result.estimate_summary.copy()
        if not chunk_estimate_summary.empty:
            chunk_estimate_summary["panel_chunk_index"] = int(chunk_index)
            chunk_estimate_summary["panel_chunk_n_participants"] = int(len(participant_chunk))
            estimate_summary_frames.append(chunk_estimate_summary)

        chunk_candidate_frame = panel_result.candidate_frame.copy()
        if not chunk_candidate_frame.empty:
            chunk_candidate_frame["panel_chunk_index"] = int(chunk_index)
            candidate_frames.append(chunk_candidate_frame)

        chunk_stage_summary = panel_result.stage_summary.copy()
        if not chunk_stage_summary.empty:
            chunk_stage_summary["panel_chunk_index"] = int(chunk_index)
            chunk_stage_summary["panel_chunk_n_participants"] = int(len(participant_chunk))
            stage_summary_frames.append(chunk_stage_summary)

        timing_rows.append(
            {
                "participant_id": pd.NA,
                "panel_chunk_index": int(chunk_index),
                "step": "panel_global_search",
                "seconds": float(panel_result.total_runtime_seconds),
            }
        )
        total_best_loglik += float(panel_result.best_loglik)
        if len(participant_chunks) == 1:
            shared_estimates = dict(panel_result.shared_estimates)

        if include_postfit_reconstruction:
            for participant_id, hourly_data in chunk_data_by_participant.items():
                estimate_row = chunk_participant_estimates.loc[
                    chunk_participant_estimates["participantidentifier"].astype(str) == str(participant_id)
                ]
                if estimate_row.empty:
                    continue
                participant_params = {
                    str(key): float(estimate_row.iloc[0][key])
                    for key in estimate_row.columns
                    if key not in {
                        "participantidentifier",
                        "panel_unit_loglik",
                        "k_fitbit_pooling_mode",
                        "panel_chunk_index",
                        "panel_chunk_size_requested",
                        "panel_chunk_n_participants",
                    }
                    and pd.notna(estimate_row.iloc[0][key])
                }
                run_start = perf_counter()
                run_result = run_step_pomp_if2(
                    hourly_data,
                    initial_params=participant_params,
                    free_params=[],
                    particles=int(stage_configs[-1].evaluation_particles),
                    mif_iterations=0,
                    random_seed=int(global_seed) + hash(str(participant_id)) % 10_000,
                    cooling_fraction=float(stage_configs[-1].cooling_fraction),
                    rw_sd_scale=float(stage_configs[-1].rw_sd_scale),
                    simulation_count=1,
                    pfilter_reps=1,
                    backward_smoother_enabled=False,
                    evaluation_particles=int(stage_configs[-1].evaluation_particles),
                    evaluation_pfilter_reps=int(stage_configs[-1].evaluation_pfilter_reps),
                    ar_order=int(architecture.ar_order),
                    seasonal_lag_hours=int(getattr(architecture, "seasonal_lag_hours", 0)),
                    missingness_mode=str(getattr(architecture, "missingness_mode", "standard")),
                )
                runtime = float(perf_counter() - run_start)
                timing_rows.append(
                    {
                        "participant_id": str(participant_id),
                        "panel_chunk_index": int(chunk_index),
                        "step": "panel_postfit_filter",
                        "seconds": runtime,
                    }
                )
                runtime_breakdown = run_result.runtime_breakdown.copy()
                if not runtime_breakdown.empty and {"step", "seconds"}.issubset(runtime_breakdown.columns):
                    for runtime_row in runtime_breakdown.itertuples(index=False):
                        timing_rows.append(
                            {
                                "participant_id": str(participant_id),
                                "panel_chunk_index": int(chunk_index),
                                "step": f"panel_postfit_{str(getattr(runtime_row, 'step'))}",
                                "seconds": float(getattr(runtime_row, "seconds")),
                            }
                        )
                    run_total_series = pd.to_numeric(
                        runtime_breakdown.loc[runtime_breakdown["step"].astype(str) == "run_total", "seconds"],
                        errors="coerce",
                    )
                    if run_total_series.notna().any():
                        run_total_seconds = float(run_total_series.sum())
                        timing_rows.append(
                            {
                                "participant_id": str(participant_id),
                                "panel_chunk_index": int(chunk_index),
                                "step": "panel_postfit_host_overhead",
                                "seconds": float(max(runtime - run_total_seconds, 0.0)),
                            }
                        )
                hourly_frames.append(
                    _prediction_table_from_run(
                        run_result,
                        participant_id=str(participant_id),
                        model_family=model_family,
                        pooling_mode=pooling_mode,
                        covariate_mode=covariate_mode,
                        mask_id=mask_id,
                    )
                )
                reward_frame = _reward_table_from_run(
                    run_result,
                    participant_id=str(participant_id),
                    model_family=model_family,
                    pooling_mode=pooling_mode,
                    mask_id=mask_id,
                )
                reward_frame = reward_frame.merge(
                    _compute_masked_hour_counts(run_result.plot_frame, run_result.reward_frame),
                    how="left",
                    on="decision_id",
                )
                reward_frame["artificially_masked_hours"] = pd.to_numeric(
                    reward_frame.get("artificially_masked_hours_y"), errors="coerce"
                ).fillna(
                    pd.to_numeric(reward_frame.get("artificially_masked_hours_x"), errors="coerce")
                ).fillna(0).astype(int)
                reward_frames.append(reward_frame)

    participant_estimates = pd.concat(participant_estimate_frames, ignore_index=True) if participant_estimate_frames else pd.DataFrame()
    estimate_summary = pd.concat(estimate_summary_frames, ignore_index=True) if estimate_summary_frames else pd.DataFrame()
    candidate_frame = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    stage_summary = pd.concat(stage_summary_frames, ignore_index=True) if stage_summary_frames else pd.DataFrame()

    return {
        "kind": "panel_pomp",
        "participant_estimates": participant_estimates,
        "estimate_summary": estimate_summary,
        "shared_estimates": shared_estimates,
        "candidate_frame": candidate_frame,
        "stage_summary": stage_summary,
        "best_loglik": float(total_best_loglik),
        "hourly_prediction_frame": pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame(),
        "reward_window_frame": pd.concat(reward_frames, ignore_index=True) if reward_frames else pd.DataFrame(),
        "timing_frame": pd.DataFrame(timing_rows),
    }


def _participant_params_from_estimate_row(estimate_row: pd.Series) -> dict[str, float]:
    skip_columns = {
        "participantidentifier",
        "participant_id",
        "panel_unit_loglik",
        "k_fitbit_pooling_mode",
        "panel_chunk_index",
        "panel_chunk_size_requested",
        "panel_chunk_n_participants",
    }
    params: dict[str, float] = {}
    for key, value in estimate_row.items():
        if str(key) in skip_columns:
            continue
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric_value):
            params[str(key)] = float(numeric_value)
    return params


def _participant_estimate_lookup_column(participant_estimates: pd.DataFrame) -> str | None:
    for candidate in ("participantidentifier", "participant_id", "PARTICIPANTIDENTIFIER"):
        if candidate in participant_estimates.columns:
            return candidate
    return None


def _stable_participant_seed(base_seed: int, participant_id: str) -> int:
    offset = sum((index + 1) * ord(char) for index, char in enumerate(str(participant_id)))
    return int(base_seed) + int(offset % 1_000_000)


def build_backward_smoother_artifact_from_masked_fit(
    masked_fit_artifact: dict[str, object],
    *,
    data_by_participant: dict[str, HourlyStepModelData],
    architecture_key: str,
    model_family: str,
    pooling_mode: str,
    covariate_mode: str,
    mask_id: str,
    stage_configs: list[GlobalSearchStageConfig],
    global_seed: int,
    source_fit_identity: str,
    smoother_method: str,
    backward_smoother_particles: int,
    backward_smoother_trajectories: int,
    backward_smoother_seed: int,
) -> dict[str, object]:
    if str(smoother_method) != "backward_particle":
        raise ValueError(f"Unsupported smoother_method for v1 smoother stage: {smoother_method!r}")
    architecture = ARCHITECTURE_SPEC_LOOKUP[architecture_key]
    # Always replay from participant_estimates (not shared_estimates) so each
    # participant uses the exact fitted vector produced by the masked fit stage.
    participant_estimates = masked_fit_artifact.get("participant_estimates", pd.DataFrame())
    if participant_estimates is None or participant_estimates.empty:
        raise ValueError("Cannot build smoother artifact: masked fit participant_estimates are empty.")
    participant_id_column = _participant_estimate_lookup_column(participant_estimates)
    if participant_id_column is None:
        raise ValueError("Cannot build smoother artifact: participant_estimates missing participant identifier column.")

    filtered_hourly_frames: list[pd.DataFrame] = []
    smoothed_hourly_frames: list[pd.DataFrame] = []
    filtered_reward_frames: list[pd.DataFrame] = []
    smoothed_reward_frames: list[pd.DataFrame] = []
    smoothed_imputation_frames: list[pd.DataFrame] = []
    smoother_summary_frames: list[pd.DataFrame] = []
    timing_rows: list[dict[str, object]] = []
    run_status_rows: list[dict[str, object]] = []

    for participant_id in sorted(data_by_participant):
        estimate_row = participant_estimates.loc[
            participant_estimates[participant_id_column].astype(str) == str(participant_id)
        ]
        if estimate_row.empty:
            run_status_rows.append(
                {
                    "participant_id": str(participant_id),
                    "status": "missing_participant_estimate",
                }
            )
            continue
        participant_params = _participant_params_from_estimate_row(estimate_row.iloc[0])
        if not participant_params:
            run_status_rows.append(
                {
                    "participant_id": str(participant_id),
                    "status": "missing_numeric_parameters",
                }
            )
            continue
        participant_data = data_by_participant[str(participant_id)]
        participant_seed = _stable_participant_seed(int(backward_smoother_seed), str(participant_id))
        run_start = perf_counter()
        # Replay uses masked inputs only; held-out truth columns remain in the frame
        # for post-hoc benchmarking and are never used as observations.
        run_result = run_step_pomp_if2(
            participant_data,
            initial_params=participant_params,
            free_params=[],
            particles=int(stage_configs[-1].evaluation_particles),
            mif_iterations=0,
            random_seed=participant_seed,
            cooling_fraction=float(stage_configs[-1].cooling_fraction),
            rw_sd_scale=float(stage_configs[-1].rw_sd_scale),
            simulation_count=1,
            pfilter_reps=int(stage_configs[-1].evaluation_pfilter_reps),
            backward_smoother_particles=int(backward_smoother_particles),
            backward_smoother_trajectories=int(backward_smoother_trajectories),
            backward_smoother_seed=participant_seed,
            backward_smoother_enabled=True,
            evaluation_particles=int(stage_configs[-1].evaluation_particles),
            evaluation_pfilter_reps=int(stage_configs[-1].evaluation_pfilter_reps),
            ar_order=int(architecture.ar_order),
            seasonal_lag_hours=int(getattr(architecture, "seasonal_lag_hours", 0)),
            missingness_mode=str(getattr(architecture, "missingness_mode", "standard")),
        )
        runtime_seconds = float(perf_counter() - run_start)
        timing_rows.append(
            {
                "participant_id": str(participant_id),
                "step": "smoother_postfit_replay",
                "seconds": runtime_seconds,
            }
        )
        runtime_breakdown = run_result.runtime_breakdown.copy()
        if not runtime_breakdown.empty and {"step", "seconds"}.issubset(runtime_breakdown.columns):
            for runtime_row in runtime_breakdown.itertuples(index=False):
                timing_rows.append(
                    {
                        "participant_id": str(participant_id),
                        "step": f"smoother_{str(getattr(runtime_row, 'step'))}",
                        "seconds": float(getattr(runtime_row, "seconds")),
                    }
                )

        smoother_summary = run_result.backward_smoother_summary.copy()
        smoother_summary["participant_id"] = str(participant_id)
        method_series = smoother_summary.get("method", pd.Series(dtype=str)).astype(str).str.lower()
        # v1 aggregate smoothing surfaces only backward-particle smoother output.
        if method_series.empty or (~method_series.eq("backward_particle_smoother")).any():
            raise ValueError(
                "Backward smoother artifact requested only backward-particle outputs, "
                f"but participant {participant_id} returned method(s) {method_series.tolist()}."
            )
        smoother_summary_frames.append(smoother_summary)

        filtered_hourly = _prediction_table_from_frames(
            run_result.plot_frame,
            run_result.reward_frame,
            participant_id=str(participant_id),
            model_family=model_family,
            pooling_mode=pooling_mode,
            covariate_mode=covariate_mode,
            mask_id=mask_id,
            predicted_observed_column="fitbit_steps_obs_fitted",
            predicted_latent_column="latent_fitbit_scale_steps",
        )
        filtered_reward = _reward_table_from_frames(
            run_result.reward_frame,
            participant_id=str(participant_id),
            model_family=model_family,
            pooling_mode=pooling_mode,
            mask_id=mask_id,
            predicted_latent_reward_column="latent_reward_24h",
        )
        filtered_reward = filtered_reward.merge(
            _compute_masked_hour_counts(run_result.plot_frame, run_result.reward_frame),
            how="left",
            on="decision_id",
        )
        filtered_reward["artificially_masked_hours"] = pd.to_numeric(
            filtered_reward.get("artificially_masked_hours_y"), errors="coerce"
        ).fillna(
            pd.to_numeric(filtered_reward.get("artificially_masked_hours_x"), errors="coerce")
        ).fillna(0).astype(int)

        smoothed_hourly = _prediction_table_from_frames(
            run_result.backward_smoothed_frame,
            run_result.backward_smoothed_reward_frame,
            participant_id=str(participant_id),
            model_family=model_family,
            pooling_mode=pooling_mode,
            covariate_mode=covariate_mode,
            mask_id=mask_id,
            predicted_observed_column="fitbit_steps_obs_fitted_backward",
            predicted_latent_column="latent_fitbit_scale_steps_backward",
        )
        smoothed_reward = _reward_table_from_frames(
            run_result.backward_smoothed_reward_frame,
            participant_id=str(participant_id),
            model_family=model_family,
            pooling_mode=pooling_mode,
            mask_id=mask_id,
            predicted_latent_reward_column="latent_reward_24h",
        )
        smoothed_reward = smoothed_reward.merge(
            _compute_masked_hour_counts(run_result.backward_smoothed_frame, run_result.backward_smoothed_reward_frame),
            how="left",
            on="decision_id",
        )
        smoothed_reward["artificially_masked_hours"] = pd.to_numeric(
            smoothed_reward.get("artificially_masked_hours_y"), errors="coerce"
        ).fillna(
            pd.to_numeric(smoothed_reward.get("artificially_masked_hours_x"), errors="coerce")
        ).fillna(0).astype(int)

        smoothed_imputation = run_result.backward_smoothed_imputation_frame.copy()
        smoothed_imputation["participant_id"] = str(participant_id)
        smoothed_imputation["model_family"] = str(model_family)
        smoothed_imputation["pooling_mode"] = str(pooling_mode)
        smoothed_imputation["mask_id"] = str(mask_id)

        filtered_hourly_frames.append(filtered_hourly)
        smoothed_hourly_frames.append(smoothed_hourly)
        filtered_reward_frames.append(filtered_reward)
        smoothed_reward_frames.append(smoothed_reward)
        smoothed_imputation_frames.append(smoothed_imputation)
        run_status_rows.append(
            {
                "participant_id": str(participant_id),
                "status": "done",
                "runtime_seconds": runtime_seconds,
            }
        )

    return {
        "kind": "smoother",
        "schema_version": SMOOTHER_TASK_CACHE_SCHEMA_VERSION,
        "source_fit_identity": str(source_fit_identity),
        "model_family": str(model_family),
        "pooling_mode": str(pooling_mode),
        "mask_id": str(mask_id),
        "smoother_method": str(smoother_method),
        "backward_smoother_particles": int(backward_smoother_particles),
        "backward_smoother_trajectories": int(backward_smoother_trajectories),
        "backward_smoother_seed": int(backward_smoother_seed),
        "participant_estimates_snapshot": participant_estimates.copy(),
        "hourly_filtered_frame": pd.concat(filtered_hourly_frames, ignore_index=True) if filtered_hourly_frames else pd.DataFrame(),
        "hourly_smoothed_frame": pd.concat(smoothed_hourly_frames, ignore_index=True) if smoothed_hourly_frames else pd.DataFrame(),
        "reward_filtered_frame": pd.concat(filtered_reward_frames, ignore_index=True) if filtered_reward_frames else pd.DataFrame(),
        "reward_smoothed_frame": pd.concat(smoothed_reward_frames, ignore_index=True) if smoothed_reward_frames else pd.DataFrame(),
        "imputation_smoothed_frame": pd.concat(smoothed_imputation_frames, ignore_index=True) if smoothed_imputation_frames else pd.DataFrame(),
        "backward_smoother_summary": pd.concat(smoother_summary_frames, ignore_index=True) if smoother_summary_frames else pd.DataFrame(),
        "timing_frame": pd.DataFrame(timing_rows),
        "participant_run_status": pd.DataFrame(run_status_rows),
    }


def _core_masking_benchmark_from_fit_artifact(
    fit_artifact: dict[str, object],
    *,
    model_family: str,
    pooling_mode: str,
    mask_id: str,
    benchmark_spec: BenchmarkSpec,
    smoother_artifact: dict[str, object] | None = None,
) -> dict[str, object]:
    filtered_hourly = fit_artifact["hourly_prediction_frame"].copy()
    filtered_reward = fit_artifact["reward_window_frame"].copy()
    if filtered_hourly.empty:
        return {
            "benchmark_key": benchmark_spec.key,
            "benchmark_label": benchmark_spec.label,
            "hourly_benchmark_frame": pd.DataFrame(),
            "daily_subtotal_frame": pd.DataFrame(),
            "summary_frame": pd.DataFrame(),
        }

    def _summarize_estimate_type(
        *,
        hourly_frame: pd.DataFrame,
        reward_frame: pd.DataFrame,
        estimate_column: str,
        estimate_type: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        hourly_benchmark = build_masked_hourly_benchmark_frame(
            hourly_frame,
            truth_column="fitbit_truth_for_eval",
            mask_flag_column="fitbit_masked_for_eval",
            positive_truth_only=True,
            estimate_columns=[estimate_column],
        )
        daily_subtotal = build_heldout_subtotal_benchmark_frame(
            hourly_frame,
            reward_frame,
            truth_column="fitbit_truth_for_eval",
            estimate_column=estimate_column,
            mask_flag_column="fitbit_masked_for_eval",
        )
        hourly_summary = summarize_masked_hourly_reconstruction(
            hourly_frame,
            truth_column="fitbit_truth_for_eval",
            estimate_column=estimate_column,
            mask_flag_column="fitbit_masked_for_eval",
            positive_truth_only=True,
        )
        subtotal_summary = summarize_heldout_subtotal_rmse_sqrtm(
            hourly_frame,
            reward_frame,
            truth_column="fitbit_truth_for_eval",
            estimate_column=estimate_column,
            mask_flag_column="fitbit_masked_for_eval",
        )
        daily_truth = pd.to_numeric(daily_subtotal.get("heldout_true_subtotal"), errors="coerce")
        daily_estimate = pd.to_numeric(daily_subtotal.get("heldout_predicted_subtotal"), errors="coerce")
        daily_pair = pd.DataFrame({"truth": daily_truth, "estimate": daily_estimate}).dropna()
        daily_correlation = (
            float(daily_pair["truth"].corr(daily_pair["estimate"])) if len(daily_pair) >= 2 else np.nan
        )
        summary = pd.DataFrame(
            [
                {
                    "benchmark_key": str(benchmark_spec.key),
                    "benchmark_label": str(benchmark_spec.label),
                    "benchmark_version": str(benchmark_spec.version),
                    "model_family": str(model_family),
                    "pooling_mode": str(pooling_mode),
                    "mask_id": str(mask_id),
                    "estimate_type": str(estimate_type),
                    "hourly_masked_rmse": float(pd.to_numeric(hourly_summary["rmse"], errors="coerce").iloc[0]) if not hourly_summary.empty else np.nan,
                    "hourly_masked_mae": float(pd.to_numeric(hourly_summary["mae"], errors="coerce").iloc[0]) if not hourly_summary.empty else np.nan,
                    "hourly_masked_correlation": float(pd.to_numeric(hourly_summary["correlation"], errors="coerce").iloc[0]) if not hourly_summary.empty else np.nan,
                    "daily_correlation": daily_correlation,
                    "heldout_subtotal_rmse_sqrtm": float(pd.to_numeric(subtotal_summary["rmse_sqrtm"], errors="coerce").iloc[0]) if not subtotal_summary.empty else np.nan,
                    "mean_signed_subtotal_error": float(pd.to_numeric(subtotal_summary["mean_signed_subtotal_error"], errors="coerce").iloc[0]) if not subtotal_summary.empty else np.nan,
                    "mean_signed_normalized_subtotal_error": float(pd.to_numeric(subtotal_summary["mean_signed_normalized_subtotal_error"], errors="coerce").iloc[0]) if not subtotal_summary.empty else np.nan,
                    "n_windows_used": int(pd.to_numeric(subtotal_summary["n_windows_used"], errors="coerce").fillna(0).iloc[0]) if not subtotal_summary.empty else 0,
                    "mean_masked_hours": float(pd.to_numeric(subtotal_summary["mean_masked_hours"], errors="coerce").iloc[0]) if not subtotal_summary.empty else np.nan,
                }
            ]
        )
        hourly_benchmark["estimate_type"] = str(estimate_type)
        daily_subtotal["estimate_type"] = str(estimate_type)
        return hourly_benchmark, daily_subtotal, summary

    # Smoother benchmarks are additive rows (`estimate_type`) and never replace
    # the filtered baseline rows.
    hourly_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    filtered_hourly_benchmark, filtered_daily_subtotal, filtered_summary = _summarize_estimate_type(
        hourly_frame=filtered_hourly,
        reward_frame=filtered_reward,
        estimate_column="latent_fitbit_scale_steps",
        estimate_type="filter",
    )
    hourly_frames.append(filtered_hourly_benchmark)
    daily_frames.append(filtered_daily_subtotal)
    summary_frames.append(filtered_summary)

    if smoother_artifact is not None:
        smoothed_hourly = smoother_artifact.get("hourly_smoothed_frame", pd.DataFrame()).copy()
        smoothed_reward = smoother_artifact.get("reward_smoothed_frame", pd.DataFrame()).copy()
        if not smoothed_hourly.empty and "latent_fitbit_scale_steps_backward" in smoothed_hourly.columns:
            smoothed_hourly_benchmark, smoothed_daily_subtotal, smoothed_summary = _summarize_estimate_type(
                hourly_frame=smoothed_hourly,
                reward_frame=smoothed_reward,
                estimate_column="latent_fitbit_scale_steps_backward",
                estimate_type="smoothed",
            )
            hourly_frames.append(smoothed_hourly_benchmark)
            daily_frames.append(smoothed_daily_subtotal)
            summary_frames.append(smoothed_summary)

    return {
        "benchmark_key": benchmark_spec.key,
        "benchmark_label": benchmark_spec.label,
        "benchmark_version": benchmark_spec.version,
        "hourly_benchmark_frame": pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame(),
        "daily_subtotal_frame": pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        "summary_frame": pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
    }


def benchmark_from_fit_artifact(
    fit_artifact: dict[str, object],
    *,
    model_family: str,
    pooling_mode: str,
    mask_id: str,
    benchmark_spec: BenchmarkSpec,
    smoother_artifact: dict[str, object] | None = None,
) -> dict[str, object]:
    if benchmark_spec.key == "core_masking":
        return _core_masking_benchmark_from_fit_artifact(
            fit_artifact,
            model_family=model_family,
            pooling_mode=pooling_mode,
            mask_id=mask_id,
            benchmark_spec=benchmark_spec,
            smoother_artifact=smoother_artifact,
        )
    raise ValueError(f"Unknown benchmark spec: {benchmark_spec.key}")


def _artifact_dir(root: Path, artifact_type: str, identity: str) -> Path:
    return root / artifact_type / identity


def _artifact_paths(root: Path, artifact_type: str, identity: str) -> dict[str, Path]:
    artifact_dir = _artifact_dir(root, artifact_type, identity)
    return {
        "dir": artifact_dir,
        "pickle": artifact_dir / "artifact.pkl",
        "meta": artifact_dir / "meta.json",
    }


def _save_artifact(root: Path, artifact_type: str, identity: str, payload: object, meta: dict[str, object]) -> dict[str, Path]:
    paths = _artifact_paths(root, artifact_type, identity)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _write_pickle(paths["pickle"], payload)
    _write_json(paths["meta"], meta)
    return paths


def _load_artifact(root: Path, artifact_type: str, identity: str) -> object:
    paths = _artifact_paths(root, artifact_type, identity)
    return _read_pickle(paths["pickle"])


def _artifact_ready(paths: dict[str, Path], force_recompute: bool) -> bool:
    if force_recompute:
        return False
    return paths["pickle"].exists() and paths["meta"].exists()


def _active_benchmark_specs(resolved_config: dict[str, object]) -> list[BenchmarkSpec]:
    raw = resolved_config.get("benchmark_keys")
    if raw is None:
        return list(BENCHMARK_SPECS.values())
    if isinstance(raw, str):
        keys = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        keys = [str(value).strip() for value in raw if str(value).strip()]
    if not keys:
        return list(BENCHMARK_SPECS.values())
    specs: list[BenchmarkSpec] = []
    for key in keys:
        if key not in BENCHMARK_SPECS:
            raise ValueError(f"Unknown benchmark key: {key}")
        specs.append(BENCHMARK_SPECS[key])
    return specs


def _aggregate_benchmark_summary(benchmark_summary: pd.DataFrame) -> pd.DataFrame:
    if benchmark_summary.empty:
        return pd.DataFrame()
    id_columns = [
        column
        for column in ["benchmark_key", "benchmark_label", "model_family", "pooling_mode", "estimate_type"]
        if column in benchmark_summary.columns
    ]
    numeric_columns = [
        column
        for column in benchmark_summary.columns
        if column not in {*id_columns, "mask_id", "benchmark_version"}
        and pd.api.types.is_numeric_dtype(pd.to_numeric(benchmark_summary[column], errors="coerce"))
    ]
    agg_spec: dict[str, str] = {}
    for column in numeric_columns:
        if column.startswith("n_") or column.endswith("_count") or column.endswith("_used"):
            agg_spec[column] = "sum"
        else:
            agg_spec[column] = "mean"
    if not agg_spec:
        return benchmark_summary.loc[:, id_columns].drop_duplicates().reset_index(drop=True)
    return (
        benchmark_summary.groupby(id_columns, as_index=False)
        .agg(agg_spec)
        .sort_values(id_columns)
        .reset_index(drop=True)
    )


def ensure_base_artifact(
    csv_path: Path,
    *,
    cache_root: Path,
    resolved_config: dict[str, object],
    force_recompute: bool,
) -> tuple[dict[str, object], dict[str, object], str]:
    base_identity_payload = {
        "csv_path": str(csv_path.resolve()),
        "csv_size_bytes": int(csv_path.stat().st_size),
        "csv_modified_ns": int(csv_path.stat().st_mtime_ns),
        "analysis_start": resolved_config["analysis_start_utc_iso"],
        "analysis_end": resolved_config["analysis_end_utc_iso"],
        "participant_set": resolved_config["participant_set"],
        "assumed_local_timezone": resolved_config["assumed_local_timezone"],
        "cache_version": resolved_config["cache_version"],
        "artifact_tag": resolved_config["artifact_tag"],
    }
    identity = _sha_identity(base_identity_payload)
    paths = _artifact_paths(cache_root, "base", identity)
    if _artifact_ready(paths, force_recompute):
        return _load_artifact(cache_root, "base", identity), base_identity_payload, identity

    start_utc = pd.Timestamp(resolved_config["analysis_start_utc_iso"]).tz_convert("UTC")
    end_utc = pd.Timestamp(resolved_config["analysis_end_utc_iso"]).tz_convert("UTC")
    lookback_start = start_utc - pd.Timedelta(hours=48)
    summary = compute_participant_summary(csv_path, start_utc=start_utc, end_utc=end_utc)
    summary["eligible_from_flag"] = pd.to_numeric(summary["fitbit_missing_fraction_from_flag"], errors="coerce") < 0.5
    summary["eligible_from_notna"] = pd.to_numeric(summary["fitbit_missing_fraction_from_notna"], errors="coerce") < 0.5
    full_set = summary.loc[summary["eligible_from_flag"], "participantidentifier"].astype(str).tolist()
    small_set = sorted(full_set)[:5]
    cohort_frame = _load_csv_slice(
        csv_path,
        participant_ids=full_set,
        start_utc=lookback_start,
        end_utc=end_utc,
    )
    mismatch = summary.loc[summary["eligible_from_flag"] != summary["eligible_from_notna"], [
        "participantidentifier",
        "fitbit_missing_fraction_from_flag",
        "fitbit_missing_fraction_from_notna",
        "missingness_flag_vs_notna_absdiff",
    ]].reset_index(drop=True)
    payload = {
        "participant_summary": summary,
        "full_set_ids": full_set,
        "small_set_ids": small_set,
        "cohort_frame": cohort_frame,
        "cohort_mismatch": mismatch,
    }
    meta = {
        **base_identity_payload,
        "full_set_n": len(full_set),
        "small_set_n": len(small_set),
        "mismatch_n": int(len(mismatch)),
    }
    _save_artifact(cache_root, "base", identity, payload, meta)
    return payload, base_identity_payload, identity


def build_task_manifest(
    *,
    resolved_config: dict[str, object],
    active_model_families: list[str],
    active_pooling_modes: list[str],
    mask_ids: list[str],
    benchmark_specs: list[BenchmarkSpec],
    smoother_scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fit_scope = str(resolved_config["fit_scope"])
    for model_family in active_model_families:
        for pooling_mode in _pooling_modes_for_model_family(model_family, active_pooling_modes):
            if fit_scope in {"unmasked_only", "both"}:
                rows.append(
                    {
                        "task_id": f"unmasked::{model_family}::{pooling_mode}",
                        "artifact_type": "unmasked_fit",
                        "participant_set": resolved_config["participant_set"],
                        "model_family": model_family,
                        "pooling_mode": pooling_mode,
                        "covariate_mode": resolved_config["covariate_mode"],
                        "masked": False,
                        "mask_replicate": pd.NA,
                        "status": "needs_run",
                        "runtime_seconds": np.nan,
                    }
                )
            if fit_scope in {"masked_only", "both", "benchmarks_from_cache_only"}:
                for mask_id in mask_ids:
                    rows.append(
                        {
                            "task_id": f"masked::{model_family}::{pooling_mode}::{mask_id}",
                            "artifact_type": "masked_fit",
                            "participant_set": resolved_config["participant_set"],
                            "model_family": model_family,
                            "pooling_mode": pooling_mode,
                            "covariate_mode": resolved_config["covariate_mode"],
                            "masked": True,
                            "mask_replicate": mask_id,
                            "status": "needs_run",
                            "runtime_seconds": np.nan,
                        }
                    )
                    if str(smoother_scope) != "none":
                        rows.append(
                            {
                                "task_id": f"smoother::{model_family}::{pooling_mode}::{mask_id}",
                                "artifact_type": "smoother",
                                "participant_set": resolved_config["participant_set"],
                                "model_family": model_family,
                                "pooling_mode": pooling_mode,
                                "covariate_mode": resolved_config["covariate_mode"],
                                "masked": True,
                                "mask_replicate": mask_id,
                                "status": "needs_run",
                                "runtime_seconds": np.nan,
                            }
                        )
                    for benchmark_spec in benchmark_specs:
                        rows.append(
                            {
                                "task_id": f"benchmark::{benchmark_spec.key}::{model_family}::{pooling_mode}::{mask_id}",
                                "artifact_type": "benchmark",
                                "benchmark_key": benchmark_spec.key,
                                "benchmark_label": benchmark_spec.label,
                                "participant_set": resolved_config["participant_set"],
                                "model_family": model_family,
                                "pooling_mode": pooling_mode,
                                "covariate_mode": resolved_config["covariate_mode"],
                                "masked": True,
                                "mask_replicate": mask_id,
                                "status": "needs_run",
                                "runtime_seconds": np.nan,
                            }
                        )
    rows.append(
        {
            "task_id": "derived::all",
            "artifact_type": "derived",
            "participant_set": resolved_config["participant_set"],
            "model_family": ",".join(active_model_families),
            "pooling_mode": ",".join(active_pooling_modes),
            "covariate_mode": resolved_config["covariate_mode"],
            "masked": pd.NA,
            "mask_replicate": pd.NA,
            "status": "needs_run",
            "runtime_seconds": np.nan,
        }
    )
    return pd.DataFrame(rows)


def run_cache_first_pipeline(
    csv_path: Path,
    *,
    resolved_config: dict[str, object],
    cache_paths: dict[str, Path],
    active_model_families: list[str],
    active_pooling_modes: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> AggregatePipelineResult:
    force_recompute = str(resolved_config["cache_mode"]) == "force_recompute"
    fit_scope = str(resolved_config["fit_scope"])
    smoother_scope = _normalize_smoother_scope(resolved_config.get("smoother_scope", "none"))
    smoother_method = _normalize_smoother_method(resolved_config.get("smoother_method", "backward_particle"))
    # Benchmark-only reruns should still derive new benchmark artifacts from saved masked fits.
    allow_fit_compute = fit_scope != "benchmarks_from_cache_only"
    allow_benchmark_compute = True
    allow_smoother_compute = smoother_scope == "compute_missing_only"
    need_unmasked = fit_scope in {"unmasked_only", "both", "benchmarks_from_cache_only"}
    need_masked = fit_scope in {"masked_only", "both", "benchmarks_from_cache_only"}
    need_smoother = smoother_scope != "none"
    if progress_callback is None:
        progress_callback = lambda _message: None

    progress_log_path = cache_paths["logs"] / f"aggregate_progress_{resolved_config['config_hash']}.jsonl"
    progress_manifest_path = cache_paths["logs"] / f"aggregate_progress_manifest_{resolved_config['config_hash']}.csv"
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")

    base_artifact, _, base_identity = ensure_base_artifact(
        csv_path,
        cache_root=cache_paths["root"],
        resolved_config=resolved_config,
        force_recompute=force_recompute,
    )
    participant_ids = list(
        base_artifact["small_set_ids"] if str(resolved_config["participant_set"]) == "small_set" else base_artifact["full_set_ids"]
    )
    mask_ids = [f"mask_rep_{index + 1:02d}" for index in range(int(resolved_config["n_mask_replicates"]))]
    benchmark_specs = _active_benchmark_specs(resolved_config)
    task_manifest = build_task_manifest(
        resolved_config=resolved_config,
        active_model_families=active_model_families,
        active_pooling_modes=active_pooling_modes,
        mask_ids=mask_ids,
        benchmark_specs=benchmark_specs,
        smoother_scope=smoother_scope,
    )
    task_rows = task_manifest.to_dict("records")
    timing_rows: list[dict[str, object]] = []
    task_row_lookup = {str(row["task_id"]): row for row in task_rows}

    def _write_progress_manifest() -> None:
        pd.DataFrame(task_rows).to_csv(progress_manifest_path, index=False)

    def _progress_counts() -> tuple[int, int, int]:
        total = len(task_rows)
        done = sum(1 for row in task_rows if str(row.get("status")) in {"done", "cached"})
        running = sum(1 for row in task_rows if str(row.get("status")) == "running")
        return done, running, total

    def emit_progress(message: str, **extra: object) -> None:
        done, running, total = _progress_counts()
        progress_callback(f"[{done}/{total} done, {running} running] {message}")
        _append_jsonl(
            progress_log_path,
            {
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                "message": str(message),
                "progress_done": done,
                "progress_running": running,
                "progress_total": total,
                **extra,
            },
        )

    def update_task_status(
        task_id: str,
        status: str,
        *,
        runtime_seconds: float | None = None,
    ) -> None:
        row = task_row_lookup.get(str(task_id))
        if row is None:
            return
        row["status"] = str(status)
        if runtime_seconds is not None:
            row["runtime_seconds"] = float(runtime_seconds)
        _write_progress_manifest()

    _write_progress_manifest()
    emit_progress(
        "Aggregate pipeline started",
        participant_set=resolved_config["participant_set"],
        model_families=list(active_model_families),
        pooling_modes=list(active_pooling_modes),
        fit_scope=fit_scope,
        smoother_scope=smoother_scope,
        smoother_method=smoother_method,
    )

    cohort_frame = base_artifact["cohort_frame"].copy()
    cohort_frame = cohort_frame.loc[cohort_frame["PARTICIPANTIDENTIFIER"].astype(str).isin(participant_ids)].copy()
    start_utc = pd.Timestamp(resolved_config["analysis_start_utc_iso"]).tz_convert("UTC")
    end_utc = pd.Timestamp(resolved_config["analysis_end_utc_iso"]).tz_convert("UTC")
    assumed_timezone = str(resolved_config["assumed_local_timezone"])
    stage_configs = [GlobalSearchStageConfig(**row) for row in resolved_config["search_stage_settings"]]
    search_windows = resolved_config["search_windows"]
    default_backward_particles = int(stage_configs[-1].evaluation_particles) if stage_configs else 200
    backward_smoother_particles = int(
        resolved_config.get("backward_smoother_particles", default_backward_particles)
    )
    backward_smoother_trajectories = int(resolved_config.get("backward_smoother_trajectories", 80))
    backward_smoother_seed = int(
        resolved_config.get("backward_smoother_seed", int(resolved_config["global_seed"]) + 70_000)
    )

    mask_payloads: dict[str, dict[str, object]] = {}
    if need_masked and (allow_fit_compute or (need_smoother and allow_smoother_compute)):
        for replicate_index, mask_id in enumerate(mask_ids):
            identity = _sha_identity(
                {
                    "base_identity": base_identity,
                    "mask_regime": resolved_config["mask_regime"],
                    "missing_fraction": resolved_config["missing_fraction"],
                    "mask_seed": resolved_config["mask_seed"],
                    "mask_replicate": int(replicate_index),
                    "participant_ids": participant_ids,
                }
            )
            paths = _artifact_paths(cache_paths["root"], "masks", identity)
            if _artifact_ready(paths, force_recompute):
                mask_payload = _load_artifact(cache_paths["root"], "masks", identity)
            else:
                emit_progress(f"Building mask artifact {mask_id}", task_type="mask", mask_id=mask_id)
                masked_frame, mask_summary = make_masked_cohort_frame(
                    cohort_frame,
                    participant_ids=participant_ids,
                    mask_regime=str(resolved_config["mask_regime"]),
                    missing_fraction=float(resolved_config["missing_fraction"]),
                    mask_seed=int(resolved_config["mask_seed"]),
                    replicate_index=replicate_index,
                )
                mask_payload = {
                    "mask_id": mask_id,
                    "masked_cohort_frame": masked_frame,
                    "mask_summary": mask_summary,
                }
                _save_artifact(
                    cache_paths["root"],
                    "masks",
                    identity,
                    mask_payload,
                    {
                        "mask_id": mask_id,
                        "mask_regime": resolved_config["mask_regime"],
                        "missing_fraction": resolved_config["missing_fraction"],
                        "mask_seed": resolved_config["mask_seed"],
                        "mask_replicate": replicate_index,
                    },
                )
            mask_payloads[mask_id] = mask_payload

    unmasked_fits: dict[tuple[str, str], dict[str, object]] = {}
    masked_fits: dict[tuple[str, str, str], dict[str, object]] = {}
    smoother_artifacts: dict[tuple[str, str, str], dict[str, object]] = {}
    benchmark_artifacts: dict[tuple[str, str, str, str], dict[str, object]] = {}

    for model_family in active_model_families:
        emit_progress(f"Preparing {model_family} data objects", model_family=model_family, stage="prepare_data")
        architecture_key = POMP_FAMILY_TO_ARCHITECTURE.get(model_family)
        if architecture_key is None and model_family != "arma_aic":
            raise ValueError(f"Unknown model_family: {model_family}")
        model_pooling_modes = _pooling_modes_for_model_family(model_family, active_pooling_modes)

        def build_data_map(source_frame: pd.DataFrame) -> dict[str, HourlyStepModelData]:
            data_map: dict[str, HourlyStepModelData] = {}
            for participant_id in participant_ids:
                participant_frame = source_frame.loc[source_frame["PARTICIPANTIDENTIFIER"].astype(str) == str(participant_id)].copy()
                data_map[str(participant_id)] = build_csv_hourly_model_data(
                    participant_frame,
                    participant_id=str(participant_id),
                    start_utc=start_utc,
                    end_utc=end_utc,
                    assumed_timezone=assumed_timezone,
                    covariate_mode=str(resolved_config["covariate_mode"]),
                    architecture_key=architecture_key or "ar1_empirical",
                )
            return data_map

        for pooling_mode in model_pooling_modes:
            if need_unmasked:
                identity = _unmasked_fit_identity(
                    resolved_config=resolved_config,
                    participant_ids=participant_ids,
                    search_windows=search_windows,
                    model_family=model_family,
                    pooling_mode=pooling_mode,
                )
                paths = _artifact_paths(cache_paths["root"], "fits/unmasked", identity)
                unmasked_task_id = f"unmasked::{model_family}::{pooling_mode}"
                if _artifact_ready(paths, force_recompute):
                    artifact = _load_artifact(cache_paths["root"], "fits/unmasked", identity)
                    unmasked_fits[(model_family, pooling_mode)] = artifact
                    update_task_status(unmasked_task_id, "cached")
                    emit_progress(
                        f"Using cached unmasked fit for {model_family} / {pooling_mode}",
                        task_id=unmasked_task_id,
                        model_family=model_family,
                        pooling_mode=pooling_mode,
                        artifact_type="unmasked_fit",
                        status="cached",
                    )
                else:
                    if allow_fit_compute:
                        task_start = perf_counter()
                        update_task_status(unmasked_task_id, "running")
                        emit_progress(
                            f"Running unmasked fit for {model_family} / {pooling_mode}",
                            task_id=unmasked_task_id,
                            model_family=model_family,
                            pooling_mode=pooling_mode,
                            artifact_type="unmasked_fit",
                            status="running",
                        )
                        data_by_participant = build_data_map(cohort_frame)
                        if model_family == "arma_aic":
                            artifact = run_arma_model(
                                data_by_participant,
                                rl=int(resolved_config["RL"]),
                                model_family=model_family,
                                pooling_mode=str(pooling_mode),
                                covariate_mode=str(resolved_config["covariate_mode"]),
                                mask_id=None,
                            )
                        else:
                            artifact = run_panel_pomp_model(
                                data_by_participant,
                                architecture_key=architecture_key,
                                pooling_mode=str(pooling_mode),
                                search_windows=search_windows,
                                stage_configs=stage_configs,
                                global_seed=int(resolved_config["global_seed"]),
                                model_family=model_family,
                                covariate_mode=str(resolved_config["covariate_mode"]),
                                mask_id=None,
                                warm_start_estimates=None,
                                panel_chunk_size=(
                                    None
                                    if resolved_config.get("panel_chunk_size") is None
                                    else int(resolved_config["panel_chunk_size"])
                                ),
                            )
                        total_runtime = float(perf_counter() - task_start)
                        timing_rows.append(
                            {
                                "task_id": f"unmasked::{model_family}::{pooling_mode}",
                                "model_family": model_family,
                                "pooling_mode": pooling_mode,
                                "mask_id": pd.NA,
                                "artifact_type": "unmasked_fit",
                                "runtime_seconds": total_runtime,
                            }
                        )
                        _save_artifact(
                            cache_paths["root"],
                            "fits/unmasked",
                            identity,
                            artifact,
                            {
                                "model_family": model_family,
                                "pooling_mode": pooling_mode,
                                "covariate_mode": resolved_config["covariate_mode"],
                                "runtime_seconds": total_runtime,
                                "fit_identity": identity,
                                "schema_version": AGGREGATE_TASK_CACHE_SCHEMA_VERSION,
                            },
                        )
                        unmasked_fits[(model_family, pooling_mode)] = artifact
                        update_task_status(unmasked_task_id, "done", runtime_seconds=total_runtime)
                        emit_progress(
                            f"Finished unmasked fit for {model_family} / {pooling_mode}",
                            task_id=unmasked_task_id,
                            model_family=model_family,
                            pooling_mode=pooling_mode,
                            artifact_type="unmasked_fit",
                            status="done",
                            runtime_seconds=total_runtime,
                        )

            if need_masked:
                for replicate_index, mask_id in enumerate(mask_ids):
                    mask_payload = mask_payloads.get(mask_id)
                    identity = _masked_fit_identity(
                        resolved_config=resolved_config,
                        participant_ids=participant_ids,
                        search_windows=search_windows,
                        model_family=model_family,
                        pooling_mode=pooling_mode,
                        mask_id=mask_id,
                    )
                    paths = _artifact_paths(cache_paths["root"], "fits/masked", identity)
                    masked_task_id = f"masked::{model_family}::{pooling_mode}::{mask_id}"
                    if _artifact_ready(paths, force_recompute):
                        artifact = _load_artifact(cache_paths["root"], "fits/masked", identity)
                        masked_fits[(model_family, pooling_mode, mask_id)] = artifact
                        update_task_status(masked_task_id, "cached")
                        emit_progress(
                            f"Using cached masked fit for {model_family} / {pooling_mode} / {mask_id}",
                            task_id=masked_task_id,
                            model_family=model_family,
                            pooling_mode=pooling_mode,
                            mask_id=mask_id,
                            artifact_type="masked_fit",
                            status="cached",
                        )
                    elif allow_fit_compute and mask_payload is not None:
                        task_start = perf_counter()
                        update_task_status(masked_task_id, "running")
                        emit_progress(
                            f"Running masked fit for {model_family} / {pooling_mode} / {mask_id}",
                            task_id=masked_task_id,
                            model_family=model_family,
                            pooling_mode=pooling_mode,
                            mask_id=mask_id,
                            artifact_type="masked_fit",
                            status="running",
                        )
                        data_by_participant = build_data_map(mask_payload["masked_cohort_frame"])
                        warm_start = unmasked_fits.get((model_family, pooling_mode), {}).get("participant_estimates")
                        if warm_start is None:
                            unmasked_identity = _unmasked_fit_identity(
                                resolved_config=resolved_config,
                                participant_ids=participant_ids,
                                search_windows=search_windows,
                                model_family=model_family,
                                pooling_mode=pooling_mode,
                            )
                            unmasked_paths = _artifact_paths(cache_paths["root"], "fits/unmasked", unmasked_identity)
                            if _artifact_ready(unmasked_paths, force_recompute=False):
                                cached_unmasked_artifact = _load_artifact(
                                    cache_paths["root"],
                                    "fits/unmasked",
                                    unmasked_identity,
                                )
                                unmasked_fits[(model_family, pooling_mode)] = cached_unmasked_artifact
                                warm_start = cached_unmasked_artifact.get("participant_estimates")
                        if model_family == "arma_aic":
                            artifact = run_arma_model(
                                data_by_participant,
                                rl=int(resolved_config["RL"]),
                                model_family=model_family,
                                pooling_mode=str(pooling_mode),
                                covariate_mode=str(resolved_config["covariate_mode"]),
                                mask_id=mask_id,
                            )
                        else:
                            artifact = run_panel_pomp_model(
                                data_by_participant,
                                architecture_key=architecture_key,
                                pooling_mode=str(pooling_mode),
                                search_windows=search_windows,
                                stage_configs=stage_configs,
                                global_seed=int(resolved_config["global_seed"]) + 5000,
                                model_family=model_family,
                                covariate_mode=str(resolved_config["covariate_mode"]),
                                mask_id=mask_id,
                                warm_start_estimates=warm_start,
                                panel_chunk_size=(
                                    None
                                    if resolved_config.get("panel_chunk_size") is None
                                    else int(resolved_config["panel_chunk_size"])
                                ),
                            )
                        artifact["mask_summary"] = mask_payload["mask_summary"]
                        total_runtime = float(perf_counter() - task_start)
                        timing_rows.append(
                            {
                                "task_id": f"masked::{model_family}::{pooling_mode}::{mask_id}",
                                "model_family": model_family,
                                "pooling_mode": pooling_mode,
                                "mask_id": mask_id,
                                "artifact_type": "masked_fit",
                                "runtime_seconds": total_runtime,
                            }
                        )
                        _save_artifact(
                            cache_paths["root"],
                            "fits/masked",
                            identity,
                            artifact,
                            {
                                "model_family": model_family,
                                "mask_id": mask_id,
                                "pooling_mode": pooling_mode,
                                "runtime_seconds": total_runtime,
                                "fit_identity": identity,
                                "schema_version": AGGREGATE_TASK_CACHE_SCHEMA_VERSION,
                            },
                        )
                        masked_fits[(model_family, pooling_mode, mask_id)] = artifact
                        update_task_status(masked_task_id, "done", runtime_seconds=total_runtime)
                        emit_progress(
                            f"Finished masked fit for {model_family} / {pooling_mode} / {mask_id}",
                            task_id=masked_task_id,
                            model_family=model_family,
                            pooling_mode=pooling_mode,
                            mask_id=mask_id,
                            artifact_type="masked_fit",
                            status="done",
                            runtime_seconds=total_runtime,
                        )

                    smoother_task_id = f"smoother::{model_family}::{pooling_mode}::{mask_id}"
                    smoother_identity: str | None = None
                    smoother_key = (model_family, pooling_mode, mask_id)
                    if need_smoother:
                        supported_smoother, smoother_support_reason = _smoother_supported_for_configuration(
                            model_family=model_family,
                            pooling_mode=str(pooling_mode),
                            architecture_key=architecture_key,
                        )
                        if not supported_smoother:
                            update_task_status(smoother_task_id, "unsupported")
                            emit_progress(
                                (
                                    "Skipping smoother for unsupported configuration "
                                    f"{model_family} / {pooling_mode} / {mask_id}: {smoother_support_reason}"
                                ),
                                task_id=smoother_task_id,
                                model_family=model_family,
                                pooling_mode=pooling_mode,
                                mask_id=mask_id,
                                artifact_type="smoother",
                                status="unsupported",
                                reason=smoother_support_reason,
                            )
                        elif smoother_key not in masked_fits:
                            update_task_status(smoother_task_id, "skipped_missing_fit_cache")
                            emit_progress(
                                f"Skipping smoother for {model_family} / {pooling_mode} / {mask_id} because masked fit cache is missing",
                                task_id=smoother_task_id,
                                model_family=model_family,
                                pooling_mode=pooling_mode,
                                mask_id=mask_id,
                                artifact_type="smoother",
                                status="skipped_missing_fit_cache",
                            )
                        else:
                            smoother_identity = _smoother_identity(
                                source_fit_identity=identity,
                                model_family=model_family,
                                pooling_mode=str(pooling_mode),
                                mask_id=mask_id,
                                smoother_method=smoother_method,
                                backward_smoother_particles=backward_smoother_particles,
                                backward_smoother_trajectories=backward_smoother_trajectories,
                                backward_smoother_seed=backward_smoother_seed,
                            )
                            smoother_paths = _artifact_paths(cache_paths["root"], "smoothers", smoother_identity)
                            if _artifact_ready(smoother_paths, force_recompute):
                                smoother_artifacts[smoother_key] = _load_artifact(
                                    cache_paths["root"],
                                    "smoothers",
                                    smoother_identity,
                                )
                                update_task_status(smoother_task_id, "cached")
                                emit_progress(
                                    f"Using cached smoother for {model_family} / {pooling_mode} / {mask_id}",
                                    task_id=smoother_task_id,
                                    model_family=model_family,
                                    pooling_mode=pooling_mode,
                                    mask_id=mask_id,
                                    artifact_type="smoother",
                                    status="cached",
                                )
                            elif not allow_smoother_compute:
                                update_task_status(smoother_task_id, "skipped_missing_fit_cache")
                                emit_progress(
                                    f"Skipping smoother compute for {model_family} / {pooling_mode} / {mask_id} (smoother_scope={smoother_scope})",
                                    task_id=smoother_task_id,
                                    model_family=model_family,
                                    pooling_mode=pooling_mode,
                                    mask_id=mask_id,
                                    artifact_type="smoother",
                                    status="skipped_missing_fit_cache",
                                )
                            else:
                                smoother_task_start = perf_counter()
                                update_task_status(smoother_task_id, "running")
                                emit_progress(
                                    f"Running smoother for {model_family} / {pooling_mode} / {mask_id}",
                                    task_id=smoother_task_id,
                                    model_family=model_family,
                                    pooling_mode=pooling_mode,
                                    mask_id=mask_id,
                                    artifact_type="smoother",
                                    status="running",
                                )
                                if mask_payload is None:
                                    masked_frame, mask_summary = make_masked_cohort_frame(
                                        cohort_frame,
                                        participant_ids=participant_ids,
                                        mask_regime=str(resolved_config["mask_regime"]),
                                        missing_fraction=float(resolved_config["missing_fraction"]),
                                        mask_seed=int(resolved_config["mask_seed"]),
                                        replicate_index=replicate_index,
                                    )
                                    mask_payload = {
                                        "mask_id": mask_id,
                                        "masked_cohort_frame": masked_frame,
                                        "mask_summary": mask_summary,
                                    }
                                smoother_data_by_participant = build_data_map(mask_payload["masked_cohort_frame"])
                                try:
                                    smoother_artifact = build_backward_smoother_artifact_from_masked_fit(
                                        masked_fits[smoother_key],
                                        data_by_participant=smoother_data_by_participant,
                                        architecture_key=architecture_key,
                                        model_family=model_family,
                                        pooling_mode=str(pooling_mode),
                                        covariate_mode=str(resolved_config["covariate_mode"]),
                                        mask_id=mask_id,
                                        stage_configs=stage_configs,
                                        global_seed=int(resolved_config["global_seed"]),
                                        source_fit_identity=identity,
                                        smoother_method=smoother_method,
                                        backward_smoother_particles=backward_smoother_particles,
                                        backward_smoother_trajectories=backward_smoother_trajectories,
                                        backward_smoother_seed=backward_smoother_seed,
                                    )
                                except Exception as smoother_exc:
                                    update_task_status(smoother_task_id, "failed")
                                    emit_progress(
                                        f"Failed smoother for {model_family} / {pooling_mode} / {mask_id}: {smoother_exc}",
                                        task_id=smoother_task_id,
                                        model_family=model_family,
                                        pooling_mode=pooling_mode,
                                        mask_id=mask_id,
                                        artifact_type="smoother",
                                        status="failed",
                                    )
                                    smoother_artifact = None
                                if smoother_artifact is not None:
                                    smoother_runtime = float(perf_counter() - smoother_task_start)
                                    timing_rows.append(
                                        {
                                            "task_id": smoother_task_id,
                                            "model_family": model_family,
                                            "pooling_mode": pooling_mode,
                                            "mask_id": mask_id,
                                            "artifact_type": "smoother",
                                            "runtime_seconds": smoother_runtime,
                                        }
                                    )
                                    _save_artifact(
                                        cache_paths["root"],
                                        "smoothers",
                                        smoother_identity,
                                        smoother_artifact,
                                        {
                                            "source_fit_identity": identity,
                                            "model_family": model_family,
                                            "pooling_mode": pooling_mode,
                                            "mask_id": mask_id,
                                            "smoother_method": smoother_method,
                                            "backward_smoother_particles": backward_smoother_particles,
                                            "backward_smoother_trajectories": backward_smoother_trajectories,
                                            "backward_smoother_seed": backward_smoother_seed,
                                            "runtime_seconds": smoother_runtime,
                                            "schema_version": SMOOTHER_TASK_CACHE_SCHEMA_VERSION,
                                        },
                                    )
                                    smoother_artifacts[smoother_key] = smoother_artifact
                                    update_task_status(smoother_task_id, "done", runtime_seconds=smoother_runtime)
                                    emit_progress(
                                        f"Finished smoother for {model_family} / {pooling_mode} / {mask_id}",
                                        task_id=smoother_task_id,
                                        model_family=model_family,
                                        pooling_mode=pooling_mode,
                                        mask_id=mask_id,
                                        artifact_type="smoother",
                                        status="done",
                                        runtime_seconds=smoother_runtime,
                                    )

                    for benchmark_spec in benchmark_specs:
                        benchmark_task_id = f"benchmark::{benchmark_spec.key}::{model_family}::{pooling_mode}::{mask_id}"
                        benchmark_identity = _benchmark_identity(
                            masked_fit_identity=identity,
                            model_family=model_family,
                            pooling_mode=pooling_mode,
                            mask_id=mask_id,
                            benchmark_key=benchmark_spec.key,
                            benchmark_version=benchmark_spec.version,
                            smoother_identity=smoother_identity
                            if smoother_identity is not None and smoother_key in smoother_artifacts
                            else None,
                        )
                        benchmark_paths = _artifact_paths(cache_paths["root"], "benchmarks", benchmark_identity)
                        benchmark_key = (model_family, pooling_mode, mask_id, benchmark_spec.key)
                        if _artifact_ready(benchmark_paths, force_recompute):
                            benchmark_artifacts[benchmark_key] = _load_artifact(
                                cache_paths["root"], "benchmarks", benchmark_identity
                            )
                            update_task_status(benchmark_task_id, "cached")
                            emit_progress(
                                f"Using cached benchmark {benchmark_spec.key} for {model_family} / {pooling_mode} / {mask_id}",
                                task_id=benchmark_task_id,
                                model_family=model_family,
                                pooling_mode=pooling_mode,
                                mask_id=mask_id,
                                benchmark_key=benchmark_spec.key,
                                artifact_type="benchmark",
                                status="cached",
                            )
                        elif (model_family, pooling_mode, mask_id) in masked_fits and allow_benchmark_compute:
                            update_task_status(benchmark_task_id, "running")
                            emit_progress(
                                f"Building benchmark {benchmark_spec.key} for {model_family} / {pooling_mode} / {mask_id}",
                                task_id=benchmark_task_id,
                                model_family=model_family,
                                pooling_mode=pooling_mode,
                                mask_id=mask_id,
                                benchmark_key=benchmark_spec.key,
                                artifact_type="benchmark",
                                status="running",
                            )
                            benchmark_payload = benchmark_from_fit_artifact(
                                masked_fits[(model_family, pooling_mode, mask_id)],
                                model_family=model_family,
                                pooling_mode=str(pooling_mode),
                                mask_id=mask_id,
                                benchmark_spec=benchmark_spec,
                                smoother_artifact=smoother_artifacts.get((model_family, pooling_mode, mask_id)),
                            )
                            _save_artifact(
                                cache_paths["root"],
                                "benchmarks",
                                benchmark_identity,
                                benchmark_payload,
                                {
                                    "benchmark_key": benchmark_spec.key,
                                    "benchmark_label": benchmark_spec.label,
                                    "benchmark_version": benchmark_spec.version,
                                    "model_family": model_family,
                                    "mask_id": mask_id,
                                    "pooling_mode": pooling_mode,
                                    "masked_fit_identity": identity,
                                    "smoother_identity": smoother_identity
                                    if smoother_identity is not None and smoother_key in smoother_artifacts
                                    else None,
                                    "schema_version": AGGREGATE_TASK_CACHE_SCHEMA_VERSION,
                                },
                            )
                            benchmark_artifacts[benchmark_key] = benchmark_payload
                            update_task_status(benchmark_task_id, "done")
                            emit_progress(
                                f"Finished benchmark {benchmark_spec.key} for {model_family} / {pooling_mode} / {mask_id}",
                                task_id=benchmark_task_id,
                                model_family=model_family,
                                pooling_mode=pooling_mode,
                                mask_id=mask_id,
                                benchmark_key=benchmark_spec.key,
                                artifact_type="benchmark",
                                status="done",
                            )

    comparison_rows: list[pd.DataFrame] = []
    parameter_rows: list[pd.DataFrame] = []
    timing_summary = pd.DataFrame(timing_rows)
    for (model_family, pooling_mode), artifact in unmasked_fits.items():
        participant_estimates = artifact["participant_estimates"].copy()
        participant_estimates["model_family"] = model_family
        participant_estimates["pooling_mode"] = str(pooling_mode)
        parameter_rows.append(participant_estimates)
    for (_, _, _, _), benchmark_payload in benchmark_artifacts.items():
        comparison_rows.append(benchmark_payload["summary_frame"].copy())
    benchmark_summary = pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()
    parameter_summary = pd.concat(parameter_rows, ignore_index=True) if parameter_rows else pd.DataFrame()
    model_comparison = _aggregate_benchmark_summary(benchmark_summary)

    derived_identity = _sha_identity(
        {
            "kind": "derived",
            "config_hash": resolved_config["config_hash"],
            "active_model_families": active_model_families,
            "active_pooling_modes": active_pooling_modes,
            "participant_ids": participant_ids,
        }
    )
    derived_artifact = {
        "model_comparison": model_comparison,
        "benchmark_summary": benchmark_summary,
        "parameter_summary": parameter_summary,
        "timing_summary": timing_summary,
        "unmasked_fits": unmasked_fits,
        "masked_fits": masked_fits,
        "smoother_artifacts": smoother_artifacts,
        "benchmark_artifacts": benchmark_artifacts,
        "benchmark_specs": [spec.key for spec in benchmark_specs],
        "participant_ids": participant_ids,
    }
    _save_artifact(
        cache_paths["root"],
        "derived",
        derived_identity,
        derived_artifact,
        {
            "config_hash": resolved_config["config_hash"],
            "n_models": len(active_model_families),
            "n_pooling_modes": len(active_pooling_modes),
            "n_participants": len(participant_ids),
        },
    )
    update_task_status("derived::all", "done")
    emit_progress(
        "Aggregate pipeline finished",
        artifact_type="derived",
        status="done",
        n_models=len(active_model_families),
        n_pooling_modes=len(active_pooling_modes),
        n_participants=len(participant_ids),
    )

    for row in task_rows:
        artifact_type = str(row["artifact_type"])
        task_key_unmasked = (str(row["model_family"]), str(row["pooling_mode"]))
        task_key_masked = (str(row["model_family"]), str(row["pooling_mode"]), str(row["mask_replicate"]))
        task_key_smoother = (str(row["model_family"]), str(row["pooling_mode"]), str(row["mask_replicate"]))
        task_key_benchmark = (
            str(row["model_family"]),
            str(row["pooling_mode"]),
            str(row["mask_replicate"]),
            str(row.get("benchmark_key")),
        )
        current_status = str(row.get("status"))
        if current_status not in {"needs_run", "running"}:
            continue
        if artifact_type == "unmasked_fit" and task_key_unmasked in unmasked_fits:
            row["status"] = "done"
        elif artifact_type == "masked_fit" and task_key_masked in masked_fits:
            row["status"] = "done"
        elif artifact_type == "smoother" and task_key_smoother in smoother_artifacts:
            row["status"] = "done"
        elif artifact_type == "benchmark" and task_key_benchmark in benchmark_artifacts:
            row["status"] = "done"
        elif artifact_type == "derived":
            row["status"] = "done"
        elif str(resolved_config["fit_scope"]) == "benchmarks_from_cache_only":
            row["status"] = "skipped_missing_cache"
        else:
            row["status"] = "missing"

    task_manifest = pd.DataFrame(task_rows)
    _write_progress_manifest()
    return AggregatePipelineResult(
        base_artifact=base_artifact,
        task_manifest=task_manifest,
        derived_artifact=derived_artifact,
        timing_summary=timing_summary,
    )
