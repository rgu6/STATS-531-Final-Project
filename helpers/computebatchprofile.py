from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import pandas as pd

from helpers.computedraft_pipeline import (
    POMP_FAMILY_TO_ARCHITECTURE,
    _initial_params_for_pooling,
    _start_box_for_initial_params,
    build_csv_hourly_model_data,
    ensure_base_artifact,
)
from src.ihs_eda.step_pomp import GlobalSearchStageConfig, run_multistage_panel_step_pomp_search


@dataclass(frozen=True, slots=True)
class BatchProfileScenario:
    scenario_id: str
    n_starts: int
    particles: int
    mif_iterations: int
    evaluation_particles: int
    evaluation_pfilter_reps: int


def batch_profile_request_root_from_project(project_root: Path) -> Path:
    return Path(project_root) / "cache" / "computebatchprofile" / "requests"


def batch_profile_request_to_dict(
    *,
    label: str,
    benchmark_tag: str,
    cache_version: str,
    scenario_set: str,
    participant_set: str,
    participant_count: int,
    panel_batch_size: int,
    model_family: str,
    pooling_mode: str,
    covariate_mode: str,
    analysis_start_utc_iso: str,
    analysis_end_utc_iso: str,
    assumed_local_timezone: str,
    random_seed: int,
    warmup_run: bool,
    trace_enabled: bool,
    scenarios: list[BatchProfileScenario],
) -> dict[str, object]:
    settings = {
        "benchmark_tag": str(benchmark_tag),
        "cache_version": str(cache_version),
        "scenario_set": str(scenario_set),
        "participant_set": str(participant_set),
        "participant_count": int(participant_count),
        "panel_batch_size": int(panel_batch_size),
        "model_family": str(model_family),
        "pooling_mode": str(pooling_mode),
        "covariate_mode": str(covariate_mode),
        "analysis_start_utc_iso": str(analysis_start_utc_iso),
        "analysis_end_utc_iso": str(analysis_end_utc_iso),
        "assumed_local_timezone": str(assumed_local_timezone),
        "random_seed": int(random_seed),
        "warmup_run": bool(warmup_run),
        "trace_enabled": bool(trace_enabled),
    }
    request_core = {
        "version": "v1",
        "label": str(label),
        "settings": settings,
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()[:10]
    return {
        **request_core,
        "request_hash": request_hash,
    }


def save_batch_profile_request(project_root: Path, payload: dict[str, object]) -> Path:
    root = batch_profile_request_root_from_project(project_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    label = str(payload.get("label", "profile_grid")).strip().lower().replace(" ", "_")
    request_hash = str(payload.get("request_hash", "request"))
    path = root / f"{timestamp}_{label}_{request_hash}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def resolve_batch_profile_request_path(project_root: Path, request_file: str | Path) -> Path:
    raw = Path(str(request_file))
    if raw.is_absolute():
        return raw.resolve()
    candidate = (Path(project_root) / raw).resolve()
    if candidate.exists():
        return candidate
    return (batch_profile_request_root_from_project(project_root) / raw).resolve()


def load_batch_profile_request(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Batch profile request payload must be a JSON object.")
    return payload


def default_batch_profile_scenarios(scenario_set: str = "factor_grid") -> list[BatchProfileScenario]:
    mode = str(scenario_set).strip().lower()
    if mode == "smoke":
        return [
            BatchProfileScenario(
                scenario_id="smoke_s1_p20_i1",
                n_starts=1,
                particles=20,
                mif_iterations=1,
                evaluation_particles=20,
                evaluation_pfilter_reps=1,
            )
        ]
    if mode == "factor_grid":
        return [
            BatchProfileScenario(
                scenario_id="base_s1_p100_i1",
                n_starts=1,
                particles=100,
                mif_iterations=1,
                evaluation_particles=100,
                evaluation_pfilter_reps=1,
            ),
            BatchProfileScenario(
                scenario_id="starts5_s5_p100_i1",
                n_starts=5,
                particles=100,
                mif_iterations=1,
                evaluation_particles=100,
                evaluation_pfilter_reps=1,
            ),
            BatchProfileScenario(
                scenario_id="iter5_s1_p100_i5",
                n_starts=1,
                particles=100,
                mif_iterations=5,
                evaluation_particles=100,
                evaluation_pfilter_reps=1,
            ),
            BatchProfileScenario(
                scenario_id="particles400_s1_p400_i1",
                n_starts=1,
                particles=400,
                mif_iterations=1,
                evaluation_particles=400,
                evaluation_pfilter_reps=1,
            ),
        ]
    if mode == "full_grid":
        scenarios: list[BatchProfileScenario] = []
        for n_starts in [1, 5]:
            for particles in [100, 400]:
                for mif_iterations in [1, 5]:
                    scenarios.append(
                        BatchProfileScenario(
                            scenario_id=f"s{n_starts}_p{particles}_i{mif_iterations}",
                            n_starts=n_starts,
                            particles=particles,
                            mif_iterations=mif_iterations,
                            evaluation_particles=particles,
                            evaluation_pfilter_reps=1,
                        )
                    )
        return scenarios
    raise ValueError(f"Unknown scenario_set: {scenario_set}")


def _profile_root(project_root: Path, *, cache_version: str, benchmark_tag: str) -> Path:
    return project_root / "cache" / "computebatchprofile" / str(cache_version) / str(benchmark_tag)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _base_resolved_config(
    *,
    cache_version: str,
    benchmark_tag: str,
    analysis_start_utc_iso: str,
    analysis_end_utc_iso: str,
    assumed_local_timezone: str,
    participant_set: str,
) -> dict[str, object]:
    return {
        "analysis_start_utc_iso": str(analysis_start_utc_iso),
        "analysis_end_utc_iso": str(analysis_end_utc_iso),
        "assumed_local_timezone": str(assumed_local_timezone),
        "participant_set": str(participant_set),
        "cache_version": str(cache_version),
        "artifact_tag": f"computebatchprofile::{benchmark_tag}",
    }


def _default_search_windows() -> dict[str, tuple[float, float]]:
    return {
        "phi": (-0.95, 0.95),
        "sigma": (0.05, 2.00),
        "k_fitbit": (0.50, 5.00),
        "log_k_fitbit": (-0.6931471805599453, 1.6094379124341003),
        "mu_log_k_fitbit": (-0.6931471805599453, 1.6094379124341003),
        "log_tau_log_k_fitbit": (-0.6931471805599453, 1.6094379124341003),
    }


def _selected_participant_ids(base_artifact: dict[str, object], *, participant_set: str, participant_count: int) -> list[str]:
    source_ids = (
        base_artifact["small_set_ids"]
        if str(participant_set).strip().lower() == "small_set"
        else base_artifact["full_set_ids"]
    )
    return list(map(str, source_ids[: int(participant_count)]))


def run_batching_profile_suite(
    csv_path: Path,
    *,
    project_root: Path,
    benchmark_tag: str = "gpu_default",
    cache_version: str = "v1",
    scenario_set: str = "factor_grid",
    participant_set: str = "full_set",
    participant_count: int = 1,
    panel_batch_size: int = 1,
    model_family: str = "ar1",
    pooling_mode: str = "unit_specific",
    covariate_mode: str = "none",
    analysis_start_utc_iso: str = "2024-07-01T00:00:00+00:00",
    analysis_end_utc_iso: str = "2024-09-30T23:00:00+00:00",
    assumed_local_timezone: str = "America/New_York",
    random_seed: int = 2026,
    warmup_run: bool = True,
    trace_enabled: bool = True,
    scenarios: list[BatchProfileScenario] | None = None,
) -> dict[str, object]:
    project_root = Path(project_root)
    root = _profile_root(project_root, cache_version=cache_version, benchmark_tag=benchmark_tag)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    base_artifact, _, _ = ensure_base_artifact(
        Path(csv_path),
        cache_root=root / "shared_cache",
        resolved_config=_base_resolved_config(
            cache_version=cache_version,
            benchmark_tag=benchmark_tag,
            analysis_start_utc_iso=analysis_start_utc_iso,
            analysis_end_utc_iso=analysis_end_utc_iso,
            assumed_local_timezone=assumed_local_timezone,
            participant_set=participant_set,
        ),
        force_recompute=False,
    )

    participant_ids = _selected_participant_ids(
        base_artifact,
        participant_set=participant_set,
        participant_count=participant_count,
    )
    if not participant_ids:
        raise ValueError("No participants available for the requested batching profile run.")

    architecture_key = POMP_FAMILY_TO_ARCHITECTURE[str(model_family)]
    start_utc = pd.Timestamp(str(analysis_start_utc_iso)).tz_convert("UTC")
    end_utc = pd.Timestamp(str(analysis_end_utc_iso)).tz_convert("UTC")
    cohort_frame = base_artifact["cohort_frame"]
    data_by_participant = {}
    for participant_id in participant_ids:
        participant_frame = cohort_frame.loc[
            cohort_frame["PARTICIPANTIDENTIFIER"].astype(str) == str(participant_id)
        ].copy()
        data_by_participant[str(participant_id)] = build_csv_hourly_model_data(
            participant_frame,
            participant_id=str(participant_id),
            start_utc=start_utc,
            end_utc=end_utc,
            assumed_timezone=assumed_local_timezone,
            covariate_mode=covariate_mode,
            architecture_key=architecture_key,
        )

    initial_params_by_participant = {
        str(participant_id): _initial_params_for_pooling(
            data_by_participant[str(participant_id)],
            architecture_key=architecture_key,
            pooling_mode=pooling_mode,
        )
        for participant_id in participant_ids
    }
    sample_params = initial_params_by_participant[participant_ids[0]]
    search_windows = _default_search_windows()
    start_box = _start_box_for_initial_params(sample_params, search_windows)
    free_params = list(sample_params.keys())

    scenarios = list(scenarios) if scenarios is not None else default_batch_profile_scenarios(scenario_set)
    scenario_rows: list[dict[str, object]] = []
    stage_profile_frames: list[pd.DataFrame] = []
    trace_rows: list[dict[str, object]] = []

    for scenario in scenarios:
        stage_configs = [
            GlobalSearchStageConfig(
                name="Profile Stage",
                n_starts=int(scenario.n_starts),
                keep_top=1,
                particles=int(scenario.particles),
                mif_iterations=int(scenario.mif_iterations),
                cooling_fraction=0.50,
                rw_sd_scale=0.02,
                evaluation_particles=int(scenario.evaluation_particles),
                evaluation_pfilter_reps=int(scenario.evaluation_pfilter_reps),
                n_fresh_random_starts=0,
                n_jittered_restarts=0,
            )
        ]

        common_kwargs = dict(
            data_by_participant=data_by_participant,
            initial_params_by_participant=initial_params_by_participant,
            free_params=free_params,
            start_box=start_box,
            stage_configs=stage_configs,
            random_seed=int(random_seed),
            ar_order=1,
            missingness_mode="standard",
            k_fitbit_pooling_mode="unit_specific",
            include_current_theta_as_start=True,
            jitter_scale=0.10,
            box_shrink_factor_per_stage=1.00,
            collect_stage_profile=True,
        )

        if warmup_run:
            run_multistage_panel_step_pomp_search(**common_kwargs)

        trace_dir = run_dir / "traces" / str(scenario.scenario_id)
        trace_started = False
        if trace_enabled:
            trace_dir.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(trace_dir))
            trace_started = True

        try:
            result = run_multistage_panel_step_pomp_search(**common_kwargs)
        finally:
            if trace_started:
                jax.profiler.stop_trace()

        stage_profile = result.stage_profile_details.copy()
        if stage_profile.empty:
            raise ValueError(f"No stage profile rows were produced for scenario {scenario.scenario_id}.")
        stage_profile.insert(0, "scenario_id", str(scenario.scenario_id))
        stage_profile.insert(1, "participant_count", int(len(participant_ids)))
        stage_profile.insert(2, "panel_batch_size", int(min(int(panel_batch_size), len(participant_ids))))
        stage_profile_frames.append(stage_profile)

        first_stage = stage_profile.iloc[0]
        scenario_rows.append(
            {
                "scenario_id": str(scenario.scenario_id),
                "participant_count": int(len(participant_ids)),
                "panel_batch_size": int(min(int(panel_batch_size), len(participant_ids))),
                "n_starts": int(scenario.n_starts),
                "particles": int(scenario.particles),
                "mif_iterations": int(scenario.mif_iterations),
                "evaluation_particles": int(scenario.evaluation_particles),
                "evaluation_pfilter_reps": int(scenario.evaluation_pfilter_reps),
                "candidate_build_seconds": float(first_stage["candidate_build_seconds"]),
                "theta_panel_build_seconds": float(first_stage["theta_panel_build_seconds"]),
                "mif_wall_seconds": float(first_stage["mif_wall_seconds"]),
                "pfilter_wall_seconds": float(first_stage["pfilter_wall_seconds"]),
                "results_materialize_seconds": float(first_stage["results_materialize_seconds"]),
                "approx_host_setup_seconds": float(first_stage["approx_host_setup_seconds"]),
                "approx_device_wall_seconds": float(first_stage["approx_device_wall_seconds"]),
                "approx_host_copyback_seconds": float(first_stage["approx_host_copyback_seconds"]),
                "stage_runtime_seconds": float(first_stage["stage_runtime_seconds"]),
                "search_total_runtime_seconds": float(result.total_runtime_seconds),
                "best_loglik": float(result.best_loglik),
                "trace_dir": str(trace_dir) if trace_enabled else None,
            }
        )
        trace_rows.append(
            {
                "scenario_id": str(scenario.scenario_id),
                "trace_dir": str(trace_dir) if trace_enabled else None,
            }
        )

    scenario_summary = pd.DataFrame(scenario_rows)
    if not scenario_summary.empty:
        base_mask = (
            pd.to_numeric(scenario_summary.get("n_starts"), errors="coerce").eq(1)
            & pd.to_numeric(scenario_summary.get("particles"), errors="coerce").eq(100)
            & pd.to_numeric(scenario_summary.get("mif_iterations"), errors="coerce").eq(1)
            & pd.to_numeric(scenario_summary.get("evaluation_particles"), errors="coerce").eq(100)
            & pd.to_numeric(scenario_summary.get("evaluation_pfilter_reps"), errors="coerce").eq(1)
        )
        if bool(base_mask.any()):
            base_row = scenario_summary.loc[base_mask].iloc[0]
            base_time = float(base_row["search_total_runtime_seconds"])
            if base_time > 0:
                scenario_summary["runtime_vs_base"] = pd.to_numeric(
                    scenario_summary["search_total_runtime_seconds"],
                    errors="coerce",
                ) / base_time
                scenario_summary["base_scenario_id"] = str(base_row["scenario_id"])

    stage_profile_details = pd.concat(stage_profile_frames, ignore_index=True) if stage_profile_frames else pd.DataFrame()
    trace_manifest = pd.DataFrame(trace_rows)

    scenario_summary_path = run_dir / "scenario_summary.csv"
    stage_profile_path = run_dir / "stage_profile_details.csv"
    trace_manifest_path = run_dir / "trace_manifest.csv"
    scenario_summary.to_csv(scenario_summary_path, index=False)
    stage_profile_details.to_csv(stage_profile_path, index=False)
    trace_manifest.to_csv(trace_manifest_path, index=False)

    manifest = {
        "run_dir": str(run_dir),
        "benchmark_tag": str(benchmark_tag),
        "cache_version": str(cache_version),
        "scenario_set": str(scenario_set),
        "participant_set": str(participant_set),
        "participant_count": int(len(participant_ids)),
        "participant_ids": list(map(str, participant_ids)),
        "panel_batch_size": int(min(int(panel_batch_size), len(participant_ids))),
        "model_family": str(model_family),
        "pooling_mode": str(pooling_mode),
        "covariate_mode": str(covariate_mode),
        "analysis_start_utc_iso": str(analysis_start_utc_iso),
        "analysis_end_utc_iso": str(analysis_end_utc_iso),
        "assumed_local_timezone": str(assumed_local_timezone),
        "warmup_run": bool(warmup_run),
        "trace_enabled": bool(trace_enabled),
        "search_windows": {name: list(bounds) for name, bounds in search_windows.items()},
        "scenario_summary_path": str(scenario_summary_path),
        "stage_profile_path": str(stage_profile_path),
        "trace_manifest_path": str(trace_manifest_path),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "notes": {
            "approx_device_wall_seconds": "Synchronized wall time spent inside panel.mif and panel.pfilter calls. This is the best available approximation to CUDA work from Python timing.",
            "approx_host_setup_seconds": "Host-side candidate generation, theta assembly, and PanelPomp object construction before the batched GPU calls.",
            "approx_host_copyback_seconds": "Host-side results materialization after the batched GPU calls.",
            "trace_files": "When trace_enabled=true, each scenario writes a JAX profiler trace directory for exact timeline inspection in TensorBoard or Chrome trace tools.",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "scenario_summary": scenario_summary,
        "stage_profile_details": stage_profile_details,
        "trace_manifest": trace_manifest,
    }
