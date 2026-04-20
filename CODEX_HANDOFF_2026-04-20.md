# Codex Handoff

Date: 2026-04-20

## Repo

- Project repo: `STATS-531-Final-Project`
- Important sibling code: `../eda`

## Main changes made in this session

- Added cache-first aggregate workflow and RL presets in `computedraft.qmd`
- Added timing benchmark harness in `computetimetest.qmd`
- Added timing helper in `helpers/computetimetest_benchmarks.py`
- Added runtime estimator page in `../eda/pages/10_Compute_Time_Estimator.py`
- Patched estimator logic in `../eda/src/ihs_eda/step_pomp/time_estimator.py`
- Patched benchmark helper to support request replay and inference-only timing
- Patched benchmark/helper timing instrumentation and partial progress files
- Updated `.gitignore` to ignore cache artifacts and generated benchmark outputs

## Current benchmark/estimator behavior

- `computetimetest.qmd` now runs in inference-only mode for timing validation
- It skips post-fit reconstruction during timing runs
- Request validation compares predicted inference time to measured inference time
- The estimator page now treats inference time as the primary estimate
- Reconstruction / artifact cost is shown separately as auxiliary/full cost

## Important implementation details

- Panel search itself is still the shared app/backend path from `../eda`
- The benchmark harness uses `include_postfit_reconstruction=False` in the shared wrapper
- The current panel backend does not automatically split an oversized participant batch
- However, `pypomp` panel code processes units with chunked/scanned logic, so participant count contributes less to peak VRAM than a naive `participants x particles` rule
- Starts and evaluation reps are stronger concurrent-memory multipliers than participant count alone

## Latest timing results that matter

### Request `3b57210e6d`

- AR(1), partial pooled k, unmasked only
- 78 participants
- 78 participants per batch
- 1 start each stage
- 10 particles / 10 eval particles
- inference-only actual time: about 92 seconds

### Request `5643ba2268`

- AR(1), partial pooled k, unmasked only
- 78 participants
- 78 participants per batch
- 2 starts each stage
- 20 particles / 20 eval particles / 2 iterations / 2 eval reps
- inference-only actual time: about 136 seconds

### Request `5f1b1c89b4`

- AR(1), partial pooled k, unmasked only
- 78 participants
- 78 participants per batch
- 4 starts each stage
- 40 particles / 40 eval particles / 4 iterations / 4 eval reps
- inference-only actual time: about 228.5 seconds

## Current uncertainty / open question

- We do not yet have a clean hard memory-breakpoint model for `participants_per_batch = 78`
- JAX appears to preallocate most VRAM, so `nvidia-smi` near-full memory does not directly reveal the true wall
- The next useful work item is better per-stage progress/heartbeat for long single-scenario runs and possibly memory-profiling mode with preallocation disabled for diagnostic runs

## Git status expectation

Cache artifacts are ignored now. The meaningful source changes are in:

- `.gitignore`
- `computedraft.qmd`
- `computetimetest.qmd`
- `helpers/`
- possibly `finalreport.qmd` if you had prior local edits

## How to continue on another machine

1. Push or otherwise copy the repo changes to the second machine.
2. Pull/open the same repo there.
3. Start a new Codex chat.
4. Tell Codex to read this file:
   - `STATS-531-Final-Project/CODEX_HANDOFF_2026-04-20.md`
5. If needed, also mention:
   - benchmark requests live under `cache/computetimetest/requests/`
   - cache outputs are ignored by git and will not transfer unless copied manually

## Suggested opener for the next Codex chat

Read `STATS-531-Final-Project/CODEX_HANDOFF_2026-04-20.md` and continue from there. The benchmark harness is now inference-only, and I want to continue tuning the runtime estimator / overnight RL=2 workflow.
