# Post-Fit Smoother Proposal

## Goal

Add a post-fit smoothing stage that:

1. reuses the parameters already learned by the RL=2 model fits,
2. performs a forward re-filter pass with those fixed parameters,
3. performs a backward smoothing pass using the stored forward-pass history,
4. caches the smoother outputs separately from the fit artifacts, so changing smoother settings does not force refitting the model.

This proposal is about system design only. It does not propose changing the mathematical fit itself.

## Current State Of The Code

### 1. Parameter fitting is already cached correctly

The aggregate fit pipeline in [helpers/computedraft_pipeline.py](./helpers/computedraft_pipeline.py) caches each fit artifact under:

- `cache/computedraft/<cache_version>/<artifact_tag>/fits/unmasked/<fit_identity>/`
- `cache/computedraft/<cache_version>/<artifact_tag>/fits/masked/<fit_identity>/`

The cache identity already includes the inputs that matter for fitted parameters:

- participant set
- analysis window
- model family
- pooling mode
- mask regime and mask id
- search stage settings
- search windows
- seed

So the fitted parameters are already safely reusable.

### 2. The aggregate panel pipeline already does a post-fit per-participant rerun

After a panel fit finishes, `run_panel_pomp_model(...)` currently loops over participants and calls `run_step_pomp_if2(...)` with:

- `free_params=[]`
- `mif_iterations=0`
- participant-specific fitted parameters from `participant_estimates`

This is already the correct structural idea for a post-fit reconstruction pass. The important current detail is that it explicitly sets:

- `backward_smoother_enabled=False`

So today the aggregate pipeline only persists filtered reconstructions, not smoothed ones.

### 3. There is already smoother code in `model.py`

`eda/src/ihs_eda/step_pomp/model.py` already contains:

- an approximate two-sided smoother: `_build_approximate_smoothed_frame(...)`
- a backward particle smoother: `_run_backward_particle_smoother(...)`
- fallback logic for unsupported architectures: `_build_backward_fallback_frame(...)`

`run_step_pomp_if2(...)` already returns all of these:

- `smoothed_frame`
- `smoothed_reward_frame`
- `backward_smoothed_frame`
- `backward_smoothed_reward_frame`
- `backward_smoother_summary`

So the core algorithmic pieces are not missing. The missing part is orchestration, persistence, and cache separation.

### 4. Current limitations of the existing smoother implementation

The backward particle smoother is currently only available for:

- AR(1)
- no seasonal lag
- `missingness_mode="standard"`

For AR(2), seasonal, or explicit wear-missingness models, the code already falls back to a labeled non-backward smoother path.

This is acceptable for the immediate RL=2 use case, because the current RL=2 run is:

- `model_family="ar1"`
- `masked_only`
- standard missingness

## Systems Involved

### `eda/src/ihs_eda/step_pomp/model.py`

This is the single-participant execution layer.

Relevant responsibilities:

- run fixed-parameter filter passes
- produce filtered latent summaries
- produce approximate smoothed summaries
- produce backward particle smoother summaries

This is the right place for the actual smoothing mechanics.

### `eda/src/ihs_eda/step_pomp/panel_search.py`

This is the panel fitting layer.

Relevant responsibilities:

- run multi-start panel IF2 search
- return participant-level fitted parameters
- return shared estimates
- return stage diagnostics

This is not where smoothing should live. Smoothing should consume the output of this system, not be embedded inside the panel search itself.

### `helpers/computedraft_pipeline.py`

This is the orchestration and cache system for aggregate report runs.

Relevant responsibilities:

- build fit identities
- load and save fit artifacts
- emit progress logs and manifest rows
- run masked benchmarks
- build derived report artifacts

This is the correct place to add a new cached "post-fit smoother" stage.

### `eda/src/ihs_eda/step_pomp/evaluation.py`

This is the benchmarking layer.

The existing benchmark helpers are already close to what we need because they accept configurable estimate columns. That means we can benchmark:

- filtered predictions
- approximate smoothed predictions
- backward-smoothed predictions

without inventing a second benchmarking framework.

### `computedraft.qmd` and `finalreport.qmd`

These are consumers.

They should not own the smoothing logic. They should only request cached artifacts and display tables/plots.

## Design Principle

The smoother must be a separate cached artifact that depends on a completed fit artifact.

That means:

- changing smoother particles or trajectories should not invalidate fitted parameters
- rerendering the QMD should not refit models just to regenerate smoother outputs
- a failed smoother run should be restartable without touching the fit cache

In short: fit cache and smoother cache must be separate layers.

## Recommended Architecture

### 1. Add a new cached artifact type: `smoother`

Recommended cache root:

- `cache/computedraft/<cache_version>/<artifact_tag>/smoothers/<smoother_identity>/`

Recommended identity inputs:

- `kind="smoother"`
- `source_fit_identity`
- `model_family`
- `pooling_mode`
- `mask_id`
- `smoother_method`
- `backward_smoother_particles`
- `backward_smoother_trajectories`
- `backward_smoother_seed`
- `schema_version`

The crucial dependency is `source_fit_identity`. That is what makes the smoother reuse the exact fitted parameters already cached.

### 2. Source parameters from `participant_estimates`, not from `shared_estimates`

This matters especially for partial pooling.

The safer source of truth is:

- `artifact["participant_estimates"]`

not:

- `artifact["shared_estimates"]`

Reason:

- each participant row already contains the full parameter vector needed for the post-fit rerun
- this avoids depending on separate shared-field interpretation
- it is robust even if shared hyperparameter reporting has a bug

For partial pooling, that means the smoother rerun should use the participant row fields such as:

- `log_k_fitbit`
- `mu_log_k_fitbit`
- `log_tau_log_k_fitbit`

if those are present in the participant estimates.

### 3. Keep smoothing as a post-fit replay, not a new fit

The smoother stage should not optimize anything.

It should:

1. load the cached fit artifact,
2. extract participant parameter rows,
3. rebuild each participant data object,
4. rerun a fixed-parameter forward pass,
5. run backward smoothing,
6. persist the smoother outputs.

That makes the smoother deterministic given:

- the fit artifact
- the smoother configuration
- the smoother seed

### 4. Recommended artifact contents

The smoother artifact should be lighter than a full fit artifact. It should store what the report and benchmarks need, not the full internal particle cloud.

Recommended contents:

- `kind`
- `source_fit_identity`
- `participant_estimates_snapshot`
- `hourly_smoothed_frame`
- `reward_smoothed_frame`
- `imputation_smoothed_frame`
- `backward_smoother_summary`
- `timing_frame`

Optional:

- `hourly_filtered_frame` only if we want direct filtered-vs-smoothed comparisons from the smoother artifact alone

Not recommended to cache:

- full particle history
- raw simulated trajectories
- the `Pomp` object

Those will make artifacts much larger without helping the report.

## Two Implementation Paths

### Option A: Conservative integration using the existing `run_step_pomp_if2(...)`

Approach:

- keep using the current per-participant post-fit rerun path
- flip `backward_smoother_enabled=True`
- persist smoothed and backward-smoothed frames into a new smoother artifact

Pros:

- minimal new model code
- lowest implementation risk
- uses code paths already exercised in `model.py`

Cons:

- it still performs some work we may not need for a smoother-only job
- `run_step_pomp_if2(...)` also does simulation and approximate smoothing
- the backward smoother internally reruns its own NumPy particle filter because `pypomp` does not expose the full filtered particle cloud

This means Option A is reliable but not maximally efficient.

### Option B: Dedicated smoother-only runner

Approach:

- add a new single-participant entry point such as `run_step_pomp_postfit_smoother(...)`
- skip IF2 entirely
- skip simulation entirely
- run exactly:
  1. forward particle filter with fixed parameters
  2. backward FFBSi smoother

Pros:

- matches the intended "forward pass then backward pass" design directly
- avoids extra simulation overhead
- cleaner separation between fitting and smoothing

Cons:

- more code to add
- more testing surface

## Recommendation

For the first implementation, I recommend Option A at the orchestration level, but with a clear interface boundary so it can be upgraded to Option B later without changing the cache contract.

Reason:

- the smoother logic already exists and is close to usable
- the risky part is not the math, it is the pipeline and cache wiring
- the user-facing need is "reuse fitted parameters and cache the smoother", which Option A satisfies

If runtime becomes a problem, we can refactor the execution engine behind the same artifact format.

## Proposed Aggregate Pipeline Changes

### New stage after fit, before or alongside benchmark derivation

For each cached fit artifact:

1. check whether a smoother artifact already exists
2. if yes, load it
3. if no, build it from the fit artifact
4. then optionally derive smoother-specific benchmarks from it

Suggested task ids:

- `smoother::<model_family>::<pooling_mode>::<mask_id>`

This matches the current manifest style and keeps progress visible from the terminal.

### Keep current fit artifacts unchanged

Do not rewrite the fit artifact format just to add smoothing.

That would create unnecessary cache invalidation and make reruns more fragile.

Instead:

- fit artifacts remain the source of fitted parameters
- smoother artifacts remain downstream products

### Derived summary handling

There are two reasonable ways to surface smoothed outputs:

1. keep a separate smoother-derived artifact
2. extend the existing `derived` artifact to include smoother summaries when available

I recommend:

- keep the smoother cache as its own artifact type
- allow the `derived` artifact to reference or summarize it

That preserves the clean layering:

- fits
- smoothers
- benchmarks / derived summaries

## Benchmarking Proposal

The current benchmark code already supports evaluating alternate estimate columns.

That means we can compare:

- filtered latent predictions
- approximate smoothed latent predictions
- backward-smoothed latent predictions

on the same masked hourly and held-out subtotal benchmarks.

Suggested benchmark naming:

- `filtered`
- `smoothed_approx`
- `smoothed_backward`

This should be stored as an extra label or estimate type, not as a separate benchmark framework.

## Report Output Proposal

The smoother outputs should support:

- hourly imputation plots
- 24-hour reward reconstruction plots
- filtered vs smoothed overlay plots
- masked benchmark tables comparing filtered and smoothed outputs

The report should not trigger recomputation directly. It should read cached smoother artifacts the same way it already reads cached fits.

## Important Constraints

### 1. Architecture support

The current backward particle smoother is implemented only for:

- AR(1)
- non-seasonal
- standard missingness

For other model families, the pipeline should either:

- use the approximate smoother path and label it explicitly, or
- skip smoothing with a clear method summary

### 2. Partial pooling robustness

The smoother should use the participant parameter rows actually used for post-fit filtering.

It should not reconstruct participant parameter vectors indirectly from shared summaries.

### 3. Cache separation

The smoother must not change fit identities.

This is the main requirement if we want previously fitted RL=2 parameters to remain reusable.

## Suggested Minimal Configuration Surface

Not implementing yet, but these are the likely controls we will eventually want:

- `smoother_scope`
  - `none`
  - `from_cache_only`
  - `compute_missing_only`

- `smoother_method`
  - `backward_particle`
  - `approximate_only`
  - `auto`

- `backward_smoother_particles`
- `backward_smoother_trajectories`
- `backward_smoother_seed`

- `smoother_benchmark_mode`
  - `filtered_only`
  - `smoothed_only`
  - `compare_filtered_and_smoothed`

## Open Questions For Discussion

1. Do we want smoothing only for the masked RL=2 runs, or also for future unmasked fits?
2. Do we want the report to compare filtered and smoothed benchmarks side by side, or replace filtered metrics with smoothed ones?
3. For the first implementation, is the conservative reuse of `run_step_pomp_if2(...)` acceptable, or do we want a dedicated smoother-only runner immediately?
4. What smoother particle count and trajectory count are acceptable for runtime?
5. Do we want to store full smoothed hourly frames, or only reward-level summaries plus imputation summaries?

## Recommended Next Step

Once the current RL=2 run finishes, the next implementation should be:

1. define the smoother artifact identity and cache location,
2. add a smoother-from-fit stage in `computedraft_pipeline.py`,
3. have it consume cached `participant_estimates`,
4. persist backward-smoothed hourly and reward outputs,
5. add optional benchmark comparison against the existing filtered outputs.

That plan preserves the existing fit cache and gives a clean path to richer smoothing later.
