from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import gammaln, logsumexp


LOG2PI = float(np.log(2.0 * np.pi))


@dataclass(slots=True)
class BatchedSmootherResult:
    participant_ids: list[str]
    x_particles: np.ndarray
    n_particles: np.ndarray
    smoothed_weights: np.ndarray
    filtered_x_mean: np.ndarray
    filtered_n_mean: np.ndarray
    filtered_x_q10: np.ndarray
    filtered_x_q90: np.ndarray
    filtered_n_q10: np.ndarray
    filtered_n_q90: np.ndarray
    smoothed_x_mean: np.ndarray
    smoothed_n_mean: np.ndarray
    smoothed_x_q10: np.ndarray
    smoothed_x_q90: np.ndarray
    smoothed_n_q10: np.ndarray
    smoothed_n_q90: np.ndarray
    loglik_by_participant: np.ndarray
    mean_ess_by_participant: np.ndarray
    min_ess_by_participant: np.ndarray


def _nb_logpmf_mean_disp(obs: jax.Array, mean: jax.Array, disp: jax.Array) -> jax.Array:
    obs = jnp.maximum(jnp.round(obs), 0.0)
    mean = jnp.maximum(mean, 1e-8)
    disp = jnp.maximum(disp, 1e-6)
    return (
        gammaln(obs + disp)
        - gammaln(disp)
        - gammaln(obs + 1.0)
        + disp * (jnp.log(disp) - jnp.log(disp + mean))
        + obs * (jnp.log(mean) - jnp.log(disp + mean))
    )


def _systematic_resample_indices(weights: jax.Array, uniforms: jax.Array) -> jax.Array:
    n_particles = int(weights.shape[-1])
    positions = (uniforms[:, None] + jnp.arange(n_particles, dtype=weights.dtype)[None, :]) / float(n_particles)
    cumulative = jnp.cumsum(weights, axis=1)
    return jnp.sum(positions[:, :, None] > cumulative[:, None, :], axis=2)


def _normal_logpdf(next_state: jax.Array, prev_mean: jax.Array, sigma: jax.Array) -> jax.Array:
    sigma = jnp.maximum(sigma, 1e-8)
    standardized = (next_state - prev_mean) / sigma
    return -0.5 * (LOG2PI + 2.0 * jnp.log(sigma) + jnp.square(standardized))


@jax.jit
def _run_batched_particle_filter(
    baseline_log_mean: jax.Array,
    observed_steps: jax.Array,
    phi: jax.Array,
    sigma: jax.Array,
    dispersion: jax.Array,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n_units, n_times = baseline_log_mean.shape
    n_particles = int(phi.shape[1])

    x0 = jnp.zeros((n_units, n_particles), dtype=jnp.float32)
    loglik0 = jnp.zeros((n_units,), dtype=jnp.float32)
    ess0 = jnp.zeros((n_units, n_times), dtype=jnp.float32)
    fx0 = jnp.zeros((n_units, n_times), dtype=jnp.float32)
    fn0 = jnp.zeros((n_units, n_times), dtype=jnp.float32)

    def step(carry, t_index):
        key, x_prev, loglik_accum, ess_arr, filtered_x_arr, filtered_n_arr = carry
        key, key_x, key_n, key_r = jax.random.split(key, 4)
        x_noise = jax.random.normal(key_x, shape=(n_units, n_particles), dtype=jnp.float32)
        x_pred = phi * x_prev + sigma * x_noise
        lam = jnp.exp(jnp.clip(baseline_log_mean[:, t_index][:, None] + x_pred, -6.0, 12.0))
        n_pred = jax.random.poisson(key_n, lam, shape=lam.shape).astype(jnp.float32)

        observed_t = observed_steps[:, t_index][:, None]
        observed_mask = ~jnp.isnan(observed_t)
        logweights = jnp.where(
            observed_mask,
            _nb_logpmf_mean_disp(observed_t, n_pred, dispersion),
            0.0,
        )
        max_logw = jnp.max(logweights, axis=1, keepdims=True)
        shifted = jnp.exp(jnp.clip(logweights - max_logw, -700.0, 700.0))
        weight_sum = jnp.sum(shifted, axis=1, keepdims=True)
        safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
        weights = shifted / safe_weight_sum
        loglik_t = jnp.squeeze(max_logw, axis=1) + jnp.log(jnp.squeeze(safe_weight_sum, axis=1)) - jnp.log(float(n_particles))
        x_mean = jnp.sum(weights * x_pred, axis=1)
        n_mean = jnp.sum(weights * n_pred, axis=1)
        ess_t = 1.0 / jnp.sum(jnp.square(weights), axis=1)

        uniforms = jax.random.uniform(key_r, shape=(n_units,), dtype=jnp.float32)
        resample_idx = _systematic_resample_indices(weights, uniforms)
        x_resampled = jnp.take_along_axis(x_pred, resample_idx, axis=1)
        n_resampled = jnp.take_along_axis(n_pred, resample_idx, axis=1)

        ess_arr = ess_arr.at[:, t_index].set(ess_t)
        filtered_x_arr = filtered_x_arr.at[:, t_index].set(x_mean)
        filtered_n_arr = filtered_n_arr.at[:, t_index].set(n_mean)
        return (
            key,
            x_resampled,
            loglik_accum + loglik_t,
            ess_arr,
            filtered_x_arr,
            filtered_n_arr,
        ), (x_resampled, n_resampled, weights)

    (_, _, loglik, ess, filtered_x, filtered_n), outputs = jax.lax.scan(
        step,
        (rng_key, x0, loglik0, ess0, fx0, fn0),
        jnp.arange(n_times),
    )
    x_particles, n_particles_store, filter_weights = outputs
    return (
        jnp.swapaxes(x_particles, 0, 1),
        jnp.swapaxes(n_particles_store, 0, 1),
        jnp.swapaxes(filter_weights, 0, 1),
        filtered_x,
        filtered_n,
        ess,
        loglik,
    )


@jax.jit
def _run_backward_marginal_smoother(
    x_particles: jax.Array,
    phi: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    n_units, n_times, n_particles = x_particles.shape
    uniform_weights = jnp.full((n_units, n_particles), 1.0 / float(n_particles), dtype=jnp.float32)

    def step(w_next, reverse_index):
        t_index = n_times - 2 - reverse_index
        x_t = x_particles[:, t_index, :]
        x_next = x_particles[:, t_index + 1, :]
        transition_logpdf = _normal_logpdf(
            x_next[:, None, :],
            phi[:, :, None] * x_t[:, :, None],
            sigma[:, :, None],
        )
        log_den = logsumexp(transition_logpdf, axis=1, keepdims=True)
        log_terms = jnp.log(jnp.clip(w_next, 1e-30, 1.0))[:, None, :] + transition_logpdf - log_den
        log_w_t = logsumexp(log_terms, axis=2)
        log_w_t = log_w_t - logsumexp(log_w_t, axis=1, keepdims=True)
        w_t = jnp.exp(log_w_t)
        return w_t, w_t

    if n_times <= 1:
        return uniform_weights[:, None, :]

    _, reverse_weights = jax.lax.scan(step, uniform_weights, jnp.arange(n_times - 1))
    reverse_weights = jnp.swapaxes(reverse_weights, 0, 1)
    smoothed_prefix = reverse_weights[:, ::-1, :]
    return jnp.concatenate([smoothed_prefix, uniform_weights[:, None, :]], axis=1)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> list[np.ndarray]:
    flat_values = values.reshape((-1, values.shape[-1]))
    flat_weights = weights.reshape((-1, weights.shape[-1]))
    order = np.argsort(flat_values, axis=1)
    sorted_values = np.take_along_axis(flat_values, order, axis=1)
    sorted_weights = np.take_along_axis(flat_weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    results: list[np.ndarray] = []
    for quantile in quantiles:
        idx = np.argmax(cumulative >= float(quantile), axis=1)
        quantile_values = sorted_values[np.arange(sorted_values.shape[0]), idx]
        results.append(quantile_values.reshape(values.shape[:-1]))
    return results


def run_batched_masked_ar1_smoother(
    *,
    participant_ids: list[str],
    hourly_frames: list[pd.DataFrame],
    phi: np.ndarray,
    sigma: np.ndarray,
    k_fitbit: np.ndarray,
    n_particles: int,
    random_seed: int,
) -> BatchedSmootherResult:
    baseline = np.stack(
        [
            pd.to_numeric(frame["baseline_log_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            for frame in hourly_frames
        ],
        axis=0,
    )
    observed = np.stack(
        [
            pd.to_numeric(frame["fitbit_steps_obs"], errors="coerce").to_numpy(dtype=np.float32)
            for frame in hourly_frames
        ],
        axis=0,
    )
    phi_arr = np.asarray(phi, dtype=np.float32).reshape((-1, 1))
    sigma_arr = np.asarray(sigma, dtype=np.float32).reshape((-1, 1))
    k_arr = np.asarray(k_fitbit, dtype=np.float32).reshape((-1, 1))

    x_particles, n_particles_store, _, filtered_x, filtered_n, ess, loglik = _run_batched_particle_filter(
        jnp.asarray(baseline),
        jnp.asarray(observed),
        jnp.broadcast_to(jnp.asarray(phi_arr), (len(participant_ids), n_particles)),
        jnp.broadcast_to(jnp.asarray(sigma_arr), (len(participant_ids), n_particles)),
        jnp.broadcast_to(jnp.asarray(k_arr), (len(participant_ids), n_particles)),
        jax.random.key(int(random_seed)),
    )
    smoothed_weights = _run_backward_marginal_smoother(
        x_particles,
        jnp.broadcast_to(jnp.asarray(phi_arr), (len(participant_ids), n_particles)),
        jnp.broadcast_to(jnp.asarray(sigma_arr), (len(participant_ids), n_particles)),
    )

    x_particles_np = np.asarray(x_particles, dtype=np.float32)
    n_particles_np = np.asarray(n_particles_store, dtype=np.float32)
    smoothed_weights_np = np.asarray(smoothed_weights, dtype=np.float32)
    filtered_x_np = np.asarray(filtered_x, dtype=np.float32)
    filtered_n_np = np.asarray(filtered_n, dtype=np.float32)
    ess_np = np.asarray(ess, dtype=np.float32)
    loglik_np = np.asarray(loglik, dtype=np.float32)

    filtered_uniform_weights = np.full(x_particles_np.shape, 1.0 / float(n_particles), dtype=np.float32)
    filtered_x_q10, filtered_x_q90 = _weighted_quantiles(x_particles_np, filtered_uniform_weights, (0.10, 0.90))
    filtered_n_q10, filtered_n_q90 = _weighted_quantiles(n_particles_np, filtered_uniform_weights, (0.10, 0.90))
    smoothed_x_mean = np.sum(smoothed_weights_np * x_particles_np, axis=2)
    smoothed_n_mean = np.sum(smoothed_weights_np * n_particles_np, axis=2)
    smoothed_x_q10, smoothed_x_q90 = _weighted_quantiles(x_particles_np, smoothed_weights_np, (0.10, 0.90))
    smoothed_n_q10, smoothed_n_q90 = _weighted_quantiles(n_particles_np, smoothed_weights_np, (0.10, 0.90))

    return BatchedSmootherResult(
        participant_ids=list(map(str, participant_ids)),
        x_particles=x_particles_np,
        n_particles=n_particles_np,
        smoothed_weights=smoothed_weights_np,
        filtered_x_mean=filtered_x_np,
        filtered_n_mean=filtered_n_np,
        filtered_x_q10=filtered_x_q10.astype(np.float32),
        filtered_x_q90=filtered_x_q90.astype(np.float32),
        filtered_n_q10=filtered_n_q10.astype(np.float32),
        filtered_n_q90=filtered_n_q90.astype(np.float32),
        smoothed_x_mean=smoothed_x_mean.astype(np.float32),
        smoothed_n_mean=smoothed_n_mean.astype(np.float32),
        smoothed_x_q10=smoothed_x_q10.astype(np.float32),
        smoothed_x_q90=smoothed_x_q90.astype(np.float32),
        smoothed_n_q10=smoothed_n_q10.astype(np.float32),
        smoothed_n_q90=smoothed_n_q90.astype(np.float32),
        loglik_by_participant=loglik_np,
        mean_ess_by_participant=ess_np.mean(axis=1).astype(np.float32),
        min_ess_by_participant=ess_np.min(axis=1).astype(np.float32),
    )
