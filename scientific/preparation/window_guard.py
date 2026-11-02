#!/usr/bin/env python3
"""CPU-only finite-window Poisson/FIFO-batch guard; no target-response fitting.

Public functions simulate_window_means and quantile_record can be reused for a
new calibration-only rate grid. Quantiles concern one replica's WINDOW MEAN,
not individual request latency and not simultaneous fleet/window coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

SEED = 2026102301
SIMULATIONS = 1024
WARMUP_S = 10.0
MEASURED_S = 45.0
DETERMINISTIC_C2_THRESHOLD = 1e-12


def interpolated_parameters(profile, rate):
    points = sorted(profile['points'], key=lambda p: p['rate_requests_s'])
    rates = np.array([p['rate_requests_s'] for p in points])
    if not rates[0] - 1e-10 <= rate <= rates[-1] + 1e-10:
        raise ValueError('Rate is outside the measured calibration domain')
    latency = float(np.interp(rate, rates, [p['L_s'] for p in points]))
    c2 = float(np.interp(rate, rates, [p['cs2'] for p in points]))
    return latency, c2


def nominal_response(rate, latency, c2, batch):
    rho = rate * latency / batch
    return ((batch - 1) / (2 * rate) + latency
            + (1 / batch + c2) * latency * rho / (2 * (1 - rho))) if rho < 1 else math.inf


def stream_seed(config, rate, master_seed=SEED):
    """Order-independent stream; input observations never affect random draws."""
    digest = hashlib.sha256(f'{master_seed}|{config}|{float(rate).hex()}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def batch_completions(ready_times, service_times):
    """Exact FIFO work-conserving queue, initially empty (last completion=0)."""
    cumulative = np.cumsum(service_times, axis=-1)
    previous = np.concatenate([np.zeros((*cumulative.shape[:-1], 1)), cumulative[..., :-1]], axis=-1)
    offsets = np.maximum.accumulate(np.maximum(ready_times - previous, 0.0), axis=-1)
    return cumulative + offsets


def draw_service(rng, shape, latency, c2):
    if c2 <= DETERMINISTIC_C2_THRESHOLD:
        return np.full(shape, latency)
    return rng.gamma(shape=1 / c2, scale=latency * c2, size=shape)


def means_from_arrivals(arrivals, service_times, batch, warmup_s, horizon_s):
    """Arrivals include surplus exponential draws past horizon, never requests.

    Each B consecutive real arrivals forms a full batch. Only the final partial
    batch is padded at horizon. Warmup requests can share a measured batch.
    Completion after horizon is included for every real measured request.
    """
    count = np.sum(arrivals < horizon_s, axis=1)
    batch_count = (count + batch - 1) // batch
    ready = arrivals[:, batch - 1::batch].copy()
    partial = (count % batch) != 0
    row_indices = np.flatnonzero(partial)
    ready[row_indices, batch_count[row_indices] - 1] = horizon_s
    completions = batch_completions(ready, service_times)
    responses = np.repeat(completions, batch, axis=1) - arrivals
    measured = (arrivals >= warmup_s) & (arrivals < horizon_s)
    denominator = measured.sum(axis=1)
    numerator = np.where(measured, responses, 0.0).sum(axis=1)
    means = np.full(len(arrivals), np.nan)
    np.divide(numerator, denominator, out=means, where=denominator > 0)
    return means, denominator


def simulate_window_means(rate, latency, c2, batch, *, simulations=SIMULATIONS,
                          seed=SEED, warmup_s=WARMUP_S, measured_s=MEASURED_S,
                          chunk_size=256):
    """Return iid finite-window means and measured counts, using bounded memory.

    Interarrival exponentials produce exact Poisson processes. Array length is
    enlarged if needed, so the conservative initial width never truncates a
    path. A single gamma duration is sampled per batch, including final padding.
    Padding contributes no request, response denominator, or utility.
    """
    if (not all(math.isfinite(v) for v in (rate, latency, c2, batch))
            or rate <= 0 or latency <= 0 or c2 < 0 or batch < 1 or int(batch) != batch):
        raise ValueError('Invalid queue parameter')
    if (not all(math.isfinite(v) for v in (warmup_s, measured_s))
            or warmup_s < 0 or measured_s <= 0):
        raise ValueError('Warmup must be nonnegative and measured duration positive and finite')
    if (not isinstance(simulations, (int, np.integer)) or simulations < 1
            or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError('Simulation and chunk counts must be positive integers')
    batch = int(batch)
    rng = np.random.default_rng(seed)
    horizon = warmup_s + measured_s
    expected = rate * horizon
    width = int(math.ceil((expected + 14 * math.sqrt(expected + 1) + 64) / batch)) * batch
    all_means, all_counts = [], []
    for start in range(0, simulations, chunk_size):
        size = min(chunk_size, simulations - start)
        arrivals = np.cumsum(rng.exponential(1 / rate, size=(size, width)), axis=1)
        while np.any(arrivals[:, -1] < horizon):
            extra_width = max(batch, (width // 2 // batch) * batch)
            extra = arrivals[:, -1:] + np.cumsum(rng.exponential(1 / rate, size=(size, extra_width)), axis=1)
            arrivals = np.concatenate([arrivals, extra], axis=1)
        services = draw_service(rng, (size, arrivals.shape[1] // batch), latency, c2)
        means, counts = means_from_arrivals(arrivals, services, batch, warmup_s, horizon)
        all_means.append(means)
        all_counts.append(counts)
    return np.concatenate(all_means), np.concatenate(all_counts)


def quantile_record(profile, rate, *, simulations=SIMULATIONS, master_seed=SEED,
                    return_samples=False):
    """Summarize one simulation; optionally return its window means and counts."""
    config, batch = profile['physical_config_id'], int(profile['B'])
    latency, c2 = interpolated_parameters(profile, rate)
    seed = stream_seed(config, rate, master_seed)
    means, counts = simulate_window_means(rate, latency, c2, batch, simulations=simulations, seed=seed)
    valid = means[np.isfinite(means)]
    if len(valid) < 8:
        raise ValueError('At least eight nonempty simulated windows are required; '
                         'increase the number of simulations or choose a larger rate')
    quantiles = np.quantile(valid, [0.025, 0.5, 0.90, 0.95, 0.975, 0.99], method='linear')
    record = dict(config=config, rate_requests_s=rate, batch=batch, L_s=latency, C2=c2,
                  rho=rate * latency / batch, nominal_stationary_R_s=nominal_response(rate, latency, c2, batch),
                  service_distribution='deterministic' if c2 <= DETERMINISTIC_C2_THRESHOLD else 'gamma',
                  simulations=simulations, seed=seed, empty_measured_windows=int(np.sum(counts == 0)),
                  mean_measured_requests=float(counts.mean()), mean_window_R_s=float(valid.mean()),
                  window_mean_standard_deviation_s=float(valid.std(ddof=1)))
    record.update(dict(zip(['q025_window_mean_s', 'q50_window_mean_s', 'q90_window_mean_s',
                            'q95_window_mean_s', 'q975_window_mean_s', 'q99_window_mean_s'], map(float, quantiles))))
    # Splits expose Monte Carlo variability, not a guarantee/confidence interval.
    split_q95 = [float(np.quantile(part, 0.95)) for part in np.array_split(valid, 8)]
    record.update(q95_eight_split_min_s=min(split_q95), q95_eight_split_max_s=max(split_q95))
    return (record, means, counts) if return_samples else record


def run_checks():
    rng = np.random.default_rng(115108)
    arrivals = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    zero_queue = batch_completions(arrivals, np.full(arrivals.shape, 0.1)) - arrivals
    assert np.allclose(zero_queue, 0.1, atol=1e-12)
    arrivals_b2 = np.array([[0.0, 0.2, 1.0, 1.2, 2.0, 9.0]])
    # Third batch has one real arrival at 2 s, padded at 3 s, response 1.1 s.
    means, counts = means_from_arrivals(arrivals_b2, np.full((1, 3), 0.1), 2, 0, 3)
    assert counts[0] == 5 and abs(means[0] - 0.38) < 1e-12
    ready = np.cumsum(rng.exponential(0.2, size=(7, 1000)), axis=1)
    svc = rng.gamma(2, 0.08, size=ready.shape)
    expected = np.zeros_like(ready)
    for j in range(ready.shape[1]):
        expected[:, j] = np.maximum(ready[:, j], expected[:, j-1] if j else 0) + svc[:, j]
    assert np.allclose(expected, batch_completions(ready, svc), rtol=1e-11, atol=1e-11)
    # Independent long stationary M/G/1 paths: discard first 10k arrivals/path.
    rate, latency, c2 = 20.0, 0.03, 0.5
    ready = np.cumsum(rng.exponential(1/rate, size=(32, 120000)), axis=1)
    svc = draw_service(rng, ready.shape, latency, c2)
    path_means = (batch_completions(ready, svc) - ready)[:, 10000:].mean(axis=1)
    theoretical = latency + rate * latency**2 * (1 + c2) / (2 * (1 - rate * latency))
    standard_error = path_means.std(ddof=1) / math.sqrt(len(path_means))
    tolerance = max(5 * standard_error, theoretical * .01)
    assert abs(path_means.mean() - theoretical) < tolerance
    return dict(zero_queue_response_s=0.1, zero_queue_pass=True,
                final_partial_batch_mean_s=float(means[0]), final_partial_batch_pass=True,
                prefix_max_matches_recursive_queue=True, mg1_pass=True,
                mg1_theoretical_mean_s=theoretical, mg1_simulated_mean_s=float(path_means.mean()),
                mg1_standard_error_s=float(standard_error), mg1_tolerance_s=float(tolerance),
                mg1_paths=32, mg1_arrivals_per_path=120000, mg1_discarded_arrivals_per_path=10000)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--config', help='Physical configuration ID in the measured catalog')
    parser.add_argument('--rate', type=float, help='Requests per second inside the measured domain')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--simulations', type=int, default=SIMULATIONS)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--output', type=Path, help='New JSON file; existing files are refused')
    parser.add_argument('--samples-output', type=Path,
                        help='Optional new NPZ with sampled window means, request counts and metadata')
    args = parser.parse_args()
    if args.checks_only:
        print(json.dumps(run_checks(), indent=2))
        return
    if args.catalog is None or args.config is None or args.rate is None or args.output is None:
        parser.error('--catalog, --config, --rate, and --output are required for a new simulation')
    if args.simulations < 8:
        parser.error('--simulations must be at least 8')
    profiles = {p['physical_config_id']: p for p in json.loads(args.catalog.read_text())['profiles']}
    if args.config not in profiles:
        parser.error('Configuration not present in the measured catalog')
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    if args.samples_output is not None:
        if args.samples_output.resolve() == args.output.resolve():
            parser.error('--output and --samples-output must name different files')
        if args.samples_output.exists():
            parser.error('Samples output already exists; choose a new file')
    result = quantile_record(profiles[args.config], args.rate,
                             simulations=args.simulations, master_seed=args.seed,
                             return_samples=args.samples_output is not None)
    if args.samples_output is not None:
        result, means, counts = result
    result.update(scope='New CPU simulation using the selected seed and parameters',
                  warmup_s=WARMUP_S, measured_s=MEASURED_S,
                  catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest())
    if args.samples_output is not None:
        metadata = dict(schema='window-response-samples-v1',
                        config=result['config'], rate_requests_s=result['rate_requests_s'],
                        batch=result['batch'], L_s=result['L_s'], C2=result['C2'],
                        service_distribution=result['service_distribution'],
                        master_seed=args.seed, seed=result['seed'], simulations=args.simulations,
                        warmup_s=WARMUP_S, measured_s=MEASURED_S, chunk_size=256,
                        units='seconds', quantile_method='linear',
                        quantile_population='Finite entries of window_means_s',
                        empty_window_representation='NaN mean and zero measured requests',
                        valid_windows=int(np.isfinite(means).sum()),
                        rng='numpy.random.default_rng', numpy_version=np.__version__,
                        catalog_sha256=result['catalog_sha256'],
                        simulator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        args.samples_output.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_output.open('xb') as stream:
            np.savez_compressed(stream, window_means_s=means, measured_requests=counts,
                                metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
        result['samples'] = dict(filename=args.samples_output.name,
                                 sha256=hashlib.sha256(args.samples_output.read_bytes()).hexdigest(),
                                 array='window_means_s', sample_count=len(means),
                                 quantile_method='linear')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.output)
    if args.samples_output is not None:
        print(args.samples_output)


if __name__ == '__main__':
    main()
