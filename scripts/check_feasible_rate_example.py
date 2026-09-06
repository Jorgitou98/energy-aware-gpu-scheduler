#!/usr/bin/env python3
"""Recalculate both Figure 2 intervals and optionally replay the simulations."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scientific" / "planning"))
sys.path.insert(0, str(ROOT / "scientific" / "preparation"))
from guard_intervals import compile_guard_intervals
from rate_intervals import direct_intervals, response_values
from window_guard import simulate_window_means, stream_seed


def check_feasible_rate_example(root=ROOT, *, resimulate=False):
    examples = Path(root) / "evidence" / "model_examples"
    source = examples / "feasible_rate_nominal.json"
    data = json.loads(source.read_bytes())
    profile, deadline = data["profile"], data["deadline_s"]
    intervals, segments = direct_intervals(profile, deadline)
    if len(intervals) != 1 or not np.allclose(intervals[0], data["nominal_interval_requests_s"],
                                             rtol=0, atol=1e-9):
        raise ValueError("The nominal interval differs from the Figure 2 calibration")

    evidence = examples / "figure2b"
    simulation = json.loads((evidence / "admission_example_data_1024.json").read_bytes())
    reference = json.loads((evidence / "verification.json").read_bytes())
    protocol = {"N_sim": 1024, "warmup_s": 10, "window_s": 45, "seed": 2026102517,
                "deadline_s": deadline}
    if (any(simulation[key] != value for key, value in protocol.items())
            or simulation["profile"] != dict(profile, configuration=data["configuration"])):
        raise ValueError("The Figure 2(b) simulation does not match the nominal example or protocol")
    nodes = simulation["nodes"]
    rates = np.array([node["rate"] for node in nodes], dtype=float)
    sample_count = protocol["N_sim"]
    if (len(rates) != 9 or not np.all(np.isfinite(rates)) or np.any(np.diff(rates) <= 0)
            or reference["rates"] != len(rates) or reference["simulations_per_rate"] != sample_count
            or reference["total_windows"] != sample_count * len(rates)
            or reference["seed"] != protocol["seed"]):
        raise ValueError("The Figure 2(b) rate grid or sample counts differ from the archived run")

    nominal, latencies, _ = response_values(profile, rates)
    points = np.array(profile["points"])
    variability = np.interp(rates, points[:, 0], points[:, 2])
    quantiles, replay_difference = [], 0.0
    with np.load(evidence / "window_samples_9x1024.npz", allow_pickle=False) as archive:
        expected = {"rates"} | {f"rate_{i}_{field}" for i in range(len(rates))
                                for field in ("window_means_s", "request_counts")}
        if set(archive.files) != expected or not np.array_equal(archive["rates"], rates):
            raise ValueError("The Figure 2(b) sample archive does not match the rate grid")
        for i, (rate, node, latency, c2) in enumerate(zip(rates, nodes, latencies, variability)):
            means, counts = archive[f"rate_{i}_window_means_s"], archive[f"rate_{i}_request_counts"]
            if (means.shape != (sample_count,) or means.dtype.kind != "f"
                    or not np.all(np.isfinite(means) & (means > 0))
                    or counts.shape != means.shape or counts.dtype.kind not in "iu"
                    or np.any(counts <= 0)):
                raise ValueError(f"Invalid Figure 2(b) window samples at rate {rate}")
            if resimulate:
                replayed, replay_counts = simulate_window_means(
                    rate, latency, c2, profile["B"], simulations=sample_count,
                    seed=stream_seed(data["configuration"], rate, protocol["seed"]),
                    warmup_s=protocol["warmup_s"], measured_s=protocol["window_s"], chunk_size=256)
                if (not np.array_equal(counts, replay_counts)
                        or not np.allclose(means, replayed, rtol=0, atol=1e-12)):
                    raise ValueError(f"Figure 2(b) replay differs from the saved windows at rate {rate}")
                replay_difference = max(replay_difference, float(np.max(np.abs(means - replayed))))
                means = replayed
            q05, q50, q95 = np.quantile(means, [.05, .50, .95], method="linear")
            checks = {"L": latency, "C2": c2, "nominal": nominal[i], "q05": q05, "q50": q50,
                      "q95": q95, "mean": means.mean(), "p_pass": np.mean(means <= deadline)}
            for field, value in checks.items():
                if not np.isclose(value, node[field], rtol=0, atol=1e-12):
                    raise ValueError(f"Figure 2(b) {field} differs at rate {rate}")
            if not np.allclose(means[:24], node["display_window_means_s"], rtol=0, atol=1e-12):
                raise ValueError(f"Figure 2(b) sample prefix differs at rate {rate}")
            quantiles.append(float(q95))

    factors = np.array(quantiles) / nominal
    corrected, _ = compile_guard_intervals(list(zip(rates, factors)), profile["B"], deadline,
                                            profile["points"], data["configuration"])
    if (len(corrected) != 1 or not np.allclose(corrected[0], reference["admitted_interval"], rtol=0, atol=1e-9)
            or not np.allclose(intervals[0], reference["nominal_interval"], rtol=0, atol=1e-9)
            or not intervals[0][0] <= corrected[0][0] < corrected[0][1] <= intervals[0][1]):
        raise ValueError("The reproduced Figure 2(b) interval differs from the archived interval")

    return {
        "deadline_s": deadline,
        "nominal_interval_requests_s": intervals[0],
        "admitted_interval_requests_s": corrected[0],
        "simulations_per_rate": sample_count,
        "total_windows": len(rates) * sample_count,
        "seed": protocol["seed"],
        "window_s": protocol["window_s"],
        "warmup_s": protocol["warmup_s"],
        "rates_requests_s": rates.tolist(),
        "p95_window_means_s": quantiles,
        "simulation_replayed": resimulate,
        "max_abs_replayed_mean_difference_s": replay_difference if resimulate else None,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resimulate", action="store_true", help="Replay all 9,216 windows using the saved seed.")
    args = parser.parse_args()
    print(json.dumps(check_feasible_rate_example(resimulate=args.resimulate), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
