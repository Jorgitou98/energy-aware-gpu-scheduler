#!/usr/bin/env python3
"""Recalculate the archived Residency comparison from archived samples."""

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence" / "validation" / "residency"


def mean_sample(record, field):
    return fmean(sample[field] for sample in record["samples"])


def check(actual, expected):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Residency result differs from the archived comparison: {actual} vs {expected}")


def main():
    source = json.loads((DATA / "source.json").read_bytes())
    records = {}
    for name, identity in source["files"].items():
        raw = (DATA / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
            raise ValueError(f"Residency acquisition has changed: {name}")
        records[Path(name).stem] = json.loads(raw)

    calibration = records["summary"]
    rates = records["protocol"]["rates"]
    residency = {1: [], 2: []}
    for measurement in calibration["individual"]:
        index, replica = measurement["trial"], measurement["config"]
        before, after = records[f"base_{index}"], records[f"base_{index + 1}"]
        loaded = records[f"loaded_c{replica}_{index}"]
        t0, t1 = mean_sample(before, "t"), mean_sample(after, "t")
        weight = (mean_sample(loaded, "t") - t0) / (t1 - t0)
        baseline = (1 - weight) * mean_sample(before, "power_W") + weight * mean_sample(after, "power_W")
        residency[replica].append(mean_sample(loaded, "power_W") - baseline)
    base = mean_sample(records["base_4"], "power_W")
    prediction = (base + sum(fmean(costs) for costs in residency.values())
                  + sum(alpha * rate for alpha, rate in zip(calibration["alpha_J_per_request"], rates)))
    target_path = DATA / "archive_power_targets.json"
    targets = json.loads(target_path.read_bytes())["targets"]
    errors = []
    for target in targets:
        observation = records[target["target"]]
        if observation["rates"] != rates:
            raise ValueError("Concurrent arrival rates differ from the Residency protocol")
        measured = mean_sample(observation, "power_W")
        error = 100 * abs(prediction - measured) / measured
        check(prediction, target["predicted_power_W"])
        check(measured, target["measured_power_W"])
        check(error, target["absolute_error_percent"])
        errors.append(error)
    print(json.dumps({
        "predicted_power_W": prediction,
        "targets": len(errors),
        "mean_error_percent": fmean(errors),
        "median_error_percent": median(errors),
        "maximum_error_percent": max(errors),
    }, indent=2))


if __name__ == "__main__":
    main()
