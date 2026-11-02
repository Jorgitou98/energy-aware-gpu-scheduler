#!/usr/bin/env python3
"""Generate new workloads from the paper's design, without claiming archived runs."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def comparison_cases(seed, draws=14):
    """Draw independent base-rate vectors within each experimental condition."""
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    # task, minimum quality, rate range, tight deadline, relaxed deadline
    roles = (
        ("detection", .45, (8.8, 11.9), .12, .15),
        ("segmentation", .55, (15., 30.), .12, .18),
        ("detection", .35, (7., 12.), .18, .25),
        ("segmentation", .64, (15., 30.), .12, .18),
        ("detection", .35, (7., 12.), .18, .25),
        ("segmentation", .55, (15., 30.), .12, .18),
        ("segmentation", .55, (40., 83.), .18, .25),
    )
    cases = []
    for cap in (145., 165.):
        for regime in ("tight", "relaxed"):
            for multiplier in (1., 1.08):
                rates = [[float(rng.uniform(*role[2])) for role in roles]
                         for _ in range(draws)]
                for draw, base_rates in enumerate(rates):
                    services = [dict(id=sid, task=role[0], qmin=role[1],
                        slo=role[3 if regime == "tight" else 4],
                        lam=multiplier * base_rates[sid], base_rate=base_rates[sid], ca2=1.)
                        for sid, role in enumerate(roles)]
                    cases.append(dict(case=f"generated_{len(cases):03d}",
                        draw=draw, gpus=3, cap_W=cap, response_regime=regime,
                        arrival_multiplier=multiplier, services=services))
    return cases


def scaling_cases(catalog, seeds):
    """Draw nested workloads using the supplied measured batch-1 references."""
    references = {}
    for task, arch in (("detection", "fasterrcnn_resnet50_fpn_v2"),
                       ("segmentation", "deeplabv3_resnet50")):
        profiles = [p for p in catalog["profiles"] if p.get("arch") == arch
                    and p.get("task") == task and p.get("B") == 1
                    and p.get("slices") == 2 and p.get("freq_MHz") == 900]
        if len(profiles) != 1:
            raise ValueError(f"Scaling requires exactly one measured {arch}, batch 1, k=2, "
                             "900 MHz reference. Supply that measurement; no substitute is inferred.")
        p = profiles[0]
        calibration = p.get("calibration", p)
        if not isinstance(calibration, dict):
            raise ValueError(f"Invalid calibration for scaling reference {arch}")
        latency = calibration.get("busy_L_s")
        if (calibration.get("measurement_status", p.get("measurement_status")) != "measured"
                or not isinstance(p.get("physical_config_id"), str)
                or not p["physical_config_id"]
                or any(isinstance(p[key], bool) or not isinstance(p[key], int)
                       for key in ("B", "slices", "freq_MHz"))
                or isinstance(latency, bool) or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency <= 0):
            raise ValueError(f"Scaling reference {arch} must be measured and have "
                             "a valid configuration and positive finite busy_L_s")
        references[task] = dict(physical_config_id=p["physical_config_id"], busy_L_s=latency)
    cases = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        services = []
        for sid in range(128):
            task = "detection" if sid % 2 == 0 else "segmentation"
            qmin = float(rng.choice([.35, .45] if task == "detection" else [.55, .59, .64]))
            beta = float(rng.uniform(.25, .55))
            slo = float(rng.choice([.18, .30, .45] if task == "detection" else [.12, .20, .30]))
            services.append(dict(id=sid, task=task, qmin=qmin, beta=beta, slo=slo,
                reference=references[task], ca2=1., lam=1.2 * beta / references[task]["busy_L_s"]))
        for count in (16, 32, 64, 128):
            cases.append(dict(case=f"generated_s{seed}_n{count}", n=count, gpus=count // 2,
                              cap_W=165., services=deepcopy(services[:count])))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--family", choices=("comparison", "scaling"), default="comparison")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed for workload generation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    catalog = json.loads(args.catalog.read_text())
    try:
        cases = (comparison_cases(args.seed) if args.family == "comparison" else
                 scaling_cases(catalog, range(args.seed, args.seed + 10)))
    except ValueError as error:
        parser.error(str(error))
    document = dict(protocol="new_workloads_from_final_paper_design", seed=args.seed,
        scope="New generated inputs from the selected seed and parameters",
        generation_choices="NumPy default_rng; documented deterministic role and draw order",
        catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest(), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
