#!/usr/bin/env python3
"""Replay Figure 6 with the archived occupancy experiment and compare results."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POLICIES = ("LeastAllocated", "MostAllocated")


def checked_files(directory, entries):
    for relative, expected in entries.items():
        path = directory / relative
        actual = sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"Replay input checksum differs: {relative}")


def read_json(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Check one workload instead of all 112")
    args = parser.parse_args()

    manifest = read_json(HERE / "replay_manifest.json")
    checked_files(HERE, manifest["frozen_files"])
    checked_files(ROOT, manifest["repository_files"])

    catalog = read_json(ROOT / "data/inputs/catalog.json")
    cases = {case["case"]: case for case in read_json(ROOT / "results/planning/inputs.json")["cases"]}
    publication = read_json(ROOT / "results/occupancy/publication_data.json")
    archived = {row["case"]: row for row in publication["rows"]}
    master_seed = manifest["master_seed"]
    if set(cases) != set(archived) or len(cases) != 112 or publication["master_seed"] != master_seed:
        raise ValueError("Expected the complete archived Figure 6 cohort and its seed")
    if args.case is not None and args.case not in cases:
        parser.error(f"Unknown workload: {args.case}")

    sys.path.insert(0, str(HERE / "frozen/planning"))
    from adapter import load_model
    from score_construction import plan

    selected = [args.case] if args.case is not None else sorted(cases)
    counts = {policy: {"feasible": 0, "construction_failed": 0} for policy in POLICIES}
    for case_id in selected:
        model = load_model(cases[case_id], catalog)
        seed = int.from_bytes(sha256(f"{master_seed}|{case_id}".encode()).digest()[:8], "big")
        for policy in POLICIES:
            expected = archived[case_id]["results"][policy]
            result = plan(model, policy, seed=seed)
            validated = bool(result.get("assignment_json"))
            if validated:
                model.validate(result)
            ratio = result["ratio"] if validated else None
            same_ratio = (ratio is None and expected["ratio"] is None) or (
                ratio is not None and expected["ratio"] is not None
                and math.isclose(ratio, expected["ratio"], rel_tol=0, abs_tol=1e-12))
            if (result["status"] != expected["status"] or validated != expected["validated"]
                    or not same_ratio):
                raise ValueError(f"Archived outcome differs: {case_id} {policy}")
            counts[policy][result["status"]] += 1

    print(json.dumps({"status": "passed", "master_seed": master_seed,
                      "workloads": len(selected), "policy_results": counts}, indent=2))


if __name__ == "__main__":
    main()
