#!/usr/bin/env python3
"""Prepare, validate, or run an explicitly supplied workload in a new output file."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficient_mig import plan_case, validate_plan


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_plan(document, *, case, admission, input_hashes, result_key=None):
    """Accept a standalone deployment or a record emitted by this CLI."""
    if not isinstance(document, dict):
        raise ValueError("The plan file must contain a JSON object")
    if document.get("assignment_json"):
        if result_key is not None:
            raise ValueError("--plan-result applies only to a planning record, not a standalone deployment")
        return document, None
    results = document.get("results")
    if not isinstance(results, dict):
        raise ValueError("The plan file must contain a deployment or a run_case planning record")
    for field, expected in (("case", case), ("admission", admission)):
        if document.get(field) != expected:
            raise ValueError(f"Saved planning record has a different {field}")
    for field, expected in document.get("input_sha256", {}).items():
        if field in input_hashes and input_hashes[field] != expected:
            raise ValueError(f"Saved planning record has a different {field} input hash")
    primary = {"greedy-c": "greedy_construction", "greedy": "greedy",
               "greedy-beam": "full", "greedy-beam-bounded": "full", "exact": "exact",
               "fixed-frequency": "fixed_frequency", "sequential": "sequential",
               "least-allocated": "least-allocated", "most-allocated": "most-allocated"}
    selected = result_key or primary.get(document.get("method"))
    plan = results.get(selected)
    if not isinstance(plan, dict) or not plan.get("assignment_json"):
        raise ValueError("Selected result has no feasible deployment to validate; "
                         "a preparation or failed search is not a plan")
    return plan, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/inputs/catalog.json")
    parser.add_argument("--cases", type=Path, required=True, help="Explicit workload input; new runs do not reproduce archived paper timings")
    parser.add_argument("--grid", type=Path, default=ROOT / "data/inputs/admission_grid.csv")
    parser.add_argument("--case", required=True)
    parser.add_argument("--admission", choices=("p95", "p99", "nominal"), default="p95")
    parser.add_argument("--method", choices=("prepare", "greedy-c", "greedy", "greedy-beam", "greedy-beam-bounded", "exact", "fixed-frequency", "sequential", "least-allocated", "most-allocated"), default="prepare")
    parser.add_argument("--beam-seconds", type=float, help="Optional time limit for a NEW run; omitted means complete the bounded-width search")
    parser.add_argument("--exact-seconds", type=float, help="Limit each exact solve; omitted means no time limit")
    parser.add_argument("--seed", type=int,
                        help="Required for occupancy policies to draw among candidates tied at the best priority")
    parser.add_argument("--warm-start", action="store_true", help="Initialize joint Exact with Greedy+Beam")
    parser.add_argument("--validate-plan", type=Path, help="Validate a saved plan without running a search")
    parser.add_argument("--plan-result", help="Optional result key within a saved planning record; defaults to its method's primary result")
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args()
    occupancy_method = args.method in ("least-allocated", "most-allocated")
    if occupancy_method and args.seed is None:
        parser.error("--seed is required for least-allocated and most-allocated")
    if not occupancy_method and args.seed is not None:
        parser.error("--seed applies only to least-allocated and most-allocated")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new file")
    if args.validate_plan is not None and (args.method != "prepare" or args.warm_start):
        parser.error("--validate-plan cannot be combined with a search or --warm-start")
    if args.plan_result is not None and args.validate_plan is None:
        parser.error("--plan-result requires --validate-plan")
    if args.warm_start and args.method != "exact":
        parser.error("--warm-start applies only to --method exact")
    if args.method != "prepare" and args.output is None:
        parser.error("New planning runs require --output")
    catalog = json.loads(args.catalog.read_text())
    workloads = json.loads(args.cases.read_text())
    cases = workloads["cases"] if isinstance(workloads, dict) else workloads
    case = next((row for row in cases if row["case"] == args.case), None)
    if case is None:
        parser.error(f"Case not found: {args.case}")
    expected = workloads.get("catalog_sha256") if isinstance(workloads, dict) else None
    if expected is not None and expected != digest(args.catalog):
        parser.error("Catalog hash differs from the workload input binding")
    input_hashes = dict(catalog=digest(args.catalog), cases=digest(args.cases))
    if args.admission != "nominal":
        input_hashes["grid"] = digest(args.grid)
    if args.validate_plan:
        document = json.loads(args.validate_plan.read_text())
        try:
            plan, selected = saved_plan(document, case=args.case, admission=args.admission,
                                       input_hashes=input_hashes, result_key=args.plan_result)
            metrics = validate_plan(case, catalog, plan, admission=args.admission, grid=args.grid)
        except ValueError as error:
            parser.error(str(error))
        result = dict(case=args.case, admission=args.admission, validated=True,
                      plan_sha256=digest(args.validate_plan), selected_result=selected, **metrics)
    else:
        result = plan_case(case, catalog, method=args.method, admission=args.admission,
                           grid=args.grid, beam_seconds=args.beam_seconds,
                           exact_seconds=args.exact_seconds, warm_start=args.warm_start,
                           random_seed=args.seed)
    result["input_sha256"] = input_hashes
    result["scientific_source_sha256"] = {
        path.name: digest(path) for path in sorted((ROOT / "scientific/planning").glob("*.py"))}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
        print(str(args.output))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
