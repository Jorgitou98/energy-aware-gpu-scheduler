#!/usr/bin/env python3
"""Audit closed planning reruns without solving, then export portable timings.

The full audit uses the frozen input compiler to validate saved deployments;
it never calls a solver. The compact check uses only the Python standard
library and the exported files, with no acquisition data or scientific runtime.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
EXPECTED = [f"unified_{i:03d}" for i in range(112)]
FREQUENCIES = [900, 1200, 1440]
PHASES = ["build_seconds", "greedy_seconds", "beam_seconds", "exact_seconds",
          "validate_convert_seconds"]
TERMINAL = {"optimal", "infeasible_in_calibrated_domain",
            "individual_infeasible_in_calibrated_domain"}
METHODS = {"fixed_frequency": "fixed_frequency_seconds",
           "sequential": "sequential_seconds"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Naive acquisition timestamp")
    return result


def number(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"Invalid numeric value: {label}")
    require(value > 0 if positive else value >= 0, f"Invalid sign: {label}")
    return float(value)


def close(first, second, label, rel=1e-7, absolute=1e-9):
    require(math.isclose(first, second, rel_tol=rel, abs_tol=absolute),
            f"Mismatch {label}: {first!r} != {second!r}")


def safe_path(root, relative):
    path = root / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root.resolve()),
            f"Unsafe relative manifest path: {relative}")
    return path


def verify_files(root, files):
    require(isinstance(files, dict) and bool(files), "Empty file manifest")
    for name, digest in files.items():
        path = safe_path(root, name)
        require(path.is_file(), f"Missing manifest file: {name}")
        require(sha(path) == digest, f"Changed manifest file: {name}")


def write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)


def summary_stats(values):
    values = [number(v, "summary value", positive=True) for v in values]
    return {"n": len(values), "mean_seconds": statistics.fmean(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values), "max_seconds": max(values),
            "lower_bound": False}


def membership(plan):
    members = Counter()
    for group in json.loads(plan["assignment_json"]):
        members[(group["s"], group["g"], group["model"], group["k"])] += group["n"]
    return members


def static_check(root, require_freeze=True):
    protocol = read(root / "protocol.json")
    references = read(root / "references.json")
    require(protocol["schema"] == "planning_full_timing_112_v1", "Unexpected protocol schema")
    require(protocol["cases"] == EXPECTED, "The protocol must contain exactly the original 112 cases")
    require(sorted(references) == EXPECTED, "Reference cohort differs")
    require(protocol["host"] == "montblanc" and protocol["workers"] == 1
            and protocol["threads"] == 1 and protocol["gpu"] is False,
            "Acquisition host/resource policy differs")
    require(protocol["cpu_affinity"] == list(range(32)), "CPU affinity policy differs")
    require(protocol["frequencies_MHz"] == FREQUENCIES and protocol["beam_seconds"] == 10,
            "Fixed-frequency initialization policy differs")
    require(protocol["worker_timeout_s"] == 1800, "Worker budget differs")
    all_cases = []
    for name in ("unified_cases.json", "additional_cases.json"):
        all_cases.extend(read(root / "runtime" / name)["cases"])
    require(len(all_cases) == 112 and sorted(c["case"] for c in all_cases) == EXPECTED,
            "Original workload inputs differ")
    indexed = {case["case"]: case for case in all_cases}
    fixed_statuses = Counter()
    refinements = 0
    for case in EXPECTED:
        reference = references[case]
        require(reference["case"] == indexed[case], f"Changed workload: {case}")
        require(sorted(map(int, reference["fixed_trials"])) == FREQUENCIES,
                f"Reference frequencies differ: {case}")
        feasible = {f for f, p in reference["fixed_trials"].items() if p["status"] == "optimal"}
        require(set(reference["sequential_trials"]) == feasible, f"Reference refinements differ: {case}")
        fixed_statuses.update(p["status"] for p in reference["fixed_trials"].values())
        refinements += len(reference["sequential_trials"])
        require(all(p["status"] == "optimal" for p in reference["sequential_trials"].values()),
                f"Uncertified archived refinement: {case}")
    require(dict(fixed_statuses) == {"optimal": 129, "infeasible_in_calibrated_domain": 207}
            and refinements == 129, "Unexpected archived trial counts")
    fingerprint = None
    if (root / "freeze.json").exists():
        frozen = read(root / "freeze.json")
        verify_files(root, frozen["files"])
        mandatory = {"run_timing.py", "protocol.json", "references.json"}
        mandatory.update(str(p.relative_to(root)) for p in (root / "runtime").rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts)
        require(mandatory <= set(frozen["files"]), "Freeze omits acquisition inputs or sources")
        fingerprint = sha(root / "freeze.json")
    else:
        require(not require_freeze, "Missing freeze.json")
    return protocol, references, fingerprint


def check_certificate(plan, label):
    require(plan["status"] in TERMINAL, f"Unresolved trial: {label}")
    number(plan["seconds"], label + " solver seconds", positive=True)
    if plan["status"] == "optimal":
        require(bool(plan.get("assignment_json")), f"Missing optimal assignment: {label}")
        require(bool(plan.get("history")), f"Missing fractional certificate: {label}")
        terminal = plan["history"][-1]
        require(terminal["status"] == 2, f"Nonoptimal terminal subproblem: {label}")
        require(math.isfinite(terminal["residual"])
                and abs(terminal["residual"]) <= 1e-7 * max(1, plan["utility"]),
                f"Fractional residual exceeds tolerance: {label}")
        close(plan["ratio"], plan["utility"] / plan["power"], label + " objective")
        close(plan["ratio"], terminal["ratio"], label + " terminal objective")
    else:
        require(not plan.get("assignment_json"), f"Assignment on infeasible trial: {label}")


def check_reference(plan, reference, label):
    require(plan["status"] == reference["status"], f"Archived status differs: {label}")
    if reference["ratio"] is not None:
        close(plan["ratio"], reference["ratio"], label + " archived objective")


def trial_index(plans, key, label):
    result = {str(p[key]): p for p in plans}
    require(len(result) == len(plans), f"Duplicate trial: {label}")
    return result


def audit(root, outdir, archive_root):
    protocol, references, fingerprint = static_check(root)
    results = root / "results"
    outcome = read(results / "outcome.json")
    started = read(results / "started.json")
    require(outcome["status"] == "complete" and outcome["exit_code"] == 0,
            "Acquisition is not successfully closed")
    require(outcome["expected_cases"] == outcome["completed_cases"] == 112,
            "Acquisition does not close all 112 cases")
    require(outcome["no_live_worker"] is True, "Acquisition has not confirmed worker exit")
    require(outcome["fingerprint"] == started["fingerprint"] == fingerprint,
            "Campaign fingerprint mismatch")
    outcome_records = {r["case"]: r for r in outcome["records"]}
    require(len(outcome["records"]) == 112 and sorted(outcome_records) == EXPECTED,
            "Duplicate or missing closed cases")
    actual = sorted(p.parent.name for p in results.glob("*/complete.json"))
    require(actual == EXPECTED, "Unexpected completed result cohort")
    require(not list(results.glob("*/timeout.json")), "Censored worker cannot enter full timing summary")
    require(not list(results.glob("*/failure.json")), "Failed worker cannot enter full timing summary")
    sys.path.insert(0, str(root / "runtime" / "planning"))
    import adapter  # Frozen validation/compiler only: no solve call or Gurobi import.
    catalog = read(root / "runtime" / "catalog_unified.json")
    source_manifest = {"freeze.json": fingerprint, "protocol.json": sha(root / "protocol.json"),
                       "references.json": sha(root / "references.json"),
                       "results/outcome.json": sha(results / "outcome.json"),
                       "results/started.json": sha(results / "started.json")}
    archived_source_manifest = {}
    case_audits, per_case, environments = [], [], []
    fixed_statuses, sequential_statuses = Counter(), Counter()
    previous_closed = instant(started["started_at"])
    for case in EXPECTED:
        paths = {name: results / case / (name + ".json")
                 for name in ("complete", "closed", "launch")}
        record, closed, launch = (read(paths[name]) for name in ("complete", "closed", "launch"))
        log = results / (case + ".log")
        require(closed["case"] == launch["case"] == record["case"] == case,
                f"Case identity mismatch: {case}")
        require(closed["exit_code"] == 0 and closed["sha256"] == sha(paths["complete"])
                and closed["log_sha256"] == sha(log), f"Closed hashes/exit mismatch: {case}")
        require(outcome_records[case] == {k: closed[k] for k in ("case", "sha256", "log_sha256")},
                f"Outcome record differs: {case}")
        require(record["schema"] == "planning_full_timing_case_v1" and record["status"] == "complete"
                and record["fingerprint"] == fingerprint, f"Invalid completion record: {case}")
        require(record["reference_sha256"] == references[case]["source_sha256"],
                f"Archived source hash differs: {case}")
        require(record["host"].split(".")[0] == protocol["host"]
                and record["threads"] == 1 and record["gpu"] is False
                and record["cpu_affinity"] == list(range(32)), f"Resource policy differs: {case}")
        require(all(record["numerical_thread_environment"].get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
                f"Numerical thread policy differs: {case}")
        require(record["cuda_visible_devices"] == "", f"CUDA visibility differs: {case}")
        require(record["pid"] == launch["pid"], f"Worker PID differs: {case}")
        # launch.json is written immediately after Popen; the process can start
        # before its launch receipt, so only closure/next-launch order is required.
        launched, finished, closed_at = (instant(launch["started_at"]),
                                       instant(record["finished_at"]), instant(closed["closed_at"]))
        require(previous_closed <= launched <= closed_at and finished <= closed_at,
                f"Workers overlap or timestamps are invalid: {case}")
        require(instant(record["started_at"]) <= finished, f"Negative timer chronology: {case}")
        previous_closed = closed_at
        for path in [*paths.values(), log]:
            source_manifest[str(path.relative_to(root))] = sha(path)
        reference = references[case]
        original = safe_path(archive_root, reference["source"])
        require(sha(original) == reference["source_sha256"], f"Archived case source changed: {case}")
        archived_source_manifest[reference["source"]] = reference["source_sha256"]
        for kind in ("fixed", "sequential"):
            for freq, archived in reference[kind + "_trials"].items():
                source = original.parent / f"{kind}_{freq}.exact.json"
                require(sha(source) == archived["source_sha256"], f"Archived trial source changed: {case}/{kind}_{freq}")
                check_reference(read(source)["result"], archived, f"{case}/{kind}_{freq} original")
                archived_source_manifest[str(source.relative_to(archive_root))] = archived["source_sha256"]
        model = adapter.load_model(reference["case"], catalog)
        require(len(model.options) == reference["option_count"], f"Compiled options differ: {case}")
        fixed = trial_index(record["fixed_trials"], "frequency_MHz", case + " fixed")
        sequential = trial_index(record["sequential_trials"], "seed_frequency_MHz", case + " sequential")
        require(set(fixed) == set(reference["fixed_trials"]) and set(sequential) == set(reference["sequential_trials"]),
                f"Trial coverage differs: {case}")
        feasible = {f: p for f, p in fixed.items() if p.get("assignment_json")}
        require(set(sequential) == set(feasible), f"Not all new fixed optima were refined: {case}")
        expected_attempt_keys = {"fixed_" + f for f in fixed} | {"sequential_" + f for f in sequential}
        require(set(record["attempts"]) == expected_attempt_keys, f"Missing/extra solve attempts: {case}")
        for kind, trials in (("fixed", fixed), ("sequential", sequential)):
            for freq, plan in trials.items():
                label = f"{case}/{kind}_{freq}"
                check_certificate(plan, label)
                check_reference(plan, reference["fixed_trials" if kind == "fixed" else "sequential_trials"][freq], label)
                attempts = record["attempts"][kind + "_" + freq]
                require(bool(attempts), f"No recorded solve attempt: {label}")
                require(all(a["status"] not in TERMINAL for a in attempts[:-1]),
                        f"Repeated certified solve: {label}")
                check_certificate(attempts[-1], label + " terminal attempt")
                require(attempts[-1]["status"] == plan["status"], f"Terminal attempt differs: {label}")
                if plan.get("assignment_json"):
                    model.validate(plan)
                    close(attempts[-1]["ratio"], plan["ratio"], label + " saved conversion")
                    if kind == "fixed":
                        require(all(g["f"] == int(freq) for g in json.loads(plan["assignment_json"])),
                                f"Fixed frequency violated: {label}")
                    else:
                        require(plan["sequential_frozen_membership"] is True and plan["mip_start"] is True,
                                f"Sequential seed/frozen mode missing: {label}")
                        require(membership(plan) == membership(feasible[freq]),
                                f"Sequential membership differs from NEW fixed seed: {label}")
                        require(plan["ratio"] >= feasible[freq]["ratio"] - 1e-8,
                                f"Sequential decreased seeded objective: {label}")
                (fixed_statuses if kind == "fixed" else sequential_statuses).update([plan["status"]])
        for key, trials, reference_key in (("fixed_result", list(feasible.values()), "fixed_ratio"),
                                            ("sequential_result", list(sequential.values()), "sequential_ratio")):
            selected = record[key]
            require(any(selected == p for p in trials), f"Selected result is not a saved trial: {case}/{key}")
            close(selected["ratio"], max(p["ratio"] for p in trials), case + " winner")
            close(selected["ratio"], reference[reference_key], case + " archived winner")
            require(selected["ratio"] <= reference["exact_ratio"] * (1 + 1e-7),
                    f"Result exceeds archived exact reference: {case}/{key}")
        require(record["archival_statuses_and_objectives_match"] is True
                and record["current_certificates_valid"] is True,
                f"Runner did not confirm archive comparisons: {case}")
        times = {key: number(record[key], case + "/" + key, positive=True) for key in
                 ("preparation_seconds", "fixed_frequency_seconds", "sequential_seconds",
                  "sequential_increment_seconds", "worker_seconds_before_output")}
        require(times["preparation_seconds"] <= times["fixed_frequency_seconds"]
                <= times["sequential_seconds"] <= times["worker_seconds_before_output"]
                <= protocol["worker_timeout_s"], f"Impossible method/worker times: {case}")
        close(times["sequential_increment_seconds"], times["sequential_seconds"] - times["fixed_frequency_seconds"],
              case + " cumulative Sequential time", rel=1e-10)
        require([p["frequency_MHz"] for p in record["phases"]] == FREQUENCIES,
                f"Timing phase coverage/order differs: {case}")
        phase_sum = 0.0
        for phase in record["phases"]:
            phase_sum += sum(number(phase[key], case + "/" + key) for key in PHASES)
            solve_sum = sum(number(a["seconds"], case + " attempt seconds", positive=True)
                            for a in record["attempts"]["fixed_" + str(phase["frequency_MHz"])] )
            require(solve_sum <= phase["exact_seconds"] + 1e-6, f"Solver interval exceeds phase: {case}")
        require(times["preparation_seconds"] + phase_sum <= times["fixed_frequency_seconds"] + 1e-6,
                f"Fixed timing phases exceed total: {case}")
        sequential_solve_sum = sum(a["seconds"] for key, attempts in record["attempts"].items()
                                   if key.startswith("sequential_") for a in attempts)
        require(sequential_solve_sum <= times["sequential_increment_seconds"] + 1e-6,
                f"Refinement time exceeds cumulative increment: {case}")
        environment = {k: record[k] for k in ("host", "python", "numpy", "gurobi", "threads", "gpu", "cpu_affinity")}
        if environment not in environments:
            environments.append(environment)
        case_audits.append({"case": case, "record_sha256": closed["sha256"],
                            "closed_sha256": sha(paths["closed"]), "log_sha256": closed["log_sha256"],
                            "reference_sha256": reference["source_sha256"],
                            "fixed_trials": 3, "sequential_trials": len(sequential),
                            "plans_validated": len(feasible) + len(sequential),
                            "archived_statuses_and_objectives_match": True,
                            "new_fixed_seed_memberships_preserved": True})
        per_case.append({"case": case, **times,
                         "fixed_frequency_ratio": record["fixed_result"]["ratio"],
                         "sequential_ratio": record["sequential_result"]["ratio"],
                         "exact_reference_ratio": reference["exact_ratio"],
                         "record_sha256": closed["sha256"],
                         "reference_sha256": reference["source_sha256"]})
    require(previous_closed <= instant(outcome["finished_at"]), "Outcome precedes final worker closure")
    require(dict(fixed_statuses) == {"optimal": 129, "infeasible_in_calibrated_domain": 207}
            and dict(sequential_statuses) == {"optimal": 129}, "Certificate totals differ")
    require(len(environments) == 1, "Inconsistent software/resource environments")
    require(not outdir.exists(), f"Output already exists; preserve it and choose another directory: {outdir}")
    outdir.mkdir(parents=True)
    audit_document = {"schema": "planning_full_timing_audit_v1", "status": "passed",
        "audited_at": stamp(), "fingerprint": fingerprint,
        "auditor_sha256": sha(Path(__file__).resolve()), "cases": 112,
        "fixed_trials": 336, "fixed_statuses": dict(fixed_statuses),
        "sequential_trials": 129, "sequential_statuses": dict(sequential_statuses),
        "optimal_certificates_checked": 258, "saved_plans_revalidated_without_solving": 258,
        "new_fixed_seed_memberships_checked": 129,
        "closed_hashes_verified": True, "sequential_workers_verified": True,
        "archived_reference_files_verified": len(archived_source_manifest),
        "acquisition_outcome_exit_code": 0, "no_live_worker_at_campaign_closure": True,
        "environments": environments, "per_case": case_audits, "source_manifest": source_manifest,
        "archived_source_manifest": archived_source_manifest,
        "limitations": [
            "Infeasible trials expose terminal status from the frozen solver, not a separate proof object; all 207 agree with the archived statuses.",
            "Optimality audit checks saved terminal optimal subproblems and fractional residuals; no solver is rerun.",
            "Resource audit checks saved affinity/thread declarations and frozen runner assertions; this is not a continuous system-wide profiler.",
            "The closed outcome confirms worker exit; this audit does not query the remote host or test its lock.",
            "The Fixed-frequency and Sequential timings come from a separate acquisition from the earlier Figure 5 timings."
        ]}
    write_new(outdir / "audit.json", audit_document)
    summary = {"schema": "planning_full_timing_summary_v1", "status": "audited_complete",
        "cases": 112, "case_ids": EXPECTED, "fingerprint": fingerprint,
        "audit_sha256": sha(outdir / "audit.json"), "protocol_sha256": sha(root / "protocol.json"),
        "acquisition_outcome_sha256": sha(results / "outcome.json"),
        "host": protocol["host"], "threads": 1, "workers": 1, "gpu": False,
        "cpu_affinity": list(range(32)), "worker_timeout_s": 1800,
        "aggregation": "Arithmetic mean over the same 112 complete cases; one fresh process per case.",
        "primary_timer": protocol["primary_timer"],
        "excluded_from_primary_timer": protocol["excluded_from_primary_timer"],
        "cache_policy": protocol["cache_policy"],
        "comparability": "The other Figure 5 timings are historical sums of components. Fixed-frequency and Sequential were remeasured in a separate acquisition on the same host.",
        "methods": {method: summary_stats(row[field] for row in per_case) for method, field in METHODS.items()},
        "sequential_includes_fixed_frequency": True,
        "objective_data_policy": "Figure 5 uses archived objective data and physical results.",
        "per_case": per_case}
    write_new(outdir / "timing_summary.json", summary)
    (outdir / "protocol.json").write_bytes((root / "protocol.json").read_bytes())
    (outdir / "freeze.json").write_bytes((root / "freeze.json").read_bytes())
    (outdir / "audit_summary.py").write_bytes(Path(__file__).resolve().read_bytes())
    manifest = {"schema": "planning_full_timing_plot_input_manifest_v1", "cases": 112,
                "fingerprint": fingerprint, "files": {p.name: sha(p) for p in sorted(outdir.iterdir()) if p.is_file()},
                "raw_required_to_plot": False,
                "verification": "python3 audit_summary.py --verify-compact ."}
    write_new(outdir / "plot_input_manifest.json", manifest)
    verify_compact(outdir)
    return {"status": "passed", "output": str(outdir), "cases": 112, "methods": summary["methods"]}


def verify_compact(directory):
    manifest = read(directory / "plot_input_manifest.json")
    require(manifest["schema"] == "planning_full_timing_plot_input_manifest_v1", "Unexpected portable manifest")
    verify_files(directory, manifest["files"])
    require(set(manifest["files"]) == {"audit.json", "timing_summary.json", "protocol.json",
                                       "freeze.json", "audit_summary.py"}, "Portable file set differs")
    summary, audit_document, protocol = (read(directory / name) for name in
                                        ("timing_summary.json", "audit.json", "protocol.json"))
    frozen = read(directory / "freeze.json")
    require(summary["schema"] == "planning_full_timing_summary_v1" and summary["status"] == "audited_complete"
            and audit_document["schema"] == "planning_full_timing_audit_v1" and audit_document["status"] == "passed",
            "Compact audit has not passed")
    require(summary["cases"] == audit_document["cases"] == manifest["cases"] == 112,
            "Compact cohort size differs")
    require(summary["case_ids"] == protocol["cases"] == EXPECTED, "Compact case list differs")
    require(summary["audit_sha256"] == sha(directory / "audit.json")
            and summary["protocol_sha256"] == sha(directory / "protocol.json")
            and audit_document["auditor_sha256"] == sha(directory / "audit_summary.py"), "Compact provenance differs")
    require(summary["fingerprint"] == audit_document["fingerprint"] == manifest["fingerprint"]
            == sha(directory / "freeze.json"), "Compact acquisition fingerprint differs")
    require(frozen["files"]["protocol.json"] == summary["protocol_sha256"]
            == audit_document["source_manifest"]["protocol.json"]
            and frozen["files"]["references.json"] == audit_document["source_manifest"]["references.json"]
            and audit_document["source_manifest"]["freeze.json"] == summary["fingerprint"],
            "Compact protocol/reference freeze chain differs")
    require(summary["host"] == protocol["host"] == "montblanc"
            and summary["threads"] == protocol["threads"] == 1
            and summary["workers"] == protocol["workers"] == 1
            and summary["gpu"] is protocol["gpu"] is False
            and summary["cpu_affinity"] == protocol["cpu_affinity"] == list(range(32))
            and summary["worker_timeout_s"] == protocol["worker_timeout_s"] == 1800,
            "Compact execution policy differs")
    require(audit_document["fixed_trials"] == 336 and audit_document["sequential_trials"] == 129
            and audit_document["optimal_certificates_checked"] == 258
            and audit_document["new_fixed_seed_memberships_checked"] == 129
            and audit_document["fixed_statuses"] == {"optimal": 129, "infeasible_in_calibrated_domain": 207}
            and audit_document["sequential_statuses"] == {"optimal": 129}
            and audit_document["archived_reference_files_verified"] == 577,
            "Compact certificate counts differ")
    require(audit_document["closed_hashes_verified"] is True
            and audit_document["acquisition_outcome_exit_code"] == 0
            and summary["acquisition_outcome_sha256"] == audit_document["source_manifest"]["results/outcome.json"],
            "Compact acquisition closure differs")
    require(list(METHODS) == list(summary["methods"]), "Compact timing methods differ")
    require([r["case"] for r in summary["per_case"]] == [r["case"] for r in audit_document["per_case"]] == EXPECTED,
            "Compact per-case cohort differs")
    for row, checked in zip(summary["per_case"], audit_document["per_case"]):
        require(row["record_sha256"] == checked["record_sha256"]
                and row["reference_sha256"] == checked["reference_sha256"], "Compact case provenance differs")
        require(0 < row["fixed_frequency_seconds"] <= row["sequential_seconds"], "Compact cumulative times differ")
        close(row["sequential_increment_seconds"], row["sequential_seconds"] - row["fixed_frequency_seconds"],
              row["case"] + " compact cumulative time", rel=1e-10)
    for method, field in METHODS.items():
        expected = summary_stats(row[field] for row in summary["per_case"])
        require(summary["methods"][method] == expected, f"Compact aggregate differs: {method}")
    return {"status": "passed", "cases": 112, "raw_required": False,
            "methods": summary["methods"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE, help="Acquisition directory containing frozen sources and closed results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--static-check", action="store_true", help="Check protocol/inputs only; permit an as-yet absent freeze; create no outputs")
    mode.add_argument("--verify-compact", type=Path, help="Verify exported Figure 5 timing inputs without acquisition data")
    parser.add_argument("--outdir", type=Path, help="New output directory; default: ROOT/analysis/completed_v1")
    parser.add_argument("--archive-root", type=Path, help="Repository containing the archived planning JSON references; default: ROOT/../..")
    args = parser.parse_args()
    if args.verify_compact:
        result = verify_compact(args.verify_compact.resolve())
    elif args.static_check:
        _, _, fingerprint = static_check(args.root.resolve(), require_freeze=False)
        result = {"status": "static_checks_passed", "cases": 112, "fingerprint": fingerprint,
                  "acquisition_results_checked": False, "scientific_outputs_created": False}
    else:
        root = args.root.resolve()
        result = audit(root, args.outdir.resolve() if args.outdir else root / "analysis" / "completed_v1",
                       args.archive_root.resolve() if args.archive_root else root.parent.parent)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
