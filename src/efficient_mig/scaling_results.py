"""Audit the frozen scaling inputs and descriptive publication records.

Calculations use the standard library in a separate arithmetic implementation.
The checks validate the recorded statistics and their consistency with the
method exports, frozen workload inputs and renderer data.
"""
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIZES = (16, 32, 64, 128)
METHODS = ("exact_cold", "exact_initialized", "greedy_beam", "greedy")
FIELDS = ("time_s", "ratio", "exact_ratio", "loss_percent")
STATISTICS = ("mean", "median", "min", "max", "sample_std", "q1", "q3")


def _quantile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _statistics(values):
    ordered = sorted(values)
    count = len(ordered)
    average = math.fsum(ordered) / count
    return {
        "n": count,
        "mean": average,
        "median": _quantile(ordered, .5),
        "min": ordered[0],
        "max": ordered[-1],
        "sample_std": math.sqrt(math.fsum((x - average) ** 2 for x in ordered) / (count - 1)),
        "q1": _quantile(ordered, .25),
        "q3": _quantile(ordered, .75),
    }


def audit_scaling(root=ROOT):
    """Return per-check evidence and recomputed summaries for Figure 8.

Missing or malformed files raise an error. Well-formed but inconsistent values
produce failed checks, so callers can include them in a broader audit report.
"""
    root = Path(root)
    directory = root / "results/scaling"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    inputs = json.loads((root / "data/inputs/scaling_cases.json").read_text(encoding="utf-8"))["cases"]
    catalog = json.loads((root / "data/inputs/catalog.json").read_text(encoding="utf-8"))["profiles"]
    checks = []

    def check(name, actual, expected, tolerance=1e-9):
        numeric = (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                   and isinstance(expected, (int, float)) and not isinstance(expected, bool))
        difference = abs(actual - expected) if numeric else None
        passed = (math.isfinite(actual) and math.isfinite(expected) and difference <= tolerance
                  if numeric else actual == expected)
        checks.append({"name": "scaling/" + name, "passed": bool(passed),
                       "actual": actual, "expected": expected,
                       "difference": difference, "tolerance": tolerance})

    def index(rows, fields, label):
        keys = [tuple(row[field] for field in fields) for row in rows]
        check(label + "/unique", len(keys), len(set(keys)), 0)
        return dict(zip(keys, rows))

    def finite(values, positive=False):
        return all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   and math.isfinite(x) and (x > 0 if positive else True) for x in values)

    references = {}
    for identifier, task, architecture in (
            ("fr_b1_k2_f900", "detection", "fasterrcnn_resnet50_fpn_v2"),
            ("dl_b1_k2_f900", "segmentation", "deeplabv3_resnet50")):
        matches = [profile for profile in catalog if profile["physical_config_id"] == identifier]
        prefix = "catalog/" + identifier
        check(prefix + "/unique-reference", len(matches), 1, 0)
        if len(matches) != 1:
            continue
        profile = matches[0]
        references[identifier] = profile
        for field, expected in (("measurement_status", "measured"), ("task", task),
                                ("arch", architecture), ("B", 1), ("slices", 2), ("freq_MHz", 900)):
            check(prefix + "/" + field, profile.get(field), expected, 0)
        calibration = profile.get("calibration", profile)
        check(prefix + "/positive-latency", finite([calibration.get("busy_L_s")], positive=True), True)

    check("methods", sorted(summary["methods"]), sorted(METHODS))
    check("series-methods", sorted(summary["series"]), sorted(METHODS))
    check("sizes", summary["sizes"], list(SIZES))
    check("cases-per-size", summary["cases_per_size"], 10, 0)
    check("workloads", summary["workloads"], 40, 0)
    check("observations", summary["observations"], 160, 0)
    seeds = summary["retained_seeds"] + summary["additional_seeds"]
    check("seed-count", len(seeds), 10, 0)
    check("seed-uniqueness", len(set(seeds)), len(seeds), 0)
    expected_cases = {f"s{seed}_n{size}" for seed in seeds for size in SIZES}
    case_inputs = index(inputs, ("case",), "inputs")
    check("input-count", len(inputs), 40, 0)
    check("input-identities", sorted(k[0] for k in case_inputs), sorted(expected_cases))
    check("input-size-counts", dict(Counter(row["n"] for row in inputs)), {n: 10 for n in SIZES})
    for row in inputs:
        name = "inputs/" + row["case"]
        size, seed, services = row["n"], row["mix_seed"], row["services"]
        check(name + "/identity", row["case"], f"s{seed}_n{size}")
        check(name + "/size", size in SIZES, True)
        check(name + "/services", len(services), size, 0)
        check(name + "/service-identities", [s["service_id"] for s in services], list(range(size)))
        check(name + "/gpus", row["gpus"], size // 2, 0)
        check(name + "/power-cap", row["cap_W"], 165., 0)
        check(name + "/demand-multiplier", row["demand_multiplier"], 1.2, 0)
        check(name + "/slo-multiplier", row["slo_multiplier"], 1., 0)
        check(name + "/placement-seed", row["placement_seed"], 0, 0)
        for service in services:
            prefix = name + "/service" + str(service["service_id"])
            task = "detection" if service["service_id"] % 2 == 0 else "segmentation"
            check(prefix + "/task", service["task"], task)
            check(prefix + "/quality", service["qmin"] in ((.35, .45) if task == "detection" else (.55, .59, .64)), True)
            check(prefix + "/slo-choice", service["base_slo"] in ((.18, .30, .45) if task == "detection" else (.12, .20, .30)), True)
            check(prefix + "/slo", service["slo"], service["base_slo"] * row["slo_multiplier"])
            check(prefix + "/ca2", service["ca2"], 1., 0)
            ref = service["reference"]
            check(prefix + "/reference", ref["physical_config_id"],
                  "fr_b1_k2_f900" if task == "detection" else "dl_b1_k2_f900")
            profile = references.get(ref["physical_config_id"])
            check(prefix + "/catalog-reference", profile is not None, True)
            if profile is not None:
                check(prefix + "/reference-latency", ref["busy_L_s"],
                      profile.get("calibration", profile).get("busy_L_s"), 1e-12)
            valid = finite([service["beta"], ref["busy_L_s"], service["lam"]], positive=True)
            check(prefix + "/finite-positive", valid, True)
            if valid:
                check(prefix + "/beta-range", .25 <= service["beta"] <= .55, True)
                check(prefix + "/arrival-rate", service["lam"],
                      row["demand_multiplier"] * service["beta"] / ref["busy_L_s"])
        largest = case_inputs.get((f"s{seed}_n128",))
        check(name + "/nested-prefix", services == largest["services"][:size] if largest else False, True)

    rows = [{"method": method, **row} for method, series in summary["series"].items() for row in series]
    observations = index(rows, ("method", "case"), "observations")
    check("row-count", len(rows), 160, 0)
    for method in METHODS:
        selected = summary["series"].get(method, [])
        check(method + "/identities", sorted(row["case"] for row in selected), sorted(expected_cases))
        check(method + "/size-counts", dict(Counter(row["services"] for row in selected)), {n: 10 for n in SIZES})
    for row in rows:
        name = row["method"] + "/" + row["case"]
        input_row = case_inputs.get((row["case"],))
        check(name + "/input-size", row["services"], input_row["n"] if input_row else None)
        valid = finite([row[f] for f in FIELDS])
        check(name + "/finite-values", valid, True)
        if not valid:
            continue
        check(name + "/positive-values", all(row[f] > 0 for f in ("time_s", "ratio", "exact_ratio")), True)
        if row["exact_ratio"] <= 0:
            continue
        exacts = [observations[(m, row["case"])]["ratio"] for m in METHODS[:2]
                  if (m, row["case"]) in observations]
        if exacts and finite(exacts, positive=True):
            check(name + "/paired-exact-reference", row["exact_ratio"], max(exacts))
        check(name + "/loss-formula", row["loss_percent"], 100 * (1 - row["ratio"] / row["exact_ratio"]))
        check(name + "/ratio-bound", row["ratio"] <= row["exact_ratio"] + 1e-9, True)
        check(name + "/nonnegative-loss", row["loss_percent"] >= -1e-9, True)
        if row["method"] in METHODS[:2]:
            check(name + "/exact-reference-loss", row["loss_percent"], 0., 1e-7)

    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    source_files = protocol["result_source"]["files"]
    source_paths = {path.relative_to(root).as_posix(): path
                    for path in (directory / "source").glob("*.json")}
    check("source/methods", sorted(source_files), sorted(METHODS))
    check("source/inventory", sorted(entry["path"] for entry in source_files.values()),
          sorted(source_paths))
    source_count = 0
    for method in METHODS:
        name = "source/" + method
        declared = source_files.get(method, {})
        path = source_paths.get(declared.get("path"))
        check(name + "/file", path is not None, True)
        if path is None:
            continue
        payload = path.read_bytes()
        check(name + "/sha256", hashlib.sha256(payload).hexdigest(), declared["sha256"])
        export = json.loads(payload)
        check(name + "/schema", export["schema"], f"scalability_{method}_v1")
        export_method = "original_serial_greedy_beam" if method == "greedy_beam" else method
        check(name + "/method", export["method"], export_method)
        cases = index(export["cases"], ("case",), name + "/cases")
        check(name + "/identities", sorted(case for case, in cases), sorted(expected_cases))
        source_count += len(export["cases"])
        for (case,), result in cases.items():
            observation = observations.get((method, case))
            if observation is None:
                continue
            values = {
                "services": result["services"],
                "time_s": result["timings_seconds"]["total_with_preparation_seconds"],
                "ratio": result["objective"]["ratio"],
                "exact_ratio": result["optimum_ratio"],
                "loss_percent": result["loss_percent"],
            }
            prefix = name + "/" + case
            check(prefix + "/finite-values", finite(values.values()), True)
            for field, value in values.items():
                check(prefix + "/" + field, value, observation[field], 0)

    record_directory = directory / "cases"
    record_paths = {path.stem: path for path in record_directory.glob("*.json")}
    check("case-records/identities", sorted(record_paths), sorted(expected_cases))
    source = {"path": "results/scaling/summary.json",
              "sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest()}
    for case in sorted(expected_cases & record_paths.keys()):
        record = json.loads(record_paths[case].read_text(encoding="utf-8"))
        name = "case-records/" + case
        check(name + "/fields", sorted(record),
              sorted(("schema", "case", "services", "source", "methods")))
        check(name + "/schema", record.get("schema"), "scaling_case_summary_v1")
        check(name + "/case", record.get("case"), case)
        input_row = case_inputs.get((case,))
        check(name + "/services", record.get("services"), input_row["n"] if input_row else None)
        check(name + "/source", record.get("source"), source)
        methods = record.get("methods", {})
        check(name + "/methods", sorted(methods), sorted(METHODS))
        for method in METHODS:
            values = methods.get(method, {})
            check(name + "/" + method + "/fields", sorted(values), sorted(FIELDS))
            expected = observations.get((method, case), {})
            for field in FIELDS:
                check(name + "/" + method + "/" + field,
                      values.get(field), expected.get(field), 0)

    with (directory / "observations.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        check("csv/observation-columns", sorted(reader.fieldnames), sorted(("method", "case", "services", *FIELDS)))
        csv_rows = list(reader)
    csv_observations = index(csv_rows, ("method", "case"), "csv/observations")
    check("csv/observation-identities", sorted(csv_observations), sorted(observations))
    for key in observations.keys() & csv_observations.keys():
        for field in ("services", *FIELDS):
            check("csv/" + "/".join(key) + "/" + field,
                  float(csv_observations[key][field]), observations[key][field], 0)

    aggregates = {}
    check("aggregate-methods", sorted(summary["aggregates"]), sorted(METHODS))
    for method in METHODS:
        supplied = index(summary["aggregates"].get(method, []), ("services",), method + "/aggregates")
        check(method + "/aggregate-sizes", sorted(k[0] for k in supplied), list(SIZES))
        aggregates[method] = []
        for size in SIZES:
            group = [row for row in summary["series"].get(method, []) if row["services"] == size]
            check(f"{method}/{size}/group-size", len(group), 10, 0)
            target = supplied.get((size,))
            if len(group) < 2 or target is None:
                continue
            actual = {"services": size, "cases": len(group)}
            check(f"{method}/{size}/cases", target["cases"], len(group), 0)
            for field in ("time_s", "loss_percent"):
                values = [row[field] for row in group]
                if not finite(values):
                    continue
                actual[field] = _statistics(values)
                check(f"{method}/{size}/{field}/statistic-keys", sorted(target[field]), sorted(("n", *STATISTICS)))
                for statistic, value in actual[field].items():
                    check(f"{method}/{size}/{field}/{statistic}", target[field][statistic], value)
            aggregates[method].append(actual)

    with (directory / "summary.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_columns = ["method", "services", "cases"] + [f"{field}_{stat}" for field in ("time_s", "loss_percent") for stat in STATISTICS]
        check("csv/aggregate-columns", sorted(reader.fieldnames), sorted(expected_columns))
        csv_summary = list(reader)
    csv_groups = index(csv_summary, ("method", "services"), "csv/aggregates")
    check("csv/aggregate-identities", sorted(csv_groups), sorted((m, str(n)) for m in METHODS for n in SIZES))
    for method, groups in aggregates.items():
        for group in groups:
            key = (method, str(group["services"]))
            if key not in csv_groups:
                continue
            check("csv/" + "/".join(key) + "/cases", int(csv_groups[key]["cases"]), group["cases"], 0)
            for field in ("time_s", "loss_percent"):
                for statistic in STATISTICS:
                    if field in group:
                        check("csv/" + "/".join(key) + "/" + field + "/" + statistic,
                              float(csv_groups[key][field + "_" + statistic]), group[field][statistic])

    tree = ast.parse((root / "figures/scalability.py").read_text(encoding="utf-8"))
    embedded = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("DATA", "SOURCE_SHA256", "DATA_SHA256"):
                    embedded[target.id] = ast.literal_eval(node.value)
    check("figure/literal-data", embedded.get("DATA") == summary["series"], True)
    canonical = json.dumps(embedded.get("DATA"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    check("figure/data-hash", hashlib.sha256(canonical).hexdigest(), embedded.get("DATA_SHA256"))
    check("figure/source-hash", hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
          embedded.get("SOURCE_SHA256"))
    failures = [row for row in checks if not row["passed"]]
    return {"passed": not failures, "failed": len(failures), "checks": checks, "failures": failures,
            "summary": {"workloads": len(inputs), "observations": len(rows),
                        "source_observations": source_count, "cases_per_size": 10,
                        "aggregates": aggregates,
                        "scope": "Consistency of method exports, frozen inputs, publication records and renderer data."}}
