"""Recompute publication statistics from supplied records and predictions.

Comparisons use separate publication summaries. Normalization and derived
quantities follow each dataset's metadata; embedded targets do not replace
calculation from individual observations.
"""
from copy import deepcopy
import csv
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("Paired MIG", "Full GPU", "Batched MIG", "Residency")
POWER_INPUT = "data/tables/power_holdout_original.csv"
METHODS = ("Fixed-frequency", "Sequential", "Greedy-C", "Greedy",
           "Greedy+Beam", "LeastAllocated", "MostAllocated")
OCCUPANCY_METHODS = METHODS[-2:]
METHOD_KEYS = {"fixed_frequency": "Fixed-frequency", "sequential": "Sequential",
               "greedy_construction": "Greedy-C", "greedy": "Greedy", "full": "Greedy+Beam"}
STATS = ("mean", "median", "maximum")
SCOPE = ("Table 6 power is recomputed from author-supplied comparison values. "
         "The two Table 7 measured occupancy rows are publication summaries.")


def _read(root, name):
    return json.loads((Path(root) / name).read_text(encoding="utf-8"))


def _require(condition, context):
    if not condition:
        raise ValueError(context)


def _number(value, context, *, positive=False, nonnegative=False):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), context + ": expected a finite number")
    _require(not positive or value > 0, context + ": expected a positive number")
    _require(not nonnegative or value >= 0, context + ": expected a nonnegative number")
    return float(value)


def _check(checks, dataset, metric, actual, target, tolerance=1e-9):
    actual = _number(actual, dataset + "/" + metric)
    target = _number(target, dataset + "/" + metric + " target")
    difference = abs(actual - target)
    row = {"dataset": dataset, "metric": metric, "actual": actual,
           "target": target, "absolute_difference": difference,
           "tolerance": tolerance, "passed": difference <= tolerance}
    checks.append(row)
    _require(row["passed"], f"{dataset}/{metric}: {actual} != {target} (tolerance {tolerance})")


def _unique(rows, field, context):
    values = [row[field] for row in rows]
    _require(all(isinstance(v, str) and v for v in values), context + ": invalid identifier")
    _require(len(values) == len(set(values)), context + ": duplicate " + field)
    return set(values)


def _summary(values):
    _require(bool(values), "Cannot summarize empty observations")
    return {"mean": mean(values), "median": median(values), "maximum": max(values)}


def summarize_supplied_power(path):
    """Calculate absolute percentage errors from author-supplied power pairs."""
    raw = Path(path).read_bytes()
    reader = csv.DictReader(StringIO(raw.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    comparison_field = "power_measured_W"
    required_fields = {"group", "target_id", "power_predicted_W", comparison_field,
                       "absolute_error_pct"}
    _require(len(headers) == len(required_fields) and set(headers) == required_fields,
             "Expected group, target_id, power_predicted_W, power_measured_W, and absolute_error_pct")
    by_group = {name: {} for name in GROUPS}
    for row in reader:
        context = f"Power CSV line {reader.line_num}"
        _require(None not in row and all(value is not None for value in row.values()),
                 context + ": incomplete or extra fields")
        name, target = row["group"].strip(), row["target_id"].strip()
        _require(name in by_group and bool(target), context + ": invalid group or target")
        group = by_group[name]
        _require(target not in group, context + ": duplicate target in group")
        predicted = _number(float(row["power_predicted_W"]), context, positive=True)
        comparison = _number(float(row[comparison_field]), context, positive=True)
        reported = _number(float(row["absolute_error_pct"]), context, nonnegative=True)
        error = 100 * abs(predicted - comparison) / comparison
        _require(math.isclose(error, reported, rel_tol=0, abs_tol=1e-9),
                 context + ": supplied error differs from computed error")
        group[target] = error
    _require(all(by_group.values()), "Expected at least one target in each power group")
    return {
        "source_kind": "author_supplied_power_comparison",
        "source_file": Path(path).name,
        "source_sha256": sha256(raw).hexdigest(),
        "input_comparison_column": comparison_field,
        "groups": {name: {"n": len(group), **_summary(group.values())}
                   for name, group in by_group.items()},
    }


def _compare_summary(checks, dataset, summary, published, *, beam=False):
    rounded = {}
    for key in STATS:
        precision = 3 if beam and key == "mean" else 2
        rounded[key] = round(summary[key], precision)
        _check(checks, dataset, key + "_published", rounded[key], published[key], 0)
    return rounded


def _predicted_losses(root, checks):
    pairs = _read(root, "data/tables/predicted_efficiency_pairs.json")["pairs"]
    keyed = [(r["case"], r["method"]) for r in pairs]
    _require(len(keyed) == len(set(keyed)), "Predicted pairs: duplicate case/method")
    expected_cases = {f"unified_{i:03d}" for i in range(112)}
    losses = {name: {} for name in METHODS}
    max_difference = 0.0
    for row in pairs:
        _require(row["method"] in METHOD_KEYS, "Unknown predicted method")
        _require(row["case"] in expected_cases, "Unknown predicted workload")
        _require(row["reference"] == "Exact", "Predicted reference must be Exact")
        value = _number(row["modeled_utility_per_J"], "Predicted efficiency", positive=True)
        reference = _number(row["reference_modeled_utility_per_J"], "Predicted reference", positive=True)
        loss = 100 * (1 - value / reference)
        max_difference = max(max_difference, abs(loss - _number(row["loss_percent"], "Predicted loss")))
        losses[METHOD_KEYS[row["method"]]][row["case"]] = loss
    _check(checks, "Table 7 predicted", "maximum_loss_formula_difference", max_difference, 0)
    occupancy = _read(root, "results/occupancy/publication_data.json")["rows"]
    _require(_unique(occupancy, "case", "Occupancy") == expected_cases,
             "Occupancy: expected all 112 attempted workloads")
    for row in occupancy:
        for name in ("LeastAllocated", "MostAllocated"):
            result = row["results"][name]
            ratio = result["ratio"]
            if ratio is None:
                _require(result["status"] != "feasible", "Occupancy feasible result has no ratio")
                continue
            _require(result["status"] == "feasible", "Occupancy ratio supplied for failed case")
            reference = _number(row["greedy_construction"]["ratio"], "Occupancy reference", positive=True)
            ratio = _number(ratio, "Occupancy efficiency", positive=True)
            losses[name][row["case"]] = 100 * (1 - ratio / reference)
    return losses, expected_cases


def recompute_tables(root=ROOT):
    """Return rounded tables plus a full-precision audit; fail on inconsistency."""
    root = Path(root)
    published = _read(root, "data/tables/published_tables.json")
    _require(tuple(published["table6"]) == GROUPS, "Unexpected Table 6 groups")
    _require(tuple(published["table7"]) == METHODS, "Unexpected Table 7 methods")
    table6, table7 = deepcopy(published["table6"]), deepcopy(published["table7"])
    checks, details6, details7 = [], {}, {}
    observations6 = _read(root, "data/tables/validation_errors.json")
    _require(observations6["schema"] == "table6-validation-v1", "Unexpected Table 6 observations schema")
    groups = observations6["groups"]
    expected_groups = {(name, "response") for name in GROUPS[:-1]}
    _require(len(groups) == len(expected_groups) and
             {(g["group"], g["metric"]) for g in groups} == expected_groups,
             "Table 6: missing, duplicated or unknown group")
    response_targets = _read(root, "data/tables/response_group_statistics.json")["groups"]
    all_ids = set()
    power = summarize_supplied_power(root / POWER_INPUT)
    for name, group in power["groups"].items():
        context = f"Table 6/{name}/power"
        _check(checks, context, "records_n", group["n"],
               published["table6_power_source"]["target_counts"][name], 0)
        summary = {stat: group[stat] for stat in STATS}
        table6[name]["power"] = _compare_summary(checks, context, summary, published["table6"][name]["power"])
        details6[name] = {"power": {"statistics": summary, "records_n": group["n"],
                                    "data_representation": "author-supplied power pairs in watts",
                                    "source": POWER_INPUT}}
    for group in groups:
        name, metric = group["group"], group["metric"]
        context = f"Table 6/{name}/{metric}"
        records = group["records"]
        identifiers = _unique(records, "id", context)
        _require(not identifiers & all_ids, "Table 6: duplicate identifier across groups")
        all_ids.update(identifiers)
        _require(group["unit"] == "percent", context + ": unexpected unit")
        _check(checks, context, "records_n", len(records), group["n"], 0)
        paired_response = name == "Batched MIG" and metric == "response"
        representation = "record identifiers and absolute percentage errors"
        if paired_response:
            _require(group["value_representation"] == "response_pairs_seconds",
                     context + ": expected response pairs in seconds")
            values = []
            for row in records:
                identifier = row["id"]
                _require(identifier == f'{row["configuration"]}/{row["stage"]}',
                         context + ": unexpected configuration/stage identifier")
                predicted = _number(row["predicted_response_s"], context, positive=True)
                measured = _number(row["measured_response_s"], context, positive=True)
                error = 100 * abs(predicted - measured) / measured
                _check(checks, context, identifier + "/absolute_error_percent",
                       error, row["absolute_error_percent"])
                values.append(error)
            representation = "Figure 4 analytical predictions and 45 s observed responses in seconds"
        else:
            values = [_number(r["absolute_error_percent"], context, nonnegative=True) for r in records]
        summary = _summary(values)
        if paired_response:
            _check(checks, context, "source_n", len(values), group["source_n"], 0)
            for stat in STATS:
                _check(checks, context, stat, summary[stat], group["target_statistics"][stat])
        else:
            authority = response_targets[name]
            _require(group["value_representation"] == "heldout_response_absolute_errors",
                     context + ": expected archived holdout response errors")
            for row in records:
                identifier = f'{row["host"]}/{row["label"]}/{row["actor"]}'
                _require(row["id"] == identifier and row["host"] and row["actor"],
                         context + ": unexpected holdout target identifier")
                _require(row["label"].startswith("full_") == (name == "Full GPU"),
                         context + ": unexpected holdout target group")
            _require(group["source_n"] == authority["n"], context + ": source n mismatch")
            _check(checks, context, "source_n", len(values), authority["n"], 0)
            summary["p90"] = float(np.quantile(values, .9, method="linear"))
            for stat in (*STATS, "p90"):
                _check(checks, context, stat, summary[stat], authority[stat])
            representation = "archived absolute response errors with holdout target identifiers"
        table6[name][metric] = _compare_summary(checks, context, summary, published["table6"][name][metric])
        details6.setdefault(name, {})[metric] = {
            "statistics": summary, "records_n": len(values),
            "source_reported_n": group["source_n"],
            "data_representation": representation}
    _require(table6["Residency"]["response"] is None, "Residency response is unreported")
    details6["Residency"]["response"] = None

    predicted, expected_cases = _predicted_losses(root, checks)
    observations7 = _read(root, "data/tables/measured_efficiency_pairs.json")
    _require(observations7["schema"] == "table7-efficiency-v1", "Unexpected Table 7 observations schema")
    methods = observations7["methods"]
    archived_methods = METHODS[:-len(OCCUPANCY_METHODS)]
    _require(len(methods) == len(archived_methods) and {m["method"] for m in methods} == set(archived_methods),
             "Table 7: missing, duplicated or unknown method")
    by_name = {m["method"]: m for m in methods}
    exact_repetitions = {}
    all_ids = set()
    for name in archived_methods:
        method, source = by_name[name], published["table7"][name]
        context = "Table 7/" + name
        reference_name = "Exact"
        _require(source["reference"] == method["reference"] == reference_name, context + ": wrong reference")
        _require(source["attempts"] == method["attempts"] == 112, context + ": wrong attempt count")
        records = method["records"]
        _check(checks, context, "source_measured_pairs_n", len(records), method["source_measured_pairs_n"], 0)
        _check(checks, context, "measured_pairs_n", len(records), source["measured_pairs_n"], 0)
        identifiers = _unique(records, "id", context)
        _require(not identifiers & all_ids, "Table 7: duplicate identifier across methods")
        all_ids.update(identifiers)
        cases = _unique(records, "case", context)
        _require(cases == set(predicted[name]), context + ": observations cohort differs from source predictions")
        excluded = method["excluded_cases"]
        _require(len(excluded) == len(set(excluded)) and set(excluded) == expected_cases - cases,
                 context + ": exclusions do not preserve failed workloads")
        _check(checks, context, "records_n", len(records), method["n"], 0)
        _check(checks, context, "predicted_n", len(predicted[name]), source["predicted_pairs_n"], 0)
        values, formula_difference = [], 0.0
        for row in records:
            repetitions = row["repetitions"]
            _require(len(repetitions) == 2 and {r["repetition"] for r in repetitions} == {0, 1},
                     context + ": expected complete repetitions 0 and 1")
            ratios = {}
            for side in ("method", "exact"):
                ratios[side] = mean(
                    _number(r[side]["completed_utility"], context + " completed utility", positive=True)
                    / _number(r[side]["energy_through_drain_J"], context + " energy", positive=True)
                    for r in repetitions
                )
            baseline = {r["repetition"]: r["exact"] for r in repetitions}
            _require(exact_repetitions.setdefault(row["case"], baseline) == baseline,
                     context + ": inconsistent Exact observations for workload")
            value, reference = ratios["method"], ratios["exact"]
            loss = 100 * (1 - value / reference)
            stated_loss = _number(row["loss_percent"], context + " loss")
            formula_difference = max(formula_difference, abs(loss - stated_loss))
            values.append(loss)
        _check(checks, context, "maximum_loss_formula_difference", formula_difference, 0)
        summary = _summary(values)
        for stat in STATS:
            _check(checks, context + "/observations", stat, summary[stat], method["target_statistics"][stat])
        prediction_summary = _summary(list(predicted[name].values()))
        table7[name]["predicted"] = _compare_summary(checks, context + "/predicted", prediction_summary,
                                                       source["predicted"], beam=name == "Greedy+Beam")
        table7[name]["measured"] = _compare_summary(checks, context + "/observations", summary,
                                                      source["measured"], beam=name == "Greedy+Beam")
        details7[name] = {"predicted_statistics": prediction_summary, "statistics": summary,
                          "predicted_pairs_n": len(predicted[name]), "records_n": len(records),
                          "measured_pairs_n": source["measured_pairs_n"], "excluded_cases": sorted(excluded),
                          "reference": reference_name,
                          "data_representation": "completed utility and energy through drain for two repetitions"}
    for name in OCCUPANCY_METHODS:
        source = published["table7"][name]
        context = "Table 7/" + name
        _require(source["reference"] == "Greedy-C" and source["attempts"] == 112,
                 context + ": unexpected prediction reference or attempt count")
        _require(source["measured_pairs_n"] is None, context + ": measured cohort size remains unreported")
        _check(checks, context, "predicted_n", len(predicted[name]), source["predicted_pairs_n"], 0)
        prediction_summary = _summary(list(predicted[name].values()))
        table7[name]["predicted"] = _compare_summary(checks, context + "/predicted", prediction_summary,
                                                     source["predicted"])
        for stat in STATS:
            _number(source["measured"][stat], context + "/reported/" + stat)
        details7[name] = {"predicted_statistics": prediction_summary, "statistics": source["measured"],
                          "predicted_pairs_n": len(predicted[name]), "records_n": 0,
                          "measured_pairs_n": None, "reference": source["reference"],
                          "data_representation": "publication summary only"}
    return {"table6": table6, "table7": table7, "audit": {
        "scope": SCOPE, "passed": True, "checks": checks,
        "counts": {"table6_errors": sum(g["n"] for g in groups),
                   "table6_power_targets": sum(g["n"] for g in power["groups"].values()),
                   "table6_residency_targets": power["groups"]["Residency"]["n"],
                   "table7_losses": sum(m["n"] for m in methods),
                   "table7_predicted_pairs": sum(len(v) for v in predicted.values())},
        "table6_power_source": {key: value for key, value in power.items() if key != "groups"},
        "table6_full_precision": details6, "table7_full_precision": details7}}


def recompute_planning_timings(root=ROOT):
    """Validate stage sums/cohorts and recompute the two supplied timing means."""
    root = Path(root)
    observations = _read(root, "results/planning/timings.json")
    published = _read(root, "results/planning/published_timings.json")
    source = _read(root, "results/planning/full_timing/timing_summary.json")
    stages = source["per_case"]
    _require(observations["schema"] == "planning-timings-v1" and observations["units"] == "seconds",
             "Unexpected planning timing schema or units")
    expected_cases = {f"unified_{i:03d}" for i in range(112)}
    _require(_unique(stages, "case", "Planning stage source") == expected_cases,
             "Planning stage source: incomplete cohort")
    stage_by_case = {r["case"]: r for r in stages}
    rows = observations["rows"]
    keyed = [(r["case"], r["method"]) for r in rows]
    methods = ("fixed_frequency", "sequential")
    _require(len(keyed) == len(set(keyed)), "Planning timings: duplicate case/method")
    _require(set(keyed) == {(c, m) for c in expected_cases for m in methods},
             "Planning timings: incomplete or unexpected cohort")
    _require(observations["attempts_per_method"] == published["attempts_per_method"] == 112,
             "Planning timings: incorrect denominator")
    checks, means, details = [], {}, {}
    max_stage_difference, max_preprocessing_difference = 0.0, 0.0
    max_total_difference = 0.0
    for row in rows:
        values = {key: _number(row[key], "Planning " + key, nonnegative=True)
                  for key in ("time_s", "preprocessing_seconds", "search_seconds")}
        max_stage_difference = max(max_stage_difference,
                                  abs(values["time_s"] - values["preprocessing_seconds"] - values["search_seconds"]))
        max_preprocessing_difference = max(max_preprocessing_difference,
            abs(values["preprocessing_seconds"] - _number(
                stage_by_case[row["case"]]["preparation_seconds"],
                "Source preprocessing", nonnegative=True)))
        max_total_difference = max(max_total_difference,
            abs(values["time_s"] - _number(
                stage_by_case[row["case"]][row["method"] + "_seconds"],
                "Source total", positive=True)))
    _check(checks, "Planning timings", "maximum_stage_sum_difference", max_stage_difference, 0)
    _check(checks, "Planning timings", "maximum_source_preprocessing_difference", max_preprocessing_difference, 0)
    _check(checks, "Planning timings", "maximum_source_total_difference", max_total_difference, 0)
    for method in methods:
        values = [r["time_s"] for r in rows if r["method"] == method]
        means[method] = mean(values)
        _check(checks, "Planning/" + method, "n", len(values), 112, 0)
        _check(checks, "Planning/" + method, "mean_seconds", means[method],
               published["mean_planning_seconds"][method])
        details[method] = {"n": len(values), "mean_seconds": means[method],
                           "median_seconds": median(values), "maximum_seconds": max(values)}
    return {"scope": "Per-workload times checked against the archived full-timing run and publication means.",
            "passed": True, "mean_planning_seconds": means, "checks": checks,
            "counts": {"planning_times": len(rows)}, "statistics": details}
