"""Recompute figure inputs from supplied records and sample arrays.

Normalization, component definitions and sample assumptions follow the dataset
metadata. Numerical relationships are checked against the supplied inputs.
"""
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("nominal_full", "p95_full", "p99_full")
STATUSES = ("slo_met", "slo_not_met", "infeasible")


def _read(root, name):
    return json.loads((Path(root) / name).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(actual, expected, context, atol=1e-9):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _require(actual.shape == expected.shape and np.isfinite(actual).all()
             and np.isfinite(expected).all()
             and np.allclose(actual, expected, rtol=0, atol=atol),
             "Numerical mismatch: " + context)


def _index(rows, fields, label):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        _require(key not in indexed, "Duplicate " + label + ": " + str(key))
        indexed[key] = row
    return indexed


def _nonnegative(values, label, positive=False):
    values = np.asarray(values, dtype=float)
    _require(np.isfinite(values).all() and np.all(values > 0 if positive else values >= 0),
             "Invalid " + label)


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def aggregate_admission_blocks(blocks, cases, p99_infeasible_cases):
    """Aggregate one acquisition window per feasible policy/workload from replica summaries.

    Workload compliance requires every replica mean to meet its SLO. Request
    percentages share the nominal offered population, including requests in
    infeasible P99 workloads. Matching uses saved trace hashes and service counts.
    """
    cases, infeasible = set(cases), set(p99_infeasible_cases)
    _require(cases and infeasible <= cases, "Invalid admission case sets")
    records = _index(blocks, ("case", "policy", "gpu"), "admission board")
    expected = {(case, policy, gpu) for case in cases for policy in POLICIES
                if policy != "p99_full" or case not in infeasible for gpu in range(3)}
    _require(set(records) == expected, "Incomplete admission board cohort")
    services = defaultdict(dict)
    traces = defaultdict(set)
    compliant = defaultdict(list)
    windows = 0
    for (case, policy, gpu), block in records.items():
        _require(block["name"] == f"{case}__{policy}__g{gpu}__r2", "Invalid admission r2 board name")
        _require(isinstance(block["alias"], bool), "Invalid admission alias flag")
        _require(all(_valid_sha256(block[name]) for name in ("marker_sha256", "replay_sha256")),
                 "Invalid admission board source hash")
        _require(isinstance(block["trace_id"], str) and block["trace_id"].endswith("::" + case),
                 "Invalid admission trace identity")
        traces[case].add(block["trace_id"])
        replicas = _index(block["replicas"], ("replica_id",), "admission replica")
        for (replica_id,), replica in replicas.items():
            service = replica["service_id"]
            _require(service in {str(n) for n in range(7)}
                     and replica_id.startswith(f"g{gpu}_s{service}_"), "Invalid admission replica identity")
            n, met = replica["n"], replica["met"]
            _require(type(n) is int and type(met) is int and n > 0 and 0 <= met <= n,
                     "Invalid admission request counts")
            _nonnegative([replica["mean"]], "admission replica mean")
            _nonnegative([replica["slo"]], "admission replica SLO", positive=True)
            _require(type(replica["mean_met"]) is bool
                     and replica["mean_met"] == (replica["mean"] <= replica["slo"]),
                     "Admission mean_met disagrees with replica mean")
            _require(replica["mean"] * n + 1e-10 >= (n - met) * replica["slo"]
                     and (met != n or replica["mean"] <= replica["slo"] + 1e-10),
                     "Admission response mean is incompatible with request compliance counts")
            _require(_valid_sha256(replica["trace_sha256"]), "Invalid admission service trace hash")
            key = (case, policy)
            saved = services[key].setdefault(service, {"n": 0, "met": 0, "slo": replica["slo"],
                                                       "trace_sha256": replica["trace_sha256"]})
            _close(replica["slo"], saved["slo"], "admission service SLO", atol=0)
            _require(replica["trace_sha256"] == saved["trace_sha256"],
                     "Admission service trace differs between replicas")
            saved["n"] += n
            saved["met"] += met
            compliant[key].append(replica["mean_met"])
            windows += 1
    _require(all(len(values) == 1 for values in traces.values()), "Admission trace identity differs across policies")
    _require(all(set(entries) == {str(n) for n in range(7)} for entries in services.values()),
             "Incomplete admission service cohort")
    for (case, policy), entries in services.items():
        for service, entry in entries.items():
            reference = services[(case, "nominal_full")][service]
            _require(entry["n"] == reference["n"], "Admission per-service request counts differ across policies")
            _require(entry["trace_sha256"] == reference["trace_sha256"],
                     "Admission service trace differs across policies")
            _close(entry["slo"], reference["slo"], "admission shared service SLO", atol=0)
    offered = sum(row["n"] for (case, policy), rows in services.items()
                  if policy == "nominal_full" for row in rows.values())
    policies = []
    for policy in POLICIES:
        selected = [(case, policy) for case in sorted(cases) if (case, policy) in services]
        met_workloads = sum(all(compliant[key]) for key in selected)
        served = sum(row["n"] for key in selected for row in services[key].values())
        within = sum(row["met"] for key in selected for row in services[key].values())
        missed, unserved = served - within, offered - served
        policies.append({"key": policy,
                         "counts": {"slo_met": met_workloads, "slo_not_met": len(selected) - met_workloads,
                                    "infeasible": len(cases) - len(selected)},
                         "requests": {"offered": offered, "served": served, "within_slo": within,
                                      "above_slo": missed, "unserved": unserved,
                                      "within_percent": 100 * within / offered,
                                      "above_percent": 100 * missed / offered,
                                      "unserved_percent": 100 * unserved / offered}})
    return {"policies": policies, "request_denominator": offered, "replica_windows": windows,
            "physical_blocks": len(records), "workload_policy_records": len(services),
            "matched_service_traces": len(cases) * 7}


def recompute_admission(root=ROOT, verify_requests=True):
    """Recompute combined Figure 7 from replica aggregates and modeled objectives.

    ``verify_requests`` remains accepted for caller compatibility. Saved request
    counts, trace hashes and source checksums are always validated.
    """
    root = Path(root)
    supplied = _read(root, "evidence/admission/observations.json")
    published = _read(root, "evidence/admission/publication_data.json")
    source_names = {"evidence/admission/physical_blocks_r2.json",
                    "evidence/admission/objective_observations.json", "evidence/admission/preprocessing.json"}
    _require(set(supplied["sources"]) == source_names, "Incomplete admission source identities")
    for name, expected in supplied["sources"].items():
        _require(_valid_sha256(expected) and hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                 "Admission source checksum mismatch: " + name)
    cases = {f"unified_{n:03d}" for n in range(112)}
    _require(set(supplied["cases"]) == cases and len(supplied["cases"]) == 112,
             "Invalid admission workload cohort")
    _require(supplied["denominator"] == 112 and supplied["repetition"] == 2,
             "Invalid admission denominator or repetition")
    objectives = _read(root, "evidence/admission/objective_observations.json")
    rows = _index(objectives["rows"], ("case",), "admission objective")
    _require(set(rows) == {(case,) for case in cases}, "Incomplete admission objective cohort")
    infeasible = {case for (case,), row in rows.items() if row["p99_classification"] == "exact_infeasible"}
    _require(infeasible == set(supplied["p99_infeasible_cases"]) == set(objectives["p99_infeasible_cases"]),
             "Admission infeasible case identities disagree")
    blocks = _read(root, "evidence/admission/physical_blocks_r2.json")
    _require(supplied["physical_input"]["record_count"] == len(blocks)
             and supplied["physical_input"]["sha256"] == supplied["sources"]["evidence/admission/physical_blocks_r2.json"],
             "Admission physical source metadata mismatch")
    aggregate = aggregate_admission_blocks(blocks, cases, infeasible)
    for entry in aggregate["policies"][1:]:
        prefix = entry["key"].split("_")[0]
        losses = []
        for (case,), row in rows.items():
            _require(row["nominal_exact_status"] == "optimal", "Uncertified nominal objective reference")
            _require(row["p99_classification"] in {"exact_infeasible", "feasible_plan"},
                     "Invalid P99 objective classification")
            objective = row[prefix + "_heuristic_objective"]
            if prefix == "p99" and case in infeasible:
                _require(objective is None and row[prefix + "_loss_vs_nominal_optimum_percent"] is None,
                         "Unexpected P99 objective for an infeasible workload")
                continue
            reference = row["nominal_optimal_objective"]
            _nonnegative([reference, objective], "admission objective", positive=True)
            loss = 100 * (1 - objective / reference)
            _require(loss >= -1e-10, "Negative admission objective loss")
            _close(loss, row[prefix + "_loss_vs_nominal_optimum_percent"], "admission objective loss")
            losses.append(max(0.0, loss))
        values = np.asarray(losses)
        _require(len(values) == 112 - entry["counts"]["infeasible"], "Admission objective and physical cohorts differ")
        entry["loss"] = {"n": len(values), "min": float(values.min()), "max": float(values.max()),
                         "mean": float(values.mean()), "median": float(np.median(values))}
    timing = _read(root, "evidence/admission/preprocessing.json")
    seconds = timing["shared_mc_wall_seconds"]
    _nonnegative([seconds], "shared admission wall time", positive=True)
    _require(timing["timer"] == "time.perf_counter" and timing["shared_policies"] == ["p95_full", "p99_full"],
             "Invalid admission preprocessing timing scope")
    result = {"schema": "publication-admission-r2-v1", "denominator": len(cases), "repetition": 2,
              "event": "Every replica window-mean response <= its service SLO in repetition r2",
              "request_denominator": aggregate["request_denominator"],
              "request_denominator_definition": "All offered requests, including those in infeasible workloads",
              "policies": aggregate["policies"], "shared_mc_wall_seconds": seconds}
    for key in ("schema", "denominator", "repetition", "event", "request_denominator",
                "request_denominator_definition"):
        _require(result[key] == published[key], "Admission publication metadata mismatch: " + key)
    _require([p["key"] for p in published["policies"]] == list(POLICIES), "Invalid publication policy order")
    for actual, target in zip(result["policies"], published["policies"]):
        for section in ("counts", "requests", "loss"):
            _require((section in actual) == (section in target), "Admission publication section mismatch")
            if section in actual:
                _require(set(actual[section]) == set(target[section]), "Admission publication field mismatch")
                for key, value in actual[section].items():
                    _close(value, target[section][key], f"admission {actual['key']}/{section}/{key}")
    _close(seconds, published["shared_mc_wall_seconds"], "published admission wall time")
    summary = {"passed": True, "repetition": 2,
               **{key: value for key, value in aggregate.items() if key != "request_denominator"},
               "offered_requests_per_policy": aggregate["request_denominator"],
               "objective_losses": sum(p.get("loss", {}).get("n", 0) for p in result["policies"]),
               "shared_mc_wall_seconds": seconds,
               "scope": "Replica means and request counts from one acquisition campaign; matching service trace hashes; "
                        "per-workload modeled objective losses; shared preprocessing wall time."}
    return {"data": result, "summary": summary}


def recompute_response_data(root=ROOT):
    """Validate response summaries, then derive Figure 4's window-mean P95.

    The stored summary's ``response_s`` is the arithmetic mean of
    the 1024 window means. Figure 4 uses their linear P95. Both statistics
    are computed from the same sample arrays.
    """
    root = Path(root)
    published = _read(root, "evidence/response_time/publication_data.json")
    components = _index(_read(root, "evidence/response_time/component_observations.json")["records"],
                        ("cell_id",), "response component")
    observations = _read(root, "evidence/response_time/long_components.json")
    _require(observations["schema"] == "long-queue-errors-v2", "Unexpected long queue error schema")
    long_rows = _index(observations["records"], ("cell_id",), "long response component")
    rows = _index([dict(zip(published["columns"], row)) for row in published["rows"]],
                  ("cell_id",), "publication response")
    _require(len(rows) == 69 and set(rows) == set(components) == set(long_rows),
             "Response component cohorts do not match")
    result = deepcopy(published)
    recomputed = []
    maximum_difference = 0.0
    p95_minus_mean_differences = []
    with np.load(root / "evidence/response_time/simulated_window_means.npz", allow_pickle=False) as samples:
        _require(set(samples.files) == {key[0] for key in rows}, "Simulation sample cohort mismatch")
        for key, target in rows.items():
            row = deepcopy(target)
            component, long = components[key], long_rows[key]
            predicted, short = component["prediction_s"], component["measured_short_components_s"]
            _require(component["config_id"] == row["config_id"] and component["load_fraction"] == row["load_fraction"],
                     "Response configuration/load mismatch")
            _nonnegative(list(predicted.values()) + list(short.values()), "response components")
            _nonnegative([short["response"], component["measured_long_response_s"]],
                         "total response", positive=True)
            for label, parts in (("predicted", predicted), ("short measured", short)):
                _close(sum(parts[name] for name in ("execution", "batchfill", "queue")),
                       parts["response"], label + " response component sum")
            _require(long["load_fraction"] == row["load_fraction"], "Long queue load mismatch")
            _require(long["window_s"] == {"short": 45, "long": 900},
                     "Expected Figure 4 observation windows of 45 and 900 seconds")
            queue_error = long["queue_error_percent"]
            _nonnegative([queue_error], "Long-window queue error")
            for name in ("execution", "batchfill", "queue"):
                row[name + "_short"] = 100 * abs(predicted[name] - short[name]) / short["response"]
            row["batchfill_long"] = (100 * abs(predicted["batchfill"] - component["measured_long_batchfill_s"])
                                     / component["measured_long_response_s"])
            row["queue_long"] = queue_error
            # Panel (b) compares predictions with the short measured response.
            row["measured_response_s"] = short["response"]
            row["analytical_response_s"] = predicted["response"]
            values = samples[key[0]]
            _require(values.shape == (1024,), "Invalid response simulation sample shape")
            _nonnegative(values, "response simulation samples")
            row["response_s"] = float(values.mean())
            for field in published["columns"][2:]:
                _close(row[field], target[field], "response " + key[0] + "/" + field)
                maximum_difference = max(maximum_difference, abs(row[field] - target[field]))
            # Check the stored sample mean, then compute P95 across window means.
            # This percentile describes window means, not individual requests.
            row["response_s"] = float(np.quantile(values, .95, method="linear"))
            p95_minus_mean_differences.append(row["response_s"] - target["response_s"])
            recomputed.append([row[field] for field in published["columns"]])
    result["rows"] = recomputed
    result["simulation_estimator"] = {
        "statistic": "percentile", "percentile": 95, "quantile_method": "linear",
        "samples_per_cell": 1024, "sample_unit": "mean response in one simulated window, seconds",
        "source": "evidence/response_time/simulated_window_means.npz",
        "stored_summary_statistic": "arithmetic mean of the 1024 window means",
    }
    result["recomputation_scope"] = (
        "Figure 4 values and 45 s and 900 s window metadata are checked against the "
        "supplied records. Panel (b) uses linear P95 of the 1024 window means per cell.")
    return {"data": result, "summary": {"passed": True, "cells": 69, "long_component_pairs": 69,
            "saved_simulation_samples": 69 * 1024,
            "maximum_absolute_reference_recomputation_difference": maximum_difference,
            "reference_difference_scope": "Components and sample means versus the supplied response summary; P95 is calculated separately.",
            "p95_minus_mean_range_s": [min(p95_minus_mean_differences), max(p95_minus_mean_differences)],
            "simulation_estimator": result["simulation_estimator"],
            "scope": result["recomputation_scope"]}}


def audit_grid_samples(root=ROOT):
    """Check simulation source identities, sample prefixes and admission quantiles."""
    root = Path(root)
    metadata = _read(root, "data/inputs/admission_grid_samples.json")
    _require(metadata["source"] == "data/inputs/admission_grid.csv"
             and metadata["array_file"] == "admission_grid_samples.npz"
             and metadata["quantile_method"] == "linear" and metadata["units"] == "seconds",
             "Invalid admission-grid metadata")
    with (root / metadata["source"]).open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    _require(len(source) == 1323 and len(metadata["rows"]) == len(source), "Admission-grid row count mismatch")
    _index(source, ("config", "rate"), "admission-grid source row")
    mapped = _index(metadata["rows"], ("array_row",), "admission-grid array row")
    _require(set(mapped) == {(i,) for i in range(len(source))}, "Incomplete admission-grid row mapping")
    for i, target in enumerate(source):
        row = mapped[(i,)]
        _require(row["config"] == target["config"], "Admission-grid configuration mapping mismatch")
        _close(row["rate_requests_s"], float(target["rate"]), "admission-grid rate mapping", atol=1e-12)
    source_name = "data/inputs/admission_simulation_source.npz"
    binding = metadata.get("simulation_source", {})
    _require(binding.get("path") == source_name, "Invalid admission simulation-source path")
    _require(binding.get("sha256") == hashlib.sha256((root / source_name).read_bytes()).hexdigest(),
             "Admission simulation-source hash mismatch")
    _require(metadata.get("sample_sets_are_nested_prefixes") is True,
             "Admission sample sets must be nested prefixes")
    unique_windows = len(source) * 1024
    _require(type(metadata.get("unique_simulated_windows")) is int
             and metadata["unique_simulated_windows"] == unique_windows,
             "Admission unique simulated-window count mismatch")
    source_fields = {"means", "counts", "seed", "rate", "config", "L_s", "C2"}
    with np.load(root / source_name, allow_pickle=False) as archive:
        _require(set(archive.files) == source_fields, "Unexpected admission simulation-source fields")
        simulation = {name: archive[name] for name in source_fields}
    _require(simulation["means"].shape == (len(source), 1024)
             and simulation["counts"].shape == simulation["means"].shape,
             "Admission simulation-source sample shape mismatch")
    _nonnegative(simulation["means"], "admission simulation-source means")
    _require(np.issubdtype(simulation["counts"].dtype, np.integer)
             and np.all(simulation["counts"] >= 0),
             "Invalid admission simulation-source request counts")
    _require(all(simulation[name].shape == (len(source),)
                 for name in ("seed", "rate", "config", "L_s", "C2")),
             "Admission simulation-source row shape mismatch")
    _require(np.issubdtype(simulation["seed"].dtype, np.integer)
             and np.all(simulation["seed"] >= 0), "Invalid admission simulation-source seeds")
    _require(np.array_equal(simulation["config"], [row["config"] for row in source]),
             "Admission simulation-source configuration mapping mismatch")
    for name in ("rate", "L_s", "C2"):
        _nonnegative(simulation[name], "admission simulation-source " + name, positive=name != "C2")
        _close(simulation[name], [float(row[name]) for row in source],
               "admission simulation-source " + name, atol=1e-12)
    for i in range(len(source)):
        seed = mapped[(i,)].get("seed")
        _require(type(seed) is int and seed == int(simulation["seed"][i]),
                 "Admission-grid row seed mismatch")
    expected = {f"window_means_s_n{n}" for n in (256, 512, 1024)}
    _require(set(metadata["arrays"]) == expected, "Unexpected admission-grid array metadata")
    reports = {}
    total = 0
    with np.load(root / "data/inputs/admission_grid_samples.npz", allow_pickle=False) as samples:
        _require(set(samples.files) == expected, "Unexpected admission-grid arrays")
        for n in (256, 512, 1024):
            key = f"window_means_s_n{n}"
            values = samples[key]
            _require(values.shape == (1323, n) and metadata["arrays"][key]["shape"] == [1323, n],
                     "Admission-grid array shape mismatch")
            _nonnegative(values, "admission-grid samples")
            quantiles = np.quantile(values, [.95, .99], axis=1, method="linear")
            maximum_difference = 0.0
            for index, q in enumerate((95, 99)):
                target = np.array([float(row[f"q{q}_{n}"]) for row in source])
                _close(quantiles[index], target, f"admission-grid P{q}, n={n}")
                maximum_difference = max(maximum_difference, float(np.max(np.abs(quantiles[index] - target))))
            _require(np.array_equal(values, simulation["means"][:, :n]),
                     f"Admission-grid source prefix mismatch, n={n}")
            total += values.size
            reports[str(n)] = {"rows": len(source), "quantiles_checked": 2 * len(source),
                               "maximum_absolute_difference_s": maximum_difference}
    _require(type(metadata.get("stored_entries_including_prefixes")) is int
             and metadata["stored_entries_including_prefixes"] == total,
             "Admission stored sample-entry count mismatch")
    return {"passed": True, "grid_rows": 1323, "window_means": total, "quantiles_checked": 7938,
            "unique_simulated_windows": unique_windows, "sample_sets_are_nested_prefixes": True,
            "quantile_method": "linear", "sample_sizes": reports,
            "scope": "Recorded simulation identities, parameters and seeds checked against the grid; "
                     "sample arrays checked as exact source prefixes. P95/P99 use linear "
                     "interpolation and are checked against the admission table."}


def audit_figure_inputs(root=ROOT):
    return {"admission": recompute_admission(root)["summary"],
            "response_time": recompute_response_data(root)["summary"],
            "admission_grid": audit_grid_samples(root)}
