"""Check observations figure inputs and reject plausible damaged datasets."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from efficient_mig import figure_results as observations

RESPONSE_COLUMNS = {"analytical": "analytical_response_s", "simulated": "response_s"}


def sorted_linear_p95(values):
    """Independent percentile definition, without NumPy's quantile routine."""
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * .95
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


class FigureResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = observations.audit_figure_inputs(ROOT)

    def changed_json(self, filename, modify):
        original = observations._read

        def read(root, name):
            value = original(root, name)
            if name == filename:
                modify(value)
            return value

        return patch.object(observations, "_read", side_effect=read)

    def test_all_request_windows_and_admission_aggregates(self):
        result = self.audit["admission"]
        self.assertEqual(result["repetition"], 2)
        self.assertEqual(result["physical_blocks"], 969)
        self.assertEqual(result["replica_windows"], 2571)
        self.assertEqual(result["workload_policy_records"], 323)
        self.assertEqual(result["matched_service_traces"], 784)
        self.assertEqual(result["offered_requests_per_policy"], 819508)
        self.assertEqual(result["objective_losses"], 211)
        self.assertEqual([p["counts"]["slo_met"] for p in result["policies"]], [47, 92, 95])
        self.assertAlmostEqual(result["policies"][1]["loss"]["mean"], 1.3769792775900034)
        self.assertAlmostEqual(result["policies"][2]["loss"]["mean"], 3.6024881412125382)

    def test_response_components_and_stored_means_recompute(self):
        result = self.audit["response_time"]
        self.assertEqual(result["cells"], 69)
        self.assertEqual(result["saved_simulation_samples"], 70656)
        self.assertLess(result["maximum_absolute_reference_recomputation_difference"], 1e-10)
        self.assertEqual(result["simulation_estimator"]["percentile"], 95)
        self.assertEqual(result["simulation_estimator"]["samples_per_cell"], 1024)

    def test_response_uses_p95_of_window_means_independently_of_stored_mean(self):
        data = observations.recompute_response_data(ROOT)["data"]
        column = data["columns"].index("response_s")
        before = (ROOT / "evidence/response_time/publication_data.json").read_bytes()
        with np.load(ROOT / "evidence/response_time/simulated_window_means.npz", allow_pickle=False) as samples:
            for row in data["rows"]:
                with self.subTest(cell_id=row[0]):
                    values = samples[row[0]]
                    self.assertEqual(len(values), 1024)
                    self.assertAlmostEqual(row[column], sorted_linear_p95(values), delta=1e-12)
                    self.assertFalse(np.isclose(row[column], values.mean(), rtol=1e-10, atol=1e-10))
        observations.recompute_response_data(ROOT)
        self.assertEqual(before, (ROOT / "evidence/response_time/publication_data.json").read_bytes())

    def test_every_grid_quantile_matches(self):
        result = self.audit["admission_grid"]
        self.assertEqual(result["window_means"], 2370816)
        self.assertEqual(result["unique_simulated_windows"], 1354752)
        self.assertIs(result["sample_sets_are_nested_prefixes"], True)
        self.assertEqual(result["quantiles_checked"], 7938)
        for report in result["sample_sizes"].values():
            self.assertLessEqual(report["maximum_absolute_difference_s"], 1e-9)

    def test_duplicate_admission_block_rejected(self):
        with self.changed_json("evidence/admission/physical_blocks_r2.json",
                               lambda data: data[1].update(deepcopy(data[0]))):
            with self.assertRaisesRegex(ValueError, "Duplicate admission board"):
                observations.recompute_admission(ROOT)

    def test_status_inconsistent_with_replica_mean_rejected(self):
        def modify(data):
            replica = data[0]["replicas"][0]
            replica["mean_met"] = not replica["mean_met"]

        with self.changed_json("evidence/admission/physical_blocks_r2.json", modify):
            with self.assertRaisesRegex(ValueError, "mean_met disagrees"):
                observations.recompute_admission(ROOT)

    def test_request_count_mismatch_between_policies_rejected(self):
        def modify(data):
            data[0]["replicas"][0]["n"] += 1

        with self.changed_json("evidence/admission/physical_blocks_r2.json", modify):
            with self.assertRaisesRegex(ValueError, "request counts differ across policies"):
                observations.recompute_admission(ROOT)

    def test_request_count_within_slo_cannot_exceed_served(self):
        def modify(data):
            replica = data[0]["replicas"][0]
            replica["met"] = replica["n"] + 1

        with self.changed_json("evidence/admission/physical_blocks_r2.json", modify):
            with self.assertRaisesRegex(ValueError, "Invalid admission request counts"):
                observations.recompute_admission(ROOT)

    def test_admission_trace_identity_mismatch_rejected(self):
        def modify(data):
            replica = data[0]["replicas"][0]
            replica["trace_sha256"] = "0" * 64

        with self.changed_json("evidence/admission/physical_blocks_r2.json", modify):
            with self.assertRaisesRegex(ValueError, "service trace differs"):
                observations.recompute_admission(ROOT)

    def test_objective_loss_must_match_source_objectives(self):
        def modify(data):
            data["rows"][0]["p95_heuristic_objective"] *= .95

        with self.changed_json("evidence/admission/objective_observations.json", modify):
            with self.assertRaisesRegex(ValueError, "admission objective loss"):
                observations.recompute_admission(ROOT)

    def test_wrong_long_queue_error_is_rejected(self):
        def modify(data):
            data["records"][0]["queue_error_percent"] += .001

        with self.changed_json("evidence/response_time/long_components.json", modify):
            with self.assertRaisesRegex(ValueError, "/queue_long"):
                observations.recompute_response_data(ROOT)

    def test_long_queue_window_durations_are_checked(self):
        def modify(data):
            data["records"][0]["window_s"] = {"short": 30, "long": 600}

        with self.changed_json("evidence/response_time/long_components.json", modify):
            with self.assertRaisesRegex(ValueError, "observation windows of 45 and 900 seconds"):
                observations.recompute_response_data(ROOT)

    def test_grid_row_mapping_tampering_rejected(self):
        def modify(data):
            data["rows"][0]["config"] = "unknown_config"

        with self.changed_json("data/inputs/admission_grid_samples.json", modify):
            with self.assertRaisesRegex(ValueError, "configuration mapping"):
                observations.audit_grid_samples(ROOT)

    def test_grid_sample_tampering_rejected(self):
        original_load = np.load

        class AlteredArrays:
            def __init__(self, *args, **kwargs):
                self.archive = original_load(*args, **kwargs)
                self.files = self.archive.files

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.archive.close()

            def __getitem__(self, name):
                value = self.archive[name]
                if name == "window_means_s_n256":
                    value[0] *= 2
                return value

        with patch.object(observations.np, "load", side_effect=AlteredArrays):
            with self.assertRaisesRegex(ValueError, "admission-grid P95"):
                observations.audit_grid_samples(ROOT)

    def test_grid_simulation_source_binding_tampering_rejected(self):
        for field, value, message in (("path", "data/inputs/other.npz", "source path"),
                                      ("sha256", "0" * 64, "source hash")):
            with self.subTest(field=field):
                def modify(data):
                    data["simulation_source"][field] = value

                with self.changed_json("data/inputs/admission_grid_samples.json", modify):
                    with self.assertRaisesRegex(ValueError, message):
                        observations.audit_grid_samples(ROOT)

    def test_grid_row_seed_must_match_simulation_source(self):
        def modify(data):
            data["rows"][0]["seed"] += 1

        with self.changed_json("data/inputs/admission_grid_samples.json", modify):
            with self.assertRaisesRegex(ValueError, "row seed mismatch"):
                observations.audit_grid_samples(ROOT)

    def test_grid_stored_entry_count_must_match_arrays(self):
        def modify(data):
            data["stored_entries_including_prefixes"] += 1

        with self.changed_json("data/inputs/admission_grid_samples.json", modify):
            with self.assertRaisesRegex(ValueError, "stored sample-entry count mismatch"):
                observations.audit_grid_samples(ROOT)

    def test_grid_reordered_prefix_rejected_when_quantiles_are_unchanged(self):
        original_load = np.load

        class ReorderedArrays:
            def __init__(self, *args, **kwargs):
                self.archive = original_load(*args, **kwargs)
                self.files = self.archive.files

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.archive.close()

            def __getitem__(self, name):
                value = self.archive[name]
                if name == "window_means_s_n256":
                    before = np.quantile(value[0], [.95, .99], method="linear")
                    value[0, [0, 1]] = value[0, [1, 0]]
                    np.testing.assert_array_equal(
                        np.quantile(value[0], [.95, .99], method="linear"), before)
                return value

        with patch.object(observations.np, "load", side_effect=ReorderedArrays):
            with self.assertRaisesRegex(ValueError, "source prefix mismatch"):
                observations.audit_grid_samples(ROOT)

    def test_repeated_figure_loads_preserve_embedded_data(self):
        for name in ("solution_method_comparison", "response_time_errors"):
            with self.subTest(figure=name):
                spec = importlib.util.spec_from_file_location(name, ROOT / "figures" / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                global_name = "DATA_JSON" if name == "response_time_errors" else "DATA"
                original = deepcopy(getattr(module, global_name))
                loader = module.load_statistics if name == "response_time_errors" else module.load_data
                first = loader()
                second = loader()
                self.assertEqual(first, second)
                self.assertIsNot(first, second)
                # A caller changing a nested result cannot corrupt the embedded data
                # or a separately loaded result (including the timing map).
                if name == "solution_method_comparison":
                    first["timings"]["mean_planning_seconds"]["fixed_frequency"] = -1
                    self.assertGreater(second["timings"]["mean_planning_seconds"]["fixed_frequency"], 0)
                else:
                    first[0]["rows"][0][3] = -1
                    self.assertGreater(second[0]["rows"][0][3], 0)
                self.assertEqual(getattr(module, global_name), original)

    def test_rehashed_mismatched_embedded_copy_is_rejected(self):
        spec = importlib.util.spec_from_file_location("response_mismatched_copy_test", ROOT / "figures/response_time_errors.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mismatched = json.loads(module.DATA_JSON)
        mismatched["rows"][0][3] *= 1.01
        module.DATA_JSON = json.dumps(mismatched)
        module.DATA_SHA256 = hashlib.sha256(module.DATA_JSON.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "authoritative publication_data"):
            module.load_statistics()

    def test_canonical_embedded_data_sign_and_quartiles(self):
        spec = importlib.util.spec_from_file_location("response_embedded_test", ROOT / "figures/response_time_errors.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        digest = hashlib.sha256(module.DATA_JSON.encode()).hexdigest()
        self.assertEqual(digest, module.DATA_SHA256)
        self.assertEqual(module.signed_error_percent(predicted=8, measured=10), 20)
        self.assertEqual(module.signed_error_percent(predicted=12, measured=10), -20)
        self.assertIn("Positive = underprediction", module.SIGN_CONVENTION)
        embedded = json.loads(module.DATA_JSON)
        canonical = json.loads((ROOT / "evidence/response_time/publication_data.json").read_text(encoding="utf-8"))
        self.assertEqual(embedded, canonical)
        rows = [dict(zip(embedded["columns"], row)) for row in embedded["rows"]]
        self.assertEqual(len(rows), 69)
        self.assertEqual(len({row["cell_id"] for row in rows}), 69)
        self.assertEqual(len({row["config_id"] for row in rows}), 23)
        groups = {"a": {}, "b": {}}
        for load in module.LOADS:
            selected = [row for row in rows if row["load_fraction"] == load]
            self.assertEqual(len(selected), 23)
            groups["a"][str(load)] = {
                component: module.statistics([row[component] for row in selected])
                for component, *_ in module.SERIES}
            groups["b"][str(load)] = {}
            for method, column in RESPONSE_COLUMNS.items():
                prediction_minus_measurement_errors = np.array([
                    100 * (row[column] - row["measured_response_s"]) / row["measured_response_s"]
                    for row in selected])
                actual = module.statistics([
                    module.signed_error_percent(row[column], row["measured_response_s"])
                    for row in selected])
                groups["b"][str(load)][method] = actual
                self.assertAlmostEqual(actual["mean"], -float(prediction_minus_measurement_errors.mean()))
                self.assertAlmostEqual(actual["median"], -float(np.median(prediction_minus_measurement_errors)))
                self.assertAlmostEqual(actual["p25"], -float(np.quantile(prediction_minus_measurement_errors, .75)))
                self.assertAlmostEqual(actual["p75"], -float(np.quantile(prediction_minus_measurement_errors, .25)))
            self.assertGreater(groups["b"][str(load)]["analytical"]["mean"], 0)
        self.assertAlmostEqual(groups["b"]["0.3"]["analytical"]["mean"], 3.285738915875302)

    def test_canonical_runtime_sign_and_quartiles(self):
        # Independently derive P95 from saved window means; all other columns
        # retain the measured responses in the supplied data.
        spec = importlib.util.spec_from_file_location("response_runtime_sign_test", ROOT / "figures/response_time_errors.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, groups = module.load_statistics()
        publication = json.loads((ROOT / "evidence/response_time/publication_data.json").read_text(encoding="utf-8"))
        rows = [dict(zip(publication["columns"], row)) for row in publication["rows"]]
        with np.load(ROOT / "evidence/response_time/simulated_window_means.npz", allow_pickle=False) as samples:
            for row in rows:
                row["response_s"] = sorted_linear_p95(samples[row["cell_id"]])
        for load in module.LOADS:
            selected = [row for row in rows if row["load_fraction"] == load]
            for panel, names in (("a", [name for name, *_ in module.SERIES]),
                                 ("b", RESPONSE_COLUMNS)):
                for name in names:
                    if panel == "a":
                        values = np.array([row[name] for row in selected])
                    else:
                        # Calculate from per-cell P95 predictions, independently of
                        # the loader's error and statistics helper functions.
                        values = np.array([
                            100 * (row["measured_response_s"] - row[RESPONSE_COLUMNS[name]])
                            / row["measured_response_s"] for row in selected])
                    expected = {"mean": float(values.mean()), "median": float(np.median(values)),
                                "p25": float(np.quantile(values, .25)),
                                "p75": float(np.quantile(values, .75))}
                    actual = groups[panel][str(load)][name]
                    self.assertEqual(actual["n"], 23)
                    for key, value in expected.items():
                        self.assertAlmostEqual(actual[key], value, delta=1e-12)
        figure = module.make_figure(groups)
        try:
            self.assertEqual(figure.axes[1].get_ylabel(), "Signed error (% underestimated)")
            lower, upper = figure.axes[1].get_ylim()
            self.assertLess(lower, 0)
            self.assertGreater(upper, 0)
            for by_method in groups["b"].values():
                for stats in by_method.values():
                    for field in ("mean", "median", "p25", "p75"):
                        self.assertLess(lower, stats[field])
                        self.assertGreater(upper, stats[field])
        finally:
            module.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
