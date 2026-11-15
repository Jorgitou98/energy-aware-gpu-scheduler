"""Check publication values, retained observations and request-level denominators."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, median
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tables", ROOT / "scripts/build_tables.py")
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)
from efficient_mig.figure_results import recompute_admission


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def figure_data(name):
    tree = ast.parse((ROOT / "figures" / name).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("DATA", "DATA_JSON") for t in node.targets):
            result = ast.literal_eval(node.value)
            return json.loads(result) if isinstance(result, str) else result
    raise AssertionError("No literal publication data: " + name)


def response_plot_module():
    spec = importlib.util.spec_from_file_location(
        "response_runtime_test", ROOT / "figures/response_time_errors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublishedTableTests(unittest.TestCase):
    def test_validation_table_matches_published_values(self):
        t6, _ = tables.load_tables()
        expected = {
            "Paired MIG": ((1.59, 1.29, 3.47), (8.15, 5.26, 44.86)),
            "Full GPU": ((1.72, 1.41, 4.27), (8.19, 5.32, 38.76)),
            "Batched MIG": ((1.91, 1.46, 4.90), (5.19, 3.48, 58.49)),
            "Residency": ((0.77, 0.77, 0.93), None),
        }
        for name, groups in expected.items():
            for field, values in zip(("power", "response"), groups):
                if values is None:
                    self.assertIsNone(t6[name][field])
                else:
                    self.assertEqual(tuple(t6[name][field][key] for key in ("mean", "median", "maximum")), values)

    def test_efficiency_table_matches_published_values(self):
        _, t7 = tables.load_tables()
        expected = {
            "Fixed-frequency": ((11.63, 11.71, 12.77), (10.86, 10.91, 14.04)),
            "Sequential": ((4.26, 2.02, 9.32), (4.10, 2.47, 9.49)),
            "Greedy-C": ((4.70, 2.71, 19.21), (4.45, 3.24, 20.12)),
            "Greedy": ((4.27, 2.30, 18.41), (4.01, 3.24, 19.31)),
            "Greedy+Beam": ((0.042, 0.00, 1.05), (0.023, 0.00, 1.12)),
            "LeastAllocated": ((12.05, 13.23, 34.49), (13.51, 14.80, 36.01)),
            "MostAllocated": ((13.12, 12.37, 29.09), (14.54, 13.33, 31.50)),
        }
        self.assertEqual(set(t7), set(expected))
        for name, values in expected.items():
            for field, triplet in zip(("predicted", "measured"), values):
                self.assertEqual(tuple(t7[name][field][key] for key in ("mean", "median", "maximum")), triplet)
            self.assertEqual(t7[name]["reference"], "Greedy-C" if name in ("LeastAllocated", "MostAllocated") else "Exact")
            full_cohort = name in ("Fixed-frequency", "Sequential", "Greedy+Beam")
            self.assertEqual(t7[name]["measured_pairs_n"], 112 if full_cohort else 97)

    def test_archived_residency_targets_have_recalculable_errors(self):
        rows = read_json("evidence/validation/residency/archive_power_targets.json")["targets"]
        self.assertEqual(len(rows), 2)
        errors = []
        for row in rows:
            error = 100 * abs(row["predicted_power_W"] - row["measured_power_W"]) / row["measured_power_W"]
            self.assertAlmostEqual(error, row["absolute_error_percent"])
            errors.append(error)
        self.assertEqual(tuple(round(value, 2) for value in (mean(errors), median(errors), max(errors))),
                         (0.77, 0.77, 0.93))

    def test_retained_individual_predictions_match_plot_and_table(self):
        pairs = read_json("data/tables/predicted_efficiency_pairs.json")["pairs"]
        plot = {r["case"]: r for r in figure_data("solution_method_comparison.py")["rows"]}
        keys = {"fixed_frequency": "Fixed-frequency", "sequential": "Sequential", "greedy_construction": "Greedy-C", "greedy": "Greedy", "full": "Greedy+Beam"}
        _, table = tables.load_tables()
        self.assertEqual(len(pairs), 530)
        for row in pairs:
            observed = plot[row["case"]]
            self.assertEqual(row["modeled_utility_per_J"], observed[row["method"]]["ratio"])
            self.assertEqual(row["reference_modeled_utility_per_J"], observed["exact_ratio"])
            self.assertAlmostEqual(row["loss_percent"], 100 * (1 - row["modeled_utility_per_J"] / row["reference_modeled_utility_per_J"]))
        for method, label in keys.items():
            losses = [p["loss_percent"] for p in pairs if p["method"] == method]
            self.assertEqual(len(losses), 97 if method in ("greedy", "greedy_construction") else 112)
            for key, value in zip(("mean", "median", "maximum"), (mean(losses), median(losses), max(losses))):
                precision = 3 if label == "Greedy+Beam" and key == "mean" else 2
                self.assertEqual(round(value, precision), table[label]["predicted"][key])


class FigureDataTests(unittest.TestCase):
    def test_all_reference_figures_have_correct_hashes(self):
        manifest = read_json("figures/figure_manifest.json")
        self.assertEqual([r["figure"] for r in manifest["figures"]], list(range(1, 9)))
        self.assertEqual(manifest["figures"][7]["script"], "scalability.py")
        for row in manifest["figures"]:
            p = ROOT / "figures" / row["reference"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), row["reference_sha256"])
            self.assertTrue((ROOT / "figures" / f"plot_figure{row['figure']}.py").is_file())

    def test_canonical_response_rows_match_embedded_and_runtime_data(self):
        archived = figure_data("response_time_errors.py")
        saved = read_json("evidence/response_time/publication_data.json")
        runtime, _ = response_plot_module().load_statistics()
        self.assertEqual(saved, archived)
        self.assertEqual(saved["columns"], archived["columns"])
        self.assertEqual(saved["columns"], runtime["columns"])
        self.assertEqual(len(saved["rows"]), 69)
        self.assertEqual(len(runtime["rows"]), 69)
        self.assertEqual(len(set(row[0] for row in saved["rows"])), 69)
        self.assertEqual(len(set(row[1] for row in saved["rows"])), 23)
        self.assertEqual(Counter(row[2] for row in saved["rows"]), {0.3: 23, 0.6: 23, 0.8: 23})
        for expected, actual in zip(saved["rows"], runtime["rows"]):
            self.assertEqual(actual[:2], expected[:2])
            unchanged = [i for i, name in enumerate(saved["columns"])
                         if i >= 2 and name != "response_s"]
            np.testing.assert_allclose([actual[i] for i in unchanged],
                                       [expected[i] for i in unchanged], rtol=0, atol=1e-12)
        self.assertEqual(runtime["simulation_estimator"]["percentile"], 95)

        baseline = {row[0]: dict(zip(archived["columns"], row)) for row in archived["rows"]}
        components = {row["cell_id"]: row for row in
                      read_json("evidence/response_time/component_observations.json")["records"]}
        sample_metadata = {row["cell_id"]: row for row in
                           read_json("evidence/response_time/simulation_samples.json")["arrays"]}
        self.assertEqual(set(baseline), set(components))
        self.assertEqual(set(baseline), set(sample_metadata))
        self.assertEqual(set(baseline), {row[0] for row in saved["rows"]})
        for values in saved["rows"]:
            row = dict(zip(saved["columns"], values))
            old = baseline[row["cell_id"]]
            self.assertEqual((row["config_id"], row["load_fraction"]),
                             (old["config_id"], old["load_fraction"]))
            self.assertEqual(row["analytical_response_s"], old["analytical_response_s"])
            self.assertEqual(row, old)
            analytical = row["analytical_response_s"]

            # All 69 analytical decompositions can be checked from
            # their recorded model parameters, without an external fixture.
            parameters = sample_metadata[row["cell_id"]]["source_simulation_parameters"]
            batch = parameters["batch"]
            rate = parameters["rate_requests_s"]
            latency = parameters["L_s"]
            rho = rate * latency / batch
            expected_parts = [latency, (batch - 1) / (2 * rate),
                              (1 / batch + parameters["C2"]) * latency * rho / (2 * (1 - rho))]
            prediction = components[row["cell_id"]]["prediction_s"]
            np.testing.assert_allclose(
                [prediction[name] for name in ("execution", "batchfill", "queue")],
                expected_parts, rtol=0, atol=1e-12)
            self.assertEqual(prediction["response"], old["analytical_response_s"])
            self.assertAlmostEqual(sum(expected_parts), analytical, delta=1e-12)

    def test_admission_counts_losses_and_denominators(self):
        plot = recompute_admission(ROOT)["data"]
        saved = read_json("evidence/admission/publication_data.json")
        self.assertEqual(saved["policies"], plot["policies"])
        self.assertEqual(plot["denominator"], 112)
        self.assertEqual(plot["repetition"], 2)
        expected = [(47, 65, 0), (92, 20, 0), (95, 4, 13)]
        for policy, counts in zip(plot["policies"], expected):
            self.assertEqual(tuple(policy["counts"][k] for k in ("slo_met", "slo_not_met", "infeasible")), counts)
            self.assertEqual(sum(policy["counts"].values()), 112)
        self.assertEqual(round(plot["policies"][1]["loss"]["mean"], 2), 1.38)
        self.assertEqual(round(plot["policies"][2]["loss"]["mean"], 2), 3.60)
        self.assertEqual(plot["policies"][2]["loss"]["n"], 99)
        self.assertEqual(round(plot["shared_mc_wall_seconds"]), 140)

    def test_admission_figure_plots_workloads_and_offered_requests(self):
        spec = importlib.util.spec_from_file_location(
            "combined_admission_plot", ROOT / "figures/SLO_compliance.py")
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        figure, _ = renderer.make_figure(renderer.load_data())
        try:
            workloads, requests = figure.axes
            np.testing.assert_array_equal(
                [[bar.get_height() for bar in group] for group in workloads.containers],
                [[47, 92, 95], [65, 20, 4], [0, 0, 13]])
            expected = np.array([[756864, 772208, 683649], [62644, 47300, 34938], [0, 0, 100921]])
            np.testing.assert_allclose(
                [[bar.get_height() for bar in group] for group in requests.containers],
                100 * expected / 819508, rtol=0, atol=1e-10)
            footnotes = [label.get_text() for label in figure.texts]
            self.assertEqual(sum("140.4 s (shared)" in note for note in footnotes), 1)
            self.assertTrue(any("819,508 offered requests" in note for note in footnotes))
        finally:
            renderer.plt.close(figure)

    def test_occupancy_pairs_preserve_failures_and_reference(self):
        data = figure_data("kubernetes_policies.py")
        self.assertEqual(len(data["rows"]), 112)
        published = read_json("results/occupancy/publication_data.json")
        self.assertEqual(published["rows"], data["rows"])
        for policy, expected_mean in (("LeastAllocated", 12.05), ("MostAllocated", 13.12)):
            losses = [100 * (1 - row["results"][policy]["ratio"] / row["greedy_construction"]["ratio"]) for row in data["rows"] if row["results"][policy]["ratio"] is not None]
            self.assertEqual(len(losses), 97)
            self.assertEqual(round(mean(losses), 2), expected_mean)

    def test_scaling_means_and_individual_loss_formula(self):
        data = figure_data("scalability.py")
        saved = read_json("results/scaling/summary.json")["series"]
        expected_time = {"greedy": 5.20, "greedy_beam": 63.00, "exact_cold": 1308.08, "exact_initialized": 842.78}
        for method, rows in data.items():
            self.assertEqual(len(rows), 40)
            self.assertEqual(Counter(r["services"] for r in rows), {16: 10, 32: 10, 64: 10, 128: 10})
            self.assertEqual({r["case"] for r in rows}, {r["case"] for r in data["exact_cold"]})
            by_case = {r["case"]: r for r in saved[method]}
            for row in rows:
                self.assertAlmostEqual(row["loss_percent"], 100 * (1 - row["ratio"] / row["exact_ratio"]))
                for key in ("time_s", "ratio", "loss_percent"):
                    self.assertEqual(row[key], by_case[row["case"]][key])
            self.assertEqual(round(mean(r["time_s"] for r in rows if r["services"] == 128), 2), expected_time[method])
        self.assertEqual(round(mean(r["loss_percent"] for r in data["greedy_beam"] if r["services"] == 128), 2), .01)
        self.assertEqual(round(mean(r["loss_percent"] for r in data["greedy"] if r["services"] == 128), 2), 0.40)
        self.assertGreater(min(r["time_s"] for r in data["greedy_beam"] if r["services"] == 128), 45)


class RequestMeasurementsTests(unittest.TestCase):
    def test_retained_simulation_samples_reproduce_the_paper_reference_means(self):
        metadata = read_json("evidence/response_time/simulation_samples.json")
        publication = read_json("evidence/response_time/publication_data.json")
        plotted = {row[0]: dict(zip(publication["columns"], row)) for row in publication["rows"]}
        archive = ROOT / "evidence/response_time/simulated_window_means.npz"
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), metadata["archive_sha256"])
        self.assertEqual(len(plotted), 69)
        self.assertEqual(len(metadata["arrays"]), 69)
        self.assertEqual({entry["cell_id"] for entry in metadata["arrays"]}, set(plotted))
        total_samples = 0
        with np.load(archive, allow_pickle=False) as samples:
            self.assertEqual(set(samples.files), set(plotted))
            for entry in metadata["arrays"]:
                values = samples[entry["cell_id"]]
                self.assertEqual(values.shape, (1024,))
                self.assertEqual(entry["values"], len(values))
                self.assertTrue(np.all(np.isfinite(values) & (values > 0)))
                self.assertAlmostEqual(float(values.mean()),
                                       plotted[entry["cell_id"]]["response_s"], delta=1e-12)
                self.assertAlmostEqual(float(values.mean()), entry["mean_response_s"], delta=1e-12)
                digest = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
                self.assertEqual(digest, entry["array_sha256"])
                total_samples += len(values)
                window = entry["source_simulation_window"]
                self.assertEqual(window["duration_s"], 45)
                self.assertAlmostEqual(window["duration_s"], window["arrival_end_s"] - window["arrival_start_s"])
        self.assertEqual(total_samples, 69 * 1024)

    def test_observation_and_simulation_windows_are_consistent(self):
        components = read_json("evidence/response_time/component_observations.json")["records"]
        simulations = {row["cell_id"]: row["source_simulation_window"] for row in
                       read_json("evidence/response_time/simulation_samples.json")["arrays"]}
        self.assertEqual(len(components), 69)
        self.assertEqual(set(simulations), {row["cell_id"] for row in components})
        for record in components:
            with self.subTest(cell_id=record["cell_id"]):
                short = record["source_observation_windows"]["short"]
                long = record["source_observation_windows"]["long"]
                simulation = simulations[record["cell_id"]]
                for window, duration in ((short, 45), (long, 20 * 45)):
                    self.assertEqual(window["arrival_window_s"], duration)
                    self.assertAlmostEqual(window["arrival_end_s"] - window["arrival_start_s"], duration)
                self.assertEqual(simulation["duration_s"], 45)
                self.assertEqual(simulation["arrival_start_s"], short["arrival_start_s"])
                self.assertEqual(simulation["arrival_end_s"], short["arrival_end_s"])
                self.assertEqual(simulation["full_arrival_horizon_s"], long["arrival_end_s"])

    def test_retained_components_reproduce_available_errors(self):
        records = read_json("evidence/response_time/component_observations.json")["records"]
        plotted = {r[0]: r for r in figure_data("response_time_errors.py")["rows"]}
        self.assertEqual(len(records), 69)
        for record in records:
            row = plotted[record["cell_id"]]
            predicted = record["prediction_s"]
            observed = record["measured_short_components_s"]
            for component, column in (("execution", 6), ("batchfill", 7), ("queue", 9)):
                error = 100 * abs(predicted[component] - observed[component]) / observed["response"]
                self.assertAlmostEqual(error, row[column], places=10)
            long_error = 100 * abs(predicted["batchfill"] - record["measured_long_batchfill_s"]) / record["measured_long_response_s"]
            self.assertAlmostEqual(long_error, row[8], places=10)

    def test_request_compliance_uses_offered_denominator_for_every_policy(self):
        aggregates = recompute_admission(ROOT)["data"]
        blocks = read_json("evidence/admission/physical_blocks_r2.json")
        served_counts = Counter()
        within_counts = Counter()
        for block in blocks:
            policy = block["policy"]
            for replica in block["replicas"]:
                served_counts[policy] += replica["n"]
                within_counts[policy] += replica["met"]

        self.assertEqual(aggregates["request_denominator"], 819508)
        expected = [(756864, 62644, 0), (772208, 47300, 0), (683649, 34938, 100921)]
        fields = ("within_slo", "above_slo", "unserved")
        percent_fields = ("within_percent", "above_percent", "unserved_percent")
        for policy, counts in zip(aggregates["policies"], expected):
            requests = policy["requests"]
            self.assertEqual(requests["offered"], 819508)
            self.assertEqual(requests["served"], served_counts[policy["key"]])
            self.assertEqual(requests["within_slo"], within_counts[policy["key"]])
            self.assertEqual(tuple(requests[field] for field in fields), counts)
            self.assertEqual(sum(counts), requests["offered"])
            for field, count in zip(percent_fields, counts):
                self.assertAlmostEqual(requests[field], 100 * count / requests["offered"])


if __name__ == "__main__":
    unittest.main()
