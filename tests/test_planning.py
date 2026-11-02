"""CPU-only checks of the planner and the final paper's workload design."""

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from efficient_mig import build_model, plan_case, validate_plan

spec = importlib.util.spec_from_file_location("generate_workloads", ROOT / "scientific/preparation/generate_workloads.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class PlanningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "data/inputs/catalog.json").read_text())
        # Software fixture, not a measured workload or a paper experiment.
        cls.case = dict(case="test_software_fixture", gpus=2, cap_W=165., services=[
            dict(id=0, task="detection", qmin=.35, slo=.45, lam=10., ca2=1.)])
        cls.plan = plan_case(cls.case, cls.catalog, method="greedy")["results"]["greedy"]

    def test_returned_plan_satisfies_constraints(self):
        metrics = validate_plan(self.case, self.catalog, self.plan)
        self.assertGreater(metrics["ratio"], 0)
        self.assertEqual(metrics["replicas"], 1)

    def test_corrupted_objective_is_rejected(self):
        damaged = deepcopy(self.plan)
        damaged["ratio"] *= 1.1
        with self.assertRaisesRegex(ValueError, "Incorrect ratio"):
            validate_plan(self.case, self.catalog, damaged)

    def test_corrupted_deployment_metadata_is_rejected(self):
        # Metrics and lookup IDs remain plausible; the actual deployment field
        # must still agree with the calibrated alternative it claims to use.
        for field, value in (("f", 1), ("model", "another-variant"),
                             ("physical_config_id", "another-configuration"),
                             ("k", 4), ("response", -1.0)):
            with self.subTest(field=field):
                damaged = deepcopy(self.plan)
                groups = json.loads(damaged["assignment_json"])
                groups[0][field] = value
                damaged["assignment_json"] = json.dumps(groups)
                with self.assertRaisesRegex(ValueError, "Incorrect assignment field"):
                    validate_plan(self.case, self.catalog, damaged)

    def test_invalid_quality_and_boolean_parameters_are_rejected(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.subTest(qmin=value):
                case = deepcopy(self.case)
                case["services"][0]["qmin"] = value
                with self.assertRaisesRegex(ValueError, "minimum quality"):
                    build_model(case, self.catalog)
        case = dict(self.case, gpus=True)
        with self.assertRaisesRegex(ValueError, "gpu_count"):
            build_model(case, self.catalog)

    def test_empty_workload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one service"):
            plan_case(dict(self.case, services=[]), self.catalog)

    def test_compiled_workload_is_independent_of_caller_mutation(self):
        case = deepcopy(self.case)
        model = build_model(case, self.catalog)
        case["services"][0]["lam"] *= 2
        self.assertTrue(model.validate(self.plan))

    def test_invalid_group_rates_and_counts_are_rejected(self):
        model = build_model(self.case, self.catalog)
        original = json.loads(self.plan["assignment_json"])
        for field, value in (("rate", float("nan")), ("rate", float("inf")),
                             ("rate", True), ("n", float("nan")),
                             ("n", float("inf")), ("g", False)):
            with self.subTest(field=field, value=value):
                groups = deepcopy(original)
                groups[0][field] = value
                with self.assertRaises(ValueError):
                    model.from_groups(groups)

    def test_grid_replacement_does_not_reuse_cached_quantiles(self):
        temporary_root = ROOT / "build" / "test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            grid = Path(folder) / "grid.csv"
            grid.write_bytes((ROOT / "data/inputs/admission_grid.csv").read_bytes())
            first = build_model(self.case, self.catalog, grid=grid)
            self.assertTrue(first.options)
            with grid.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            # Deliberately infeasible software fixture, never a paper input.
            for row in rows:
                row["q95_1024"] = 1e6
            with grid.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            second = build_model(self.case, self.catalog, grid=grid)
            self.assertFalse(second.options)
            self.assertTrue(first.validate(self.plan))

    def test_response_domain_checks_survive_optimized_python(self):
        # Run with -O: scientific validity must not depend on Python asserts.
        import subprocess
        source = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from rate_intervals import response_values; "
            "response_values({'B': 1, 'points': [[1, .01, 1], [2, .01, 1]]}, [3])"
        )
        result = subprocess.run([sys.executable, "-O", "-c", source,
                                 str(ROOT / "scientific/planning")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the measured domain", result.stderr)

    def test_admission_modes_remain_independent(self):
        with patch.dict(os.environ, {"ADMISSION_QUANTILE_FIELD": "q99_1024"}):
            first = build_model(self.case, self.catalog, admission="p95")
            build_model(self.case, self.catalog, admission="p99")
            build_model(self.case, self.catalog, admission="nominal")
            again = build_model(self.case, self.catalog, admission="p95")
            self.assertEqual(os.environ["ADMISSION_QUANTILE_FIELD"], "q99_1024")
        first.validate(self.plan)
        again.validate(self.plan)
        self.assertEqual(first.options, again.options)

    def test_occupancy_draws_from_all_candidates_tied_at_best_priority(self):
        model = build_model(self.case, self.catalog)
        rate = self.case["services"][0]["lam"]
        tolerance = 1e-9
        choices = [o for o in model.options if o["g"] == 0
                   and o["lower"] - tolerance <= rate <= o["upper"] + tolerance
                   and o["k"] <= 4
                   and model.active_base(o["f"]) + o["peak"] <= model.cap + tolerance]
        self.assertGreater(len({o["k"] for o in choices}), 1)
        for method in ("least-allocated", "most-allocated"):
            for seed in (0, 1, 7, 101):
                with self.subTest(method=method, seed=seed):
                    expected_index = random.Random(seed).randrange(len(choices))
                    record = plan_case(self.case, self.catalog, method=method, random_seed=seed)
                    result = record["results"][method]
                    model.validate(result)
                    group = json.loads(result["assignment_json"])[0]
                    trace = result["insertion_trace"][0]
                    self.assertEqual(group["rid"], choices[expected_index]["rid"])
                    self.assertEqual(trace["random_pool_size"], len(choices))
                    self.assertEqual(trace["random_choice_index"], expected_index)
                    self.assertEqual(trace["free_slices_after"], 4 - group["k"])
                    self.assertEqual(record["random_seed"], seed)
                    again = plan_case(self.case, self.catalog, method=method, random_seed=seed)
                    self.assertEqual(result["assignment_json"], again["results"][method]["assignment_json"])

    def test_occupancy_requires_explicit_seed(self):
        for method in ("least-allocated", "most-allocated"):
            with self.assertRaisesRegex(ValueError, "explicit integer random_seed"):
                plan_case(self.case, self.catalog, method=method)

    def test_greedy_c_matches_greedy_construction(self):
        direct = plan_case(self.case, self.catalog, method="greedy-c")["results"]["greedy_construction"]
        staged = plan_case(self.case, self.catalog, method="greedy")["results"]["greedy_construction"]
        self.assertAlmostEqual(direct["ratio"], staged["ratio"], places=12)

    def test_complete_small_beam_has_no_implicit_time_limit(self):
        record = plan_case(self.case, self.catalog, method="greedy-beam")
        self.assertIsNone(record["beam_seconds"])
        self.assertTrue(record["results"]["beam"]["beam_finished"])
        self.assertGreaterEqual(record["results"]["full"]["ratio"], self.plan["ratio"])
        validate_plan(self.case, self.catalog, record["results"]["full"])

    def test_core_does_not_import_gurobi(self):
        self.assertNotIn("gurobipy", sys.modules)

    def test_exact_boundary_cleanup_preserves_traffic(self):
        model = build_model(self.case, self.catalog)
        option = next(o for o in model.options if o["k"] == 1 and o["lower"] * 2 < o["upper"])
        group = dict(option, n=2, rate=option["upper"])
        reduced = model.remove_boundary_surplus([group])[0]
        self.assertEqual(reduced["n"], 1)
        self.assertEqual(reduced["rate"], group["rate"])
        self.assertLessEqual(model.totals([reduced])[0], model.totals([group])[0])
        above = dict(group, rate=option["upper"] + 1e-5)
        self.assertEqual(model.remove_boundary_surplus([above])[0]["n"], 2)

    def test_exact_boundary_cleanup_yields_valid_guarded_deployment(self):
        initial = build_model(self.case, self.catalog)
        option = next(o for o in initial.options if o["k"] == 1 and o["lower"] * 2 < o["upper"])
        case = deepcopy(self.case)
        case["services"][0]["lam"] = option["upper"]
        model = build_model(case, self.catalog)
        selected = next(o for o in model.options if o["g"] == 0
                        and o["physical_config_id"] == option["physical_config_id"]
                        and o["alternative_id"] == option["alternative_id"])
        reduced = model.remove_boundary_surplus([dict(selected, n=2, rate=option["upper"])])
        self.assertEqual(reduced[0]["n"], 1)
        plan = model.from_groups(reduced)
        self.assertTrue(model.validate(plan))


class WorkloadDesignChecks(unittest.TestCase):
    def test_comparison_design_draws_independently_within_each_condition(self):
        cases = generator.comparison_cases(seed=1234)
        self.assertEqual(len(cases), 112)
        groups = Counter((c["cap_W"], c["response_regime"], c["arrival_multiplier"]) for c in cases)
        self.assertEqual(len(groups), 8)
        self.assertEqual(set(groups.values()), {14})
        vectors = set()
        draws = defaultdict(set)
        for case in cases:
            condition = (case["cap_W"], case["response_regime"], case["arrival_multiplier"])
            draws[condition].add(case["draw"])
            self.assertEqual(len(case["services"]), 7)
            vectors.add(tuple(s["base_rate"] for s in case["services"]))
            for service in case["services"]:
                self.assertAlmostEqual(service["lam"], service["base_rate"] * case["arrival_multiplier"])
        self.assertEqual(len(vectors), 112)
        self.assertTrue(all(values == set(range(14)) for values in draws.values()))
        self.assertEqual(cases, generator.comparison_cases(seed=1234))
        self.assertNotEqual(cases, generator.comparison_cases(seed=4321))

    def test_missing_scaling_reference_is_not_imputed(self):
        catalog = json.loads((ROOT / "data/inputs/catalog.json").read_text())
        catalog["profiles"] = [p for p in catalog["profiles"]
                               if p["physical_config_id"] != "dl_b1_k2_f900"]
        with self.assertRaisesRegex(ValueError, "measured deeplabv3_resnet50, batch 1"):
            generator.scaling_cases(catalog, range(10))

    def test_scaling_design_has_ten_nested_sequences_with_measured_references(self):
        catalog = json.loads((ROOT / "data/inputs/catalog.json").read_text())
        seeds = range(1234, 1244)
        cases = generator.scaling_cases(catalog, seeds)
        self.assertEqual(len(cases), 40)
        self.assertEqual(Counter(c["n"] for c in cases), {16: 10, 32: 10, 64: 10, 128: 10})
        self.assertEqual(cases, generator.scaling_cases(catalog, seeds))
        self.assertNotEqual(cases, generator.scaling_cases(catalog, range(4321, 4331)))
        references = {p["task"]: p for p in catalog["profiles"]
                      if p["physical_config_id"] in ("fr_b1_k2_f900", "dl_b1_k2_f900")}
        for offset in range(0, 40, 4):
            full = cases[offset + 3]["services"]
            for case in cases[offset:offset + 4]:
                self.assertEqual(case["services"], full[:case["n"]])
                self.assertEqual(case["gpus"], case["n"] // 2)
                self.assertEqual(case["cap_W"], 165.)
                for service in case["services"]:
                    reference = references[service["task"]]
                    self.assertEqual(service["reference"]["physical_config_id"],
                                     reference["physical_config_id"])
                    self.assertEqual(service["reference"]["busy_L_s"], reference["busy_L_s"])
                    self.assertGreaterEqual(service["beta"], .25)
                    self.assertLessEqual(service["beta"], .55)
                    self.assertAlmostEqual(service["lam"], 1.2 * service["beta"] / reference["busy_L_s"])
        # A caller can adjust one generated workload without changing another.
        cases[0]["services"][0]["reference"]["busy_L_s"] = 1.
        self.assertNotEqual(cases[0]["services"][0]["reference"],
                            cases[1]["services"][0]["reference"])

    def test_ambiguous_and_invalid_scaling_references_are_rejected(self):
        catalog = json.loads((ROOT / "data/inputs/catalog.json").read_text())
        index = next(i for i, p in enumerate(catalog["profiles"])
                     if p["physical_config_id"] == "dl_b1_k2_f900")
        duplicate = deepcopy(catalog)
        duplicate["profiles"].append(deepcopy(duplicate["profiles"][index]))
        with self.assertRaisesRegex(ValueError, "exactly one measured deeplabv3_resnet50"):
            generator.scaling_cases(duplicate, [1234])
        for field, value in (("measurement_status", "estimated"), ("physical_config_id", ""),
                             ("busy_L_s", 0.), ("busy_L_s", -1.),
                             ("busy_L_s", float("nan")), ("busy_L_s", float("inf")),
                             ("busy_L_s", True), ("B", True)):
            with self.subTest(field=field, value=value):
                damaged = deepcopy(catalog)
                damaged["profiles"][index][field] = value
                with self.assertRaisesRegex(ValueError, "Scaling reference"):
                    generator.scaling_cases(damaged, [1234])

    def test_scaling_cli_generates_forty_new_workloads(self):
        import subprocess
        temporary_root = ROOT / "build" / "test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            output = Path(folder) / "scaling.json"
            command = [sys.executable, str(ROOT / "scientific/preparation/generate_workloads.py"),
                       "--catalog", str(ROOT / "data/inputs/catalog.json"), "--family", "scaling",
                       "--seed", "1234", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads(output.read_text())
            self.assertEqual(len(document["cases"]), 40)
            self.assertEqual(document["cases"][0]["case"], "generated_s1234_n16")
            self.assertEqual(document["cases"][-1]["case"], "generated_s1243_n128")
            self.assertIn("New generated inputs", document["scope"])


class CustomCatalogFrequencyChecks(unittest.TestCase):
    def test_baseline_only_frequencies_do_not_abort_available_profile_trials(self):
        # Software fixture: retain a valid single-frequency profile catalog
        # while keeping all independently measured device baselines.
        catalog = json.loads((ROOT / "data/inputs/catalog.json").read_text())
        catalog["profiles"] = [p for p in catalog["profiles"] if p["freq_MHz"] == 1200]
        catalog["expected_profiles"] = len(catalog["profiles"])
        catalog["exclusions"] = []
        case = dict(case="single_frequency_fixture", gpus=2, cap_W=165., services=[
            dict(id=0, task="detection", qmin=.35, slo=.45, lam=10., ca2=1.)])
        model = build_model(case, catalog)
        self.assertEqual(set(model.base_by_frequency), {900, 1200, 1440})
        self.assertTrue(model.options)

        for method in ("fixed-frequency", "sequential"):
            calls = []

            def fake_solve(compiled, *, seed_plan, frozen_groups=None, **kwargs):
                # Exercise real catalog compilation, initialization, conversion
                # and validation; only the licensed optimizer is replaced.
                compiled.validate(seed_plan)
                calls.append((set(p["freq_MHz"] for p in compiled.profiles.values()),
                              frozen_groups is not None))
                return dict(seed_plan, status="feasible_limit", seconds=0.0)

            with self.subTest(method=method), patch.object(type(model), "solve", fake_solve):
                result = plan_case(case, catalog, method=method)["results"]
                self.assertEqual([r["frequency_MHz"] for r in result["fixed_frequency_trials"]], [1200])
                self.assertEqual(calls, [({1200}, False)] +
                                 ([({1200}, True)] if method == "sequential" else []))
                selected = result["sequential" if method == "sequential" else "fixed_frequency"]
                self.assertTrue(model.validate(selected))


if __name__ == "__main__":
    unittest.main()
