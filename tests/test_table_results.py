"""Verify observations statistics and reject data corruption hidden by aggregates."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from efficient_mig.table_results import recompute_planning_timings, recompute_tables


INPUTS = (
    "data/tables/published_tables.json", "data/tables/response_group_statistics.json",
    "data/tables/power_holdout_original.csv", "data/tables/predicted_efficiency_pairs.json",
    "results/occupancy/publication_data.json", "data/tables/validation_errors.json",
    "data/tables/measured_efficiency_pairs.json", "results/planning/timings.json",
    "results/planning/published_timings.json", "results/planning/full_timing/timing_summary.json",
)


class TableResultsTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "build" / "test_table_results"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in INPUTS:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)

    def change(self, name, callback):
        path = self.root / name
        data = json.loads(path.read_text(encoding="utf-8"))
        callback(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_recomputed_tables_match_publication_and_preserve_unknowns(self):
        result = recompute_tables(self.root)
        published = json.loads((ROOT / "data/tables/published_tables.json").read_text(encoding="utf-8"))
        self.assertEqual(result["table6"], published["table6"])
        self.assertEqual(result["table7"], published["table7"])
        self.assertEqual(result["audit"]["counts"]["table6_errors"], 183)
        self.assertEqual(result["audit"]["counts"]["table7_losses"], 530)
        self.assertTrue(all(c["passed"] for c in result["audit"]["checks"]))
        self.assertIsNone(result["table6"]["Residency"]["response"])
        self.assertEqual([row["measured_pairs_n"] for row in result["table7"].values()],
                         [112, 112, 97, 97, 112, None, None])
        for group, count in (("Paired MIG", 53), ("Full GPU", 42), ("Batched MIG", 69), ("Residency", 2)):
            power = result["audit"]["table6_full_precision"][group]["power"]
            self.assertEqual(power["source"], "data/tables/power_holdout_original.csv")
            self.assertEqual(power["records_n"], count)
        for name in ("LeastAllocated", "MostAllocated"):
            occupancy = result["audit"]["table7_full_precision"][name]
            self.assertEqual(occupancy["data_representation"], "publication summary only")
            self.assertEqual(occupancy["records_n"], 0)
            self.assertEqual(occupancy["predicted_pairs_n"], 97)

    def test_power_comparison_recalculates_each_supplied_error(self):
        path = self.root / "data/tables/power_holdout_original.csv"
        rows = path.read_text(encoding="utf-8").splitlines()
        original, _, _ = rows[1].rpartition(",")
        rows[1] = original + ",0.0"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "supplied error differs from computed error"):
            recompute_tables(self.root)

    def test_individual_error_change_is_not_hidden_by_embedded_targets(self):
        def corrupt(data):
            group = data["groups"][0]
            group["records"][0]["absolute_error_percent"] += 10
            group["target_statistics"]["mean"] += 10 / group["n"]
        self.change("data/tables/validation_errors.json", corrupt)
        with self.assertRaisesRegex(ValueError, "Table 6/Paired MIG/response/mean"):
            recompute_tables(self.root)

    def test_duplicate_errors_are_rejected_even_if_statistics_do_not_change(self):
        def corrupt(data):
            data["groups"][0]["records"][1]["id"] = data["groups"][0]["records"][0]["id"]
        self.change("data/tables/validation_errors.json", corrupt)
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            recompute_tables(self.root)

    def test_measured_response_cannot_change_without_recomputing_its_error(self):
        def corrupt(data):
            group = next(g for g in data["groups"]
                         if (g["group"], g["metric"]) == ("Batched MIG", "response"))
            group["records"][0]["measured_response_s"] *= 2
        self.change("data/tables/validation_errors.json", corrupt)
        with self.assertRaisesRegex(ValueError, r"Table 6/Batched MIG/response/.+/absolute_error_percent"):
            recompute_tables(self.root)

    def test_holdout_record_identity_must_agree_with_its_group(self):
        def corrupt(data):
            group = next(g for g in data["groups"]
                         if (g["group"], g["metric"]) == ("Paired MIG", "response"))
            row = group["records"][0]
            row["label"] = "full_" + row["label"]
            row["id"] = "/".join((row["host"], row["label"], row["actor"]))
        self.change("data/tables/validation_errors.json", corrupt)
        with self.assertRaisesRegex(ValueError, "Paired MIG/response: unexpected holdout target group"):
            recompute_tables(self.root)

    def test_nonfinite_values_are_rejected(self):
        self.change("data/tables/validation_errors.json",
                    lambda d: d["groups"][0]["records"][0].update(absolute_error_percent=float("nan")))
        with self.assertRaisesRegex(ValueError, "finite number"):
            recompute_tables(self.root)

    def test_wrong_case_is_rejected_even_if_losses_are_unchanged(self):
        def corrupt(data):
            method = next(m for m in data["methods"] if m["method"] == "Greedy-C")
            method["records"][0]["case"] = "unified_000"
        self.change("data/tables/measured_efficiency_pairs.json", corrupt)
        with self.assertRaisesRegex(ValueError, "cohort differs"):
            recompute_tables(self.root)

    def test_attempted_failures_are_not_silently_dropped(self):
        def corrupt(data):
            method = next(m for m in data["methods"] if m["method"] == "Greedy")
            method["excluded_cases"].pop()
        self.change("data/tables/measured_efficiency_pairs.json", corrupt)
        with self.assertRaisesRegex(ValueError, "exclusions"):
            recompute_tables(self.root)

    def test_measured_cohort_is_not_inferred_from_predicted_records(self):
        def corrupt(data):
            data["table7"]["MostAllocated"]["measured_pairs_n"] = 97
        self.change("data/tables/published_tables.json", corrupt)
        with self.assertRaisesRegex(ValueError, "measured cohort size remains unreported"):
            recompute_tables(self.root)

    def test_inconsistent_archived_exact_reference_is_rejected(self):
        def corrupt(data):
            repetition = data["methods"][1]["records"][0]["repetitions"][0]
            repetition["exact"]["completed_utility"] *= 1.01
            repetition["exact"]["energy_through_drain_J"] *= 1.01
        self.change("data/tables/measured_efficiency_pairs.json", corrupt)
        with self.assertRaisesRegex(ValueError, "inconsistent Exact observations"):
            recompute_tables(self.root)

    def test_planning_means_are_recomputed_from_every_attempt(self):
        result = recompute_planning_timings(self.root)
        self.assertEqual(result["counts"]["planning_times"], 224)
        self.assertAlmostEqual(result["mean_planning_seconds"]["fixed_frequency"], 3.889549093354227)
        self.assertAlmostEqual(result["mean_planning_seconds"]["sequential"], 3.927745122246311)
        self.assertTrue(all(c["passed"] for c in result["checks"]))

    def test_timings_preserving_mean_must_still_equal_their_stage_sum(self):
        def corrupt(data):
            data["rows"][0]["time_s"] += .1
            data["rows"][2]["time_s"] -= .1
        self.change("results/planning/timings.json", corrupt)
        with self.assertRaisesRegex(ValueError, "maximum_stage_sum_difference"):
            recompute_planning_timings(self.root)

    def test_changed_preprocessing_with_same_total_is_rejected(self):
        def corrupt(data):
            data["rows"][0]["preprocessing_seconds"] += .1
            data["rows"][0]["search_seconds"] -= .1
        self.change("results/planning/timings.json", corrupt)
        with self.assertRaisesRegex(ValueError, "maximum_source_preprocessing_difference"):
            recompute_planning_timings(self.root)

    def test_swapped_totals_with_preserved_mean_are_rejected(self):
        def corrupt(data):
            first, second = data["rows"][0], data["rows"][2]
            first["time_s"], second["time_s"] = second["time_s"], first["time_s"]
            for row in (first, second):
                row["search_seconds"] = row["time_s"] - row["preprocessing_seconds"]
        self.change("results/planning/timings.json", corrupt)
        with self.assertRaisesRegex(ValueError, "maximum_source_total_difference"):
            recompute_planning_timings(self.root)

    def test_timing_duplicate_case_does_not_pass_with_correct_row_count(self):
        def corrupt(data):
            data["rows"][2]["case"] = data["rows"][0]["case"]
        self.change("results/planning/timings.json", corrupt)
        with self.assertRaisesRegex(ValueError, "duplicate case/method"):
            recompute_planning_timings(self.root)


if __name__ == "__main__":
    unittest.main()
