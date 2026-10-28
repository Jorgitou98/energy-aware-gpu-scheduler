"""Regression checks for scaling records, descriptive statistics and inputs."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scaling_results", ROOT / "src/efficient_mig/scaling_results.py")
SCALING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALING)
FILES = ("results/scaling/summary.json", "results/scaling/observations.csv",
         "results/scaling/summary.csv", "data/inputs/scaling_cases.json",
         "data/inputs/catalog.json", "figures/scalability.py", "results/scaling/protocol.json")
FILES += tuple(path.relative_to(ROOT).as_posix()
               for directory in ("cases", "source")
               for path in sorted((ROOT / "results/scaling" / directory).glob("*.json")))


class ScalingResultsTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "build/test_tmp"
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="scaling_", dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)

    def read(self, name="results/scaling/summary.json"):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def write(self, data, name="results/scaling/summary.json"):
        (self.root / name).write_text(json.dumps(data), encoding="utf-8")

    def assertFailure(self, name):
        result = SCALING.audit_scaling(self.root)
        self.assertFalse(result["passed"])
        self.assertTrue(any(row["name"] == "scaling/" + name for row in result["failures"]), result["failures"])
        return result

    def test_current_records_and_all_statistics_agree(self):
        result = SCALING.audit_scaling(self.root)
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["summary"]["workloads"], 40)
        self.assertEqual(result["summary"]["observations"], 160)
        self.assertEqual(result["summary"]["source_observations"], 160)
        self.assertEqual(result["summary"]["cases_per_size"], 10)
        means = [group["time_s"]["mean"] for group in result["summary"]["aggregates"]["greedy_beam"]]
        self.assertEqual([round(value, 2) for value in means], [11., 18., 31., 63.])

    def test_source_checksum_detects_byte_changes(self):
        protocol = self.read("results/scaling/protocol.json")
        source = protocol["result_source"]["files"]["greedy_beam"]
        path = self.root / source["path"]
        path.write_bytes(path.read_bytes() + b"\n")
        result = self.assertFailure("source/greedy_beam/sha256")
        self.assertEqual(result["failed"], 1)

    def test_rehashed_source_time_must_still_match_the_observation(self):
        name = "results/scaling/protocol.json"
        protocol = self.read(name)
        source = protocol["result_source"]["files"]["greedy_beam"]
        export = self.read(source["path"])
        row = export["cases"][0]
        row["timings_seconds"]["total_with_preparation_seconds"] += 1
        self.write(export, source["path"])
        source["sha256"] = hashlib.sha256((self.root / source["path"]).read_bytes()).hexdigest()
        self.write(protocol, name)
        result = self.assertFailure("source/greedy_beam/" + row["case"] + "/time_s")
        self.assertEqual(result["failed"], 1)

    def test_changed_time_is_detected_against_statistics(self):
        data = self.read()
        data["series"]["greedy_beam"][0]["time_s"] += 1.
        self.write(data)
        self.assertFailure("greedy_beam/16/time_s/mean")

    def test_case_record_cannot_substitute_a_different_observation(self):
        name = "results/scaling/cases/s359_n128.json"
        data = self.read(name)
        data["methods"]["greedy_beam"]["time_s"] += 1.
        self.write(data, name)
        self.assertFailure("case-records/s359_n128/greedy_beam/time_s")

    def test_case_record_source_hash_must_identify_the_summary(self):
        name = "results/scaling/cases/s359_n128.json"
        data = self.read(name)
        data["source"]["sha256"] = "0" * 64
        self.write(data, name)
        self.assertFailure("case-records/s359_n128/source")

    def test_missing_case_record_is_detected(self):
        (self.root / "results/scaling/cases/s359_n128.json").unlink()
        self.assertFailure("case-records/identities")

    def test_population_standard_deviation_is_rejected(self):
        data = self.read()
        group = data["aggregates"]["greedy_beam"][0]
        group["time_s"]["sample_std"] *= math.sqrt(9 / 10)
        self.write(data)
        self.assertFailure("greedy_beam/16/time_s/sample_std")

    def test_wrong_quartile_is_detected_without_changing_mean(self):
        data = self.read()
        data["aggregates"]["exact_cold"][3]["time_s"]["q1"] += 1.
        self.write(data)
        self.assertFailure("exact_cold/128/time_s/q1")

    def test_reference_substitution_fails_even_with_consistent_loss(self):
        data = self.read()
        row = data["series"]["greedy_beam"][0]
        row["exact_ratio"] *= 1.01
        row["loss_percent"] = 100 * (1 - row["ratio"] / row["exact_ratio"])
        self.write(data)
        result = self.assertFailure("greedy_beam/" + row["case"] + "/paired-exact-reference")
        self.assertFalse(any(f["name"].endswith("/" + row["case"] + "/loss-formula") for f in result["failures"]))

    def test_missing_case_fails_even_if_declared_totals_are_unchanged(self):
        data = self.read()
        data["series"]["greedy"].pop()
        self.write(data)
        self.assertFailure("greedy/identities")

    def test_duplicate_csv_observation_is_rejected(self):
        path = self.root / "results/scaling/observations.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[-1] = rows[-2]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.assertFailure("csv/observations/unique")

    def test_csv_summary_cannot_hide_a_different_mean(self):
        path = self.root / "results/scaling/summary.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        row = rows[0]
        row["time_s_mean"] = str(float(row["time_s_mean"]) + .1)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.assertFailure("csv/" + row["method"] + "/" + row["services"] + "/time_s/mean")

    def test_nested_sequence_is_checked_separately_from_rate_formula(self):
        name = "data/inputs/scaling_cases.json"
        data = self.read(name)
        row = next(c for c in data["cases"] if c["n"] == 16)
        service = row["services"][0]
        service["beta"] += .001
        service["lam"] = row["demand_multiplier"] * service["beta"] / service["reference"]["busy_L_s"]
        self.write(data, name)
        self.assertFailure("inputs/" + row["case"] + "/nested-prefix")

    def test_coherent_reference_and_rate_changes_fail_against_catalog(self):
        name = "data/inputs/scaling_cases.json"
        data = self.read(name)
        for row in data["cases"]:
            for service in row["services"]:
                service["reference"]["busy_L_s"] *= 2
                service["lam"] /= 2
        self.write(data, name)
        first = data["cases"][0]
        result = self.assertFailure("inputs/" + first["case"] + "/service0/reference-latency")
        self.assertEqual(sum(f["name"].endswith("/reference-latency") for f in result["failures"]), 2400)
        self.assertFalse(any(f["name"].endswith(("/arrival-rate", "/nested-prefix")) for f in result["failures"]))

    def test_reference_profile_must_be_unique_and_measured(self):
        name = "data/inputs/catalog.json"
        original = self.read(name)
        identifier = "fr_b1_k2_f900"
        for change in ("duplicate", "unmeasured"):
            with self.subTest(change=change):
                data = json.loads(json.dumps(original))
                profile = next(p for p in data["profiles"] if p["physical_config_id"] == identifier)
                if change == "duplicate":
                    data["profiles"].append(profile.copy())
                    expected = "/unique-reference"
                else:
                    profile["measurement_status"] = "inferred"
                    expected = "/measurement_status"
                self.write(data, name)
                self.assertFailure("catalog/" + identifier + expected)

    def test_rehashed_figure_data_must_still_match_records(self):
        data = self.read()["series"]
        data["greedy_beam"][0]["time_s"] += 1.
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        source_hash = hashlib.sha256((self.root / "results/scaling/summary.json").read_bytes()).hexdigest()
        text = ("DATA = " + repr(data) + "\nDATA_SHA256 = " + repr(hashlib.sha256(canonical).hexdigest())
                + "\nSOURCE_SHA256 = " + repr(source_hash) + "\n")
        (self.root / "figures/scalability.py").write_text(text, encoding="utf-8")
        result = self.assertFailure("figure/literal-data")
        self.assertFalse(any(f["name"] == "scaling/figure/data-hash" for f in result["failures"]))


if __name__ == "__main__":
    unittest.main()
