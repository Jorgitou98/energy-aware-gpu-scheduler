"""CPU-only CLI round trips and saved-input binding checks."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_case_cli", ROOT / "scripts/run_case.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class PlanningCLI(unittest.TestCase):
    def test_bounded_beam_saved_run_round_trip(self):
        temporary_root = ROOT / "build/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            folder = Path(folder)
            cases, output = folder / "cases.json", folder / "bounded.json"
            cases.write_text(json.dumps({"cases": [dict(
                case="software_fixture", gpus=2, cap_W=165., services=[dict(
                    id=0, task="detection", qmin=.35, slo=.45, lam=10., ca2=1.)])]}),
                encoding="utf-8")
            base = [sys.executable, str(ROOT / "scripts/run_case.py"),
                    "--cases", str(cases), "--case", "software_fixture"]
            run = subprocess.run(base + ["--method", "greedy-beam-bounded",
                                         "--output", str(output)],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            document = json.loads(output.read_text())
            self.assertEqual(document["method"], "greedy-beam-bounded")
            self.assertIsNone(document["beam_seconds"])
            self.assertEqual(document["results"]["full"]["algorithm"], "greedy_and_beam_bounded")
            source = ROOT / "scientific/planning/heuristics_bounded.py"
            self.assertEqual(document["scientific_source_sha256"][source.name],
                             hashlib.sha256(source.read_bytes()).hexdigest())
            validation = subprocess.run(base + ["--validate-plan", str(output)],
                                        capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(validation.returncode, 0, validation.stderr)
            checked = json.loads(validation.stdout)
            self.assertTrue(checked["validated"])
            self.assertEqual(checked["selected_result"], "full")
            self.assertEqual(checked["ratio"], document["results"]["full"]["ratio"])

    def test_saved_run_and_standalone_plan_round_trip(self):
        temporary_root = ROOT / "build/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            folder = Path(folder)
            cases, output, raw = (folder / name for name in ("cases.json", "run.json", "plan.json"))
            cases.write_text(json.dumps({"cases": [dict(
                case="software_fixture", gpus=2, cap_W=165., services=[dict(
                    id=0, task="detection", qmin=.35, slo=.45, lam=10., ca2=1.)])]}),
                encoding="utf-8")
            base = [sys.executable, str(ROOT / "scripts/run_case.py"),
                    "--cases", str(cases), "--case", "software_fixture"]
            run = subprocess.run(base + ["--method", "greedy", "--output", str(output)],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            document = json.loads(output.read_text())
            raw.write_text(json.dumps(document["results"]["greedy"]), encoding="utf-8")
            for path, expected in ((output, "greedy"), (raw, None)):
                with self.subTest(format=path.name):
                    validation = subprocess.run(base + ["--validate-plan", str(path)],
                                                capture_output=True, text=True, cwd=ROOT)
                    self.assertEqual(validation.returncode, 0, validation.stderr)
                    checked = json.loads(validation.stdout)
                    self.assertTrue(checked["validated"])
                    self.assertEqual(checked["selected_result"], expected)
                    self.assertEqual(checked["ratio"], document["results"]["greedy"]["ratio"])

    def test_saved_record_binding_and_failed_result_are_rejected(self):
        record = dict(case="case", admission="p95", method="greedy",
                      input_sha256=dict(catalog="catalog-hash", cases="cases-hash", grid="grid-hash"),
                      results=dict(greedy={"assignment_json": "[]"}))
        hashes = dict(record["input_sha256"])
        for field in ("case", "admission", "catalog", "cases", "grid"):
            damaged = deepcopy(record)
            if field in ("case", "admission"):
                damaged[field] = "changed"
            else:
                damaged["input_sha256"][field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "different"):
                cli.saved_plan(damaged, case="case", admission="p95", input_hashes=hashes)
        record["results"]["greedy"] = {"status": "construction_failed"}
        with self.assertRaisesRegex(ValueError, "no feasible deployment"):
            cli.saved_plan(record, case="case", admission="p95", input_hashes=hashes)


if __name__ == "__main__":
    unittest.main()
