"""CPU regressions for acquisition setup; never call CUDA or NVIDIA tools."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "scientific/hardware"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_module("hardware_test_window_guard", ROOT / "scientific/preparation/window_guard.py")


class AcquisitionSetupTests(unittest.TestCase):
    def run_cli(self, path, *arguments):
        return subprocess.run([sys.executable, str(path), *arguments],
                              capture_output=True, text=True, timeout=20)

    def test_cpu_entry_points_need_no_posix_lock_or_cuda_import(self):
        paths = ("profiling/batch_worker.py", "profiling/calibrate.py",
                 "execution/replay.py", "execution/summarize.py")
        for relative in paths:
            with self.subTest(entry_point=relative):
                result = self.run_cli(HARDWARE / relative, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_run_cannot_be_combined_with_gpu_execution(self):
        for mode in ("--dry-run", "--prepare"):
            with self.subTest(mode=mode):
                result = self.run_cli(HARDWARE / "execution/replay.py",
                                      "--selection", "unused.json", "--run", mode)
                self.assertEqual(result.returncode, 2)
                self.assertIn("not allowed with argument", result.stderr)

    def test_nonfinite_replay_windows_rejected_before_reading_selection(self):
        for relative in ("execution/replay.py", "profiling/gpu_support.py"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(entry_point=relative, value=value):
                    result = self.run_cli(HARDWARE / relative, "--selection",
                                          "unused.json", "--dry-run", "--duration-s=" + value)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("Invalid", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_gpu_entry_points_use_the_same_advisory_lock(self):
        # This cross-entry-point invariant prevents concurrent campaigns from
        # accepting distinct locks for the same physical device index.
        names = ("profiling/batch_worker.py", "profiling/calibrate.py",
                 "profiling/gpu_support.py", "execution/replay.py")
        locks = []
        for name in names:
            tree = ast.parse((HARDWARE / name).read_text(encoding="utf-8"))
            found = [node for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                     and any(isinstance(part, ast.Constant) and isinstance(part.value, str)
                             and part.value.endswith(".lock") for part in node.values)]
            self.assertEqual(len(found), 1, name)
            locks.append(ast.dump(found[0], include_attributes=False))
        self.assertEqual(len(set(locks)), 1)

    def test_batch_launch_hashes_the_distributed_worker(self):
        support = load_module("hardware_test_profiling_support", HARDWARE / "profiling/gpu_support.py")
        cleanup = load_module("hardware_test_cleanup", HARDWARE / "profiling/cleanup.py")
        with patch.dict(sys.modules, {"gpu_support": support, "cleanup": cleanup}):
            worker = load_module("hardware_test_batch_worker", HARDWARE / "profiling/batch_worker.py")
        fake_lock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=Mock())
        scratch = ROOT / "build/test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            base = Path(temporary)
            plan = base / "plan.json"
            plan.write_text(json.dumps({"power_cap_W": 165,
                "configurations": [{"lane": 0, "id": "fixture"}]}), encoding="utf-8")
            args = types.SimpleNamespace(outdir=base / "run", gpu=0, lane=0,
                                         plan=plan, only=None)
            written = []
            def smi(*arguments):
                if "-q" in arguments:
                    return "MIG Mode\nCurrent: Enabled\nAccounting"
                if "--query-gpu=power.limit" in arguments:
                    return "165"
                return "GPU-fixture"
            with (patch.dict(sys.modules, {"fcntl": fake_lock}),
                  patch.object(worker.os, "geteuid", return_value=1000, create=True),
                  patch("builtins.open", mock_open()),
                  patch.object(support, "assert_empty"),
                  patch.object(support, "smi", side_effect=smi),
                  patch.object(support, "atomic_json", side_effect=lambda path, value: written.append(value)),
                  patch.object(support, "NVML", side_effect=RuntimeError("stop before telemetry"))):
                with self.assertRaisesRegex(RuntimeError, "stop before telemetry"):
                    worker.run(args)
            launch = written[0]
            self.assertEqual(set(launch["source_sha256"]),
                             {"batch_worker.py", "gpu_support.py", "cleanup.py"})
            self.assertTrue(all(len(value) == 64 for value in launch["source_sha256"].values()))


class SimulationInputTests(unittest.TestCase):
    def test_invalid_simulation_parameters_are_rejected(self):
        for kwargs in ({"rate": float("nan")}, {"latency": float("inf")},
                       {"c2": -1}, {"batch": 1.5}, {"simulations": 0},
                       {"chunk_size": 0}, {"warmup_s": -1}, {"measured_s": 0}):
            parameters = dict(rate=2, latency=.1, c2=0, batch=1,
                              simulations=8, warmup_s=0, measured_s=1)
            parameters.update(kwargs)
            with self.subTest(parameters=kwargs), self.assertRaises(ValueError):
                guard.simulate_window_means(**parameters)

    def test_integral_batch_float_has_same_simulation_as_integer(self):
        options = dict(rate=2, latency=.1, c2=0, simulations=8,
                       warmup_s=0, measured_s=1)
        integer = guard.simulate_window_means(batch=2, **options)
        floating = guard.simulate_window_means(batch=2.0, **options)
        for expected, observed in zip(integer, floating):
            np.testing.assert_equal(expected, observed)

    def test_too_few_nonempty_windows_have_an_explicit_error(self):
        profile = {"physical_config_id": "fixture", "B": 1,
                   "points": [{"rate_requests_s": 1, "L_s": .1, "cs2": 0}]}
        with patch.object(guard, "simulate_window_means",
                          return_value=(np.full(8, np.nan), np.zeros(8))):
            with self.assertRaisesRegex(ValueError, "eight nonempty"):
                guard.quantile_record(profile, 1, simulations=8)


if __name__ == "__main__":
    unittest.main()
