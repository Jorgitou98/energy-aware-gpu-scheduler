"""CPU-only rejection of altered deployments and incompatible replay campaigns."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "scientific/hardware/execution"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = load("replay_validation_support", EXECUTION / "gpu_support.py")
profiling = load("replay_validation_profiling", ROOT / "scientific/hardware/profiling/gpu_support.py")
with patch.dict(sys.modules, {"gpu_support": support}):
    replay = load("replay_validation_runner", EXECUTION / "replay.py")
with patch.dict(sys.modules, {"replay": replay}):
    summarizer = load("replay_validation_summary", EXECUTION / "summarize.py")


def selection():
    return {"cases": [dict(case_id="fixture", cap_W=165.,
        services=[dict(service_id="0", lambda_rps=2., slo_s=.5)],
        policies={"fixture": {"gpus": [dict(logical_gpu=0, frequency_MHz=900,
            replicas=[dict(service_id="0", arch="fcn_resnet50", B=1,
                           slices=1, rate_rps=2., n=1)])]}})]}


class ReplayInputTests(unittest.TestCase):
    def test_integral_numbers_keep_the_requested_configuration(self):
        document = selection()
        board = document["cases"][0]["policies"]["fixture"]["gpus"][0]
        board["replicas"][0]["B"] = 1.0
        result = replay.normalize_selection(document)[0]
        self.assertEqual(result["policies"]["fixture"]["gpus"][0]["replicas"][0]["B"], 1)

    def test_fractional_or_boolean_deployment_fields_are_rejected(self):
        for location, field in (("board", "logical_gpu"), ("board", "frequency_MHz"),
                                ("replica", "B"), ("replica", "slices"), ("replica", "n")):
            for value in (1.5, True, float("nan"), float("inf")):
                document = selection()
                board = document["cases"][0]["policies"]["fixture"]["gpus"][0]
                target = board if location == "board" else board["replicas"][0]
                target[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replay.normalize_selection(document)

    def test_invalid_layout_and_idle_mode_are_rejected(self):
        for field, value in (("layout_slices", [1.5]), ("mode", "typo"), ("mode", "idle_fixed")):
            document = selection()
            document["cases"][0]["policies"]["fixture"]["gpus"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replay.normalize_selection(document)


class ProfilingInputTests(unittest.TestCase):
    def document(self):
        document = selection()
        replica = document["cases"][0]["policies"]["fixture"]["gpus"][0]["replicas"][0]
        replica["catalog_id"] = "catfixture"
        return document

    def test_profiling_runner_rejects_fractional_counts(self):
        for field in ("B", "slices", "n"):
            document = self.document()
            document["cases"][0]["policies"]["fixture"]["gpus"][0]["replicas"][0][field] = 1.5
            with self.subTest(field=field), self.assertRaises(ValueError):
                profiling.normalize_selection(document)

    def test_case_and_policy_identifiers_cannot_escape_run_directory(self):
        for value in ("../../outside", "/absolute", "..", "C:\\outside"):
            for field in ("case", "policy"):
                document = self.document()
                case = document["cases"][0]
                if field == "case":
                    case["case_id"] = value
                else:
                    case["policies"][value] = case["policies"].pop("fixture")
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    profiling.normalize_selection(document)

    def test_retinanet_uses_detection_inputs_without_explicit_task(self):
        document = self.document()
        document["cases"][0]["policies"]["fixture"]["gpus"][0]["replicas"][0]["arch"] = "retinanet_resnet50_fpn"
        replica = profiling.normalize_selection(document)[0]["replicas"][0]
        self.assertEqual(replica["task"], "detection")
        self.assertEqual(replica["caller_shape"], [3, 640, 640])

    def test_clock_cleanup_stops_if_ownership_is_uncertain(self):
        with (patch.object(profiling, "assert_empty", side_effect=RuntimeError("GPU is not empty")),
              patch.object(profiling, "smi") as smi):
            with self.assertRaisesRegex(RuntimeError, "not empty"):
                profiling.unlock_empty_gpu(0)
            smi.assert_not_called()


class ReplayCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "build/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.jobs, _ = replay.prepare_jobs(replay.normalize_selection(selection()), 1, 0, 10)
        self.protocol = {"fixture_protocol": 1}

    def campaign(self, name, *, jobs=None, protocol=None):
        root = self.root / name
        root.mkdir()
        (root / "protocol.json").write_text(json.dumps(self.protocol if protocol is None else protocol))
        (root / "prepared_jobs.json").write_text(json.dumps(self.jobs if jobs is None else jobs))
        return root

    def completed_record(self, root, job, *, protocol=None):
        directory = root / "blocks/fixture/attempt_001"
        directory.mkdir(parents=True)
        results, summaries = [], []
        for replica, plan in zip(job["replicas"], job["plans"]):
            records = [dict(request_id=r["request_id"], measured=r["measured"],
                            finish=r["arrival_offset"] + .1, response_s=.1,
                            batch_index=i, padded_batch=False)
                       for i, r in enumerate(plan["requests"])]
            results.append(dict(records=records, batches=[]))
            summary = replay.timing_summary(records, plan["requests"], replica["slo_s"])
            summary.update(replica_id=replica["replica_id"], service_id=replica["service_id"],
                           completed_utility=summary["completed"] * replica["utility_per_request"])
            summaries.append(summary)
        record = dict(job=job, status="complete", results=results, replica_summaries=summaries,
                      window=dict(start=0., end=10.),
                      energy_window=dict(energy_J=500., mean_power_W=50., power_cap_flag_samples=0),
                      energy_through_drain=dict(energy_J=550.))
        path = directory / "replay.json"
        path.write_text(json.dumps(record))
        manifest = dict(attempt=directory.name, status="complete",
            files={"replay.json": replay.file_hash(path)},
            execution_identity=replay.execution_identity(job),
            fingerprint=replay.digest(dict(job=job, protocol=replay.digest(
                self.protocol if protocol is None else protocol))))
        (directory.parent / "COMPLETE").write_text(json.dumps(manifest))

    def test_complete_matching_campaign_is_summarized(self):
        root = self.campaign("lane0")
        self.completed_record(root, self.jobs[0])
        rows = summarizer.summarize(iter([root]))
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(rows[0]["mean_power_W"], 50.)
        self.assertTrue(rows[0]["execution_feasible"])
        self.assertGreater(rows[0]["utility_per_J"], 0.)

    def test_consistent_lanes_keep_missing_boards_visible(self):
        lanes = [self.campaign("lane0"), self.campaign("lane1")]
        rows = summarizer.summarize(lanes)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "incomplete")
        self.assertEqual(rows[0]["missing_boards"], [0])

    def test_different_protocols_cannot_be_pooled(self):
        lanes = [self.campaign("lane0"), self.campaign("lane1", protocol={"fixture_protocol": 2})]
        with self.assertRaisesRegex(ValueError, "different replay protocols"):
            summarizer.summarize(lanes)

    def test_different_prepared_jobs_cannot_be_pooled(self):
        changed = deepcopy(self.jobs)
        changed[0]["power_cap_W"] = 145.
        lanes = [self.campaign("lane0"), self.campaign("lane1", jobs=changed)]
        with self.assertRaisesRegex(ValueError, "different prepared jobs"):
            summarizer.summarize(lanes)

    def test_completed_record_must_match_prepared_job(self):
        root = self.campaign("lane0")
        changed = deepcopy(self.jobs[0])
        changed["power_cap_W"] = 145.
        self.completed_record(root, changed)
        with self.assertRaisesRegex(ValueError, "does not match its prepared job"):
            summarizer.summarize([root])

    def test_completed_record_must_match_protocol(self):
        root = self.campaign("lane0")
        self.completed_record(root, self.jobs[0], protocol={"fixture_protocol": 2})
        with self.assertRaisesRegex(ValueError, "different protocol"):
            summarizer.summarize([root])


if __name__ == "__main__":
    unittest.main()
