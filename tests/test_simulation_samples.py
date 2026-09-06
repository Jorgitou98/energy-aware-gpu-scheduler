"""Check direct sample export from new CPU window-response simulations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scientific/preparation/window_guard.py'
spec = importlib.util.spec_from_file_location('sample_export_window_guard', SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class SimulationSamplesTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'build/test_tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.profile = dict(physical_config_id='sample_export_fixture', B=2,
                            points=[dict(rate_requests_s=.01, L_s=.02, cs2=.2),
                                    dict(rate_requests_s=10., L_s=.03, cs2=.3)])
        self.catalog = self.directory / 'catalog.json'
        self.catalog.write_text(json.dumps({'profiles':[self.profile]}), encoding='utf-8')

    def run_cli(self, output, samples=None, *, seed=12345, rate=2.):
        arguments = [sys.executable, str(SCRIPT), '--catalog', str(self.catalog),
                     '--config', self.profile['physical_config_id'], '--rate', str(rate),
                     '--simulations', '64', '--seed', str(seed), '--output', str(output)]
        if samples is not None:
            arguments.extend(['--samples-output', str(samples)])
        return subprocess.run(arguments, capture_output=True, text=True, timeout=20)

    def load_samples(self, path):
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), {'window_means_s', 'measured_requests', 'metadata_json'})
            return (saved['window_means_s'].copy(), saved['measured_requests'].copy(),
                    json.loads(saved['metadata_json'].item()))

    def test_summary_quantiles_use_exported_samples_and_preserve_empty_windows(self):
        output, samples = self.directory/'summary.json', self.directory/'samples.npz'
        completed = self.run_cli(output, samples, rate=.01)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        record = json.loads(output.read_text())
        means, counts, metadata = self.load_samples(samples)
        self.assertEqual(means.shape, (64,))
        self.assertEqual(counts.shape, (64,))
        np.testing.assert_array_equal(np.isnan(means), counts == 0)
        self.assertGreater(record['empty_measured_windows'], 0)
        self.assertEqual(record['empty_measured_windows'], int((counts == 0).sum()))
        valid = means[np.isfinite(means)]
        for quantile, field in ((.025, 'q025'), (.5, 'q50'), (.9, 'q90'),
                                (.95, 'q95'), (.975, 'q975'), (.99, 'q99')):
            self.assertEqual(float(np.quantile(valid, quantile, method='linear')),
                             record[field+'_window_mean_s'])
        self.assertEqual(float(valid.mean()), record['mean_window_R_s'])
        self.assertEqual(metadata['valid_windows'], len(valid))
        self.assertEqual(metadata['quantile_method'], 'linear')
        self.assertEqual(metadata['master_seed'], 12345)
        self.assertEqual(metadata['seed'], record['seed'])
        for field in ('config', 'rate_requests_s', 'batch', 'L_s', 'C2', 'simulations',
                      'warmup_s', 'measured_s', 'catalog_sha256'):
            self.assertEqual(metadata[field], record[field])
        self.assertEqual(record['samples']['sha256'], hashlib.sha256(samples.read_bytes()).hexdigest())
        self.assertEqual(metadata['simulator_sha256'], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())

    def test_seed_repeats_actual_arrays_and_export_keeps_default_summary(self):
        runs = []
        for name, seed in (('first', 12345), ('same', 12345), ('other', 12346)):
            output, samples = self.directory/(name+'.json'), self.directory/(name+'.npz')
            completed = self.run_cli(output, samples, seed=seed)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            runs.append((json.loads(output.read_text()), *self.load_samples(samples)))
        np.testing.assert_array_equal(runs[0][1], runs[1][1])
        np.testing.assert_array_equal(runs[0][2], runs[1][2])
        self.assertEqual(runs[0][3], runs[1][3])
        self.assertFalse(np.array_equal(runs[0][1], runs[2][1]))
        record, actual_means, actual_counts, _ = runs[0]
        latency, c2 = guard.interpolated_parameters(self.profile, 2.)
        direct_means, direct_counts = guard.simulate_window_means(
            2., latency, c2, 2, simulations=64, seed=record['seed'])
        np.testing.assert_array_equal(actual_means, direct_means)
        np.testing.assert_array_equal(actual_counts, direct_counts)
        default = self.directory/'default.json'
        completed = self.run_cli(default)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn('samples', json.loads(default.read_text()))
        record.pop('samples')
        self.assertEqual(record, json.loads(default.read_text()))
        self.assertEqual(completed.stdout.strip(), str(default))

    def test_existing_outputs_and_identical_paths_are_refused(self):
        for existing in ('summary', 'samples'):
            with self.subTest(existing=existing):
                output = self.directory/(existing+'-summary.json')
                samples = self.directory/(existing+'-samples.npz')
                occupied = output if existing == 'summary' else samples
                occupied.write_bytes(b'keep existing contents')
                completed = self.run_cli(output, samples)
                self.assertEqual(completed.returncode, 2)
                self.assertIn('already exists', completed.stderr)
                self.assertEqual(occupied.read_bytes(), b'keep existing contents')
                self.assertFalse((samples if existing == 'summary' else output).exists())
        path = self.directory/'same-output'
        completed = self.run_cli(path, path)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('must name different files', completed.stderr)
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
