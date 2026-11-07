"""Regression checks for the independent audit's failure behavior."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        'audit_independent', ROOT / 'scripts/audit_independent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentAuditTests(unittest.TestCase):
    def test_import_does_not_run_audit(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module = load_auditor()
        self.assertEqual(output.getvalue(), '')
        self.assertTrue(callable(module.run_audit))

    def test_changed_record_fails_against_publication_statistics(self):
        module = load_auditor()
        temp_root = ROOT / 'build/test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='independent_audit_', dir=temp_root) as tmp:
            temp = Path(tmp)
            self.assertTrue(temp.resolve().is_relative_to(temp_root.resolve()))
            inputs = (
                'results/scaling/summary.json',
                'results/scaling/observations.csv',
                'results/scaling/summary.csv',
                'results/scaling/protocol.json',
                'data/inputs/scaling_cases.json',
                'data/inputs/catalog.json',
                'figures/scalability.py',
                'data/tables/published_tables.json',
                'data/tables/response_group_statistics.json',
                'data/tables/validation_errors.json',
                'data/tables/power_holdout_original.csv',
                'data/tables/measured_efficiency_pairs.json',
                'data/tables/predicted_efficiency_pairs.json',
                'results/occupancy/publication_data.json',
                'results/planning/timings.json',
                'results/planning/published_timings.json',
                'results/planning/full_timing/timing_summary.json',
                'evidence/admission/observations.json',
                'evidence/admission/physical_blocks_r2.json',
                'evidence/admission/objective_observations.json',
                'evidence/admission/preprocessing.json',
                'evidence/admission/publication_data.json',
                'evidence/response_time/long_components.json',
                'evidence/response_time/component_observations.json',
                'evidence/response_time/publication_data.json',
                'evidence/response_time/simulated_window_means.npz',
                'data/inputs/admission_grid.csv',
                'data/inputs/admission_grid_samples.json',
                'data/inputs/admission_grid_samples.npz',
                'data/inputs/admission_simulation_source.npz',
            )
            for name in inputs:
                destination = temp / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            shutil.copytree(ROOT / 'results/scaling/cases', temp / 'results/scaling/cases')
            shutil.copytree(ROOT / 'results/scaling/source', temp / 'results/scaling/source')
            table = temp / 'data/tables/validation_errors.json'
            data = json.loads(table.read_text(encoding='utf-8'))
            data['groups'][0]['records'][0]['absolute_error_percent'] += 0.5
            table.write_text(json.dumps(data), encoding='utf-8')
            # Publication summaries stay unchanged: the audit must detect that
            # an individual observation no longer reproduces those summaries.
            outdir = temp / 'audit'
            with contextlib.redirect_stdout(io.StringIO()):
                status = module.main([
                    '--root', str(temp),
                    '--outdir', str(outdir)])
            result = json.loads((outdir / 'independent_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(status, 1)
            self.assertFalse(result['passed'])
            self.assertGreater(result['failed'], 0)
            self.assertTrue(any(
                failure['name'] == 't6/Paired MIG/response/mean/source'
                for failure in result['failures']))
            checks = json.loads((outdir / 'independent_checks.json').read_text(encoding='utf-8'))
            self.assertEqual(len(checks), result['checks'])

    def test_independent_audit_rejects_mean_substitution_for_p95(self):
        module = load_auditor()
        original = module.recompute_response_data

        def use_mean(root):
            result = original(root)
            reference = json.loads((root / 'evidence/response_time/publication_data.json').read_text(encoding='utf-8'))
            result['data']['rows'] = reference['rows']
            return result

        temp_root = ROOT / 'build/test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='p95_audit_', dir=temp_root) as tmp:
            with patch.object(module, 'recompute_response_data', side_effect=use_mean):
                result = module.run_audit(ROOT, tmp)
        self.assertFalse(result['passed'])
        failures = [check for check in result['failures']
                    if check['name'].endswith('/current-window-mean-p95')]
        self.assertEqual(len(failures), 69)

    def test_independent_audit_rejects_reordered_prefix_with_same_quantiles(self):
        module = load_auditor()
        sample_file = ROOT / 'data/inputs/admission_grid_samples.npz'
        with np.load(sample_file, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        row = arrays['window_means_s_n256'][0]
        quantiles = np.quantile(row, [.95, .99], method='linear')
        self.assertNotEqual(row[0], row[1])
        row[[0, 1]] = row[[1, 0]]
        np.testing.assert_array_equal(np.quantile(row, [.95, .99], method='linear'), quantiles)
        temp_root = ROOT / 'build/test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='prefix_audit_', dir=temp_root) as tmp:
            altered = Path(tmp) / 'altered.npz'
            np.savez_compressed(altered, **arrays)
            original_load = np.load

            def load(path, *args, **kwargs):
                if Path(path).resolve() == sample_file.resolve():
                    path = altered
                return original_load(path, *args, **kwargs)

            with patch.object(module.np, 'load', side_effect=load):
                result = module.run_audit(ROOT, Path(tmp) / 'audit')
        self.assertFalse(result['passed'])
        self.assertEqual([check['name'] for check in result['failures']],
                         ['grid/256/source-prefix'])
        self.assertEqual(result['summaries']['grid']['window_means'], 2370816)
        self.assertEqual(result['summaries']['grid']['unique_simulated_windows'], 1354752)


if __name__ == '__main__':
    unittest.main()
