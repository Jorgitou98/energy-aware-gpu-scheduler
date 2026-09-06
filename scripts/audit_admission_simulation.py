#!/usr/bin/env python3
"""Replay every admission-grid simulation with its recorded parameters and seed.

The audit checks source identities and sample prefixes, runs the CPU simulator,
and compares window means and request counts. By default two workers are used
and the report is written to build/audit/admission_simulation.json.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from efficient_mig.figure_results import audit_grid_samples

ABSOLUTE_TOLERANCE_S = 1e-12


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_simulator(root):
    path = Path(root) / 'scientific/preparation/window_guard.py'
    spec = importlib.util.spec_from_file_location('admission_replay_simulator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initialize(root, source_path, source_sha256, batches, protocol):
    global _SIMULATOR, _SOURCE, _BATCHES, _PROTOCOL
    _SIMULATOR = _load_simulator(root)
    if _sha256(Path(source_path)) != source_sha256:
        raise ValueError('Admission simulation-source hash mismatch')
    with np.load(source_path, allow_pickle=False) as archive:
        _SOURCE = {key: archive[key].copy() for key in archive.files}
    _BATCHES, _PROTOCOL = batches, protocol


def _replay(index):
    means, counts = _SIMULATOR.simulate_window_means(
        float(_SOURCE['rate'][index]), float(_SOURCE['L_s'][index]),
        float(_SOURCE['C2'][index]), _BATCHES[index],
        simulations=_PROTOCOL['simulations_per_row'], seed=int(_SOURCE['seed'][index]),
        warmup_s=_PROTOCOL['warmup_s'], measured_s=_PROTOCOL['measured_s'],
        chunk_size=_PROTOCOL['chunk_size'])
    expected = _SOURCE['means'][index]
    difference = np.abs(means-expected)
    maximum = float(difference.max()) if np.isfinite(difference).all() else None
    means_ok = bool(np.allclose(means, expected, rtol=0., atol=ABSOLUTE_TOLERANCE_S))
    counts_ok = bool(np.array_equal(counts, _SOURCE['counts'][index]))
    return dict(row=index, config=str(_SOURCE['config'][index]),
                rate_requests_s=float(_SOURCE['rate'][index]), passed=means_ok and counts_ok,
                means_exact=bool(np.array_equal(means, expected)), counts_exact=counts_ok,
                means_within_tolerance=means_ok, max_mean_absolute_difference_s=maximum,
                count_mismatches=int(np.sum(counts != _SOURCE['counts'][index])))


def run_audit(root=ROOT, workers=2):
    """Return a report after validating inputs and replaying all recorded rows."""
    root = Path(root).resolve()
    if type(workers) is not int or workers < 1:
        raise ValueError('Workers must be a positive integer')
    start = time.perf_counter()
    validation = audit_grid_samples(root)
    metadata = json.loads((root/'data/inputs/admission_grid_samples.json').read_text(encoding='utf-8'))
    protocol = metadata['simulation_protocol']
    for key in ('simulations_per_row', 'chunk_size'):
        if type(protocol.get(key)) is not int or protocol[key] < 1:
            raise ValueError('Invalid simulation protocol: '+key)
    for key in ('warmup_s', 'measured_s'):
        value = protocol.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key == 'measured_s' and value == 0)):
            raise ValueError('Invalid simulation protocol: '+key)
    source_path = root / metadata['simulation_source']['path']
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    if protocol['simulations_per_row'] != source['means'].shape[1]:
        raise ValueError('Simulation protocol sample count mismatch')
    with (root/metadata['source']).open(encoding='utf-8', newline='') as stream:
        grid = list(csv.DictReader(stream))
    catalog = {p['physical_config_id']: p for p in
               json.loads((root/'data/inputs/catalog.json').read_text(encoding='utf-8'))['profiles']}
    simulator = _load_simulator(root)
    batches = []
    for index, row in enumerate(grid):
        profile = catalog[row['config']]
        batch = int(row['batch'])
        if batch < 1 or profile['B'] != batch:
            raise ValueError(f'Catalog/grid batch mismatch at row {index}')
        batches.append(batch)
        latency, c2 = simulator.interpolated_parameters(profile, float(row['rate']))
        for key, value in (('L_s', latency), ('C2', c2)):
            if (not math.isfinite(value)
                    or abs(value-float(source[key][index])) > ABSOLUTE_TOLERANCE_S):
                raise ValueError(f'Catalog/source {key} mismatch at row {index}')
        for key in ('rate', 'L_s', 'C2'):
            if abs(float(row[key])-float(source[key][index])) > ABSOLUTE_TOLERANCE_S:
                raise ValueError(f'Grid/source {key} mismatch at row {index}')
    names = ('data/inputs/admission_grid_samples.json', metadata['source'],
             metadata['simulation_source']['path'], 'data/inputs/admission_grid_samples.npz',
             'data/inputs/catalog.json', 'scientific/preparation/window_guard.py')
    identities = {name: _sha256(root/name) for name in names}
    rows = []
    initialization = (str(root), str(source_path), metadata['simulation_source']['sha256'], batches, protocol)
    workers = min(workers, len(grid))
    if workers == 1:
        _initialize(*initialization)
        rows = [_replay(index) for index in range(len(grid))]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                                 initializer=_initialize, initargs=initialization) as pool:
            futures = [pool.submit(_replay, index) for index in range(len(grid))]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 100 == 0:
                    print(f'Replayed {len(rows)}/{len(grid)} rows.', flush=True)
    for name, expected in identities.items():
        if _sha256(root/name) != expected:
            raise ValueError('Input changed during replay: '+name)
    failures = sorted((row for row in rows if not row['passed']), key=lambda row: row['row'])
    differences = [row['max_mean_absolute_difference_s'] for row in rows]
    return dict(schema='admission-simulation-audit-v1', passed=not failures,
                scope='Direct CPU simulation replay using recorded row parameters and seeds.',
                rows_checked=len(rows), windows_checked=len(rows)*source['means'].shape[1],
                means_exact_rows=sum(row['means_exact'] for row in rows),
                means_within_tolerance_rows=sum(row['means_within_tolerance'] for row in rows),
                counts_exact_rows=sum(row['counts_exact'] for row in rows),
                max_mean_absolute_difference_s=max(differences) if all(v is not None for v in differences) else None,
                total_count_mismatches=sum(row['count_mismatches'] for row in rows),
                absolute_tolerance_s=ABSOLUTE_TOLERANCE_S, relative_tolerance=0.,
                protocol=protocol, source_files=identities, input_validation=validation,
                numpy_version=np.__version__, workers=workers, failures=failures,
                elapsed_s=time.perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=min(2, os.cpu_count() or 1),
                        help='Number of CPU worker processes (default: up to 2)')
    parser.add_argument('--outdir', type=Path, default=ROOT/'build/audit',
                        help='Directory for admission_simulation.json')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        report = run_audit(workers=args.workers)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        report = dict(schema='admission-simulation-audit-v1', passed=False, error=str(error))
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir/'admission_simulation.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(output)
    if not report['passed']:
        print('Admission simulation audit failed: '+report.get('error', 'sample mismatch'), file=sys.stderr)
        return 1
    print(f"Admission simulation replay passed: {report['rows_checked']} rows; "
          f"maximum mean difference {report['max_mean_absolute_difference_s']:.3g} s.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
