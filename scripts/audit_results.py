#!/usr/bin/env python3
"""Check arithmetic consistency of supplied records and publication summaries.

The command validates arithmetic and input identities without running new
physical experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from efficient_mig.table_results import recompute_tables, recompute_planning_timings
from efficient_mig.figure_results import audit_figure_inputs
from audit_independent import run_audit
from check_feasible_rate_example import check_feasible_rate_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'build/audit')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Check the current input identities before performing independent calculations.
    provenance = json.loads((ROOT / 'provenance/input_manifest.json').read_text(encoding='utf-8'))
    for name, expected in provenance['files'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected['sha256']:
            raise ValueError(f'Input changed: {name}')
    tables = recompute_tables(ROOT)
    timings = recompute_planning_timings(ROOT)
    figures = audit_figure_inputs(ROOT)
    figures['feasible_rate_example'] = check_feasible_rate_example(ROOT)
    independent = run_audit(ROOT, args.outdir)
    if independent['failed']:
        raise ValueError(f"Independent audit found {independent['failed']} failed checks")
    result = {
        'schema': 'experiment-results-audit-v1',
        'passed': True,
        'consistency_passed': True,
        'scope': 'Arithmetic consistency of the distributed records.',
        'calculation_scope': {
            'figure2_statistic': 'Feasible-rate interval from the P95 of 1024 window means at each of nine rates.',
            'figure4_statistic': 'P95 across 1024 window means per configuration and load.',
            'admission_arrays': 'Window-response samples used to compute the tabulated q95/q99 values.',
            'hardware_acquisition': 'Separate workflow using scientific/hardware/.',
        },
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
        'source_files_checked': len(provenance['files']),
        'tables': tables,
        'planning_timings': timings,
        'figure_inputs': figures,
        'independent': independent,
    }
    (args.outdir / 'reproducibility.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"Arithmetic consistency passed: {independent['checks']:,} checks in a separate implementation.")


if __name__ == '__main__':
    main()
