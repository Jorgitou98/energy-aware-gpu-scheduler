#!/usr/bin/env python3
"""Verify, audit, or reproduce figures and tables from saved results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.git', '.venv', '__pycache__', 'build', 'runs', '.pytest_cache', '.mypy_cache'}


def verify():
    manifest = json.loads((ROOT / 'provenance/manifest.json').read_text(encoding='utf-8'))
    errors = []
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if '\\' in name or ':' in name or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid manifest path')
        if not path.is_file():
            errors.append(name + ': missing')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            errors.append(name + ': checksum changed')
        elif path.stat().st_size != expected['bytes']:
            errors.append(name + ': size changed')
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if (not path.is_file() or any(part in EXCLUDED for part in rel.parts)
                or rel.as_posix() == 'provenance/manifest.json' or path.suffix == '.pyc'):
            continue
        if rel.as_posix() not in manifest['files']:
            errors.append(rel.as_posix() + ': not listed in manifest')
    if errors:
        raise ValueError('\n'.join(errors))
    print(f"Verified {len(manifest['files'])} distributed files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'audit', 'figures', 'tables'))
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
        return
    destination = (args.outdir or ROOT / 'build' / args.command).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Render in an empty directory so repeated invocations replace only completed outputs.
    with tempfile.TemporaryDirectory(prefix='.render-', dir=destination) as temporary:
        work = Path(temporary)
        if args.command == 'figures':
            manifest = json.loads((ROOT / 'figures/figure_manifest.json').read_text(encoding='utf-8'))
            commands = [ROOT / f"figures/plot_figure{entry['figure']}.py" for entry in manifest['figures']]
        elif args.command == 'tables':
            commands = [ROOT / 'scripts/build_tables.py']
        else:
            commands = [ROOT / 'scripts/audit_results.py']
        for script in commands:
            subprocess.run([sys.executable, str(script), '--outdir', str(work)], check=True, cwd=ROOT)
        for path in work.rglob('*'):
            if path.is_file():
                target = destination / path.relative_to(work)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    print(f'Outputs: {destination}')


if __name__ == '__main__':
    main()
