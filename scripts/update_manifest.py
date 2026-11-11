#!/usr/bin/env python3
"""Refresh checksums after an intentional change to the public artifact."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '__pycache__', 'build', 'runs', '.pytest_cache', '.mypy_cache'}


def main():
    files = {}
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED for part in rel.parts): continue
        if rel.as_posix() == 'provenance/manifest.json' or path.suffix == '.pyc': continue
        files[rel.as_posix()] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    out = ROOT / 'provenance/manifest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'schema': 'efficient-mig-inference-manifest-v1', 'files': files}, indent=2) + '\n', encoding='utf-8')
    print(f'{len(files)} files, {sum(x["bytes"] for x in files.values())} bytes')


if __name__ == '__main__':
    main()
