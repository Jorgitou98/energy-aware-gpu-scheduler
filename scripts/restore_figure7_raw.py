#!/usr/bin/env python3
"""Verify or restore the compressed acquisition archive for Figure 7."""

import argparse
import json
from pathlib import Path

from archive_parts import restore_parts, verify_parts


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "evidence/admission/raw_r2"
ARCHIVE_NAME = "figure7_raw_completo.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true", help="Check hashes without writing the ZIP.")
    action.add_argument("--output", type=Path, help="Path for the restored ZIP (default: build/figure7/).")
    args = parser.parse_args()
    manifest = json.loads((PARTS / "archive_manifest.json").read_bytes())
    archive = manifest["archive"]
    if manifest["schema"] != "figure7-raw-archive-v1" or archive["name"] != ARCHIVE_NAME:
        raise ValueError("Invalid Figure 7 raw archive manifest")
    if args.verify_only:
        verify_parts(PARTS, archive, manifest["parts"])
        print(f"Verified {archive['name']} ({archive['bytes']:,} bytes, SHA-256 {archive['sha256']}).")
        return

    try:
        output = restore_parts(PARTS, archive, manifest["parts"], args.output or ROOT / "build/figure7" / ARCHIVE_NAME)
    except FileExistsError as error:
        parser.error(str(error))
    print(f"Restored {output} ({archive['bytes']:,} bytes, SHA-256 {archive['sha256']}).")


if __name__ == "__main__":
    main()
