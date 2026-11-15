#!/usr/bin/env python3
"""Verify or restore the acquisition archive for Table 7 from its Git-tracked parts."""

import argparse
import json
from pathlib import Path

from archive_parts import restore_parts, verify_parts


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/table7"
ARCHIVE_NAME = "table7_trazas_registros.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true", help="Check hashes without writing the ZIP.")
    action.add_argument("--output", type=Path, help="Path for the restored ZIP (default: build/table7/).")
    args = parser.parse_args()
    try:
        manifest = json.loads((EVIDENCE / "archive_manifest.json").read_bytes())
        if manifest["schema"] != "table7-raw-acquisition-archive-v1" or manifest["filename"] != ARCHIVE_NAME:
            raise ValueError("Invalid Table 7 raw archive manifest")
        archive = {"name": ARCHIVE_NAME, "bytes": manifest["bytes"], "sha256": manifest["sha256"]}
        details = f"{archive['bytes']:,} bytes, SHA-256 {archive['sha256']}"
        if args.verify_only:
            verify_parts(EVIDENCE / "parts", archive, manifest["parts"])
            print(f"Verified {ARCHIVE_NAME} ({details}).")
            return
        output = restore_parts(
            EVIDENCE / "parts", archive, manifest["parts"], args.output or ROOT / "build/table7" / ARCHIVE_NAME
        )
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Verification or restoration failed: {error}\n")
    print(f"Restored {output} ({details}).")


if __name__ == "__main__":
    main()
