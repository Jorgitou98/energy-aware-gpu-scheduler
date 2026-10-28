#!/usr/bin/env python3
"""Verify or restore the compressed acquisition archive for Figure 7."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "evidence/admission/raw_r2"
ARCHIVE_NAME = "figure7_raw_completo.zip"


def read_parts(directory, output=None):
    manifest = json.loads((directory / "archive_manifest.json").read_bytes())
    archive = manifest["archive"]
    if (manifest["schema"] != "figure7-raw-archive-v1" or archive["name"] != ARCHIVE_NAME
            or not manifest["parts"]):
        raise ValueError("Invalid Figure 7 raw archive manifest")
    digest = sha256()
    total_bytes = 0
    for index, part in enumerate(manifest["parts"], start=1):
        name = f"{ARCHIVE_NAME}.part{index:02d}"
        if part["name"] != name:
            raise ValueError(f"Expected archive part {name} at position {index}")
        part_digest = sha256()
        part_bytes = 0
        with (directory / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                part_digest.update(chunk)
                digest.update(chunk)
                part_bytes += len(chunk)
                if output is not None:
                    output.write(chunk)
        if part_bytes != part["bytes"] or part_digest.hexdigest() != part["sha256"]:
            raise ValueError(f"Invalid size or SHA-256 for {name}")
        total_bytes += part_bytes
    if total_bytes != archive["bytes"] or digest.hexdigest() != archive["sha256"]:
        raise ValueError("The reassembled Figure 7 archive differs from the supplied ZIP")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true", help="Check hashes without writing the ZIP.")
    action.add_argument("--output", type=Path, help="Path for the restored ZIP (default: build/figure7/).")
    args = parser.parse_args()
    if args.verify_only:
        archive = read_parts(PARTS)
        print(f"Verified {archive['name']} ({archive['bytes']:,} bytes, SHA-256 {archive['sha256']}).")
        return

    output = (args.output or ROOT / "build/figure7" / ARCHIVE_NAME).absolute()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".figure7-", dir=output.parent) as temporary:
        staging = Path(temporary) / ARCHIVE_NAME
        with staging.open("wb") as stream:
            archive = read_parts(PARTS, stream)
        # Publish the verified file atomically without replacing an existing output.
        if os.name == "nt":
            staging.rename(output)
        else:
            os.link(staging, output)
    print(f"Restored {output} ({archive['bytes']:,} bytes, SHA-256 {archive['sha256']}).")


if __name__ == "__main__":
    main()
