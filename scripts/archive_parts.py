"""Verify and restore ZIP archives distributed as ordered byte fragments."""

from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def verify_parts(directory, archive, parts, output=None):
    """Check each part and the combined ZIP, optionally writing the bytes."""
    if not parts:
        raise ValueError("The archive manifest contains no parts")
    archive_name = archive["name"]
    combined_digest = sha256()
    total_bytes = 0
    for index, part in enumerate(parts, start=1):
        name = f"{archive_name}.part{index:02d}"
        if part["name"] != name:
            raise ValueError(f"Expected archive part {name} at position {index}")
        part_digest = sha256()
        part_bytes = 0
        with (directory / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                part_digest.update(chunk)
                combined_digest.update(chunk)
                part_bytes += len(chunk)
                if output is not None:
                    output.write(chunk)
        if part_bytes != part["bytes"] or part_digest.hexdigest() != part["sha256"]:
            raise ValueError(f"Invalid size or SHA-256 for {name}")
        total_bytes += part_bytes
    if total_bytes != archive["bytes"] or combined_digest.hexdigest() != archive["sha256"]:
        raise ValueError(f"The reassembled {archive_name} differs from the supplied ZIP")


def restore_parts(directory, archive, parts, output):
    """Publish a verified ZIP atomically without replacing an existing file."""
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".archive-", dir=output.parent) as temporary:
        staging = Path(temporary) / archive["name"]
        with staging.open("wb") as stream:
            verify_parts(directory, archive, parts, stream)
        if os.name == "nt":
            staging.rename(output)
        else:
            os.link(staging, output)
    return output
