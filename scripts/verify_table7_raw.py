#!/usr/bin/env python3
"""Verify the Table 7 acquisition ZIP and its final analysis cohort."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence/table7/archive_manifest.json"


def digest(stream):
    checksum = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_certificates(archive, prefix, index, lane, expected_count):
    directory = lane + "/blocks/"
    markers = {path: json.loads(archive.read(prefix + path)) for path in index
               if path.startswith(directory) and path.endswith("/COMPLETE")}
    require(len(markers) == expected_count, f"Unexpected certificate count in {lane}")
    aliases = 0
    for path, marker in markers.items():
        require(marker["status"] == "complete", f"Invalid certificate status: {path}")
        if "equivalent_to" in marker:
            source = f'{lane}/{marker["equivalent_to"]}/COMPLETE'
            require(source in markers and "equivalent_to" not in markers[source],
                    f"Expected a physical source in the same lane: {path}")
            require(index[source]["sha256"] == marker["source_manifest_sha256"],
                    f"Alias source hash differs: {path}")
            aliases += 1
            continue
        attempt = PurePosixPath(path).parent / marker["attempt"]
        require("replay.json" in marker["files"], f"Missing request and telemetry record: {path}")
        for name, expected in marker["files"].items():
            member = (attempt / name).as_posix()
            require(index.get(member, {}).get("sha256") == expected,
                    f"Certified file hash differs: {member}")
    return set(markers), aliases


def verify_repetitions(archive, prefix, index, archive_name):
    observations = json.loads((ROOT / "data/tables/measured_efficiency_pairs.json").read_bytes())
    policies = {"LeastAllocated", "MostAllocated"}
    compact = {}
    for method in observations["methods"]:
        policy = method["method"]
        if policy not in policies:
            continue
        require(method["source"]["archive"] == archive_name, f"Archive name differs: {policy}")
        for member, expected in method["source"]["members"].items():
            require(member.startswith(prefix) and index.get(member.removeprefix(prefix), {}).get("sha256") == expected,
                    f"Archived analysis member hash differs: {member}")
        for row in method["records"]:
            for run in row["repetitions"]:
                key = (row["case"], policy, run["repetition"])
                require(key not in compact, f"Duplicate repository repetition: {key}")
                compact[key] = run["method"]

    summary_name = prefix + "analysis/inputs/measured_summary.json"
    summary = json.loads(archive.read(summary_name))
    measured = {}
    for deployment in summary["deployments"]:
        key = (deployment["case_id"], deployment["policy"], deployment["repetition"])
        require(deployment["status"] == "complete" and key not in measured,
                f"Unexpected archived repetition: {key}")
        measured[key] = {field: deployment[field] for field in ("completed_utility", "energy_through_drain_J")}
    require(compact == measured, "Repository repetitions differ from the archived measured summary")
    return len(compact)


def verify(path, *, full=False):
    manifest = json.loads(MANIFEST.read_bytes())
    require(path.stat().st_size == manifest["bytes"], "ZIP size differs from manifest")
    with path.open("rb") as stream:
        require(digest(stream) == manifest["sha256"], "ZIP SHA-256 differs from manifest")
    prefix = manifest["root"] + "/"
    with ZipFile(path) as archive:
        members = archive.infolist()
        names = {info.filename for info in members}
        require(len(members) == len(names) == manifest["members"], "Unexpected ZIP member inventory")
        index = json.loads(archive.read(prefix + "SHA256.json"))
        require(names == {prefix + name for name in index} | {prefix + "SHA256.json"},
                "ZIP members differ from SHA256.json")
        for name, record in index.items():
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name
                    and ":" not in name, f"Unsafe member name: {name}")
            require(archive.getinfo(prefix + name).file_size == record["bytes"],
                    f"ZIP member size differs: {name}")
        origin = json.loads(archive.read(prefix + "REMOTE_SHA256.json"))
        require(len(origin) == manifest["origin_index_members"], "Unexpected origin index size")
        require(all(index.get(name) == record for name, record in origin.items()),
                "The origin SHA-256 inventory differs from the local inventory")
        repetitions = verify_repetitions(archive, prefix, index, manifest["filename"])
        final = {}
        alias_count = 0
        for lane, count in manifest["final_lanes"].items():
            final[lane], aliases = read_certificates(archive, prefix, index, lane, count)
            alias_count += aliases
        failed = manifest["failed_lane"]
        retained = manifest["retained_failed_lane_certificates"]
        original_markers, _ = read_certificates(archive, prefix, index, failed, retained)
        recovery = "results/recovery_01/lane1"
        recovered_markers = {recovery + name.removeprefix(failed) for name in original_markers}
        require(recovered_markers <= final[recovery], "A retained certificate is missing from recovery")
        for name in original_markers:
            recovered = recovery + name.removeprefix(failed)
            require(index[name] == index[recovered], f"Retained certificate differs: {name}")
        require(len(final[recovery] - recovered_markers) == manifest["new_recovery_certificates"],
                "Unexpected count of certificates added in recovery")
        total = sum(map(len, final.values()))
        require(total == manifest["logical_blocks"] and alias_count == manifest["aliases"]
                and total - alias_count == manifest["physical_blocks"], "Cohort totals differ")
        if full:
            for position, (name, record) in enumerate(index.items(), 1):
                with archive.open(prefix + name) as member:
                    require(digest(member) == record["sha256"], f"ZIP member SHA-256 differs: {name}")
                if position % 2000 == 0:
                    print(f"Verified {position}/{len(index)} members", flush=True)
    return {"sha256": manifest["sha256"], "members": len(members), "origin_members": len(origin),
            "verified_member_contents": full, "final_certificates": {lane: len(paths) for lane, paths in final.items()},
            "repository_repetitions": repetitions, "retained_certificates": retained, "logical_blocks": total,
            "physical_blocks": total - alias_count, "aliases": alias_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--full", action="store_true", help="Also stream and hash every indexed member")
    args = parser.parse_args()
    try:
        result = verify(args.archive, full=args.full)
    except (BadZipFile, KeyError, ValueError, OSError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
