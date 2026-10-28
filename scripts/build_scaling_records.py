#!/usr/bin/env python3
"""Export one compact record per workload from the scaling summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "results/scaling/summary.json"
METHODS = ("exact_cold", "exact_initialized", "greedy_beam", "greedy")
SIZES = (16, 32, 64, 128)
SEEDS = (317, 331, 347, 359, 373, 389, 401, 419, 433, 449)
FIELDS = ("time_s", "ratio", "exact_ratio", "loss_percent")
EXPECTED = {f"s{seed}_n{size}": size for seed in SEEDS for size in SIZES}


def build_records(source_bytes):
    data = json.loads(source_bytes)
    if (data.get("methods") != list(METHODS)
            or data.get("sizes") != list(SIZES)
            or data.get("retained_seeds", []) + data.get("additional_seeds", []) != list(SEEDS)
            or data.get("workloads") != len(EXPECTED)
            or data.get("observations") != len(EXPECTED) * len(METHODS)
            or data.get("cases_per_size") != len(SEEDS)):
        raise ValueError("Unexpected scaling workload or method declarations")
    series = data.get("series")
    if not isinstance(series, dict) or set(series) != set(METHODS):
        raise ValueError("Expected exactly the four scaling methods")
    source = {"path": SOURCE, "sha256": hashlib.sha256(source_bytes).hexdigest()}
    records = {
        case: {"schema": "scaling_case_summary_v1", "case": case,
               "services": services, "source": source, "methods": {}}
        for case, services in EXPECTED.items()
    }
    for method in METHODS:
        rows = series[method]
        if not isinstance(rows, list) or len(rows) != len(EXPECTED):
            raise ValueError(f"Expected 40 records for {method}")
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid record for {method}")
            case = row.get("case")
            if (not isinstance(case, str) or re.fullmatch(r"s[0-9]+_n[0-9]+", case) is None
                    or case not in EXPECTED or case in seen):
                raise ValueError(f"Invalid or duplicate workload identity for {method}: {case!r}")
            if type(row.get("services")) is not int or row["services"] != EXPECTED[case]:
                raise ValueError(f"Inconsistent service count for {method}/{case}")
            values = {field: row.get(field) for field in FIELDS}
            if not all(type(value) in (int, float) and math.isfinite(value)
                       for value in values.values()):
                raise ValueError(f"Expected finite numerical fields for {method}/{case}")
            if values["time_s"] < 0 or values["ratio"] <= 0 or values["exact_ratio"] <= 0:
                raise ValueError(f"Invalid time or objective ratio for {method}/{case}")
            records[case]["methods"][method] = values
            seen.add(case)
        if seen != set(EXPECTED):
            raise ValueError(f"Incomplete workload coverage for {method}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/scaling/cases")
    args = parser.parse_args()
    records = build_records((ROOT / SOURCE).read_bytes())
    args.outdir.mkdir(parents=True, exist_ok=True)
    for case, record in sorted(records.items()):
        (args.outdir / f"{case}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n")
    print(f"Exported {len(records)} workload records ({len(records) * len(METHODS)} method observations).")


if __name__ == "__main__":
    main()
