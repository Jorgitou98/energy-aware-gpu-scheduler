# Table 7 GPU acquisition records

The repository stores `table7_trazas_registros.zip` in 30 binary parts of up
to 48 MiB under [`parts/`](parts/). Together they occupy 1.49 GB. The
[archive manifest](archive_manifest.json) records their order, sizes and
SHA-256 digests, the digest of the original ZIP and the final certificate
counts. The ZIP expands to approximately 8.6 GB.

The ZIP preserves the original inputs, per-request arrival, start, completion
and response times, NVML telemetry, runner and restoration logs, `COMPLETE`
certificates, acquisition and analysis code, and the local and source integrity
inventories. Its `analysis/` directory includes the compact inputs and output,
the analysis README and the lane audits. Its Greedy-C reference is the
certified October 2026 summary. The occupancy measurements were acquired on
15 November 2026 in a separate campaign.

## Verify and reproduce

From the repository root, reconstruct the archive and regenerate the table
from the repository's compact inputs.

```sh
python scripts/restore_table7_raw.py
python scripts/verify_table7_raw.py build/table7/table7_trazas_registros.zip --full
python reproduce.py tables
```

The restore command verifies each part and the complete ZIP before publishing
`build/table7/table7_trazas_registros.zip`. Use `--verify-only` to check the
parts without writing the ZIP or `--output PATH` to choose its destination.

The verifier checks the ZIP digest, both hash inventories, the final certificate
counts, the certificate hashes carried into the recovery, and the 388
repetitions used in the repository's compact inputs. `--full` streams
each member through the ZIP CRC and SHA-256 checks. The table builder writes
`build/tables/table7.tex` and its numerical audit. It uses completed utility
divided by energy through drain for each repetition, averages the two
efficiencies per workload, and calculates `100 * (1 - method / reference)`.
Each occupancy policy has 97 measured pairs among 112 attempted workloads.

To reproduce the archived analysis, extract the verified ZIP and run the
analysis script from the extracted `table7_raw` directory.

```sh
python -m zipfile -e build/table7/table7_trazas_registros.zip build/table7
cd build/table7/table7_raw
python -X utf8 analysis/reproduce.py
```

This command verifies the compact input hashes and regenerates the archived
`analysis/table7.tex` and `analysis/results.json`. To build the deployment
summary directly from the certified records, run the archived summarizer on a
Linux CPU host or WSL, from the same extracted directory. It uses the POSIX
`fcntl` module through the archived acquisition library.

```sh
python3 package/summarize_112.py results/lane0 results/recovery_01/lane1 results/lane2 --output summary_recomputed.json
```

`summary_recomputed.json` can be compared with
`analysis/inputs/measured_summary.json` by case, policy and repetition. Use
the verification, analysis and summarization commands above for offline
reproduction. The archive also preserves the original GPU acquisition tools.

## Recovery and counting

The original `results/lane1` records the failed attempt (`NVML power unavailable`).
The recovery at `results/recovery_01/lane1` reuses 202 certified blocks from
that attempt and adds 186 certified blocks. Use the three final roots in the
command above exactly once each. The original lane is historical evidence and
contains the 202 certificates already present in recovery.

The three final lanes each have 388 certificates. Their combined 1,164 logical
blocks represent 1,134 physical acquisitions and 30 aliases. An alias shares
the physical observation identified in its certificate within the same lane.
The lane 0 and lane 2 audit logs also preserve the original checker outcome
and the correction to the expected integer-versus-float window metadata.
