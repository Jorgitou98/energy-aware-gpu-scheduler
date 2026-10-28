# Figure 6 replay

Run `python experiments/occupancy/verify_seed0.py` from the repository root to
replay the 112 workloads for both occupancy policies. Add `--case unified_012`
to check a single workload. The script uses Python 3.10 or later and NumPy and
was verified with Python 3.12.14 and NumPy 2.5.4 on Windows. It validates every
completed plan and compares its status and modeled utility per joule with the
saved results in `results/occupancy/publication_data.json`.

The frozen code and admission grid come from the archived 2 November 2026
occupancy experiment. The repository workload file supplies the same 112 case
records. Its catalog supplies the same calibration values, with source paths
relative to the original workspace. `replay_manifest.json` records the frozen
file checksums, the repository inputs and the checksums of the original case
and catalog files. The frozen grid is the grid consumed in that experiment.

The experiment and the public planner choose uniformly from all feasible
insertion candidates with the best target-GPU priority. Both policies rank GPUs
by free slices before insertion. For each workload, the Figure 6 replay derives
the seed from the first eight bytes of the SHA-256 digest of `0|case_id`, read as
a big-endian integer. New runs through the public planner take the per-run seed
directly via `--seed`. See the [planning guide](../../docs/planning.md).
