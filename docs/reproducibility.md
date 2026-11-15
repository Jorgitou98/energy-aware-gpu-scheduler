# Reproduction guide

## Scope and prerequisites

The default workflow renders the supplied records and builds tables. The
[coverage below](#manuscript-results-and-source-coverage) identifies which
results can be recalculated or replayed and which are available at summary level.

- Python 3.10 or newer; a pinned environment is available for Python 3.12.
- NumPy and Matplotlib from `requirements.txt`; no CUDA, compiler, Gurobi or
  private assets for the default workflow.
- Internet access for initial dependency installation; the default reproduction
  commands themselves use local inputs.
- Keep the complete directory structure. The scripts import local modules by
  path; `pip install .` is not required or supported.

The dependency ranges in `requirements.txt` support compatible environments.
For fixed dependency versions, use Python 3.12 and
`requirements/reproduction-py312.txt`. A different operating system or font
stack may change figure layout without changing numerical values.

There are three separate workflows:

1. **Inspect and rebuild saved results:** the commands below run on a CPU and
   write tables, figures and reports to `build/`.
2. **Run new planning/simulation experiments:** follow [planning](planning.md),
   choose explicit inputs and write results to `runs/`.
3. **Acquire new GPU measurements:** follow [hardware](hardware.md) on a
   prepared Linux/A30 host. This is optional and has separate dependencies.

The default workflow runs the Python entry points directly. Keep the repository
structure intact; entry points resolve the implementation and data relative
to their own files.

## Create an isolated environment

Linux/macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/reproduction-py312.txt
.venv/bin/python reproduce.py verify
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/reproduction-py312.txt
.\.venv\Scripts\python.exe reproduce.py verify
```

Using the environment's Python directly avoids changing PowerShell execution
policy. Below, `python` means that environment's interpreter (activate it or
replace the command with its full path).

If Python 3.12 is unavailable, create the environment with a Python version
of at least 3.10 and install `requirements.txt` instead. This selects compatible
dependency versions rather than the pinned Python 3.12 environment. Install any
optional requirements using that same interpreter.

## Run the checks and build the outputs

```sh
python reproduce.py verify
python -m unittest discover -s tests -v
python reproduce.py audit
python reproduce.py tables
python reproduce.py figures
```

| Command | Output and meaning |
| --- | --- |
| `verify` | All distributed files match `provenance/manifest.json`; detects missing, altered and unexpected files |
| unit tests | CPU checks of records, mathematical constraints and error handling; no new GPU or Gurobi run |
| `audit` | `build/audit/reproducibility.json`, arithmetic checks and summary |
| `tables` | `build/tables/table6.tex`, `table7.tex`, `table_statistics.json` |
| `figures` | `build/figures/figure1.pdf` through `figure8.pdf`, descriptive aliases, PNGs for generated plots and audit sidecars |

Run each step successfully before continuing. The checksum command requires
only the Python standard library; the numerical commands need NumPy, and
plotting needs Matplotlib. The optional exact solver and hardware dependencies
are not imported to build the saved results.

The full figure build exports the Figure 2 reference PDF. The audit recomputes
both of its intervals, using the saved simulation windows for panel (b). Figure 4(a)
shows component errors for the 45 s and 900 s windows. Panel (b) compares the
analytical mean and linear P95 of each target's 1,024 simulated window means
with the supplied short-window mean response. The audit checks the arithmetic
and numerical agreement with the supplied reference.

## Manuscript results and source coverage

Run each figure with `python figures/plot_figureN.py --outdir build/figures`,
replacing `N` with its number. Use `python reproduce.py tables` to generate
the LaTeX for Tables 6 and 7. The statistics report names the record sources
and the cohorts used to compute the table entries.

| Result | Calculation and source coverage |
| --- | --- |
| Figure 1 | Recalculate the concurrent-run means from the original NVML records in `evidence/validation/residency/` and combine the supplied coefficients in `evidence/model_examples/replica_power.json`. |
| Figure 2 | `python scripts/check_feasible_rate_example.py` recalculates the nominal interval from the calibrated profile and the P95 interval from the 9,216 saved window means in `evidence/model_examples/figure2b/`. Add `--resimulate` to run all nine rates with the recorded seed and compare the window means and request counts. |
| Figure 3 | The diagram renders from code. Its objective-bound annotation applies to the optional bounded variant. The recorded performance results use the standard variant. |
| Figure 4 | Regenerate both panels from the supplied inputs and check the numerical reference. Panel (b) computes linear P95. |
| Figure 5 | Recalculate all six timings and objective-loss distributions from `results/planning/`. The compact timing archive includes the Fixed-frequency and Sequential protocol and verification command. New timings depend on the CPU and solver. |
| Figure 6 | Recompute summaries from the 112-workload seed-0 results. `python experiments/occupancy/verify_seed0.py` replays both policies using their frozen experiment ranking, grid and seed derivation. |
| Figure 7 | Rebuild both panels from the 969 logical block summaries, modeled objective pairs and shared preprocessing record. The [raw acquisition archive](#figure-7-raw-acquisition-archive) supplies the underlying per-request timestamps and GPU telemetry for the campaign. |
| Table 1 | Notation reference. |
| Table 2 | The five architecture/quality groups and 13 variants are recorded in `data/inputs/catalog.json`, which contains 101 retained configuration profiles. |
| Table 3 | The Batched MIG configuration plan and acquisition protocol are in `evidence/validation/`. The protocol records 240 planned stages with load-dependent windows. The archived response comparisons cover 69 of 72 planned held-out targets. |
| Table 4 | Inspect the 112 frozen cases in `results/planning/inputs.json`. They contain 14 distinct arrival-rate vectors in each load, SLO and power-cap condition. |
| Table 5 | The [planning guide](planning.md) maps methods to the implementation. The public planner uses the same occupancy ranking and tie-breaking as the Figure 6 replay, which supplies the original admission grid and workload seeds. |
| Table 6 | Recompute the four power rows from the 166 pairs in `data/tables/power_holdout_original.csv`. The response rows use archived Paired MIG and Full GPU errors and the 69 Figure 4 observations for Batched MIG. Run `python scripts/check_residency_power.py` to check the separate archived Residency comparison from NVML samples, calibration records and processing coefficients. |
| Table 7 | Recompute all predicted and measured rows. The first five methods use Exact as the reference. LeastAllocated and MostAllocated each use 97 complete pairs relative to the archived Greedy-C measurements. [Restore the raw ZIP from the Git parts](#table-7-raw-acquisition-archive) for analysis of the certified runs. |

## Figure 7 raw acquisition archive

The original `figure7_raw_completo.zip` is distributed in 11 parts of up to
48 MiB under `evidence/admission/raw_r2/`. The adjacent `archive_manifest.json`
gives the part order, sizes and SHA-256 digests. From the repository root, use:

```sh
python scripts/restore_figure7_raw.py --verify-only
python scripts/restore_figure7_raw.py
```

The first command verifies the parts and the reconstructed byte stream in
place. The second restores the original 524 MB ZIP to
`build/figure7/figure7_raw_completo.zip`. The ZIP contains the two compressed
acquisition lanes and `documentacion_y_hashes.zip` with inventories and
checksums. When extracting the lanes for inspection, use the same destination
directory for both so their original `results/` paths are retained. The
unpacked lanes occupy about 2.8 GB.

The archive stores 797 distinct physical runs and 172 logical aliases from
one acquisition campaign. Timestamps use the acquisition clock. Use their
differences to calculate elapsed times within a run. The default audit computes
the published results from the compact summaries and checks the input hashes.
Rerunning the GPU experiment is a separate procedure that also requires the
models, images and execution environment.

## Table 7 raw acquisition archive

The 30 parts tracked under [`evidence/table7/parts/`](../evidence/table7/parts/)
contain the original 1.49 GB ZIP. From the repository root:

```sh
python scripts/restore_table7_raw.py
python scripts/verify_table7_raw.py build/table7/table7_trazas_registros.zip --full
python reproduce.py tables
```

Use `--verify-only` to check the SHA-256 of each part and of the combined byte
stream in place. The restore command performs those checks while reconstructing
the ZIP. The full archive check verifies the ZIP members, certificates and
the repetitions used by the table builder. See the
[extraction, analysis and recovery commands](../evidence/table7/) for the raw traces.

## Additional checks

For scaling, `python reproduce.py audit` verifies the four method exports
and compares their 160 observations with the saved figure data. The
[scaling guide](../results/scaling/#verification-and-reproduction) lists the
focused unit tests and the commands to render the figure and regenerate
its compact case records.

To replay the admission-grid simulation separately and verify the archived
CPU timings:

```sh
python scripts/audit_admission_simulation.py --workers 2
python results/planning/full_timing/audit_summary.py --verify-compact results/planning/full_timing
```

Individual commands and explicit destinations:

```sh
python figures/plot_figure8.py --outdir build/scaling
python reproduce.py tables --outdir build/custom-tables
python scientific/preparation/window_guard.py --checks-only
python scientific/hardware/profiling/gpu_support.py --self-test
```

Use destinations under `build/` or `runs/`, which are excluded from checksums
and Git. Other destinations inside the repository become unexpected files for
`verify`. Batch reproduction renders into a temporary directory before copying
completed outputs. Reference PDFs and data are never replaced by these commands.

To keep a record of a local reproduction, retain the command output,
`build/audit/` and the generated tables/figures.
Record the revision and installed environment alongside the outputs, for example:

```sh
git rev-parse HEAD
python --version
python -m pip freeze
```

The Git command applies to a checkout with history; a source ZIP may not
contain that history. Numerical agreement is the comparison criterion for
saved results. New planning runtimes and PDF bytes can vary with the platform.

## Visual comparison

The optional PDF comparison requires `requirements/visual-audit.txt` (already
included in the Python 3.12 pins) and Poppler's `pdftoppm` on PATH:

```sh
python scripts/compare_figure_pdfs.py . build/validation/visual
```

Inspect the side-by-side images and `figure_comparison.json`. Font, layout and
text differences are reported separately, allowing numerical changes to be
distinguished from layout or font changes.

## Input preservation and metadata

Input identities are recorded in `provenance/input_manifest.json`. The broader
`provenance/manifest.json` includes source code and documentation too. After an
intentional code/documentation change, inspect the diff, then run:

```sh
python scripts/update_manifest.py
python reproduce.py verify
```

Do not refresh hashes merely to silence a changed-input error. New acquisitions
belong in separate outputs with their own inputs, seeds, commands and environment.
Use the input manifest to identify the exact data associated with each run.

Generated PDF plots omit automatic creation/modification dates, and the
algorithm SVG omits its build date and uses stable IDs. This avoids meaningless
timestamp-only changes between builds.
Reference PDFs have their own metadata, independent of regenerated outputs.
Git tracks content history independently of filesystem and PDF dates.

## New experiments and external inputs

[Planning](planning.md) describes solver-free heuristics, optional exact solves,
explicit workloads and output validation. [Hardware](hardware.md) documents
Linux GPU acquisition, required privileges and preparation of external assets.
Scalability includes 40 frozen workload inputs using batch-1 reference profiles
and 160 method-case records with total planning times, objective ratios and
losses. The included generators produce new cohorts from the seed supplied on
the command line; reproducing the figures uses the frozen records directly.

## Troubleshooting

- **Missing NumPy/Matplotlib:** install requirements with the same interpreter
  used to run the scripts. Do not mix a system `pip` with another Python.
- **Checksum failure:** restore the distributed inputs or inspect the intended source
  change. Line-ending conversions can alter hashes; preserve `.gitattributes`.
- **No `pdftoppm`:** numerical reproduction still works; install Poppler for
  optional visual comparison.
- **No Gurobi/license or no A30:** use the default workflow and CPU heuristics;
  the exact/hardware workflows have separate requirements.
