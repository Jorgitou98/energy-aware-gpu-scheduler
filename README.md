# Energy-Aware GPU Scheduler

This repository provides GPU scheduling code, result datasets, figure and
table builders, and tools for running new experiments. The planner
jointly selects inference variants, MIG allocations, replica placement, traffic
allocation and GPU frequency to maximize utility per joule under quality,
mean-response-time and power constraints. The included implementation targets
NVIDIA A30 GPUs; the reported comparison covers 112 seven-service workloads
and scheduler scaling across 40 workloads up to 128 services.

The default workflow rebuilds tables and figures from the distributed records
on a CPU. New planning runs and GPU acquisition use separate entry points.
The [data guide](docs/data.md) describes the records, units and aggregation
rules; [hardware](docs/hardware.md) lists the external assets required for
new physical experiments.

The [manuscript coverage table](docs/reproducibility.md#manuscript-results-and-source-coverage)
lists the source records and replay commands by figure and table. The Figure 4
entry point renders the P95 figure included in the manuscript.

## Start here

| Goal | Guide |
| --- | --- |
| Use the scheduler with your workloads | [Planning guide](docs/planning.md) |
| Check the archive and rebuild tables/figures | [Reproduction guide](docs/reproducibility.md) |
| Find a paper result and its inputs | [Figure guide](docs/figures.md), [data guide](docs/data.md) |
| Verify the physical occupancy results for Table 7 | [Acquisition archive and recovery](evidence/table7/) |
| Run a new workload through the planner | [Planning guide](docs/planning.md) |
| Collect new physical measurements | [Hardware guide](docs/hardware.md) |

## Build and test

Use Python 3.10 or later. Python 3.12 has a pinned reproduction environment.
There is no native compilation step or package installation with `pip install .`:
run the Python entry points from the repository root. The default workflow
needs neither a GPU nor Gurobi.

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: use .\.venv\Scripts\python.exe instead of python below.
python -m pip install -r requirements.txt
python reproduce.py verify
python -m unittest discover -s tests -v
python reproduce.py audit
python reproduce.py tables
python reproduce.py figures
```

For the pinned Python 3.12 environment, install
`requirements/reproduction-py312.txt` instead of `requirements.txt`. See the
[reproduction guide](docs/reproducibility.md) for complete commands on each
platform, optional PDF comparison and troubleshooting.

The outputs appear under `build/`: audit reports, `tables/table6.tex`,
`tables/table7.tex`, and `figures/figure1.pdf` through `figure8.pdf`.

For the Table 7 acquisition records, run `python scripts/restore_table7_raw.py`
to restore the original ZIP from the parts tracked in Git. The
[Table 7 guide](evidence/table7/) describes integrity checks, extraction and
analysis of the three final lanes.

Figures 1 and 3-8 are rendered from code. Figure 2 is exported from its supplied
PDF to preserve its annotations. The audit recomputes both Figure 2 intervals.
Use `python scripts/check_feasible_rate_example.py --resimulate` to rerun the
1,024 simulations per rate for panel (b).

`verify` checks file identity. The tests and `audit` check arithmetic and code
behavior, including all 969 GPU admission blocks. The separate audit
implementation recomputes statistics from the same distributed inputs.

## Workflows and requirements

| Activity | Included support | Workflow |
| --- | --- | --- |
| Saved tables and figures | Records, renderers, table builder and reference PDFs | Render the records and export the Figure 2 PDF. Check its intervals with the audit. |
| Internal consistency | File hashes, separate arithmetic audit and regression tests | Check the distributed records |
| New CPU planning runs | Catalog, explicit workloads, heuristics and plan validation | Select inputs; exact solves use Gurobi and a license |
| New workload/queue simulations | Seeded generators and mathematical self-checks | Supply a seed and explicit parameters |
| New GPU measurements | Linux acquisition, replay and aggregation tools | Prepare A30 hardware, images, checkpoints, privileges and explicit plans |

## Use the scheduler

After installing the Python requirements, generate a workload set and run a
planning method. For example, from the repository root:

```sh
python scientific/preparation/generate_workloads.py --catalog data/inputs/catalog.json --seed 2026 --output runs/example/workloads.json
python scripts/run_case.py --cases runs/example/workloads.json --case generated_028 --method greedy --output runs/example/greedy.json
```

The output records the selected method, input hashes, search status and, for a
feasible result, its deployment and objective. Each command exposes its options
with `--help`. The [planning guide](docs/planning.md) describes the available
methods, admission policies and input format.

## Reported results

These values summarize the publication records. The [data guide](docs/data.md)
describes denominators, feasible cohorts and available measurement fields.

| Result | Reported value |
| --- | --- |
| Fixed-frequency / Sequential mean predicted loss against Exact | 11.63% / 4.26% |
| Greedy+Beam mean predicted / measured loss | 0.042% / 0.023% |
| Mean-response SLO compliance, Nominal / P95 / P99 | 47 / 92 / 95 of 112 workloads |
| P99 admission | 4 SLO failures, 13 infeasible workloads |
| P95 / P99 mean objective loss on feasible cohorts | 1.38% / 3.60% |
| Offered requests within SLO, Nominal / P95 / P99 | 92.36% / 94.23% / 83.42% |
| Greedy+Beam at 128 services | 63.00 s mean, 0.01% mean loss |
| Cold / initialized Exact at 128 services | 1308.08 s / 842.78 s mean |

A mean-response SLO concerns a replica's mean within a planning interval.
Figure 7 uses one acquisition campaign with 819,508 offered requests per
policy. P99 leaves 100,921 requests unserved. The 140.4 s CPU preprocessing
cost is shared by P95 and P99. Scaling
results concern scheduler computations for logical fleets: ten distinct workloads
at each of 16, 32, 64 and 128 services, evaluated with four methods.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/efficient_mig/` | Public planning wrapper and saved-result calculations |
| `scientific/planning/` | Admission, placement, heuristic and exact formulations |
| `scientific/preparation/` | New workload and simulation generators |
| `scientific/hardware/` | Optional Linux GPU acquisition and replay |
| `data/` | Catalog, admission inputs and table records |
| `results/` | Planning, occupancy and scaling observations |
| `evidence/` | Figure inputs and processing assumptions |
| `figures/` | Nine entry points, renderers and reference PDFs |
| `scripts/` | Audits, table generation, plan execution and manifest maintenance |
| `docs/` | Reproduction, data interpretation and experiment guides |
| `tests/` | CPU consistency and regression tests |
| `provenance/` | Source descriptions and checksums |
| `build/`, `runs/` | Generated outputs and new experiments; ignored by Git |

Keep the supplied datasets and references unchanged. Write new experiments
under `runs/`, with their inputs, seeds and environment.

## Citation and license

[CITATION.cff](CITATION.cff) records the authors and accompanying paper title.
The DOI, publication status and repository URL are left unset until confirmed.
Project-owned code and documentation use the [MIT License](LICENSE).
Third-party dependencies, weights and datasets retain their own terms; see
[external components](THIRD_PARTY_NOTICES.md).
