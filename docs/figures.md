# Publication figures

Run `python reproduce.py figures` to produce all eight figures, or
`python figures/plot_figureN.py --outdir build/custom` for one figure. Paths
are resolved independently of the current working directory. The batch renders
in a temporary directory before replacing completed outputs.

These commands render the supplied inputs; they do not repeat the physical
experiments. Figure 4 calculates the P95 of 1,024 window means for each
configuration/rate pair. The interpretation of each result is described below.

| Figure | Content | Inputs |
| --- | --- | --- |
| 1 | Replica power aggregation | Calibration coefficients and two concurrent-run means in `evidence/model_examples/replica_power.json` |
| 2 | Feasible traffic intervals and admission | `figures/reference/feasible_rates_admission.pdf`, `evidence/model_examples/feasible_rate_nominal.json`, and the data in `evidence/model_examples/figure2b/` |
| 3 | Greedy and beam-search diagram | `create_heuristic_diagram.py` |
| 4 | Repository P95 response-time calculation | 69 reference JSON targets, component values and window-mean arrays under `evidence/response_time/` |
| 5 | Solution methods | Per-workload objectives and `results/planning/timings.json` |
| 6 | Kubernetes occupancy policies | Seed-0 outcomes in `results/occupancy/` and frozen replay in `experiments/occupancy/` |
| 7 | Workload and request SLO admission | `evidence/admission/observations.json` and its three source records |
| 8 | Scalability | 160 observations, four methods and 40 workloads |

Each figure has `figureN.pdf` and a descriptive PDF name. Generated figures
also have PNGs. Numerical summaries and processing metadata accompany the
outputs. Fonts and Matplotlib versions can affect rendering without changing
the numerical values.

## Interpretation

Figure 1 sums the device baseline once and adds both replicas' residency
and rate-dependent power. The observed bar is the mean of two concurrent runs.
The two concurrent-run means are recalculated from the original NVML
samples under `evidence/validation/residency/`. Figure 1 uses its supplied
calibration coefficients. A separate archived Residency analysis uses
these run means with a slightly different prediction and agrees with the
rounded Residency statistics computed from the Table 6 power CSV.

Figure 2 exports the supplied PDF. The DeepLabV3 calibration gives the
nominal interval [18.23, 47.05] requests/s at a 160 ms deadline. For panel (b),
the calculation uses the P95 of 1,024 window means at each of nine rates.
It interpolates the correction between these rates and finds the crossings
at [18.76, 45.62] requests/s. Each simulation uses 10 s of warm-up followed
by a 45 s observation window. Run `python scripts/check_feasible_rate_example.py`
to verify both intervals from the repository inputs. Add `--resimulate` to
regenerate the windows on a CPU from the recorded seed and compare them with
the saved window means and request counts.

Figure 3 renders the diagram in the manuscript. Its objective upper-bound
pruning label describes the optional `greedy-beam-bounded` method. The
archived results use `greedy-beam`, which rejects branches based on remaining
slice capacity and applies the beam width and signature limits. An upper-bound
pruning claim for the evaluated method would require a separate evaluation.

Figure 4(a) shows component errors for the 45 s and 900 s windows. Panel (b)
calculates linear P95 from 1,024 supplied window means for each of 69 targets
and uses signed error `100 * (observed - predicted) / observed`. Negative values
indicate overestimation. The audit checks window metadata and numerical agreement
with the supplied reference.

The admission-grid arrays provide the window-response samples used to calculate
P95/P99 for all 1,323 rows and the three sample sizes. Their metadata records
row identities, units, array shapes, simulation parameters, per-row seeds and
the linear quantile method. The 256- and 512-window arrays are exact prefixes
of the corresponding 1,024-window simulation output.

Figure 5 pairs efficiency with Exact by workload. Fixed-frequency and
Sequential means use their 112 end-to-end timing observations per method,
including preprocessing and search. These two methods were remeasured on
Montblanc in a separate acquisition from the other Figure 5 methods. The
renderer recomputes the other four time labels from their archived per-case
observations. The displayed values match the manuscript.
The saved workload inputs contain 112 cases in eight strata, with 14 cases
per stratum. The group-level statistics use the feasible cases in each stratum.

Figure 7 combines workload and request compliance from one acquisition campaign
with shared arrival traces across the admission policies.
In panel (a), Nominal, P95 and P99 meet every replica's mean-response SLO in
47, 92 and 95 of 112 workloads. P99 leaves 13 workloads infeasible and
misses the SLO in 4 feasible workloads. Its 3.60% mean modeled objective loss
is over 99 feasible workloads, whereas P95's 1.38% is over all 112. P95 and P99
share one 140.4 s wall-time measurement on the CPU host.

Panel (b) counts all 819,508 offered requests per policy, including 100,921
unserved requests from the P99 infeasible workloads. Nominal, P95 and P99
meet the per-request SLO for 92.36%, 94.23% and 83.42% of offered requests.
Counts and mean-response flags are aggregated from 2,571 replica windows
across 969 logical GPU blocks. Of these, 797 are physical runs and 172 reuse
recordings of equivalent executions. The repository also distributes
the matching per-request records, GPU telemetry and completion certificates
in the [raw acquisition archive](reproducibility.md#figure-7-raw-acquisition-archive).

Figure 8 uses ten distinct workloads at each of 16, 32, 64 and 128 services,
with the same 40 cases evaluated by all four methods. Lines show arithmetic
means, bands show interquartile ranges, and points show individual workloads.
The bands describe workload variation, not confidence intervals. Loss is
`100 * (1 - ratio / exact_ratio)` for each matched case; annotations show
mean paired losses. Quartiles use linear interpolation at `(n-1)*p`.

The figure uses `total_with_preparation_seconds` from the four JSON exports
in `results/scaling/source/`. The [scaling protocol](../results/scaling/protocol.json)
describes the timing fields, workload design and statistical conventions.

The 40 frozen inputs use batch-1 references and are distributed in
`data/inputs/scaling_cases.json`. The figure renders publication records from
`results/scaling/`, also embedded as ordinary dictionaries in `scalability.py`.
The compact case records contain total planning times, objective ratios and
losses. These results describe scheduler computation for logical fleets.
The rendering command builds the figure from the recorded observations.

`figures/figure_manifest.json` identifies the reference PDFs and plotting
programs. The reference inputs provide comparison targets for the numerical
verification.

## Optional visual comparison

Install `requirements/visual-audit.txt` and make Poppler's `pdftoppm`
available on PATH, then run:

```sh
python scripts/compare_figure_pdfs.py . build/validation/visual
```

This rasterizes each figure and its reference at 120 dpi and writes
comparisons plus `figure_comparison.json`. Pixel equality, PDF text and page
dimensions are reported separately to distinguish data and layout differences.
