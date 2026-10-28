# Data coverage and interpretation

The records are organized by the figures and tables they support. Plotting and
arithmetic checks use the files distributed with the repository.

| Dataset | Granularity available |
| --- | --- |
| Configuration catalog | 101 profiles labeled measured, with load-dependent values and calibration summaries |
| Feasible-rate example, Figure 2 | Nominal profile and 1,024 simulated window means and request counts at each of nine arrival rates |
| Admission grid | 1,323 configuration/rate rows; 1,354,752 simulated windows and 2,370,816 stored entries including prefixes |
| Response-time plot, Figure 4 entry point | 69 supplied configuration/rate summaries and 1,024 stored window means per pair for the repository P95 calculation |
| Solution-method Figure 5 | Per-workload objectives, certificates and 112 timing records for each of six methods |
| Occupancy Figure 6 | Per-workload observations for 97 feasible workloads per policy |
| Admission Figure 7 | One acquisition campaign with 969 logical GPU blocks and 2,571 replica windows, plus 211 objective loss comparisons, shared preprocessing time, request records and GPU telemetry |
| Scaling Figure 8 | 40 frozen workload inputs and 160 method-case observations; ten workloads per size |
| Tables 6 and 7 | Validation pairs, percentage-error records, modeled efficiencies and paired physical deployment summaries |

## Table statistics

The four power rows of Table 6 use the 166 target pairs in
`data/tables/power_holdout_original.csv`. The CSV contains
53 Paired MIG, 42 Full GPU, 69 Batched MIG and two Residency targets. The
Paired MIG power cohort excludes one throttled run. Each
row gives a target ID, a prediction and a comparison value in watts.
The builder calculates absolute percentage errors from the watt values
and checks agreement with the row errors and the reference table statistics.

The separate Residency audit uses original NVML samples and
`archive_power_targets.json` under `evidence/validation/residency/`. It
reproduces the observed means from the archived trials. Its rounded power
error statistics also agree with the CSV-based Residency row.

Table 6 response statistics use 72 Paired MIG and 42 Full GPU archived
holdout errors. Batched MIG uses the analytical predictions and the 69
observed responses from the 45 s windows in Figure 4.
`data/tables/validation_errors.json` records target IDs and source digests.

Table 7 recomputes predicted losses from `predicted_efficiency_pairs.json`
and the occupancy results. Measured losses for the five Exact-relative
methods use completed utility and energy through drain from two archived
repetitions per workload in `measured_efficiency_pairs.json`. Efficiency
is the mean of the two utility-per-joule values. Loss is
`100 * (1 - method_efficiency / reference_efficiency)`. A negative value
indicates higher measured efficiency than the reference. The paired counts
are 112 each for Fixed-frequency, Sequential and Greedy+Beam, and 97 each
for Greedy-C and Greedy. The two measured occupancy rows are manuscript
summaries only. The archived GPU runs used different deployments from
the seed-0 occupancy experiment used for Figure 6 and the predicted columns.

## Response and admission processing

Inputs under `evidence/response_time/` reproduce the manuscript's Figure 4.
The 69 targets have paired 45 s and 900 s component windows and 1,024
simulated 45 s window means each. The [figure guide](figures.md#interpretation)
describes the statistics and the checks against the supplied reference.

Admission-grid samples are in `data/inputs/admission_grid_samples.npz`, with
row identities, units and array shapes in the JSON sidecar. Each row identifies
one configuration and request rate. P95 and P99 in `admission_grid.csv` are
computed from these window-response samples using linear quantile interpolation.
The planner reads the precomputed table; the audits recompute all 7,938
percentiles from the arrays and compare them with the corresponding table entries.

Each row contains 1,024 simulated window means. The 256- and 512-window
arrays are exact prefixes of that row: there are 1,354,752 simulated windows
and 2,370,816 stored entries when the prefixes are included.
`data/inputs/admission_simulation_source.npz` retains the full means, measured
request counts, per-row seeds, configuration names, rates and service-time
parameters. Each simulation uses a 10-second warmup and a 45-second measurement
window; the sidecar records the simulation settings and source checksum.
The analytical nominal-response column and service-time moments are distinct
from the empirical mean and variance of the window-response samples.

`evidence/admission/physical_blocks_r2.json` contains the 969 logical GPU blocks
from the Figure 7 acquisition campaign. The 2,571 replica windows supply request
counts, counts within the SLO, mean response times and service SLOs. The workload panel
uses 112 attempted cases per policy. The request panel uses 819,508 offered
requests per policy, including the unserved requests in P99-infeasible
workloads. The audit checks matching trace identifiers and offered counts by
workload and service across feasible deployments. Modeled objective losses
use `objective_observations.json`; the 140.4 s shared wall-time measurement on the CPU host is in
`preprocessing.json`. The replica summaries include trace identifiers and
checksums for each block. The matching compressed archive is distributed in
`evidence/admission/raw_r2/`. It contains request arrival, start and finish
times and NVML telemetry for 797 distinct physical runs. The remaining 172
logical blocks reuse a recorded run through a completion certificate. The
100,921 unserved P99 requests contribute to the offered denominator and have
no physical timestamps. See [restoring the raw archive](reproducibility.md#figure-7-raw-acquisition-archive).

## Planning and scaling

Per-case planning records support objective and feasibility inspection.
`results/planning/timings.json` provides 112 Fixed-frequency and 112 Sequential
times. Preparation and total times are checked against the archived
`results/planning/full_timing/timing_summary.json`.

The saved `results/planning/inputs.json` contains 112 workloads, grouped into
eight strata of 14 cases by power cap, SLO regime and load. Its normalized
arrival-rate vectors are distinct across cases. The workload generator creates
new inputs by drawing fourteen independent base arrival-rate vectors within
each combination of two loads, two SLO regimes and two power caps. This yields
112 distinct vectors; supply an explicit seed for each run.

`data/inputs/scaling_cases.json` contains the 40 frozen scalability inputs:
ten nested workload sequences at 16, 32, 64 and 128 services. Each case has
`n/2` logical GPUs and a 165 W per-GPU cap. Arrival rates follow
`lambda = 1.2 * beta / reference_busy_time`, with `beta` in `[0.25, 0.55]`.
The detection and segmentation references are `fr_b1_k2_f900` and
`dl_b1_k2_f900`: batch 1, two slices and 900 MHz.

`results/scaling/observations.csv` and `summary.json` contain the same 160
publication observations for cold Exact, initialized Exact, Greedy+Beam and
Greedy. Every loss is paired with the same case's Exact ratio:
`100 * (1 - ratio / exact_ratio)`. The summary JSON and CSV report arithmetic
means, medians, extrema, sample standard deviations (`n-1`) and quartiles
(linear interpolation at `(n-1)*p`). Ten cases per size are distinct workloads,
not ten timing repetitions. The figure shows means, interquartile bands and
all individual observations.

The four [method exports](../results/scaling/source/) contain per-case timing,
objectives and selected plans. The [scaling audit](../results/scaling/#verification-and-reproduction)
checks their hashes and matches each record to the published observations.
The frozen workload inputs contain service roles, reference profiles, rates
and constraints and can be selected explicitly with the public planning CLI.

## Verification

1. **Rendering:** Figures 1 and 3-8 are generated and Figure 2 is exported from
   its publication PDF. Tables 6-7 are calculated from the records.
2. **Numerical verification:** `python reproduce.py audit` checks identities,
   both Figure 2 intervals, 2,571 replica windows and 7,938 grid quantiles.
   The independent audit also checks the replica windows and grid quantiles
   using separate arithmetic.
   Unit tests also cover request, planning, occupancy and scaling statistics.
3. **Acquisition:** planning and hardware tools can produce observations with
   explicit inputs and the required hardware or solver.

`provenance/sources.json` describes the data sources. The manifest verifies
the distributed files, including unexpected files. Run
`python reproduce.py verify`; after an intentional modification,
`python scripts/update_manifest.py` refreshes the checksums.

The configuration catalog includes calibration summaries, load-dependent
measurements and source identifiers (paths and digests). Hardware workflows collect new
request observations and GPU telemetry using the environment and external
assets described in [hardware](hardware.md).
