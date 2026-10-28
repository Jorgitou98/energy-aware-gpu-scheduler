# Scalability (Figure 8)

The dataset contains 40 workloads and 160 observations: ten nested workload
sequences at 16, 32, 64 and 128 services, each evaluated with cold Exact,
initialized Exact, Greedy+Beam and Greedy.

## Files

- `source/`: the four supplied method-specific JSON exports. Their SHA-256 hashes
  are recorded in `protocol.json`.
- `summary.json`: all observations and descriptive statistics.
- `observations.csv`: the same 160 method-case records in tabular form.
- `summary.csv`: statistics for each method and service count.
- `cases/`: one compact record per workload, containing the four methods and
  the path and SHA-256 of `summary.json` from which the values are copied.
- `protocol.json`: workload design, reference profiles and statistical
  conventions.
- `../../data/inputs/scaling_cases.json`: the 40 frozen workload inputs.
- `../../figures/scalability.py`: a standalone renderer with the same numerical
  observations as ordinary Python dictionaries.

## Verification and reproduction

Run these commands from the repository root:

```sh
python reproduce.py verify
python reproduce.py audit
python -m unittest discover -s tests -p test_scaling_results.py
python figures/plot_figure8.py --outdir build/scaling
python scripts/build_scaling_records.py --outdir build/scaling/cases
```

The audit checks the SHA-256 of each export against `protocol.json` and
matches all 160 source observations to `summary.json` by method and case.
It compares the service count, planning time, objective ratio, paired Exact
ratio and loss. It also recomputes the descriptive statistics and checks the
CSV files, compact records and renderer's embedded data against the summary.
The last two commands render the figure and regenerate the compact records
from the saved publication observations.

## Workload design

Each case uses `n/2` logical GPUs with a 165 W cap. Its arrival rates are
`1.2 * beta / reference_busy_time`, where `beta` lies in `[0.25, 0.55]`.
The detection and segmentation reference profiles are `fr_b1_k2_f900` and
`dl_b1_k2_f900`, both with batch 1, two slices and 900 MHz.

## Statistics and interpretation

Loss is `100 * (1 - ratio / exact_ratio)`, paired by workload. Means are
arithmetic. Sample standard deviations use denominator `n-1`; quartiles use
linear interpolation at `(n-1)*p`. The figure draws means, interquartile bands
and every observation. Ten workloads at one size are distinct inputs, not
repeated timings of the same input. The bands are not confidence intervals.

Mean Greedy+Beam times at 16, 32, 64 and 128 services are 11, 18, 31 and 63 s,
with sample standard deviations of 2.2, 3.6, 6.2 and 12.6 s. Their mean paired
losses are 0.00%, 0.01%, 0.02% and 0.01%. The corresponding mean Greedy losses
are 0.18%, 0.27%, 0.33% and 0.40%. At 128 services, the other mean times are
1308.08 s for cold Exact, 842.78 s for initialized Exact and 5.20 s for Greedy.

For each method, `time_s` comes from `total_with_preparation_seconds` in the
corresponding JSON export. The Greedy+Beam export also reports a separately
scoped `controller_wall_seconds` field for 39 workloads. The figure uses
the declared planning total across all four methods.

The publication records and frozen inputs support rendering, arithmetic and
input-design checks using total planning times, objective ratios and losses.
New planning runs use the separate interfaces in the
[planning guide](../../docs/planning.md).
