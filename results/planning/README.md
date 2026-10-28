# Main 112-workload comparison: Figure 5

`deployment_data.json` contains the per-workload objectives. Loss statistics
are paired with Exact by workload. Greedy-C and Greedy complete 97 of 112
workloads; Greedy+Beam and the exact baselines complete all 112. Failed
constructions remain in the attempted cohort.

`timings.json` contains 112 Fixed-frequency and 112 Sequential observations
from the archived Montblanc run in `full_timing/timing_summary.json`.
The checks compare the preparation and total time for each case against that
source. The search time is the total minus preparation. Sequential includes
the Fixed-frequency phase and the subsequent refinements. Verify the archived
112-case acquisition summary and protocol with:

```sh
python results/planning/full_timing/audit_summary.py --verify-compact results/planning/full_timing
```

The frozen protocol uses the figure number from the acquisition campaign.
The planning comparison is Figure 5 in the current numbering.

`published_timings.json` and `planning_statistics.json` contain the reference
mean times: 3.89 s Fixed-frequency, 3.93 s Sequential, 1.98 s Greedy-C,
1.98 s Greedy, 3.97 s Greedy+Beam, and 4.72 s Exact.

Plan records support objective and feasibility inspection against the explicit
workload inputs. Their objectives match the corresponding figure values.
`inputs.json` contains 112 seven-service cases in eight strata of 14 cases.
Each case provides the service rates, SLOs, quality thresholds, GPU count and
power cap required by the planner.

Fixed-frequency and Sequential were remeasured together on Montblanc. The
other Figure 5 times come from an earlier acquisition on the same host.
The renderer uses `stage_timing_audit.json` for Greedy-C, Greedy and Greedy+Beam,
and `exact_timing_observations.json` for Exact. The Exact measurement in those
records includes the initializer and exact solve. New timings depend on the
CPU and solver configuration.
