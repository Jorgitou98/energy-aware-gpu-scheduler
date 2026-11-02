# Planning with explicit workload inputs

The model specializes the paper's formulation to identical four-slice NVIDIA
A30 GPUs. The included catalog contains 101 configurations labeled measured at 900,
1200 and 1440 MHz. Sizes 1, 2 and 4 give the legal A30 allocations. GPU
frequency is shared by all replicas on that GPU. The objective uses expected
traffic-dependent power; the power constraint reserves busy power.

## Prepare or validate supplied inputs

`scripts/run_case.py` compiles alternatives, runs a selected planning method,
or validates a saved deployment for an explicit workload and case. Its
`--help` output describes the available modes and arguments. New planning
outputs belong under `runs/`; existing output files are not overwritten.
Saved input records support the
supplied objective records described in `results/planning/README.md`. They
contain 112 cases with explicit service rates, quality thresholds, SLOs,
GPU counts and power caps. Each saved case is selected by its identifier.

Validation uses the workload, case and admission mode associated with the
selected result. Saved method results and standalone deployment JSON objects
are supported. Validation checks input bindings
when available, traffic, quality, replica counts, frequency, slices, power,
latency and stored metrics without solving. A failed search or a preparation
record has no deployment to validate.

## Generate new workload inputs

`scientific/preparation/generate_workloads.py` creates seeded comparison or
scaling workloads for new experiments. The comparison design uses fourteen independently drawn
base-rate vectors within each combination of two load multipliers, two SLO
regimes and two power caps, for 112 distinct workloads. Generated
inputs are explicitly labeled as new; no published measurements are replaced.
Some generated workloads return
`construction_failed` under Greedy; this means that heuristic did not find a
plan, not a proof that the workload is infeasible.
The scaling generator uses the measured batch-1, two-slice, 900 MHz reference
profiles `fr_b1_k2_f900` and `dl_b1_k2_f900`. It requires those exact profiles
rather than substituting another batch size. New generated workloads are
separate from the 40 frozen inputs in `data/inputs/scaling_cases.json`.
Both generated and frozen cases can be supplied to the planning interface.

## Algorithms and admission

- `greedy-c` performs construction only.
- `greedy` adds four local-refinement passes.
- `greedy-beam` adds width-512 Beam search, retaining at most two states per
  signature in its first 256 positions. Mixed replica arrangements use at most
  two alternatives and four replicas. The best complete deployment is retained.
- `greedy-beam-bounded` adds the admissible upper-bound pruning illustrated in
  Figure 3 to the same Greedy initialization and Beam arrangement space. A
  partial deployment is pruned when its bound cannot improve the validated
  complete Greedy seed. Beam width and signature limits still apply, so completion does
  not certify a global optimum.
- `least-allocated` prioritizes GPUs with more free slices before insertion.
  `most-allocated` prioritizes GPUs with fewer. Both choose uniformly from all
  feasible insertion candidates with the best priority and require an integer
  `--seed`. The Python API takes the seed as `random_seed`.
- `exact`, `fixed-frequency` and `sequential` require Gurobi. Install
  `requirements/solver.txt` and provide a suitable license outside the repo.

Run an occupancy policy through the public planner from the repository root.

```sh
python scripts/run_case.py --cases results/planning/inputs.json --case unified_012 --method least-allocated --seed 0 --output runs/example/least-allocated.json
```

The seed controls the draws for this run and is recorded in the output. To
reproduce the published Figure 6 outcomes, use the [frozen replay](../experiments/occupancy/),
which uses the archived admission grid and derives a seed for each workload
from master seed 0.

The supplied performance results correspond to `greedy-beam`, which does not
prune by an objective upper bound. `greedy-beam-bounded` has no published
performance evaluation. It is available as a separate method for new runs.

The bound independently takes the maximum remaining utility and minimum direct
power over each service's arrangements, relaxing placement constraints. Occupied
GPUs keep their frequency; unused GPUs may take the lowest remaining active
baseline or stay idle. Thus an active baseline below idle cannot cause an unsafe
prune. Floating-point bounds are rounded outwards, and near-equal candidates are
retained. If Greedy finds no feasible seed, the variant searches without objective
pruning. New output records include the pruning threshold and counters.

Time limits and initialization are explicit run settings. Beam and Exact have
no default time limit; the Exact limit applies to each solve, not the complete
pipeline. A time-limited incumbent is not an optimality certificate. Exact
supports cold execution or a new Greedy+Beam initializer. Fixed-frequency solves all catalog frequencies;
Sequential refines each feasible fixed plan with memberships and counts held
fixed. Both retain the best result. These exact-method initializers use
`greedy-beam`.

`results/scaling/protocol.json` describes the frozen workload design and
statistical conventions used for the supplied scaling records.

Admission defaults to `p95`; `--admission nominal` and `--admission p99` are
explicit alternatives. Quantiles concern each replica's window-mean response,
not individual-request latency or a fleet-wide joint probability. The measured
rate domain, stability, quality, slice, frequency and power checks apply in all
modes. Alternative admission grids can be provided with `--grid`.

## New simulation and acquisition

`scientific/preparation/window_guard.py` provides the portable queue-simulation
interface and mathematical self-checks. New simulations use explicitly chosen
inputs, seeds and measurement parameters and produce separate observations.
Use `--samples-output runs/example/samples.npz` alongside `--output` to save
the sampled window means, request counts and simulation metadata. The saved
arrays support direct recalculation of the summary quantiles using the recorded
linear interpolation method.

To replay the supplied admission-grid simulations with their recorded per-row
seeds and parameters, run:

```sh
python scripts/audit_admission_simulation.py --workers 2
```

This CPU audit compares all 1,354,752 window means and request counts with
`data/inputs/admission_simulation_source.npz`. Counts must match exactly;
means allow an absolute floating-point tolerance of `1e-12` seconds. The report
is written to `build/audit/admission_simulation.json`. Ordinary figure rendering
uses the saved records and does not rerun these simulations.

The CPU-only unit tests cover the wrapper and constraints without Gurobi.
Hardware measurements require the external assets and GPU environment listed
in [hardware](hardware.md). Reproduction of supplied figures does not invoke
these acquisition scripts.
