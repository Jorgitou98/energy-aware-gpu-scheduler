# Planning implementation

- `model_adapter.py` compiles the measured catalog, validates deployments and
  implements the fractional MILP with Dinkelbach iterations.
- `rate_intervals.py` compiles nominal mean-response intervals without
  extrapolating beyond the measured rate domain.
- `guard_intervals.py` intersects those intervals with the selected P95 or P99
  finite-window admission curve. Quantiles concern window means, not requests.
- `heuristics.py` provides one-replica Greedy construction, four local-refinement
  passes and multi-replica Beam search. Beam width is 512, with a per-signature
  cap of 2 in its first 256 positions. Mixed arrangements use at most two
  alternatives and four replicas. Uniform arrangements retain the fleet bound.
- `heuristics_bounded.py` provides the `greedy-beam-bounded` alternative, adding
  the admissible upper-bound pruning illustrated in Figure 3. It retains the
  same arrangement space and Beam limits; finishing this search is not a global
  optimality certificate. The supplied performance results use `heuristics.py`
  without this pruning. The bounded variant has no published performance
  evaluation.
- `occupancy.py` provides LeastAllocated and MostAllocated construction based on
  the target GPU's free slices before insertion. Each insertion samples
  uniformly from all feasible candidates with the best priority. New runs
  require `--seed`, which is recorded in the output.

Use `scripts/run_case.py` with explicit workload inputs. A time limit is optional
for new Beam runs; none is silently imposed. New runtime measurements are not
the paper's published timings. Exact optimization requires Gurobi and a license;
catalog compilation, validation and the heuristics do not.

This implementation specializes the general formulation to identical four-slice
NVIDIA A30 GPUs. With supported instance sizes 1, 2 and 4, the slice-capacity
constraint admits precisely the partial allocations of legal A30 layouts.
The power cap uses busy-power reservation, while the objective uses expected
traffic-dependent power and includes inactive-device power.
