"""Occupancy baselines from the final paper's method table.

Rank target GPUs by free slices before insertion and draw uniformly from
all feasible candidates tied at the best priority using an explicit seed.
Utility and power rank the GreedyScore control. Power constrains feasibility
for every policy.
"""
from time import perf_counter
import random
TOLERANCE = 1e-9


def _unique_options(problem, service):
    return [option for option in problem.options if option["s"] == service and option["g"] == 0]


def _totals(problem, assignments):
    return problem.totals(assignments)


def _result(problem, assignments, started, **metadata):
    if assignments is None:
        return dict(status="construction_failed", seconds=perf_counter() - started, **metadata)
    result = problem.from_groups(assignments)
    problem.validate(result)
    return dict(status="feasible", seconds=perf_counter() - started, **result, **metadata)

POLICIES = ("GreedyScore", "LeastAllocated", "MostAllocated")

def placement_priority(policy, occupied_slices):
    """Integer priority based only on the target's current partial deployment."""
    slices = occupied_slices
    if not 0 <= slices <= 4:
        raise ValueError("Invalid four-slice occupancy")
    if policy == "GreedyScore":
        return 0
    if policy == "MostAllocated":
        return 100 * slices // 4
    if policy == "LeastAllocated":
        return 100 * (4 - slices) // 4
    raise ValueError(policy)


def plan(problem, policy, *, seed=None, include_trace=False):
    """Original Greedy-C candidates with pre-insertion GPU-priority ranking.

    GreedyScore uses the efficiency score as a matched control.
    Feasibility and service/candidate generation match Greedy construction.
    """
    if policy not in POLICIES:
        raise ValueError(policy)
    if policy != "GreedyScore" and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("Occupancy policies require an explicit integer random seed")
    rng = random.Random(seed) if policy != "GreedyScore" else None
    started = perf_counter()
    options = {}
    for service, specification in enumerate(problem.ss):
        rate = specification["lam"]
        options[service] = [item for item in _unique_options(problem, service)
                            if item["lower"] - TOLERANCE <= rate
                            <= item["upper"] + TOLERANCE]
    if any(not choices for choices in options.values()):
        return _result(problem, None, started, algorithm="greedy_c_occupancy", policy=policy, seed=seed,
                           refinement_passes=0, insertion_trace=[])
    order = sorted(options, key=lambda service:
                   (len(options[service]), -problem.ss[service]["lam"]))
    configurations = {}

    def local_configs(members):
        key = tuple(sorted(members))
        if key in configurations:
            return configurations[key]
        feasible = []

        def visit(position, selected, frequency, slices, peak):
            if position == len(key):
                feasible.append(tuple(selected))
                return
            service = key[position]
            for option in options[service]:
                if frequency is not None and option["f"] != frequency:
                    continue
                if slices + option["k"] > 4:
                    continue
                if problem.active_base(option["f"]) + peak + option["peak"] > problem.cap + TOLERANCE:
                    continue
                item = dict(option, n=1, rate=problem.ss[service]["lam"])
                visit(position + 1, selected + [item], option["f"],
                      slices + option["k"], peak + option["peak"])

        if len(key) <= 4:
            visit(0, [], None, 0, 0.0)
        configurations[key] = feasible
        return feasible

    assignments = []
    q = 0.0
    trace = []
    for service in order:
        power, utility = _totals(problem, assignments)
        active = sorted({item["g"] for item in assignments})
        targets = active + ([next(g for g in range(problem.G) if g not in active)]
                            if len(active) < problem.G else [])
        best, best_score, best_delta = None, float("-inf"), float("inf")
        best_priority = float("-inf")
        best_gpu = best_free = None
        tied_candidates = []
        for gpu in targets:
            members = [item["s"] for item in assignments if item["g"] == gpu]
            occupied_before = sum(item["k"] * item["n"] for item in assignments
                                  if item["g"] == gpu)
            priority = placement_priority(policy, occupied_before)
            # All feasible (a2) candidates remain eligible. Resident variants
            # may change, as in Greedy-C; this is not a fixed-reservation test.
            for local in local_configs(members + [service]):
                candidate = [item for item in assignments if item["g"] != gpu]
                candidate += [dict(item, g=gpu) for item in local]
                if policy == "GreedyScore":
                    new_power, new_utility = _totals(problem, candidate)
                    delta_power = new_power - power
                    score = new_utility - utility - q * delta_power
                    if (score > best_score + TOLERANCE or
                            (abs(score - best_score) <= TOLERANCE and delta_power < best_delta)):
                        best, best_score, best_delta = candidate, score, delta_power
                        best_priority = priority
                        best_gpu, best_free = gpu, 4 - occupied_before
                else:
                    if priority > best_priority:
                        best_priority = priority
                        tied_candidates = []
                    if priority == best_priority:
                        tied_candidates.append((candidate, gpu, 4 - occupied_before))
        choice_index = None
        if rng is not None and tied_candidates:
            choice_index = rng.randrange(len(tied_candidates))
            best, best_gpu, best_free = tied_candidates[choice_index]
            best_delta = _totals(problem, best)[0] - power
            best_score = None
        if best is None:
            return _result(problem, None, started, algorithm="greedy_c_occupancy", policy=policy, seed=seed,
                           refinement_passes=0, insertion_trace=trace if include_trace else [])
        assignments = best
        power, utility = _totals(problem, assignments)
        prior_utility = trace[-1]["utility"] if trace else 0.0
        trace.append(dict(service=service, q_before=q, delta_power=best_delta,
                          delta_utility=utility - prior_utility,
                          score=best_score, power=power, utility=utility,
                          target_gpu=best_gpu, free_slices_before=best_free,
                          placement_priority=best_priority,
                          random_pool_size=len(tied_candidates) if rng is not None else None,
                          random_choice_index=choice_index,
                          free_slices_after=4 - sum(item["k"] * item["n"] for item in assignments
                                                    if item["g"] == best_gpu),
                          q_after=utility / power,
                          elapsed_seconds=perf_counter() - started,
                          configurations=[(item["s"], item["g"], item["rid"])
                                          for item in assignments]))
        q = utility / power

    return _result(problem, assignments, started, algorithm="greedy_c_occupancy",
                   policy=policy, seed=seed, refinement_passes=0,
                   insertion_trace=trace if include_trace else [])
