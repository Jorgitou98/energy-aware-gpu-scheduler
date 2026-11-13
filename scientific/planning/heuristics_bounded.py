"""Beam variant with an admissible completion bound and a Greedy fallback.

The placement rules, arrangement families and width/diversity selection come
from ``heuristics``. This variant additionally discards partial deployments
whose optimistic final U/P cannot improve the validated complete seed.
The width limit remains heuristic; finishing this search is not an optimality
certificate. Load through the planning interface's ``greedy-beam-bounded``
method, which supplies the shared ``heuristics`` module.
"""

import json
import math
from time import perf_counter

import heuristics as base


def _up(value):
    return math.nextafter(value, math.inf)


def _down(value):
    return max(0.0, math.nextafter(value, -math.inf))


def _utility_upper(assignments):
    total = 0.0
    for item in assignments:
        total = _up(total + _up(item["u"] * item["rate"]))
    return total


def _direct_lower(assignments):
    total = 0.0
    for item in assignments:
        residency = _down(item["intercept"] * item["n"])
        traffic = _down(item["slope"] * item["rate"])
        total = _down(total + _down(residency + traffic))
    return total


class CompletionBounds:
    """Relax packing and independently maximize utility/minimize power.

    For each remaining service, the upper numerator uses its most useful
    arrangement and the lower denominator uses its cheapest direct power.
    Occupied GPUs retain their frequency. Each unused GPU is allowed the
    lesser of idle power and any remaining arrangement's active baseline,
    even when that optimistic combination could not actually be packed.
    This handles active baselines below idle without assuming power increases.

    All model coefficients are nonnegative. Directed floating-point rounding
    enlarges the numerator and reduces the denominator; a nonpositive lower
    denominator disables pruning. A separate comparison margin retains ties.
    """

    def __init__(self, problem, candidates, order, *, deadline=None):
        self.problem = problem
        size = len(order)
        self.utility = [0.0] * (size + 1)
        self.direct_power = [0.0] * (size + 1)
        self.unused_base = [problem.idle] * (size + 1)
        for level in range(size - 1, -1, -1):
            service = order[level]
            utility = 0.0
            direct_power = math.inf
            baseline = self.unused_base[level + 1]
            for arrangement in candidates[service]:
                base._check_deadline(deadline)
                utility = max(utility, _utility_upper(arrangement))
                direct_power = min(direct_power, _direct_lower(arrangement))
                for item in arrangement:
                    baseline = min(baseline, problem.active_base(item["f"]))
            self.utility[level] = _up(utility + self.utility[level + 1])
            self.direct_power[level] = _down(direct_power + self.direct_power[level + 1])
            self.unused_base[level] = baseline

    def upper_ratio(self, state, next_level):
        utility = _up(_utility_upper(state.assignments) + self.utility[next_level])
        power = _down(_direct_lower(state.assignments) + self.direct_power[next_level])
        for frequency, slices, _ in state.resources:
            baseline = (self.problem.active_base(frequency) if slices
                        else self.unused_base[next_level])
            power = _down(power + baseline)
        if power <= 0:
            return math.inf
        return _up(utility / power)


def _cannot_improve(upper, threshold):
    # Retain equality and nearly equal candidates rather than pruning an
    # improvement hidden by finite-precision objective calculations.
    return (upper < threshold and
            not math.isclose(upper, threshold, rel_tol=1e-12, abs_tol=1e-12))


def beam(problem, seed_plan=None, *, width=512, per_signature=2, progress=None,
         time_limit=None):
    """Run bounded-width search with completion-bound pruning against the seed.

    Without a feasible seed there is no pruning threshold. A supplied seed is
    rebuilt and validated before use; its stored ratio is never trusted.
    Deadline checks cover arrangements, bound preparation and placements.
    """
    if (isinstance(width, bool) or not isinstance(width, int) or width < 1 or
            isinstance(per_signature, bool) or not isinstance(per_signature, int)
            or per_signature < 1):
        raise ValueError("Beam width and signature cap must be positive integers")
    if time_limit is not None and (isinstance(time_limit, bool) or
                                  not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be positive and finite")
    started = perf_counter()
    deadline = None if time_limit is None else started + time_limit
    best = None
    if seed_plan and seed_plan.get("assignment_json"):
        best = dict(status="feasible", **problem.from_groups(
            json.loads(seed_plan["assignment_json"])))
        problem.validate(best)
    threshold = best["ratio"] if best is not None else None
    expanded = pruned = evaluations = 0
    finished = timed_out = False
    levels = []

    def result(**extra):
        record = dict(best or {"status": "no_solution_limit" if timed_out else "construction_failed"},
                      seconds=perf_counter() - started,
                      algorithm="beam_bound_pruning", beam_width=width,
                      per_signature=per_signature, beam_expanded=expanded,
                      beam_finished=finished, beam_timed_out=timed_out,
                      time_limit_seconds=time_limit, beam_levels=levels,
                      bound_pruned=pruned, bound_evaluations=evaluations,
                      pruning_reference_ratio=threshold, **extra)
        if progress is not None:
            progress(dict(event="complete", elapsed_seconds=record["seconds"],
                          beam_finished=finished, status=record["status"],
                          best_ratio=record.get("ratio"), expanded=expanded,
                          bound_pruned=pruned))
        return record

    try:
        candidates = [base.arrangements(problem, service, deadline=deadline)
                      for service in range(len(problem.ss))]
        if any(not choices for choices in candidates):
            return result(reason="no_arrangements")
        order = sorted(range(len(problem.ss)), key=lambda service:
                       (len(candidates[service]), -problem.ss[service]["lam"]))
        minima = [min(sum(item["k"] for item in arrangement)
                      for arrangement in candidates[service]) for service in order]
        bounds = (CompletionBounds(problem, candidates, order, deadline=deadline)
                  if threshold is not None else None)
    except base._BeamDeadline:
        timed_out = True
        return result(interrupted_phase="preparation")

    def discard(state, next_level):
        nonlocal pruned, evaluations
        if bounds is None:
            return False
        evaluations += 1
        if _cannot_improve(bounds.upper_ratio(state, next_level), threshold):
            pruned += 1
            return True
        return False

    states = [base.State((), tuple((0, 0, 0.0) for _ in range(problem.G)),
                         problem.G * problem.idle, 0.0)]
    best_complete_state = None
    for level, service in enumerate(order):
        remaining_slices = sum(minima[level + 1:])
        selector = base.StreamingBeam(width, per_signature)
        level_pruned = pruned
        try:
            for state in states:
                base._check_deadline(deadline)
                if discard(state, level):
                    continue
                occupied = sum(slices for _, slices, _ in state.resources)
                for arrangement in candidates[service]:
                    base._check_deadline(deadline)
                    if (occupied + sum(item["k"] for item in arrangement)
                            + remaining_slices > 4 * problem.G):
                        continue
                    for extension in base._extensions(problem, state, arrangement,
                                                       deadline=deadline):
                        expanded += 1
                        if level == len(order) - 1 and (best_complete_state is None or
                                extension.utility / extension.power >
                                best_complete_state.utility / best_complete_state.power):
                            best_complete_state = extension
                        if discard(extension, level + 1):
                            continue
                        selector.add(extension, base.deployment_key(extension.assignments, problem.G))
        except base._BeamDeadline:
            timed_out = True
        states = selector.finish()
        finished = not timed_out and (not states or level == len(order) - 1)
        report = dict(event="level_complete", level=level + 1, service=service,
                      candidates=selector.generated, retained=len(states),
                      peak_retained_state_slots=selector.peak_retained,
                      exact_duplicate_hits_while_retained=selector.duplicate_hits,
                      expanded=expanded, elapsed_seconds=perf_counter() - started,
                      bound_pruned=pruned - level_pruned,
                      best_ratio=best.get("ratio") if best else None,
                      beam_finished=finished)
        levels.append(report)
        if progress is not None:
            progress(report)
        if timed_out or finished:
            break

    if best_complete_state is not None:
        candidate = problem.from_groups(best_complete_state.assignments)
        problem.validate(candidate)
        if best is None or candidate["ratio"] > best["ratio"]:
            best = dict(status="feasible", **candidate)
    return result()
