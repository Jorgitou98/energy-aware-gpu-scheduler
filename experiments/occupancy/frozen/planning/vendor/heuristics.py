"""Algorithms 2 and 3 with explicit calibrated per-frequency GPU baselines.

This module has no solver, array-library, or GPU dependency. ``problem`` is the
profile-derived model interface preserved under ``frozen``: options, services,
fleet constants, ``from_groups``, and ``validate``. The implementation assumes
the study's identical four-slice GPUs and common-frequency constraint.

There is no implicit time budget, size cutoff, optimistic-bound stopping rule,
or merging of distinct deployments that happen to have the same footprint.
"""

from collections import Counter
from dataclasses import dataclass
import heapq
from itertools import combinations
from math import ceil, ulp
from time import perf_counter
import json


TOLERANCE = 1e-9


class _BeamDeadline(Exception):
    pass


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise _BeamDeadline


def _unique_options(problem, service):
    return [option for option in problem.options
            if option["s"] == service and option["g"] == 0]


def minimum_replicas(rate, upper):
    """Smallest integer count that admits a positive aggregate request rate."""
    if rate <= 0 or upper <= 0:
        raise ValueError("Positive rate and rate capacity are required")
    count = max(1, ceil(rate / upper))
    # Multiplication/division can put an exact capacity boundary one ulp above
    # its integer quotient. Match that boundary without a scale-wide epsilon.
    if count > 1 and abs(rate - (count - 1) * upper) <= 4 * ulp(float(rate)):
        count -= 1
    return count


def _totals(problem, assignments):
    return problem.totals(assignments)


def _result(problem, assignments, started, **metadata):
    if assignments is None:
        return dict(status="construction_failed", seconds=perf_counter() - started,
                    **metadata)
    result = problem.from_groups(assignments)
    problem.validate(result)
    return dict(status="feasible", seconds=perf_counter() - started,
                **result, **metadata)


def greedy(problem, *, include_trace=False):
    """Algorithm 2: one replica per service and four fixed-membership passes.

    Each insertion can reselect all configurations on its target GPU. Its
    increase in U-qP is evaluated against the actual current deployment.
    All GPUs pay their idle contribution, including during partial placement.
    """
    started = perf_counter()
    options = {}
    for service, specification in enumerate(problem.ss):
        rate = specification["lam"]
        options[service] = [item for item in _unique_options(problem, service)
                            if item["lower"] - TOLERANCE <= rate
                            <= item["upper"] + TOLERANCE]
    if any(not choices for choices in options.values()):
        return _result(problem, None, started, algorithm="manuscript_greedy")
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

    def best_local(members, q):
        best = None
        best_score = float("-inf")
        best_power = float("inf")
        for candidate in local_configs(members):
            power = problem.active_base(candidate[0]["f"]) - problem.idle + sum(
                item["intercept"] + item["slope"] * item["rate"]
                for item in candidate)
            utility = sum(item["u"] * item["rate"] for item in candidate)
            score = utility - q * power
            if (score > best_score + TOLERANCE or
                    (abs(score - best_score) <= TOLERANCE and power < best_power)):
                best, best_score, best_power = candidate, score, power
        return best

    assignments = []
    q = 0.0
    trace = []
    for service in order:
        power, utility = _totals(problem, assignments)
        active = sorted({item["g"] for item in assignments})
        targets = active + ([next(g for g in range(problem.G) if g not in active)]
                            if len(active) < problem.G else [])
        best, best_score, best_delta = None, float("-inf"), float("inf")
        for gpu in targets:
            members = [item["s"] for item in assignments if item["g"] == gpu]
            local = best_local(members + [service], q)
            if local is None:
                continue
            candidate = [item for item in assignments if item["g"] != gpu]
            candidate += [dict(item, g=gpu) for item in local]
            new_power, new_utility = _totals(problem, candidate)
            delta_power = new_power - power
            score = new_utility - utility - q * delta_power
            if (score > best_score + TOLERANCE or
                    (abs(score - best_score) <= TOLERANCE and delta_power < best_delta)):
                best, best_score, best_delta = candidate, score, delta_power
        if best is None:
            return _result(problem, None, started, algorithm="manuscript_greedy")
        assignments = best
        power, utility = _totals(problem, assignments)
        prior_utility = trace[-1]["utility"] if trace else 0.0
        trace.append(dict(service=service, q_before=q, delta_power=best_delta,
                          delta_utility=utility - prior_utility,
                          score=best_score, power=power, utility=utility,
                          q_after=utility / power,
                          elapsed_seconds=perf_counter() - started,
                          configurations=[(item["s"], item["g"], item["rid"])
                                          for item in assignments]))
        q = utility / power

    best = assignments
    best_ratio = q
    memberships = {gpu: [item["s"] for item in assignments if item["g"] == gpu]
                   for gpu in sorted({item["g"] for item in assignments})}
    refinements = []
    for pass_index in range(4):
        previous_q = q
        assignments = [dict(item, g=gpu) for gpu, members in memberships.items()
                       for item in best_local(members, q)]
        power, utility = _totals(problem, assignments)
        q = utility / power
        if q > best_ratio:
            best, best_ratio = assignments, q
        refinements.append(dict(pass_index=pass_index, q_before=previous_q,
                                q_after=q, best_ratio=best_ratio,
                                configurations=[(item["s"], item["g"], item["rid"])
                                                for item in assignments]))
    metadata = dict(algorithm="manuscript_greedy", refinement_passes=4)
    if include_trace:
        metadata["insertion_trace"] = trace
        metadata["refinement_trace"] = refinements
    return _result(problem, best, started, **metadata)


def arrangements(problem, service, *, deadline=None):
    """Uniform families plus two-alternative families with at most four replicas."""
    demand = problem.ss[service]["lam"]
    options = _unique_options(problem, service)
    result = []
    seen = set()

    def add(pattern):
        if sum(item["k"] * count for item, count in pattern) > 4 * problem.G:
            return
        rates = [item["lower"] * count for item, count in pattern]
        remaining = demand - sum(rates)
        if remaining < -TOLERANCE:
            return
        if sum(item["upper"] * count for item, count in pattern) < demand - TOLERANCE:
            return
        for index in sorted(range(len(pattern)), key=lambda i: pattern[i][0]["slope"]):
            item, count = pattern[index]
            extra = min(max(0.0, remaining), item["upper"] * count - rates[index])
            rates[index] += extra
            remaining -= extra
        if remaining > TOLERANCE:
            return
        replicas = []
        for (item, count), rate in zip(pattern, rates):
            if rate <= 0:
                # A zero-traffic replica is inactive. Its removal does not change
                # the admitted request allocation and avoids counting idle copies.
                continue
            # Section 4.1's minimum-count dominance also applies after mixed
            # traffic allocation. Reducing the count preserves the lower rate
            # bound (rate >= old_count * lower), while ceil(rate / upper)
            # preserves the upper bound. Do not retain unused positive copies.
            count = minimum_replicas(rate, item["upper"])
            replicas.extend(dict(item, n=1, rate=rate / count) for _ in range(count))
        key = tuple(sorted((item["rid"], item["rate"].hex()) for item in replicas))
        if key in seen:
            return
        seen.add(key)
        result.append(tuple(sorted(replicas,
                                  key=lambda item: (-item["k"], -item["peak"], item["rid"]))))

    for option in options:
        _check_deadline(deadline)
        count = minimum_replicas(demand, option["upper"])
        if count <= problem.R:
            add([(option, count)])
    for first, second in combinations(options, 2):
        _check_deadline(deadline)
        if first["model"] != second["model"]:
            continue
        for count_first in range(1, 4):
            for count_second in range(1, 5 - count_first):
                add([(first, count_first), (second, count_second)])
    return result


def deployment_key(assignments, gpu_count):
    """Exact assignment identity modulo permutations of identical GPUs.

    Rates are compared as binary floating-point values, never rounded. Service
    identity, alternative identity, and replica traffic remain in the key.
    """
    contents = [[] for _ in range(gpu_count)]
    for item in assignments:
        contents[item["g"]].append((item["s"], item["rid"], item["n"],
                                     float(item["rate"]).hex()))
    return tuple(sorted(tuple(sorted(gpu)) for gpu in contents))


@dataclass(frozen=True)
class State:
    assignments: tuple
    resources: tuple
    power: float
    utility: float


def _extensions(problem, state, replicas, *, deadline=None):
    """Place a complete arrangement, skipping only provably identical GPUs."""
    def visit(index, assignments, resources):
        _check_deadline(deadline)
        if index == len(replicas):
            power, utility = _totals(problem, assignments)
            yield State(assignments, resources, power, utility)
            return
        replica = replicas[index]
        equivalent = set()
        for gpu, (frequency, slices, peak) in enumerate(resources):
            content = tuple(sorted((item["s"], item["rid"], float(item["rate"]).hex())
                                   for item in assignments if item["g"] == gpu))
            if content in equivalent:
                continue
            equivalent.add(content)
            if frequency not in (0, replica["f"]):
                continue
            if slices + replica["k"] > 4:
                continue
            if problem.active_base(replica["f"]) + peak + replica["peak"] > problem.cap + TOLERANCE:
                continue
            updated = list(resources)
            updated[gpu] = (replica["f"], slices + replica["k"], peak + replica["peak"])
            yield from visit(index + 1, assignments + (dict(replica, g=gpu),), tuple(updated))

    yield from visit(0, state.assignments, state.resources)


def select_beam(states, width=512, per_signature=2):
    """Rank by current U/P; reserve half the beam for occupancy diversity."""
    ranked = sorted(states, key=lambda state: -(state.utility / state.power))
    selected = []
    selected_indices = set()
    counts = Counter()
    for index, state in enumerate(ranked):
        if len(selected) >= width // 2:
            break
        signature = tuple(sorted((frequency, slices)
                                 for frequency, slices, _ in state.resources))
        if counts[signature] < per_signature:
            selected.append(state)
            selected_indices.add(index)
            counts[signature] += 1
    for index, state in enumerate(ranked):
        if len(selected) >= width:
            break
        if index not in selected_indices:
            selected.append(state)
    return selected


@dataclass(frozen=True)
class RankedState:
    state: State
    key: tuple
    signature: tuple
    rank: tuple


class StreamingBeam:
    """Exact, bounded-memory implementation of deduplication and beam selection.

    Let G be the globally best ``width`` distinct states, and D the best
    ``width//2`` states with at most ``per_signature`` per signature. The result
    is D followed by the best unselected states from G. No other state can be
    selected by Algorithm 3. Both rank thresholds only tighten as candidates
    arrive, so a duplicate of a state discarded from both sets can never become
    eligible later. Exact keys are therefore needed only for retained states.

    Rank includes arrival order to reproduce the stable sort in select_beam.
    The heaps are periodically compacted so removed entries cannot grow with
    the size of the expanded level. At width 512, at most 768 full candidate
    states are retained, independent of the number of generated extensions.
    """

    def __init__(self, width=512, per_signature=2):
        self.width = width
        self.reserve = width // 2
        self.per_signature = per_signature
        self.global_states = {}
        self.diverse_states = {}
        self.global_heap = []
        self.diverse_heap = []
        self.signatures = {}
        self.generated = 0
        self.duplicate_hits = 0
        self.peak_retained = 0

    def _remove_diverse(self, key):
        record = self.diverse_states.pop(key)
        members = self.signatures[record.signature]
        members.remove(key)
        if not members:
            del self.signatures[record.signature]

    def _diverse_worst(self):
        while self.diverse_heap and self.diverse_heap[0][1] not in self.diverse_states:
            heapq.heappop(self.diverse_heap)
        return self.diverse_heap[0]

    def add(self, state, key):
        sequence = self.generated
        self.generated += 1
        if key in self.global_states or key in self.diverse_states:
            self.duplicate_hits += 1
            return
        signature = tuple(sorted((frequency, slices) for frequency, slices, _ in state.resources))
        record = RankedState(state, key, signature, (state.utility / state.power, -sequence))

        if len(self.global_states) < self.width:
            self.global_states[key] = record
            heapq.heappush(self.global_heap, (record.rank, key))
        elif record.rank > self.global_heap[0][0]:
            _, removed = heapq.heapreplace(self.global_heap, (record.rank, key))
            del self.global_states[removed]
            self.global_states[key] = record

        if self.reserve:
            members = self.signatures.get(signature, [])
            worst_same = min((self.diverse_states[item] for item in members),
                             key=lambda item: item.rank, default=None)
            beats_signature = len(members) < self.per_signature or record.rank > worst_same.rank
            beats_global = (len(self.diverse_states) < self.reserve or
                            record.rank > self._diverse_worst()[0])
            if beats_signature and beats_global:
                if len(members) == self.per_signature:
                    self._remove_diverse(worst_same.key)
                elif len(self.diverse_states) == self.reserve:
                    self._remove_diverse(self._diverse_worst()[1])
                self.diverse_states[key] = record
                self.signatures.setdefault(signature, []).append(key)
                heapq.heappush(self.diverse_heap, (record.rank, key))
                if len(self.diverse_heap) > 2 * self.reserve:
                    self.diverse_heap = [(item.rank, item.key) for item in self.diverse_states.values()]
                    heapq.heapify(self.diverse_heap)
        # A record in both selections is shared. Counting slots is a cheap
        # upper bound on retained full states, without an O(width) key union.
        retained = len(self.global_states) + len(self.diverse_states)
        self.peak_retained = max(self.peak_retained, retained)

    def finish(self):
        selected = sorted(self.diverse_states.values(), key=lambda item: item.rank, reverse=True)
        keys = set(self.diverse_states)
        for record in sorted(self.global_states.values(), key=lambda item: item.rank, reverse=True):
            if len(selected) >= self.width:
                break
            if record.key not in keys:
                selected.append(record)
        return [record.state for record in selected]


def beam(problem, seed_plan=None, *, width=512, per_signature=2, progress=None,
         time_limit=None):
    """Algorithm 3 with an optional explicit wall-time limit, no bound pruning.

    Deadline checks include arrangement generation and recursive placements.
    A fully assigned candidate already visited at the final level is retained
    on interruption; a partial deployment is never returned as feasible.
    """
    if width < 1 or per_signature < 1:
        raise ValueError("Beam width and signature cap must be positive")
    started = perf_counter()
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    deadline = None if time_limit is None else started + time_limit
    best = (dict(status="feasible", **problem.from_groups(json.loads(seed_plan["assignment_json"])))
            if seed_plan and seed_plan.get("assignment_json") else None)
    if best is not None:
        problem.validate(best)
    try:
        candidates = [arrangements(problem, service, deadline=deadline)
                      for service in range(len(problem.ss))]
    except _BeamDeadline:
        return dict(best or {"status": "no_solution_limit"}, seconds=perf_counter()-started,
                    beam_finished=False, beam_timed_out=True, beam_expanded=0,
                    time_limit_seconds=time_limit, algorithm="manuscript_beam",
                    interrupted_phase="arrangement_generation")
    if any(not choices for choices in candidates):
        result = dict(best or {"status": "construction_failed"},
                    seconds=perf_counter() - started, beam_finished=False,
                    algorithm="manuscript_beam", beam_expanded=0)
        if progress is not None:
            progress(dict(event="complete", elapsed_seconds=result["seconds"],
                          beam_finished=False, status=result["status"],
                          best_ratio=result.get("ratio"), reason="no_arrangements"))
        return result
    order = sorted(range(len(problem.ss)),
                   key=lambda service: (len(candidates[service]), -problem.ss[service]["lam"]))
    minima = [min(sum(item["k"] for item in arrangement)
                  for arrangement in candidates[service]) for service in order]
    states = [State((), tuple((0, 0, 0.0) for _ in range(problem.G)),
                    problem.G * problem.idle, 0.0)]
    expanded = 0
    finished = False
    levels = []
    timed_out = False
    best_complete_state = None
    for level, service in enumerate(order):
        remaining_slices = sum(minima[level + 1:])
        selector = StreamingBeam(width, per_signature)
        try:
            for state in states:
                _check_deadline(deadline)
                occupied = sum(slices for _, slices, _ in state.resources)
                for arrangement in candidates[service]:
                    _check_deadline(deadline)
                    if occupied + sum(item["k"] for item in arrangement) + remaining_slices > 4 * problem.G:
                        continue
                    for extension in _extensions(problem, state, arrangement, deadline=deadline):
                        expanded += 1
                        key = deployment_key(extension.assignments, problem.G)
                        selector.add(extension, key)
                        if level == len(order)-1 and (best_complete_state is None or
                                extension.utility/extension.power > best_complete_state.utility/best_complete_state.power):
                            best_complete_state = extension
        except _BeamDeadline:
            timed_out = True
        states = selector.finish()
        if not states:
            break
        if level == len(order) - 1:
            finished = not timed_out
            for state in states:
                result = problem.from_groups(state.assignments)
                problem.validate(result)
                if best is None or result["ratio"] > best["ratio"]:
                    best = dict(status="feasible", **result)
        report = dict(event="level_complete", level=level + 1, service=service,
                      candidates=selector.generated, retained=len(states),
                      peak_retained_state_slots=selector.peak_retained,
                      exact_duplicate_hits_while_retained=selector.duplicate_hits,
                      expanded=expanded, elapsed_seconds=perf_counter() - started,
                      best_ratio=best.get("ratio") if best else None,
                      beam_finished=finished)
        levels.append(report)
        if progress is not None:
            progress(report)
        if timed_out:
            break
    if best_complete_state is not None:
        candidate = problem.from_groups(best_complete_state.assignments)
        problem.validate(candidate)
        if best is None or candidate["ratio"] > best["ratio"]:
            best = dict(status="feasible", **candidate)
    result = dict(best or {"status": "no_solution_limit" if timed_out else "construction_failed"},
                seconds=perf_counter() - started, beam_finished=finished,
                algorithm="manuscript_beam", beam_expanded=expanded,
                beam_width=width, per_signature=per_signature, beam_levels=levels,
                beam_timed_out=timed_out, time_limit_seconds=time_limit)
    if progress is not None:
        progress(dict(event="complete", elapsed_seconds=result["seconds"],
                      beam_finished=finished, status=result["status"],
                      best_ratio=result.get("ratio"), expanded=expanded))
    return result


def proposed(problem, *, progress=None):
    """Run both manuscript stages for every size, without an implicit cutoff.

    Call ``greedy`` explicitly for the greedy-only ablation. ``progress`` is an
    optional observational callback after each fully expanded beam level.
    """
    started = perf_counter()
    seed = greedy(problem)
    result = beam(problem, seed, progress=progress)
    return dict(result, seconds=perf_counter() - started,
                mode="greedy_and_beam")


# Source is retained unchanged outside this isolated calibration adapter.
SOURCE_SHA256 = '8860ad561d211d1c5cc6ebaab0acb84816e1c7bd42c1588cc6e1650369166040'
# Adaptations: totals delegation, active baseline in local score and both cap checks.
