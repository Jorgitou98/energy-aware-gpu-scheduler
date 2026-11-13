"""Check completion bounds against exhaustive, small software fixtures."""

from itertools import product
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from efficient_mig import planning


class TinyProblem:
    """One-replica choices, with independent resource and objective checks."""

    def __init__(self, choices, *, gpus=2, idle=10.0, bases=None):
        self.G, self.R, self.idle, self.cap = gpus, 4 * gpus, idle, 100.0
        self.base_by_frequency = bases or {900: 10.0, 1200: 14.0}
        self.ss = [dict(lam=1.0) for _ in choices]
        self.options = []
        for service, alternatives in enumerate(choices):
            for alternative in alternatives:
                rid = len(self.options)
                self.options.append(dict(s=service, g=0, rid=rid,
                    model=f"variant_{rid}", lower=0.0, upper=2.0,
                    k=alternative.get("k", 1), f=alternative.get("f", 900),
                    peak=alternative.get("peak", 3.0),
                    intercept=alternative.get("intercept", 1.0),
                    slope=alternative.get("slope", 1.0), u=alternative["u"]))

    def active_base(self, frequency):
        return self.base_by_frequency[frequency]

    def totals(self, assignments):
        # Sum complete per-GPU power, independently of the planner's
        # fleet-idle-plus-increments implementation.
        powers = []
        for gpu in range(self.G):
            groups = [r for r in assignments if r["g"] == gpu]
            if not groups:
                powers.append(self.idle)
            else:
                powers.append(self.active_base(groups[0]["f"]) + math.fsum(
                    r["intercept"] * r["n"] + r["slope"] * r["rate"]
                    for r in groups))
        return math.fsum(powers), math.fsum(r["u"] * r["rate"] for r in assignments)

    def from_groups(self, assignments):
        assignments = list(assignments)
        power, utility = self.totals(assignments)
        return dict(power=power, utility=utility, ratio=utility / power,
                    replicas=sum(r["n"] for r in assignments),
                    active=len({r["g"] for r in assignments}),
                    assignment_json=json.dumps(assignments))

    def feasible_prefix(self, assignments):
        if len({r["s"] for r in assignments}) != len(assignments):
            return False
        for r in assignments:
            if r["n"] != 1 or r["rate"] != 1.0 or not 0 <= r["g"] < self.G:
                return False
        for gpu in range(self.G):
            groups = [r for r in assignments if r["g"] == gpu]
            if not groups:
                continue
            if len({r["f"] for r in groups}) != 1:
                return False
            if sum(r["k"] for r in groups) > 4:
                return False
            if self.active_base(groups[0]["f"]) + sum(r["peak"] for r in groups) > self.cap:
                return False
        return True

    def validate(self, result):
        assignments = json.loads(result["assignment_json"])
        if not self.feasible_prefix(assignments) or {r["s"] for r in assignments} != set(range(len(self.ss))):
            raise ValueError("Invalid complete fixture deployment")
        rebuilt = self.from_groups(assignments)
        if any(not math.isclose(result[k], rebuilt[k], rel_tol=1e-12)
               for k in ("power", "utility", "ratio")):
            raise ValueError("Incorrect fixture objective")
        return True


def enumerate_placements(problem, services, prefix=()):
    """Enumerate literal option/GPU products, without heuristic placement code."""
    choices = [[dict(o, g=gpu, n=1, rate=1.0)
                for o in problem.options if o["s"] == service
                for gpu in range(problem.G)] for service in services]
    for selected in product(*choices):
        assignments = tuple(prefix) + selected
        if problem.feasible_prefix(assignments):
            yield assignments


class BoundedBeamChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = planning._runtime("nominal", str(planning.DEFAULT_GRID))[1]
        cls.bounded = planning._bounded_runtime(cls.base)

    def fixture(self):
        return TinyProblem([
            [dict(u=3.0, k=3), dict(u=6.0, f=1200, intercept=4.0)],
            [dict(u=5.0, k=2), dict(u=4.0, f=1200)],
            [dict(u=2.0), dict(u=3.0, f=1200, intercept=2.0)],
        ])

    def state(self, problem, assignments):
        resources = []
        for gpu in range(problem.G):
            selected = [r for r in assignments if r["g"] == gpu]
            resources.append((selected[0]["f"] if selected else 0,
                              sum(r["k"] for r in selected),
                              sum(r["peak"] for r in selected)))
        power, utility = problem.totals(assignments)
        return self.base.State(tuple(assignments), tuple(resources), power, utility)

    def test_every_small_completion_is_below_its_prefix_bound(self):
        problem = self.fixture()
        candidates = [self.base.arrangements(problem, s) for s in range(len(problem.ss))]
        order = [0, 1, 2]
        bound = self.bounded.CompletionBounds(problem, candidates, order)
        comparisons = 0
        for next_level in range(len(order) + 1):
            for prefix in enumerate_placements(problem, order[:next_level]):
                upper = bound.upper_ratio(self.state(problem, prefix), next_level)
                for completion in enumerate_placements(problem, order[next_level:], prefix):
                    actual = problem.from_groups(completion)["ratio"]
                    self.assertGreaterEqual(upper, actual,
                        (next_level, prefix, completion, upper, actual))
                    comparisons += 1
        self.assertGreater(comparisons, 40)

    def test_opening_lower_baseline_gpu_can_reduce_total_power(self):
        problem = TinyProblem([[dict(u=8.0)], [dict(u=2.0, f=1200)]],
                              bases={900: 10.0, 1200: 1.0})
        first = dict(problem.options[0], g=0, n=1, rate=1.0)
        second = dict(problem.options[1], g=1, n=1, rate=1.0)
        state = self.state(problem, [first])
        completed = problem.from_groups([first, second])
        candidates = [self.base.arrangements(problem, s) for s in range(2)]
        bound = self.bounded.CompletionBounds(problem, candidates, [0, 1])
        self.assertEqual(state.power, 22.0)
        self.assertEqual(completed["power"], 15.0)
        self.assertAlmostEqual(completed["ratio"], 2 / 3)
        self.assertGreaterEqual(bound.upper_ratio(state, 1), completed["ratio"])

    def test_wide_beam_matches_exhaustive_optimum_and_prunes(self):
        problem = self.fixture()
        plans = [problem.from_groups(a) for a in enumerate_placements(problem, [0, 1, 2])]
        optimum = max(plans, key=lambda r: r["ratio"])
        seed = sorted(plans, key=lambda r: r["ratio"])[len(plans) // 2]
        self.assertLess(seed["ratio"], optimum["ratio"])
        original = self.base.beam(problem, seed, width=10000)
        result = self.bounded.beam(problem, seed, width=10000)
        self.assertEqual(original["ratio"], optimum["ratio"])
        self.assertEqual(result["ratio"], optimum["ratio"])
        self.assertGreater(result["bound_pruned"], 0)
        self.assertGreaterEqual(result["bound_evaluations"], result["bound_pruned"])
        self.assertEqual(result["algorithm"], "beam_bound_pruning")
        self.assertTrue(result["beam_finished"])
        self.assertTrue(problem.validate(result))

    def test_seedless_search_finds_exhaustive_optimum(self):
        problem = self.fixture()
        expected = max(problem.from_groups(a)["ratio"]
                       for a in enumerate_placements(problem, [0, 1, 2]))
        result = self.bounded.beam(problem, width=10000)
        self.assertEqual(result["ratio"], expected)
        self.assertIsNone(result["pruning_reference_ratio"])
        self.assertEqual(result["bound_pruned"], 0)
        self.assertTrue(problem.validate(result))

    def test_fully_pruned_search_returns_valid_seed(self):
        problem = TinyProblem([[dict(u=1.0), dict(u=10.0)]], gpus=1)
        inferior, superior = [dict(o, n=1, rate=1.0) for o in problem.options]
        seed = problem.from_groups([superior])
        # A feasible seed may use a family omitted from the bounded candidate
        # generator. It must survive even when every generated branch loses.
        with patch.object(self.base, "arrangements", return_value=[(inferior,)]):
            result = self.bounded.beam(problem, seed)
        self.assertEqual(result["assignment_json"], seed["assignment_json"])
        self.assertGreater(result["bound_pruned"], 0)
        self.assertTrue(result["beam_finished"])
        self.assertFalse(result["beam_timed_out"])
        self.assertTrue(problem.validate(result))

    def test_near_equal_bound_does_not_prune(self):
        problem = TinyProblem([[dict(u=1.0), dict(u=1.0 + 5e-14)]], gpus=1)
        inferior, superior = [dict(o, n=1, rate=1.0) for o in problem.options]
        seed = problem.from_groups([superior])
        with patch.object(self.base, "arrangements", return_value=[(inferior,)]):
            result = self.bounded.beam(problem, seed)
        self.assertEqual(result["bound_pruned"], 0)
        self.assertGreater(result["beam_expanded"], 0)
        self.assertEqual(result["ratio"], seed["ratio"])

    def test_zero_utility_does_not_require_positive_incumbent(self):
        problem = TinyProblem([[dict(u=0.0)]], gpus=1)
        plan = problem.from_groups([dict(problem.options[0], n=1, rate=1.0)])
        result = self.bounded.beam(problem, plan)
        self.assertEqual(result["ratio"], 0.0)
        self.assertTrue(result["beam_finished"])
        self.assertTrue(problem.validate(result))

    def test_timeout_preserves_complete_seed(self):
        problem = self.fixture()
        seed = problem.from_groups(next(enumerate_placements(problem, [0, 1, 2])))
        result = self.bounded.beam(problem, seed, time_limit=1e-300)
        self.assertTrue(result["beam_timed_out"])
        self.assertFalse(result["beam_finished"])
        self.assertEqual(result["assignment_json"], seed["assignment_json"])
        self.assertTrue(problem.validate(result))

    def test_timeout_during_search_retains_improved_complete_candidate(self):
        problem = TinyProblem([[dict(u=1.0), dict(u=10.0)]], gpus=1)
        seed = problem.from_groups([dict(problem.options[0], n=1, rate=1.0)])
        original_extensions = self.base._extensions

        def interrupt_after_improvement(*args, **kwargs):
            for state in original_extensions(*args, **kwargs):
                yield state
                if state.utility / state.power > seed["ratio"]:
                    raise self.base._BeamDeadline

        with patch.object(self.base, "_extensions", side_effect=interrupt_after_improvement):
            result = self.bounded.beam(problem, seed)
        self.assertTrue(result["beam_timed_out"])
        self.assertFalse(result["beam_finished"])
        self.assertGreater(result["ratio"], seed["ratio"])
        self.assertTrue(problem.validate(result))

    def test_underflow_cannot_turn_a_positive_bound_into_zero(self):
        utility = math.nextafter(0.0, math.inf)
        problem = TinyProblem([[dict(u=utility, intercept=0.0, slope=0.0)]], gpus=1)
        item = dict(problem.options[0], n=1, rate=0.5)
        bounds = self.bounded.CompletionBounds(problem, [[(item,)]], [0])
        upper = bounds.upper_ratio(self.state(problem, []), 0)
        exact = Fraction(utility) * Fraction(0.5) / Fraction(problem.idle)
        self.assertGreaterEqual(Fraction(upper), exact)


if __name__ == "__main__":
    unittest.main()
