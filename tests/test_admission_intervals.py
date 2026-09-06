"""Analytical boundary checks independent of the supplied experiment records."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial import Polynomial


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scientific/planning" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rates = load("interval_test_rates", "rate_intervals.py")
with patch.dict(sys.modules, {"rate_intervals": rates}):
    guard = load("interval_test_guard", "guard_intervals.py")


class AdmissionBoundaryTests(unittest.TestCase):
    def test_roundoff_at_a_double_root_does_not_erase_boundary(self):
        # This representation can yield 0.5 +/- 1.05e-8j in the eigensolver.
        polynomial = Polynomial([.5500000000000003, -2.2, 2.2])
        self.assertEqual(len(rates.roots_unit(polynomial)), 1)
        self.assertAlmostEqual(rates.roots_unit(polynomial)[0], .5, places=12)

    def test_positive_minimum_is_not_a_root(self):
        polynomial = Polynomial([.25 + 1e-9, -1., 1.])
        self.assertEqual(rates.roots_unit(polynomial), [])

    def test_guard_tangent_admits_only_the_analytical_minimum(self):
        # B=2, L=.1, SCV=0: R(r)=1/(2r)+.1+.00125r/(1-.05r).
        # Its unique minimum is R(10)=.175. With factor 1.1 the deadline
        # .1925 admits only r=10, while the nominal model admits a wide interval.
        profile = dict(physical_config_id="tangent", B=2,
                       points=[[5., .1, 0.], [15., .1, 0.]])
        guard._guard_intervals.cache_clear()
        with patch.object(guard, "_nodes", return_value={"tangent": [(5., 1.1), (15., 1.1)]}):
            intervals, _ = guard.guarded_intervals(profile, .1925)
        self.assertEqual(len(intervals), 1)
        np.testing.assert_allclose(intervals[0], [10., 10.], atol=1e-6, rtol=0)
        self.assertFalse(rates.member([7., 13.], intervals).any())
        self.assertTrue(rates.member([10.], intervals)[0])

    def test_disjoint_nearby_intervals_do_not_create_reversed_interval(self):
        self.assertEqual(guard._intersect([[1., 2.]], [[2. + 5e-11, 3.]]), [])
        self.assertEqual(guard._intersect([[1., 2.]], [[2., 3.]]), [[2., 2.]])

    def test_midpoint_roundoff_tolerance_cannot_admit_a_whole_segment(self):
        profile = dict(physical_config_id="near_tangent", B=2,
                       points=[[5., .1, 0.], [15., .1, 0.]])
        guard._guard_intervals.cache_clear()
        with patch.object(guard, "_nodes", return_value={"near_tangent": [(5., 1.1), (15., 1.1)]}):
            intervals, _ = guard.guarded_intervals(profile, .1925 - 5e-11)
        self.assertEqual(intervals, [])


if __name__ == "__main__":
    unittest.main()
