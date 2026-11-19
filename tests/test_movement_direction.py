"""Behavioral correctness checks for the revised offline kinematics."""
import unittest
import numpy as np
from src.movement_direction import estimate_movement


class TestMovementDirection(unittest.TestCase):
    def test_cardinal_and_irregular_linear_motion(self):
        rng = np.random.default_rng(13)
        t = 4000 + np.cumsum(rng.uniform(.009, .011, 1000))
        for vx, vy, angle in [(0, 10, 0), (10, 0, 90), (0, -10, 180), (-10, 0, 270)]:
            r = estimate_movement(np.c_[t, 100 + vx * t, 500 + vy * t])
            self.assertGreater(r['valid'].sum(), 950)
            np.testing.assert_allclose(r['speed'][r['valid']], 10, atol=1e-8)
            np.testing.assert_allclose(r['direction'][r['valid']], angle, atol=1e-8)

    def test_stationary_is_undefined_not_north(self):
        t = np.arange(1000) * .01
        r = estimate_movement(np.c_[t, np.ones_like(t), np.ones_like(t)])
        self.assertFalse(r['valid'].any())
        self.assertTrue(np.isnan(r['direction']).all())

    def test_gap_and_missing_positions_are_not_bridged(self):
        t = np.arange(1000) * .01
        t[500:] += 1
        x = 10 * t
        x[500:] += 1000  # This discontinuity must not contaminate neighboring slopes.
        p = np.c_[t, x, np.zeros_like(t)]
        p[250, 1] = np.nan
        r = estimate_movement(p)
        self.assertTrue(np.isnan(r['speed'][490:510]).all())
        self.assertTrue(np.isnan(r['speed'][240:261]).all())
        np.testing.assert_allclose(r['speed'][r['valid']], 10, atol=1e-9)

    def test_centered_timestamp_on_acceleration(self):
        t = np.arange(1000) * .01
        r = estimate_movement(np.c_[t, .5 * t * t, 2 * t])
        good = r['valid']
        np.testing.assert_allclose(r['velocity'][good, 0], t[good], atol=1e-10)
        np.testing.assert_allclose(r['velocity'][good, 1], 2, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
