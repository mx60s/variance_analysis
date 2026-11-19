import unittest
import numpy as np
from src.aligned_analysis import (
    coalesce_tracking_timestamps,
    align_tracking,
    lag_without_wrap,
    grouped_maps,
    poisson_argmax,
)
from src.decoding import xy_bayesian_decoding


class AlignmentTests(unittest.TestCase):
    def test_alignment_rejects_gaps_and_outside_recording(self):
        t = np.array([0, 0.01, 0.02, 1.0, 1.01])
        p = np.c_[t, t, t, t]
        q, idx, v = align_tracking(p, np.array([-0.02, 0, 0.02, 0.04, 0.5, 1.0, 1.02]))
        self.assertFalse(v[0])
        self.assertFalse(v[3])
        self.assertTrue(v[1])
        self.assertTrue(np.isnan(q[~v, 1:]).all())

    def test_duplicate_heading_mean_is_circular(self):
        p, n = coalesce_tracking_timestamps(
            np.array([[0, 1, 2, 359], [0, 3, 4, 1], [0.01, 5, 6, 90]])
        )
        self.assertEqual(n, 1)
        np.testing.assert_allclose(p[0, 1:3], [2, 3])
        self.assertAlmostEqual(p[0, 3] % 360, 0)

    def test_lag_does_not_wrap(self):
        np.testing.assert_allclose(
            lag_without_wrap(np.arange(4.0), 2), [2, 3, np.nan, np.nan], equal_nan=True
        )

    def test_fold_samples_not_duplicated(self):
        rates = np.arange(12.0)[:, None]
        row = np.repeat([0, 1], 6)
        col = np.zeros(12, dtype=int)
        a, b, o = grouped_maps(rates, row, col, 2, 1, np.ones(12, bool), floor=2)
        self.assertEqual(o.sum(), 12)
        np.testing.assert_allclose((a + b)[:, 0, 0] / 2, [2.5, 8.5])

    def test_fast_decoder_matches_reference(self):
        rng = np.random.default_rng(8)
        r = rng.uniform(0.1, 5, (3, 4, 6))
        r[0, 0] = np.nan
        r[1, 1, 2] = np.nan
        k = rng.poisson(0.8, (41, 6))
        edgesx = np.arange(5)
        edgesy = np.arange(4)
        expected, _ = xy_bayesian_decoding(
            r, k, 0.1, np.ones(41), edgesx, edgesy, nan_to_zero=False
        )
        idx = poisson_argmax(r.reshape(12, 6), k, 0.1)
        np.testing.assert_allclose(expected, np.c_[idx % 4 + 0.5, idx // 4 + 0.5])


if __name__ == "__main__":
    unittest.main()
