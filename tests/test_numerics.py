import unittest
import numpy as np
from src.explained_variance import get_splits, masked_zoom, quantify_variance_corr
from src.decoding import xy_bayesian_decoding, make_cv_masks


class NumericalRegression(unittest.TestCase):
    def test_split_internal_nans_odd_counts_and_no_mutation(self):
        a = np.array([1.0, np.nan, 2.0, 5.0]).reshape(1, 1, 4, 1)
        before = a.copy()
        total, f0, f1, o0, o1 = get_splits(a, seed=1)
        np.testing.assert_equal(a, before)
        self.assertAlmostEqual(total.item(), 8 / 3)
        self.assertEqual(o0.item(), 1)
        self.assertEqual(o1.item(), 2)
        self.assertAlmostEqual(
            (f0.item() * o0.item() + f1.item() * o1.item()) / 3, 8 / 3
        )

    def test_split_rejects_ambiguous_occupancy(self):
        a = np.ones((1, 1, 4, 2))
        a[0, 0, 0, 1] = np.nan
        with self.assertRaises(ValueError):
            get_splits(a)

    def test_interpolation_preserves_signed_values(self):
        a = np.array([[-2.0, 1.0], [-1.0, 2.0]])
        np.testing.assert_allclose(masked_zoom(a, (1, 1)), a)

    def test_constant_pearson_slices_are_undefined(self):
        m = np.ones((4, 4))
        off, rel = quantify_variance_corr(m, m, np.ones_like(m))
        self.assertTrue(np.isnan(off))
        self.assertTrue(np.isnan(rel))

    def test_decoder_never_selects_unobserved_states(self):
        rates = np.array([[[np.nan], [10.0]]])
        edges = np.array([0.0, 1.0, 2.0])
        for fill in [True, False]:
            xy, _ = xy_bayesian_decoding(
                rates,
                np.zeros((1, 1)),
                1.0,
                np.ones(1),
                edges,
                np.array([0.0, 1.0]),
                nan_to_zero=fill,
            )
            np.testing.assert_allclose(xy, [[1.5, 0.5]])

    def test_cv_has_no_train_test_overlap_and_tests_each_sample_once(self):
        masks = make_cv_masks(17, 4)
        np.testing.assert_equal(np.sum([~m for m in masks], axis=0), np.ones(17))


class SpatialSmoothing(unittest.TestCase):
    def test_flattened_row_edges_are_not_neighbors(self):
        from src.explained_variance import smooth_location_hd

        a = np.zeros((25, 12))
        a[4, :] = 1
        smoothed = smooth_location_hd(a, (0.5, 0))
        self.assertLess(smoothed[5, 0], 1e-8)
        self.assertGreater(smoothed[9, 0], 0.01)


if __name__ == "__main__":
    unittest.main()
