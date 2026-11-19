import unittest
import numpy as np
from src.ev_summary import correlation_summary
from src.explained_variance import _corr_matrix


class FastCorrelationTests(unittest.TestCase):
    def test_matches_reference_with_pairwise_missingness_and_constants(self):
        rng = np.random.default_rng(73)
        for missing in [0, 0.1, 0.7]:
            for offset in [0, 1000]:
                a = rng.normal(size=(23, 29)) + offset
                b = rng.normal(size=a.shape) + offset
                a[0] = 3
                b[3] = 4
                a[rng.random(a.shape) < missing] = np.nan
                b[rng.random(b.shape) < missing] = np.nan
                c = _corr_matrix(a, b)
                diag = np.diag(c).copy()
                np.fill_diagonal(c, np.nan)
                expected = (np.nanmean(c), np.nanmean(diag))
                np.testing.assert_allclose(
                    correlation_summary(a, b), expected, atol=1e-10
                )

    def test_tiny_pairwise_variance_and_zero_tails(self):
        rng = np.random.default_rng(15)
        a = rng.normal(size=(35, 41))
        b = rng.normal(size=a.shape)
        a[:15, :20] = 1e-7 * rng.normal(size=(15, 20))
        b[10:30, :25] = 0
        a[rng.random(a.shape) < 0.2] = np.nan
        b[rng.random(b.shape) < 0.2] = np.nan
        c = _corr_matrix(a, b)
        diag = np.diag(c).copy()
        np.fill_diagonal(c, np.nan)
        np.testing.assert_allclose(
            correlation_summary(a, b), (np.nanmean(c), np.nanmean(diag)), atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
