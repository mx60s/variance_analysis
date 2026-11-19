import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.ann.score_utils import (
    _compute_field_stability,
    _compute_single_heatmap_fields_info_new,
    _compute_single_heatmap_directional_scores_new,
)
from src.ann.classify_neurons import classify_neurons, score_neurons
from src.ann.fig import plot_cell_type_pie


class ClassificationRegression(unittest.TestCase):
    def test_field_stability_ignores_outside_bins(self):
        # The field is anti-correlated; a strongly correlated surrounding map
        # must not rescue it, regardless of the surrounding amplitudes.
        field = np.zeros((10, 10))
        field[:3, :3] = 1
        a = np.arange(100, dtype=float).reshape(10, 10) + 100
        b = a.copy()
        b[:3, :3] = a[:3, :3][::-1, ::-1]
        r = _compute_field_stability(field, a, b, min_included_value=0, min_bins=9)
        self.assertAlmostEqual(r, -1)

    def test_field_size_and_stats_refer_only_to_field(self):
        x, y = np.mgrid[:17, :17]
        m = np.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 9)
        h = np.repeat(m.reshape(-1, 1), 90, axis=1)
        n, area, peak, mean, var = _compute_single_heatmap_fields_info_new(
            h, 16, 16, 0, 0, 15, 144
        )
        self.assertEqual(n[0], 1)
        self.assertLess(area[0], 144)
        self.assertGreater(mean[0], 0)
        self.assertLessEqual(mean[0], peak[0])
        n2, *_ = _compute_single_heatmap_fields_info_new(h, 16, 16, 0, 0, 200, 250)
        self.assertEqual(n2[0], 0)

    def test_constants_have_no_fields_or_direction(self):
        for v in [0.0, 1.0, -1.0]:
            h = np.full((289, 90), v)
            self.assertEqual(
                _compute_single_heatmap_fields_info_new(h, 16, 16, 0, 0, 15, 144)[0][0],
                0,
            )
            self.assertEqual(_compute_single_heatmap_directional_scores_new(h)[0], 0.0)

    def test_partition_covers_all_eight_active_combinations_and_dead(self):
        chart = np.zeros((9, 12), dtype=object)
        for i in range(8):
            chart[i, 0] = 1
            chart[i, 1] = int(bool(i & 1))
            chart[i, 2] = []
            chart[i, 3] = []
            chart[i, 9] = float(bool(i & 2))
            chart[i, 10] = float(bool(i & 4))
        groups = classify_neurons(chart)
        self.assertEqual(sorted(n for v in groups.values() for n in v), list(range(9)))
        self.assertTrue(all(len(v) == 1 for v in groups.values()))

    def test_pie_preserves_rare_category_denominator(self):
        keys = [
            "exclusive_place_cells_indices",
            "exclusive_border_cells_indices",
            "exclusive_direction_cells_indices",
            "place_and_border_not_direction_cells_indices",
            "place_and_direction_not_border_cells_indices",
            "border_and_direction_not_place_cells_indices",
            "place_border_direction_cells_indices",
            "active_no_type_indices",
            "dead_neurons_indices",
        ]
        groups = {k: [] for k in keys}
        groups[keys[0]] = list(range(199))
        groups[keys[1]] = [199]
        fig = plot_cell_type_pie(groups)
        wedges = fig.axes[0].patches
        self.assertEqual(len(wedges), 2)
        self.assertAlmostEqual(
            wedges[0].theta2 - wedges[0].theta1, 360 * 199 / 200, places=4
        )
        self.assertIn("Rare types", [t.get_text() for t in fig.axes[0].texts])
        plt.close(fig)

    def test_rectangular_grid_and_explicit_output(self):
        with TemporaryDirectory() as d:
            p = Path(d)
            np.savez(p / "a.npz", activations=np.ones((80, 4, 2)))
            path, _ = score_neurons(
                [p / "a.npz"],
                dict(
                    env_bounds=(0, 7, 0, 9),
                    n_rotations=4,
                    model_name="test",
                    output_layer="layer",
                ),
                base_dir=p,
                n_visits=1,
                use_old_place_method=False,
                output_dir=p / "review",
            )
            self.assertEqual(np.load(path, allow_pickle=True).shape, (2, 12))


if __name__ == "__main__":
    unittest.main()
