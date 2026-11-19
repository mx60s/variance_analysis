"""Prevent silent mixing of activations when an interrupted extraction resumes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from regenerate_model_ev import validate_resume_identity


class ResumeIdentityTests(unittest.TestCase):
    def test_matching_identity_and_fresh_run(self):
        identity = {"model": "resnet50", "device": "mps", "feature_ids": [4, 2]}
        validate_resume_identity({}, identity)
        validate_resume_identity(
            {"completed_locations": 10, "input_identity": identity}, identity
        )

    def test_changed_and_legacy_identity_rejected(self):
        identity = {"device": "mps", "feature_ids": [4, 2]}
        for previous in [
            None,
            {"device": "cuda", "feature_ids": [4, 2]},
            {"device": "mps", "feature_ids": [2, 4]},
        ]:
            with self.subTest(previous=previous), self.assertRaises(ValueError):
                validate_resume_identity(
                    {"completed_locations": 10, "input_identity": previous}, identity
                )


class ModelRunnerIntegrationTests(unittest.TestCase):
    def test_tiny_model_runs_both_grids_and_skips_completed_output(self):
        import json
        import tempfile
        from collections import OrderedDict
        from unittest.mock import patch
        import numpy as np
        import torch
        import regenerate_model_ev as runner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "results/review_20261116/cnns/resnet50_avgpool"
            clean.mkdir(parents=True)
            (clean / "provenance.json").write_text(
                json.dumps({"feature_ids": [2, 0, 1]})
            )
            (clean / "classifications.json").write_text(
                json.dumps({"exclusive_place_cells_indices": [1]})
            )
            for name in [
                "scripts/regenerate_model_ev.py",
                "scripts/regenerate_ev.py",
                "src/ann/get_activations.py",
                "src/ann/predictive_coder.py",
                "src/ev_summary.py",
                "src/explained_variance.py",
            ]:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("fixture")
            images = root / "data/minecraft/panorama-90yaw-pitch"
            images.mkdir(parents=True)
            rng = np.random.default_rng(4)
            for i in range(289):
                np.savez(
                    images / f"loc_{i}_0.npz",
                    rgb=rng.integers(0, 255, (360, 2, 2, 3), dtype=np.uint8),
                    yaw=np.repeat(np.arange(90) * 4, 4),
                    pitch=np.tile([-3, -1, 1, 3], 90),
                )
            toy = torch.nn.Sequential(
                OrderedDict([("avgpool", torch.nn.AdaptiveAvgPool2d(1))])
            )
            prep = lambda a: torch.from_numpy(a).float().permute(0, 3, 1, 2) / 255
            out = root / "new_results"
            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner.models, "resnet50", return_value=toy) as construct,
                patch.object(runner, "_make_tensor_preprocess", return_value=prep),
            ):
                runner.run("resnet50", ["avgpool"], out, "cpu", batch_size=128)
                runner.run("resnet50", ["avgpool"], out, "cpu", batch_size=128)
                self.assertEqual(construct.call_count, 1)
            dest = out / "resnet50_avgpool"
            for name in ["ev", "ev_native"]:
                with np.load(dest / f"{name}.npz") as z:
                    self.assertEqual(z["ev"].shape, (3, 2))
                    np.testing.assert_array_equal(z["unit_ids"], [2, 0, 1])
                    np.testing.assert_array_equal(z["is_place"], [False, True, False])
            self.assertFalse(list(dest.glob("temporary*")))
            marker = json.loads((out / "resnet50_resume.json").read_text())
            self.assertEqual(marker["completed_locations"], 289)
            self.assertIn("input_identity", marker)
