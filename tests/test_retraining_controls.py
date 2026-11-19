import hashlib
import json

import numpy as np
import torch

from scripts.retraining.evaluate_controls import run
from src.ann.predictive_coder import PredictiveCoder


def test_current_only_agrees_with_identical_history(tmp_path):
    """The history ablation must not change a sequence of identical frames."""
    torch.set_num_threads(2)
    torch.manual_seed(9)
    rgb = np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    np.savez(tmp_path / "episode.npz", rgb=np.repeat(rgb[None], 10, axis=0))
    manifest = {
        "complete": True,
        "episodes": [
            {
                "id": "episode",
                "split": "test",
                "sha256": hashlib.sha256(
                    (tmp_path / "episode.npz").read_bytes()
                ).hexdigest(),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    architecture = {"layers": [2, 2], "seq_len": 10, "causal_normalization": True}
    model = PredictiveCoder(3, 3, **architecture)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": architecture,
            "mean": [0.5] * 3,
            "std": [0.2] * 3,
            "step": 0,
            "dataset_sha256": hashlib.sha256(
                (tmp_path / "manifest.json").read_bytes()
            ).hexdigest(),
        },
        tmp_path / "weights.pt",
    )
    result = run(tmp_path, tmp_path / "weights.pt", tmp_path / "controls.json")
    errors = result["mean_rgb_mse"]
    assert errors["copy_current_frame"] == 0
    np.testing.assert_allclose(
        errors["real_history"], errors["current_frame_only"], rtol=1e-5, atol=1e-7
    )
    assert all(np.isfinite(value) for value in errors.values())
