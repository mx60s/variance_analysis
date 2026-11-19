import hashlib
import json

import numpy as np
import pytest

from scripts.retraining.train import load_data
from scripts.retraining.trajectories import make_path


def dataset(root):
    episodes = []
    for i, split in enumerate(["train", "validation", "test"]):
        poses, _ = make_path(i, frames=10)
        rgb = np.random.default_rng(i).integers(0, 256, (10, 64, 64, 3), dtype=np.uint8)
        p = root / f"{i}.npz"
        np.savez(p, rgb=rgb, state=poses, simulation_time=np.arange(10) * 0.1)
        episodes.append(
            {
                "id": str(i),
                "split": split,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "complete": True,
        "episodes": episodes,
        "dt_seconds": 0.1,
        "speed_blocks_per_second": 1.5,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_integrity_and_training_only_statistics(tmp_path):
    dataset(tmp_path)
    data, mean, std, _ = load_data(tmp_path)
    reference = data["train"][0] / 255.0
    np.testing.assert_allclose(mean, reference.mean(axis=(0, 1, 2)))
    np.testing.assert_allclose(std, reference.std(axis=(0, 1, 2)))
    with (tmp_path / "1.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_data(tmp_path)


def test_rejects_cross_split_duplicates(tmp_path):
    manifest = dataset(tmp_path)
    (tmp_path / "1.npz").write_bytes((tmp_path / "0.npz").read_bytes())
    manifest["episodes"][1]["sha256"] = manifest["episodes"][0]["sha256"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        load_data(tmp_path)
