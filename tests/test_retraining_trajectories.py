import importlib.util
from pathlib import Path

import numpy as np
import pytest

p = Path(__file__).resolve().parents[1] / "scripts/retraining/trajectories.py"
spec = importlib.util.spec_from_file_location("trajectories", p)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def test_motion_and_bounds():
    for seed in range(100):
        state, _ = t.make_path(seed)
        t.validate(state, 0.1, 1.5)
        assert (state[:, :2] >= [120.75, 197.75]).all()
        assert (state[:, :2] <= [136.25, 213.25]).all()
        omega = ((np.diff(state[:, 2]) + 180) % 360 - 180) / 0.1
        assert np.max(np.abs(np.diff(omega))) / 0.1 <= np.degrees(2.5) + 1e-8
    assert np.array_equal(t.make_path(123)[0], t.make_path(123)[0])


def test_rejects_teleport_and_wrong_heading():
    state, _ = t.make_path(1)
    state[10, 0] += 2
    with pytest.raises(ValueError):
        t.validate(state, 0.1, 1.5)
    state, _ = t.make_path(1)
    state[5, 2] += 90
    with pytest.raises(ValueError):
        t.validate(state, 0.1, 1.5)
