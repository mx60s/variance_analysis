"""Seeded, constant-displacement camera paths; simulation dt differs from render wall time.

Yaw is Minecraft degrees, atan2(-dx,dz). Paths use a smooth correlated angular
velocity and anticipatory wall steering, with rejection instead of speed changes.
Episodes are independent; no training window may cross an episode reset.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def make_path(seed, frames=100, dt=0.1, speed=1.5):
    if frames < 2 or dt <= 0 or speed <= 0:
        raise ValueError("Invalid trajectory parameters")
    rng = np.random.default_rng(seed)
    lo = np.array([120.75, 197.75])
    hi = np.array([136.25, 213.25])
    for attempt in range(1000):
        pos = rng.uniform(lo, hi)
        theta = rng.uniform(-np.pi, np.pi)
        omega = 0.0
        rows = []
        for i in range(frames):
            direction = np.array([-np.sin(theta), np.cos(theta)])
            rows.append([*pos, np.degrees(theta) % 360])
            next_pos = pos + speed * dt * direction
            if np.any(next_pos < lo) or np.any(next_pos > hi):
                break
            # Smooth steering begins before the edge, preserving forward speed.
            repulsion = 2.0 * (
                np.maximum(0, 3 - (pos - lo)) ** 2 - np.maximum(0, 3 - (hi - pos)) ** 2
            )
            target = direction + repulsion
            target_theta = np.arctan2(-target[0], target[1])
            turn = (target_theta - theta + np.pi) % (2 * np.pi) - np.pi
            proposal = (
                omega
                + (-omega / 1.5 + 2.0 * turn) * dt
                + 0.5 * np.sqrt(dt) * rng.normal()
            )
            omega += np.clip(proposal - omega, -2.5 * dt, 2.5 * dt)
            omega = np.clip(omega, -1.5, 1.5)
            theta += omega * dt
            pos = next_pos
        if len(rows) == frames:
            state = np.array(rows)
            validate(state, dt, speed)
            return state, attempt
    raise RuntimeError("Could not generate bounded trajectory")


def validate(state, dt, speed, position_tolerance=1e-6, heading_tolerance=1e-5):
    if state.ndim != 2 or state.shape[1] != 3 or not np.isfinite(state).all():
        raise ValueError("Invalid poses")
    delta = np.diff(state[:, :2], axis=0)
    actual_speed = np.linalg.norm(delta, axis=1) / dt
    heading = np.degrees(np.arctan2(-delta[:, 0], delta[:, 1]))
    error = np.abs((heading - state[:-1, 2] + 180) % 360 - 180)
    turns = np.abs((np.diff(state[:, 2]) + 180) % 360 - 180)
    if np.max(np.abs(actual_speed - speed)) > position_tolerance / dt:
        raise ValueError("Speed mismatch")
    if error.max() > heading_tolerance:
        raise ValueError("Heading not aligned with movement")
    if turns.max() > np.degrees(1.5 * dt) + heading_tolerance:
        raise ValueError("Abrupt heading change")
    return {
        "speed_min": float(actual_speed.min()),
        "speed_max": float(actual_speed.max()),
        "max_heading_error_deg": float(error.max()),
        "max_yaw_step_deg": float(turns.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261116)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "dt_seconds": 0.1,
        "speed_blocks_per_second": 1.5,
        "trajectory": "kinematic camera; fixed simulated time, variable rendering wall time",
        "episodes": [],
    }
    for i in range(a.episodes):
        seed = a.seed + i
        state, rejections = make_path(seed, a.frames)
        split = "train" if i % 10 < 8 else ("validation" if i % 10 == 8 else "test")
        np.save(out / f"{i:05d}.npy", state)
        manifest["episodes"].append(
            dict(
                id=f"{i:05d}",
                seed=seed,
                split=split,
                rejections=rejections,
                **validate(state, 0.1, 1.5),
            )
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Validated {a.episodes} trajectories")


if __name__ == "__main__":
    main()
