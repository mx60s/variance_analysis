"""Audit saved RGB/pose trajectories and package their exact training inputs."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from scripts.retraining.trajectories import validate


def audit(root, archive=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete collection")
    seen = set()
    seeds = set()
    states = []
    splits = []
    ids = []
    records = []
    for ep in manifest["episodes"]:
        p = root / ep.get("file", ep["id"] + ".npz")
        if hashlib.sha256(p.read_bytes()).hexdigest() != ep["sha256"]:
            raise ValueError("File integrity failure")
        if ep["seed"] in seeds:
            raise ValueError("Duplicate random seed")
        seeds.add(ep["seed"])
        with np.load(p, allow_pickle=False) as f:
            rgb = f["rgb"]
            s = f["state"]
            target = f["requested_state"]
            t = f["simulation_time"]
            if rgb.shape != (100, 64, 64, 3) or rgb.dtype != np.uint8:
                raise ValueError("Invalid images")
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError("Repeated image episode")
            seen.add(digest)
            if np.std(rgb) < 1 or np.mean(rgb[1:] != rgb[:-1]) < 0.001:
                raise ValueError("Blank or frozen episode")
            if np.max(np.abs(s[:, :2] - target[:, :2])) >= 1e-4:
                raise ValueError("Rendered position mismatch")
            if np.max(np.abs((s[:, 2] - target[:, 2] + 180) % 360 - 180)) >= 0.002:
                raise ValueError("Rendered heading mismatch")
            if not np.allclose(t, np.arange(len(s)) * 0.1):
                raise ValueError("Invalid simulated times")
            if not (
                np.diff(f["render_timestamp"].astype("datetime64[us]"))
                > np.timedelta64(0, "us")
            ).all():
                raise ValueError("Non-increasing render timestamps")
            if not (np.diff(f["wall_time"]) > 0).all():
                raise ValueError("Non-increasing capture times")
            record = validate(s, 0.1, 1.5, 3e-4, 0.1)
            record["wall_seconds"] = float(f["wall_time"][-1] - f["wall_time"][0])
            records.append(record)
            states.append(s)
            splits.append(ep["split"])
            ids.append(ep["id"])
    allstate = np.stack(states)

    def coverage(s):
        h = np.histogram2d(
            s[:, :, 0].ravel(),
            s[:, :, 1].ravel(),
            bins=17,
            range=[[120, 137], [197, 214]],
        )[0]
        return {
            "visited_bins": int((h > 0).sum()),
            "total_bins": 289,
            "histogram": h.astype(int).tolist(),
        }

    result = {
        "passed": True,
        "episodes": len(states),
        "frames": sum(len(s) for s in states),
        "rgb_shape": [64, 64, 3],
        "speed_target_blocks_per_simulated_second": 1.5,
        "dt_simulated_seconds": 0.1,
        "speed_min": min(r["speed_min"] for r in records),
        "speed_max": max(r["speed_max"] for r in records),
        "max_heading_error_deg": max(r["max_heading_error_deg"] for r in records),
        "max_yaw_step_deg": max(r["max_yaw_step_deg"] for r in records),
        "median_wall_seconds_per_100_frames": float(
            np.median([r["wall_seconds"] for r in records])
        ),
        "splits": {
            split: {
                "episodes": splits.count(split),
                **coverage(allstate[np.array(splits) == split]),
            }
            for split in ["train", "validation", "test"]
        },
        "all_coverage": coverage(allstate),
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "interpretation": "Kinematic camera paths rendered by Minecraft1.11.2. Constant speed refers to fixed simulation timestamps; wall-clock rendering is asynchronous. Episode resets are excluded from training windows.",
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        root / "poses.npz",
        state=allstate,
        split=np.array(splits),
        episode_id=np.array(ids),
    )
    if archive:
        with tarfile.open(archive, "w") as tar:
            for ep in manifest["episodes"]:
                name = ep.get("file", ep["id"] + ".npz")
                tar.add(root / name, arcname="smooth-v1/" + name)
            for name in ["manifest.json", "audit.json", "poses.npz"] + [
                s["file"] for s in manifest.get("shards", [])
            ]:
                tar.add(root / name, arcname="smooth-v1/" + name)
    return result
