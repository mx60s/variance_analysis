"""Frozen-checkpoint controls: real history, current frame only, and fixed blur baselines.

No training or model selection occurs here. All controls use the same held-out
windows as the primary final evaluation. Blur sizes are fixed in advance.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.ann.predictive_coder import PredictiveCoder


def run(data_root, checkpoint, output):
    torch.set_num_threads(4)
    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        ck["dataset_sha256"]
        != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Checkpoint/data mismatch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PredictiveCoder(3, 3, **ck["architecture"]).to(device).eval()
    model.load_state_dict(ck["model"])
    mean = torch.tensor(ck["mean"], dtype=torch.float32, device=device)[
        None, None, :, None, None
    ]
    std = torch.tensor(ck["std"], dtype=torch.float32, device=device)[
        None, None, :, None, None
    ]
    kernels = {}
    for k, sigma in [(3, 1.0), (7, 2.0)]:
        coords = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2
        g = torch.exp(-coords.square() / (2 * sigma * sigma))
        g /= g.sum()
        kernels[k] = (g[:, None] * g[None, :])[None, None].expand(3, 1, k, k)
    records = []
    with torch.no_grad():
        for ep in manifest["episodes"]:
            if ep["split"] != "test":
                continue
            p = root / ep.get("file", ep["id"] + ".npz")
            if hashlib.sha256(p.read_bytes()).hexdigest() != ep["sha256"]:
                raise ValueError("Data checksum mismatch")
            with np.load(p) as f:
                rgb = f["rgb"].copy()
            raw = (
                torch.from_numpy(
                    np.stack([rgb[i : i + 10] for i in range(0, len(rgb) - 9, 9)])
                )
                .to(device)
                .float()
                .permute(0, 1, 4, 2, 3)
                / 255
            )
            x = (raw - mean) / std
            current = x[:, :-1]
            target = raw[:, 1:]
            real = model(current) * std + mean
            batch, length = current.shape[:2]
            static = (
                model(current.reshape(batch * length, 1, 3, 64, 64)).reshape(
                    batch, length, 3, 64, 64
                )
                * std
                + mean
            )
            pred = {
                "real_history": real,
                "current_frame_only": static,
                "copy_current_frame": raw[:, :-1],
            }
            for k, kernel in kernels.items():
                radius = k // 2
                inp = F.pad(
                    raw[:, :-1].reshape(-1, 3, 64, 64),
                    (radius, radius, radius, radius),
                    mode="reflect",
                )
                pred["blur_" + str(k)] = F.conv2d(inp, kernel, groups=3).reshape(
                    batch, length, 3, 64, 64
                )
            rec = {"episode_id": ep["id"]}
            for key, value in pred.items():
                mse = float((value - target).square().mean())
                if not np.isfinite(mse):
                    raise ValueError("Nonfinite control metric")
                rec[key] = mse
            records.append(rec)
    keys = [
        "real_history",
        "current_frame_only",
        "copy_current_frame",
        "blur_3",
        "blur_7",
    ]
    means = {k: float(np.mean([e[k] for e in records])) for k in keys}
    ci = {}
    for control in keys[1:]:
        d = np.array([e["real_history"] - e[control] for e in records])
        boot = (
            np.random.default_rng(1701).choice(d, (10000, len(d)), replace=True).mean(1)
        )
        ci[control] = np.quantile(boot, [0.025, 0.975]).tolist()
    result = {
        "dataset_sha256": ck["dataset_sha256"],
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_step": ck["step"],
        "episodes": records,
        "mean_rgb_mse": means,
        "real_history_minus_control_episode_bootstrap_95ci": ci,
        "control_definition": "Current-frame-only runs the same frozen network on length-one inputs. Fixed Gaussian current-frame blur: 3x3 sigma1 and7x7 sigma2, reflect padding. These are evaluation-time controls, not separately trained models.",
        "evaluation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return result
