"""Fresh causal predictive coder with audited episode splits and durable checkpoints."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scripts.retraining.recipe_model import make_model
from scripts.retraining.trajectories import validate
from src.ann.predictive_coder import PredictiveCoder


def load_data(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Incomplete collection")
    data = {s: [] for s in ("train", "validation", "test")}
    seen = set()
    pose_seen = set()
    for ep in manifest["episodes"]:
        p = root / ep.get("file", ep["id"] + ".npz")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if ep.get("sha256") != digest:
            raise ValueError("Dataset checksum mismatch")
        with np.load(p, allow_pickle=False) as f:
            rgb = f["rgb"].copy()
            state = f["state"].copy()
            if (
                rgb.dtype != np.uint8
                or rgb.ndim != 4
                or rgb.shape[1:] != (64, 64, 3)
                or len(rgb) < 10
            ):
                raise ValueError("Invalid episode")
            if len(state) != len(rgb) or not np.isfinite(state).all():
                raise ValueError("Invalid state")
            if "speed_blocks_per_second" in manifest:
                validate(
                    state,
                    manifest["dt_seconds"],
                    manifest["speed_blocks_per_second"],
                    position_tolerance=3e-4,
                    heading_tolerance=0.1,
                )
                if not np.allclose(
                    f["simulation_time"], np.arange(len(rgb)) * manifest["dt_seconds"]
                ):
                    raise ValueError("Invalid simulation time")
        h = hashlib.sha256(rgb.tobytes()).hexdigest()
        ph = hashlib.sha256(state.tobytes()).hexdigest()
        if h in seen or ph in pose_seen:
            raise ValueError("Duplicate episodes")
        seen.add(h)
        pose_seen.add(ph)
        data[ep["split"]].append(rgb)
    if any(not x for x in data.values()):
        raise ValueError("Missing split")
    total = sum(x.size // 3 for x in data["train"])
    sums = sum(x.sum(axis=(0, 1, 2), dtype=np.float64) for x in data["train"])
    squares = sum(
        np.square(x.astype(np.float64)).sum(axis=(0, 1, 2)) for x in data["train"]
    )
    mean = sums / total / 255
    std = np.sqrt(squares / total - (sums / total) ** 2) / 255
    if np.min(std) < 1e-4:
        raise ValueError("Degenerate training data")
    return (
        data,
        mean,
        std,
        hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
    )


def run(
    data_root,
    output,
    steps=12600,
    max_seconds=5400,
    resume=False,
    final=False,
    seed=20261116,
):
    torch.set_num_threads(4)
    if steps < 1 or max_seconds <= 0:
        raise ValueError("Positive update and runtime limits required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    data, measured_mean, measured_std, identity = load_data(data_root)
    mean = np.array([121.6697, 149.3242, 154.9510]) / 255
    std = np.array([40.7521, 47.7267, 103.2739]) / 255
    chunks = [
        (k, i) for k, e in enumerate(data["train"]) for i in range(0, len(e) - 9, 10)
    ]
    batches_per_epoch = (len(chunks) + 63) // 64
    if steps != 200 * batches_per_epoch:
        raise ValueError("Recipe requires exactly 200 epochs")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not resume:
        raise ValueError(
            "Existing run requires explicit resume or a new output directory"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-6
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, epochs=200, steps_per_epoch=batches_per_epoch
    )
    order = None
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)[
        None, None, :, None, None
    ]
    std_t = torch.tensor(std, dtype=torch.float32, device=device)[
        None, None, :, None, None
    ]
    rng = np.random.default_rng(seed)
    start = 0
    best = float("inf")
    history = []
    if resume:
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if ck["seed"] != seed:
            raise ValueError("Resume seed changed")
        if ck["dataset_sha256"] != identity:
            raise ValueError("Resume dataset changed")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        order = ck["epoch_order"]
        if ck["steps_total"] != steps:
            raise ValueError("Schedule changed")
        start = ck["step"]
        best = ck["best_validation"]
        history = ck["history"]
        rng.bit_generator.state = ck["numpy_generator"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])

    def tensor(windows):
        x = (
            torch.from_numpy(np.stack(windows))
            .to(device)
            .float()
            .permute(0, 1, 4, 2, 3)
            / 255
        )
        return (x - mean_t) / std_t

    def evaluate(split, cap=None, preview=False):
        model.eval()
        squared = 0.0
        normalized_squared = 0.0
        copy_error = 0.0
        mean_error = 0.0
        n = 0
        examples = []
        windows = [
            (k, e, i)
            for k, e in enumerate(data[split])
            for i in range(0, len(e) - 9, 9)
        ]
        episode_model = np.zeros(len(data[split]))
        episode_copy = np.zeros(len(data[split]))
        episode_n = np.zeros(len(data[split]), dtype=int)
        if cap:
            windows = [
                windows[i]
                for i in np.linspace(
                    0, len(windows) - 1, min(cap, len(windows)), dtype=int
                )
            ]
        with torch.no_grad():
            for j in range(0, len(windows), 16):
                x = tensor([e[i : i + 10] for k, e, i in windows[j : j + 16]])
                pred = model(x[:, :-1])
                target = x[:, 1:]
                squared += float(((pred - target) * std_t).square().sum())
                normalized_squared += float((pred - target).square().sum())
                copy_error += float(((x[:, :-1] - target) * std_t).square().sum())
                mean_error += float((target * std_t).square().sum())
                n += target.numel()
                if preview:
                    model_errors = (
                        ((pred - target) * std_t)
                        .square()
                        .mean(dim=(1, 2, 3, 4))
                        .cpu()
                        .numpy()
                    )
                    copy_errors = (
                        ((x[:, :-1] - target) * std_t)
                        .square()
                        .mean(dim=(1, 2, 3, 4))
                        .cpu()
                        .numpy()
                    )
                    for (k, e, i), me, ce in zip(
                        windows[j : j + 16], model_errors, copy_errors
                    ):
                        episode_model[k] += me
                        episode_copy[k] += ce
                        episode_n[k] += 1

                if preview and not examples:
                    v = torch.stack((x[0, :9], target[0], pred[0])) * std_t + mean_t
                    v = (
                        (v.clamp(0, 1) * 255)
                        .byte()
                        .permute(0, 1, 3, 4, 2)
                        .cpu()
                        .numpy()
                    )
                    canvas = np.concatenate(
                        [np.concatenate(row, axis=1) for row in v], axis=0
                    )
                    Image.fromarray(canvas).save(out / (split + "_predictions.png"))
                    examples = [True]
        detail = {}
        if preview:
            model_ep = episode_model / episode_n
            copy_ep = episode_copy / episode_n
            difference = model_ep - copy_ep
            bootstrap = (
                np.random.default_rng(1701)
                .choice(difference, (10000, len(difference)), replace=True)
                .mean(axis=1)
            )
            detail = {
                "episode_model_rgb_mse": model_ep.tolist(),
                "episode_persistence_rgb_mse": copy_ep.tolist(),
                "model_minus_persistence_mse_episode_bootstrap_95ci": np.quantile(
                    bootstrap, [0.025, 0.975]
                ).tolist(),
            }
        return {
            **detail,
            "rgb_mse": squared / n,
            "normalized_mse": normalized_squared / n,
            "persistence_rgb_mse": copy_error / n,
            "training_mean_rgb_mse": mean_error / n,
            "windows": len(windows),
            "pixels_channels": n,
        }

    def check_causality():
        errors = {}
        x = tensor([data["train"][0][:10]])
        for training in [True, False]:
            model.train(training)
            with torch.no_grad():
                y = model(x)
                xx = x.clone()
                xx[:, 5:] += 3
                yy = model(xx)
                error = float((y[:, :5] - yy[:, :5]).abs().max())
                if not np.isfinite(error) or error > 1e-5:
                    raise ValueError("Future leakage: " + str(error))
                errors["training_mode" if training else "evaluation_mode"] = error
        return errors

    initial_causality = check_causality()
    initial = evaluate("validation")
    print("initial", initial, flush=True)
    started = time.monotonic()
    step = start

    def save(is_best=False):
        ck = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch_order": order,
            "steps_total": steps,
            "recipe": "authors SGD/OneCycle 200 epochs; causal normalization",
            "recipe_model_sha256": hashlib.sha256(
                Path(__import__("inspect").getfile(make_model)).read_bytes()
            ).hexdigest(),
            "step": step,
            "best_validation": best,
            "history": history,
            "dataset_sha256": identity,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "architecture": {
                "causal_normalization": True,
                "layers": [2, 2, 2, 2],
                "num_heads": 2,
                "seq_len": 10,
            },
            "numpy_generator": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "seed": seed,
            "torch_version": torch.__version__,
            "training_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "model_code_sha256": hashlib.sha256(
                Path(__import__("inspect").getfile(PredictiveCoder)).read_bytes()
            ).hexdigest(),
        }
        torch.save(ck, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")
        if is_best:
            torch.save(ck, out / "best.tmp")
            (out / "best.tmp").replace(out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    while step < steps and time.monotonic() - started < max_seconds:
        model.train()
        batch_index = step % batches_per_epoch
        if batch_index == 0:
            order = rng.permutation(len(chunks)).tolist()
        windows = [
            data["train"][chunks[j][0]][chunks[j][1] : chunks[j][1] + 10]
            for j in order[batch_index * 64 : (batch_index + 1) * 64]
        ]
        x = tensor(windows)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)[:, :-1]
        loss = torch.nn.functional.mse_loss(prediction, x[:, 1:])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        gn = torch.sqrt(
            sum(
                p.grad.detach().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        if not torch.isfinite(gn):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        scheduler.step()
        step += 1
        if step % batches_per_epoch == 0 or step == steps:
            val = evaluate("validation")
            improved = val["normalized_mse"] < best
            best = min(best, val["normalized_mse"])
            row = {
                "step": step,
                "epoch": step // batches_per_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "momentum": optimizer.param_groups[0]["momentum"],
                "training_normalized_mse": float(loss.detach()),
                "gradient_norm": float(gn),
                "validation": val,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            save(improved)
            print(json.dumps(row), flush=True)
    save()
    if step != steps:
        raise RuntimeError(
            "Training budget ended; resume required before final evaluation"
        )
    if not (out / "best.pt").exists():
        raise RuntimeError("No validation checkpoint completed")
    best_ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model"])
    final_causality = check_causality()
    torch.save(
        {
            "model": best_ck["model"],
            "architecture": best_ck["architecture"],
            "mean": mean.tolist(),
            "std": std.tolist(),
            "step": best_ck["step"],
            "dataset_sha256": identity,
            "training_code_sha256": best_ck["training_code_sha256"],
            "model_code_sha256": best_ck["model_code_sha256"],
            "recipe_model_sha256": best_ck["recipe_model_sha256"],
        },
        out / "inference.pt",
    )

    result = {
        "steps": step,
        "best_step": best_ck["step"],
        "elapsed_seconds": time.monotonic() - started,
        "dataset_sha256": identity,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "episodes": {k: len(v) for k, v in data.items()},
        "initial_validation": initial,
        "validation": evaluate("validation", preview=True),
        "device": device,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "fresh_initialization": True,
        "causality_max_abs_error_initial": initial_causality,
        "causality_max_abs_error_selected_checkpoint": final_causality,
        "test_evaluated": final,
        "architecture": "causal per-frame GroupNorm + per-frame latent LayerNorm",
        "objective": "normalized next-frame pixel MSE; SGD momentum .9; OneCycle max_lr .1; batch64; 200 shuffled epochs; nonoverlapping 10-frame chunks",
        "epochs": step / batches_per_epoch,
        "training_chunks": len(chunks),
        "measured_training_mean": measured_mean.tolist(),
        "measured_training_std": measured_std.tolist(),
        "validation_selection": "full validation RGB-channel-normalized MSE, corrected averaging",
        "recipe": {
            "num_heads": 2,
            "epochs": 200,
            "batch_size": 64,
            "max_lr": 0.1,
            "weight_decay": 5e-6,
            "nesterov": False,
            "gradient_clipping": False,
            "rgb_statistics": "authors fixed values",
            "dataset": "fresh smooth-v1",
        },
    }
    if final:
        result["test"] = evaluate("test", preview=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-seconds", type=int, default=600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--seed", type=int, default=20261116)
    a = p.parse_args()
    run(a.data, a.output, a.steps, a.max_seconds, a.resume, a.final, a.seed)
