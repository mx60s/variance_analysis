"""Bounded training/throughput pilot on PNG trajectory episodes (not panoramas).

Each episode directory must contain 0.png ... 9.png (or longer). Train and
validation directories must be separate episodes. Uses a new causal-normalization
architecture; never loads or overwrites the historical checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.ann.predictive_coder import PredictiveCoder


class Episodes(Dataset):
    def __init__(self, root, seq_len=10):
        self.episodes = []
        self.seq_len = seq_len
        for directory in sorted(Path(root).iterdir()):
            if not directory.is_dir():
                continue
            frames = sorted(
                (p for p in directory.glob("*.png") if p.stem.isdigit()),
                key=lambda p: int(p.stem),
            )
            if len(frames) >= seq_len:
                self.episodes.append(frames)
        if not self.episodes:
            raise ValueError(f"No complete trajectory episodes in {root}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        # Fixed first window makes pilot throughput reproducible. Full training
        # needs a declared window-sampling policy, rather than silent random crops.
        frames = self.episodes[i][: self.seq_len]
        x = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in frames])
        if x.shape[1:] != (64, 64, 3):
            raise ValueError("Expected 64x64 RGB frames")
        x = torch.from_numpy(x.copy()).float().permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([121.6697, 149.3242, 154.9510]).view(1, 3, 1, 1) / 255
        std = torch.tensor([40.7521, 47.7267, 103.2739]).view(1, 3, 1, 1) / 255
        x = (x - mean) / std
        return x, x[1:]


def run(
    train_root,
    val_root,
    output,
    device="cpu",
    steps=100,
    batch_size=16,
    max_seconds=600,
    seed=413,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if steps < 1 or max_seconds <= 0:
        raise ValueError("Positive steps and runtime required")
    tr, va = Episodes(train_root), Episodes(val_root)
    trdirs = {str(f[0].parent.resolve()) for f in tr.episodes}
    vadirs = {str(f[0].parent.resolve()) for f in va.episodes}
    if trdirs & vadirs:
        raise ValueError("Training and validation episodes overlap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = PredictiveCoder(3, 3, [2, 2], 10, causal_normalization=True).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-6
    )
    loader = DataLoader(
        tr,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    timings = []
    losses = []
    t = time.monotonic()
    it = iter(loader)
    for step in range(steps):
        if time.monotonic() - t >= max_seconds:
            break
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        t0 = time.monotonic()
        model.train()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x)[:, :-1], y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        timings.append(time.monotonic() - t0)
        losses.append(float(loss.detach().cpu()))
    model.eval()
    vals = []
    with torch.no_grad():
        for i, (x, y) in enumerate(DataLoader(va, batch_size=batch_size)):
            vals.append(
                float(
                    torch.nn.functional.mse_loss(
                        model(x.to(device))[:, :-1], y.to(device)
                    ).cpu()
                )
            )
            if i >= 2:
                break
    result = dict(
        architecture="causal normalization: per-frame LayerNorm and GroupNorm; requires new training",
        pilot_only=True,
        seed=seed,
        steps_completed=len(losses),
        train_episodes=len(tr),
        validation_episodes=len(va),
        batch_size=batch_size,
        median_step_seconds=float(np.median(timings[1:] or timings)),
        train_losses=losses,
        validation_batch_losses=vals,
        elapsed_seconds=time.monotonic() - t,
        parameters=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__,
        training_episode_paths=sorted(trdirs),
        validation_episode_paths=sorted(vadirs),
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": result,
        },
        output / "pilot.pt",
    )
    (output / "pilot.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_paths")}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seconds", type=int, default=600)
    a = p.parse_args()
    run(a.train, a.validation, a.output, a.device, a.steps, a.batch_size, a.max_seconds)


if __name__ == "__main__":
    main()
