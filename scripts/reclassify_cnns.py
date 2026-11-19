"""Fresh multi-layer CNN activations and classification on the existing panoramas.

Runs every CNN layer used by the manuscript, records sampled feature IDs, and
keeps outputs separate. The untrained control uses an explicit new seed (413).
"""

import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from src.ann.get_activations import _make_tensor_preprocess
from reclassify import classify, sha256


def run(model_name, layers, output, device="mps"):
    torch.manual_seed(413)
    if model_name == "resnet50":
        core = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        weights = "ResNet50_Weights.IMAGENET1K_V2"
    elif model_name == "resnet50_untrained":
        core = models.resnet50(weights=None)
        weights = "random initialization; seed 413"
    else:
        core = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        weights = "VGG16_Weights.IMAGENET1K_V1"
    model = create_feature_extractor(core, {l: l for l in layers}).to(device).eval()
    preprocess = _make_tensor_preprocess()
    files = sorted((ROOT / "data/minecraft/panorama-90yaw").glob("loc_*.npz"))
    if not files:
        raise FileNotFoundError("No panoramas")
    dirs = {l: output / f"{model_name}_{l}" for l in layers}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        if (d / "activations.npz").exists():
            raise FileExistsError(d)
    buffers = {}
    ids = {}
    locations = []
    inputs = []
    t = time.time()
    with torch.inference_mode():
        for i, p in enumerate(files):
            with np.load(p) as z:
                rgb = z["rgb"]
            locations.append([float(v) for v in p.stem.removeprefix("loc_").split("_")])
            inputs.append({"file": p.name, "sha256": sha256(p)})
            for start in range(0, len(rgb), 16):
                batch = rgb[start : start + 16]
                outs = model(preprocess(batch).to(device))
                for layer, act in outs.items():
                    act = act.flatten(1)
                    if layer not in ids:
                        ids[layer] = (
                            torch.randperm(
                                act.shape[1],
                                generator=torch.Generator().manual_seed(413),
                            )[:8000]
                            if act.shape[1] > 8000
                            else torch.arange(act.shape[1])
                        )
                        buffers[layer] = np.lib.format.open_memmap(
                            dirs[layer] / "activations.npy",
                            mode="w+",
                            dtype="float32",
                            shape=(len(files), len(rgb), len(ids[layer])),
                        )
                    buffers[layer][i, start : start + len(batch)] = (
                        act.index_select(1, ids[layer].to(device)).cpu().numpy()
                    )
            if i % 30 == 0:
                print(
                    model_name, i + 1, len(files), round(time.time() - t, 1), flush=True
                )
    for layer in layers:
        buffers[layer].flush()
        p = dirs[layer] / "activations.npz"
        np.savez_compressed(
            p,
            activations=buffers[layer],
            locations=locations,
            unit_ids=ids[layer].numpy(),
        )
        (dirs[layer] / "provenance.json").write_text(
            json.dumps(
                {
                    "weights": weights,
                    "seed": 413,
                    "feature_ids": ids[layer].tolist(),
                    "panoramas": inputs,
                    "torch": torch.__version__,
                    "device": device,
                    "evaluation": "fixed-pitch static panoramas, nominal heading bins; see predictive-coder input QA for yaw deviations",
                },
                indent=2,
            )
        )
    del model, core, buffers
    if device == "mps":
        torch.mps.empty_cache()
    for layer in layers:
        p = dirs[layer] / "activations.npz"
        classify(p, model_name, layer, dirs[layer], compare_ref="e44fee3")
        classify(
            p,
            model_name,
            layer,
            dirs[layer] / "tanni_iterative",
            field_method="tanni_iterative",
        )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--device", default="cpu")
    args = a.parse_args()
    for name, layers in [
        ("resnet50", ["layer1", "layer3", "layer4", "avgpool"]),
        ("resnet50_untrained", ["avgpool"]),
        ("vgg16", ["features.9"]),
    ]:
        run(name, layers, args.output, args.device)
