"""Stage explicit inference inputs and reference activations, never credentials."""

import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {
    "predictive_coder": ["conv1", "down1", "pre_pred", "post_pred"],
    "resnet50": ["layer1", "layer3", "layer4", "avgpool"],
    "resnet50_untrained": ["avgpool"],
    "vgg16": ["features.9"],
}


def prepare(stage):
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        ROOT / "src",
        stage / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (stage / "scripts").mkdir(exist_ok=True)
    for name in ["regenerate_model_ev.py", "regenerate_ev.py"]:
        shutil.copy2(ROOT / "scripts" / name, stage / "scripts" / name)
    # The inference runner needs only this utility, not classification's optional
    # plotting dependencies. Keep this stage self-contained and explicit.
    (stage / "scripts/reclassify.py").write_text(
        'import hashlib\ndef sha256(path):\n with open(path,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()\n'
    )
    first = sorted((ROOT / "data/minecraft/panorama-90yaw").glob("loc_*.npz"))[0]
    with np.load(first) as z:
        rgb = z["rgb"][:10]
    reference = stage / "data/crossdevice"
    reference.mkdir(parents=True, exist_ok=True)
    for model, layers in LAYERS.items():
        ref = {"rgb": rgb}
        for layer in layers:
            relative = (
                Path("results/review_20261116")
                / ("predictive_coder" if model == "predictive_coder" else "cnns")
                / (layer if model == "predictive_coder" else f"{model}_{layer}")
            )
            src, dest = ROOT / relative, stage / relative
            dest.mkdir(parents=True, exist_ok=True)
            for name in ["provenance.json", "classifications.json"]:
                shutil.copy2(src / name, dest / name)
            ref[layer] = np.load(src / "activations.npy", mmap_mode="r")[0, :10].copy()
        np.savez_compressed(reference / f"{model}.npz", **ref)
    # Exact weights, rather than assuming seeded initialization matches between
    # ARM and x86. These reproduce the seed used by the clean classification run.
    torch.manual_seed(413)
    torch.save(
        models.resnet50(weights=None).state_dict(), reference / "untrained_state.pt"
    )
    print("Prepared minimal stage", stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    prepare(parser.parse_args().stage)
