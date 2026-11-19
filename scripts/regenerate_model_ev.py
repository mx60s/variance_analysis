"""Regenerate pitch-fold EV with recorded feature IDs and bounded-memory scoring.

Native and interpolated grids share the same freshly extracted folds. Completed
locations can be resumed; temporary folds are deleted only after both requested
scores are saved. Historical activation caches and paper figures are untouched.
"""

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from src.ann.get_activations import _make_tensor_preprocess, load_model
from reclassify import sha256
from regenerate_ev import score, summarize

MODEL_LAYERS = {
    "resnet50": ["layer1", "layer3", "layer4", "avgpool"],
    "resnet50_untrained": ["avgpool"],
    "vgg16": ["features.9"],
    "predictive_coder": ["conv1", "down1", "pre_pred", "post_pred"],
}


def predictive_outputs(core, x):
    """Encode identical static frames once, retaining the exact temporal model."""
    e = core.encoder
    a = torch.relu(e.bn1(e.conv1(x[:, 0])))
    b = e.down1(e.maxpool(a))
    f = e.down2(b)
    f = f[:, None].expand(-1, 10, -1, -1, -1).contiguous()
    pre = f[:, -1]
    f = core._normalize_latents(core.attention_1(f, f, f, core.mask)[0] + f)
    f = core._normalize_latents(core.ffn_1(f) + f)
    f = core._normalize_latents(core.attention_2(f, f, f, core.mask)[0] + f)
    f = core._normalize_latents(core.ffn_2(f) + f)
    return {"conv1": a, "down1": b, "pre_pred": pre, "post_pred": f[:, -1]}


def save_json(path, obj):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def save_scores(path, data):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


def validate_resume_identity(resume, identity):
    """Refuse mixed-input, mixed-device or unverified legacy resume markers."""
    if (
        resume.get("completed_locations", 0)
        and resume.get("input_identity") != identity
    ):
        raise ValueError(
            "Resume identity differs or is missing; use a fresh output directory"
        )


def run(name, layers, out, device, batch_size=16, native=True, crossdevice_dir=None):
    pred = name == "predictive_coder"
    base = ROOT / "results/review_20261116"
    clean = {
        l: base
        / ("predictive_coder" if pred else "cnns")
        / (l if pred else f"{name}_{l}")
        for l in layers
    }
    dirs = {l: out / f"{name}_{l}" for l in layers}
    grids = {"ev": (1, 5)}
    if native:
        grids["ev_native"] = (1, 1)
    pending = [
        l for l in layers if not all((dirs[l] / f"{g}.npz").exists() for g in grids)
    ]
    if not pending:
        return
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    feature_ids = {}
    for l in layers:
        provenance = json.loads((clean[l] / "provenance.json").read_text())
        feature_ids[l] = np.asarray(
            provenance.get("unit_ids", provenance.get("feature_ids"))
        )
    files = sorted((ROOT / "data/minecraft/panorama-90yaw-pitch").glob("loc_*.npz"))
    if len(files) != 289:
        raise ValueError(f"Expected 289 panorama locations; found {len(files)}")
    marker = out / f"{name}_resume.json"
    resume = json.loads(marker.read_text()) if marker.exists() else {}
    completed = resume.get("completed_locations", 0)
    if not 0 <= completed <= len(files):
        raise ValueError("Invalid completed-location count")
    identity = {
        "model": name,
        "device": device,
        "torch_version": torch.__version__,
        "feature_ids": {l: feature_ids[l].tolist() for l in layers},
        "panoramas": {p.name: sha256(p) for p in files},
        "extraction_code": {
            f: sha256(ROOT / f)
            for f in [
                "scripts/regenerate_model_ev.py",
                "src/ann/get_activations.py",
                "src/ann/predictive_coder.py",
            ]
        },
    }
    if pred:
        identity["checkpoint"] = sha256(ROOT / "data/weights/predictive_coder.ckpt")
    if crossdevice_dir is not None and name == "resnet50_untrained":
        identity["untrained_state"] = sha256(crossdevice_dir / "untrained_state.pt")
    validate_resume_identity(resume, identity)
    folds = {}
    for l in pending:
        paths = [dirs[l] / f"temporary_fold{k}.npy" for k in range(2)]
        if completed and not all(p.exists() for p in paths):
            raise RuntimeError(
                f"Missing retained folds for {name}/{l}; use a fresh output directory to re-extract"
            )
        folds[l] = [
            np.lib.format.open_memmap(
                p,
                mode="r+" if completed else "w+",
                dtype="float32",
                shape=(len(files), 90, len(feature_ids[l])),
            )
            for p in paths
        ]
    for l, pair in folds.items():
        if any(
            f.shape != (len(files), 90, len(feature_ids[l])) or f.dtype != np.float32
            for f in pair
        ):
            raise ValueError(f"Retained fold shape/dtype mismatch: {name}/{l}")
    equivalence = resume.get("static_encoding_max_difference", {})
    crossdevice = resume.get("crossdevice_max_abs_difference", {})
    if completed < len(files):
        torch.manual_seed(413)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ids = {l: torch.tensor(feature_ids[l], device=device) for l in pending}
        wrapper = None
        if pred:
            wrapper, prep = load_model(name, ROOT, "post_pred", device)
            core = wrapper.core
            model = core
        else:
            if name == "resnet50":
                core = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            elif name == "resnet50_untrained":
                core = models.resnet50(weights=None)
                if crossdevice_dir is not None:
                    core.load_state_dict(
                        torch.load(
                            crossdevice_dir / "untrained_state.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
            else:
                core = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model = (
                create_feature_extractor(core, {l: l for l in pending})
                .to(device)
                .eval()
            )
            prep = _make_tensor_preprocess()
        with torch.inference_mode():
            if crossdevice_dir is not None:
                with np.load(crossdevice_dir / f"{name}.npz") as check:
                    rgb = check["rgb"]
                    x = prep(np.repeat(rgb[:, None], 10, axis=1) if pred else rgb).to(
                        device
                    )
                    acts = predictive_outputs(core, x) if pred else model(x)
                    for l in pending:
                        actual = (
                            acts[l].flatten(1).index_select(1, ids[l]).cpu().numpy()
                        )
                        np.testing.assert_allclose(
                            actual, check[l], rtol=1e-3, atol=1e-4
                        )
                        crossdevice[l] = float(np.max(abs(actual - check[l])))
                print("CROSSDEVICE", name, crossdevice, flush=True)
            start_time = time.time()
            for i in range(completed, len(files)):
                with np.load(files[i]) as z:
                    rgb = z["rgb"]
                if len(rgb) != 360:
                    raise ValueError(f"Expected 360 views: {files[i]}")
                locacts = {l: [] for l in pending}
                for start in range(0, 360, batch_size):
                    batch = rgb[start : start + batch_size]
                    x = prep(
                        np.repeat(batch[:, None], 10, axis=1) if pred else batch
                    ).to(device)
                    acts = predictive_outputs(core, x) if pred else model(x)
                    if pred and not equivalence:
                        for l in pending:
                            ref = (
                                core.get_encoder_latents(x, l)[:, -1]
                                if l in ["conv1", "down1"]
                                else core.get_latents(
                                    x, which=0 if l == "pre_pred" else -1
                                )[:, -1]
                            )
                            equivalence[l] = float((ref - acts[l]).abs().max())
                            torch.testing.assert_close(
                                acts[l], ref, rtol=2e-4, atol=2e-5
                            )
                        print(
                            "predictive static-encoding equivalence",
                            equivalence,
                            flush=True,
                        )
                    for l in pending:
                        locacts[l].append(
                            acts[l].flatten(1).index_select(1, ids[l]).cpu().numpy()
                        )
                for l in pending:
                    a = np.concatenate(locacts[l]).reshape(90, 4, -1)
                    folds[l][0][i] = a[:, [0, 2]].mean(1)
                    folds[l][1][i] = a[:, [1, 3]].mean(1)
                    for f in folds[l]:
                        f.flush()
                save_json(
                    marker,
                    {
                        "completed_locations": i + 1,
                        "input_identity": identity,
                        "device": device,
                        "static_encoding_max_difference": equivalence,
                        "crossdevice_max_abs_difference": crossdevice,
                    },
                )
                if i % 30 == 0:
                    print(
                        name,
                        i + 1,
                        len(files),
                        round(time.time() - start_time, 1),
                        flush=True,
                    )
        del model, core, wrapper, ids, acts, x
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    inputs, locs = [], []
    for p in files:
        with np.load(p) as z:
            yaw, pitch = z["yaw"], z["pitch"]
        if (
            len(yaw) != 360
            or len(pitch) != 360
            or not np.isfinite(yaw).all()
            or not np.isfinite(pitch).all()
        ):
            raise ValueError(f"Invalid angle metadata: {p}")
        ep = abs(pitch - np.tile([-3, -1, 1, 3], 90))
        ey = abs((yaw - np.repeat(np.arange(90) * 4, 4) + 180) % 360 - 180)
        inputs.append(
            {
                "file": p.name,
                "sha256": sha256(p),
                "pitch_error_max": float(ep.max()),
                "yaw_error_max": float(ey.max()),
                "pitch_errors_over_005": int((ep > 0.05).sum()),
                "yaw_errors_over_005": int((ey > 0.05).sum()),
            }
        )
        locs.append([float(v) for v in p.stem[4:].split("_")])
    for l in pending:
        classes = json.loads((clean[l] / "classifications.json").read_text())
        for grid, zoom in grids.items():
            target = dirs[l] / f"{grid}.npz"
            if target.exists():
                continue
            d = score(
                *folds[l], np.full((len(files), 90), 4.0), zoom=zoom, cutoff=False
            )
            d["is_place"] = np.isin(
                np.arange(len(feature_ids[l])), classes["exclusive_place_cells_indices"]
            )
            d["unit_ids"] = feature_ids[l]
            save_scores(target, d)
            meta = summarize(d)
            meta.update(
                model=name,
                layer=l,
                grid=grid,
                device=resume.get("device", device),
                source_panorama_hashes=inputs,
                locations=locs,
                classification=str(clean[l]),
                folds=[[0, 2], [1, 3]],
                pitch_degrees=[-3, -1, 1, 3],
                evaluation="pitch perturbation folds, not measurement-noise replicates; static repeated-image protocol",
                static_encoding_max_difference=equivalence,
                crossdevice_max_abs_difference=crossdevice,
                code_sha256={
                    f: sha256(ROOT / f)
                    for f in [
                        "scripts/regenerate_model_ev.py",
                        "scripts/regenerate_ev.py",
                        "src/ev_summary.py",
                        "src/explained_variance.py",
                    ]
                },
            )
            if pred:
                meta["checkpoint_sha256"] = sha256(
                    ROOT / "data/weights/predictive_coder.ckpt"
                )
            if crossdevice_dir is not None:
                meta["crossdevice_reference_sha256"] = sha256(
                    crossdevice_dir / f"{name}.npz"
                )
                if name == "resnet50_untrained":
                    meta["untrained_state_sha256"] = sha256(
                        crossdevice_dir / "untrained_state.pt"
                    )
            save_json(
                dirs[l] / ("summary.json" if grid == "ev" else "native_summary.json"),
                meta,
            )
            print(
                name,
                l,
                grid,
                {k: meta[k] for k in ["n", "median", "axis0_greater"]},
                flush=True,
            )
        folds[l].clear()
        gc.collect()
        for p in dirs[l].glob("temporary_fold*.npy"):
            p.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--native", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crossdevice-dir", type=Path)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_LAYERS), default=list(MODEL_LAYERS)
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for name in args.models:
        run(
            name,
            MODEL_LAYERS[name],
            args.output,
            args.device,
            args.batch_size,
            args.native,
            args.crossdevice_dir,
        )
