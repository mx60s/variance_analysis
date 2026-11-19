"""Matched-panorama CNN extraction; existing EV and classification helpers."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from reclassify import classify
from regenerate_ev import score, summarize
from src.ann.get_activations import _make_tensor_preprocess

LAYERS = {
    "resnet50": ["layer1", "layer3", "layer4", "avgpool"],
    "resnet50_untrained": ["avgpool"],
    "vgg16": ["features.9"],
}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def extract(name, commit=lambda: None):
    torch.set_num_threads(4)
    torch.manual_seed(413)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path("/persist/panoramas-centers-v1")
    out = Path("/persist/cnn-matched-v1") / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    episodes = sorted(manifest["episodes"], key=lambda e: e["location"])
    if not manifest["complete"] or len(episodes) != 289:
        raise ValueError("Incomplete panoramas")
    if (out / "extraction.json").exists():
        raise FileExistsError("Use a fresh run")
    if name == "resnet50":
        core = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif name == "resnet50_untrained":
        core = models.resnet50(weights=None)
        core.load_state_dict(
            torch.load(
                "/project/untrained_state.pt", map_location="cpu", weights_only=True
            )
        )
    else:
        core = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    state = hashlib.sha256()
    for k, v in core.state_dict().items():
        state.update(k.encode())
        state.update(v.cpu().numpy().tobytes())
    model = create_feature_extractor(core, {l: l for l in LAYERS[name]}).cuda().eval()
    prep = _make_tensor_preprocess()
    identity = {
        "checkpoint_sha256": state.hexdigest(),
        "manifest_sha256": sha(root / "manifest.json"),
        "model": name,
        "code_sha256": sha(__file__),
    }
    locations = np.array([e["location"] for e in episodes])
    ids = {}
    arrays = {}
    shapes = {}
    t = time.monotonic()
    refdiff = {}
    with torch.inference_mode():
        # Match saved CPU-reference outputs on archived sample images before using new views.
        with np.load("/project/reference/" + name + ".npz") as check:
            x = prep(check["rgb"]).cuda()
            actual = model(x)
            for l in LAYERS[name]:
                provenance = json.loads(
                    (
                        ROOT / "provenance" / f"{name}_{l}" / "provenance.json"
                    ).read_text()
                )
                selected = np.asarray(
                    provenance.get("unit_ids", provenance.get("feature_ids")),
                    dtype=np.int64,
                )
                ids[l] = torch.tensor(selected, device="cuda")
                shapes[l] = list(actual[l].shape[1:])
                got = actual[l].flatten(1).index_select(1, ids[l]).cpu().numpy()
                np.testing.assert_allclose(got, check[l], rtol=1e-3, atol=1e-4)
                refdiff[l] = float(np.max(abs(got - check[l])))
                d = out / l
                d.mkdir(exist_ok=True)
                np.save(d / "unit_ids.npy", selected)
                np.save(d / "locations.npy", locations)
                arrays[l] = {
                    s: np.lib.format.open_memmap(
                        d / (s + ".npy"),
                        mode="w+",
                        dtype="float32",
                        shape=(289, 90, len(selected)),
                    )
                    for s in ["clean", "fold0", "fold1"]
                }
        for k, ep in enumerate(episodes):
            p = root / ep["file"]
            if sha(p) != ep["sha256"]:
                raise ValueError("Bad image hash")
            with np.load(p) as z:
                rgb = z["rgb"]
                np.testing.assert_allclose(
                    z["pitch"], np.tile([0, -3, -1, 1, 3], 90), atol=0.002
                )
                np.testing.assert_allclose(
                    (z["yaw"] - np.repeat(np.arange(0, 360, 4), 5) + 180) % 360 - 180,
                    0,
                    atol=0.002,
                )
            acts = {l: [] for l in ids}
            for j in range(0, 450, 32):
                a = model(prep(rgb[j : j + 32]).cuda())
                for l in ids:
                    acts[l].append(
                        a[l].flatten(1).index_select(1, ids[l]).cpu().numpy()
                    )
            for l in ids:
                a = np.concatenate(acts[l]).reshape(90, 5, -1)
                if not np.isfinite(a).all():
                    raise ValueError("Nonfinite activations")
                arrays[l]["clean"][k] = a[:, 0]
                arrays[l]["fold0"][k] = a[:, [1, 3]].mean(1)
                arrays[l]["fold1"][k] = a[:, [2, 4]].mean(1)
            if k % 30 == 0:
                print(
                    name,
                    "extract",
                    k + 1,
                    289,
                    round(time.monotonic() - t, 1),
                    flush=True,
                )
    for pair in arrays.values():
        for a in pair.values():
            a.flush()
    status = {
        "identity": identity,
        "complete": True,
        "layer_shapes": shapes,
        "reference_max_abs_difference": refdiff,
        "seconds": time.monotonic() - t,
    }
    (out / "extraction.json").write_text(json.dumps(status, indent=2))
    commit()
    return status


def analyze_layer(name, layer, commit=lambda: None):
    root = Path("/persist/cnn-matched-v1") / name

    root = Path(root)
    d = root / layer
    status = json.loads((root / "extraction.json").read_text())
    if not status["complete"]:
        raise ValueError("Extraction incomplete")
    ids = np.load(d / "unit_ids.npy")
    locations = np.load(d / "locations.npy")
    clean = np.load(d / "clean.npy", mmap_mode="r")
    a = np.load(d / "fold0.npy", mmap_mode="r")
    b = np.load(d / "fold1.npy", mmap_mode="r")
    summaries = {}
    for score_name, zoom in [("ev_native", (1, 1)), ("ev", (1, 5))]:
        path = d / (score_name + ".npz")
        if path.exists():
            with np.load(path) as z:
                result = dict(z)
        else:
            result = score(a, b, np.full((289, 90), 4.0), zoom=zoom, cutoff=False)
            np.savez_compressed(path, **result, unit_ids=ids)
        summaries[score_name] = summarize(result)
        print(layer, score_name, summaries[score_name], flush=True)
        commit()
    cache = Path("/tmp") / (layer + "_clean.npz")
    if not (d / "classifications.json").exists():
        np.savez(cache, activations=clean, locations=locations)
        classify(cache, name, layer, d, field_method="tanni_iterative_no_stability")
        cache.unlink()
    policy = json.loads((d / "summary.json").read_text()).get("policy", "")
    if not policy.startswith("tanni_iterative_no_stability"):
        raise ValueError("Existing classification cache uses an earlier method. Reclassify into a fresh output directory with the iterative procedure before resuming.")
    groups = json.loads((d / "classifications.json").read_text())
    rng = np.random.default_rng(20261116 + LAYERS[name].index(layer))
    chosen = list(map(int, rng.choice(len(ids), 8, replace=False)))
    reasons = {i: "Random sample (unconditional)" for i in chosen}
    for key, count in [
        ("exclusive_place_cells_indices", 4),
        ("exclusive_direction_cells_indices", 2),
        ("place_and_direction_not_border_cells_indices", 2),
    ]:
        candidates = sorted(set(groups[key]) - set(chosen))
        if candidates:
            selected = list(
                map(
                    int,
                    rng.choice(candidates, min(count, len(candidates)), replace=False),
                )
            )
            chosen.extend(selected)
            reasons.update(
                {i: "Random within " + key.removesuffix("_indices") for i in selected}
            )
    left = sorted(set(range(len(ids))) - set(chosen))
    extra = list(map(int, rng.choice(left, 16 - len(chosen), replace=False)))
    chosen += extra
    reasons.update({i: "Random sample (fill remaining slots)" for i in extra})
    labels = {i: k.removesuffix("_indices") for k, v in groups.items() for i in v}
    neurons = [
        {
            "cache_index": i,
            "unit_id": int(ids[i]),
            "tensor_index": list(
                map(int, np.unravel_index(ids[i], status["layer_shapes"][layer]))
            ),
            "selection": reasons[i],
            "classification": labels[i],
        }
        for i in chosen
    ]
    np.savez_compressed(
        d / "gallery_maps.npz",
        maps=clean[:, :, chosen].transpose(2, 0, 1).reshape(16, 17, 17, 90),
        unit_ids=ids[chosen],
        cache_indices=chosen,
        locations=locations,
    )
    (d / "gallery_selection.json").write_text(json.dumps(neurons, indent=2))
    summaries.update(
        layer=layer,
        checkpoint_sha256=status["identity"]["checkpoint_sha256"],
        classification_counts={k: len(v) for k, v in groups.items()},
        selection="8 unconditional random, up to4 place/2HD/2mixed, remaining random; seed20261116+layer_index",
        limitations="Static-image CNN evaluation; pitch-defined folds reflect visual perturbations, not independent measurement noise.",
        code_sha256={
            f: sha(ROOT / f)
            for f in [
                "scripts/finalization/cnn_matched.py",
                "scripts/regenerate_ev.py",
                "src/ev_summary.py",
                "src/explained_variance.py",
                "src/ann/classify_neurons.py",
                "src/ann/score_utils.py",
            ]
        },
    )
    (d / "analysis_summary.json").write_text(json.dumps(summaries, indent=2))
    commit()
    return summaries
