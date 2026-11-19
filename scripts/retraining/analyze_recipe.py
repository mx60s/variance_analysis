"""Fresh panorama extraction and existing EV/classification for a causal checkpoint."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from regenerate_ev import score, summarize
from scripts.retraining.recipe_model import make_model

LAYERS = ["conv1", "down1", "pre_pred", "post_pred"]


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def outputs(core, x):
    e = core.encoder
    a = torch.relu(e.bn1(e.conv1(x)))
    b = e.down1(e.maxpool(a))
    f = e.down2(b)[:, None].expand(-1, 10, -1, -1, -1).contiguous()
    pre = f[:, -1]
    f = core._normalize_latents(core.attention_1(f, f, f, core.mask)[0] + f)
    f = core._normalize_latents(core.ffn_1(f) + f)
    f = core._normalize_latents(core.attention_2(f, f, f, core.mask)[0] + f)
    f = core._normalize_latents(core.ffn_2(f) + f)
    return {"conv1": a, "down1": b, "pre_pred": pre, "post_pred": f[:, -1]}


def extract(checkpoint, panoramas, output, commit=lambda: None):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    core = make_model(ck["architecture"]).cuda().eval()
    core.load_state_dict(ck["model"], strict=True)
    root = Path(panoramas)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"] or len(manifest["episodes"]) != 289:
        raise ValueError("Incomplete panorama grid")
    training_manifest = Path(checkpoint).parent.parent / "smooth-v1" / "manifest.json"
    train_manifest = json.loads(training_manifest.read_text())
    if sha(training_manifest) != ck["dataset_sha256"]:
        raise ValueError("Training manifest mismatch")
    for key in [
        "world_files_sha256",
        "minecraft_version",
        "malmo_version",
        "camera_y",
        "camera_fov_degrees",
        "render_options",
    ]:
        if manifest[key] != train_manifest[key]:
            raise ValueError("Training/evaluation rendering mismatch: " + key)
    episodes = sorted(manifest["episodes"], key=lambda e: e["location"])
    locations = np.array([e["location"] for e in episodes])
    expected = np.array(
        [(x, z) for x in np.arange(120.5, 137, 1) for z in np.arange(197.5, 214, 1)]
    )
    np.testing.assert_array_equal(locations, expected)
    identity = {
        "checkpoint_sha256": sha(checkpoint),
        "manifest_sha256": sha(root / "manifest.json"),
        "analysis_code_sha256": sha(__file__),
        "recipe_model_sha256": sha(ROOT / "scripts/retraining/recipe_model.py"),
        "model_code_sha256": sha(ROOT / "src/ann/predictive_coder.py"),
    }
    if (
        identity["recipe_model_sha256"] != ck["recipe_model_sha256"]
        or identity["model_code_sha256"] != ck["model_code_sha256"]
    ):
        raise ValueError("Checkpoint source mismatch")
    marker = out / "extraction.json"
    if marker.exists():
        status = json.loads(marker.read_text())
        if status["identity"] != identity:
            raise ValueError("Resume identity changed")
        if status["complete"]:
            return status
        raise ValueError("Partial extraction requires a fresh destination")
    marker.write_text(json.dumps({"identity": identity, "complete": False}))
    means = torch.tensor(ck["mean"], device="cuda", dtype=torch.float32)[
        None, :, None, None
    ]
    stds = torch.tensor(ck["std"], device="cuda", dtype=torch.float32)[
        None, :, None, None
    ]
    arrays = {}
    ids = {}
    shapes = {}
    equivalence = {}
    started = time.monotonic()
    with torch.inference_mode():
        for k, ep in enumerate(episodes):
            path = root / ep["file"]
            if sha(path) != ep["sha256"]:
                raise ValueError("Panorama checksum mismatch")
            with np.load(path) as z:
                rgb = z["rgb"]
                yaw = z["yaw"]
                pitch = z["pitch"]
                np.testing.assert_allclose(
                    (yaw - np.repeat(np.arange(0, 360, 4), 5) + 180) % 360 - 180,
                    0,
                    atol=0.002,
                )
                np.testing.assert_allclose(
                    pitch, np.tile([0, -3, -1, 1, 3], 90), atol=0.002
                )
                if rgb.shape != (450, 64, 64, 3):
                    raise ValueError("Wrong image shape")
            location = {l: [] for l in LAYERS}
            for j in range(0, 450, 64):
                x = (
                    torch.from_numpy(rgb[j : j + 64]).cuda().float().permute(0, 3, 1, 2)
                    / 255
                )
                x = (x - means) / stds
                acts = outputs(core, x)
                if not ids:
                    for l, a in acts.items():
                        shapes[l] = list(a.shape[1:])
                        n = a[0].numel()
                        old = (
                            ROOT
                            / "results/review_20261116/predictive_coder"
                            / l
                            / "provenance.json"
                        )
                        provenance = json.loads(old.read_text())
                        selected = np.asarray(
                            provenance.get("unit_ids", provenance.get("feature_ids")),
                            dtype=np.int64,
                        )
                        if (
                            len(set(selected)) != len(selected)
                            or selected.min() < 0
                            or selected.max() >= n
                        ):
                            raise ValueError("Invalid original unit IDs")
                        ids[l] = torch.tensor(selected, device="cuda")
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
                    xx = x[:2, None].expand(-1, 10, -1, -1, -1).contiguous()
                    for l in LAYERS:
                        ref = (
                            core.get_encoder_latents(xx, l)[:, -1]
                            if l in ["conv1", "down1"]
                            else core.get_latents(
                                xx, which=0 if l == "pre_pred" else -1
                            )[:, -1]
                        )
                        torch.testing.assert_close(
                            acts[l][:2], ref, rtol=2e-4, atol=2e-5
                        )
                        equivalence[l] = float((acts[l][:2] - ref).abs().max())
                for l in LAYERS:
                    a = acts[l].flatten(1).index_select(1, ids[l]).cpu().numpy()
                    if not np.isfinite(a).all():
                        raise ValueError("Nonfinite activations")
                    location[l].append(a)
            for l in LAYERS:
                a = np.concatenate(location[l]).reshape(90, 5, -1)
                arrays[l]["clean"][k] = a[:, 0]
                arrays[l]["fold0"][k] = a[:, [1, 3]].mean(1)
                arrays[l]["fold1"][k] = a[:, [2, 4]].mean(1)
            if k % 30 == 0:
                print(
                    "extract",
                    k + 1,
                    289,
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
    for d in arrays.values():
        for a in d.values():
            a.flush()
    status = {
        "identity": identity,
        "complete": True,
        "layer_shapes": shapes,
        "feature_counts": {l: len(ids[l]) for l in LAYERS},
        "static_equivalence": equivalence,
        "seconds": time.monotonic() - started,
        "protocol": "10 identical frames; 0-pitch classification; pitch [-3,+1] vs [-1,+3] folds; original recorded feature IDs; raw signed EV",
    }
    marker.write_text(json.dumps(status, indent=2))
    commit()
    return status


def analyze_layer(root, layer, commit=lambda: None):
    from reclassify import classify

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
    for name, zoom in [("ev_native", (1, 1)), ("ev", (1, 5))]:
        path = d / (name + ".npz")
        if path.exists():
            with np.load(path) as z:
                result = dict(z)
        else:
            result = score(a, b, np.full((289, 90), 4.0), zoom=zoom, cutoff=False)
            np.savez_compressed(path, **result, unit_ids=ids)
        summaries[name] = summarize(result)
        print(layer, name, summaries[name], flush=True)
        commit()
    cache = Path("/tmp") / (layer + "_clean.npz")
    if not (d / "classifications.json").exists():
        np.savez(cache, activations=clean, locations=locations)
        classify(cache, "predictive_coder", layer, d, field_method="luo")
        cache.unlink()
    policy = json.loads((d / "summary.json").read_text()).get("policy", "")
    if not policy.startswith("luo"):
        raise ValueError("Existing classification cache uses an earlier method. Reclassify into a fresh output directory with the iterative procedure before resuming.")
    groups = json.loads((d / "classifications.json").read_text())
    rng = np.random.default_rng(20261116 + LAYERS.index(layer))
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
        limitations="Static repeated-image evaluation; pitch-defined folds reflect visual perturbations, not independent measurement noise; post_pred rectified for classification only; EV and gallery retain signed activations.",
        code_sha256={
            f: sha(ROOT / f)
            for f in [
                "scripts/retraining/analyze_recipe.py",
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
