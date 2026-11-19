"""Recompute classification from fresh checkpoint activations or an existing cache.

Fresh predictive-coder evaluation uses the existing repeated-static-frame protocol.
Outputs are isolated, with original feature IDs and input hashes; no paper results
are overwritten. Example: python scripts/reclassify.py --output results/review/run1
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.ann.classify_neurons import score_neurons, classify_neurons
from src.ann.fig import plot_cell_type_pie


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def classify(path, model, layer, output, compare_ref=None, field_method="adapted"):
    with np.load(path) as z:
        acts, locations = z["activations"], z["locations"]
    xs, ys = np.unique(locations[:, 0]), np.unique(locations[:, 1])
    expected = np.array([(x, y) for x in xs for y in ys])
    if not np.array_equal(locations, expected):
        raise ValueError("Cache locations must be a complete, sorted Cartesian grid")
    config = dict(
        env_bounds=(0, len(xs) - 1, 0, len(ys) - 1),
        n_rotations=acts.shape[1],
        model_name=model,
        output_layer=layer,
    )
    del acts
    kw = dict(
        model_reps_paths=[path],
        config=config,
        base_dir=ROOT,
        n_visits=1,
        use_old_place_method=False,
        pixel_min_threshold=15,
    )
    chart_path, _ = score_neurons(**kw, output_dir=output, field_method=field_method)
    chart = np.load(chart_path, allow_pickle=True)
    groups = classify_neurons(chart)
    summary = {
        "counts": {k: len(v) for k, v in groups.items()},
        "total": len(chart),
        "activation_sha256": sha256(path),
        "source_hashes": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                ROOT / "src/ann/classify_neurons.py",
                ROOT / "src/ann/score_utils.py",
                ROOT / "src/ann/tanni_fields.py",
            ]
        },
        "policy": field_method
        + "; field-mask and field-statistics fixed; signed post_pred rectified as before",
    }
    (output / "classifications.json").write_text(json.dumps(groups, indent=2))
    fig = plot_cell_type_pie(groups)
    fig.suptitle(f"{model}/{layer} • n={len(chart):,}", fontsize=12)
    fig.savefig(output / "classification.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    if compare_ref:
        # Execute the user's versioned pre-fix code on exactly the same activations.
        legacy = {"__name__": "legacy_classification"}
        for file in ["src/ann/score_utils.py", "src/ann/classify_neurons.py"]:
            source = subprocess.check_output(
                ["git", "show", f"{compare_ref}:{file}"], cwd=ROOT, text=True
            )
            source = source.replace("from .score_utils import *", "")
            if file.endswith("classify_neurons.py"):
                source = source.replace(
                    'base_dir / "data/ann_activations" / f"{model_name}_{output_layer}"',
                    "base_dir",
                )
            exec(compile(source, file, "exec"), legacy)
        old_dir = output / "before"
        old_dir.mkdir(parents=True, exist_ok=True)
        old_path, _ = legacy["score_neurons"](**{**kw, "base_dir": old_dir})
        old_groups = legacy["classify_neurons"](np.load(old_path, allow_pickle=True))
        summary["before_counts"] = {k: len(v) for k, v in old_groups.items()}
        old_label = {n: k for k, v in old_groups.items() for n in v}
        new_label = {n: k for k, v in groups.items() for n in v}
        summary["changed_units"] = [
            n for n in new_label if new_label[n] != old_label[n]
        ]
        summary["comparison_ref"] = compare_ref
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "layer": layer,
                **{k: v for k, v in summary.items() if k != "changed_units"},
                "n_changed": len(summary.get("changed_units", [])),
            }
        ),
        flush=True,
    )


def fresh_predictive(output, layers, device, batch_size, max_locations=None):
    import torch
    from src.ann.get_activations import load_model

    files = sorted((ROOT / "data/minecraft/panorama-90yaw").glob("loc_*.npz"))
    if not files:
        raise FileNotFoundError("No panorama files")
    if max_locations:
        files = files[:max_locations]
    inputs = [{"file": p.name, "sha256": sha256(p)} for p in files]
    checkpoint = ROOT / "data/weights/predictive_coder.ckpt"
    for layer in layers:
        dest = output / layer
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "activations.npz"
        if target.exists():
            raise FileExistsError(f"Refusing stale cache: {target}")
        model, preprocess = load_model("predictive_coder", ROOT, layer, device)
        t = time.time()
        selected = None
        buf = None
        locs = []
        angle_deviations = []
        with torch.inference_mode():
            for i, p in enumerate(files):
                with np.load(p) as z:
                    rgb = z["rgb"]
                    yaw = z["yaw"]
                    pitch = z["pitch"]
                    if (
                        len(rgb) != 90
                        or not np.allclose(pitch, 0)
                        or not np.all(np.isfinite(yaw))
                    ):
                        raise ValueError(f"Unexpected panorama shape/pitch: {p}")
                    err = np.abs((yaw - np.arange(90) * 4 + 180) % 360 - 180)
                    if err.max() > 0.01:
                        angle_deviations.append(
                            {
                                "file": p.name,
                                "max_yaw_error_degrees": float(err.max()),
                                "affected_frames": int(np.sum(err > 0.01)),
                            }
                        )
                xy = [float(v) for v in p.stem.removeprefix("loc_").split("_")]
                locs.append(xy)
                for start in range(0, len(rgb), batch_size):
                    batch = np.repeat(rgb[start : start + batch_size, None], 10, axis=1)
                    out = model(preprocess(batch).to(device))[:, -1].flatten(1)
                    if selected is None:
                        selected = (
                            torch.randperm(
                                out.shape[1],
                                generator=torch.Generator().manual_seed(413),
                            )[: min(8000, out.shape[1])]
                            if out.shape[1] > 8000
                            else torch.arange(out.shape[1])
                        )
                        buf = np.lib.format.open_memmap(
                            dest / "activations.npy",
                            mode="w+",
                            dtype="float32",
                            shape=(len(files), 90, len(selected)),
                        )
                    buf[i, start : start + len(batch)] = (
                        out.index_select(1, selected.to(device)).cpu().numpy()
                    )
                if i % 30 == 0:
                    print(
                        f"{layer}: {i + 1}/{len(files)} locations, {time.time() - t:.1f}s",
                        flush=True,
                    )
        buf.flush()
        np.savez_compressed(
            target, activations=buf, locations=locs, unit_ids=selected.numpy()
        )
        metadata = {
            "checkpoint_sha256": sha256(checkpoint),
            "panoramas": inputs,
            "angle_deviations": angle_deviations,
            "angle_policy": "preserve historical nominal 4-degree index bins; actual yaw deviations logged, not corrected",
            "unit_seed": 413,
            "unit_ids": selected.tolist(),
            "device": device,
            "torch": torch.__version__,
            "evaluation": "ten repetitions of each static image, final timestep; NOT natural-history evaluation",
            "seconds": time.time() - t,
            "git_base": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
        (dest / "provenance.json").write_text(json.dumps(metadata, indent=2))
        del buf, model
        if device == "mps":
            torch.mps.empty_cache()
        yield layer, target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--layers", nargs="+", default=["conv1", "down1", "pre_pred", "post_pred"]
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--model", default="predictive_coder")
    ap.add_argument("--compare-ref")
    ap.add_argument(
        "--field-method", choices=["adapted", "tanni_iterative", "tanni_iterative_no_stability"], default="adapted"
    )
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.cache:
        if len(a.layers) != 1:
            ap.error("--cache requires exactly one --layers entry")
        classify(a.cache, a.model, a.layers[0], a.output, a.compare_ref, a.field_method)
    else:
        for layer, path in fresh_predictive(a.output, a.layers, a.device, a.batch_size):
            classify(
                path,
                "predictive_coder",
                layer,
                a.output / layer,
                a.compare_ref,
                a.field_method,
            )


if __name__ == "__main__":
    main()
