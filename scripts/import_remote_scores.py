"""Validate and import only completed, identity-matched model score archives."""

import hashlib
import argparse
import io
import json
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "resnet50": ["layer1", "layer3", "layer4", "avgpool"],
    "resnet50_untrained": ["avgpool"],
    "vgg16": ["features.9"],
    "predictive_coder": ["conv1", "down1", "pre_pred", "post_pred"],
}


def import_scores(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    panoramas = ROOT / "data/minecraft/panorama-90yaw-pitch"
    expected_hashes = {}
    for path in panoramas.glob("loc_*.npz"):
        with path.open("rb") as f:
            expected_hashes[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError("ZIP integrity check failed")
        names = set(z.namelist())
        allowed = {"remote_run.json"}
        for model, layers in MODELS.items():
            for layer in layers:
                allowed.update(
                    f"{model}_{layer}/{n}"
                    for n in [
                        "ev.npz",
                        "ev_native.npz",
                        "summary.json",
                        "native_summary.json",
                    ]
                )
        if names - allowed:
            raise ValueError(f"Unexpected archive paths: {names - allowed}")
        count = 0
        for model, layers in MODELS.items():
            for layer in layers:
                folder = f"{model}_{layer}"
                required = {
                    f"{folder}/{n}" for n in ["ev.npz", "ev_native.npz", "summary.json"]
                }
                if not required.issubset(names):
                    continue
                clean = (
                    ROOT
                    / "results/review_20261116"
                    / ("predictive_coder" if model == "predictive_coder" else "cnns")
                    / (layer if model == "predictive_coder" else folder)
                )
                prov = json.loads((clean / "provenance.json").read_text())
                ids = np.array(prov.get("unit_ids", prov.get("feature_ids")))
                classes = json.loads((clean / "classifications.json").read_text())
                mask = np.isin(
                    np.arange(len(ids)), classes["exclusive_place_cells_indices"]
                )
                payload = {}
                for stem in ["ev", "ev_native"]:
                    data = z.read(f"{folder}/{stem}.npz")
                    with np.load(io.BytesIO(data)) as a:
                        np.testing.assert_array_equal(a["unit_ids"], ids)
                        np.testing.assert_array_equal(a["is_place"], mask)
                        for key in ["ev", "normalized", "reliability"]:
                            if a[key].shape != (len(ids), 2):
                                raise ValueError(
                                    f"Incorrect score dimensions: {folder}/{key}"
                                )
                    payload[stem + ".npz"] = data
                for name in ["summary.json", "native_summary.json"]:
                    path = f"{folder}/{name}"
                    if path not in names:
                        continue
                    meta = json.loads(z.read(path))
                    if meta["model"] != model or meta["layer"] != layer:
                        raise ValueError("Metadata model/layer mismatch")
                    if (
                        len(meta["source_panorama_hashes"]) != len(expected_hashes)
                        or {
                            x["file"]: x["sha256"]
                            for x in meta["source_panorama_hashes"]
                        }
                        != expected_hashes
                    ):
                        raise ValueError(
                            "Panorama identities differ from local evaluation data"
                        )
                    if model == "predictive_coder":
                        with (ROOT / "data/weights/predictive_coder.ckpt").open(
                            "rb"
                        ) as f:
                            if (
                                meta["checkpoint_sha256"]
                                != hashlib.file_digest(f, "sha256").hexdigest()
                            ):
                                raise ValueError("Predictive checkpoint hash differs")
                    meta["remote_classification_path"] = meta["classification"]
                    meta["classification"] = str(clean)
                    payload[name] = json.dumps(meta, indent=2).encode()
                dest = output / folder
                dest.mkdir(exist_ok=True)
                for name, data in payload.items():
                    temp = dest / (name + ".tmp")
                    temp.write_bytes(data)
                    temp.replace(dest / name)
                count += 1
        if "remote_run.json" in names:
            (output / "remote_run.json").write_bytes(z.read("remote_run.json"))
    print(
        "Imported", count, "completed model/layers with matched feature IDs and classes"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import_scores(a.archive, a.output)
