"""Bounded remote inference under the user's estimate-before-spending authorization.

L4/A10 + 4 CPU + 16 GiB: about $1.12–1.42/h; 30-minute function timeout, one container,
zero retries. Allow $5 for build/setup overhead; this is not a billing hard cap.
Only the explicit stage, evaluation panoramas and existing checkpoint are sent.
"""

import os
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(os.environ.get("EV_REMOTE_STAGE", "/project"))
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.1",
        "numpy==2.5.4",
        "scipy==1.18.1",
        "numba==0.68.0",
        "matplotlib==3.11.2",
        "threadpoolctl==3.7.0",
        "transformers==5.20.2",
    )
    .env(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(STAGE, remote_path="/project", ignore=["**/__pycache__/**"])
    .add_local_dir(
        ROOT / "data/minecraft/panorama-90yaw-pitch",
        remote_path="/project/data/minecraft/panorama-90yaw-pitch",
    )
    .add_local_file(
        ROOT / "data/weights/predictive_coder.ckpt",
        remote_path="/project/data/weights/predictive_coder.ckpt",
    )
)
app = modal.App("variance-review-inference")


@app.function(
    image=image,
    gpu=["L4", "A10"],
    cpu=4,
    memory=16384,
    timeout=1800,
    max_containers=1,
    retries=0,
)
def infer(selected_models: list[str]):
    import subprocess, time, zipfile, io, json

    layer_counts = {
        "resnet50": 4,
        "resnet50_untrained": 1,
        "vgg16": 1,
        "predictive_coder": 4,
    }
    if not selected_models or any(m not in layer_counts for m in selected_models):
        raise ValueError("Unknown or empty model selection")
    expected_layers = sum(layer_counts[m] for m in selected_models)
    start = time.time()
    print("REMOTE START", flush=True)
    root = Path("/project")
    output = root / "results/review_20261116/revalidation/models"
    failure = None
    try:
        result = subprocess.run(
            [
                "python",
                "scripts/regenerate_model_ev.py",
                "--output",
                str(output),
                "--device",
                "cuda",
                "--batch-size",
                "64",
                "--native",
                "--crossdevice-dir",
                "/project/data/crossdevice",
                "--models",
                *selected_models,
            ],
            cwd=root,
            check=False,
            timeout=1740,
        )
        if result.returncode:
            failure = f"Inference exited with status {result.returncode}"
    except subprocess.TimeoutExpired:
        failure = "Inference exceeded its 1740-second subprocess limit"
    files = [p for p in output.glob("*/*.npz") if p.name in ["ev.npz", "ev_native.npz"]]
    files += list(output.glob("*/*summary.json"))
    complete = (
        len(list(output.glob("*/ev.npz"))) == expected_layers
        and len(list(output.glob("*/ev_native.npz"))) == expected_layers
        and failure is None
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(output)))
        z.writestr(
            "remote_run.json",
            json.dumps(
                {
                    "seconds": time.time() - start,
                    "complete": complete,
                    "selected_models": selected_models,
                    "failure": failure,
                    "gpu": subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        text=True,
                    ).strip(),
                    "cpu": 4,
                    "memory_mib": 16384,
                    "timeout_seconds": 1800,
                    "tf32": False,
                    "training": False,
                }
            ),
        )
    return buf.getvalue()


@app.local_entrypoint()
def main(
    output: str, models: str = "resnet50,resnet50_untrained,vgg16,predictive_coder"
):
    print("Submitting bounded inference call", flush=True)
    data = infer.remote(models.split(","))
    Path(output).write_bytes(data)
    import io, json, zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        status = json.loads(archive.read("remote_run.json"))
    print("Saved remote results", output, len(data), "bytes", status)
    if not status["complete"]:
        raise RuntimeError(
            "Partial results were saved; inspect remote_run.json before continuing"
        )
