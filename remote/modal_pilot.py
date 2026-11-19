"""Prepared remote pilot; importing this file does not start compute.

After reviewing the cost estimate and uploading trajectory data to the named
volume, run: modal run remote/modal_pilot.py --train /data/train --validation /data/validation
No credentials are embedded or forwarded to the container.
"""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "numpy==2.5.4", "Pillow==13.0.0")
    .add_local_dir(
        ROOT / "src", remote_path="/project/src", ignore=["**/__pycache__/**"]
    )
    .add_local_file(
        ROOT / "scripts/train_pilot.py", remote_path="/project/train_pilot.py"
    )
)
app = modal.App("variance-training-pilot")
volume = modal.Volume.from_name("variance-training-data", create_if_missing=False)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=1800,
    max_containers=1,
    retries=0,
    volumes={"/data": volume},
)
def pilot(train: str, validation: str):
    import sys, time

    sys.path.insert(0, "/project")
    from train_pilot import run

    output = f"/data/pilot-results/{int(time.time())}"
    result = run(
        train,
        validation,
        output,
        device="cuda",
        steps=200,
        batch_size=32,
        max_seconds=600,
    )
    volume.commit()
    return {k: v for k, v in result.items() if not k.endswith("_paths")}


@app.local_entrypoint()
def main(train: str = "/data/train", validation: str = "/data/validation"):
    print(pilot.remote(train, validation))
