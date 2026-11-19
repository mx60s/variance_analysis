from pathlib import Path
import modal

ROOT = Path("/Users/mvonebers/variance_analysis")
HERE = Path(__file__).resolve().parent
app = modal.App("variance-paper-recipe")
volume = modal.Volume.from_name("variance-minecraft-render", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "numpy==2.5.4", "Pillow==13.0.0")
    .add_local_dir(ROOT / "src", "/project/src", ignore=["**/__pycache__/**"])
    .add_local_dir(
        ROOT / "scripts/retraining",
        "/project/scripts/retraining",
        ignore=["**/__pycache__/**"],
    )
)


@app.function(
    image=image,
    gpu=["L4", "A10"],
    cpu=4,
    memory=8192,
    timeout=6000,
    max_containers=1,
    retries=0,
    volumes={"/persist": volume},
)
def train(
    steps: int = 12600,
    seconds: int = 5400,
    resume: bool = False,
    final: bool = True,
    data: str = "smooth-v1",
    run_name: str = "paper-recipe-v1",
    seed: int = 20261116,
):
    import sys, pathlib, json, shutil

    if (
        data == "smooth-pilot"
        and not pathlib.Path("/persist/smooth-pilot/manifest.json").exists()
    ):
        source = pathlib.Path("/persist/smooth-v1")
        m = json.loads((source / "manifest.json").read_text())
        if len(m["episodes"]) != 20 or not m["complete"]:
            raise ValueError("Pilot collection is not complete")
        target = pathlib.Path("/persist/smooth-pilot")
        target.mkdir(exist_ok=True)
        for ep in m["episodes"]:
            shutil.copyfile(source / (ep["id"] + ".npz"), target / (ep["id"] + ".npz"))
        shutil.copyfile(source / "manifest.json", target / "manifest.json")
        volume.commit()
    sys.path.insert(0, "/project")
    from scripts.retraining.train_recipe import run

    try:
        return run(
            "/persist/" + data,
            "/persist/" + run_name,
            steps,
            seconds,
            resume,
            final,
            seed,
        )
    finally:
        volume.commit()


@app.local_entrypoint()
def main(
    steps: int = 12600,
    seconds: int = 5400,
    resume: bool = False,
    final: bool = True,
    data: str = "smooth-v1",
    run_name: str = "paper-recipe-v1",
    seed: int = 20261116,
):
    import json

    (HERE / "training_result.json").write_text(
        json.dumps(
            train.remote(steps, seconds, resume, final, data, run_name, seed), indent=2
        )
    )
