from pathlib import Path
import modal

image = modal.Image.from_id("im-AC4XTSzcKIQLLFHwhR4jP0").apt_install(
    "x11-xserver-utils", "mesa-utils"
)
volume = modal.Volume.from_name("variance-minecraft-render", create_if_missing=False)
HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/mvonebers/variance_analysis")
image = (
    image.add_local_dir(
        HERE.parent / "upstream/masters-thesis/environments/small-world",
        "/source-world",
    )
    .add_local_dir(HERE / "paths_v2", "/paths")
    .add_local_file(ROOT / "scripts/retraining/collect_malmo.py", "/collect_malmo.py")
)
app = modal.App("variance-minecraft-collect-parallel")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    retries=0,
    max_containers=8,
    volumes={"/persist": volume},
)
def collect(start: int, episodes: int = 60):
    import os, subprocess, pathlib, shutil, time, json, socket, tarfile

    output = pathlib.Path("/persist/smooth-v1/shards") / str(start)
    output.mkdir(parents=True, exist_ok=True)
    root = next(pathlib.Path("/opt/malmo").iterdir())
    work = pathlib.Path("/tmp/buildroot")
    work.mkdir()
    with tarfile.open("/persist/build-cache.tar") as t:
        t.extractall(work, filter="data")
    mc = work / "Minecraft"
    world = pathlib.Path("/tmp/world-copy")
    if not world.exists():
        shutil.copytree("/source-world", world)
    env = dict(
        os.environ,
        GRADLE_USER_HOME=str(work / "gradle"),
        GRADLE_OPTS="-Xmx3g -XX:ActiveProcessorCount=4",
        JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64",
        DISPLAY=":99",
        LIBGL_ALWAYS_SOFTWARE="1",
        MALMO_XSD_PATH=str(root / "Schemas"),
    )
    xvfb = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "800x600x24", "-ac"], env=env
    )
    (mc / "run/config").mkdir(parents=True, exist_ok=True)
    (mc / "run/config/malmomodCLIENT.cfg").write_text(
        "malmoports {\n I:portOverride=10000\n}\n"
    )
    (mc / "run/options.txt").write_text(
        "renderDistance:4\nbobView:false\nmaxFps:120\nfancyGraphics:false\nrenderClouds:false\nparticles:2\nfov:0.0\n"
    )
    log = open(output / "client.log", "w")
    client = subprocess.Popen(
        ["./gradlew", "runClient"],
        cwd=mc,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if client.poll() is not None:
                raise RuntimeError(
                    "Client exited: " + (output / "client.log").read_text()[-6000:]
                )
            try:
                with socket.create_connection(("127.0.0.1", 10000), timeout=1):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError(
                "Client startup timed out: "
                + (output / "client.log").read_text()[-6000:]
            )
        with open(output / "collection.log", "a") as f:
            proc = subprocess.Popen(
                [
                    "python3.6",
                    "/collect_malmo.py",
                    "--malmo",
                    str(root),
                    "--world",
                    str(world),
                    "--paths",
                    "/paths",
                    "--output",
                    str(output),
                    "--episodes",
                    str(episodes),
                    "--start",
                    str(start),
                ],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            while proc.poll() is None:
                time.sleep(20)
                volume.commit()
            if proc.returncode:
                raise RuntimeError((output / "collection.log").read_text()[-6000:])
        return json.loads((output / "manifest.json").read_text())
    finally:
        import signal

        try:
            os.killpg(client.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        xvfb.terminate()
        log.close()
        volume.commit()


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=300,
    retries=0,
    volumes={"/persist": volume},
)
def aggregate():
    import pathlib, json, hashlib

    volume.reload()
    root = pathlib.Path("/persist/smooth-v1")
    base = json.loads((root / "manifest.json").read_text())
    if len(base["episodes"]) != 20 or not base["complete"]:
        raise ValueError("Pilot is incomplete")
    manifests = []
    for start in range(20, 500, 60):
        p = root / "shards" / str(start) / "manifest.json"
        m = json.loads(p.read_text())
        if not m["complete"] or len(m["episodes"]) != 60:
            raise ValueError("Incomplete shard")
        if m["world_files_sha256"] != base["world_files_sha256"]:
            raise ValueError("World mismatch")
        for ep in m["episodes"]:
            ep["file"] = "shards/" + str(start) + "/" + ep["id"] + ".npz"
            if not (root / ep["file"]).exists():
                raise ValueError("Missing episode")
            base["episodes"].append(ep)
        manifests.append(
            {
                "file": str(p.relative_to(root)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "collector_sha256": m["collector_sha256"],
            }
        )
    if sorted(e["id"] for e in base["episodes"]) != ["%05d" % i for i in range(500)]:
        raise ValueError("Missing or repeated episode IDs")
    base["shards"] = manifests
    base["complete"] = True
    (root / "manifest.tmp.json").write_text(json.dumps(base, indent=2))
    (root / "manifest.tmp.json").replace(root / "manifest.json")
    volume.commit()
    return {
        "episodes": len(base["episodes"]),
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
    }


@app.local_entrypoint()
def main():
    import json

    results = list(collect.starmap([(s, 60) for s in range(20, 500, 60)]))
    (HERE / "shard_results.json").write_text(json.dumps(results, indent=2))
    (HERE / "full_collection_result.json").write_text(
        json.dumps(aggregate.remote(), indent=2)
    )
