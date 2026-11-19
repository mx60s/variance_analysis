from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App("variance-minecraft-render")
volume = modal.Volume.from_name("variance-minecraft-render", create_if_missing=True)
image = (
    modal.Image.from_registry("ubuntu:18.04", add_python="3.12")
    .apt_install(
        "openjdk-8-jdk",
        "python3.6",
        "python3-pip",
        "libpython3.6",
        "xvfb",
        "xauth",
        "libgl1-mesa-dri",
        "libglu1-mesa",
        "libxcursor1",
        "libxrandr2",
        "libasound2",
        "unzip",
        "curl",
        "ca-certificates",
        "libboost-all-dev",
    )
    .add_local_file(HERE / "malmo.zip", "/tmp/malmo.zip", copy=True)
    .run_commands(
        "mkdir -p /opt/malmo && unzip -q /tmp/malmo.zip -d /opt/malmo",
        "python3.6 -m pip install --index-url https://pypi.org/simple numpy==1.19.5",
        "find /opt/malmo -name gradlew -exec chmod +x {} \\;",
    )
)
image = image.add_local_file(HERE / "download_assets.py", "/download_assets.py")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/persist": volume},
)
def setup():
    import subprocess, os, pathlib, shutil, tarfile, time

    root = next(pathlib.Path("/opt/malmo").iterdir())
    work = pathlib.Path("/tmp/buildroot")
    work.mkdir()
    if pathlib.Path("/persist/build-cache.tar").exists():
        with tarfile.open("/persist/build-cache.tar") as t:
            t.extractall(work, filter="data")
    else:
        if pathlib.Path("/persist/gradle").exists():
            shutil.copytree("/persist/gradle", work / "gradle")
        shutil.copytree(root / "Minecraft", work / "Minecraft")
        shutil.copytree(root / "Schemas", work / "Schemas")
    mc = work / "Minecraft"
    p = mc / "build.gradle"
    s = (
        p.read_text()
        .replace("jcenter()", "mavenCentral()")
        .replace(
            "http://files.minecraftforge.net/maven", "https://maven.minecraftforge.net"
        )
    )
    p.write_text(s)
    env = dict(
        os.environ,
        GRADLE_USER_HOME=str(work / "gradle"),
        JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64",
        GRADLE_OPTS="-Xmx3g -XX:ActiveProcessorCount=4",
    )
    subprocess.run(["python", "/download_assets.py", str(work)], check=True)
    with open("/tmp/setup.log", "w") as f:
        proc = subprocess.Popen(
            ["./gradlew", "setupDecompWorkspace", "build", "--stacktrace"],
            cwd=mc,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        try:
            code = proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code = 124
    shutil.copyfile("/tmp/setup.log", "/persist/setup.log")
    with tarfile.open("/persist/build-cache.tmp.tar", "w") as t:
        for name in ["gradle", "Minecraft", "Schemas"]:
            t.add(work / name, arcname=name)
    pathlib.Path("/persist/build-cache.tmp.tar").replace("/persist/build-cache.tar")
    volume.commit()
    result = {
        "returncode": code,
        "tail": pathlib.Path("/tmp/setup.log").read_text()[-18000:],
    }
    print(result)
    return result


@app.local_entrypoint()
def main():
    import json

    (HERE / "render_setup_result.json").write_text(json.dumps(setup.remote(), indent=2))
