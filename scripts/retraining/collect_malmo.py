"""Python 3.6-compatible Malmo collector, using render metadata to verify each pose.

This samples a kinematic path, not a real-time physics rollout. Simulation dt is
stored separately from wall-clock render timestamps. Never trains across resets.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--malmo", required=True)
    p.add_argument("--world", required=True)
    p.add_argument("--paths", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.malmo) / "Python_Examples"))
    import MalmoPython

    os.environ["MALMO_XSD_PATH"] = str(Path(a.malmo) / "Schemas")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(a.paths) / "manifest.json").read_text())
    world_files = {
        str(p.relative_to(a.world)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(a.world).rglob("*"))
        if p.is_file()
    }
    manifest.update(
        world_files_sha256=world_files,
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        minecraft_version="1.11.2",
        malmo_version="0.37.0",
        camera_y=4.0,
        camera_pitch=0.0,
        camera_fov_degrees=70,
        render_options={
            "renderDistance": 4,
            "bobView": False,
            "maxFps": 120,
            "fancyGraphics": False,
            "renderClouds": False,
            "particles": 2,
        },
    )

    xml = f"""<Mission xmlns="http://ProjectMalmo.microsoft.com"><About><Summary>Verified smooth heading-aligned trajectories</Summary></About><ModSettings><MsPerTick>16</MsPerTick></ModSettings><ServerSection><ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time><Weather>clear</Weather><AllowSpawning>false</AllowSpawning></ServerInitialConditions><ServerHandlers><FileWorldGenerator src="{a.world}" forceReset="true"/></ServerHandlers></ServerSection><AgentSection mode="Creative"><Name>TrajectoryCollector</Name><AgentStart><Placement x="128" y="4" z="205" yaw="0" pitch="0"/></AgentStart><AgentHandlers><AbsoluteMovementCommands/><ContinuousMovementCommands/><MissionQuitCommands/><ObservationFromFullStats/><VideoProducer want_depth="false"><Width>64</Width><Height>64</Height></VideoProducer></AgentHandlers></AgentSection></Mission>"""
    host = MalmoPython.AgentHost()
    mission = MalmoPython.MissionSpec(xml, True)
    deadline = time.monotonic() + 120
    while True:
        try:
            host.startMission(mission, MalmoPython.MissionRecordSpec())
            break
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)
    while not host.peekWorldState().has_mission_begun:
        if time.monotonic() > deadline:
            raise RuntimeError("Mission did not begin")
        time.sleep(0.1)
    host.sendCommand("move 0")
    host.sendCommand("turn 0")
    host.sendCommand("setPitch 0")
    collected = []
    try:
        for ep in manifest["episodes"][a.start : a.start + a.episodes]:
            target = out / (ep["id"] + ".npz")
            if target.exists():
                ep = dict(ep, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                collected.append(ep)
                continue
            poses = np.load(str(Path(a.paths) / (ep["id"] + ".npy")))
            pixels = []
            rendered = []
            timestamps = []
            wall = []
            for index, (x, z, yaw) in enumerate(poses):
                host.getWorldState()
                host.sendCommand(f"tp {x:.9f} 4.0 {z:.9f}")
                host.sendCommand(f"setYaw {yaw:.9f}")
                deadline = time.monotonic() + 15
                matched = None
                while time.monotonic() < deadline:
                    ws = host.getWorldState()
                    if not ws.is_mission_running:
                        raise RuntimeError("Mission terminated")
                    for frame in ws.video_frames:
                        err = abs((frame.yaw - yaw + 180) % 360 - 180)
                        if (
                            abs(frame.xPos - x) < 1e-4
                            and abs(frame.zPos - z) < 1e-4
                            and abs(frame.yPos - 4) < 1e-4
                            and err < 0.002
                            and abs(frame.pitch) < 0.002
                        ):
                            matched = frame
                    if matched is not None:
                        break
                    time.sleep(0.005)
                if matched is None:
                    raise RuntimeError(
                        "Render pose mismatch episode {} frame {}".format(
                            ep["id"], index
                        )
                    )
                frame = matched
                if (frame.height, frame.width, frame.channels) != (64, 64, 3):
                    raise RuntimeError("Wrong RGB dimensions")
                pixels.append(
                    np.frombuffer(frame.pixels, dtype=np.uint8)
                    .reshape(64, 64, 3)
                    .copy()
                )
                rendered.append([frame.xPos, frame.zPos, frame.yaw])
                timestamps.append(str(frame.timestamp))
                wall.append(time.monotonic())
            rgb = np.stack(pixels)
            if np.std(rgb) < 1 or np.mean(rgb[1:] != rgb[:-1]) < 0.001:
                raise RuntimeError("Blank or frozen rendering")
            tmp = out / (ep["id"] + ".tmp.npz")
            np.savez_compressed(
                str(tmp),
                rgb=rgb,
                state=np.array(rendered),
                requested_state=poses,
                simulation_time=np.arange(len(poses)) * 0.1,
                render_timestamp=np.array(timestamps),
                wall_time=np.array(wall),
            )
            tmp.replace(target)
            ep = dict(ep, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            collected.append(ep)
            progress = dict(
                manifest,
                episodes=collected,
                complete=len(collected) == a.episodes,
                collector="Malmo 0.37.0 / Minecraft 1.11.2, verified rendered poses",
            )
            (out / "manifest.json").write_text(json.dumps(progress, indent=2))
            print(
                "Collected",
                ep["id"],
                rgb.shape,
                "render_seconds",
                round(wall[-1] - wall[0], 2),
                flush=True,
            )
    finally:
        host.sendCommand("quit")


if __name__ == "__main__":
    main()
