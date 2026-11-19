"""Fetch official Minecraft assets over HTTPS and verify their SHA-1 identifiers."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import sys

root = Path(sys.argv[1])
assets = root / "gradle/caches/minecraft/assets"
index = assets / "indexes/1.11.json"
if not index.exists():
    version_path = root / "gradle/caches/minecraft/versionJsons/1.11.2.json"
    if not version_path.exists():
        versions = json.loads(
            urllib.request.urlopen(
                "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json",
                timeout=30,
            ).read()
        )
        entry = next(v for v in versions["versions"] if v["id"] == "1.11.2")
        raw = urllib.request.urlopen(entry["url"], timeout=30).read()
        if hashlib.sha1(raw).hexdigest() != entry["sha1"]:
            raise ValueError("Version metadata hash mismatch")
        version_path.parent.mkdir(parents=True, exist_ok=True)
        version_path.write_bytes(raw)
    version = json.loads(version_path.read_text())
    data = urllib.request.urlopen(version["assetIndex"]["url"], timeout=30).read()
    if hashlib.sha1(data).hexdigest() != version["assetIndex"]["sha1"]:
        raise ValueError("Asset index hash mismatch")
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes(data)
objects = json.loads(index.read_text())["objects"]


def fetch(item):
    h = item["hash"]
    target = assets / "objects" / h[:2] / h
    if target.exists() and hashlib.sha1(target.read_bytes()).hexdigest() == h:
        return 0
    raw = urllib.request.urlopen(
        "https://resources.download.minecraft.net/" + h[:2] + "/" + h, timeout=60
    ).read()
    if len(raw) != item["size"] or hashlib.sha1(raw).hexdigest() != h:
        raise ValueError("Asset hash mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return len(raw)


with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    total = sum(pool.map(fetch, objects.values()))
print("Verified assets", len(objects), "downloaded bytes", total, flush=True)
