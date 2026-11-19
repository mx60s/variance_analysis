"""Fetch only curated NWB behavior and selected spike times through HTTP ranges.

Never downloads raw ephys. Requires h5py/fsspec/aiohttp/certifi. Public DANDI
release 000939/0.261016.2310; outputs carry asset IDs and source metadata.
"""

import argparse, json, ssl, urllib.request, hashlib, re, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import certifi, fsspec, h5py, numpy as np

VERSION = "0.261016.2310"


def fetch(job):
    asset, out = job
    out = Path(out)
    stem = Path(asset["path"]).name
    session = re.search(r"sub-(.*?)_ses-(.*?)_behavior", stem)
    name = session[1] + "-" + session[2]
    target = out / (name + ".npz")
    if target.exists():
        return name, "cached"
    ctx = ssl.create_default_context(cafile=certifi.where())
    t = time.time()
    with urllib.request.urlopen(
        f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/",
        context=ctx,
        timeout=60,
    ) as f:
        meta = json.load(f)
    url = next(u for u in meta["contentUrl"] if "s3.amazonaws.com" in u)
    fs = fsspec.filesystem("https", ssl=ctx)
    with fs.open(url, "rb", block_size=1024 * 1024, cache_type="blockcache") as stream:
        with h5py.File(stream, "r") as f:
            ep = f["intervals/epochs"]
            ends = ep["tags_index"][:]
            starts = np.r_[0, ends[:-1]]
            tags = ep["tags"][:]
            selected = [
                i
                for i, (a, b) in enumerate(zip(starts, ends))
                if b"wake_square" in tags[a:b]
            ]
            if len(selected) != 1:
                raise ValueError(f"{name}: expected one square epoch, found {selected}")
            e = selected[0]
            begin, end = float(ep["start_time"][e]), float(ep["stop_time"][e])
            p = f["processing/behavior/Position/position"]
            h = f["processing/behavior/CompassDirection/head-direction"]
            pt = p["timestamps"][:]
            ht = h["timestamps"][:]
            xy = p["data"][:]
            hd = h["data"][:]
            if not np.array_equal(pt, ht):
                raise ValueError("Position/heading clock mismatch")
            if (
                p["data"].attrs["unit"] != "centimeters"
                or h["data"].attrs["unit"] != "radians"
            ):
                raise ValueError("Unexpected NWB units")
            mask = (pt > begin) & (pt < end)
            positions = np.c_[pt[mask], xy[mask], np.rad2deg(hd[mask]) % 360]
            units = f["units"]
            ids = units["id"][:]
            select = units["is_excitatory"][:].astype(bool) & units[
                "is_head_direction"
            ][:].astype(bool)
            indices = units["spike_times_index"][:]
            starts = np.r_[0, indices[:-1]]
            spikes = []
            for i in np.flatnonzero(select):
                ts = units["spike_times"][starts[i] : indices[i]]
                ts = ts[(ts > begin) & (ts < end)]
                spikes.append(np.c_[ts, np.full(len(ts), ids[i])])
            spikes = np.concatenate(spikes)
            spikes = spikes[np.argsort(spikes[:, 0])]
            provenance = {
                "dandiset": "000939",
                "version": VERSION,
                "asset_id": asset["asset_id"],
                "path": asset["path"],
                "digest": meta["digest"],
                "square_epoch": [begin, end],
                "position_unit": str(p["data"].attrs["unit"]),
                "heading_unit": "degrees (converted from radians)",
                "heading_reference": str(h["reference_frame"][()]),
                "position_reference": str(p["reference_frame"][()]),
                "n_selected": int(select.sum()),
                "seconds": time.time() - t,
                "subset_policy": "tracking and excitatory HD-cell spike times only; no raw ephys",
            }
    with open(target.with_suffix(".tmp"), "wb") as f:
        np.savez_compressed(
            f, positions=positions, spikes=spikes, neuron_ids=ids[select]
        )
    target.with_suffix(".tmp").replace(target)
    provenance["subset_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    return name, provenance["n_selected"], len(positions), round(time.time() - t, 1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--workers", type=int, default=3)
    args = a.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = json.loads(args.manifest.read_text())["results"]
    with ProcessPoolExecutor(args.workers) as pool:
        tasks = {pool.submit(fetch, (x, str(args.output))): x["path"] for x in assets}
        errors = []
        for task in as_completed(tasks):
            try:
                print("FETCHED", task.result(), flush=True)
            except Exception as e:
                errors.append((tasks[task], str(e)))
                print("ERROR", tasks[task], repr(e), flush=True)
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
