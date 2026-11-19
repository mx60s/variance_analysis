"""Isolated cached-map EV rerun; saves raw scores, reliability and normalized scores."""

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.ev_summary import quantify_fast
from src.explained_variance import (
    smooth_location_hd,
    nan_aware_gaussian_filter,
    masked_zoom,
)


def score(f0, f1, occ, spatial=True, sigma=(1, 1), zoom=None, cutoff=True):
    if f0.ndim == 4:
        f0 = f0.reshape(-1, *f0.shape[2:])
        f1 = f1.reshape(-1, *f1.shape[2:])
        occ = occ.reshape(f0.shape[:2])
    if zoom is None:
        zoom = (max(f0.shape[:2]) / f0.shape[0], max(f0.shape[:2]) / f0.shape[1])
    oz = masked_zoom(occ, zoom)
    positive = oz[oz > 0]
    minimum = np.percentile(positive / 2, 10) if cutoff and len(positive) else 0
    res = []
    t = time.time()
    with threadpool_limits(limits=1):
        for n in range(f0.shape[-1]):
            # Bounded contiguous blocks prevent neuron-strided mmap reads from
            # repeatedly faulting the entire multi-gigabyte activation archive.
            if n % 128 == 0:
                blocks = [np.array(f[:, :, n : n + 128], copy=True) for f in [f0, f1]]
            maps = [
                smooth_location_hd(f[:, :, n % 128], sigma)
                if spatial
                else nan_aware_gaussian_filter(f[:, :, n % 128], sigma, (0, 1))
                for f in blocks
            ]
            a, b = [masked_zoom(f, zoom) for f in maps]
            res.append(
                [quantify_fast(a, b, oz, axis=ax, min_occ=minimum) for ax in [0, 1]]
            )
            if (n + 1) % 1000 == 0:
                print("score", n + 1, round(time.time() - t, 1), flush=True)
    res = np.asarray(res)
    ev = 1 - res[:, :, 0]
    reliability = res[:, :, 1]
    norm = np.divide(
        reliability - res[:, :, 0],
        reliability,
        out=np.full_like(ev, np.nan),
        where=reliability > 0,
    )
    return dict(
        ev=ev, reliability=reliability, normalized=norm, min_occ=minimum, zoom=zoom
    )


def summarize(d):
    ev = d["ev"]
    v = np.isfinite(ev).all(1)
    return dict(
        n=len(ev),
        n_finite=int(v.sum()),
        median=np.nanmedian(ev, axis=0).tolist(),
        axis0_greater=float(np.mean(ev[v, 0] > ev[v, 1])) if v.any() else None,
        median_reference=np.nanmedian(1 - d["reliability"], axis=0).tolist(),
        median_normalized=np.nanmedian(d["normalized"], axis=0).tolist(),
    )


def cached(out):
    out.mkdir(parents=True, exist_ok=True)
    paths = list(
        (ROOT / "results/rodents/pfeiffer").glob("Rat*/ratemap_xyhd_20x20x60.npz")
    ) + list((ROOT / "results/rodents/dusz").glob("Dataset_*/*/fold_rate_*.npz"))
    for p in sorted(paths):
        if "xyhd" in p.name and "20x20" not in p.name:
            continue
        if "hdmd" in p.name and "90x90" not in p.name:
            continue
        name = "_".join(p.parts[-3:]).removesuffix(".npz")
        target = out / (name + ".npz")
        if target.exists():
            continue
        print("START", name, flush=True)
        with np.load(p) as z:
            occ = z["occupancy"]
            if "fold0_map" in z:
                f0, f1 = z["fold0_map"], z["fold1_map"]
            else:
                f0 = np.load(p.with_name(p.name.replace("ratemap_", "fold0_")))[
                    "rate_map"
                ]
                f1 = np.load(p.with_name(p.name.replace("ratemap_", "fold1_")))[
                    "rate_map"
                ]
        d = score(
            f0,
            f1,
            occ,
            spatial="xyhd" in p.name,
            sigma=(1, 2) if "pfeiffer" in str(p) else (1, 1),
        )
        np.savez_compressed(target, **d)
        summary = summarize(d)
        summary["source"] = str(p)
        target.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(name, summary, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    cached(args.output)
