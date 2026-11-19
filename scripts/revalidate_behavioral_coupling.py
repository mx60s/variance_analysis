"""Audit HD–MD coupling using the curated runner's aligned time-bin convention."""

import sys, json, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.aligned_analysis import (
    coalesce_tracking_timestamps,
    align_tracking,
    average_in_bins,
)
from src.rodent_data import get_md_from_xy
from regenerate_biology import sessions


def main(out):
    rows = []
    hist = np.zeros(36, dtype=int)
    total = within = 0
    independent_weighted = 0.0
    for kind, d, r in sessions():
        if kind != "dusz":
            continue
        with np.load(
            ROOT / f"results/review_20261116/revalidation/curated_inputs/{r}.npz"
        ) as z:
            pos = z["positions"]
            spikes = z["spikes"]
        pos, _ = coalesce_tracking_timestamps(pos)
        dt = np.diff(pos[:, 0])
        velocity = np.linalg.norm(np.gradient(pos[:, 1:3], pos[:, 0], axis=0), axis=1)
        bad = (
            np.r_[False, dt > 1.5 * np.median(dt)]
            | np.r_[dt > 1.5 * np.median(dt), False]
        )
        velocity[bad] = np.nan
        good = np.isfinite(velocity)
        den = gaussian_filter1d(good.astype(float), 3, mode="nearest")
        num = gaussian_filter1d(np.where(good, velocity, 0), 3, mode="nearest")
        velocity = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        velocity[bad] = np.nan
        edges = np.arange(spikes[0, 0], spikes[-1, 0] + 0.05, 0.05)
        aligned, idx, valid = align_tracking(pos, edges)
        vel = average_in_bins(pos[:, 0], velocity, edges)
        md = (get_md_from_xy(pos, smoothing_window=20) + 90) % 360
        delta = (aligned[:, 3] - md[idx] + 180) % 360 - 180
        v = valid & np.isfinite(delta) & np.isfinite(vel) & (vel >= 1)
        angles = delta[v]
        headings = aligned[v, 3] % 360
        movements = np.sort(md[idx][v] % 360)
        extended = np.r_[movements - 360, movements, movements + 360]
        counts = np.searchsorted(
            extended, headings + 30, side="right"
        ) - np.searchsorted(extended, headings - 30, side="left")
        independent = float(counts.mean() / len(movements))
        independent_weighted += independent * len(angles)
        n = len(angles)
        k = int((abs(angles) <= 30).sum())
        total += n
        within += k
        hist += np.histogram(angles, np.linspace(-180, 180, 37))[0]
        rows.append(
            dict(
                session=r,
                n_bins=n,
                seconds=n * 0.05,
                within_30_percent=100 * k / n,
                marginal_independence_reference_percent=100 * independent,
                mean_cosine=float(np.cos(np.deg2rad(angles)).mean()),
            )
        )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "behavioral_coupling.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = dict(
        sessions=len(rows),
        n_bins=total,
        pooled_within_30_percent=100 * within / total,
        uniform_reference_percent=100 / 6,
        marginal_independence_reference_percent=100 * independent_weighted / total,
        angular_histogram_counts=hist.tolist(),
        angular_bin_edges=np.linspace(-180, 180, 37).tolist(),
        policy="Curated clockwise-from-+Y HD and MD; same nearest-center alignment and velocity preprocessing as curated EV; 50-ms bins; speed >=1 cm/s; gaps excluded",
    )
    (out / "behavioral_coupling.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
