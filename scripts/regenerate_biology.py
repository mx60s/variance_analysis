"""Regenerate biological EV and held-out decoding with explicit time alignment.

Retains original bin sizes, smoothing, cell selection and velocity thresholds.
EV compares MD lags 0/1/3 without wraparound. Saves numerical outputs and
provenance, not bulky posterior arrays or duplicate dense rate-map archives.
"""

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits
from src.rodent_data import (
    load_pfeiffer_data,
    load_dusz_data,
    build_spike_rates,
    compute_velocity_series,
    get_md_from_xy,
)
from src.aligned_analysis import (
    coalesce_tracking_timestamps,
    align_tracking,
    average_in_bins,
    lag_without_wrap,
    labels,
    grouped_maps,
    poisson_argmax,
)
from src.explained_variance import nan_aware_gaussian_filter
from src.decoding import make_cv_masks, gaussian_filter_hd
from regenerate_ev import score, summarize
from reclassify import sha256


def sessions():
    for r in range(1, 5):
        yield "pfeiffer", 0, str(r)
    for p in sorted(
        (ROOT / "results/rodents/dusz").glob("Dataset_*/*/fold_rate_xyhd_20x20x90.npz")
    ):
        yield "dusz", int(p.parts[-3].split("_")[-1]), p.parent.name


def duplication_audit(pos, edges, xe, ye, he, fast, nearest=False):
    idx = np.searchsorted(edges, pos[:, 0], side="right") - 1
    if nearest:
        right = np.clip(np.searchsorted(edges, pos[:, 0]), 1, len(edges) - 1)
        left = right - 1
        idx = np.where(
            abs(pos[:, 0] - edges[left]) <= abs(edges[right] - pos[:, 0]), left, right
        )
    x, y, h = labels(pos[:, 1], xe), labels(pos[:, 2], ye), labels(pos[:, 3], he)
    v = (idx >= 0) & (idx < len(edges) - 1) & (x >= 0) & (y >= 0) & (h >= 0)
    idx = idx[v]
    xy = (y * (len(xe) - 1) + x)[v]
    h = h[v]
    T = len(edges) - 1
    nx = np.bincount(
        np.unique(idx * (len(xe) * len(ye)) + xy) // (len(xe) * len(ye)), minlength=T
    )
    nh = np.bincount(np.unique(idx * (len(he) - 1) + h) // (len(he) - 1), minlength=T)
    assigned = nx * nh
    eligible = fast & (assigned > 0)
    return dict(
        n_eligible=int(eligible.sum()),
        fraction_multiple_states=float(np.mean(assigned[eligible] > 1)),
        mean_state_assignments=float(np.mean(assigned[eligible])),
        max_state_assignments=int(assigned[eligible].max()),
    )


def curves(rates, lab, valid, nstates):
    idx = np.flatnonzero(valid & (lab >= 0))
    occ = np.bincount(lab[idx], minlength=nstates)
    sums = np.zeros((nstates, rates.shape[1]))
    np.add.at(sums, lab[idx], rates[idx])
    maps = np.divide(
        sums, occ[:, None], out=np.full_like(sums, np.nan), where=occ[:, None] > 0
    )
    return maps, occ


def decoding(kind, rates, pos, md, valid, vel, tau, xe, ye, output, xy_unit=None):
    threshold = 5 if kind == "pfeiffer" else 2
    he = np.linspace(0, 360 + 1e-6, 361)
    nx = len(xe) - 1
    ny = len(ye) - 1
    lx, ly = labels(pos[:, 1], xe), labels(pos[:, 2], ye)
    xy = np.where((lx >= 0) & (ly >= 0), ly * nx + lx, -1)
    labs = {"xy": xy, "hd": labels(pos[:, 3], he)}
    if md is not None:
        labs["md"] = labels(md, he)
    good = valid & np.isfinite(vel) & (vel >= threshold)
    foldresults = []
    arrays = {}
    start = time.time()
    with threadpool_limits(limits=1):
        for fold, train in enumerate(make_cv_masks(len(rates), 4)):
            test = np.flatnonzero(~train)
            counts = rates[test] * tau
            if kind == "dusz":
                counts = gaussian_filter1d(counts, 0.1 / tau, axis=0, mode="nearest")
            for axis, lab in labs.items():
                nstates = nx * ny if axis == "xy" else 360
                m, occ = curves(rates, lab, good & train, nstates)
                if kind == "dusz":
                    m[occ < 10] = np.nan
                if axis != "xy":
                    m = gaussian_filter_hd(m, 2 if kind == "pfeiffer" else 3)
                elif kind == "pfeiffer":
                    m = nan_aware_gaussian_filter(
                        m.reshape(ny, nx, -1), (2, 2, 0), ()
                    ).reshape(nstates, -1)
                eligible = good[test] & (lab[test] >= 0)
                ti = test[eligible]
                sc = counts[eligible]
                pred = poisson_argmax(
                    m, sc, tau, offset=1e-12 if kind == "dusz" and axis != "xy" else 0
                )
                baseline = np.full(len(ti), int(np.argmax(occ)))
                if axis == "xy":
                    centers = np.stack(
                        np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2), -1
                    ).reshape(-1, 2)
                    err = np.linalg.norm(
                        centers[np.maximum(pred, 0)] - pos[ti, 1:3], axis=1
                    )
                    baseerr = np.linalg.norm(centers[baseline] - pos[ti, 1:3], axis=1)
                else:
                    centers = (he[:-1] + he[1:]) / 2
                    truth = pos[ti, 3] if axis == "hd" else md[ti]
                    err = np.abs(
                        (centers[np.maximum(pred, 0)] - truth + 180) % 360 - 180
                    )
                    baseerr = np.abs((centers[baseline] - truth + 180) % 360 - 180)
                err[pred < 0] = np.nan
                arrays[f"{axis}_fold{fold}_errors"] = err
                arrays[f"{axis}_fold{fold}_baseline"] = baseerr
                arrays[f"{axis}_fold{fold}_timebins"] = ti
                foldresults.append(
                    dict(
                        axis=axis,
                        fold=fold,
                        n_test=len(ti),
                        n_decoded=int(np.isfinite(err).sum()),
                        median_error=float(np.nanmedian(err)),
                        median_baseline=float(np.median(baseerr)),
                        unit=(
                            xy_unit
                            or (
                                "cm"
                                if kind == "pfeiffer"
                                else "legacy coordinate units"
                            )
                        )
                        if axis == "xy"
                        else "degrees",
                    )
                )
            print("decode fold", fold, round(time.time() - start, 1), flush=True)
    np.savez_compressed(output / "decoding.npz", **arrays)
    (output / "decoding.json").write_text(json.dumps(foldresults, indent=2))


def run(
    kind,
    d,
    r,
    output,
    no_decoding=False,
    fine=False,
    native=False,
    hdmd_bins=90,
    curated_dir=None,
):
    dest = output / f"{kind}_{d}_{r}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        return
    print("SESSION", kind, d, r, flush=True)
    t = time.time()
    if kind == "pfeiffer":
        source = ROOT / f"data/rodents/pfeiffer/Rat{r}/Open1"
        pos, spikes, ids = load_pfeiffer_data(source)
        vseries = compute_velocity_series(pos)
        tau = 0.25
    elif curated_dir is not None:
        source = Path(curated_dir)
        z = np.load(source / (r + ".npz"))
        spikes, pos, ids = z["spikes"], z["positions"], z["neuron_ids"]
        tau = 0.05
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
        vseries = np.c_[pos[:, 0], velocity]
    else:
        source = ROOT / f"data/rodents/dusz/Dataset_{d}/{r}/Data"
        spikes, pos, cell, vseries = load_dusz_data(ROOT / "data/rodents/dusz", d, r)
        ids = np.flatnonzero((cell["hd"] == 1) & (cell["ex"] == 1))
        tau = 0.05
    original_tau = tau
    if fine:
        tau = 0.05 if kind == "pfeiffer" else 0.01
    xy_floor = round(5 * original_tau / tau)
    md_floor = round(10 * original_tau / tau)
    pos, duplicate_timestamps = coalesce_tracking_timestamps(pos)
    edges = np.arange(spikes[0, 0], spikes[-1, 0] + tau, tau)
    rates = build_spike_rates(spikes, edges, tau, ids)
    vel = average_in_bins(vseries[:, 0], vseries[:, 1], edges)
    aligned, ti, valid = align_tracking(pos, edges)
    nhead = 60 if kind == "pfeiffer" else 90
    bounds = (
        [(0, 201), (0, 201)]
        if kind == "pfeiffer"
        else [
            (float(np.nanmin(pos[:, j])), float(np.nanmax(pos[:, j]))) for j in [1, 2]
        ]
    )
    xe, ye = [np.linspace(a, b + 1e-6, 21) for a, b in bounds]
    he = np.linspace(0, 360 + 1e-6, nhead + 1)
    x, y, h = (
        labels(aligned[:, 1], xe),
        labels(aligned[:, 2], ye),
        labels(aligned[:, 3], he),
    )
    row = np.where((x >= 0) & (y >= 0), y * 20 + x, -1)
    fast = vel >= (5 if kind == "pfeiffer" else 2)
    qa = duplication_audit(pos, edges, xe, ye, he, fast, nearest=kind == "pfeiffer")
    qa.update(
        native_xy_grid=native,
        hdmd_bins=hdmd_bins,
        tau=tau,
        xy_min_occupancy_samples=xy_floor,
        hdmd_min_occupancy_samples=md_floor,
        duplicate_tracking_timestamps_merged=duplicate_timestamps,
        n_timebins=len(rates),
        n_aligned=int(valid.sum()),
        position_bounds=bounds,
        tracking_dt=float(np.median(np.diff(pos[:, 0]))),
        selected_cell_ids=ids.tolist(),
    )
    if not (dest / "xyhd.npz").exists():
        a, b, o = grouped_maps(rates, row, h, 400, nhead, valid & fast, floor=xy_floor)
        res = score(
            a,
            b,
            o,
            spatial=True,
            sigma=(1, 2) if kind == "pfeiffer" else (1, 1),
            zoom=(1, 1) if native else None,
        )
        np.savez_compressed(dest / "xyhd.npz", **res)
        (dest / "xyhd.json").write_text(json.dumps(summarize(res), indent=2))
        print("XYHD", summarize(res), flush=True)
    md = None
    if kind == "dusz":
        rawmd = get_md_from_xy(pos, smoothing_window=20)
        if curated_dir is not None:
            rawmd = (rawmd + 90) % 360  # NWB: clockwise from +Y, not clockwise from +X.
        md = rawmd[ti]
        md[~valid] = np.nan
        for lag in [0, 1, 3]:
            target = dest / f"hdmd_lag{lag}.npz"
            if target.exists():
                continue
            shifted = lag_without_wrap(rawmd, lag)[ti]
            shifted[~valid] = np.nan
            hm = labels(aligned[:, 3], np.linspace(0, 360 + 1e-6, hdmd_bins + 1))
            mm = labels(shifted, np.linspace(0, 360 + 1e-6, hdmd_bins + 1))
            a, b, o = grouped_maps(
                rates, hm, mm, hdmd_bins, hdmd_bins, valid & (vel >= 1), floor=md_floor
            )
            res = score(a, b, o, spatial=False, sigma=(hdmd_bins / 90, hdmd_bins / 90))
            np.savez_compressed(target, **res)
            target.with_suffix(".json").write_text(json.dumps(summarize(res), indent=2))
            print("HDMD", lag, summarize(res), flush=True)
    if not no_decoding and not (dest / "decoding.json").exists():
        n = 100 if kind == "pfeiffer" else 32
        dx, dy = [np.linspace(a, b + 1e-6, n + 1) for a, b in bounds]
        decoding(
            kind,
            rates,
            aligned,
            md,
            valid,
            vel,
            tau,
            dx,
            dy,
            dest,
            xy_unit="cm"
            if kind == "pfeiffer" or curated_dir is not None
            else "legacy coordinate units",
        )
    qa.update(
        seconds=time.time() - t,
        source_hashes={
            p.name: sha256(p) for p in source.iterdir() if p.suffix in [".mat", ".csv"]
        },
        code_hashes={
            f: sha256(ROOT / f)
            for f in [
                "scripts/regenerate_biology.py",
                "src/aligned_analysis.py",
                "src/rodent_data.py",
                "src/ev_summary.py",
            ]
        },
        decoding_ran=not no_decoding,
        alignment="nearest tracking sample to spike-bin center; max gap 1.5 median tracking steps",
        md_lags_samples=[0, 1, 3] if kind == "dusz" else [],
        seed=0,
    )
    if curated_dir is not None:
        qa["curated_input"] = json.loads((source / (r + ".json")).read_text())
        qa["velocity_policy"] = (
            "Norm of centered XY derivative, Gaussian sigma 3 tracking samples; cm/s"
        )
        qa["md_reference"] = "Clockwise from +Y, matching NWB head direction"
    (dest / "complete.json").write_text(json.dumps(qa, indent=2))
    print("DONE", kind, d, r, round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--no-decoding", action="store_true")
    a.add_argument("--fine-time-bins", action="store_true")
    a.add_argument("--native-grid", action="store_true")
    a.add_argument("--hdmd-bins", type=int, default=90)
    args = a.parse_args()
    for kind, d, r in sessions():
        run(
            kind,
            d,
            r,
            args.output,
            args.no_decoding,
            args.fine_time_bins,
            args.native_grid,
            args.hdmd_bins,
        )
