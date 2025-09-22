#!/usr/bin/env python3

"""
Head-direction–only Bayesian decoder (Pfeiffer-format data)

Changes vs. your original X–Y–HD version:
- Builds *only* head-direction tuning curves (H×N), i.e., the HD marginal.
- Optional cross-validation is retained; no spatial binning anywhere.
- Decodes HD only; saves decoded_hd, posterior, time_centers (no x/y outputs).
- Uses circular Gaussian smoothing (wrap) on HD curves with NaN-aware masking.
"""

import argparse
import os
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

# ---------------------------- Utilities ---------------------------- #

def compute_velocity(locations: np.ndarray, base_time: np.ndarray) -> np.ndarray:
    """Compute mean velocity (cm/s) within each time bin.

    locations: Pfeiffer Position_Data, columns [time, x, y, hd_deg]
    base_time: 1D array of bin edges
    Returns: (len(base_time)-1,) array of avg velocity per bin
    """
    if locations.shape[0] < 4:
        raise ValueError("locations must have >= 4 rows to estimate time step.")
    time_step = float(locations[3, 0] - locations[2, 0])
    # per-sample displacement magnitude (cm)
    steps = np.diff(locations[:, 1:3], axis=0)
    steps = np.vstack((steps, steps[-1]))  # pad last to align length
    speed_samples = np.linalg.norm(steps, axis=1) / max(time_step, 1e-9)

    v = np.zeros(len(base_time) - 1, dtype=float)
    for t in range(len(base_time) - 1):
        idx = np.where((locations[:, 0] >= base_time[t]) & (locations[:, 0] < base_time[t + 1]))[0]
        v[t] = 0.0 if idx.size == 0 else float(np.mean(speed_samples[idx]))
    return v


def _gaussian_filter_hd(rate_curve: np.ndarray, smooth_sigma: float) -> np.ndarray:
    """NaN-aware circular 1D Gaussian smoothing across head-direction bins.

    rate_curve: (H, N) or (H,) array; smoothing is along the first axis (HD).
    """
    if smooth_sigma is None or smooth_sigma <= 0:
        return rate_curve
    V = rate_curve.copy()
    isnan = np.isnan(V)
    V[isnan] = 0.0
    U = np.ones_like(rate_curve, dtype=float)
    U[isnan] = 0.0
    VV = gaussian_filter1d(V, sigma=smooth_sigma, axis=0, mode="wrap")
    UU = gaussian_filter1d(U, sigma=smooth_sigma, axis=0, mode="wrap")
    with np.errstate(invalid='ignore', divide='ignore'):
        out = VV / UU
    out[UU <= 0] = np.nan
    return out


def _select_time_indices_in_hd_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    hd_l: Tuple[float, float],
    fast_bins: np.ndarray,
    timebin_tol: float = 1e-4,
) -> np.ndarray:
    """Return array of *time-bin indices* whose samples fall in the given HD bin,
    optionally filtered by high-velocity bins (fast_bins contains base_time edges).
    """
    l0, l1 = hd_l
    in_hd = np.where((locations[:, 3] >= l0) & (locations[:, 3] < l1))[0]
    if in_hd.size == 0:
        return np.array([], dtype=int)

    loc_times = locations[in_hd, 0]
    # map each sample time to the nearest base_time index (edge); later we guard range
    loc_idx = np.array([int(np.abs(base_time - t).argmin()) for t in loc_times], dtype=int)

    # high-velocity mask via time matching (tolerance to avoid float issues)
    if fast_bins is not None and fast_bins.size:
        mask = np.any(np.isclose(base_time[loc_idx, None], fast_bins, atol=timebin_tol), axis=1)
        loc_idx = loc_idx[mask]

    # keep only indices that map into spike-rate rows [0..T-1]
    T = len(base_time) - 1
    return loc_idx[(loc_idx >= 0) & (loc_idx < T)]


# ------------------ HD tuning curves (marginal over XY) ------------- #

def get_hd_rate_maps_cv(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_hd_bins: int = 60,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
    cv_count: int = 4,
    seed: int = 777,
):
    """Build cross-validated HD tuning curves (H×N), plus occupancy (H).

    Returns
    -------
    occupancy_maps : list length cv_count of arrays (H,)
    cv_masks : list length cv_count of boolean arrays, True = train, False = test
    hd_edges : (H+1,) bin edges in degrees
    """
    rng = np.random.RandomState(seed)

    # velocity filtering
    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if filter_vel else None
    print(
        f"Filtering {100.0 - (len(fast_idx) / max(len(base_time)-1,1)) * 100.0:.1f}% of bins due to low velocity"
    )

    # HD bins (0..360)
    hd_edges = np.linspace(0.0, 360.0 + 1e-2, n_hd_bins + 1)

    T, N = spike_rates.shape
    assert T == len(base_time) - 1, "spike_rates must match len(base_time)-1"

    # CV masks (on time-bin indices)
    n_points = T
    folds = np.array_split(np.arange(n_points), cv_count)
    cv_masks: List[np.ndarray] = []
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False  # False = test for this fold
        cv_masks.append(mask)

    # outputs
    shape_map = (n_hd_bins, N)
    rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps    = [np.zeros((n_hd_bins,), dtype=float) for _ in range(cv_count)]

    print("Creating cross-validation HD tuning curves…")
    for l in tqdm(range(n_hd_bins), desc="HD bins"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_bins)
        if idx.size == 0:
            continue

        # store occupancy before any CV splitting
        for cv in range(cv_count):
            occupancy_maps[cv][l] = float(idx.size)

        # shuffle then split into two folds for EV calcs
        idx_shuf = idx.copy()
        rng.shuffle(idx_shuf)
        half = len(idx_shuf) // 2
        fold0 = idx_shuf[:half]
        fold1 = idx_shuf[half:]

        # compute per-CV maps
        for cv in range(cv_count):
            mask_cv = cv_masks[cv]
            valid = idx_shuf[idx_shuf < mask_cv.size]
            tr = valid[mask_cv[valid]]
            te = valid[~mask_cv[valid]]

            if tr.size:
                global_train_maps[cv][l, :] = np.nanmean(spike_rates[tr, :], axis=0)
            if te.size:
                global_test_maps[cv][l, :] = np.nanmean(spike_rates[te, :], axis=0)

            # folded (for EV)
            for fnum, fidx in enumerate([fold0, fold1]):
                if fidx.size == 0:
                    continue
                fvalid = fidx[fidx < mask_cv.size]
                ftr = fvalid[mask_cv[fvalid]]
                fte = fvalid[~mask_cv[fvalid]]
                if ftr.size:
                    folded_train_maps[cv][l, fnum, :] = np.nanmean(spike_rates[ftr, :], axis=0)
                if fte.size:
                    folded_test_maps[cv][l, fnum, :] = np.nanmean(spike_rates[fte, :], axis=0)

    return (
        global_train_maps,
        global_test_maps,
        folded_train_maps,
        folded_test_maps,
        occupancy_maps,
        cv_masks,
        hd_edges,
    )


def get_hd_rate_map_all(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_hd_bins: int = 60,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
):
    """Build a single (all-data) HD tuning curve (H, N), plus occupancy (H)."""
    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if filter_vel else None

    hd_edges = np.linspace(0.0, 360.0 + 1e-2, n_hd_bins + 1)

    T, N = spike_rates.shape
    rate_map = np.full((n_hd_bins, N), np.nan, dtype=float)
    occupancy = np.zeros((n_hd_bins,), dtype=float)

    print("Creating all-data HD tuning curve…")
    for l in tqdm(range(n_hd_bins), desc="HD bins"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_bins)
        if idx.size:
            occupancy[l] = float(idx.size)
            rate_map[l, :] = np.nanmean(spike_rates[idx, :], axis=0)
    return rate_map, occupancy, hd_edges


# ----------------------------- Decoding ----------------------------- #

def hd_bayesian_decoding(
    rate_map_hd: np.ndarray,
    spike_counts: np.ndarray,
    tau: float,
    hd_bins: np.ndarray,
    avg_vel: np.ndarray,
    vel_threshold: float,
    eps: float = 1e-10,
):
    """Head-direction Bayesian decoding with Poisson likelihood.

    rate_map_hd : (H, N) — mean firing rate (Hz)
    spike_counts : (T, N) — counts per bin (not rates)
    avg_vel : (T,) — cm/s
    Returns: decoded_hd (T,), list_of_posteriors (each (H,) or NaNs)
    """
    T = spike_counts.shape[0]
    H, N = rate_map_hd.shape

    decoded_hd = np.full(T, np.nan)
    posterior_all: List[np.ndarray] = []

    print(f"Decoding {T} time bins using {N} neurons…")
    rate_tau = rate_map_hd * tau  # expected spike count per bin

    for t in tqdm(range(T), desc="Decoding time bins"):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((H,), np.nan))
            continue

        k = spike_counts[t][None, :]  # (1,N)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k * np.log(rate_tau + eps) - rate_tau, axis=1)  # (H,)

        if np.all(np.isnan(log_like)):
            posterior_all.append(np.full((H,), np.nan))
            continue

        m = np.nanmax(log_like)
        post = np.exp(log_like - m)
        post_sum = np.nansum(post)
        if post_sum <= 0 or not np.isfinite(post_sum):
            posterior_all.append(np.full((H,), np.nan))
            continue
        post /= post_sum
        posterior_all.append(post)

        h_idx = int(np.nanargmax(post))
        decoded_hd[t] = 0.5 * (hd_bins[h_idx] + hd_bins[h_idx + 1])

    return decoded_hd, posterior_all


# ------------------------- Data Loading / Prep ---------------------- #

def load_pfeiffer_data(data_path: Path):
    """Load Pfeiffer Position and Spike data, returning arrays.
    Returns: spike_times (M,2), locations (P,4), and spike_data dict.
    """
    locations = loadmat(str(data_path / 'Position_Data.mat'))['Position_Data']
    spike_data = loadmat(str(data_path / 'Spike_Data.mat'))
    spike_times = spike_data['Spike_Data']  # columns: [time, neuron_id]
    return spike_times, locations, spike_data


def extract_excitatory_ids(spike_data: dict) -> np.ndarray:
    excit = spike_data['Excitatory_Neurons']
    excit = np.asarray(excit).ravel().astype(spike_data['Spike_Data'].dtype)
    return excit


def build_spike_rates(spike_times: np.ndarray, base_time: np.ndarray, tau: float, excit_list: np.ndarray) -> np.ndarray:
    """Vectorized per-neuron per-bin firing rates using numpy.histogram.
    Returns: spike_rates of shape (T, N_excit)
    """
    T = len(base_time) - 1
    N = len(excit_list)
    rates = np.zeros((T, N), dtype=float)
    for n_idx, neuron_id in enumerate(excit_list):
        neuron_spk = spike_times[spike_times[:, 1] == neuron_id, 0]
        if neuron_spk.size:
            counts, _ = np.histogram(neuron_spk, bins=base_time)
            rates[:, n_idx] = counts.astype(float) / max(tau, 1e-9)
    return rates


# ------------------------------- Main ------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Head-direction–only tuning + Bayesian decoding (Pfeiffer data)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/mag/variance-analysis/data/pfeiffer",
        help="Root data directory (contains Rat{rat}/Open1)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/mag/variance-analysis/results/pfeiffer",
        help="Directory to write outputs",
    )
    parser.add_argument("--rat", type=str, default="1", help="Rat ID (e.g., '1')")

    parser.add_argument("--n_hd_bins", type=int, default=60, help="Number of head-direction bins")
    parser.add_argument("--tau", type=float, default=0.120, help="Time bin size (s)")
    parser.add_argument("--vel_threshold", type=float, default=5.0, help="Velocity threshold (cm/s)")
    parser.add_argument("--cv_count", type=int, default=4, help="Number of CV folds")
    parser.add_argument("--cv_index", type=int, default=-1, help="Fold index (-1 for no CV)")
    parser.add_argument(
        "--filter_vel",
        action="store_true",
        default=True,
        help="Filter low-velocity bins",
    )
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=2.0,
        help="Sigma for Gaussian smoothing of HD curves (in bins; circular)",
    )
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        default=False,
        help="If set, save intermediate artifacts (HD rate maps, masks, etc.)",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path) / f"Rat{args.rat}" / "Open1"
    save_path = Path(args.save_path) / f"Rat{args.rat}"
    os.makedirs(save_path, exist_ok=True)

    print("\n=== Head-Direction–Only Decoding Pipeline ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_path}")
    print(f"HD bins   : {args.n_hd_bins}")
    print(
        f"tau: {args.tau:.3f}s | vel_th: {args.vel_threshold} cm/s | CV folds: {args.cv_count} | cv_index: {args.cv_index}"
    )

    t0 = time.time()

    print("\nLoading data…")
    spike_times_raw, locations, spike_data = load_pfeiffer_data(data_path)
    excit_list = extract_excitatory_ids(spike_data)
    print(
        f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_list)} | Position samples: {locations.shape[0]}"
    )

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + args.tau, args.tau)
    print(
        f"Time bins: {len(base_time)-1} bins covering {base_time[0]:.3f}–{base_time[-1]:.3f}s"
    )

    print("Binning spikes into rates…")
    spike_rates = build_spike_rates(
        spike_times_raw, base_time, args.tau, excit_list
    )  # (T, N)

    if args.cv_index >= 0:
        (
            global_train_maps,
            global_test_maps,
            folded_train_maps,
            folded_test_maps,
            occupancy_maps,
            cv_masks,
            hd_edges,
        ) = get_hd_rate_maps_cv(
            spike_rates=spike_rates,
            locations=locations,
            base_time=base_time,
            n_hd_bins=args.n_hd_bins,
            velocity_threshold=args.vel_threshold,
            filter_vel=args.filter_vel,
            cv_count=args.cv_count,
            seed=args.seed,
        )

        if not (0 <= args.cv_index < args.cv_count):
            raise ValueError(
                f"cv_index must be in [0, {args.cv_count-1}] or -1 for no CV"
            )
        rate_map_hd = global_train_maps[args.cv_index]  # (H,N)
        cv_mask = cv_masks[args.cv_index]  # True=train, False=test (over T bins)
        test_idx = np.where(~cv_mask)[0]
        print(f"Using CV fold {args.cv_index}: decoding {len(test_idx)} held-out bins.")

        # subset spikes/vel for decoding
        spike_counts = (spike_rates[test_idx, :] * args.tau).astype(float)
        avg_vel = compute_velocity(locations, base_time)[test_idx]
        time_centers = 0.5 * (
            base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)]
        )

        # smooth HD curves (circular)
        rate_map_hd = _gaussian_filter_hd(rate_map_hd, args.smooth_sigma)

        if args.save_intermediates:
            np.save(
                save_path
                / f"hd_rate_map_cv{args.cv_index}_{args.n_hd_bins}_global_train.npy",
                rate_map_hd,
            )
            np.save(
                save_path / f"cv_mask{args.cv_index}_{args.n_hd_bins}.npy", cv_mask
            )
            np.save(
                save_path / f"hd_occupancy_map_cv{args.cv_index}_{args.n_hd_bins}.npy",
                occupancy_maps[args.cv_index],
            )

    else:
        rate_map_hd, occupancy_all, hd_edges = get_hd_rate_map_all(
            spike_rates=spike_rates,
            locations=locations,
            base_time=base_time,
            n_hd_bins=args.n_hd_bins,
            velocity_threshold=args.vel_threshold,
            filter_vel=args.filter_vel,
        )
        print("Using ALL-DATA HD tuning; decoding all bins.")
        spike_counts = (spike_rates * args.tau).astype(float)
        avg_vel = compute_velocity(locations, base_time)
        time_centers = 0.5 * (base_time[:-1] + base_time[1:])

        rate_map_hd = _gaussian_filter_hd(rate_map_hd, args.smooth_sigma)

        if args.save_intermediates:
            np.save(
                save_path / f"hd_rate_map_{args.n_hd_bins}.npy",
                rate_map_hd,
            )
            np.save(
                save_path / f"hd_occupancy_map_{args.n_hd_bins}.npy",
                occupancy_all,
            )

    decoded_hd, posterior_all = hd_bayesian_decoding(
        rate_map_hd=rate_map_hd,
        spike_counts=spike_counts,
        tau=args.tau,
        hd_bins=hd_edges,
        avg_vel=avg_vel,
        vel_threshold=args.vel_threshold,
    )

    print("Saving decoding outputs…")
    if args.cv_index >= 0:
        base = f"decoded_hd_cv{args.cv_index}_{args.n_hd_bins}"
    else:
        base = f"decoded_hd_{args.n_hd_bins}"

    np.save(save_path / f"{base}.npy", decoded_hd)
    np.save(save_path / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
    np.save(save_path / f"{base}_time_centers.npy", time_centers)

    # Summary
    ok_hd = decoded_hd[~np.isnan(decoded_hd)]

    print("\nDecoding Summary:")
    print(f"Total time bins: {len(decoded_hd)}")
    print(f"Successfully decoded: {len(ok_hd)} ({100.0 * len(ok_hd) / max(len(decoded_hd),1):.1f}%)")
    if ok_hd.size:
        print(f"HD range: {np.min(ok_hd):.1f} – {np.max(ok_hd):.1f} deg")

    print(f"\nDone in {(time.time()-t0)/60.0:.2f} min.")


if __name__ == "__main__":
    main()
