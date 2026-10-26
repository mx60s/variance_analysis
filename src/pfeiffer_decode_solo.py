#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Tuple, Optional, Dict, Any

import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from src.rodent_data import load_pfeiffer_data, build_spike_rates, compute_velocity, compute_velocity_series
from src.decoding import (
    make_cv_masks, test_spike_counts, gaussian_filter_hd,
    hd_bayesian_decoding, xy_bayesian_decoding, hd_chance_decoding, xy_chance_decoding,
    compute_chance_error_analytical_hd, compute_chance_error_analytical_xy,
    compute_no_prior_chance_hd, compute_no_prior_chance_xy,
    evaluate_errors_hd, evaluate_errors_xy,
)

# utilities

def gaussian_filter_xy(rate_map: np.ndarray, smooth_sigma_xy: Tuple[float, float]) -> np.ndarray:
    sy, sx = smooth_sigma_xy
    V = rate_map.copy()
    isnan = np.isnan(V)
    V[isnan] = 0.0
    U = np.ones_like(rate_map, dtype=float)
    U[isnan] = 0.0

    if rate_map.ndim == 2:
        VV = gaussian_filter(V, sigma=(sy, sx), mode="nearest")
        UU = gaussian_filter(U, sigma=(sy, sx), mode="nearest")
    else:
        VV = np.empty_like(V)
        UU = np.empty_like(U)
        for n in range(V.shape[-1]):
            VV[..., n] = gaussian_filter(V[..., n], sigma=(sy, sx), mode="nearest")
            UU[..., n] = gaussian_filter(U[..., n], sigma=(sy, sx), mode="nearest")

    with np.errstate(invalid='ignore', divide='ignore'):
        out = VV / UU
    out[UU <= 0] = np.nan
    return out


def _nearest_fast_bins(
    loc_times: np.ndarray,
    base_time: np.ndarray,
    fast_bins: Optional[np.ndarray],
    timebin_tol: float = 1e-4,
) -> np.ndarray:
    """Nearest time-bin edge for each sample, kept only if that edge is a high-velocity bin."""
    loc_idx = np.array([int(np.abs(base_time - t).argmin()) for t in loc_times], dtype=int)

    if fast_bins is not None and fast_bins.size:
        mask = np.any(np.isclose(base_time[loc_idx, None], fast_bins, atol=timebin_tol), axis=1)
        loc_idx = loc_idx[mask]

    T = len(base_time) - 1
    return loc_idx[(loc_idx >= 0) & (loc_idx < T)]


def select_time_indices_in_hd_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    hd_l: Tuple[float, float],
    fast_bins: Optional[np.ndarray],
    timebin_tol: float = 1e-4,
) -> np.ndarray:
    """Return array of time-bin indices whose samples fall in the given HD bin, 
    filtered by high-velocity bins.
    """
    l0, l1 = hd_l
    in_hd = np.where((locations[:, 3] >= l0) & (locations[:, 3] < l1))[0]
    if in_hd.size == 0:
        return np.array([], dtype=int)
    return _nearest_fast_bins(locations[in_hd, 0], base_time, fast_bins, timebin_tol)


def select_time_indices_in_xy_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    x_win: Tuple[float, float],
    y_win: Tuple[float, float],
    fast_bins: Optional[np.ndarray],
    timebin_tol: float = 1e-4,
) -> np.ndarray:
    """Return array of *time-bin indices* whose samples fall in the given XY bin."""
    x0, x1 = x_win
    y0, y1 = y_win
    in_xy = np.where(
        (locations[:, 1] >= x0) & (locations[:, 1] < x1) &
        (locations[:, 2] >= y0) & (locations[:, 2] < y1)
    )[0]
    if in_xy.size == 0:
        return np.array([], dtype=int)
    return _nearest_fast_bins(locations[in_xy, 0], base_time, fast_bins, timebin_tol)


def _fast_bins(locations: np.ndarray, base_time: np.ndarray, velocity_threshold: float) -> Optional[np.ndarray]:
    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    print(
        f"Filtering {100.0 - (len(fast_idx) / max(len(base_time)-1,1)) * 100.0:.1f}% of bins due to low velocity"
    )
    return base_time[fast_idx] if velocity_threshold > 0.0 else None


def get_hd_rate_maps_cv(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_hd_bins: int = 60,
    velocity_threshold: float = 5.0,
    cv_count: int = 4,
):
    fast_bins = _fast_bins(locations, base_time, velocity_threshold)

    hd_edges = np.linspace(0.0, 360.0 + 1e-2, n_hd_bins + 1)

    T, N = spike_rates.shape
    cv_masks = make_cv_masks(T, cv_count)

    shape_map = (n_hd_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps    = [np.zeros((n_hd_bins,), dtype=float) for _ in range(cv_count)]

    print("Creating cross-validation HD tuning curves…")
    for l in tqdm(range(n_hd_bins), desc="HD bins"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_bins)
        if idx.size == 0:
            continue

        for cv in range(cv_count):
            # occupancy counts all (train and test) samples in the bin
            occupancy_maps[cv][l] = float(idx.size)
            
            mask_cv = cv_masks[cv]
            valid = idx[idx < mask_cv.size]
            cv_idx = valid[mask_cv[valid]]
            if cv_idx.size:
                cv_rate_maps[cv][l, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)

    return (
        cv_rate_maps,
        occupancy_maps,
        cv_masks,
        hd_edges,
    )


def get_xy_rate_maps_cv(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_xy_bins: int = 100,
    velocity_threshold: float = 5.0,
    cv_count: int = 4,
):
    fast_bins = _fast_bins(locations, base_time, velocity_threshold)

    xmin, xmax = 0., 201.
    ymin, ymax = 0., 201.
    
    n_x_bins = n_xy_bins
    n_y_bins = n_xy_bins
    
    x_edges = np.linspace(xmin, xmax + 1e-6, n_x_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_y_bins + 1)

    T, N = spike_rates.shape
    cv_masks = make_cv_masks(T, cv_count)

    shape_map = (n_y_bins, n_x_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_y_bins, n_x_bins,), dtype=float) for _ in range(cv_count)]

    print("Creating XY tuning curves")
    for iy in tqdm(range(n_y_bins), desc="Y bins"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx = select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_bins)
            if idx.size == 0:
                continue

            for cv in range(cv_count):
                occupancy_maps[cv][iy, ix] = float(idx.size)

                mask_cv = cv_masks[cv]
                valid = idx[idx < mask_cv.size]
                cv_idx = valid[mask_cv[valid]]
                if cv_idx.size:
                    cv_rate_maps[cv][iy, ix, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)

    return (
        cv_rate_maps,
        occupancy_maps,
        cv_masks,
        (x_edges, y_edges),
    )
    

def decode_solo(
    *,
    data_path: str = "./data/rodents/pfeiffer",
    save_path: str = "./results/rodents/pfeiffer",
    rat: str = "1",
    n_hd_bins: int = 360,
    n_xy_bins: int = 100,
    tau: float = 0.25,
    vel_threshold: float = 5.0,
    cv_folds: int = 4,
    seed: int = 777,
    smooth_sigma_hd: float = 2.0,
    smooth_spikes_sigma_s: float = 0.0,
    save_intermediates: bool = False,
    run_chance_decoding: bool = True,
    chance_mode: str = "mode",
) -> Dict[str, Any]:
    """
    Cross-validated decoding for BOTH head direction (HD) and location (XY).

    Parameters
    ----------
    run_chance_decoding : bool
        If True, also run chance-level decoding based on occupancy alongside
        the neural decoder. This provides a baseline for comparison.
    chance_mode : str
        'mode' - always predict most occupied bin (deterministic)
        'sample' - sample from occupancy distribution (stochastic)

    Returns a summary dict and writes per-fold artifacts to disk.
    """
    if cv_folds is None or int(cv_folds) < 2:
        raise ValueError("cv_folds must be >= 2 for CV-only script.")

    data_path = Path(data_path) / f"Rat{rat}" / "Open1"
    save_path = Path(save_path) / f"Rat{rat}"
    os.makedirs(save_path, exist_ok=True)
    
    smooth_sigma_xy = 2.0 # for 2 cm bins

    print("\n=== Pfeiffer Decoding Pipeline (CV-only): HD + XY ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_path}")
    print(f"HD bins   : {n_hd_bins}")
    print(f"XY bins   : {n_xy_bins}")
    print(f"tau: {tau:.3f}s | vel_th: {vel_threshold} cm/s | CV folds: {cv_folds}")
    print(f"Smoothing: HD σ={smooth_sigma_hd} (bins), XY σ={smooth_sigma_xy} (bins)")
    if smooth_spikes_sigma_s and smooth_spikes_sigma_s > 0:
        print(f"Temporal smoothing of spike counts: σ={smooth_spikes_sigma_s*1000:.0f} ms")
    if run_chance_decoding:
        print(f"Chance decoding: ENABLED (mode='{chance_mode}')")
    else:
        print("Chance decoding: DISABLED")

    t0 = time.time()

    # RNG for stochastic chance decoding
    rng = np.random.default_rng(seed)

    print("\nLoading data…")
    locations, spike_times_raw, excit_list = load_pfeiffer_data(data_path)
    print(
        f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_list)} | Position samples: {locations.shape[0]}"
    )

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + tau, tau)
    print(f"Time bins: {len(base_time)-1} bins covering {base_time[0]:.3f}–{base_time[-1]:.3f}s")

    spike_rates = build_spike_rates(spike_times_raw, base_time, tau, excit_list) 

    velocity_series = compute_velocity_series(locations)
    avg_vel_bins = compute_velocity(locations, base_time)

    time_centers_all = 0.5 * (base_time[:-1] + base_time[1:])
    
    (cv_rate_maps_hd, occ_maps_hd, cv_masks, hd_edges) = get_hd_rate_maps_cv(
        spike_rates=spike_rates,
        locations=locations,
        base_time=base_time,
        n_hd_bins=n_hd_bins,
        velocity_threshold=vel_threshold,
        cv_count=cv_folds,
    )

    (cv_rate_maps_xy, occ_maps_xy, _, (x_edges, y_edges)) = get_xy_rate_maps_cv(
        spike_rates=spike_rates,
        locations=locations,
        base_time=base_time,
        n_xy_bins=n_xy_bins,
        velocity_threshold=vel_threshold,
        cv_count=cv_folds,
    )

    cv_summary: Dict[str, Any] = {"folds": [], "n_folds": int(cv_folds)}

    for cv_idx in range(cv_folds):
        print(f"\n=== Decoding CV fold {cv_idx}/{cv_folds-1} (HD & XY) ===")

        # HD
        rate_map_hd = gaussian_filter_hd(cv_rate_maps_hd[cv_idx], smooth_sigma_hd)
        occ_hd = occ_maps_hd[cv_idx]
        mask_hd = cv_masks[cv_idx]  # True=train, False=test
        test_idx = np.where(~mask_hd)[0]

        sc = test_spike_counts(spike_rates, test_idx, tau, smooth_spikes_sigma_s)
        avg_vel = avg_vel_bins[test_idx]
        tc = time_centers_all[test_idx]

        decoded_hd, post_hd = hd_bayesian_decoding(
            rate_map_hd=rate_map_hd,
            spike_counts=sc,
            tau=tau,
            hd_bins=hd_edges,
            avg_vel=avg_vel,
            vel_threshold=vel_threshold,
        )

        base_hd = f"decoded_hd_cv{cv_idx}_{n_hd_bins}"
        np.save(save_path / f"{base_hd}.npy", decoded_hd)
        np.save(save_path / f"{base_hd}_posterior.npy", np.array(post_hd, dtype=object))
        np.save(save_path / f"{base_hd}_time_centers.npy", tc)
        if save_intermediates:
            np.save(save_path / f"hd_rate_map_cv{cv_idx}_{n_hd_bins}.npy", rate_map_hd)
            np.save(save_path / f"hd_occupancy_map_cv{cv_idx}_{n_hd_bins}.npy", occ_hd)

        eval_hd = evaluate_errors_hd(decoded_hd, tc, locations, velocity_series, vel_threshold)
        np.savez_compressed(save_path / f"errors_hd_cv{cv_idx}_{n_hd_bins}.npz", **eval_hd)

        # HD Chance decoding
        eval_hd_chance = None
        if run_chance_decoding:
            print(f"  -> HD chance decoding (mode='{chance_mode}')")
            decoded_hd_chance, hd_prior = hd_chance_decoding(
                occupancy=occ_hd,
                n_test_bins=len(test_idx),
                hd_edges=hd_edges,
                avg_vel=avg_vel,
                vel_threshold=vel_threshold,
                mode=chance_mode,
                rng=rng,
            )
            np.save(save_path / f"decoded_hd_chance_cv{cv_idx}_{n_hd_bins}.npy", decoded_hd_chance)
            np.save(save_path / f"hd_occupancy_prior_cv{cv_idx}_{n_hd_bins}.npy", hd_prior)
            
            eval_hd_chance = evaluate_errors_hd(decoded_hd_chance, tc, locations, velocity_series, vel_threshold)
            np.savez_compressed(save_path / f"errors_hd_chance_cv{cv_idx}_{n_hd_bins}.npz", **eval_hd_chance)
            
            # Also compute analytical expected error
            analytical_hd = compute_chance_error_analytical_hd(occ_hd, hd_edges)
            with open(save_path / f"chance_analytical_hd_cv{cv_idx}_{n_hd_bins}.json", "w") as f:
                json.dump(analytical_hd, f, indent=2)

        # XY (same folds as HD)
        rate_map_xy = cv_rate_maps_xy[cv_idx]
        occ_xy = occ_maps_xy[cv_idx]
        if smooth_sigma_xy and smooth_sigma_xy > 0:
            rate_map_xy = gaussian_filter_xy(rate_map_xy, (smooth_sigma_xy, smooth_sigma_xy))

        decoded_xy, post_xy = xy_bayesian_decoding(
            rate_map_xy=rate_map_xy,
            spike_counts=sc,
            tau=tau,
            avg_vel=avg_vel,
            x_edges=x_edges,
            y_edges=y_edges,
            vel_threshold=vel_threshold,
        )
        
        n_x_bins = len(x_edges) - 1
        n_y_bins = len(y_edges) - 1

        base_xy = f"decoded_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}"
        np.save(save_path / f"{base_xy}.npy", decoded_xy)
        np.save(save_path / f"{base_xy}_posterior.npy", np.array(post_xy, dtype=object))
        np.save(save_path / f"{base_xy}_time_centers.npy", tc)
        if save_intermediates:
            np.save(save_path / f"xy_rate_map_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", rate_map_xy)
            np.save(save_path / f"xy_occupancy_map_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", occ_xy)

        eval_xy = evaluate_errors_xy(decoded_xy, tc, locations, velocity_series, vel_threshold)
        np.savez_compressed(save_path / f"errors_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npz", **eval_xy)

        # XY Chance decoding
        eval_xy_chance = None
        if run_chance_decoding:
            print(f"  -> XY chance decoding (mode='{chance_mode}')")
            decoded_xy_chance, xy_prior = xy_chance_decoding(
                occupancy=occ_xy,
                n_test_bins=len(test_idx),
                x_edges=x_edges,
                y_edges=y_edges,
                avg_vel=avg_vel,
                vel_threshold=vel_threshold,
                mode=chance_mode,
                rng=rng,
            )
            np.save(save_path / f"decoded_xy_chance_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", decoded_xy_chance)
            np.save(save_path / f"xy_occupancy_prior_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", xy_prior)
            
            eval_xy_chance = evaluate_errors_xy(decoded_xy_chance, tc, locations, velocity_series, vel_threshold)
            np.savez_compressed(save_path / f"errors_xy_chance_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npz", **eval_xy_chance)
            
            analytical_xy = compute_chance_error_analytical_xy(occ_xy, x_edges, y_edges)
            with open(save_path / f"chance_analytical_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}.json", "w") as f:
                json.dump(analytical_xy, f, indent=2)

        fold_rec: Dict[str, Any] = {
            "cv_idx": int(cv_idx),
            "n_test_bins": int(len(test_idx)),
            "output_prefix_hd": base_hd,
            "output_prefix_xy": base_xy,
            "error_stats_hd": eval_hd["stats"],
            "error_stats_xy": eval_xy["stats"],
        }
        if run_chance_decoding:
            fold_rec["error_stats_hd_chance"] = eval_hd_chance["stats"] if eval_hd_chance else None
            fold_rec["error_stats_xy_chance"] = eval_xy_chance["stats"] if eval_xy_chance else None
        cv_summary["folds"].append(fold_rec)

    with open(save_path / f"cv_summary_hd_xy_{n_y_bins}x{n_x_bins}_{n_hd_bins}.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    # Compute and save no-prior (uniform) chance baselines
    no_prior_hd = compute_no_prior_chance_hd()
    no_prior_xy = compute_no_prior_chance_xy(x_edges, y_edges)
    no_prior_chance = {
        'hd': no_prior_hd,
        'xy': no_prior_xy,
    }
    with open(save_path / "no_prior_chance.json", "w") as f:
        json.dump(no_prior_chance, f, indent=2)
    print(f"\nNo-prior chance baselines: HD={no_prior_hd['no_prior_expected_mae_deg']:.1f}°, XY={no_prior_xy['no_prior_expected_mae_cm']:.1f}cm")

    print("\nWrote CV summary:", save_path / f"cv_summary_hd_xy_{n_y_bins}x{n_x_bins}_{n_hd_bins}.json")
    if run_chance_decoding:
        print("Chance-level decoding results saved alongside neural decoding.")
    print(f"\nDone in {(time.time()-t0)/60.0:.2f} min.")
    return cv_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-validated HD,XY rate maps from Pfeiffer + Bayesian decoding."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/rodents/pfeiffer",
        help="Root data directory (contains Rat{rat}/Open1)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/rodents/pfeiffer",
        help="Directory to write outputs",
    )
    
    # default settings replicate the original paper's methods
    parser.add_argument("--rat", type=str, default="1", help="Rat ID (e.g., '1')")
    parser.add_argument("--n_hd_bins", type=int, default=360, help="Number of head-direction bins")
    parser.add_argument("--n_xy_bins", type=int, default=100, help="Number of xy bins")
    parser.add_argument("--tau", type=float, default=0.250, help="Time bin size (s)")
    parser.add_argument("--vel_threshold", type=float, default=5.0, help="Velocity threshold (cm/s)")
    parser.add_argument("--cv_folds", type=int, default=4, help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    parser.add_argument("--smooth_sigma_hd", type=float, default=2.0, help="Sigma (in bins) for Gaussian smoothing of HD curves")
    parser.add_argument("--smooth_spikes_sigma_s", type=float, default=0.0, help="Temporal Gaussian σ for spike counts before decoding (s)")
    parser.add_argument("--save_intermediates", action="store_true", default=False, help="Save intermediate artifacts (rate maps, masks, etc.) for debugging")
    
    # Chance decoding arguments
    parser.add_argument("--run_chance_decoding", action="store_true", default=True,
                        help="Run chance-level decoding based on occupancy (default: True)")
    parser.add_argument("--no_chance_decoding", action="store_false", dest="run_chance_decoding",
                        help="Disable chance-level decoding")
    parser.add_argument("--chance_mode", type=str, default="mode", choices=["mode", "sample"],
                        help="Chance decoding mode: 'mode' (always predict most occupied bin) or 'sample' (sample from occupancy)")
    
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    decode_solo(
        data_path=args.data_path,
        save_path=args.save_path,
        rat=args.rat,
        n_hd_bins=args.n_hd_bins,
        n_xy_bins=args.n_xy_bins,
        tau=args.tau,
        vel_threshold=args.vel_threshold,
        cv_folds=args.cv_folds,
        seed=args.seed,
        smooth_sigma_hd=args.smooth_sigma_hd,
        smooth_spikes_sigma_s=args.smooth_spikes_sigma_s,
        save_intermediates=args.save_intermediates,
        run_chance_decoding=args.run_chance_decoding,
        chance_mode=args.chance_mode,
    )


if __name__ == "__main__":
    main()