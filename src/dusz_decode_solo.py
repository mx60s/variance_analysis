#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
from tqdm import tqdm

from src.rodent_data import (
    load_dusz_data, build_spike_rates, compute_avg_velocity_per_bin, get_md_from_xy,
    select_time_indices_in_angle_bin, select_time_indices_in_xy_bin,
)
from src.decoding import (
    make_cv_masks, test_spike_counts, gaussian_filter_hd,
    hd_bayesian_decoding, xy_bayesian_decoding, hd_chance_decoding, xy_chance_decoding,
    compute_chance_error_analytical_hd, compute_chance_error_analytical_xy,
    compute_no_prior_chance_hd, compute_no_prior_chance_xy,
    evaluate_errors_hd, evaluate_errors_xy,
)

# rate maps

def get_angle_rate_maps_cv(
    spike_rates: np.ndarray, positions: np.ndarray, base_time: np.ndarray, angles_deg: np.ndarray,
    n_bins: int = 60, velocity_threshold: float = 5.0, cv_count: int = 4,
    avg_vel_bins: Optional[np.ndarray] = None, desc: str = "HD bins (CV)",
):
    """CV tuning curves over a per-sample angle series (HD: positions[:, 3], or MD)."""
    fast_mask = (avg_vel_bins >= velocity_threshold) if avg_vel_bins is not None else None
    edges = np.linspace(0.0, 360.0 + 1e-6, n_bins + 1)
    T, N = spike_rates.shape
    cv_masks = make_cv_masks(T, cv_count)
    shape_map = (n_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_bins,), dtype=float) for _ in range(cv_count)]
    for l in tqdm(range(n_bins), desc=desc):
        win = (edges[l], edges[l + 1])
        idx = select_time_indices_in_angle_bin(positions, base_time, angles_deg, win, fast_mask)
        if idx.size == 0:
            continue
        for cv in range(cv_count):
            mask_cv = cv_masks[cv]  # True=train
            valid = idx[idx < mask_cv.size]
            cv_idx = valid[mask_cv[valid]]
            occupancy_maps[cv][l] = float(cv_idx.size)
            if cv_idx.size:
                cv_rate_maps[cv][l, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)
    return cv_rate_maps, occupancy_maps, cv_masks, edges


def get_xy_rate_maps_cv(
    spike_rates: np.ndarray, positions: np.ndarray, base_time: np.ndarray,
    n_x_bins: int, n_y_bins: int, velocity_threshold: float = 5.0, cv_count: int = 4,
    avg_vel_bins: Optional[np.ndarray] = None
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if avg_vel_bins is not None else None
    
    xmin, xmax = float(np.nanmin(positions[:, 1])), float(np.nanmax(positions[:, 1]))
    ymin, ymax = float(np.nanmin(positions[:, 2])), float(np.nanmax(positions[:, 2]))
    x_edges = np.linspace(xmin, xmax + 1e-6, n_x_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_y_bins + 1)
    
    T, N = spike_rates.shape
    cv_masks = make_cv_masks(T, cv_count)
    shape_map = (n_y_bins, n_x_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_y_bins, n_x_bins), dtype=float) for _ in range(cv_count)]
    for iy in tqdm(range(n_y_bins), desc="Y bins (CV)"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx = select_time_indices_in_xy_bin(positions, base_time, x_win, y_win, fast_mask)
            if idx.size == 0:
                continue
            for cv in range(cv_count):
                mask_cv = cv_masks[cv]
                valid = idx[idx < mask_cv.size]
                cv_idx = valid[mask_cv[valid]]
                occupancy_maps[cv][iy, ix] = float(cv_idx.size)
                if cv_idx.size:
                    cv_rate_maps[cv][iy, ix, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)
    return cv_rate_maps, occupancy_maps, cv_masks, (x_edges, y_edges)


def decode_solo(
    dataset: int,
    rat: str,
    *,
    data_path: Union[str, Path] = "./data/rodents/dusz",
    save_path: Union[str, Path] = "./results/rodents/dusz",
    n_hd_bins: int = 360,
    n_md_bins: int = 360,
    md_smooth_window: int = 20,
    n_x_bins: int = 32,
    n_y_bins: int = 32,
    tau: float = 0.050,
    vel_threshold: float = 2.0,
    cv_folds: int = 4,
    seed: int = 777,
    smooth_sigma_deg: float = 3.0,
    smooth_spikes_sigma_s: float = 0.100,
    save_intermediates: bool = False,
    run_chance_decoding: bool = True,
    chance_mode: str = "mode",
) -> Dict[str, Any]:
    """
    Run all categories (HD, MD, XY) cross-validated bayesian single-dimension decoding.

    Default params generally follow what was laid out in Duszkiewicz et al. 2024
    
    Parameters
    ----------
    run_chance_decoding : bool
        If True, also run chance-level decoding based on occupancy alongside
        the neural decoder. This provides a baseline for comparison.
    chance_mode : str
        'mode' - always predict most occupied bin (deterministic)
        'sample' - sample from occupancy distribution (stochastic)
    """
    if cv_folds is None or int(cv_folds) < 2:
        raise ValueError("cv_folds must be >= 2.")

    data_path = Path(data_path)
    save_root = Path(save_path) / f"Dataset_{dataset}" / rat
    os.makedirs(save_root, exist_ok=True)

    print("\n=== Dusz Decoding Pipeline (HD, MD, XY) ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_root}")
    print(f"HD bins : {n_hd_bins}  | MD bins : {n_md_bins} (MD smoothing window: {md_smooth_window} samples)")
    print(f"XY bins : {n_y_bins} x {n_x_bins}")
    print(f"tau: {tau:.3f}s | vel_th: {vel_threshold} cm/s | CV folds: {cv_folds}")
    print(f"Angle smoothing: {smooth_sigma_deg}° | Spike-count temporal smoothing: {smooth_spikes_sigma_s*1000:.0f} ms")
    if run_chance_decoding:
        print(f"Chance decoding: ENABLED (mode='{chance_mode}')")
    else:
        print("Chance decoding: DISABLED")

    t0 = time.time()
    spike_times_raw, positions, cell_types, velocity_series = load_dusz_data(data_path, dataset, rat)
    hd = np.asarray(cell_types["hd"], dtype=int).ravel()
    excit = np.asarray(cell_types["ex"], dtype=int).ravel()
    excit_ids = np.where((hd == 1) & (excit == 1))[0]
    print(f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_ids)} | Position samples: {positions.shape[0]}")

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + tau, tau)

    spike_rates = build_spike_rates(spike_times_raw, base_time, tau, excit_ids) 
    avg_vel_bins = compute_avg_velocity_per_bin(base_time, velocity_series)

    md_series_deg = get_md_from_xy(positions, smoothing_window=md_smooth_window)

    # RNG for stochastic chance decoding
    rng = np.random.default_rng(seed)

    print(f"\nCV decoding with {cv_folds} folds")

    cv_rate_maps_hd, occ_maps_hd, cv_masks, hd_edges_cv = get_angle_rate_maps_cv(
        spike_rates=spike_rates,
        positions=positions,
        base_time=base_time,
        angles_deg=positions[:, 3],
        n_bins=n_hd_bins,
        velocity_threshold=vel_threshold,
        cv_count=cv_folds,
        avg_vel_bins=avg_vel_bins,
    )
    cv_rate_maps_md, occ_maps_md, _, md_edges_cv = get_angle_rate_maps_cv(
        spike_rates=spike_rates,
        positions=positions,
        base_time=base_time,
        angles_deg=md_series_deg,
        n_bins=n_md_bins,
        velocity_threshold=vel_threshold,
        cv_count=cv_folds,
        avg_vel_bins=avg_vel_bins,
        desc="MD bins (CV)",
    )
    cv_rate_maps_xy, occ_maps_xy, _, (x_edges, y_edges) = get_xy_rate_maps_cv(
        spike_rates=spike_rates,
        positions=positions,
        base_time=base_time,
        n_x_bins=n_x_bins,
        n_y_bins=n_y_bins,
        velocity_threshold=vel_threshold,
        cv_count=cv_folds,
        avg_vel_bins=avg_vel_bins,
    )

    cv_summary_all: Dict[str, Any] = {"mode": "all", "folds": [], "n_folds": int(cv_folds)}

    for cv_idx in range(cv_folds):
        print(f"\n=== CV fold {cv_idx}/{cv_folds-1}: HD ===")
        rate_map_hd_cv = cv_rate_maps_hd[cv_idx].copy()
        occ_hd_cv = occ_maps_hd[cv_idx]

        min_occ = 10
        low = occ_hd_cv < min_occ
        if np.any(low):
            rate_map_hd_cv[low, ...] = np.nan

        rate_map_hd_cv = gaussian_filter_hd(
            rate_map_hd_cv,
            float(smooth_sigma_deg) * (n_hd_bins / 360.0),
        )

        cv_mask = cv_masks[cv_idx]
        test_idx = np.where(~cv_mask)[0]
        # temporally smooth spikes before sorting as specified in the paper
        sc = test_spike_counts(spike_rates, test_idx, tau, smooth_spikes_sigma_s)
        vel = avg_vel_bins[test_idx]
        tc = 0.5 * (base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)])

        decoded_hd, post_hd = hd_bayesian_decoding(
            rate_map_hd=rate_map_hd_cv,
            spike_counts=sc,
            tau=tau,
            hd_bins=hd_edges_cv,
            avg_vel=vel,
            vel_threshold=vel_threshold,
            rate_offset=1e-12,
            progress=True,
        )
        base = f"decoded_hd_cv{cv_idx}_{n_hd_bins}_all"
        np.save(save_root / f"{base}.npy", decoded_hd)
        np.save(save_root / f"{base}_posterior.npy", np.array(post_hd, dtype=object))
        np.save(save_root / f"{base}_time_centers.npy", tc)
        if save_intermediates:
            np.save(save_root / f"hd_rate_map_cv{cv_idx}_{n_hd_bins}.npy", rate_map_hd_cv)

        eval_hd = evaluate_errors_hd(decoded_hd, tc, positions, velocity_series, vel_threshold, std_of_abs=False)
        np.savez_compressed(save_root / f"errors_hd_cv{cv_idx}_{n_hd_bins}_all.npz", **eval_hd)

        # HD Chance decoding
        eval_hd_chance = None
        if run_chance_decoding:
            print(f"  -> HD chance decoding (mode='{chance_mode}')")
            decoded_hd_chance, hd_prior = hd_chance_decoding(
                occupancy=occ_hd_cv,
                n_test_bins=len(test_idx),
                hd_edges=hd_edges_cv,
                avg_vel=vel,
                vel_threshold=vel_threshold,
                mode=chance_mode,
                rng=rng,
            )
            np.save(save_root / f"decoded_hd_chance_cv{cv_idx}_{n_hd_bins}_all.npy", decoded_hd_chance)
            np.save(save_root / f"hd_occupancy_prior_cv{cv_idx}_{n_hd_bins}.npy", hd_prior)
            
            eval_hd_chance = evaluate_errors_hd(decoded_hd_chance, tc, positions, velocity_series, vel_threshold, std_of_abs=False)
            np.savez_compressed(save_root / f"errors_hd_chance_cv{cv_idx}_{n_hd_bins}_all.npz", **eval_hd_chance)
            
            # Also compute analytical expected error
            analytical_hd = compute_chance_error_analytical_hd(occ_hd_cv, hd_edges_cv)
            with open(save_root / f"chance_analytical_hd_cv{cv_idx}_{n_hd_bins}.json", "w") as f:
                json.dump(analytical_hd, f, indent=2)

        # MD CV
        print(f"===  CV fold {cv_idx}/{cv_folds-1}: MD ===")
        rate_map_md_cv = cv_rate_maps_md[cv_idx].copy()
        min_occ_md = max(1, int(np.ceil(0.5 / tau)))
        occ_md_cv = occ_maps_md[cv_idx]
        low_md = occ_md_cv < min_occ_md
        if np.any(low_md):
            rate_map_md_cv[low_md, ...] = np.nan
        rate_map_md_cv = gaussian_filter_hd(
            rate_map_md_cv,
            float(smooth_sigma_deg) * (n_md_bins / 360.0),
        )

        decoded_md, post_md = hd_bayesian_decoding(
            rate_map_hd=rate_map_md_cv,
            spike_counts=sc,
            tau=tau,
            hd_bins=md_edges_cv,
            avg_vel=vel,
            vel_threshold=vel_threshold,
            rate_offset=1e-12,
            progress=True,
        )
        base = f"decoded_md_cv{cv_idx}_{n_md_bins}_all"
        np.save(save_root / f"{base}.npy", decoded_md)
        np.save(save_root / f"{base}_posterior.npy", np.array(post_md, dtype=object))
        np.save(save_root / f"{base}_time_centers.npy", tc)
        if save_intermediates:
            np.save(save_root / f"md_rate_map_cv{cv_idx}_{n_md_bins}.npy", rate_map_md_cv)
            np.save(save_root / f"md_occupancy_map_cv{cv_idx}_{n_md_bins}.npy", occ_md_cv)

        eval_md = evaluate_errors_hd(decoded_md, tc, positions, velocity_series, vel_threshold,
                                     angles_deg=md_series_deg, name="md", std_of_abs=False)
        np.savez_compressed(save_root / f"errors_md_cv{cv_idx}_{n_md_bins}_all.npz", **eval_md)

        # MD Chance decoding
        eval_md_chance = None
        if run_chance_decoding:
            print(f"  -> MD chance decoding (mode='{chance_mode}')")
            decoded_md_chance, md_prior = hd_chance_decoding(
                occupancy=occ_md_cv,
                n_test_bins=len(test_idx),
                hd_edges=md_edges_cv,
                avg_vel=vel,
                vel_threshold=vel_threshold,
                mode=chance_mode,
                rng=rng,
            )
            np.save(save_root / f"decoded_md_chance_cv{cv_idx}_{n_md_bins}_all.npy", decoded_md_chance)
            np.save(save_root / f"md_occupancy_prior_cv{cv_idx}_{n_md_bins}.npy", md_prior)
            
            eval_md_chance = evaluate_errors_hd(decoded_md_chance, tc, positions, velocity_series, vel_threshold,
                                                angles_deg=md_series_deg, name="md", std_of_abs=False)
            np.savez_compressed(save_root / f"errors_md_chance_cv{cv_idx}_{n_md_bins}_all.npz", **eval_md_chance)
            
            analytical_md = compute_chance_error_analytical_hd(occ_md_cv, md_edges_cv)
            with open(save_root / f"chance_analytical_md_cv{cv_idx}_{n_md_bins}.json", "w") as f:
                json.dump(analytical_md, f, indent=2)

        # XY CV
        print(f"===  CV fold {cv_idx}/{cv_folds-1}: XY ===")
        rate_map_xy_cv = cv_rate_maps_xy[cv_idx].copy()
        occ_xy_cv = occ_maps_xy[cv_idx]
        min_occ_xy = max(1, int(np.ceil(0.5 / tau)))
        low_xy = occ_xy_cv < min_occ_xy
        if np.any(low_xy):
            rate_map_xy_cv[low_xy, ...] = np.nan

        decoded_xy, post_xy = xy_bayesian_decoding(
            rate_map_xy=rate_map_xy_cv,
            spike_counts=sc,
            tau=tau,
            avg_vel=vel,
            x_edges=x_edges,
            y_edges=y_edges,
            vel_threshold=vel_threshold,
            nan_to_zero=False,
            progress=True,
        )
        base = f"decoded_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}_all"
        np.save(save_root / f"{base}.npy", decoded_xy)
        np.save(save_root / f"{base}_posterior.npy", np.array(post_xy, dtype=object))
        np.save(save_root / f"{base}_time_centers.npy", tc)
        if save_intermediates:
            np.save(save_root / f"xy_rate_map_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", rate_map_xy_cv)
            np.save(save_root / f"xy_occupancy_map_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", occ_xy_cv)

        eval_xy = evaluate_errors_xy(decoded_xy, tc, positions, velocity_series, vel_threshold, key_suffix="")
        np.savez_compressed(save_root / f"errors_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}_all.npz", **eval_xy)

        # XY Chance decoding
        eval_xy_chance = None
        if run_chance_decoding:
            print(f"  -> XY chance decoding (mode='{chance_mode}')")
            decoded_xy_chance, xy_prior = xy_chance_decoding(
                occupancy=occ_xy_cv,
                n_test_bins=len(test_idx),
                x_edges=x_edges,
                y_edges=y_edges,
                avg_vel=vel,
                vel_threshold=vel_threshold,
                mode=chance_mode,
                rng=rng,
            )
            np.save(save_root / f"decoded_xy_chance_cv{cv_idx}_{n_y_bins}x{n_x_bins}_all.npy", decoded_xy_chance)
            np.save(save_root / f"xy_occupancy_prior_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", xy_prior)
            
            eval_xy_chance = evaluate_errors_xy(decoded_xy_chance, tc, positions, velocity_series, vel_threshold, key_suffix="")
            np.savez_compressed(save_root / f"errors_xy_chance_cv{cv_idx}_{n_y_bins}x{n_x_bins}_all.npz", **eval_xy_chance)
            
            analytical_xy = compute_chance_error_analytical_xy(occ_xy_cv, x_edges, y_edges, key_suffix="")
            with open(save_root / f"chance_analytical_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}.json", "w") as f:
                json.dump(analytical_xy, f, indent=2)

        cv_record = {
            "cv_idx": int(cv_idx),
            "n_test_bins": int(len(test_idx)),
            "error_stats_hd": eval_hd["stats"],
            "error_stats_md": eval_md["stats"],
            "error_stats_xy": eval_xy["stats"],
        }
        if run_chance_decoding:
            cv_record["error_stats_hd_chance"] = eval_hd_chance["stats"] if eval_hd_chance else None
            cv_record["error_stats_md_chance"] = eval_md_chance["stats"] if eval_md_chance else None
            cv_record["error_stats_xy_chance"] = eval_xy_chance["stats"] if eval_xy_chance else None
        cv_summary_all["folds"].append(cv_record)

    with open(save_root / "cv_summary_all.json", "w") as f:
        json.dump(cv_summary_all, f, indent=2)

    # Compute and save no-prior (uniform) chance baselines
    no_prior_hd = compute_no_prior_chance_hd()
    no_prior_xy = compute_no_prior_chance_xy(x_edges, y_edges)
    no_prior_chance = {
        'hd': no_prior_hd,
        'md': no_prior_hd,  # Same as HD since both are angular [0, 360)
        'xy': no_prior_xy,
    }
    with open(save_root / "no_prior_chance.json", "w") as f:
        json.dump(no_prior_chance, f, indent=2)
    print(f"\nNo-prior chance baselines: HD/MD={no_prior_hd['no_prior_expected_mae_deg']:.1f}°, XY={no_prior_xy['no_prior_expected_mae_cm']:.1f}cm")

    print("\n Finished all decoding with HD-based neuron filtering.")
    if run_chance_decoding:
        print(" Chance-level decoding results saved alongside neural decoding.")
    print(f"Done in {(time.time()-t0)/60.0:.2f} min.")

    return cv_summary_all


def main():
    parser = argparse.ArgumentParser(description="HD/MD/XY tuning + Bayesian decoding with CV (Dusz)")
    parser.add_argument("--data_path", type=str, default="./data/rodents/dusz", help="Root data dir containing Dataset_{id}/{rat}/Data")
    parser.add_argument("--save_path", type=str, default="./results/rodents/dusz", help="Directory to write outputs")
    parser.add_argument("--dataset", type=int, required=True, help="Dataset number (int)")
    parser.add_argument("--rat", type=str, required=True, help="Rat ID (e.g., 'R1')")

    parser.add_argument("--n_hd_bins", type=int, default=360, help="Number of head-direction bins")
    parser.add_argument("--n_md_bins", type=int, default=360, help="Number of movement-direction bins")
    parser.add_argument("--md_smooth_window", type=int, default=20, help="Smoothing window (samples) for MD computation")
    parser.add_argument("--n_x_bins", type=int, default=10, help="Number of X bins (XY)")
    parser.add_argument("--n_y_bins", type=int, default=10, help="Number of Y bins (XY)")

    parser.add_argument("--tau", type=float, default=0.050, help="Time bin size (s). Paper used 50 ms")
    parser.add_argument("--vel_threshold", type=float, default=2.0, help="Velocity threshold (cm/s). Paper used 2 cm/s")

    parser.add_argument("--cv_folds", type=int, default=4, help="Number of CV folds (>=2). Default uses CV.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")

    parser.add_argument("--smooth_sigma_deg", type=float, default=3.0, help="Gaussian sigma across angle (degrees). Paper used 3 deg")
    parser.add_argument("--smooth_spikes_sigma_s", type=float, default=0.100, help="Temporal Gaussian sigma for spike counts before decoding (s). Paper used 100 ms")

    parser.add_argument("--save_intermediates", action="store_true", default=False, help="Save intermediate artifacts")
    
    # Chance decoding arguments
    parser.add_argument("--run_chance_decoding", action="store_true", default=True, 
                        help="Run chance-level decoding based on occupancy (default: True)")
    parser.add_argument("--no_chance_decoding", action="store_false", dest="run_chance_decoding",
                        help="Disable chance-level decoding")
    parser.add_argument("--chance_mode", type=str, default="mode", choices=["mode", "sample"],
                        help="Chance decoding mode: 'mode' (always predict most occupied bin) or 'sample' (sample from occupancy)")
    
    args = parser.parse_args()

    decode_solo(
        dataset=args.dataset,
        rat=args.rat,
        data_path=args.data_path,
        save_path=args.save_path,
        n_hd_bins=args.n_hd_bins,
        n_md_bins=args.n_md_bins,
        md_smooth_window=args.md_smooth_window,
        n_x_bins=args.n_x_bins,
        n_y_bins=args.n_y_bins,
        tau=args.tau,
        vel_threshold=args.vel_threshold,
        cv_folds=args.cv_folds,
        seed=args.seed,
        smooth_sigma_deg=args.smooth_sigma_deg,
        smooth_spikes_sigma_s=args.smooth_spikes_sigma_s,
        save_intermediates=args.save_intermediates,
        run_chance_decoding=args.run_chance_decoding,
        chance_mode=args.chance_mode,
    )


if __name__ == "__main__":
    main()