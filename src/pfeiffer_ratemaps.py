#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, gaussian_filter


def compute_velocity(locations: np.ndarray, base_time: np.ndarray) -> np.ndarray:
    """Mean speed (cm/s) per time bin. locations: [t, x, y, hd]."""
    if locations.shape[0] < 4:
        raise ValueError("locations must have >= 4 rows to estimate time step.")
    time_step = float(locations[3, 0] - locations[2, 0])
    steps = np.diff(locations[:, 1:3], axis=0)
    steps = np.vstack((steps, steps[-1]))
    speed_samples = np.linalg.norm(steps, axis=1) / max(time_step, 1e-9)
    v = np.zeros(len(base_time) - 1, dtype=float)
    for t in range(len(base_time) - 1):
        idx = np.where((locations[:, 0] >= base_time[t]) & (locations[:, 0] < base_time[t + 1]))[0]
        v[t] = 0.0 if idx.size == 0 else float(np.mean(speed_samples[idx]))
    return v


def _nearest_edge_indices(base_time: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of [np.abs(base_time - t).argmin() for t in times]
    (nearest edge; exact ties resolve to the lower index, as argmin does)."""
    times = np.asarray(times, dtype=float)
    i = np.searchsorted(base_time, times)
    i = np.clip(i, 1, len(base_time) - 1)
    left = i - 1
    choose_left = (times - base_time[left]) <= (base_time[i] - times)
    return np.where(choose_left, left, i).astype(int)


def _fast_bin_mask(base_time: np.ndarray, loc_idx: np.ndarray, fast_bins: np.ndarray,
                   timebin_tol: float) -> np.ndarray:
    """Equivalent of np.any(np.isclose(base_time[loc_idx, None], fast_bins, atol=tol), axis=1)
    without materialising the (n_samples x n_fast_bins) matrix. fast_bins is sorted (a subset
    of base_time), so nearest-neighbour distance decides membership."""
    fb = np.sort(np.asarray(fast_bins, dtype=float))
    t = base_time[loc_idx]
    j = np.clip(np.searchsorted(fb, t), 0, len(fb) - 1)
    jl = np.clip(j - 1, 0, len(fb) - 1)
    d = np.minimum(np.abs(fb[j] - t), np.abs(fb[jl] - t))
    return d <= timebin_tol + 1e-8 * np.abs(t)   # np.isclose default rtol is 1e-5 on |b|; tighten but keep atol


def _select_time_indices_in_hd_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    hd_l: Tuple[float, float],
    fast_bins: Optional[np.ndarray],
    timebin_tol: float = 1e-4,
) -> np.ndarray:
    l0, l1 = hd_l
    in_hd = np.where((locations[:, 3] >= l0) & (locations[:, 3] < l1))[0]
    if in_hd.size == 0:
        return np.array([], dtype=int)

    loc_times = locations[in_hd, 0]
    loc_idx = _nearest_edge_indices(base_time, loc_times)

    if fast_bins is not None and fast_bins.size:
        mask = _fast_bin_mask(base_time, loc_idx, fast_bins, timebin_tol)
        loc_idx = loc_idx[mask]

    T = len(base_time) - 1
    return loc_idx[(loc_idx >= 0) & (loc_idx < T)]


def _select_time_indices_in_xy_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    x_win: Tuple[float, float],
    y_win: Tuple[float, float],
    fast_bins: Optional[np.ndarray],
    timebin_tol: float = 1e-4,
) -> np.ndarray:
    """Return array of *time-bin indices* whose samples fall in the given XY bin."""
    x0, x1 = x_win; y0, y1 = y_win
    in_xy = np.where(
        (locations[:, 1] >= x0) & (locations[:, 1] < x1) &
        (locations[:, 2] >= y0) & (locations[:, 2] < y1)
    )[0]
    if in_xy.size == 0:
        return np.array([], dtype=int)

    loc_times = locations[in_xy, 0]
    loc_idx = _nearest_edge_indices(base_time, loc_times)

    if fast_bins is not None and fast_bins.size:
        mask = _fast_bin_mask(base_time, loc_idx, fast_bins, timebin_tol)
        loc_idx = loc_idx[mask]

    T = len(base_time) - 1
    return loc_idx[(loc_idx >= 0) & (loc_idx < T)]


def build_spike_rates(spike_times: np.ndarray, base_time: np.ndarray, tau: float, excit_list: np.ndarray) -> np.ndarray:
    T = len(base_time) - 1
    N = len(excit_list)
    rates = np.zeros((T, N), dtype=float)
    for n_idx, neuron_id in enumerate(excit_list):
        neuron_spk = spike_times[spike_times[:, 1] == neuron_id, 0]
        if neuron_spk.size:
            counts, _ = np.histogram(neuron_spk, bins=base_time)
            rates[:, n_idx] = counts.astype(float) / max(tau, 1e-9)
    return rates

    
def xyhd_fold_rate_map(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    n_hd_bins: int,
    avg_vel_bins: np.ndarray,
    vel_threshold: float,
    filter_vel: bool,
    floor: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fast_idx = np.where(avg_vel_bins >= vel_threshold)[0]
    fast_bins = base_time[fast_idx] if filter_vel else None

    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)

    T, N = spike_rates.shape
    n_y_bins = len(y_edges) - 1
    n_x_bins = len(x_edges) - 1

    fold0_map = np.full((n_y_bins, n_x_bins, n_hd_bins, N), np.nan)
    fold1_map = np.full((n_y_bins, n_x_bins, n_hd_bins, N), np.nan)
    rate_map = np.full((n_y_bins, n_x_bins, n_hd_bins, N), np.nan)
    occupancy = np.zeros((n_y_bins, n_x_bins, n_hd_bins), dtype=float)

    for iy in tqdm(range(n_y_bins), desc="Y bins"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx_xy = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_bins)
            if idx_xy.size == 0:
                continue
            
            for ih in range(n_hd_bins):
                hd_win = (hd_edges[ih], hd_edges[ih+1])
                idx_hd = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_bins)
                if idx_hd.size == 0:
                    continue
                
                idx = np.intersect1d(idx_xy, idx_hd, assume_unique=False)
                if idx.size < floor:
                    continue
                
                half = len(idx) // 2
                np.random.shuffle(idx)
                fold0, fold1 = spike_rates[idx[:half], :], spike_rates[idx[half:], :]
                
                if fold1.shape[0] > half:
                    fold1 = fold1[:-1, :]
                
                fold0_map[iy, ix, ih, :] = np.nanmean(fold0, axis=0)
                fold1_map[iy, ix, ih, :] = np.nanmean(fold1, axis=0)
                rate_map[iy, ix, ih, :] = np.nanmean(spike_rates[idx, :], axis=0)
                occupancy[iy, ix, ih] = half * 2

    return fold0_map, fold1_map, rate_map, occupancy, (x_edges, y_edges, hd_edges)
    
def get_fold_ratemaps(args: Optional[argparse.Namespace] = None, /, **overrides: Any) -> Dict[str, Any]:
    if args is None:
        args = argparse.Namespace(**overrides)
    else:
        for k, v in overrides.items():
            setattr(args, k, v)

    d = args.__dict__
    d.setdefault("data_path", "./data/rodents/pfeiffer")
    d.setdefault("save_path", "./results/rodents/pfeiffer")
    d.setdefault("rat", "1")
    d.setdefault("n_hd_bins", 90)
    d.setdefault("n_loc_bins", 100) 
    d.setdefault("tau", 0.250)
    d.setdefault("vel_threshold", 5.0)
    d.setdefault("filter_vel", True)
    d.setdefault("floor", 5)

    data_path = Path(d["data_path"]) / f"Rat{d['rat']}" / "Open1"
    save_path = Path(d["save_path"]) / f"Rat{d['rat']}"
    os.makedirs(save_path, exist_ok=True)

    print(f"\n=== XY×HD Rate Map Builder ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_path}")
    print(f"XY bins: {d['n_loc_bins']} × {d['n_loc_bins']} | HD bins: {d['n_hd_bins']}")
    print(f"tau: {d['tau']:.3f}s | vel_th: {d['vel_threshold']} cm/s | filter_vel: {d['filter_vel']}")
    t0 = time.time()

    locations = loadmat(str(data_path / 'Position_Data.mat'))['Position_Data']
    spike_data = loadmat(str(data_path / 'Spike_Data.mat'))
    spike_times_raw = spike_data['Spike_Data']  # columns: [time, neuron_id]
    excit = spike_data['Excitatory_Neurons']
    excit_list = np.asarray(excit).ravel().astype(spike_data['Spike_Data'].dtype)
    print(f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_list)} | Position samples: {locations.shape[0]}")

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + d["tau"], d["tau"])
    print(f"Time bins: {len(base_time)-1} covering {base_time[0]:.3f}–{base_time[-1]:.3f}s")

    spike_rates = build_spike_rates(spike_times_raw, base_time, d["tau"], excit_list)  # (T, N)

    avg_vel_bins = compute_velocity(locations, base_time)

    xmin, xmax = 0, 201.0
    ymin, ymax = 0, 201.0

    n_x_bins = int(d["n_loc_bins"])
    n_y_bins = int(d["n_loc_bins"])
    x_edges = np.linspace(xmin, xmax + 1e-6, n_x_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_y_bins + 1)
    print(f"XY grid: {n_y_bins} (Y) × {n_x_bins} (X)")

    fold0_map, fold1_map, rate_map, occupancy, (x_edges, y_edges, hd_edges) = xyhd_fold_rate_map(
        spike_rates=spike_rates,
        locations=locations,
        base_time=base_time,
        x_edges=x_edges,
        y_edges=y_edges,
        n_hd_bins=d["n_hd_bins"],
        avg_vel_bins=avg_vel_bins,
        vel_threshold=d["vel_threshold"],
        filter_vel=d["filter_vel"],
        floor=d['floor']
    )

    # Save
    base = f"xyhd_{n_y_bins}x{n_x_bins}x{d['n_hd_bins']}"
    rm_save = save_path / f"ratemap_{base}.npz"
    np.savez_compressed(
        rm_save,
        rate_map=rate_map,
        occupancy=occupancy,
        x_edges=x_edges,
        y_edges=y_edges,
        hd_edges=hd_edges
    )
    
    fold0_rm = save_path / f"fold0_{base}.npz"
    np.savez_compressed(
        fold0_rm,
        rate_map=fold0_map,
        occupancy=occupancy,
        x_edges=x_edges,
        y_edges=y_edges,
        hd_edges=hd_edges
    )
    
    fold1_rm = save_path / f"fold1_{base}.npz"
    np.savez_compressed(
        fold1_rm,
        rate_map=fold1_map,
        occupancy=occupancy,
        x_edges=x_edges,
        y_edges=y_edges,
        hd_edges=hd_edges
    )
    
    print(f"\nSaved maps at: {save_path}")

    meta_path = save_path / f"{base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "tau": d["tau"],
            "vel_threshold": d["vel_threshold"],
            "filter_vel": d["filter_vel"],
            "n_loc_bins": d["n_loc_bins"],
            "n_hd_bins": d["n_hd_bins"],
        }, f, indent=2)

    elapsed_min = (time.time() - t0) / 60.0
    print(f"Done in {elapsed_min:.2f} min.")

    return {
        "out_npz": [rm_save, fold0_rm, fold1_rm,],
        "out_meta": str(meta_path),
        "rate_map_shape": tuple(rate_map.shape),
        "occupancy_shape": tuple(occupancy.shape),
        "edges": {
            "x": (float(x_edges.min()), float(x_edges.max()), int(len(x_edges)-1)),
            "y": (float(y_edges.min()), float(y_edges.max()), int(len(y_edges)-1)),
            "hd": (float(hd_edges.min()), float(hd_edges.max()), int(len(hd_edges)-1)),
        },
        "elapsed_min": float(elapsed_min),
    }
    

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    get_fold_ratemaps(args)

if __name__ == "__main__":
    main()
