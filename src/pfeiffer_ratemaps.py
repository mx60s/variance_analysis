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

N_VISITS = 500 # you don't really need to mess around with this tbh 500 visits is more than enough

# ---------------------------- Utilities ---------------------------- #

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

def times_to_bin_indices(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map timestamps to base_time bin indices (left-closed, right-open)."""
    idx = np.searchsorted(edges, times, side='right') - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    return idx[valid]

# ---------------- Selectors ---------------- #

def _select_time_indices_in_hd_bin(locations: np.ndarray, base_time: np.ndarray,
                                   hd_l: Tuple[float, float],
                                   fast_mask: Optional[np.ndarray]) -> np.ndarray:
    l0, l1 = hd_l
    idx_loc = np.where((locations[:, 3] >= l0) & (locations[:, 3] < l1))[0]
    if idx_loc.size == 0:
        return np.array([], dtype=int)
    t = locations[idx_loc, 0]
    idx = times_to_bin_indices(t, base_time)
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx

def _select_time_indices_in_xy_bin(locations: np.ndarray, base_time: np.ndarray,
                                   x_win: Tuple[float, float], y_win: Tuple[float, float],
                                   fast_mask: Optional[np.ndarray]) -> np.ndarray:
    x0, x1 = x_win; y0, y1 = y_win
    idx_loc = np.where(
        (locations[:, 1] >= x0) & (locations[:, 1] < x1) &
        (locations[:, 2] >= y0) & (locations[:, 2] < y1)
    )[0]
    if idx_loc.size == 0:
        return np.array([], dtype=int)
    t = locations[idx_loc, 0]
    idx = times_to_bin_indices(t, base_time)
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx


def build_spike_rates(spike_times: np.ndarray, base_time: np.ndarray, tau: float, excit_list: np.ndarray) -> np.ndarray:
    """Per-neuron per-bin firing rates using numpy.histogram → (T, N_excit)."""
    T = len(base_time) - 1
    N = len(excit_list)
    rates = np.zeros((T, N), dtype=float)
    for n_idx, neuron_id in enumerate(excit_list):
        neuron_spk = spike_times[spike_times[:, 1] == neuron_id, 0]
        if neuron_spk.size:
            counts, _ = np.histogram(neuron_spk, bins=base_time)
            rates[:, n_idx] = counts.astype(float) / max(tau, 1e-9)
    return rates


def xyhd_raw_rate_map(spike_rates: np.ndarray,
                  locations: np.ndarray,
                  base_time: np.ndarray,
                  n_loc_bins: int,
                  n_hd_bins: int,
                  avg_vel_bins: np.ndarray,
                  vel_threshold: float,
                  filter_vel: bool) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    fast_mask = (avg_vel_bins >= vel_threshold) if filter_vel else None
    
    xmin, xmax = float(np.nanmin(locations[:,1])), float(np.nanmax(locations[:,1]))
    ymin, ymax = float(np.nanmin(locations[:,2])), float(np.nanmax(locations[:,2]))
    x_edges = np.linspace(xmin, xmax + 1e-6, n_loc_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_loc_bins + 1)
    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)

    T, N = spike_rates.shape
    raw_rate_map = np.full((n_loc_bins, n_loc_bins, n_hd_bins, N_VISITS, N), np.nan)
    occupancy = np.zeros((n_loc_bins, n_loc_bins, n_hd_bins), dtype=float)

    for iy in tqdm(range(n_loc_bins), desc="Y bins"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_loc_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx_xy = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_mask)
            if idx_xy.size == 0:
                continue
            for ih in range(n_hd_bins):
                hd_win = (hd_edges[ih], hd_edges[ih+1])
                idx_hd = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_mask)
                if idx_hd.size == 0:
                    continue
                idx = np.intersect1d(idx_xy, idx_hd, assume_unique=False)
                np.random.shuffle(idx)
                if idx.size:
                    take = int(min(N_VISITS, idx.size))
                    raw_rate_map[iy, ix, ih, :take, :] = np.nanmean(spike_rates[idx, :][:take], axis=0)
                    occupancy[iy, ix, ih] = take

    return raw_rate_map, occupancy, (x_edges, y_edges, hd_edges)


def get_raw_ratemaps(args: Optional[argparse.Namespace] = None, /, **overrides: Any) -> Dict[str, Any]:
    """
    Build XY×HD rate maps, storing raw visits.
    Returns a small summary dict with output paths and shapes.
    """
    if args is None:
        args = argparse.Namespace(**overrides)
    else:
        for k, v in overrides.items():
            setattr(args, k, v)

    d = args.__dict__
    d.setdefault("data_path", "/home/mag/variance-analysis/data/pfeiffer")
    d.setdefault("save_path", "/home/mag/variance-analysis/results/pfeiffer")
    d.setdefault("rat", "1")
    d.setdefault("n_loc_bins", 30)
    d.setdefault("n_hd_bins", 60)
    d.setdefault("tau", 0.120)
    d.setdefault("vel_threshold", 5.0)
    d.setdefault("filter_vel", True)

    data_path = Path(d["data_path"]) / f"Rat{d['rat']}" / "Open1"
    save_path = Path(d["save_path"]) / f"Rat{d['rat']}"
    os.makedirs(save_path, exist_ok=True)

    print(f"\n=== XY×HD Rate Map Builder ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_path}")
    print(f"XY bins   : {d['n_loc_bins']} × {d['n_loc_bins']} | HD bins: {d['n_hd_bins']}")
    print(f"tau: {d['tau']:.3f}s | vel_th: {d['vel_threshold']} cm/s | filter_vel: {d['filter_vel']}")
    t0 = time.time()

    locations = loadmat(str(data_path / 'Position_Data.mat'))['Position_Data']
    spike_data = loadmat(str(data_path / 'Spike_Data.mat'))
    spike_times_raw = spike_data['Spike_Data']  # columns: [time, neuron_id]
    
    excit = spike_data['Excitatory_Neurons']
    excit_list = np.asarray(excit).ravel().astype(spike_data['Spike_Data'].dtype)
    print(f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_list)} | Position samples: {locations.shape[0]}")

    if spike_times_raw.size == 0 or locations.size == 0 or excit_list.size == 0:
        raise RuntimeError("Insufficient data: check paths and Excitatory_Neurons list.")

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + d["tau"], d["tau"])
    print(f"Time bins: {len(base_time)-1} covering {base_time[0]:.3f}–{base_time[-1]:.3f}s")

    print("Binning spikes into rates…")
    spike_rates = build_spike_rates(spike_times_raw, base_time, d["tau"], excit_list)  # (T, N)

    avg_vel_bins = compute_velocity(locations, base_time)

    # Build map
    rate_map, occupancy, (x_edges, y_edges, hd_edges) = xyhd_raw_rate_map(
        spike_rates=spike_rates,
        locations=locations,
        base_time=base_time,
        n_loc_bins=d["n_loc_bins"],
        n_hd_bins=d["n_hd_bins"],
        avg_vel_bins=avg_vel_bins,
        vel_threshold=d["vel_threshold"],
        filter_vel=d["filter_vel"],
    )

    # Save
    base = f"raw_ratemap_xyhd_{d['n_loc_bins']}x{d['n_loc_bins']}x{d['n_hd_bins']}"
    out_path = save_path / f"{base}.npz"
    np.savez_compressed(
        out_path,
        rate_map=rate_map,
        occupancy=occupancy,
        x_edges=x_edges,
        y_edges=y_edges,
        hd_edges=hd_edges
    )
    print(f"\nSaved: {out_path}")

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
        "out_npz": str(out_path),
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

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make \"raw\" XY×HD rate maps (Pfeiffer data)."
    )
    parser.add_argument("--data_path", type=str,
                        default="/home/mag/variance-analysis/data/pfeiffer",
                        help="Root dir containing Rat{rat}/Open1")
    parser.add_argument("--save_path", type=str,
                        default="/home/mag/variance-analysis/results/pfeiffer",
                        help="Directory to write outputs")
    parser.add_argument("--rat", type=str, default="1", help="Rat ID")

    # Binning
    parser.add_argument("--n_loc_bins", type=int, default=30,
                        help="Number of spatial bins per axis (X=Y=n_loc_bins)")
    parser.add_argument("--n_hd_bins", type=int, default=60,
                        help="Number of head-direction bins")

    # Timing / thresholds
    parser.add_argument("--tau", type=float, default=0.120, help="Time bin size (s)")
    parser.add_argument("--vel_threshold", type=float, default=5.0,
                        help="Velocity threshold (cm/s)")
    parser.add_argument("--filter_vel", action="store_true", default=True,
                        help="Filter low-velocity bins")

    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    decode_solo(args)

if __name__ == "__main__":
    main()
