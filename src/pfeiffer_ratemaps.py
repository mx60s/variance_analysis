#!/usr/bin/env python3
"""
Example usage:

python pfeiffer_ratemaps.py \
  --data_path /path/to/pfeiffer/Rat1/Open1 \
  --save_path ./results/Rat1 \
  --n_x_bins 30 --n_y_bins 30 --tau 0.12 \
  --filter_vel --vel_threshold 5 --smooth_sigma_xy 1
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------- Utilities ---------------------------- #

def compute_velocity(locations: np.ndarray, base_time: np.ndarray) -> np.ndarray:
    if locations.shape[0] < 4:
        raise ValueError("locations must have >= 4 rows to estimate time step.")
    time_step = float(locations[3, 0] - locations[2, 0])
    steps = np.diff(locations[:, 1:3], axis=0)
    steps = np.vstack((steps, steps[-1]))  # pad last to align length
    speed_samples = np.linalg.norm(steps, axis=1) / max(time_step, 1e-9)

    v = np.zeros(len(base_time) - 1, dtype=float)
    for t in range(len(base_time) - 1):
        idx = np.where((locations[:, 0] >= base_time[t]) & (locations[:, 0] < base_time[t + 1]))[0]
        v[t] = 0.0 if idx.size == 0 else float(np.mean(speed_samples[idx]))
    return v


def _gaussian_filter_xy(rate_map: np.ndarray, smooth_sigma_xy: Optional[Tuple[float, float]]) -> np.ndarray:
    if smooth_sigma_xy is None:
        return rate_map
    sy, sx = smooth_sigma_xy
    if (sy is None or sy <= 0) and (sx is None or sx <= 0):
        return rate_map

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

    with np.errstate(invalid="ignore", divide="ignore"):
        out = VV / UU
        out[UU <= 0] = np.nan
    return out


def _select_time_indices_in_xy_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    x_win: Tuple[float, float],
    y_win: Tuple[float, float],
    fast_bins: Optional[np.ndarray],
    timebin_tol: float = 1e-4,
) -> np.ndarray:
    x0, x1 = x_win
    y0, y1 = y_win
    in_xy = np.where(
        (locations[:, 1] >= x0) & (locations[:, 1] < x1) &
        (locations[:, 2] >= y0) & (locations[:, 2] < y1)
    )[0]
    if in_xy.size == 0:
        return np.array([], dtype=int)

    loc_times = locations[in_xy, 0]
    loc_idx = np.array([int(np.abs(base_time - t).argmin()) for t in loc_times], dtype=int)

    if fast_bins is not None and fast_bins.size:
        mask = np.any(np.isclose(base_time[loc_idx, None], fast_bins, atol=timebin_tol), axis=1)
        loc_idx = loc_idx[mask]

    T = len(base_time) - 1
    return loc_idx[(loc_idx >= 0) & (loc_idx < T)]


# ------------------ XY spatial rate maps --------------------- #

def _get_xy_edges(locations: np.ndarray, n_x_bins: int, n_y_bins: int):
    xmin, xmax = float(np.nanmin(locations[:, 1])), float(np.nanmax(locations[:, 1]))
    ymin, ymax = float(np.nanmin(locations[:, 2])), float(np.nanmax(locations[:, 2]))
    x_edges = np.linspace(xmin, xmax + 1e-6, n_x_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_y_bins + 1)
    return x_edges, y_edges


def get_xy_rate_map_all(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_x_bins: int,
    n_y_bins: int,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
    smooth_sigma_xy: Optional[float] = None,
):
    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if filter_vel else None

    x_edges, y_edges = _get_xy_edges(locations, n_x_bins, n_y_bins)

    T, N = spike_rates.shape
    rate_map = np.full((n_y_bins, n_x_bins, N), np.nan, dtype=float)
    occupancy = np.zeros((n_y_bins, n_x_bins), dtype=float)

    for iy in tqdm(range(n_y_bins), desc="Y bins"):
        y_win = (y_edges[iy], y_edges[iy + 1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix + 1])
            idx = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_bins)
            if idx.size:
                occupancy[iy, ix] = float(idx.size)
                rate_map[iy, ix, :] = np.nanmean(spike_rates[idx, :], axis=0)

    if smooth_sigma_xy and smooth_sigma_xy > 0:
        rate_map = _gaussian_filter_xy(rate_map, (smooth_sigma_xy, smooth_sigma_xy))

    return rate_map, occupancy, (x_edges, y_edges)


# ------------------------- Data Loading / Prep ---------------------- #

def load_pfeiffer_data(data_path: Path):
    locations = loadmat(str(data_path / "Position_Data.mat"))["Position_Data"]
    spike_data = loadmat(str(data_path / "Spike_Data.mat"))
    spike_times = spike_data["Spike_Data"]  # columns: [time, neuron_id]
    return spike_times, locations, spike_data


def extract_excitatory_ids(spike_data: dict) -> np.ndarray:
    excit = spike_data["Excitatory_Neurons"]
    excit = np.asarray(excit).ravel().astype(spike_data["Spike_Data"].dtype)
    return excit


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



def main():
    parser = argparse.ArgumentParser(description="Build XY rate maps (Pfeiffer) — slim, no decoding")
    parser.add_argument("--data_path", type=str, required=True, help="Path to RatX/Open1 directory containing mats")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to write outputs")

    parser.add_argument("--tau", type=float, default=0.120, help="Time bin size (s)")

    parser.add_argument("--n_x_bins", type=int, default=30, help="Number of X bins for spatial map")
    parser.add_argument("--n_y_bins", type=int, default=30, help="Number of Y bins for spatial map")
    parser.add_argument("--smooth_sigma_xy", type=float, default=1.0, help="Sigma (bins) for XY Gaussian smoothing; <=0 to disable")

    parser.add_argument("--vel_threshold", type=float, default=5.0, help="Velocity threshold (cm/s) for filtering")
    parser.add_argument("--filter_vel", action="store_true", default=False, help="Apply velocity filtering (recommended)")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_path)
    os.makedirs(save_path, exist_ok=True)

    spike_times_raw, locations, spike_data = load_pfeiffer_data(data_path)
    excit_list = extract_excitatory_ids(spike_data)

    print(f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_list)} | Position samples: {locations.shape[0]}")

    # Build time base
    start_t = float(spike_times_raw[0, 0])
    end_t = float(spike_times_raw[-1, 0])
    base_time = np.arange(start_t, end_t + args.tau, args.tau)
    print(f"Time bins: {len(base_time)-1} covering {base_time[0]:.3f}–{base_time[-1]:.3f}s (tau={args.tau:.3f}s)")

    print("Binning spikes into rates…")
    spike_rates = build_spike_rates(spike_times_raw, base_time, args.tau, excit_list)

    rate_map_xy, occupancy_xy, (x_edges, y_edges) = get_xy_rate_map_all(
        spike_rates=spike_rates,
        locations=locations,
        base_time=base_time,
        n_x_bins=args.n_x_bins,
        n_y_bins=args.n_y_bins,
        velocity_threshold=args.vel_threshold,
        filter_vel=args.filter_vel,
        smooth_sigma_xy=args.smooth_sigma_xy,
    )

    np.save(save_path / f"xy_rate_map_{args.n_y_bins}x{args.n_x_bins}.npy", rate_map_xy)
    np.save(save_path / f"xy_occupancy_{args.n_y_bins}x{args.n_x_bins}.npy", occupancy_xy)
    np.save(save_path / f"x_edges_{args.n_x_bins}.npy", x_edges)
    np.save(save_path / f"y_edges_{args.n_y_bins}.npy", y_edges)
    print("Saved:")
    print(f"  rate map -> {save_path / f'xy_rate_map_{args.n_y_bins}x{args.n_x_bins}.npy'}")
    print(f"  occupancy -> {save_path / f'xy_occupancy_{args.n_y_bins}x{args.n_x_bins}.npy'}")
    print(f"  x_edges -> {save_path / f'x_edges_{args.n_x_bins}.npy'}")
    print(f"  y_edges -> {save_path / f'y_edges_{args.n_y_bins}.npy'}")


if __name__ == "__main__":
    main()
