#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, gaussian_filter

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


def compute_velocity_series(locations: np.ndarray) -> np.ndarray:
    """Instantaneous speed series (per sample) as columns [time, speed]."""
    if locations.shape[0] < 4:
        raise ValueError("locations must have >= 4 rows to estimate time step.")
    time_step = float(locations[3, 0] - locations[2, 0])
    steps = np.diff(locations[:, 1:3], axis=0)
    steps = np.vstack((steps, steps[-1]))
    speed_samples = np.linalg.norm(steps, axis=1) / max(time_step, 1e-9)
    return np.column_stack((locations[:, 0], speed_samples))


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


def _gaussian_filter_xy(rate_map: np.ndarray, smooth_sigma_xy: Tuple[float, float]) -> np.ndarray:
    """NaN-aware 2D Gaussian smoothing over (Y, X, N) or (Y, X)."""
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
        # apply per-channel via broadcasting-safe filtering
        VV = np.empty_like(V)
        UU = np.empty_like(U)
        for n in range(V.shape[-1]):
            VV[..., n] = gaussian_filter(V[..., n], sigma=(sy, sx), mode="nearest")
            UU[..., n] = gaussian_filter(U[..., n], sigma=(sy, sx), mode="nearest")

    with np.errstate(invalid='ignore', divide='ignore'):
        out = VV / UU
    out[UU <= 0] = np.nan
    return out


def _select_time_indices_in_hd_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    hd_l: Tuple[float, float],
    fast_bins: Optional[np.ndarray],
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
    loc_idx = np.array([int(np.abs(base_time - t).argmin()) for t in loc_times], dtype=int)

    if fast_bins is not None and fast_bins.size:
        mask = np.any(np.isclose(base_time[loc_idx, None], fast_bins, atol=timebin_tol), axis=1)
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
    rng = np.random.RandomState(seed)

    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if filter_vel else None
    print(
        f"Filtering {100.0 - (len(fast_idx) / max(len(base_time)-1,1)) * 100.0:.1f}% of bins due to low velocity"
    )

    hd_edges = np.linspace(0.0, 360.0 + 1e-2, n_hd_bins + 1)

    T, N = spike_rates.shape
    assert T == len(base_time) - 1, "spike_rates must match len(base_time)-1"

    n_points = T
    folds = np.array_split(np.arange(n_points), cv_count)
    cv_masks: List[np.ndarray] = []
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)

    shape_map = (n_hd_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps    = [np.zeros((n_hd_bins,), dtype=float) for _ in range(cv_count)]

    print("Creating cross-validation HD tuning curves…")
    for l in tqdm(range(n_hd_bins), desc="HD bins"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_bins)
        if idx.size == 0:
            continue

        for cv in range(cv_count):
            occupancy_maps[cv][l] = float(idx.size)
            
            mask_cv = cv_masks[cv]
            valid = idx[idx < mask_cv.size]
            cv_idx = valid[mask_cv[valid]]
            cv_rate_maps[cv][l, :] = np.nanmean(spike_rates[cv_idx, :])

    return (
        cv_rate_maps,
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

# ------------------ XY spatial rate maps --------------------- #

def get_xy_rate_map_all(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_x_bins: int,
    n_y_bins: int,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
):
    """Build a single (all-data) XY tuning map (Y, X, N), plus occupancy (Y, X)."""
    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if filter_vel else None

    xmin, xmax = float(np.nanmin(locations[:,1])), float(np.nanmax(locations[:,1]))
    ymin, ymax = float(np.nanmin(locations[:,2])), float(np.nanmax(locations[:,2]))
    x_edges = np.linspace(xmin, xmax + 1e-6, n_x_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_y_bins + 1)

    T, N = spike_rates.shape
    rate_map = np.full((n_y_bins, n_x_bins, N), np.nan, dtype=float)
    occupancy = np.zeros((n_y_bins, n_x_bins), dtype=float)

    print("Creating all-data XY tuning map…")
    for iy in tqdm(range(n_y_bins), desc="Y bins"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_bins)
            if idx.size:
                occupancy[iy, ix] = float(idx.size)
                rate_map[iy, ix, :] = np.nanmean(spike_rates[idx, :], axis=0)
    return rate_map, occupancy, (x_edges, y_edges)

def get_xy_rate_maps_cv(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_x_bins: int,
    n_y_bins: int,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
    cv_count: int = 4,
    seed: int = 777,
):
    """Cross-validated XY maps: returns lists per fold analogous to HD."""
    rng = np.random.RandomState(seed)

    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if filter_vel else None
    print(
        f"Filtering {100.0 - (len(fast_idx) / max(len(base_time)-1,1)) * 100.0:.1f}% of bins due to low velocity"
    )

    xmin, xmax = float(np.nanmin(locations[:,1])), float(np.nanmax(locations[:,1]))
    ymin, ymax = float(np.nanmin(locations[:,2])), float(np.nanmax(locations[:,2]))
    x_edges = np.linspace(xmin, xmax + 1e-6, n_x_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_y_bins + 1)

    T, N = spike_rates.shape
    n_points = T
    folds = np.array_split(np.arange(n_points), cv_count)
    cv_masks: List[np.ndarray] = []
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)

    shape_map = (n_y_bins, n_x_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_y_bins, n_x_bins,), dtype=float) for _ in range(cv_count)]

    print("Creating cross-validation XY tuning maps…")
    for iy in tqdm(range(n_y_bins), desc="Y bins"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_bins)
            if idx.size == 0:
                continue

            for cv in range(cv_count):
                occupancy_maps[cv][iy, ix] = float(idx.size)

                mask_cv = cv_masks[cv]
                valid = idx[idx < mask_cv.size]
                cv_idx = valid[mask_cv[valid]]
                cv_rate_maps[cv][iy, ix, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)

    return (
        cv_rate_maps,
        occupancy_maps,
        cv_masks,
        (x_edges, y_edges),
    )

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
    """Head-direction Bayesian decoding with Poisson likelihood."""
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


def xy_bayesian_decoding(
    rate_map_xy: np.ndarray,   # (Y, X, N) mean firing rate (Hz)
    spike_counts: np.ndarray,  # (T, N) counts per bin
    tau: float,
    avg_vel: np.ndarray,       # (T,)
    vel_threshold: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    eps: float = 1e-10,
):
    """2D spatial Bayesian decoding with Poisson likelihood."""
    T, N = spike_counts.shape
    Y, X, N_map = rate_map_xy.shape

    decoded_xy = np.full((T, 2), np.nan)
    posterior_all: List[np.ndarray] = []

    print(f"Decoding {T} time bins using {N} neurons over {Y}×{X} grid…")
    rate_tau = rate_map_xy * tau  # expected counts per bin

    # reshape to (Y*X, N) for efficient dot ops
    rate_tau_flat = rate_tau.reshape(Y * X, N)

    for t in tqdm(range(T), desc="Decoding time bins"):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((Y, X), np.nan))
            continue

        k = spike_counts[t]  # (N,)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k[None, :] * np.log(rate_tau_flat + eps) - rate_tau_flat, axis=1)  # (Y*X,)

        if np.all(np.isnan(log_like)):
            posterior_all.append(np.full((Y, X), np.nan))
            continue

        m = np.nanmax(log_like)
        post = np.exp(log_like - m)
        s = np.nansum(post)
        if s <= 0 or not np.isfinite(s):
            posterior_all.append(np.full((Y, X), np.nan))
            continue
        post = (post / s).reshape(Y, X)
        posterior_all.append(post)

        max_idx_flat = int(np.nanargmax(post))
        iy, ix = divmod(max_idx_flat, X)
        # decode to bin centers
        x_c = 0.5 * (x_edges[ix] + x_edges[ix + 1])
        y_c = 0.5 * (y_edges[iy] + y_edges[iy + 1])
        decoded_xy[t, 0] = x_c
        decoded_xy[t, 1] = y_c

    return decoded_xy, posterior_all

# ---------------------- Error analysis helpers --------------------- #

def _unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    rad_unwrapped = np.unwrap(rad, discont=np.pi)
    return np.rad2deg(rad_unwrapped)


def interp_circular_deg(t_src: np.ndarray, t_tgt: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    t_src = t_src[order]
    angles_deg = angles_deg[order]
    ang_unwrapped = _unwrap_deg(angles_deg)
    ang_interp_unwrapped = np.interp(t_tgt, t_src, ang_unwrapped)
    ang_interp = np.mod(ang_interp_unwrapped, 360.0)
    return ang_interp


def compute_ground_truth_hd(positions: np.ndarray, time_centers: np.ndarray, velocity_series: np.ndarray, velocity_filter: float, window: float = 0.06) -> np.ndarray:
    """Interpolate true HD at decoding centers and mask by mean speed in ±window."""
    timestamps = positions[:, 0]
    hd = positions[:, 3]  # degrees
    hd_interp = interp_circular_deg(timestamps, time_centers, hd)

    v_times = velocity_series[:, 0]
    v_vals = velocity_series[:, 1]

    velocity_interp = np.full(len(time_centers), np.nan)
    for i, tc in enumerate(time_centers):
        mask = (v_times >= tc - window) & (v_times <= tc + window)
        if np.any(mask):
            velocity_interp[i] = float(np.nanmean(v_vals[mask]))

    valid_velocity = (velocity_interp >= velocity_filter) & (~np.isnan(velocity_interp))
    ground_hd_filtered = np.where(valid_velocity, hd_interp, np.nan)
    return ground_hd_filtered


def evaluate_errors(decoded_hd: np.ndarray, time_centers: np.ndarray, positions: np.ndarray, velocity_series: np.ndarray, vel_threshold: float) -> Dict[str, Any]:
    gt = compute_ground_truth_hd(positions, time_centers, velocity_series, vel_threshold)
    hd_err_abs = (decoded_hd - gt + 180.0) % 360.0 - 180.0
    total = int(hd_err_abs.size)
    valid_mask = ~np.isnan(hd_err_abs)
    valid = int(np.sum(valid_mask))
    stats = {
        'hd_mean_abs_error_deg': float(np.nanmean(np.abs(hd_err_abs))),
        'hd_median_abs_error_deg': float(np.nanmedian(np.abs(hd_err_abs))),
        'hd_std_abs_error_deg': float(np.nanstd(hd_err_abs)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors_deg': hd_err_abs}


def _interp_linear_series(t_src: np.ndarray, t_tgt: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    return np.interp(t_tgt, t_src[order], v_src[order])

def compute_ground_truth_xy(positions: np.ndarray, time_centers: np.ndarray, velocity_series: np.ndarray, velocity_filter: float, window: float = 0.06) -> np.ndarray:
    """Interpolate true XY at decoding centers and mask by mean speed in ±window."""
    t = positions[:, 0]
    x = positions[:, 1]
    y = positions[:, 2]
    x_interp = _interp_linear_series(t, time_centers, x)
    y_interp = _interp_linear_series(t, time_centers, y)

    v_times = velocity_series[:, 0]
    v_vals = velocity_series[:, 1]
    velocity_interp = np.full(len(time_centers), np.nan)
    for i, tc in enumerate(time_centers):
        mask = (v_times >= tc - window) & (v_times <= tc + window)
        if np.any(mask):
            velocity_interp[i] = float(np.nanmean(v_vals[mask]))

    valid = (velocity_interp >= velocity_filter) & (~np.isnan(velocity_interp))
    gt_xy = np.column_stack([x_interp, y_interp])
    gt_xy[~valid, :] = np.nan
    return gt_xy

def evaluate_errors_xy(decoded_xy: np.ndarray, time_centers: np.ndarray, positions: np.ndarray, velocity_series: np.ndarray, vel_threshold: float) -> Dict[str, Any]:
    gt_xy = compute_ground_truth_xy(positions, time_centers, velocity_series, vel_threshold)
    diffs = decoded_xy - gt_xy  # (T,2)
    dists = np.sqrt(np.nansum(diffs**2, axis=1))  # Euclidean cm
    total = int(dists.size)
    valid_mask = ~np.isnan(dists)
    valid = int(np.sum(valid_mask))
    stats = {
        'xy_mean_abs_error_cm': float(np.nanmean(dists)),
        'xy_median_abs_error_cm': float(np.nanmedian(dists)),
        'xy_std_abs_error_cm': float(np.nanstd(dists)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors_cm': dists}

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

# ------------------------------ Public API ------------------------------ #

def decode_solo(args: Optional[argparse.Namespace] = None, /, **overrides: Any) -> Dict[str, Any]:
    """
    Run head-direction OR location tuning + Bayesian decoding (Pfeiffer data).
    Returns a small summary dict (paths & basic stats).
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
    d.setdefault("mode", "hd")
    d.setdefault("n_hd_bins", 60)
    d.setdefault("n_x_bins", 30)
    d.setdefault("n_y_bins", 30)
    d.setdefault("tau", 0.120)
    d.setdefault("vel_threshold", 5.0)
    d.setdefault("cv_folds", None)
    d.setdefault("filter_vel", True)
    d.setdefault("seed", 777)
    d.setdefault("smooth_sigma", 2.0)
    d.setdefault("smooth_sigma_xy", 1.0)
    d.setdefault("save_intermediates", False)

    # HD convention: ignore velocity gating at decode time
    if d["mode"] == "hd":
        d["filter_vel"] = False
        d["vel_threshold"] = 0.0

    # Resolve cv_folds (prompt only if interactive)
    if d["cv_folds"] is None:
        try:
            import sys
            if sys.stdin and sys.stdin.isatty():
                user_in = input("How many CV folds? (0 = no CV): ").strip()
                d["cv_folds"] = int(user_in)
            else:
                d["cv_folds"] = 0
        except Exception:
            print("No valid input; defaulting to 0 (no CV).")
            d["cv_folds"] = 0
    if d["cv_folds"] < 0:
        raise ValueError("cv_folds must be >= 0")

    data_path = Path(d["data_path"]) / f"Rat{d['rat']}" / "Open1"
    save_path = Path(d["save_path"]) / f"Rat{d['rat']}"
    os.makedirs(save_path, exist_ok=True)

    print(f"\n=== {'Head-Direction' if d['mode']=='hd' else 'Location (XY)'} Decoding Pipeline ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_path}")
    if d["mode"] == "hd":
        print(f"HD bins   : {d['n_hd_bins']}")
    else:
        print(f"XY bins   : {d['n_y_bins']} × {d['n_x_bins']}")
    print(f"tau: {d['tau']:.3f}s | vel_th: {d['vel_threshold']} cm/s | CV folds: {d['cv_folds']}")

    t0 = time.time()

    print("\nLoading data…")
    spike_times_raw, locations, spike_data = load_pfeiffer_data(data_path)
    excit_list = extract_excitatory_ids(spike_data)
    print(
        f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_list)} | Position samples: {locations.shape[0]}"
    )

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + d["tau"], d["tau"])
    print(f"Time bins: {len(base_time)-1} bins covering {base_time[0]:.3f}–{base_time[-1]:.3f}s")

    print("Binning spikes into rates…")
    spike_rates = build_spike_rates(spike_times_raw, base_time, d["tau"], excit_list)  # (T, N)

    velocity_series = compute_velocity_series(locations)

    # ------------------------- No-CV path ------------------------- #
    if d["cv_folds"] == 0:
        if d["mode"] == "hd":
            rate_map_hd, occupancy_all, hd_edges = get_hd_rate_map_all(
                spike_rates=spike_rates,
                locations=locations,
                base_time=base_time,
                n_hd_bins=d["n_hd_bins"],
                velocity_threshold=d["vel_threshold"],
                filter_vel=d["filter_vel"],
            )
            print("Using ALL-DATA HD tuning; decoding all bins.")
            spike_counts = (spike_rates * d["tau"]).astype(float)
            avg_vel = compute_velocity(locations, base_time)
            time_centers = 0.5 * (base_time[:-1] + base_time[1:])

            rate_map_hd = _gaussian_filter_hd(rate_map_hd, d["smooth_sigma"])

            if d["save_intermediates"]:
                np.save(save_path / f"hd_rate_map_{d['n_hd_bins']}.npy", rate_map_hd)
                np.save(save_path / f"hd_occupancy_map_{d['n_hd_bins']}.npy", occupancy_all)

            decoded_hd, posterior_all = hd_bayesian_decoding(
                rate_map_hd=rate_map_hd,
                spike_counts=spike_counts,
                tau=d["tau"],
                hd_bins=hd_edges,
                avg_vel=avg_vel,
                vel_threshold=d["vel_threshold"],
            )

            print("Saving decoding outputs…")
            base = f"decoded_hd_{d['n_hd_bins']}"
            np.save(save_path / f"{base}.npy", decoded_hd)
            np.save(save_path / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
            np.save(save_path / f"{base}_time_centers.npy", time_centers)

            eval_res = evaluate_errors(decoded_hd, time_centers, locations, velocity_series, d["vel_threshold"])
            np.savez_compressed(save_path / f"errors_hd_{d['n_hd_bins']}.npz", **eval_res)
            print("Saved error analysis:", save_path / f"errors_hd_{d['n_hd_bins']}.npz")
            print("Error stats:")
            for k, v in eval_res['stats'].items():
                print(f"  {k}: {v}")

            ok_hd = decoded_hd[~np.isnan(decoded_hd)]
            print("\nDecoding Summary:")
            print(f"Total time bins: {len(decoded_hd)}")
            print(f"Successfully decoded: {len(ok_hd)} ({100.0 * len(ok_hd) / max(len(decoded_hd),1):.1f}%)")
            if ok_hd.size:
                print(f"HD range: {np.min(ok_hd):.1f} – {np.max(ok_hd):.1f} deg")

            print(f"\nDone in {(time.time()-t0)/60.0:.2f} min.")
            return {
                "mode": "hd",
                "outputs_prefix": str(save_path / base),
                "n_time_bins": int(len(decoded_hd)),
                "decoded_fraction": float(len(ok_hd) / max(len(decoded_hd), 1)),
                "save_path": str(save_path),
            }

        else:  # XY
            rate_map_xy, occupancy_xy, (x_edges, y_edges) = get_xy_rate_map_all(
                spike_rates=spike_rates,
                locations=locations,
                base_time=base_time,
                n_x_bins=d["n_x_bins"],
                n_y_bins=d["n_y_bins"],
                velocity_threshold=d["vel_threshold"],
                filter_vel=d["filter_vel"],
            )
            print("Using ALL-DATA XY tuning; decoding all bins.")
            spike_counts = (spike_rates * d["tau"]).astype(float)
            avg_vel = compute_velocity(locations, base_time)
            time_centers = 0.5 * (base_time[:-1] + base_time[1:])

            if d["smooth_sigma_xy"] and d["smooth_sigma_xy"] > 0:
                rate_map_xy = _gaussian_filter_xy(rate_map_xy, (d["smooth_sigma_xy"], d["smooth_sigma_xy"]))

            if d["save_intermediates"]:
                np.save(save_path / f"xy_rate_map_{d['n_y_bins']}x{d['n_x_bins']}.npy", rate_map_xy)
                np.save(save_path / f"xy_occupancy_map_{d['n_y_bins']}x{d['n_x_bins']}.npy", occupancy_xy)

            decoded_xy, posterior_all = xy_bayesian_decoding(
                rate_map_xy=rate_map_xy,
                spike_counts=spike_counts,
                tau=d["tau"],
                avg_vel=avg_vel,
                vel_threshold=d["vel_threshold"],
                x_edges=x_edges,
                y_edges=y_edges,
            )

            print("Saving decoding outputs…")
            base = f"decoded_xy_{d['n_y_bins']}x{d['n_x_bins']}"
            np.save(save_path / f"{base}.npy", decoded_xy)
            np.save(save_path / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
            np.save(save_path / f"{base}_time_centers.npy", time_centers)

            eval_res = evaluate_errors_xy(decoded_xy, time_centers, locations, velocity_series, d["vel_threshold"])
            np.savez_compressed(save_path / f"errors_xy_{d['n_y_bins']}x{d['n_x_bins']}.npz", **eval_res)
            print("Saved error analysis:", save_path / f"errors_xy_{d['n_y_bins']}x{d['n_x_bins']}.npz")
            print("Error stats:")
            for k, v in eval_res['stats'].items():
                print(f"  {k}: {v}")

            ok_xy = decoded_xy[~np.isnan(decoded_xy).any(axis=1)]
            print("\nDecoding Summary:")
            print(f"Total time bins: {len(decoded_xy)}")
            print(f"Successfully decoded: {len(ok_xy)} ({100.0 * len(ok_xy) / max(len(decoded_xy),1):.1f}%)")

            print(f"\nDone in {(time.time()-t0)/60.0:.2f} min.")
            return {
                "mode": "xy",
                "outputs_prefix": str(save_path / base),
                "n_time_bins": int(len(decoded_xy)),
                "decoded_fraction": float(len(ok_xy) / max(len(decoded_xy), 1)),
                "save_path": str(save_path),
            }

    # -------------------------- CV path -------------------------- #
    else:
        if d["mode"] == "hd":
            (
                rate_maps_hd,
                occupancy_maps,
                cv_masks,
                hd_edges,
            ) = get_hd_rate_maps_cv(
                spike_rates=spike_rates,
                locations=locations,
                base_time=base_time,
                n_hd_bins=d["n_hd_bins"],
                velocity_threshold=d["vel_threshold"],
                filter_vel=d["filter_vel"],
                cv_count=d["cv_folds"],
                seed=d["seed"],
            )

            cv_summary: Dict[str, Any] = {"mode": "hd", "folds": [], "n_folds": int(d["cv_folds"])}

            for cv_idx in range(d["cv_folds"]):
                print(f"\n=== Decoding CV fold {cv_idx}/{d['cv_folds']-1} (HD) ===")
                rate_map_hd = _gaussian_filter_hd(rate_maps_hd[cv_idx], d["smooth_sigma"])
                cv_mask = cv_masks[cv_idx]  # True=train, False=test (over T bins)
                test_idx = np.where(~cv_mask)[0]
                print(f"Using CV fold {cv_idx}: decoding {len(test_idx)} held-out bins.")

                spike_counts = (spike_rates[test_idx, :] * d["tau"]).astype(float)
                avg_vel_all = compute_velocity(locations, base_time)
                avg_vel = avg_vel_all[test_idx]
                time_centers = 0.5 * (
                    base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)]
                )

                if d["save_intermediates"]:
                    np.save(save_path / f"hd_rate_map_cv{cv_idx}_{d['n_hd_bins']}.npy", rate_map_hd)
                    np.save(save_path / f"cv_mask{cv_idx}_{d['n_hd_bins']}.npy", cv_mask)
                    np.save(save_path / f"hd_occupancy_map_cv{cv_idx}_{d['n_hd_bins']}.npy", occupancy_maps[cv_idx])

                decoded_hd, posterior_all = hd_bayesian_decoding(
                    rate_map_hd=rate_map_hd,
                    spike_counts=spike_counts,
                    tau=d["tau"],
                    hd_bins=hd_edges,
                    avg_vel=avg_vel,
                    vel_threshold=d["vel_threshold"],
                )

                print("Saving decoding outputs…")
                base = f"decoded_hd_cv{cv_idx}_{d['n_hd_bins']}"
                np.save(save_path / f"{base}.npy", decoded_hd)
                np.save(save_path / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
                np.save(save_path / f"{base}_time_centers.npy", time_centers)

                fold_record: Dict[str, Any] = {
                    "cv_idx": int(cv_idx),
                    "n_test_bins": int(len(test_idx)),
                    "output_prefix": base,
                }

                eval_res = evaluate_errors(decoded_hd, time_centers, locations, velocity_series, d["vel_threshold"])
                np.savez_compressed(save_path / f"errors_hd_cv{cv_idx}_{d['n_hd_bins']}.npz", **eval_res)
                print("Saved error analysis:", save_path / f"errors_hd_cv{cv_idx}_{d['n_hd_bins']}.npz")
                fold_record["error_stats"] = eval_res["stats"]

                cv_summary["folds"].append(fold_record)

            with open(save_path / f"cv_summary_hd_{d['n_hd_bins']}.json", "w") as f:
                json.dump(cv_summary, f, indent=2)
            print("\nWrote CV summary:", save_path / f"cv_summary_hd_{d['n_hd_bins']}.json")
            print(f"\nDone in {(time.time()-t0)/60.0:.2f} min.")
            return {"mode": "hd", "cv_summary": str(save_path / f"cv_summary_hd_{d['n_hd_bins']}.json")}

        else:
            (
                rate_maps_xy,
                occupancy_maps_xy,
                cv_masks,
                (x_edges, y_edges),
            ) = get_xy_rate_maps_cv(
                spike_rates=spike_rates,
                locations=locations,
                base_time=base_time,
                n_x_bins=d["n_x_bins"],
                n_y_bins=d["n_y_bins"],
                velocity_threshold=d["vel_threshold"],
                filter_vel=d["filter_vel"],
                cv_count=d["cv_folds"],
                seed=d["seed"],
            )

            cv_summary: Dict[str, Any] = {"mode": "xy", "folds": [], "n_folds": int(d["cv_folds"])}

            for cv_idx in range(d["cv_folds"]):
                print(f"\n=== Decoding CV fold {cv_idx}/{d['cv_folds']-1} (XY) ===")
                rate_map_xy = rate_maps_xy[cv_idx]
                if d["smooth_sigma_xy"] and d["smooth_sigma_xy"] > 0:
                    rate_map_xy = _gaussian_filter_xy(rate_map_xy, (d["smooth_sigma_xy"], d["smooth_sigma_xy"]))

                cv_mask = cv_masks[cv_idx]  # True=train, False=test
                test_idx = np.where(~cv_mask)[0]
                print(f"Using CV fold {cv_idx}: decoding {len(test_idx)} held-out bins.")

                spike_counts = (spike_rates[test_idx, :] * d["tau"]).astype(float)
                avg_vel_all = compute_velocity(locations, base_time)
                avg_vel = avg_vel_all[test_idx]
                time_centers = 0.5 * (
                    base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)]
                )

                if d["save_intermediates"]:
                    np.save(save_path / f"xy_rate_map_cv{cv_idx}_{d['n_y_bins']}x{d['n_x_bins']}.npy", rate_map_xy)
                    np.save(save_path / f"cv_mask_xy{cv_idx}_{d['n_y_bins']}x{d['n_x_bins']}.npy", cv_mask)
                    np.save(save_path / f"xy_occupancy_map_cv{cv_idx}_{d['n_y_bins']}x{d['n_x_bins']}.npy", occupancy_maps_xy[cv_idx])

                decoded_xy, posterior_all = xy_bayesian_decoding(
                    rate_map_xy=rate_map_xy,
                    spike_counts=spike_counts,
                    tau=d["tau"],
                    avg_vel=avg_vel,
                    vel_threshold=d["vel_threshold"],
                    x_edges=x_edges,
                    y_edges=y_edges,
                )

                print("Saving decoding outputs…")
                base = f"decoded_xy_cv{cv_idx}_{d['n_y_bins']}x{d['n_x_bins']}"
                np.save(save_path / f"{base}.npy", decoded_xy)
                np.save(save_path / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
                np.save(save_path / f"{base}_time_centers.npy", time_centers)

                fold_record: Dict[str, Any] = {
                    "cv_idx": int(cv_idx),
                    "n_test_bins": int(len(test_idx)),
                    "output_prefix": base,
                }

                eval_res = evaluate_errors_xy(decoded_xy, time_centers, locations, velocity_series, d["vel_threshold"])
                np.savez_compressed(save_path / f"errors_xy_cv{cv_idx}_{d['n_y_bins']}x{d['n_x_bins']}.npz", **eval_res)
                print("Saved error analysis:", save_path / f"errors_xy_cv{cv_idx}_{d['n_y_bins']}x{d['n_x_bins']}.npz")
                fold_record["error_stats"] = eval_res["stats"]

                cv_summary["folds"].append(fold_record)

            with open(save_path / f"cv_summary_xy_{d['n_y_bins']}x{d['n_x_bins']}.json", "w") as f:
                json.dump(cv_summary, f, indent=2)
            print("\nWrote CV summary:", save_path / f"cv_summary_xy_{d['n_y_bins']}x{d['n_x_bins']}.json")
            print(f"\nDone in {(time.time()-t0)/60.0:.2f} min.")
            return {"mode": "xy", "cv_summary": str(save_path / f"cv_summary_xy_{d['n_y_bins']}x{d['n_x_bins']}.json")}

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make HD or XY rate maps from Pfieffer and do singular Bayesian decoding."
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
    parser.add_argument("--mode", type=str, choices=["hd", "xy"], default="hd",
                        help="Decode head direction (hd) or location (xy)")
    parser.add_argument("--n_hd_bins", type=int, default=60, help="Number of head-direction bins")
    parser.add_argument("--n_x_bins", type=int, default=30, help="Number of X bins for spatial map (xy mode)")
    parser.add_argument("--n_y_bins", type=int, default=30, help="Number of Y bins for spatial map (xy mode)")
    parser.add_argument("--tau", type=float, default=0.120, help="Time bin size (s)")
    parser.add_argument("--vel_threshold", type=float, default=5.0, help="Velocity threshold (cm/s)")
    parser.add_argument("--cv_folds", type=int, default=None,
                        help="Number of CV folds; 0 = no CV. If omitted, you will be prompted.")
    parser.add_argument("--filter_vel", action="store_true", default=True, help="Filter low-velocity bins")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    parser.add_argument("--smooth_sigma", type=float, default=2.0,
                        help="Sigma for Gaussian smoothing of HD curves (in bins; circular)")
    parser.add_argument("--smooth_sigma_xy", type=float, default=1.0,
                        help="Sigma (in bins) for Gaussian smoothing of XY maps (applied identically to Y and X)")
    parser.add_argument("--save_intermediates", action="store_true", default=False,
                        help="If set, save intermediate artifacts (rate maps, masks, etc.)")
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    decode_solo(args)

if __name__ == "__main__":
    main()
