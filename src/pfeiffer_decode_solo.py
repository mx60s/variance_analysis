#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict, Any
from math import ceil

import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# utilities

def compute_velocity(locations: np.ndarray, base_time: np.ndarray) -> np.ndarray:
    time_step = float(locations[3, 0] - locations[2, 0])
    steps = np.diff(locations[:, 1:3], axis=0)
    steps = np.vstack((steps, steps[-1]))
    speed_samples = np.linalg.norm(steps, axis=1) / max(time_step, 1e-9)

    v = np.zeros(len(base_time) - 1, dtype=float)
    for t in range(len(base_time) - 1):
        idx = np.where((locations[:, 0] >= base_time[t]) & (locations[:, 0] < base_time[t + 1]))[0]
        v[t] = 0.0 if idx.size == 0 else float(np.mean(speed_samples[idx]))
    return v


def compute_velocity_series(locations: np.ndarray) -> np.ndarray:
    time_step = float(locations[3, 0] - locations[2, 0])
    steps = np.diff(locations[:, 1:3], axis=0)
    steps = np.vstack((steps, steps[-1]))
    speed_samples = np.linalg.norm(steps, axis=1) / max(time_step, 1e-9)
    
    return np.column_stack((locations[:, 0], speed_samples))


def gaussian_filter_hd(rate_curve: np.ndarray, smooth_sigma: float) -> np.ndarray:
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

    loc_times = locations[in_hd, 0]
    loc_idx = np.array([int(np.abs(base_time - t).argmin()) for t in loc_times], dtype=int)

    if fast_bins is not None and fast_bins.size:
        mask = np.any(np.isclose(base_time[loc_idx, None], fast_bins, atol=timebin_tol), axis=1)
        loc_idx = loc_idx[mask]

    T = len(base_time) - 1
    return loc_idx[(loc_idx >= 0) & (loc_idx < T)]


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

    loc_times = locations[in_xy, 0]
    loc_idx = np.array([int(np.abs(base_time - t).argmin()) for t in loc_times], dtype=int)

    if fast_bins is not None and fast_bins.size:
        mask = np.any(np.isclose(base_time[loc_idx, None], fast_bins, atol=timebin_tol), axis=1)
        loc_idx = loc_idx[mask]

    T = len(base_time) - 1
    return loc_idx[(loc_idx >= 0) & (loc_idx < T)]
    

def get_hd_rate_maps_cv(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_hd_bins: int = 60,
    velocity_threshold: float = 5.0,
    cv_count: int = 4,
    seed: int = 777,
):
    rng = np.random.RandomState(seed)

    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if velocity_threshold > 0.0 else None
    print(
        f"Filtering {100.0 - (len(fast_idx) / max(len(base_time)-1,1)) * 100.0:.1f}% of bins due to low velocity"
    )

    hd_edges = np.linspace(0.0, 360.0 + 1e-2, n_hd_bins + 1)

    T, N = spike_rates.shape

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
        idx = select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_bins)
        if idx.size == 0:
            continue

        for cv in range(cv_count):
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
    seed: int = 777,
):
    rng = np.random.RandomState(seed)

    v = compute_velocity(locations, base_time)
    fast_idx = np.where(v >= velocity_threshold)[0]
    fast_bins = base_time[fast_idx] if velocity_threshold > 0 else None
    print(
        f"Filtering {100.0 - (len(fast_idx) / max(len(base_time)-1,1)) * 100.0:.1f}% of bins due to low velocity"
    )

    xmin, xmax = 0., 201.
    ymin, ymax = 0., 201.
    
    n_x_bins = n_xy_bins
    n_y_bins = n_xy_bins
    
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

# decoding

def hd_bayesian_decoding(
    rate_map_hd: np.ndarray,
    spike_counts: np.ndarray,
    tau: float,
    hd_bins: np.ndarray,
    avg_vel: np.ndarray,
    vel_threshold: float = 0.0,
    eps: float = 1e-10,
):
    T = spike_counts.shape[0]
    H, N = rate_map_hd.shape

    decoded_hd = np.full(T, np.nan)
    posterior_all: List[np.ndarray] = []

    rate_tau = rate_map_hd * tau
    valid_h = np.any(np.isfinite(rate_tau), axis=1)
    rate_tau = np.nan_to_num(rate_tau, nan=0.0)

    for t in range(T):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((H,), np.nan))
            continue

        k = spike_counts[t][None, :] 
        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k * np.log(rate_tau + eps) - rate_tau, axis=1) 
        log_like[~valid_h] = -np.inf

        if not np.any(np.isfinite(log_like)):
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
    rate_map_xy: np.ndarray,   
    spike_counts: np.ndarray,  
    tau: float,
    avg_vel: np.ndarray,       
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    vel_threshold: float = 0.0,
    eps: float = 1e-10,
):
    T, N = spike_counts.shape
    Y, X, N_map = rate_map_xy.shape

    decoded_xy = np.full((T, 2), np.nan)
    posterior_all: List[np.ndarray] = []

    rate_tau = rate_map_xy * tau  
    rate_tau_flat = np.nan_to_num(rate_tau.reshape(Y * X, N), nan=0.0)

    for t in range(T):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((Y, X), np.nan))
            continue

        k = spike_counts[t] 

        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k[None, :] * np.log(rate_tau_flat + eps) - rate_tau_flat, axis=1)

        if np.all(~np.isfinite(log_like)):
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

        x_c = 0.5 * (x_edges[ix] + x_edges[ix + 1])
        y_c = 0.5 * (y_edges[iy] + y_edges[iy + 1])
        decoded_xy[t, 0] = x_c
        decoded_xy[t, 1] = y_c

    return decoded_xy, posterior_all


# ==================== CHANCE-LEVEL DECODING ====================
# These functions decode based purely on occupancy (the prior), ignoring neural activity.
# This provides a baseline: "how well would we do just guessing based on where the animal spends time?"


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angular distance in degrees, wrapped to [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def hd_chance_decoding(
    occupancy: np.ndarray,
    n_test_bins: int,
    hd_edges: np.ndarray,
    avg_vel: np.ndarray,
    vel_threshold: float = 0.0,
    mode: str = "mode",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chance-level decoding for HD based on occupancy distribution.
    
    Parameters
    ----------
    occupancy : np.ndarray
        Occupancy counts per HD bin (from training data).
    n_test_bins : int
        Number of test time bins to decode.
    hd_edges : np.ndarray
        Bin edges for HD.
    avg_vel : np.ndarray
        Average velocity per test bin.
    vel_threshold : float
        Velocity threshold for valid bins.
    mode : str
        'mode' - always predict most occupied bin (deterministic baseline)
        'sample' - sample from occupancy distribution (stochastic baseline)
    rng : np.random.Generator, optional
        Random generator for 'sample' mode.
        
    Returns
    -------
    decoded_hd : np.ndarray
        Chance-decoded HD values.
    occupancy_prior : np.ndarray
        Normalized occupancy distribution used as prior.
    """
    # Normalize occupancy to get prior P(HD)
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total_occ = occ.sum()
    if total_occ > 0:
        prior = occ / total_occ
    else:
        prior = np.ones_like(occ) / len(occ)  # uniform if no occupancy
    
    decoded_hd = np.full(n_test_bins, np.nan)
    
    if mode == "mode":
        # Always predict the most occupied bin
        mode_idx = int(np.argmax(prior))
        mode_hd = 0.5 * (hd_edges[mode_idx] + hd_edges[mode_idx + 1])
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            decoded_hd[t] = mode_hd
    elif mode == "sample":
        # Sample from occupancy distribution
        if rng is None:
            rng = np.random.default_rng()
        bin_centers = 0.5 * (hd_edges[:-1] + hd_edges[1:])
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            idx = rng.choice(len(prior), p=prior)
            decoded_hd[t] = bin_centers[idx]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mode' or 'sample'.")
    
    return decoded_hd, prior


def xy_chance_decoding(
    occupancy: np.ndarray,
    n_test_bins: int,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    avg_vel: np.ndarray,
    vel_threshold: float = 0.0,
    mode: str = "mode",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chance-level decoding for XY based on occupancy distribution.
    
    Parameters
    ----------
    occupancy : np.ndarray
        Occupancy counts per XY bin (shape: n_y_bins x n_x_bins, from training data).
    n_test_bins : int
        Number of test time bins to decode.
    x_edges, y_edges : np.ndarray
        Bin edges for X and Y.
    avg_vel : np.ndarray
        Average velocity per test bin.
    vel_threshold : float
        Velocity threshold for valid bins.
    mode : str
        'mode' - always predict most occupied bin (deterministic baseline)
        'sample' - sample from occupancy distribution (stochastic baseline)
    rng : np.random.Generator, optional
        Random generator for 'sample' mode.
        
    Returns
    -------
    decoded_xy : np.ndarray
        Chance-decoded XY values (shape: n_test_bins x 2).
    occupancy_prior : np.ndarray
        Normalized occupancy distribution used as prior.
    """
    Y, X = occupancy.shape
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total_occ = occ.sum()
    if total_occ > 0:
        prior = occ / total_occ
    else:
        prior = np.ones_like(occ) / occ.size
    
    decoded_xy = np.full((n_test_bins, 2), np.nan)
    
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    if mode == "mode":
        mode_idx_flat = int(np.argmax(prior))
        iy, ix = divmod(mode_idx_flat, X)
        mode_x = x_centers[ix]
        mode_y = y_centers[iy]
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            decoded_xy[t, 0] = mode_x
            decoded_xy[t, 1] = mode_y
    elif mode == "sample":
        if rng is None:
            rng = np.random.default_rng()
        prior_flat = prior.ravel()
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            idx = rng.choice(len(prior_flat), p=prior_flat)
            iy, ix = divmod(idx, X)
            decoded_xy[t, 0] = x_centers[ix]
            decoded_xy[t, 1] = y_centers[iy]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mode' or 'sample'.")
    
    return decoded_xy, prior


def compute_chance_error_analytical_hd(
    occupancy: np.ndarray,
    hd_edges: np.ndarray,
) -> Dict[str, float]:
    """
    Compute expected error analytically for HD chance decoding.
    
    This computes E[|error|] = sum_i P(true=i) * |angle_diff(i, mode)|
    For mode decoding: always guess the most occupied bin.
    
    Returns expected mean absolute error under chance model.
    """
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total = occ.sum()
    if total <= 0:
        return {'expected_mae_deg': np.nan, 'mode_bin_probability': np.nan}
    
    prior = occ / total
    n_bins = len(prior)
    bin_centers = 0.5 * (hd_edges[:-1] + hd_edges[1:])
    
    # Mode prediction
    mode_idx = int(np.argmax(prior))
    mode_angle = bin_centers[mode_idx]
    
    # Expected error: sum over true positions weighted by their probability
    expected_error = 0.0
    for i in range(n_bins):
        err = np.abs(angular_distance_deg(np.array([bin_centers[i]]), np.array([mode_angle]))[0])
        expected_error += prior[i] * err
    
    return {
        'expected_mae_deg': float(expected_error),
        'mode_bin_probability': float(prior[mode_idx]),
        'mode_bin_center_deg': float(mode_angle),
    }


def compute_chance_error_analytical_xy(
    occupancy: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> Dict[str, float]:
    """
    Compute expected error analytically for XY chance decoding.
    
    Returns expected mean absolute error (Euclidean distance) under chance model.
    """
    Y, X = occupancy.shape
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total = occ.sum()
    if total <= 0:
        return {'expected_mae_cm': np.nan, 'mode_bin_probability': np.nan}
    
    prior = occ / total
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    mode_idx_flat = int(np.argmax(prior))
    iy_mode, ix_mode = divmod(mode_idx_flat, X)
    mode_x = x_centers[ix_mode]
    mode_y = y_centers[iy_mode]
    
    expected_error = 0.0
    for iy in range(Y):
        for ix in range(X):
            dist = np.sqrt((x_centers[ix] - mode_x)**2 + (y_centers[iy] - mode_y)**2)
            expected_error += prior[iy, ix] * dist
    
    return {
        'expected_mae_cm': float(expected_error),
        'mode_bin_probability': float(prior[iy_mode, ix_mode]),
        'mode_x': float(mode_x),
        'mode_y': float(mode_y),
    }


def compute_no_prior_chance_hd() -> Dict[str, float]:
    """
    Compute expected error for HD decoding with no prior (uniform random guessing).

    If we guess uniformly at random from [0, 360), the expected absolute
    circular error is exactly 90 degrees.

    Derivation: The absolute circular error is uniformly distributed on [0, 180],
    so E[|error|] = 90.
    """
    return {
        'no_prior_expected_mae_deg': 90.0,
        'description': 'Uniform random guessing over [0, 360) degrees'
    }


def compute_no_prior_chance_xy(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> Dict[str, float]:
    """
    Compute expected error for XY decoding with no prior (uniform random guessing).

    For a square environment of side L, the expected Euclidean distance between
    two uniformly random points is approximately 0.5214 * L.

    Exact formula: L * (2 + sqrt(2) + 5*log(1+sqrt(2))) / 15
    """
    import math

    L_x = x_edges[-1] - x_edges[0]
    L_y = y_edges[-1] - y_edges[0]

    # For a square, use the exact formula
    # For a rectangle, use average of the two dimensions as approximation
    L = (L_x + L_y) / 2.0

    # Exact coefficient for square: (2 + sqrt(2) + 5*ln(1+sqrt(2))) / 15 ≈ 0.5214
    sqrt2 = math.sqrt(2)
    coeff = (2 + sqrt2 + 5 * math.log(1 + sqrt2)) / 15

    expected_error = coeff * L

    return {
        'no_prior_expected_mae_cm': float(expected_error),
        'environment_size_x_cm': float(L_x),
        'environment_size_y_cm': float(L_y),
        'description': 'Uniform random guessing over environment'
    }


# error analysis helpers

def unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    rad_unwrapped = np.unwrap(rad, discont=np.pi)
    return np.rad2deg(rad_unwrapped)


def interp_circular_deg(t_src: np.ndarray, t_tgt: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    t_src = t_src[order]
    angles_deg = angles_deg[order]
    ang_unwrapped = unwrap_deg(angles_deg)
    ang_interp_unwrapped = np.interp(t_tgt, t_src, ang_unwrapped)
    ang_interp = np.mod(ang_interp_unwrapped, 360.0)
    return ang_interp


def compute_ground_truth_hd(positions: np.ndarray, time_centers: np.ndarray, velocity_series: np.ndarray, velocity_filter: float, window: float = 0.06) -> np.ndarray:
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


def evaluate_errors_hd(decoded_hd: np.ndarray, time_centers: np.ndarray, positions: np.ndarray, velocity_series: np.ndarray, vel_threshold: float) -> Dict[str, Any]:
    gt = compute_ground_truth_hd(positions, time_centers, velocity_series, vel_threshold)
    err = (decoded_hd - gt + 180.0) % 360.0 - 180.0
    hd_err_abs = np.abs(err)
    total = int(hd_err_abs.size)
    valid_mask = ~np.isnan(hd_err_abs)
    valid = int(np.sum(valid_mask))
    stats = {
        'hd_mean_abs_error_deg': float(np.nanmean(hd_err_abs)),
        'hd_median_abs_error_deg': float(np.nanmedian(hd_err_abs)),
        'hd_std_abs_error_deg': float(np.nanstd(hd_err_abs)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors_deg': err}


def interp_linear_series(t_src: np.ndarray, t_tgt: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    return np.interp(t_tgt, t_src[order], v_src[order])


def compute_ground_truth_xy(positions: np.ndarray, time_centers: np.ndarray, velocity_series: np.ndarray, velocity_filter: float, window: float = 0.06) -> np.ndarray:
    t = positions[:, 0]
    x = positions[:, 1]
    y = positions[:, 2]
    order = np.argsort(t)
    x_interp = np.interp(time_centers, t[order], x[order])
    y_interp = np.interp(time_centers, t[order], y[order])

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
    dists = np.sqrt(np.nansum(diffs**2, axis=1)) 
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
    

def decode_solo(
    *,
    data_path: str = "/home/mag/variance-analysis/data/pfeiffer",
    save_path: str = "/home/mag/variance-analysis/results/pfeiffer",
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
    locations = loadmat(str(data_path / 'Position_Data.mat'))['Position_Data']
    spike_data = loadmat(str(data_path / 'Spike_Data.mat'))
    spike_times_raw = spike_data['Spike_Data']  # columns: [time, neuron_id]

    excit = spike_data['Excitatory_Neurons']
    excit_list = np.asarray(excit).ravel().astype(spike_data['Spike_Data'].dtype)
    
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
        seed=seed,
    )

    (cv_rate_maps_xy, occ_maps_xy, cv_masks_xy, (x_edges, y_edges)) = get_xy_rate_maps_cv(
        spike_rates=spike_rates,
        locations=locations,
        base_time=base_time,
        n_xy_bins=n_xy_bins,
        velocity_threshold=vel_threshold,
        cv_count=cv_folds,
        seed=seed,
    )

    cv_summary: Dict[str, Any] = {"folds": [], "n_folds": int(cv_folds)}

    for cv_idx in range(cv_folds):
        print(f"\n=== Decoding CV fold {cv_idx}/{cv_folds-1} (HD & XY) ===")

        # HD
        rate_map_hd = gaussian_filter_hd(cv_rate_maps_hd[cv_idx], smooth_sigma_hd)
        occ_hd = occ_maps_hd[cv_idx]
        mask_hd = cv_masks[cv_idx]  # True=train, False=test
        test_idx = np.where(~mask_hd)[0]

        sc = (spike_rates[test_idx, :] * tau).astype(float)
        if smooth_spikes_sigma_s and smooth_spikes_sigma_s > 0:
            sigma_bins_t = max(1e-9, smooth_spikes_sigma_s / tau)
            sc = gaussian_filter1d(sc, sigma=sigma_bins_t, axis=0, mode="nearest")
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

        # XY
        rate_map_xy = cv_rate_maps_xy[cv_idx]
        occ_xy = occ_maps_xy[cv_idx]
        if smooth_sigma_xy and smooth_sigma_xy > 0:
            rate_map_xy = gaussian_filter_xy(rate_map_xy, (smooth_sigma_xy, smooth_sigma_xy))
        mask_xy = cv_masks_xy[cv_idx]
        test_idx_xy = np.where(~mask_xy)[0]
        avg_vel_xy = avg_vel_bins[test_idx_xy]
        tc_xy = time_centers_all[test_idx_xy]

        sc_xy = (spike_rates[test_idx_xy, :] * tau).astype(float)
        if smooth_spikes_sigma_s and smooth_spikes_sigma_s > 0:
            sigma_bins_t = max(1e-9, smooth_spikes_sigma_s / tau)
            sc_xy = gaussian_filter1d(sc_xy, sigma=sigma_bins_t, axis=0, mode="nearest")

        decoded_xy, post_xy = xy_bayesian_decoding(
            rate_map_xy=rate_map_xy,
            spike_counts=sc_xy,
            tau=tau,
            avg_vel=avg_vel_xy,
            x_edges=x_edges,
            y_edges=y_edges,
            vel_threshold=vel_threshold,
        )
        
        n_x_bins = len(x_edges) - 1
        n_y_bins = len(y_edges) - 1

        base_xy = f"decoded_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}"
        np.save(save_path / f"{base_xy}.npy", decoded_xy)
        np.save(save_path / f"{base_xy}_posterior.npy", np.array(post_xy, dtype=object))
        np.save(save_path / f"{base_xy}_time_centers.npy", tc_xy)
        if save_intermediates:
            np.save(save_path / f"xy_rate_map_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", rate_map_xy)
            np.save(save_path / f"xy_occupancy_map_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", occ_xy)

        eval_xy = evaluate_errors_xy(decoded_xy, tc_xy, locations, velocity_series, vel_threshold)
        np.savez_compressed(save_path / f"errors_xy_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npz", **eval_xy)

        # XY Chance decoding
        eval_xy_chance = None
        if run_chance_decoding:
            print(f"  -> XY chance decoding (mode='{chance_mode}')")
            decoded_xy_chance, xy_prior = xy_chance_decoding(
                occupancy=occ_xy,
                n_test_bins=len(test_idx_xy),
                x_edges=x_edges,
                y_edges=y_edges,
                avg_vel=avg_vel_xy,
                vel_threshold=vel_threshold,
                mode=chance_mode,
                rng=rng,
            )
            np.save(save_path / f"decoded_xy_chance_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", decoded_xy_chance)
            np.save(save_path / f"xy_occupancy_prior_cv{cv_idx}_{n_y_bins}x{n_x_bins}.npy", xy_prior)
            
            eval_xy_chance = evaluate_errors_xy(decoded_xy_chance, tc_xy, locations, velocity_series, vel_threshold)
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
        default="/home/mag/variance-analysis/data/pfeiffer",
        help="Root data directory (contains Rat{rat}/Open1)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/mag/variance-analysis/results/pfeiffer",
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