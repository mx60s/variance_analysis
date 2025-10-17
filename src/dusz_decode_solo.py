#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# utilities

def _compute_avg_velocity_per_bin(base_time: np.ndarray, velocity_series: np.ndarray) -> np.ndarray:
    T = len(base_time) - 1
    avg_vel = np.full(T, np.nan, dtype=float)
    v_t = velocity_series[:, 0]
    v_v = velocity_series[:, 1]
    for i in range(T):
        m = (v_t >= base_time[i]) & (v_t < base_time[i + 1])
        if np.any(m):
            avg_vel[i] = float(np.nanmean(v_v[m]))
    return avg_vel

def _interp_circular_deg(t_src: np.ndarray, t_tgt: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    t_src = t_src[order]
    angles_deg = angles_deg[order]
    ang_unwrapped = _unwrap_deg(angles_deg)
    ang_interp_unwrapped = np.interp(t_tgt, t_src, ang_unwrapped)
    ang_interp = np.mod(ang_interp_unwrapped, 360.0)
    return ang_interp

def _times_to_bin_indices(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, times, side='right') - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    return idx[valid]

def _circular_smooth_deg(angles_deg: np.ndarray, smoothing_window: int = 5) -> np.ndarray:
    if smoothing_window is None or smoothing_window <= 1:
        return angles_deg % 360.0
    ang_rad = np.deg2rad(angles_deg % 360.0)
    sin_vals = np.sin(ang_rad)
    cos_vals = np.cos(ang_rad)
    kernel = np.ones(smoothing_window) / float(smoothing_window)
    smooth_sin = np.convolve(sin_vals, kernel, mode='same')
    smooth_cos = np.convolve(cos_vals, kernel, mode='same')
    return (np.rad2deg(np.arctan2(smooth_sin, smooth_cos)) % 360.0)


def _get_md_from_xy(positions: np.ndarray, smoothing_window: int = 20) -> np.ndarray:
    x = positions[:, 1]
    y = positions[:, 2]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    direction = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
    direction = _circular_smooth_deg(direction, smoothing_window=smoothing_window)
    # Pad to same length as positions
    direction = np.concatenate([direction, [direction[-1]]])
    return direction

def _gaussian_filter_hd(rate_curve: np.ndarray, smooth_sigma: Optional[float]) -> np.ndarray:
    """NaN-aware circular 1D Gaussian smoothing across head/MD bins."""
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


def _gaussian_filter_xy(rate_map: np.ndarray, smooth_sigma_xy: Optional[Tuple[float, float]]) -> np.ndarray:
    # unused for now
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


def _select_time_indices_in_hd_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    hd_l: Tuple[float, float],
    fast_mask: Optional[np.ndarray],
) -> np.ndarray:
    l0, l1 = hd_l
    in_hd = np.where((locations[:, 3] >= l0) & (locations[:, 3] < l1))[0]
    if in_hd.size == 0:
        return np.array([], dtype=int)
    loc_times = locations[in_hd, 0]
    idx = _times_to_bin_indices(loc_times, base_time)
    T = len(base_time) - 1
    idx = idx[(idx >= 0) & (idx < T)]
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx


def _select_time_indices_in_md_bin(
    positions: np.ndarray,
    base_time: np.ndarray,
    md_series_deg: np.ndarray,
    md_l: Tuple[float, float],
    fast_mask: Optional[np.ndarray],
) -> np.ndarray:
    l0, l1 = md_l
    in_md = np.where((md_series_deg >= l0) & (md_series_deg < l1))[0]
    if in_md.size == 0:
        return np.array([], dtype=int)
    loc_times = positions[in_md, 0]
    idx = _times_to_bin_indices(loc_times, base_time)
    T = len(base_time) - 1
    idx = idx[(idx >= 0) & (idx < T)]
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx


def _select_time_indices_in_xy_bin(
    locations: np.ndarray,
    base_time: np.ndarray,
    x_win: Tuple[float, float],
    y_win: Tuple[float, float],
    fast_mask: Optional[np.ndarray],
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
    idx = _times_to_bin_indices(loc_times, base_time)
    T = len(base_time) - 1
    idx = idx[(idx >= 0) & (idx < T)]
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx
    

def _unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    rad_unwrapped = np.unwrap(rad, discont=np.pi)
    return np.rad2deg(rad_unwrapped)


def _interp_linear_series(t_src: np.ndarray, t_tgt: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    return np.interp(t_tgt, t_src[order], v_src[order])


def compute_ground_truth_md(positions: np.ndarray, time_centers: np.ndarray, velocity_series: np.ndarray,
                            velocity_filter: float, md_series_deg: np.ndarray, window: float = 0.06) -> np.ndarray:
    """MD 'ground truth' by interpolating MD series at time_centers, masked by speed."""
    tpos = positions[:, 0]
    md_interp = _interp_circular_deg(tpos, time_centers, md_series_deg)

    v_times = velocity_series[:, 0]
    v_vals = velocity_series[:, 1]
    velocity_interp = np.full(len(time_centers), np.nan)
    for i, tc in enumerate(time_centers):
        mask = (v_times >= tc - window) & (v_times <= tc + window)
        if np.any(mask):
            velocity_interp[i] = float(np.nanmean(v_vals[mask]))

    valid_velocity = (velocity_interp >= velocity_filter) & (~np.isnan(velocity_interp))
    ground_md_filtered = np.where(valid_velocity, md_interp, np.nan)
    return ground_md_filtered


def evaluate_errors_hd(decoded_hd: np.ndarray, time_centers: np.ndarray,
                       positions: np.ndarray, velocity_series: np.ndarray, vel_threshold: float) -> Dict[str, Any]:
    
    # compute the ground truth hd
    window = 0.06
    timestamps = positions[:, 0]
    hd = positions[:, 3]
    hd_interp = _interp_circular_deg(timestamps, time_centers, hd)

    v_times = velocity_series[:, 0]
    v_vals = velocity_series[:, 1]
    velocity_interp = np.full(len(time_centers), np.nan)
    for i, tc in enumerate(time_centers):
        mask = (v_times >= tc - window) & (v_times <= tc + window)
        if np.any(mask):
            velocity_interp[i] = float(np.nanmean(v_vals[mask]))

    valid_velocity = (velocity_interp >= vel_threshold) & (~np.isnan(velocity_interp))
    gt = np.where(valid_velocity, hd_interp, np.nan)
    
    hd_err_abs = angular_distance_deg(decoded_hd, gt)
    total = int(hd_err_abs.size)
    valid = int(np.sum(~np.isnan(hd_err_abs)))
    stats = {
        'hd_mean_abs_error_deg': float(np.nanmean(np.abs(hd_err_abs))),
        'hd_median_abs_error_deg': float(np.nanmedian(np.abs(hd_err_abs))),
        'hd_std_abs_error_deg': float(np.nanstd(hd_err_abs)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors_deg': hd_err_abs}


def evaluate_errors_md(decoded_md: np.ndarray, time_centers: np.ndarray, positions: np.ndarray,
                       velocity_series: np.ndarray, vel_threshold: float, md_series_deg: np.ndarray) -> Dict[str, Any]:
    tpos = positions[:, 0]
    md_interp = _interp_circular_deg(tpos, time_centers, md_series_deg)

    window = 0.06
    v_times = velocity_series[:, 0]
    v_vals = velocity_series[:, 1]
    velocity_interp = np.full(len(time_centers), np.nan)
    for i, tc in enumerate(time_centers):
        mask = (v_times >= tc - window) & (v_times <= tc + window)
        if np.any(mask):
            velocity_interp[i] = float(np.nanmean(v_vals[mask]))

    valid_velocity = (velocity_interp >= vel_threshold) & (~np.isnan(velocity_interp))
    gt = np.where(valid_velocity, md_interp, np.nan)
    
    md_err_abs = angular_distance_deg(decoded_md, gt)
    total = int(md_err_abs.size)
    valid = int(np.sum(~np.isnan(md_err_abs)))
    stats = {
        'md_mean_abs_error_deg': float(np.nanmean(np.abs(md_err_abs))),
        'md_median_abs_error_deg': float(np.nanmedian(np.abs(md_err_abs))),
        'md_std_abs_error_deg': float(np.nanstd(md_err_abs)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors_deg': md_err_abs}


def compute_ground_truth_xy(positions: np.ndarray, time_centers: np.ndarray,
                            velocity_series: np.ndarray, velocity_filter: float,
                            window: float = 0.06) -> np.ndarray:
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


def evaluate_errors_xy(decoded_xy: np.ndarray, time_centers: np.ndarray,
                       positions: np.ndarray, velocity_series: np.ndarray, vel_threshold: float) -> Dict[str, Any]:
    gt_xy = compute_ground_truth_xy(positions, time_centers, velocity_series, vel_threshold)
    diffs = decoded_xy - gt_xy  # (T,2)
    dists = np.sqrt(np.nansum(diffs**2, axis=1))  # Euclidean
    total = int(dists.size)
    valid = int(np.sum(~np.isnan(dists)))
    stats = {
        'xy_mean_abs_error': float(np.nanmean(dists)),
        'xy_median_abs_error': float(np.nanmedian(dists)),
        'xy_std_abs_error': float(np.nanstd(dists)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors': dists}


def extract_matlab_obj(obj):
    from scipy.io.matlab._mio5_params import mat_struct
    if not (isinstance(obj, mat_struct) or isinstance(obj, dict)):
        return obj
    out: Dict[str, Any] = {}
    obj = obj.__dict__ if isinstance(obj, mat_struct) else obj
    for key, val in obj.items():
        if key in ['_fieldnames', 'type', 'time_unit', 'data_unit', 'name']:
            continue
        if isinstance(val, mat_struct) or isinstance(val, dict):
            out[key] = extract_matlab_obj(val)
        elif isinstance(val, np.ndarray):
            arr = [extract_matlab_obj(v) for v in val]
            out[key] = np.array(arr)
        else:
            out[key] = val
    return out


def load_dusz_data(base_path: Path, dataset: int, rat: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    dpath = base_path / f"Dataset_{dataset}" / rat / "Data"

    # Epochs
    epochs = loadmat(dpath / 'BehavEpochs.mat', struct_as_record=False, squeeze_me=True)
    try:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    except Exception:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    if rat == 'A3708-200317b':
        epoch_data = extract_matlab_obj(epochs['wake2Ep'])

    try:
        tracking = pd.read_csv(dpath / 'Tracking_data.csv')
        tracking = tracking.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    except Exception:
        tracking = pd.read_csv(dpath / 'Tracking_data.csv')
        tracking = tracking.set_axis(['timestamp', 'x', 'y', 'hd'], axis=1)
    positions = tracking[['timestamp', 'x', 'y', 'hd']].to_numpy()

    if not isinstance(epoch_data['start'], float):
        epoch_data['start'] = epoch_data['start'][0]
        epoch_data['stop'] = epoch_data['stop'][0]
    mask_pos = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[mask_pos]

    # Normalize positions to positive origin
    positions[:, 1] += abs(float(np.nanmin(positions[:, 1])))
    positions[:, 2] += abs(float(np.nanmin(positions[:, 2])))
    if positions[:, 1].max() < 1:  # same quirk as original
        positions[:, 1] *= 100
        positions[:, 2] *= 100

    cell_types_raw = loadmat(dpath / 'CellTypes.mat', struct_as_record=False, squeeze_me=True)
    cell_types = extract_matlab_obj(cell_types_raw.get('CellTypes', cell_types_raw))
    hd_list = np.asarray(cell_types.get('hd', cell_types_raw.get('hd'))).astype(int).ravel()
    ex_list = np.asarray(cell_types.get('ex', cell_types_raw.get('ex'))).astype(int).ravel()

    spike_raw = loadmat(dpath / 'SpikeData.mat', struct_as_record=False, squeeze_me=True)
    S = extract_matlab_obj(spike_raw.get('S', spike_raw))
    arr = []
    for i, cell in enumerate(S['C']):
        ts = np.asarray(cell['tsd']['t']).ravel()
        if ts.size:
            arr.append(np.column_stack([ts, np.full(ts.size, i, dtype=int)]))
    spike_times = np.vstack(arr) if arr else np.empty((0, 2))
    spike_times = spike_times[np.argsort(spike_times[:, 0])]
    mask_spk = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[mask_spk]

    vel_raw = loadmat(dpath / 'Velocity.mat', struct_as_record=False, squeeze_me=True)
    vel = extract_matlab_obj(vel_raw.get('vel', vel_raw))
    velocity_series = np.dstack((vel['t'], vel['data']))[0]
    mask_v = (velocity_series[:, 0] > epoch_data['start']) & (velocity_series[:, 0] < epoch_data['stop'])
    velocity_series = velocity_series[mask_v]

    return spike_times, positions, {'hd': hd_list, 'ex': ex_list}, velocity_series


def build_spike_rates(spike_times: np.ndarray, base_time: np.ndarray, tau: float, neuron_ids: np.ndarray) -> np.ndarray:
    T = len(base_time) - 1
    N = len(neuron_ids)
    rates = np.zeros((T, N), dtype=float)
    if spike_times.size == 0 or N == 0:
        return rates
    for n_idx, neuron_id in enumerate(neuron_ids):
        neuron_spk = spike_times[spike_times[:, 1] == neuron_id, 0]
        if neuron_spk.size:
            counts, _ = np.histogram(neuron_spk, bins=base_time)
            rates[:, n_idx] = counts.astype(float) / max(tau, 1e-9)
    return rates


# ------------------ HD/MD/XY tuning ------------- #

def get_hd_rate_map_all(
    spike_rates: np.ndarray, locations: np.ndarray, base_time: np.ndarray,
    n_hd_bins: int = 60, velocity_threshold: float = 5.0, avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if avg_vel_bins is not None else None
    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)
    T, N = spike_rates.shape
    rate_map = np.full((n_hd_bins, N), np.nan, dtype=float)
    occupancy = np.zeros((n_hd_bins,), dtype=float)
    for l in tqdm(range(n_hd_bins), desc="HD bins (all-data)"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_mask)
        if idx.size:
            occupancy[l] = float(idx.size)
            rate_map[l, :] = np.nanmean(spike_rates[idx, :], axis=0)
    return rate_map, occupancy, hd_edges


def get_hd_rate_maps_cv(
    spike_rates: np.ndarray, locations: np.ndarray, base_time: np.ndarray,
    n_hd_bins: int = 60, velocity_threshold: float = 5.0, cv_count: int = 4,
    avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if avg_vel_bins is not None else None
    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)
    T, N = spike_rates.shape
    folds = np.array_split(np.arange(T), cv_count)
    cv_masks = []
    for cv in range(cv_count):
        mask = np.ones(T, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)
    shape_map = (n_hd_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_hd_bins,), dtype=float) for _ in range(cv_count)]
    for l in tqdm(range(n_hd_bins), desc="HD bins (CV)"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_mask)
        if idx.size == 0:
            continue
        for cv in range(cv_count):
            mask_cv = cv_masks[cv]  # True=train
            valid = idx[idx < mask_cv.size]
            cv_idx = valid[mask_cv[valid]]
            occupancy_maps[cv][l] = float(cv_idx.size)
            if cv_idx.size:
                cv_rate_maps[cv][l, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)
    return cv_rate_maps, occupancy_maps, cv_masks, hd_edges


def get_md_rate_maps_cv(
    spike_rates: np.ndarray, positions: np.ndarray, base_time: np.ndarray, md_series_deg: np.ndarray,
    n_md_bins: int = 60, velocity_threshold: float = 5.0, cv_count: int = 4, avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if avg_vel_bins is not None else None
    md_edges = np.linspace(0.0, 360.0 + 1e-6, n_md_bins + 1)
    T, N = spike_rates.shape
    folds = np.array_split(np.arange(T), cv_count)
    cv_masks = []
    for cv in range(cv_count):
        mask = np.ones(T, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)
    shape_map = (n_md_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_md_bins,), dtype=float) for _ in range(cv_count)]
    for l in tqdm(range(n_md_bins), desc="MD bins (CV)"):
        md_win = (md_edges[l], md_edges[l + 1])
        idx = _select_time_indices_in_md_bin(positions, base_time, md_series_deg, md_win, fast_mask)
        if idx.size == 0:
            continue
        for cv in range(cv_count):
            mask_cv = cv_masks[cv]
            valid = idx[idx < mask_cv.size]
            cv_idx = valid[mask_cv[valid]]
            occupancy_maps[cv][l] = float(cv_idx.size)
            if cv_idx.size:
                cv_rate_maps[cv][l, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)
    return cv_rate_maps, occupancy_maps, cv_masks, md_edges

def get_xy_rate_maps_cv(
    spike_rates: np.ndarray, locations: np.ndarray, base_time: np.ndarray,
    n_x_bins: int, n_y_bins: int, velocity_threshold: float = 5.0, cv_count: int = 4,
    avg_vel_bins: Optional[np.ndarray] = None
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if avg_vel_bins is not None else None
    
    xmin, xmax = float(np.nanmin(locations[:, 1])), float(np.nanmax(locations[:, 1]))
    ymin, ymax = float(np.nanmin(locations[:, 2])), float(np.nanmax(locations[:, 2]))
    x_edges = np.linspace(xmin, xmax + 1e-6, n_x_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_y_bins + 1)
    
    T, N = spike_rates.shape
    folds = np.array_split(np.arange(T), cv_count)
    cv_masks = []
    for cv in range(cv_count):
        mask = np.ones(T, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)
    shape_map = (n_y_bins, n_x_bins, N)
    cv_rate_maps = [np.full(shape_map, np.nan) for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_y_bins, n_x_bins), dtype=float) for _ in range(cv_count)]
    for iy in tqdm(range(n_y_bins), desc="Y bins (CV)"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_mask)
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
    T = spike_counts.shape[0]
    H, N = rate_map_hd.shape
    decoded_hd = np.full(T, np.nan)
    posterior_all: List[np.ndarray] = []
    rate_tau = rate_map_hd * tau 
    valid_h = np.any(np.isfinite(rate_tau), axis=1)
    rt = rate_tau.copy()
    rt[~np.isfinite(rt)] = 0.0
    rt += 1e-12
    for t in tqdm(range(T), desc="Decoding HD time bins"):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((H,), np.nan))
            continue
        k = spike_counts[t][None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k * np.log(rt + eps) - rt, axis=1)  # (H,)
        log_like[~valid_h] = -np.inf
        if not np.any(np.isfinite(log_like)):
            posterior_all.append(np.full((H,), np.nan))
            continue
        m = np.nanmax(log_like)
        post = np.exp(log_like - m)
        s = post.sum()
        if not np.isfinite(s) or s <= 0:
            posterior_all.append(np.full((H,), np.nan))
            continue
        post /= s
        posterior_all.append(post)
        h_idx = int(np.nanargmax(post))
        decoded_hd[t] = 0.5 * (hd_bins[h_idx] + hd_bins[h_idx + 1])
    return decoded_hd, posterior_all


def xy_bayesian_decoding(
    rate_map_xy: np.ndarray, 
    spike_counts: np.ndarray,
    tau: float,
    avg_vel: np.ndarray, 
    vel_threshold: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    eps: float = 1e-10,
):
    T, N = spike_counts.shape
    Y, X, N_map = rate_map_xy.shape
    assert N_map == N, "rate_map_xy last dim must match spike_counts."
    decoded_xy = np.full((T, 2), np.nan)
    posterior_all: List[np.ndarray] = []
    rate_tau = rate_map_xy * tau
    rate_tau_flat = rate_tau.reshape(Y * X, N)
    for t in tqdm(range(T), desc="Decoding XY time bins"):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((Y, X), np.nan))
            continue
        k = spike_counts[t]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k[None, :] * np.log(rate_tau_flat + eps) - rate_tau_flat, axis=1)
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
        x_c = 0.5 * (x_edges[ix] + x_edges[ix + 1])
        y_c = 0.5 * (y_edges[iy] + y_edges[iy + 1])
        decoded_xy[t, 0] = x_c
        decoded_xy[t, 1] = y_c
    return decoded_xy, posterior_all


def filter_neurons_by_hd_rate(
    spike_rates: np.ndarray,
    positions: np.ndarray,
    base_time: np.ndarray,
    avg_vel_bins: Optional[np.ndarray],
    n_hd_bins: int,
    vel_threshold: float,
    smooth_sigma_deg: Optional[float],
    keep_min_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate_map_hd_all, occupancy_all, hd_edges = get_hd_rate_map_all(
        spike_rates=spike_rates,
        locations=positions,
        base_time=base_time,
        n_hd_bins=n_hd_bins,
        velocity_threshold=vel_threshold,
        avg_vel_bins=avg_vel_bins,
    )
    # ~0.5 s worth of bins at tau=0.05 -> 10 bins; keep as constant 10 bins.
    min_occ = 10
    low_occ = occupancy_all < min_occ
    if np.any(low_occ):
        rate_map_hd_all[low_occ, :] = np.nan
    if smooth_sigma_deg is not None and smooth_sigma_deg > 0:
        rate_map_hd_all = _gaussian_filter_hd(rate_map_hd_all, float(smooth_angle_deg) * (n_hd_bins / 360.0))
    max_rates = np.nanmax(rate_map_hd_all, axis=0)
    keep_mask = (max_rates >= keep_min_rate) & np.isfinite(max_rates)
    if not np.any(keep_mask):
        raise RuntimeError(f"HD neuron filter removed all neurons (threshold={keep_min_rate} spikes/s).")
    return keep_mask, rate_map_hd_all[:, keep_mask], hd_edges

def decode_solo(args):
    data_path = Path(args.data_path)
    save_root = Path(args.save_path) / f"Dataset_{args.dataset}" / args.rat
    os.makedirs(save_root, exist_ok=True)

    print("\n=== ALL Decoding Pipeline (HD → MD → XY) with CV (Dusz) ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_root}")
    print(f"HD bins : {args.n_hd_bins}  | MD bins : {args.n_md_bins} (MD smoothing window: {args.md_smooth_window} samples)")
    print(f"XY bins : {args.n_y_bins} × {args.n_x_bins}")
    print(f"tau: {args.tau:.3f}s | vel_th: {args.vel_threshold} cm/s | CV folds: {args.cv_folds}")
    print(f"Angle smoothing: {args.smooth_sigma_deg}° | Spike-count temporal smoothing: {args.smooth_spikes_sigma_s*1000:.0f} ms")

    t0 = time.time()
    spike_times_raw, positions, cell_types, velocity_series = load_dusz_data(data_path, args.dataset, args.rat)
    hd = np.asarray(cell_types['hd'], dtype=int).ravel()
    excit = np.asarray(cell_types['ex'], dtype=int).ravel()
    excit_ids = np.where((hd == 1) & (excit == 1))[0]
    print(f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_ids)} | Position samples: {positions.shape[0]}")

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + args.tau, args.tau)

    spike_rates = build_spike_rates(spike_times_raw, base_time, args.tau, excit_ids)  # (T, N)
    avg_vel_bins = _compute_avg_velocity_per_bin(base_time, velocity_series)

    # Precompute MD series
    md_series_deg = _get_md_from_xy(positions, smoothing_window=args.md_smooth_window)

    # filter by the max firing rate in the HD rate map
    keep_mask, rate_map_hd_kept, hd_edges = filter_neurons_by_hd_rate(
        spike_rates=spike_rates,
        positions=positions,
        base_time=base_time,
        avg_vel_bins=avg_vel_bins,
        n_hd_bins=args.n_hd_bins,
        vel_threshold=args.vel_threshold,
        smooth_sigma_deg=args.smooth_sigma_deg,
        keep_min_rate=0.0, # we tried this, didn't end up being necessary
    )
    spike_rates = spike_rates[:, keep_mask]
    kept_neuron_ids = np.nonzero(keep_mask)[0]
    print(f"Kept {spike_rates.shape[1]} neurons after HD max-rate filter.")
    np.save(save_root / "neurons_kept_hd_filter.npy", kept_neuron_ids)

    print(f"\nCV decoding with {args.cv_folds} folds")

    cv_rate_maps_hd, occ_maps_hd, cv_masks, hd_edges_cv = get_hd_rate_maps_cv(
        spike_rates=spike_rates,
        locations=positions,
        base_time=base_time,
        n_hd_bins=args.n_hd_bins,
        velocity_threshold=args.vel_threshold,
        cv_count=args.cv_folds,
        avg_vel_bins=avg_vel_bins,
    )

    cv_summary_all: Dict[str, Any] = {"mode": "all", "folds": [], "n_folds": int(args.cv_folds)}

    for cv_idx in range(args.cv_folds):
        print(f"\n=== CV fold {cv_idx}/{args.cv_folds-1}: HD ===")
        rate_map_hd_cv = cv_rate_maps_hd[cv_idx].copy()
        occ_hd_cv = occ_maps_hd[cv_idx]

        min_occ = 10
        low = occ_hd_cv < min_occ
        if np.any(low):
            rate_map_hd_cv[low, ...] = np.nan
            
        rate_map_hd_cv = _gaussian_filter_hd(rate_map_hd_cv, float(smooth_sigma_deg) * (args.n_hd_bins / 360.0))

        cv_mask = cv_masks[cv_idx]
        test_idx = np.where(~cv_mask)[0]
        sc_hd = (spike_rates[test_idx, :] * args.tau).astype(float)

        # temporally smooth spikes before sorting as specified in the paper
        if args.smooth_spikes_sigma_s and args.smooth_spikes_sigma_s > 0:
            sigma_bins_t = max(1e-9, args.smooth_spikes_sigma_s / args.tau)
            sc_hd = gaussian_filter1d(sc_hd, sigma=sigma_bins_t, axis=0, mode="nearest")

        vel_hd = avg_vel_bins[test_idx]
        tc_hd = 0.5 * (base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)])

        decoded_hd, post_hd = hd_bayesian_decoding(
            rate_map_hd=rate_map_hd_cv,
            spike_counts=sc_hd,
            tau=args.tau,
            hd_bins=hd_edges_cv,
            avg_vel=vel_hd,
            vel_threshold=args.vel_threshold,
        )
        base = f"decoded_hd_cv{cv_idx}_{args.n_hd_bins}_all"
        np.save(save_root / f"{base}.npy", decoded_hd)
        np.save(save_root / f"{base}_posterior.npy", np.array(post_hd, dtype=object))
        np.save(save_root / f"{base}_time_centers.npy", tc_hd)
        if args.save_intermediates:
            np.save(save_root / f"hd_rate_map_cv{cv_idx}_{args.n_hd_bins}_kept.npy", rate_map_hd_cv)

        eval_hd = evaluate_errors_hd(decoded_hd, tc_hd, positions, velocity_series, args.vel_threshold)
        np.savez_compressed(save_root / f"errors_hd_cv{cv_idx}_{args.n_hd_bins}_all.npz", **eval_hd)

        # MD CV
        print(f"===  CV fold {cv_idx}/{args.cv_folds-1}: MD ===")
        cv_rate_maps_md, occ_maps_md, _, md_edges_cv = get_md_rate_maps_cv(
            spike_rates=spike_rates,
            positions=positions,
            base_time=base_time,
            md_series_deg=md_series_deg,
            n_md_bins=args.n_md_bins,
            velocity_threshold=args.vel_threshold,
            cv_count=args.cv_folds,
            avg_vel_bins=avg_vel_bins,
        )
        rate_map_md_cv = cv_rate_maps_md[cv_idx].copy()
        min_occ_md = max(1, int(np.ceil(0.5 / args.tau)))  
        occ_md_cv = occ_maps_md[cv_idx]
        low_md = occ_md_cv < min_occ_md
        if np.any(low_md):
            rate_map_md_cv[low_md, ...] = np.nan
        rate_map_md_cv = _gaussian_filter_hd(rate_map_md_cv, float(smooth_sigma_deg) * (args.n_md_bins / 360.0))

        sc_md = (spike_rates[test_idx, :] * args.tau).astype(float)
        if args.smooth_spikes_sigma_s and args.smooth_spikes_sigma_s > 0:
            sigma_bins_t = max(1e-9, args.smooth_spikes_sigma_s / args.tau)
            sc_md = gaussian_filter1d(sc_md, sigma=sigma_bins_t, axis=0, mode="nearest")

        vel_md = avg_vel_bins[test_idx]
        tc_md = tc_hd 
        decoded_md, post_md = hd_bayesian_decoding(
            rate_map_hd=rate_map_md_cv,
            spike_counts=sc_md,
            tau=args.tau,
            hd_bins=md_edges_cv,
            avg_vel=vel_md,
            vel_threshold=args.vel_threshold,
        )
        base = f"decoded_md_cv{cv_idx}_{args.n_md_bins}_all"
        np.save(save_root / f"{base}.npy", decoded_md)
        np.save(save_root / f"{base}_posterior.npy", np.array(post_md, dtype=object))
        np.save(save_root / f"{base}_time_centers.npy", tc_md)
        if args.save_intermediates:
            np.save(save_root / f"md_rate_map_cv{cv_idx}_{args.n_md_bins}_kept.npy", rate_map_md_cv)
            np.save(save_root / f"md_occupancy_map_cv{cv_idx}_{args.n_md_bins}_kept.npy", occ_md_cv)

        eval_md = evaluate_errors_md(decoded_md, tc_md, positions, velocity_series, args.vel_threshold, md_series_deg)
        np.savez_compressed(save_root / f"errors_md_cv{cv_idx}_{args.n_md_bins}_all.npz", **eval_md)

        # XY CV
        print(f"===  CV fold {cv_idx}/{args.cv_folds-1}: XY ===")
        cv_rate_maps_xy, occ_maps_xy, _, (x_edges, y_edges) = get_xy_rate_maps_cv(
            spike_rates=spike_rates,
            locations=positions,
            base_time=base_time,
            n_x_bins=args.n_x_bins,
            n_y_bins=args.n_y_bins,
            velocity_threshold=args.vel_threshold,
            cv_count=args.cv_folds,
            avg_vel_bins=avg_vel_bins,
        )
        rate_map_xy_cv = cv_rate_maps_xy[cv_idx].copy()
        occ_xy_cv = occ_maps_xy[cv_idx]
        min_occ_xy = max(1, int(np.ceil(0.5 / args.tau)))
        low_xy = occ_xy_cv < min_occ_xy
        if np.any(low_xy):
            rate_map_xy_cv[low_xy, ...] = np.nan

        sc_xy = (spike_rates[test_idx, :] * args.tau).astype(float)
        if args.smooth_spikes_sigma_s and args.smooth_spikes_sigma_s > 0:
            sigma_bins_t = max(1e-9, args.smooth_spikes_sigma_s / args.tau)
            sc_xy = gaussian_filter1d(sc_xy, sigma=sigma_bins_t, axis=0, mode="nearest")

        vel_xy = avg_vel_bins[test_idx]
        tc_xy = tc_hd
        decoded_xy, post_xy = xy_bayesian_decoding(
            rate_map_xy=rate_map_xy_cv,
            spike_counts=sc_xy,
            tau=args.tau,
            avg_vel=vel_xy,
            vel_threshold=args.vel_threshold,
            x_edges=x_edges,
            y_edges=y_edges,
        )
        base = f"decoded_xy_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}_all"
        np.save(save_root / f"{base}.npy", decoded_xy)
        np.save(save_root / f"{base}_posterior.npy", np.array(post_xy, dtype=object))
        np.save(save_root / f"{base}_time_centers.npy", tc_xy)
        if args.save_intermediates:
            np.save(save_root / f"xy_rate_map_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}_kept.npy", rate_map_xy_cv)
            np.save(save_root / f"xy_occupancy_map_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}_kept.npy", occ_xy_cv)

        eval_xy = evaluate_errors_xy(decoded_xy, tc_xy, positions, velocity_series, args.vel_threshold)
        np.savez_compressed(save_root / f"errors_xy_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}_all.npz", **eval_xy)

        cv_record = {
            "cv_idx": int(cv_idx),
            "n_test_bins": int(len(test_idx)),
            "kept_neurons": int(spike_rates.shape[1]),
            "error_stats_hd": eval_hd["stats"],
            "error_stats_md": eval_md["stats"],
            "error_stats_xy": eval_xy["stats"],
        }
        cv_summary_all["folds"].append(cv_record)

    with open(save_root / f"cv_summary_all.json", "w") as f:
        json.dump(cv_summary_all, f, indent=2)

    print("\n Finished HD→MD→XY with HD-based neuron filtering.")
    print(f"Kept neuron index list saved to: {save_root / 'neurons_kept_hd_filter.npy'}")
    print(f"Done in {(time.time()-t0)/60.0:.2f} min.")
        

def main():
    parser = argparse.ArgumentParser(description="ALL (HD→MD→XY) tuning + Bayesian decoding with CV (Dusz)")
    parser.add_argument("--data_path", type=str, default="./data/dusz", help="Root data dir containing Dataset_{id}/{rat}/Data")
    parser.add_argument("--save_path", type=str, default="./results/dusz", help="Directory to write outputs")
    parser.add_argument("--dataset", type=int, required=True, help="Dataset number (int)")
    parser.add_argument("--rat", type=str, required=True, help="Rat ID (e.g., 'R1')")

    parser.add_argument("--n_hd_bins", type=int, default=360, help="Number of head-direction bins (1° bins recommended)")
    parser.add_argument("--n_md_bins", type=int, default=360, help="Number of movement-direction bins (1° bins recommended)")
    parser.add_argument("--md_smooth_window", type=int, default=20, help="Smoothing window (samples) for MD computation")
    parser.add_argument("--n_x_bins", type=int, default=10, help="Number of X bins (XY)")
    parser.add_argument("--n_y_bins", type=int, default=10, help="Number of Y bins (XY)")

    parser.add_argument("--tau", type=float, default=0.050, help="Time bin size (s). Paper used 50 ms")
    parser.add_argument("--vel_threshold", type=float, default=2.0, help="Velocity threshold (cm/s). Paper used 2 cm/s")

    parser.add_argument("--cv_folds", type=int, default=4, help="Number of CV folds (>=2). Default uses CV.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")

    parser.add_argument("--smooth_sigma_deg", type=float, default=3.0,
                        help="Gaussian σ across angle (degrees). Paper used 3°")

    parser.add_argument("--smooth_spikes_sigma_s", type=float, default=0.100,
                        help="Temporal Gaussian σ for spike counts before decoding (s). Paper used 100 ms")

    parser.add_argument("--save_intermediates", action="store_true", default=False, help="Save intermediate artifacts")

    args = parser.parse_args()
    if args.cv_folds is None or args.cv_folds < 2:
        raise ValueError("--cv_folds must be >= 2.")

    decode_solo(args)

if __name__ == "__main__":
    main()
