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
from openpyxl import load_workbook

# ---------------------------- Utilities ---------------------------- #

def compute_avg_velocity_per_bin(base_time: np.ndarray,
                                 velocity_series: Optional[np.ndarray],
                                 positions: np.ndarray) -> np.ndarray:
    T = len(base_time) - 1
    avg_vel = np.full(T, np.nan, dtype=float)

    v_t = velocity_series[:, 0]
    v_v = velocity_series[:, 1]
    for i in range(T):
        m = (v_t >= base_time[i]) & (v_t < base_time[i + 1])
        if np.any(m):
            avg_vel[i] = float(np.nanmean(v_v[m]))
    return avg_vel


def times_to_bin_indices(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map arbitrary timestamps to base_time bin indices (left-closed, right-open)."""
    idx = np.searchsorted(edges, times, side='right') - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    return idx[valid]

# ---------- Circular smoothing and MD from x–y (added) ---------- #

def _circular_smooth_deg(angles_deg: np.ndarray, smoothing_window: int = 5) -> np.ndarray:
    """Simple circular moving average in degrees."""
    if smoothing_window is None or smoothing_window <= 1:
        return angles_deg % 360.0
    ang_rad = np.deg2rad(angles_deg % 360.0)
    sin_vals = np.sin(ang_rad)
    cos_vals = np.cos(ang_rad)
    kernel = np.ones(smoothing_window) / float(smoothing_window)
    smooth_sin = np.convolve(sin_vals, kernel, mode='same')
    smooth_cos = np.convolve(cos_vals, kernel, mode='same')
    return (np.rad2deg(np.arctan2(smooth_sin, smooth_cos)) % 360.0)

def get_md_from_xy(positions: np.ndarray, smoothing_window: int = 200) -> np.ndarray:
    """Compute movement direction (deg, 0–360) from x,y trajectory."""
    x = positions[:, 1]
    y = positions[:, 2]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    # Note: use arctan2(-dy, dx) to match your provided script's convention.
    direction = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
    direction = _circular_smooth_deg(direction, smoothing_window=smoothing_window)
    # Pad to same length as positions
    direction = np.concatenate([direction, [direction[-1]]])
    return direction

# ---------------- Gaussian filters (NaN-aware) ---------------- #

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
    idx = times_to_bin_indices(loc_times, base_time)  # time-bin indices
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
    """Select base_time indices whose *position samples* fall in MD bin md_l."""
    l0, l1 = md_l
    in_md = np.where((md_series_deg >= l0) & (md_series_deg < l1))[0]
    if in_md.size == 0:
        return np.array([], dtype=int)
    loc_times = positions[in_md, 0]
    idx = times_to_bin_indices(loc_times, base_time)
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
    idx = times_to_bin_indices(loc_times, base_time)
    T = len(base_time) - 1
    idx = idx[(idx >= 0) & (idx < T)]
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx


# ---------------------- Error analysis helpers --------------------- #

def interp_circular_deg(t_src: np.ndarray, t_tgt: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    t_src = t_src[order]
    angles_deg = angles_deg[order]
    rad = np.deg2rad(angle_deg)
    ang_unwrapped = np.unwrap(rad, discont=np.pi)
    ang_interp_unwrapped = np.interp(t_tgt, t_src, ang_unwrapped)
    ang_interp = np.mod(ang_interp_unwrapped, 360.0)
    return ang_interp

def _interp_linear_series(t_src: np.ndarray, t_tgt: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    return np.interp(t_tgt, t_src[order], v_src[order])

def compute_ground_truth_hd(positions: np.ndarray, time_centers: np.ndarray,
                            velocity_series: np.ndarray, velocity_filter: float,
                            window: float = 0.06) -> np.ndarray:
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

def compute_ground_truth_md(positions: np.ndarray, time_centers: np.ndarray,
                            velocity_series: np.ndarray, velocity_filter: float,
                            md_series_deg: np.ndarray,
                            window: float = 0.06) -> np.ndarray:
    """MD 'ground truth' by interpolating MD series at time_centers, masked by speed."""
    # Interpolate MD over time (tied to positions timestamps)
    tpos = positions[:, 0]
    md_interp = interp_circular_deg(tpos, time_centers, md_series_deg)
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
                        positions: np.ndarray, velocity_series: np.ndarray,
                        vel_threshold: float) -> Dict[str, Any]:
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

def evaluate_errors_md(decoded_md: np.ndarray, time_centers: np.ndarray,
                       positions: np.ndarray, velocity_series: np.ndarray,
                       vel_threshold: float, md_series_deg: np.ndarray) -> Dict[str, Any]:
    gt = compute_ground_truth_md(positions, time_centers, velocity_series, vel_threshold, md_series_deg)
    md_err_abs = angular_distance_deg(decoded_md, gt)
    total = int(md_err_abs.size)
    valid_mask = ~np.isnan(md_err_abs)
    valid = int(np.sum(valid_mask))
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


def evaluate_errors_xy(decoded_xy: np.ndarray, time_centers: np.ndarray,
                        positions: np.ndarray, velocity_series: np.ndarray,
                        vel_threshold: float) -> Dict[str, Any]:
    gt_xy = compute_ground_truth_xy(positions, time_centers, velocity_series, vel_threshold)
    diffs = decoded_xy - gt_xy  # (T,2)
    dists = np.sqrt(np.nansum(diffs**2, axis=1))  # Euclidean
    total = int(dists.size)
    valid_mask = ~np.isnan(dists)
    valid = int(np.sum(valid_mask))
    stats = {
        'xy_mean_abs_error': float(np.nanmean(dists)),
        'xy_median_abs_error': float(np.nanmedian(dists)),
        'xy_std_abs_error': float(np.nanstd(dists)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors': dists}


# ------------------------- Data Loading / Prep ---------------------- #

def extract_matlab_obj(obj):
    """Recursively extract MATLAB structs into Python dictionaries/arrays."""
    from scipy.io.matlab._mio5_params import mat_struct  # local import to avoid hard dep if unused
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
            arr = []
            for v in val:
                arr.append(extract_matlab_obj(v))
            out[key] = np.array(arr)
        else:
            out[key] = val
    return out

def load_dusz_data(base_path: Path, dataset: int, rat: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Load Dusz dataset files for a given dataset/rat.
    Returns: spike_times (M,2), positions (P,4), cell_types dict, velocity_series (Q,2 or None)
    """
    dpath = base_path / f"Dataset_{dataset}" / rat / "Data"

    # Epochs
    # later if you want you can decode a few epochs from this
    epochs = loadmat(dpath / 'BehavEpochs.mat', struct_as_record=False, squeeze_me=True)
    epoch_data = None
    try:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    except:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    
    if rat == 'A3708-200317b': # this is the only rat with the square env in the second wake epoch
        epoch_data = extract_matlab_obj(epochs['wake2Ep'])
    
    # Tracking
    try: # so dumb, just one of the files uses different headers
        tracking = pd.read_csv(dpath / 'Tracking_data.csv')
        tracking = tracking.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    except:
        tracking = pd.read_csv(dpath / 'Tracking_data.csv')
        tracking = tracking.set_axis(['timestamp', 'x', 'y', 'hd'], axis=1)
        
    positions = tracking[['timestamp', 'x', 'y', 'hd']].to_numpy()
    if not isinstance(epoch_data['start'], float):
       epoch_data['start'] = epoch_data['start'][0]
       epoch_data['stop'] = epoch_data['stop'][0]
        
    mask_pos = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[mask_pos]
    # Normalize positions to positive origin (as in user's script)
    positions[:, 1] += abs(float(np.nanmin(positions[:, 1])))
    positions[:, 2] += abs(float(np.nanmin(positions[:, 2])))

    if positions[:, 1].max() < 1:
        positions[:, 1] *= 100
        positions[:, 2] *= 100 # uhhhh don't ask

    # Cell types
    cell_types_raw = loadmat(dpath / 'CellTypes.mat', struct_as_record=False, squeeze_me=True)
    cell_types = extract_matlab_obj(cell_types_raw['CellTypes'] if 'CellTypes' in cell_types_raw else cell_types_raw)
    # Fallback keys
    hd_list = np.asarray(cell_types.get('hd', cell_types_raw.get('hd'))).astype(int).ravel()
    ex_list = np.asarray(cell_types.get('ex', cell_types_raw.get('ex'))).astype(int).ravel()

    # Spikes
    spike_raw = loadmat(dpath / 'SpikeData.mat', struct_as_record=False, squeeze_me=True)
    S = extract_matlab_obj(spike_raw['S'] if 'S' in spike_raw else spike_raw)
    arr = []
    for i, cell in enumerate(S['C']):
        ts = np.asarray(cell['tsd']['t']).ravel()
        if ts.size:
            arr.append(np.column_stack([ts, np.full(ts.size, i, dtype=int)]))
    spike_times = np.vstack(arr) if arr else np.empty((0, 2))
    spike_times = spike_times[np.argsort(spike_times[:, 0])]
    mask_spk = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[mask_spk]

    vel_path = dpath / 'Velocity.mat'
    vel_raw = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
    vel = extract_matlab_obj(vel_raw['vel'] if 'vel' in vel_raw else vel_raw)
    velocity_series = np.dstack((vel['t'], vel['data']))[0]
    mask_v = (velocity_series[:, 0] > epoch_data['start']) & (velocity_series[:, 0] < epoch_data['stop'])
    velocity_series = velocity_series[mask_v]

    return spike_times, positions, {'hd': hd_list, 'ex': ex_list}, velocity_series


def extract_excitatory_ids(cell_types: Dict[str, np.ndarray]) -> np.ndarray:
    excit = np.asarray(cell_types['hd']).astype(int).ravel()
    return np.where(excit == 1)[0].astype(int)


def build_spike_rates(spike_times: np.ndarray, base_time: np.ndarray, tau: float, neuron_ids: np.ndarray) -> np.ndarray:
    """Vectorized per-neuron per-bin firing rates using numpy.histogram.
    Returns: spike_rates of shape (T, N_neurons)
    """
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
    avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if filter_vel else None
    pct_filtered = 0.0 if fast_mask is None else 100.0 * (1.0 - np.mean(fast_mask))
    print(f"Filtering {pct_filtered:.1f}% of bins due to low velocity")

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

    print("Creating cross-validation HD tuning curves…")
    for l in tqdm(range(n_hd_bins), desc="HD bins"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_mask)
        if idx.size == 0:
            continue
        for cv in range(cv_count):
            occupancy_maps[cv][l] = float(idx.size)
            mask_cv = cv_masks[cv]
            valid = idx[idx < mask_cv.size]
            cv_idx = valid[mask_cv[valid]]
            if cv_idx.size:
                cv_rate_maps[cv][l, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)
    return cv_rate_maps, occupancy_maps, cv_masks, hd_edges


def get_hd_rate_map_all(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_hd_bins: int = 60,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
    avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if filter_vel else None

    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)
    T, N = spike_rates.shape
    rate_map = np.full((n_hd_bins, N), np.nan, dtype=float)
    occupancy = np.zeros((n_hd_bins,), dtype=float)

    print("Creating all-data HD tuning curve…")
    for l in tqdm(range(n_hd_bins), desc="HD bins"):
        hd_win = (hd_edges[l], hd_edges[l + 1])
        idx = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_mask)
        if idx.size:
            occupancy[l] = float(idx.size)
            rate_map[l, :] = np.nanmean(spike_rates[idx, :], axis=0)
    return rate_map, occupancy, hd_edges

# ------------------ MD tuning curves (NEW; marginal over XY) -------- #

def get_md_rate_maps_cv(
    spike_rates: np.ndarray,
    positions: np.ndarray,
    base_time: np.ndarray,
    md_series_deg: np.ndarray,
    n_md_bins: int = 60,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
    cv_count: int = 4,
    seed: int = 777,
    avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if filter_vel else None
    pct_filtered = 0.0 if fast_mask is None else 100.0 * (1.0 - np.mean(fast_mask))
    print(f"Filtering {pct_filtered:.1f}% of bins due to low velocity")

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

    print("Creating cross-validation MD tuning curves…")
    for l in tqdm(range(n_md_bins), desc="MD bins"):
        md_win = (md_edges[l], md_edges[l + 1])
        idx = _select_time_indices_in_md_bin(positions, base_time, md_series_deg, md_win, fast_mask)
        if idx.size == 0:
            continue
        for cv in range(cv_count):
            occupancy_maps[cv][l] = float(idx.size)
            mask_cv = cv_masks[cv]
            valid = idx[idx < mask_cv.size]
            cv_idx = valid[mask_cv[valid]]
            if cv_idx.size:
                cv_rate_maps[cv][l, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)
    return cv_rate_maps, occupancy_maps, cv_masks, md_edges


def get_md_rate_map_all(
    spike_rates: np.ndarray,
    positions: np.ndarray,
    base_time: np.ndarray,
    md_series_deg: np.ndarray,
    n_md_bins: int = 60,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
    avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if filter_vel else None

    md_edges = np.linspace(0.0, 360.0 + 1e-6, n_md_bins + 1)
    T, N = spike_rates.shape
    rate_map = np.full((n_md_bins, N), np.nan, dtype=float)
    occupancy = np.zeros((n_md_bins,), dtype=float)

    print("Creating all-data MD tuning curve…")
    for l in tqdm(range(n_md_bins), desc="MD bins"):
        md_win = (md_edges[l], md_edges[l + 1])
        idx = _select_time_indices_in_md_bin(positions, base_time, md_series_deg, md_win, fast_mask)
        if idx.size:
            occupancy[l] = float(idx.size)
            rate_map[l, :] = np.nanmean(spike_rates[idx, :], axis=0)
    return rate_map, occupancy, md_edges


# ------------------ XY spatial rate maps --------------------- #

def get_xy_rate_map_all(
    spike_rates: np.ndarray,
    locations: np.ndarray,
    base_time: np.ndarray,
    n_x_bins: int,
    n_y_bins: int,
    velocity_threshold: float = 5.0,
    filter_vel: bool = True,
    avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if filter_vel else None

    xmin, xmax = float(np.nanmin(locations[:, 1])), float(np.nanmax(locations[:, 1]))
    ymin, ymax = float(np.nanmin(locations[:, 2])), float(np.nanmax(locations[:, 2]))
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
            idx = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_mask)
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
    avg_vel_bins: Optional[np.ndarray] = None,
):
    fast_mask = (avg_vel_bins >= velocity_threshold) if filter_vel else None
    pct_filtered = 0.0 if fast_mask is None else 100.0 * (1.0 - np.mean(fast_mask))
    print(f"Filtering {pct_filtered:.1f}% of bins due to low velocity")

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

    print("Creating cross-validation XY tuning maps…")
    for iy in tqdm(range(n_y_bins), desc="Y bins"):
        y_win = (y_edges[iy], y_edges[iy+1])
        for ix in range(n_x_bins):
            x_win = (x_edges[ix], x_edges[ix+1])
            idx = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_mask)
            if idx.size == 0:
                continue
            for cv in range(cv_count):
                occupancy_maps[cv][iy, ix] = float(idx.size)
                mask_cv = cv_masks[cv]
                valid = idx[idx < mask_cv.size]
                cv_idx = valid[mask_cv[valid]]
                if cv_idx.size:
                    cv_rate_maps[cv][iy, ix, :] = np.nanmean(spike_rates[cv_idx, :], axis=0)
    return cv_rate_maps, occupancy_maps, cv_masks, (x_edges, y_edges)


# ----------------------------- Decoding ----------------------------- #

def hd_bayesian_decoding(
    rate_map_hd: np.ndarray,  # (H, N)
    spike_counts: np.ndarray,  # (T, N)
    tau: float,
    hd_bins: np.ndarray,       # (H+1,)
    avg_vel: np.ndarray,       # (T,)
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
    apply_vel_gate = (vel_threshold is not None) and (vel_threshold > 0)
    for t in tqdm(range(T), desc="Decoding time bins"):
        if apply_vel_gate and (np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold):
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
    rate_map_xy: np.ndarray,  # (Y, X, N) mean firing rate (Hz)
    spike_counts: np.ndarray,  # (T, N) counts per bin
    tau: float,
    avg_vel: np.ndarray,  # (T,)
    vel_threshold: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    eps: float = 1e-10,
):
    """2D spatial Bayesian decoding with Poisson likelihood."""
    T, N = spike_counts.shape
    Y, X, N_map = rate_map_xy.shape
    assert N_map == N, "rate_map_xy last dim must match spike_counts."
    decoded_xy = np.full((T, 2), np.nan)
    posterior_all: List[np.ndarray] = []
    print(f"Decoding {T} time bins using {N} neurons over {Y}×{X} grid…")
    rate_tau = rate_map_xy * tau  # expected counts per bin
    rate_tau_flat = rate_tau.reshape(Y * X, N)
    apply_vel_gate = (vel_threshold is not None) and (vel_threshold > 0)
    for t in tqdm(range(T), desc="Decoding time bins"):
        if apply_vel_gate and (np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold):
            posterior_all.append(np.full((Y, X), np.nan))
            continue
        k = spike_counts[t]
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
        x_c = 0.5 * (x_edges[ix] + x_edges[ix + 1])
        y_c = 0.5 * (y_edges[iy] + y_edges[iy + 1])
        decoded_xy[t, 0] = x_c
        decoded_xy[t, 1] = y_c
    return decoded_xy, posterior_all


# ------------------------------- Main ------------------------------- #

def decode_solo(
    args: Optional[argparse.Namespace] = None,
    /,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Make HD, XY, or MD rate maps from Dusz data and do singular Bayesian decoding.
    Returns a small summary dict with paths & stats that can be inspected by callers.
    """
    if args is None:
        args = argparse.Namespace(**kwargs) 
    else:
        for k, v in kwargs.items():
            setattr(args, k, v)

    d = args.__dict__
    d.setdefault("data_path", "./data/dusz")
    d.setdefault("save_path", "./results/dusz")
    d.setdefault("dataset", None)
    d.setdefault("rat", None)

    d.setdefault("mode", "hd")
    d.setdefault("n_hd_bins", 17)
    d.setdefault("n_md_bins", 17)
    d.setdefault("md_smooth_window", 200)

    d.setdefault("n_x_bins", 10)
    d.setdefault("n_y_bins", 10)

    d.setdefault("tau", 0.120)
    d.setdefault("vel_threshold", 5.0)

    d.setdefault("filter_vel", True)
    d.setdefault("seed", 777)
    d.setdefault("smooth_sigma", 1.0)
    d.setdefault("save_intermediates", False)
    

    # By default, in HD/MD we often ignore velocity gating at decode time
    if d['mode'] == 'hd':
        d['filter_vel'] = False
        d['vel_threshold'] = 0.0

    data_path = Path(d['data_path'])
    save_root = Path(d['save_path']) / f"Dataset_{d['dataset']}" / d['rat']
    os.makedirs(save_root, exist_ok=True)

    print(f"\n=== {'Head-Direction' if d['mode']=='hd' else ('Movement Direction' if d['mode']=='md' else 'Location (XY)')} Decoding Pipeline (Dusz) ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_root}")
    if d['mode'] == "hd":
        print(f"HD bins : {d['n_hd_bins']}")
    elif d['mode'] == "md":
        print(f"MD bins : {d['n_md_bins']}")
        print(f"MD smoothing window (samples): {d['md_smooth_window']}")
    else:
        print(f"XY bins : {args.n_y_bins} × {args.n_x_bins}")
    print(f"tau: {d['tau']:.3f}s | vel_th: {d['vel_threshold']} | CV folds: {d['cv_folds']}")

    t0 = time.time()
    print("\nLoading data…")
    spike_times_raw, positions, cell_types, velocity_series = load_dusz_data(data_path, d['dataset'], d['rat'])
    excit = np.asarray(cell_types['hd']).astype(int).ravel()
    excit_ids = np.where(excit == 1)[0].astype(int)
    
    print(
        f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_ids)} | Position samples: {positions.shape[0]}"
    )

    if spike_times_raw.size == 0 or positions.size == 0 or excit_ids.size == 0:
        raise RuntimeError("Insufficient data: check dataset/rat paths and cell types.")

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + d['tau'], d['tau'])
    print(
        f"Time bins: {len(base_time)-1} bins covering {base_time[0]:.3f}–{base_time[-1]:.3f}s"
    )

    print("Binning spikes into rates…")
    spike_rates = build_spike_rates(spike_times_raw, base_time, d['tau'], excit_ids)  # (T, N)

    avg_vel_bins = compute_avg_velocity_per_bin(base_time, velocity_series, positions)

    # Precompute MD series if needed
    md_series_deg = None
    if d['mode'] == "md" or d['save_intermediates']:
        md_series_deg = get_md_from_xy(positions, smoothing_window=d['md_smooth_window'])

    if d['mode'] == "hd":
        (
            cv_rate_maps,
            occupancy_maps,
            cv_masks,
            hd_edges,
        ) = get_hd_rate_maps_cv(
            spike_rates=spike_rates,
            locations=positions,
            base_time=base_time,
            n_hd_bins=d['n_hd_bins'],
            velocity_threshold=d['vel_threshold'],
            filter_vel=d['filter_vel'],
            cv_count=d['cv_folds'],
            seed=d['seed'],
            avg_vel_bins=avg_vel_bins,
        )

        cv_summary: Dict[str, Any] = {"mode": "hd", "folds": [], "n_folds": int(d['cv_folds'])}
        for cv_idx in range(d['cv_folds']):
            print(f"\n=== Decoding CV fold {cv_idx}/{d['cv_folds']-1} (HD) ===")
            rate_map_hd = cv_rate_maps[cv_idx]
            cv_mask = cv_masks[cv_idx]  # True=train, False=test (over T bins)
            test_idx = np.where(~cv_mask)[0]
            print(f"Using CV fold {cv_idx}: decoding {len(test_idx)} held-out bins.")
            spike_counts = (spike_rates[test_idx, :] * d['tau']).astype(float)
            avg_vel = avg_vel_bins[test_idx]
            time_centers = 0.5 * (
                base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)]
            )
            rate_map_hd = _gaussian_filter_hd(rate_map_hd, d['smooth_sigma'])
            if d['save_intermediates']:
                np.save(save_root / f"hd_rate_map_cv{cv_idx}_{d['n_hd_bins']}_train.npy", rate_map_hd)
                np.save(save_root / f"cv_mask_hd{cv_idx}_{d['n_hd_bins']}.npy", cv_mask)
                np.save(save_root / f"hd_occupancy_map_cv{cv_idx}_{d['n_hd_bins']}.npy", occupancy_maps[cv_idx])

            decoded_hd, posterior_all = hd_bayesian_decoding(
                rate_map_hd=rate_map_hd,
                spike_counts=spike_counts,
                tau=d['tau'],
                hd_bins=hd_edges,
                avg_vel=avg_vel,
                vel_threshold=d['vel_threshold'],
            )
            print("Saving decoding outputs…")
            base = f"decoded_hd_cv{cv_idx}_{d['n_hd_bins']}"
            np.save(save_root / f"{base}.npy", decoded_hd)
            np.save(save_root / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
            np.save(save_root / f"{base}_time_centers.npy", time_centers)

            eval_res = evaluate_errors_hd(decoded_hd, time_centers, positions, velocity_series, d['vel_threshold'])
            np.savez_compressed(save_root / f"errors_hd_cv{cv_idx}_{d['n_hd_bins']}.npz", **eval_res)
            print("Saved error analysis:", save_root / f"errors_hd_cv{cv_idx}_{d['n_hd_bins']}.npz")

            fold_record: Dict[str, Any] = {
                "cv_idx": int(cv_idx),
                "n_test_bins": int(len(test_idx)),
                "output_prefix": base,
                "error_stats": eval_res["stats"],
            }
            cv_summary["folds"].append(fold_record)

        with open(save_root / f"cv_summary_hd_{d['n_hd_bins']}.json", "w") as f:
            json.dump(cv_summary, f, indent=2)
        print("\nWrote CV summary:", save_root / f"cv_summary_hd_{d['n_hd_bins']}.json")

    elif d['mode'] == "md":
        if md_series_deg is None:
            md_series_deg = get_md_from_xy(positions, smoothing_window=d['md_smooth_window'])
        (
            cv_rate_maps_md,
            occupancy_maps_md,
            cv_masks,
            md_edges,
        ) = get_md_rate_maps_cv(
            spike_rates=spike_rates,
            positions=positions,
            base_time=base_time,
            md_series_deg=md_series_deg,
            n_md_bins=d['n_md_bins'],
            velocity_threshold=d['vel_threshold'],
            filter_vel=d['filter_vel'],
            cv_count=d['cv_folds'],
            seed=d['seed'],
            avg_vel_bins=avg_vel_bins,
        )

        cv_summary: Dict[str, Any] = {"mode": "md", "folds": [], "n_folds": int(d['cv_folds'])}
        for cv_idx in range(d['cv_folds']):
            print(f"\n=== Decoding CV fold {cv_idx}/{d['cv_folds']-1} (MD) ===")
            rate_map_md = cv_rate_maps_md[cv_idx]
            cv_mask = cv_masks[cv_idx]
            test_idx = np.where(~cv_mask)[0]
            print(f"Using CV fold {cv_idx}: decoding {len(test_idx)} held-out bins.")
            spike_counts = (spike_rates[test_idx, :] * d['tau']).astype(float)
            avg_vel = avg_vel_bins[test_idx]
            time_centers = 0.5 * (
                base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)]
            )
            rate_map_md = _gaussian_filter_hd(rate_map_md, d['smooth_sigma'])
            if d['save_intermediates']:
                np.save(save_root / f"md_rate_map_cv{cv_idx}_{d['n_md_bins']}_train.npy", rate_map_md)
                np.save(save_root / f"cv_mask_md{cv_idx}_{d['n_md_bins']}.npy", cv_mask)
                np.save(save_root / f"md_occupancy_map_cv{cv_idx}_{d['n_md_bins']}.npy", occupancy_maps_md[cv_idx])

            decoded_md, posterior_all = hd_bayesian_decoding(
                rate_map_hd=rate_map_md,
                spike_counts=spike_counts,
                tau=d['tau'],
                hd_bins=md_edges,
                avg_vel=avg_vel,
                vel_threshold=d['vel_threshold'],
            )
            print("Saving decoding outputs…")
            base = f"decoded_md_cv{cv_idx}_{d['n_md_bins']}"
            np.save(save_root / f"{base}.npy", decoded_md)
            np.save(save_root / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
            np.save(save_root / f"{base}_time_centers.npy", time_centers)

            eval_res = evaluate_errors_md(decoded_md, time_centers, positions, velocity_series, d['vel_threshold'], md_series_deg)
            np.savez_compressed(save_root / f"errors_md_cv{cv_idx}_{d['n_md_bins']}.npz", **eval_res)
            print("Saved error analysis:", save_root / f"errors_md_cv{cv_idx}_{d['n_md_bins']}.npz")

            fold_record: Dict[str, Any] = {
                "cv_idx": int(cv_idx),
                "n_test_bins": int(len(test_idx)),
                "output_prefix": base,
                "error_stats": eval_res["stats"],
            }
            cv_summary["folds"].append(fold_record)

        with open(save_root / f"cv_summary_md_{d['n_md_bins']}.json", "w") as f:
            json.dump(cv_summary, f, indent=2)
        print("\nWrote CV summary:", save_root / f"cv_summary_md_{d['n_md_bins']}.json")

    else:
        (
            cv_rate_maps_xy,
            occupancy_maps_xy,
            cv_masks,
            (x_edges, y_edges),
        ) = get_xy_rate_maps_cv(
            spike_rates=spike_rates,
            locations=positions,
            base_time=base_time,
            n_x_bins=args.n_x_bins,
            n_y_bins=args.n_y_bins,
            velocity_threshold=d['vel_threshold'],
            filter_vel=d['filter_vel'],
            cv_count=d['cv_folds'],
            seed=d['seed'],
            avg_vel_bins=avg_vel_bins,
        )

        cv_summary: Dict[str, Any] = {"mode": "xy", "folds": [], "n_folds": int(d['cv_folds'])}
        for cv_idx in range(d['cv_folds']):
            print(f"\n=== Decoding CV fold {cv_idx}/{d['cv_folds']-1} (XY) ===")
            rate_map_xy = cv_rate_maps_xy[cv_idx]
            if d['smooth_sigma'] and d['smooth_sigma'] > 0:
                rate_map_xy = _gaussian_filter_xy(rate_map_xy, (d['smooth_sigma'], d['smooth_sigma']))
            cv_mask = cv_masks[cv_idx]
            test_idx = np.where(~cv_mask)[0]
            print(f"Using CV fold {cv_idx}: decoding {len(test_idx)} held-out bins.")
            spike_counts = (spike_rates[test_idx, :] * d['tau']).astype(float)
            avg_vel = avg_vel_bins[test_idx]
            
            time_centers = 0.5 * (
                base_time[test_idx] + base_time[np.minimum(test_idx + 1, len(base_time) - 1)]
            )
            if d['save_intermediates']:
                np.save(save_root / f"xy_rate_map_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}_train.npy", rate_map_xy)
                np.save(save_root / f"cv_mask_xy{cv_idx}_{args.n_y_bins}x{args.n_x_bins}.npy", cv_mask)
                np.save(save_root / f"xy_occupancy_map_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}.npy", occupancy_maps_xy[cv_idx])

            decoded_xy, posterior_all = xy_bayesian_decoding(
                rate_map_xy=rate_map_xy,
                spike_counts=spike_counts,
                tau=d['tau'],
                avg_vel=avg_vel,
                vel_threshold=d['vel_threshold'],
                x_edges=x_edges,
                y_edges=y_edges,
            )
            print("Saving decoding outputs…")
            base = f"decoded_xy_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}"
            np.save(save_root / f"{base}.npy", decoded_xy)
            np.save(save_root / f"{base}_posterior.npy", np.array(posterior_all, dtype=object))
            np.save(save_root / f"{base}_time_centers.npy", time_centers)

            eval_res = evaluate_errors_xy(decoded_xy, time_centers, positions, velocity_series, d['vel_threshold'])
            np.savez_compressed(save_root / f"errors_xy_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}.npz", **eval_res)
            print("Saved error analysis:", save_root / f"errors_xy_cv{cv_idx}_{args.n_y_bins}x{args.n_x_bins}.npz")

            fold_record: Dict[str, Any] = {
                "cv_idx": int(cv_idx),
                "n_test_bins": int(len(test_idx)),
                "output_prefix": base,
                "error_stats": eval_res["stats"],
            }
            cv_summary["folds"].append(fold_record)

        with open(save_root / f"cv_summary_xy_{args.n_y_bins}x{args.n_x_bins}.json", "w") as f:
            json.dump(cv_summary, f, indent=2)
        print("\nWrote CV summary:", save_root / f"cv_summary_xy_{args.n_y_bins}x{args.n_x_bins}.json")

    print(f"\nDone in {(time.time()-t0)/60.0:.2f} min.")

def main():
    parser = _build_parser()
    args = parser.parse_args()
    decode_solo(args)

if __name__ == "__main__":
    main()
