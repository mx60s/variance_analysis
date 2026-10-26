"""Loading and time-binning helpers shared by the Pfeiffer and Duszkiewicz pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

# loading

def load_pfeiffer_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """data_path is Rat{rat}/Open1. Returns locations [t, x, y, hd], spikes [t, neuron_id], excitatory ids."""
    locations = loadmat(str(data_path / 'Position_Data.mat'))['Position_Data']
    spike_data = loadmat(str(data_path / 'Spike_Data.mat'))
    spike_times = spike_data['Spike_Data']
    excit_list = np.asarray(spike_data['Excitatory_Neurons']).ravel().astype(spike_times.dtype)
    return locations, spike_times, excit_list


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

    positions[:, 1] += abs(float(np.nanmin(positions[:, 1])))
    positions[:, 2] += abs(float(np.nanmin(positions[:, 2])))
    positions[:, 3] = (positions[:, 3] + 360.0) % 360.0
    if positions[:, 1].max() < 1: # one of these is in diff units for some reason
        positions[:, 1] *= 100
        positions[:, 2] *= 100

    cell_types_raw = loadmat(dpath / 'CellTypes.mat', struct_as_record=False, squeeze_me=True)
    cell_types = extract_matlab_obj(cell_types_raw.get('CellTypes', cell_types_raw))
    hd_list = np.asarray(cell_types.get('hd', cell_types_raw.get('hd'))).astype(int).ravel()
    ex_list = np.asarray(cell_types.get('ex', cell_types_raw.get('ex', np.ones_like(hd_list)))).astype(int).ravel()

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

# velocity and movement direction

def compute_velocity_series(locations: np.ndarray) -> np.ndarray:
    """Per-sample speed from tracked positions, as columns [t, speed]."""
    if locations.shape[0] < 4:
        raise ValueError("locations must have >= 4 rows to estimate time step.")
    time_step = float(locations[3, 0] - locations[2, 0])
    steps = np.diff(locations[:, 1:3], axis=0)
    steps = np.vstack((steps, steps[-1]))
    speed_samples = np.linalg.norm(steps, axis=1) / max(time_step, 1e-9)
    return np.column_stack((locations[:, 0], speed_samples))


def compute_velocity(locations: np.ndarray, base_time: np.ndarray) -> np.ndarray:
    """Mean speed (cm/s) per time bin. locations: [t, x, y, hd]."""
    speed_samples = compute_velocity_series(locations)[:, 1]
    v = np.zeros(len(base_time) - 1, dtype=float)
    for t in range(len(base_time) - 1):
        idx = np.where((locations[:, 0] >= base_time[t]) & (locations[:, 0] < base_time[t + 1]))[0]
        v[t] = 0.0 if idx.size == 0 else float(np.mean(speed_samples[idx]))
    return v


def compute_avg_velocity_per_bin(base_time: np.ndarray, velocity_series: np.ndarray,
                                 positions: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean of velocity_series per time bin. If positions is given, empty bins fall back to compute_velocity."""
    T = len(base_time) - 1
    avg_vel = np.full(T, np.nan, dtype=float)
    v_t = velocity_series[:, 0]
    v_v = velocity_series[:, 1]
    for i in range(T):
        m = (v_t >= base_time[i]) & (v_t < base_time[i + 1])
        if np.any(m):
            avg_vel[i] = float(np.nanmean(v_v[m]))

    if positions is not None and np.isnan(avg_vel).any():
        fallback = compute_velocity(positions, base_time)
        nan_mask = np.isnan(avg_vel)
        avg_vel[nan_mask] = fallback[nan_mask]
    return avg_vel


def circular_smooth_deg(angles_deg: np.ndarray, smoothing_window: int = 5) -> np.ndarray:
    if smoothing_window is None or smoothing_window <= 1:
        return angles_deg % 360.0
    ang_rad = np.deg2rad(angles_deg % 360.0)
    sin_vals = np.sin(ang_rad)
    cos_vals = np.cos(ang_rad)
    kernel = np.ones(smoothing_window) / float(smoothing_window)
    smooth_sin = np.convolve(sin_vals, kernel, mode='same')
    smooth_cos = np.convolve(cos_vals, kernel, mode='same')
    return (np.rad2deg(np.arctan2(smooth_sin, smooth_cos)) % 360.0)


def get_md_from_xy(positions: np.ndarray, smoothing_window: int = 20) -> np.ndarray:
    x = positions[:, 1]
    y = positions[:, 2]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    direction = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
    direction = circular_smooth_deg(direction, smoothing_window=smoothing_window)
    # Pad to same length as positions
    direction = np.concatenate([direction, [direction[-1]]])
    return direction

# time-bin selection (Dusz: samples are assigned to the bin that contains them)

def times_to_bin_indices(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, times, side='right') - 1
    return idx[(idx >= 0) & (idx < len(edges) - 1)]


def _sample_bins(sample_times: np.ndarray, base_time: np.ndarray, fast_mask: Optional[np.ndarray]) -> np.ndarray:
    idx = times_to_bin_indices(sample_times, base_time)
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx


def select_time_indices_in_angle_bin(positions: np.ndarray, base_time: np.ndarray, angles_deg: np.ndarray,
                                     win: Tuple[float, float], fast_mask: Optional[np.ndarray]) -> np.ndarray:
    """Time bins of samples whose angle (HD: positions[:, 3], or an MD series) falls in win."""
    l0, l1 = win
    in_bin = np.where((angles_deg >= l0) & (angles_deg < l1))[0]
    if in_bin.size == 0:
        return np.array([], dtype=int)
    return _sample_bins(positions[in_bin, 0], base_time, fast_mask)


def select_time_indices_in_xy_bin(positions: np.ndarray, base_time: np.ndarray,
                                  x_win: Tuple[float, float], y_win: Tuple[float, float],
                                  fast_mask: Optional[np.ndarray]) -> np.ndarray:
    x0, x1 = x_win
    y0, y1 = y_win
    in_xy = np.where(
        (positions[:, 1] >= x0) & (positions[:, 1] < x1) &
        (positions[:, 2] >= y0) & (positions[:, 2] < y1)
    )[0]
    if in_xy.size == 0:
        return np.array([], dtype=int)
    return _sample_bins(positions[in_xy, 0], base_time, fast_mask)

# split-half rate maps

def split_half_rate_map(spike_rates: np.ndarray, row_idx: List[np.ndarray], col_idx: List[np.ndarray],
                        floor: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D rate map over (row bin, col bin) from per-bin time indices, with a random split-half
    of the time bins in each cell (uses the global np.random state).
    Returns fold0, fold1, full rate map (R, C, N) and occupancy (R, C)."""
    T, N = spike_rates.shape
    R, C = len(row_idx), len(col_idx)
    fold0_map = np.full((R, C, N), np.nan)
    fold1_map = np.full((R, C, N), np.nan)
    rate_map = np.full((R, C, N), np.nan)
    occupancy = np.zeros((R, C), dtype=float)

    for r, idx_r in enumerate(row_idx):
        if idx_r.size == 0:
            continue
        for c, idx_c in enumerate(col_idx):
            if idx_c.size == 0:
                continue
            idx = np.intersect1d(idx_r, idx_c, assume_unique=False)
            if idx.size < floor:
                continue
            np.random.shuffle(idx)
            half = len(idx) // 2
            fold0 = spike_rates[idx[:half], :]
            fold1 = spike_rates[idx[half:], :]
            if fold1.shape[0] > half:
                fold1 = fold1[:-1, :]
            fold0_map[r, c, :] = np.nanmean(fold0, axis=0)
            fold1_map[r, c, :] = np.nanmean(fold1, axis=0)
            rate_map[r, c, :] = np.nanmean(spike_rates[idx, :], axis=0)
            occupancy[r, c] = float(2 * half)

    return fold0_map, fold1_map, rate_map, occupancy
