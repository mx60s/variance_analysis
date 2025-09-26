#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, gaussian_filter

N_VISITS = 500

# ---------------------------- Utilities ---------------------------- #

def compute_velocity(locations: np.ndarray, base_time: np.ndarray) -> np.ndarray:
    """Compute mean velocity (units/s) within each time bin. locations: [t, x, y, hd]."""
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

def compute_avg_velocity_per_bin(base_time: np.ndarray,
                                 velocity_series: Optional[np.ndarray],
                                 positions: np.ndarray) -> np.ndarray:
    """Average velocity per bin (len(base_time)-1,). Falls back to position-derived velocity."""
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
    """Map timestamps to base_time bin indices (left-closed, right-open)."""
    idx = np.searchsorted(edges, times, side='right') - 1
    return idx[(idx >= 0) & (idx < len(edges) - 1)]

# ---------- Circular smoothing and MD from x–y ---------- #

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
    direction = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
    direction = _circular_smooth_deg(direction, smoothing_window=smoothing_window)
    direction = np.concatenate([direction, [direction[-1]]])  # pad
    return direction

# ---------------- Time-bin index selectors ---------------- #

def _select_time_indices_in_hd_bin(locations: np.ndarray, base_time: np.ndarray,
                                   hd_l: Tuple[float, float],
                                   fast_mask: Optional[np.ndarray]) -> np.ndarray:
    l0, l1 = hd_l
    in_hd = np.where((locations[:, 3] >= l0) & (locations[:, 3] < l1))[0]
    if in_hd.size == 0:
        return np.array([], dtype=int)
    loc_times = locations[in_hd, 0]
    idx = times_to_bin_indices(loc_times, base_time)
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx

def _select_time_indices_in_md_bin(positions: np.ndarray, base_time: np.ndarray,
                                   md_series_deg: np.ndarray, md_l: Tuple[float, float],
                                   fast_mask: Optional[np.ndarray]) -> np.ndarray:
    l0, l1 = md_l
    in_md = np.where((md_series_deg >= l0) & (md_series_deg < l1))[0]
    if in_md.size == 0:
        return np.array([], dtype=int)
    loc_times = positions[in_md, 0]
    idx = times_to_bin_indices(loc_times, base_time)
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx

def _select_time_indices_in_xy_bin(locations: np.ndarray, base_time: np.ndarray,
                                   x_win: Tuple[float, float], y_win: Tuple[float, float],
                                   fast_mask: Optional[np.ndarray]) -> np.ndarray:
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
    if fast_mask is not None:
        idx = idx[fast_mask[idx]]
    return idx

# ------------------------- Data Loading / Prep ---------------------- #

def extract_matlab_obj(obj):
    """Recursively extract MATLAB structs into Python dictionaries/arrays."""
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

def load_dusz_data(base_path: Path, dataset: int, rat: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Load Dusz dataset files for a given dataset/rat.
    Returns: spike_times (M,2), positions (P,4), cell_types dict, velocity_series (Q,2 or None)
    """
    dpath = base_path / f"Dataset_{dataset}" / rat / "Data"

    # Epochs
    epochs = loadmat(dpath / 'BehavEpochs.mat', struct_as_record=False, squeeze_me=True)
    epoch_data = None
    try:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    except:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    if rat == 'A3708-200317b':
        epoch_data = extract_matlab_obj(epochs['wake2Ep'])

    # Tracking
    try:
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

    # Normalize to positive origin
    positions[:, 1] += abs(float(np.nanmin(positions[:, 1])))
    positions[:, 2] += abs(float(np.nanmin(positions[:, 2])))

    # Some datasets are in different units, for some reason
    if positions[:, 1].max() < 1:
        positions[:, 1] *= 100
        positions[:, 2] *= 100

    # Cell types
    cell_types_raw = loadmat(dpath / 'CellTypes.mat', struct_as_record=False, squeeze_me=True)
    cell_types = extract_matlab_obj(cell_types_raw['CellTypes'] if 'CellTypes' in cell_types_raw else cell_types_raw)
    hd_list = np.asarray(cell_types.get('hd', cell_types_raw.get('hd'))).astype(int).ravel()
    ex_list = np.asarray(cell_types.get('ex', cell_types_raw.get('ex', np.ones_like(hd_list)))).astype(int).ravel()

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

    # Velocity
    vel_path = dpath / 'Velocity.mat'
    vel_raw = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
    vel = extract_matlab_obj(vel_raw['vel'] if 'vel' in vel_raw else vel_raw)
    velocity_series = np.dstack((vel['t'], vel['data']))[0]
    mask_v = (velocity_series[:, 0] > epoch_data['start']) & (velocity_series[:, 0] < epoch_data['stop'])
    velocity_series = velocity_series[mask_v]

    return spike_times, positions, {'hd': hd_list, 'ex': ex_list}, velocity_series

def build_spike_rates(spike_times: np.ndarray, base_time: np.ndarray, tau: float, neuron_ids: np.ndarray) -> np.ndarray:
    """Per-neuron per-bin firing rates using numpy.histogram → (T, N_neurons)."""
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

# ------------------ 2D HD×MD rate map ------------------ #

def hdmd_rate_map(spike_rates: np.ndarray, positions: np.ndarray, base_time: np.ndarray,
                  n_hd_bins: int, n_md_bins: int, md_series_deg: np.ndarray,
                  avg_vel_bins: np.ndarray, velocity_threshold: float
                  ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    fast_mask = (avg_vel_bins >= velocity_threshold) if velocity_threshold > 0 else None
    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)
    md_edges = np.linspace(0.0, 360.0 + 1e-6, n_md_bins + 1)

    T, N = spike_rates.shape
    raw_rate_map = np.full((n_hd_bins, n_md_bins, N_VISITS, N), np.nan)
    occupancy = np.zeros((n_hd_bins, n_md_bins), dtype=float)

    for ih in range(n_hd_bins):
        hd_win = (hd_edges[ih], hd_edges[ih + 1])
        idx_hd = _select_time_indices_in_hd_bin(positions, base_time, hd_win, fast_mask)
        if idx_hd.size == 0:
            continue
        for im in range(n_md_bins):
            md_win = (md_edges[im], md_edges[im + 1])
            idx_md = _select_time_indices_in_md_bin(positions, base_time, md_series_deg, md_win, fast_mask)
            if idx_md.size == 0:
                continue
            idx = np.intersect1d(idx_hd, idx_md, assume_unique=False)
            if idx.size:
                np.random.shuffle(idx) 
                take = int(min(N_VISITS, idx.size))
                occupancy[ih, im] = float(idx.size) 
                raw_rate_map[ih, im, :take, :] = spike_rates[idx[:take], :]

    return raw_rate_map, occupancy, (hd_edges, md_edges)

# ------------------ 3D XY×HD raw per-visit map ------------------ #

def raw_xyhd_rate_map(spike_rates: np.ndarray, locations: np.ndarray, base_time: np.ndarray,
                      n_loc_bins: int, n_hd_bins: int,
                      avg_vel_bins: np.ndarray, velocity_threshold: float,
                      arena_pad: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return:
      raw_rate_map: (Y, X, H, V, N) storing up to V=N_VISITS per-visit rate vectors,
      occupancy:    (Y, X, H) number of visits (time bins) observed,
      edges:        (x_edges, y_edges, hd_edges)
    """
    fast_mask = (avg_vel_bins >= velocity_threshold) if velocity_threshold > 0 else None

    xmin, xmax = float(np.nanmin(locations[:, 1])), float(np.nanmax(locations[:, 1]))
    ymin, ymax = float(np.nanmin(locations[:, 2])), float(np.nanmax(locations[:, 2]))
    if arena_pad and arena_pad != 0.0:
        xmin -= arena_pad; xmax += arena_pad
        ymin -= arena_pad; ymax += arena_pad
    x_edges = np.linspace(xmin, xmax + 1e-6, n_loc_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_loc_bins + 1)
    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)

    T, N = spike_rates.shape
    raw_rate_map = np.full((n_loc_bins, n_loc_bins, n_hd_bins, N_VISITS, N), np.nan, dtype=float)
    occupancy = np.zeros((n_loc_bins, n_loc_bins, n_hd_bins), dtype=float)

    for iy in range(n_loc_bins):
        y_win = (y_edges[iy], y_edges[iy + 1])
        for ix in range(n_loc_bins):
            x_win = (x_edges[ix], x_edges[ix + 1])
            idx_xy = _select_time_indices_in_xy_bin(locations, base_time, x_win, y_win, fast_mask)
            if idx_xy.size == 0:
                continue
            for ih in range(n_hd_bins):
                hd_win = (hd_edges[ih], hd_edges[ih + 1])
                idx_hd = _select_time_indices_in_hd_bin(locations, base_time, hd_win, fast_mask)
                if idx_hd.size == 0:
                    continue
                idx = np.intersect1d(idx_xy, idx_hd, assume_unique=False)
                if idx.size:
                    np.random.shuffle(idx) 
                    take = int(min(N_VISITS, idx.size))
                    occupancy[iy, ix, ih] = float(idx.size) 
                    raw_rate_map[iy, ix, ih, :take, :] = spike_rates[idx[:take], :]

    return raw_rate_map, occupancy, (x_edges, y_edges, hd_edges)


def get_raw_ratemaps(args: Optional[argparse.Namespace] = None, /, **overrides: Any) -> Dict[str, Any]:
    """
    Build rate maps for XY×HD (3D raw per-visit) or HD×MD (2D).
    Returns a summary dict (paths, shapes, edges).
    """
    
    if args is None:
        args = argparse.Namespace(**overrides)
    else:
        for k, v in overrides.items():
            setattr(args, k, v)

    d = args.__dict__
    d.setdefault("data_path", "./data/dusz")
    d.setdefault("save_path", "./results/dusz")
    d.setdefault("dataset", None)
    d.setdefault("rat", None)
    d.setdefault("mode", "xyhd")
    d.setdefault("n_hd_bins", 60)
    d.setdefault("n_md_bins", 60)
    d.setdefault("n_loc_bins", 20)
    d.setdefault("tau", 0.120)
    d.setdefault("vel_threshold", 5.0)
    d.setdefault("md_smooth_window", 200)
    d.setdefault("save_intermediates", False)

    if d["dataset"] is None or d["rat"] is None:
        raise ValueError("Both 'dataset' (int) and 'rat' (str) must be provided.")

    data_path = Path(d["data_path"])
    save_root = Path(d["save_path"]) / f"Dataset_{d['dataset']}" / d["rat"]
    os.makedirs(save_root, exist_ok=True)

    print(f"\n=== Rate Map Builder ({d['mode']}) ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_root}")
    print(f"tau: {d['tau']:.3f}s | vel_th: {d['vel_threshold']}")
    t0 = time.time()

    # Load & prep
    print("\nLoading data…")
    spike_times_raw, positions, cell_types, velocity_series = load_dusz_data(data_path, d["dataset"], d["rat"])

    # Use HD-classified neurons (per your original)
    excit = np.asarray(cell_types['hd']).astype(int).ravel()
    excit_ids = np.where(excit == 1)[0].astype(int)

    print(f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_ids)} | Position samples: {positions.shape[0]}")

    if spike_times_raw.size == 0 or positions.size == 0 or excit_ids.size == 0:
        raise RuntimeError("Insufficient data: check dataset/rat paths and cell types.")

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + d["tau"], d["tau"])
    print(f"Time bins: {len(base_time)-1} bins covering {base_time[0]:.3f}–{base_time[-1]:.3f}s")

    print("Binning spikes into rates…")
    spike_rates = build_spike_rates(spike_times_raw, base_time, d["tau"], excit_ids)  # (T, N)
    avg_vel_bins = compute_avg_velocity_per_bin(base_time, velocity_series, positions)

    out: Dict[str, Any] = {}
    edges: Dict[str, Any] = {}

    if d["mode"] == "hdmd":
        md_series_deg = get_md_from_xy(positions, smoothing_window=d["md_smooth_window"])
        rate_map, occ, (hd_edges, md_edges) = hdmd_rate_map(
            spike_rates, positions, base_time,
            d["n_hd_bins"], d["n_md_bins"], md_series_deg,
            avg_vel_bins, d["vel_threshold"]
        )
        
        out = dict(rate_map=rate_map, occupancy=occ)
        edges = {"hd": hd_edges, "md": md_edges}
        base = f"raw_rate_hdmd_{d['n_hd_bins']}x{d['n_md_bins']}"

    else:  # xyhd
        rate_map, occ, (x_edges, y_edges, hd_edges) = raw_xyhd_rate_map(
            spike_rates, positions, base_time,
            d["n_loc_bins"], d["n_hd_bins"],
            avg_vel_bins, d["vel_threshold"]
        )
        out = dict(raw_rate_map=rate_map, occupancy=occ)
        edges = {"x": x_edges, "y": y_edges, "hd": hd_edges}
        base = f"raw_rate_xyhd_{d['n_loc_bins']}x{d['n_loc_bins']}x{d['n_hd_bins']}"

    # Save results
    out_npz = save_root / f"{base}.npz"
    np.savez_compressed(out_npz, **out, **{k+"_edges": v for k, v in edges.items()})
    print(f"\nSaved: {out_npz}")

    # Optionally save metadata
    meta_path = None
    meta_path = save_root / f"{base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "mode": d["mode"],
            "tau": d["tau"],
            "vel_threshold": d["vel_threshold"],
            "n_hd_bins": d["n_hd_bins"],
            "n_md_bins": d["n_md_bins"],
            "n_loc_bins": d["n_loc_bins"],
            "md_smooth_window": d["md_smooth_window"],
        }, f, indent=2)
    print("Saved meta json.")

    elapsed_min = (time.time() - t0) / 60.0
    print(f"Done in {elapsed_min:.2f} min.")

    summary = {
        "out_npz": str(out_npz),
        "out_meta": str(meta_path) if meta_path else None,
        "mode": d["mode"],
        "shapes": {k: (np.array(v).shape if isinstance(v, np.ndarray) else None)
                   for k, v in out.items()},
        "edges_keys": list(edges.keys()),
        "elapsed_min": float(elapsed_min),
    }
    return summary

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make rate maps: XY×HD (3D raw per-visit) or HD×MD (2D)."
    )
    parser.add_argument("--data_path", type=str, default="./data/dusz",
                        help="Root data dir containing Dataset_{id}/{rat}/Data")
    parser.add_argument("--save_path", type=str, default="./results/dusz",
                        help="Directory to write outputs")
    parser.add_argument("--dataset", type=int, required=True, help="Dataset number (int)")
    parser.add_argument("--rat", type=str, required=True, help="Rat ID (e.g., 'R1')")

    parser.add_argument("--mode", type=str, choices=["xyhd", "hdmd"], default="xyhd",
                        help="Rate-map type: 3D XY×HD (raw per-visit) or 2D HD×MD")

    # Bin params
    parser.add_argument("--n_hd_bins", type=int, default=60, help="Number of head-direction bins")
    parser.add_argument("--n_md_bins", type=int, default=60, help="Number of movement-direction bins (hdmd)")
    parser.add_argument("--n_loc_bins", type=int, default=20, help="Number of X bins (xyhd)")

    # Timing / thresholds
    parser.add_argument("--tau", type=float, default=0.120, help="Time bin size (s)")
    parser.add_argument("--vel_threshold", type=float, default=1.0, help="Velocity threshold (units/s)")

    # Smoothing (note: applied only to hdmd in this script; xyhd kept raw)

    # MD computation
    parser.add_argument("--md_smooth_window", type=int, default=200,
                        help="Smoothing window (samples) for MD from XY (hdmd)")

    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    get_raw_ratemaps(args)

if __name__ == "__main__":
    main()