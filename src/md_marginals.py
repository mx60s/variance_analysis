import argparse
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy.io.matlab._mio5_params import mat_struct
import time
from pathlib import Path

current_file_path = Path().resolve()

# -----------------------------
# Helpers
# -----------------------------

def get_md_from_xy(positions, smoothing_window=200):
    """
    Compute movement direction (md) in degrees from x/y positions, with circular smoothing.
    positions: ndarray [T, 4] with columns [timestamp, x, y, hd]
    Returns: md array [T]
    """
    def circular_smooth(angles, smoothing_window=5):
        angles_rad = np.radians(angles)
        sin_vals = np.sin(angles_rad)
        cos_vals = np.cos(angles_rad)
        kernel = np.ones(smoothing_window) / smoothing_window
        smooth_sin = np.convolve(sin_vals, kernel, mode='same')
        smooth_cos = np.convolve(cos_vals, kernel, mode='same')
        smooth_angles = np.degrees(np.arctan2(smooth_sin, smooth_cos)) % 360
        return smooth_angles

    x = positions[:, 1]
    y = positions[:, 2]

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    direction = np.degrees(np.arctan2(-dy, dx))
    direction = (direction + 360) % 360
    direction = circular_smooth(direction, smoothing_window=smoothing_window)

    # align length with positions by repeating last value
    direction = np.concatenate([direction, [direction[-1]]])
    return direction


def times_to_bin_indices(times, edges):
    """Map continuous times to indices of half-open bins defined by edges.
    Returns only indices that fall within [edges[0], edges[-1])."""
    idx = np.searchsorted(edges, times, side='right') - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    return idx[valid]


def extract_matlab_obj(obj):
    """Recursively extract MATLAB structs into Python dictionaries and arrays."""
    if not (isinstance(obj, mat_struct) or isinstance(obj, dict)):
        return obj

    out = {}
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


# -----------------------------
# Core computation: MD marginals only
# -----------------------------

def get_md_marginals_cv(spike_times, positions, cell_id_list,
                        n_md_bins=17, velocity_threshold=2, velocity_data=None,
                        smooth_window=200, cv_count=4, shuffle_md=False,
                        save_path=None):
    """
    Compute firing-rate *marginals over movement direction (MD) only* with CV splits.

    For each MD bin and each CV split, we average the per-bin firing rates across train/test time bins.

    Returns (lists of length cv_count):
        md_train_maps[cv]:  shape (n_md_bins, n_neurons)
        md_test_maps[cv]:   shape (n_md_bins, n_neurons)
        cv_masks:           list of boolean masks over base bins (train=True)
        base_time:          1D array of bin edges (len = n_bins+1)
        kept_neuron_ids:    order of neurons in the output arrays
        md_bins:            MD bin edges used
    """
    assert save_path is not None, "save_path must be provided"
    os.makedirs(save_path, exist_ok=True)

    seed = 111
    np.random.seed(seed)

    tau = 0.120  # seconds (approx 8 Hz bins)

    # Movement direction (smoothed)
    md = get_md_from_xy(positions, smoothing_window=smooth_window)
    if shuffle_md:
        np.random.shuffle(md)

    md_max = 360.0

    start_time = time.time()

    # Build time-bin edges (base_time)
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + tau, tau)
    n_points = len(base_time) - 1

    # Separate spikes by neuron, keeping only requested IDs
    neuron_idxs = np.array(sorted(np.unique(spike_times[:, 1]).astype(int)))
    keep = np.isin(neuron_idxs, cell_id_list)
    kept_neuron_ids = neuron_idxs[keep]

    # Vectorized frequency maps via histogram per neuron
    n_neurons = len(kept_neuron_ids)
    freq_maps = np.empty((n_points, n_neurons), dtype=float)
    for col, nid in enumerate(kept_neuron_ids):
        st = spike_times[spike_times[:, 1] == nid][:, 0]
        counts, _ = np.histogram(st, bins=base_time)
        freq_maps[:, col] = counts / tau

    # Optional velocity filtering (set low-velocity bins to NaN)
    if velocity_data is not None:
        avg_vel = np.empty(n_points)
        for i in range(n_points):
            mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i + 1])
            if np.any(mask):
                avg_vel[i] = np.nanmean(velocity_data[mask, 1])
            else:
                avg_vel[i] = np.nan
        low_vel_bins = np.where((np.isnan(avg_vel)) | (avg_vel < velocity_threshold))[0]
        if low_vel_bins.size > 0:
            print(f"Dropping {100*low_vel_bins.size/n_points:.2f}% of time bins due to low velocity")
            freq_maps[low_vel_bins, :] = np.nan

    # MD bin edges
    md_bins = np.linspace(0.0, md_max, n_md_bins + 1)

    # CV masks over time bins
    folds = np.array_split(np.arange(n_points), cv_count)
    cv_masks = []
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False  # False => test
        cv_masks.append(mask)

    # Output arrays
    md_train_maps = [np.full((n_md_bins, n_neurons), np.nan) for _ in range(cv_count)]
    md_test_maps  = [np.full((n_md_bins, n_neurons), np.nan) for _ in range(cv_count)]

    print("Computing MD marginals...")
    # Loop over MD bins only
    for i in range(n_md_bins):
        md_mask = (md >= md_bins[i]) & (md < md_bins[i + 1])
        if not np.any(md_mask):
            continue
        # Map those position times to base_time bin indices
        md_idx = times_to_bin_indices(positions[md_mask, 0], base_time)
        if md_idx.size == 0:
            continue
        # For each CV, compute train/test means
        for cv in range(cv_count):
            mask_cv = cv_masks[cv]
            train_idx = md_idx[mask_cv[md_idx]]
            test_idx  = md_idx[~mask_cv[md_idx]]

            if train_idx.size > 0:
                md_train_maps[cv][i, :] = np.nanmean(freq_maps[train_idx, :], axis=0)
            if test_idx.size > 0:
                md_test_maps[cv][i, :] = np.nanmean(freq_maps[test_idx, :], axis=0)

    runtime = (time.time() - start_time) / 60.0
    print(f"Runtime: {runtime:.2f} minutes")

    # Save neuron ordering early for downstream alignment
    np.save(Path(save_path) / f"neuron_id_order_md_{n_md_bins}.npy", kept_neuron_ids)

    return md_train_maps, md_test_maps, cv_masks, base_time, kept_neuron_ids, md_bins


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute MD-only marginal rate maps with CV from spike and position data.")
    parser.add_argument("--dataset", required=True, type=int, help="Dataset number (required).")
    parser.add_argument("--rat", required=True, type=str, help="Rat ID (required).")
    parser.add_argument("--n_md_bins", default=17, type=int, help="Number of MD bins (default: 17)")
    parser.add_argument("--smooth_window", required=True, type=int, help="Smoothing window for MD calculation (default: 200).")
    parser.add_argument("--vel_threshold", required=False, type=int, default=2, help="Velocity threshold in cm/s (default: 2).")
    parser.add_argument("--shuffle_md", required=False, action='store_true', help="Shuffle MD for control analysis.")
    parser.add_argument("--cv_count", required=False, type=int, default=4, help="Number of CV splits (default: 4)")
    args = parser.parse_args()

    # Load epochs
    epochs = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/BehavEpochs.mat",
                     struct_as_record=False, squeeze_me=True)
    if args.dataset == 6:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    else:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])

    # Load tracking -> positions [t, x, y, hd]
    tracking_data = pd.read_csv(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Tracking_data.csv")
    tracking_data = tracking_data.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    positions = tracking_data[['timestamp', 'x', 'y', 'hd']].to_numpy()
    position_mask = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[position_mask]

    # Normalize positions (shift so min is 0)
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))

    # Load cell types and pick HD cells (to match original behavior)
    cell_types = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/CellTypes.mat",
                         struct_as_record=False, squeeze_me=True)
    hd_list = extract_matlab_obj(cell_types['hd'])

    # Load spikes and build [time, neuron_id] array
    spike_data = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/SpikeData.mat",
                         struct_as_record=False, squeeze_me=True)
    spike_data = extract_matlab_obj(spike_data['S'])
    arr = []
    for a in spike_data['C']:
        arr.append(a['tsd']['t'])
    full_arr = []
    for i, a in enumerate(arr):
        for aa in a:
            full_arr.append([aa, i])
    spike_times = np.array(full_arr)
    spike_times = spike_times[np.argsort(spike_times[:, 0])]
    spike_mask = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[spike_mask]

    # Velocity (optional)
    velocity_data = None
    vel_path = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Velocity.mat"
    if os.path.exists(vel_path):
        velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
        velocity_data = extract_matlab_obj(velocity_data['vel'])
        velocity_data = np.dstack((velocity_data['t'], velocity_data['data']))[0]
        mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
        velocity_data = velocity_data[mask]

    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"
    os.makedirs(save_path, exist_ok=True)

    cell_id_list = np.where(hd_list == 1)[0]  # match original: operate on HD cells
    print(len(cell_id_list))

    md_train_maps, md_test_maps, cv_masks, base_time, kept_neuron_ids, md_bins = get_md_marginals_cv(
        spike_times=spike_times,
        positions=positions,
        cell_id_list=cell_id_list,
        n_md_bins=args.n_md_bins,
        velocity_threshold=args.vel_threshold,
        velocity_data=velocity_data,
        smooth_window=args.smooth_window,
        cv_count=args.cv_count,
        shuffle_md=args.shuffle_md,
        save_path=save_path,
    )

    # Save outputs (one file per CV split)
    suffix = "_shuffled_md" if args.shuffle_md else ""

    # Bin edges & metadata once
    np.save(save_path / f"md_bin_edges_{args.n_md_bins}{suffix}.npy", md_bins)
    np.save(save_path / f"cv_base_time_edges{suffix}.npy", base_time)

    for cv in range(len(md_train_maps)):
        np.save(save_path / f"md_marginal_train_cv{cv}_{args.n_md_bins}{suffix}.npy", md_train_maps[cv])
        np.save(save_path / f"md_marginal_test_cv{cv}_{args.n_md_bins}{suffix}.npy", md_test_maps[cv])
        np.save(save_path / f"cv_mask{cv}_md_{args.n_md_bins}{suffix}.npy", cv_masks[cv])

    print("Done. Saved MD marginal maps, neuron ordering, bin edges, and CV masks.")


if __name__ == "__main__":
    main()
