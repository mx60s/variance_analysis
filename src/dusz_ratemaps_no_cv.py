# Simplified rate map generation without cross-validation
import argparse
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy.io.matlab._mio5_params import mat_struct
import time
from tqdm import tqdm
from pathlib import Path

current_file_path = Path().resolve()

def get_md_from_xy(positions, smoothing_window=75):
    def rolling_mean(positions):
        kernel = np.ones(smoothing_window) / smoothing_window
        return np.convolve(positions, kernel, mode='same')

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

    direction = np.degrees(np.arctan2(dy, dx))
    direction = (direction + 360) % 360
    direction = circular_smooth(direction, smoothing_window=smoothing_window)

    direction = np.concatenate([direction, [direction[-1]]])
    return direction

def times_to_bin_indices(times, edges):
    idx = np.searchsorted(edges, times, side='right') - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    return idx[valid]

def get_rate_maps_simple(spike_times, positions, excit_list, 
                         n_hd_bins=17, n_md_bins=17, 
                         velocity_threshold=2, velocity_data=None,
                         n_spatial_bins=11, smooth_window=75, shuffle_md=False):
    """
    Compute rate maps without cross-validation - using all data.
    """
    seed = 111
    np.random.seed(seed)
    
    tau = 0.120  # approx 8Hz
    
    # Create movement direction (md) signal from positions 
    md = get_md_from_xy(positions, smoothing_window=smooth_window)

    if shuffle_md:
        np.random.shuffle(md)
    
    hd = positions[:, 3]
    hd = (hd + 360) % 360

    md_max = np.max(md)
    hd_max = np.max(hd)

    start_time = time.time()

    # Build base_time vector
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + tau, tau)

    # Separate spike data by neuron (only if in excit_list)
    neuron_idxs = set(spike_times[:, 1])
    spike_mat = []
    for i in tqdm(neuron_idxs, desc="Separating spike data by neuron"):
        if int(i) in excit_list:
            spike_mat.append(spike_times[spike_times[:, 1] == i])
    n_neurons = len(spike_mat)
    print(f"# neurons: {n_neurons}")

    # Create frequency maps
    freq_maps = np.empty((len(base_time) - 1, 2, n_neurons))
    for idx, i in enumerate(tqdm(range(n_neurons), desc="Computing freq maps")):
        current_spikes = spike_mat[idx]
        current_map = np.empty((len(base_time) - 1, 2))
        current_map[:, 0] = base_time[:-1]
        for j in range(len(base_time) - 1):
            running_freq = np.sum((current_spikes[:, 0] <= base_time[j + 1]) & 
                                  (current_spikes[:, 0] > base_time[j])) / tau
            current_map[j, 1] = running_freq
        freq_maps[:, 1, idx] = current_map[:, 1]
    print("Finished computing frequency maps.")

    # Apply velocity filter if provided
    if velocity_data is not None:
        avg_vel = np.empty(len(base_time) - 1)
        for i in range(len(base_time) - 1):
            mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i + 1])
            if np.any(mask):
                avg_vel[i] = np.nanmean(velocity_data[mask, 1])
            else:
                avg_vel[i] = np.nan
        low_vel_bins = np.where((np.isnan(avg_vel)) | (avg_vel < velocity_threshold))[0]
        ratio = (len(low_vel_bins) / (len(base_time) - 1)) * 100
        print(f"Dropping {ratio:.2f}% of time bins due to low velocity")
        freq_maps[low_vel_bins, 1, :] = np.nan

    # Define bin edges
    md_bins = np.linspace(0, md_max, n_md_bins + 1)
    hd_bins = np.linspace(0, hd_max, n_hd_bins + 1)

    # Preallocate rate map array
    rate_map = np.full((n_md_bins, n_hd_bins, n_neurons), np.nan)

    print("Creating rate maps...")
    # Loop over md bins
    for i in tqdm(range(len(md_bins) - 1), desc="MD bins"):
        freq_locs = np.where((md >= md_bins[i]) & (md < md_bins[i + 1]))[0]
        if freq_locs.size > 2:
            freq_locers = times_to_bin_indices(positions[freq_locs, 0], base_time)
            
            # Loop over hd bins
            for j in tqdm(range(len(hd_bins) - 1), desc="HD bins", leave=False):
                hd_mask = (hd[freq_locs] >= hd_bins[j]) & (hd[freq_locs] < hd_bins[j + 1])
                if np.any(hd_mask):
                    hd_freq_locs = freq_locs[hd_mask]
                    hd_freq_locers = times_to_bin_indices(positions[hd_freq_locs, 0], base_time)
                    
                    # Shuffle indices as in original
                    np.random.shuffle(hd_freq_locers)
                    
                    # Calculate mean firing rate for all neurons
                    if hd_freq_locers.size > 0:
                        mean_rate = np.nanmean(freq_maps[hd_freq_locers, 1, :], axis=0)
                    else:
                        mean_rate = np.full(n_neurons, np.nan)
                    
                    rate_map[i, j, :] = mean_rate
                        
    runtime = (time.time() - start_time) / 60
    print(f"Runtime: {runtime:.2f} minutes")
    return rate_map, base_time

def extract_matlab_obj(obj):
    """
    Recursively extract MATLAB structs into Python dictionaries and arrays.
    """
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

def main():
    parser = argparse.ArgumentParser(description="Compute rate maps without CV from spike and position data.")
    parser.add_argument("--dataset", required=True, type=int, help="Dataset number (required).")
    parser.add_argument("--rat", required=True, type=str, help="Rat ID (required).")
    parser.add_argument("--n_hd_bins", default=17, type=int, help="Number of head-direction bins.")
    parser.add_argument("--n_md_bins", default=17, type=int, help="Number of movement-direction bins.")
    parser.add_argument("--n_spatial_bins", default=11, type=int, help="Number of spatial bins.")
    parser.add_argument("--smooth_window", required=True, type=int, help="Smoothing window for md calculation.")
    parser.add_argument("--vel_threshold", required=False, type=int, help="Velocity threshold.", default=2)
    parser.add_argument("--shuffle_md", required=False, action='store_true', 
                       help="Shuffle MD for downstream analysis.")
    args = parser.parse_args()

    # Load data
    epochs = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/BehavEpochs.mat", 
                     struct_as_record=False, squeeze_me=True)
    if args.dataset == 6:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    else:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    
    tracking_data = pd.read_csv(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Tracking_data.csv")
    tracking_data = tracking_data.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    positions = tracking_data[['timestamp', 'x', 'y', 'hd']].to_numpy()
    position_mask = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[position_mask]
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))
    
    # Load cell types
    cell_types = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/CellTypes.mat", 
                         struct_as_record=False, squeeze_me=True)
    hd_list = extract_matlab_obj(cell_types['hd'])
    excit_list = extract_matlab_obj(cell_types['ex'])
    good_list = extract_matlab_obj(cell_types['gd'])
    fs_list = extract_matlab_obj(cell_types['fs'])
    print('hd', np.where(hd_list)[0])
    print('excit', np.where(excit_list)[0])
    print('good', np.where(good_list)[0])
    print('fs', np.where(fs_list)[0])
    
    # Load spike data
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
    
    # Load velocity data
    velocity_data = None
    vel_path = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Velocity.mat"
    if os.path.exists(vel_path):
        velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
        velocity_data = extract_matlab_obj(velocity_data['vel'])
        velocity_data = np.dstack((velocity_data['t'], velocity_data['data']))[0]
        mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
        velocity_data = velocity_data[mask]

    # Compute simple rate maps (no CV)
    rate_map, base_time = get_rate_maps_simple(
        spike_times, 
        positions, 
        np.where(hd_list == 1)[0],
        n_hd_bins=args.n_hd_bins,
        n_md_bins=args.n_md_bins,
        n_spatial_bins=args.n_spatial_bins,
        smooth_window=args.smooth_window,
        velocity_threshold=args.vel_threshold,
        velocity_data=velocity_data,
        shuffle_md=args.shuffle_md
    )

    # Save results
    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"
    os.makedirs(save_path, exist_ok=True)

    suffix = "_shuffled_md" if args.shuffle_md else ""
    
    print(f"Saving at {save_path}/hd_md_rate_map_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy")
    np.save(f"{save_path}/hd_md_rate_map_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", rate_map)
    np.save(f"{save_path}/base_time_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", base_time)
    
if __name__ == "__main__":
    main()