import argparse
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy.io.matlab._mio5_params import mat_struct
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict

from pathlib import Path
current_file_path = Path().resolve()

def get_rate_maps(spike_times, positions, excit_list, 
                  n_hd_bins=17, n_x_bins=17, n_y_bins=17, 
                  velocity_threshold=2, velocity_data=None):
    """
    Build spatial (XY) and head-direction rate maps.

    spike_times: (N x 2) array of [timestamp, neuron_id]
    positions: (M x 4) array of [timestamp, x, y, head_direction]
    excit_list: indices of excitatory neurons
    n_hd_bins: number of bins to use for head-direction
    n_x_bins: number of bins to use for x dimension
    n_y_bins: number of bins to use for y dimension
    velocity_threshold: velocity threshold for filtering
    velocity_data: velocity data for filtering
    """
    seed = 111
    np.random.seed(seed=seed)
    
    tau = 0.120  # approx 8Hz

    # Extract position and head direction data
    x = positions[:, 1]
    y = positions[:, 2]
    hd = positions[:, 3]
    hd = (hd + 360) % 360

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    hd_max = np.max(hd)

    start_time = time.time()

    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + tau, tau)
    print(len(base_time)) 

    neuron_idxs = set(spike_times[:, 1])
    spike_mat = []
    for i in tqdm(neuron_idxs, desc="Separating spike data by neuron"):
        if int(i) in excit_list:
            spike_mat.append(spike_times[spike_times[:, 1] == i])
    n_neurons = len(spike_mat)
    print(f"# neurons: {n_neurons}")

    freq_maps = np.empty((len(base_time), 2, n_neurons))

    def compute_freq_map_for_neuron(i, spike_mat_i, base_time, tau):
        """Compute the frequency map for the i-th neuron."""
        current_map = np.vstack((base_time, np.empty(len(base_time)))).T
        for j in range(len(base_time) - 1):
            running_freq = len(np.where(
                (spike_mat_i[:, 0] <= base_time[j + 1]) &
                (spike_mat_i[:, 0] > base_time[j])
            )[0]) / tau
            current_map[j, 1] = running_freq
            
        current_map[-1, 1] = len(np.where(spike_mat_i[:, 0] > base_time[-1])[0]) / tau # get the last bin
        return current_map

    results = Parallel(n_jobs=-1)(
        delayed(compute_freq_map_for_neuron)(i, spike_mat[i], base_time, tau) 
        for i in tqdm(range(n_neurons), desc="Computing freq maps")
    )
        
    print("Unpacking frequency maps...")
    for i, current_map in enumerate(results):
        freq_maps[:, :, i] = current_map

    if velocity_data is not None:
        avg_vel = np.empty(len(base_time))
        for i in range(len(base_time) - 1):
            mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i + 1])
            if np.any(mask):
                avg_vel[i] = np.nanmean(velocity_data[mask, 1])
            else:
                avg_vel[i] = np.nan
        avg_vel[-1] = np.nan

        low_vel_bins = np.where((np.isnan(avg_vel)) | (avg_vel < velocity_threshold))[0]
        ratio = (len(low_vel_bins) / len(base_time)) * 100
        print(f"Dropping {ratio:.2f}% due to low velocity")
        freq_maps[low_vel_bins, 1, :] = np.nan
    
    # Create bins for x, y, and hd
    x_bins = np.linspace(x_min, x_max, n_x_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_y_bins + 1)
    hd_bins = np.linspace(0, hd_max, n_hd_bins + 1)

    # Initialize raw visits array with XY and HD dimensions
    raw_visits = np.zeros((n_x_bins, n_y_bins, n_hd_bins, 500, n_neurons))
    raw_visits[:] = np.nan
    
    # Initialize occupancy map
    xy_hd_occupancy_map = np.zeros((n_x_bins, n_y_bins, n_hd_bins))

    print("Creating rate maps...")
    for i in tqdm(range(len(x_bins) - 1), desc="Building rate maps (X loop)"):
        # Find positions within this X bin
        x_freq_locs = np.where(
            (x < x_bins[i + 1]) &
            (x >= x_bins[i])
        )[0]

        if x_freq_locs.size > 0:
            freq_locers = np.array([
                np.abs(base_time - positions[idx, 0]).argmin()
                for idx in freq_locs
            ])
            for j in range(len(y_bins) - 1):
                # Find positions within this Y bin (and X bin)
                y_freq_locs = np.where(
                    (y[x_freq_locs] >= y_bins[j]) &
                    (y[x_freq_locs] < y_bins[j + 1])
                )[0]
                xy_freq_locs = x_freq_locs[y_freq_locs]

                if xy_freq_locs.size > 0:
                    for k in range(len(hd_bins) - 1):
                        hd_freq_locs = np.where(
                            (hd[xy_freq_locs] >= hd_bins[k]) &
                            (hd[xy_freq_locs] < hd_bins[k + 1])
                        )[0]
                        xy_hd_freq_locs = xy_freq_locs[hd_freq_locs]

                        if xy_hd_freq_locs.size > 10:
                            # Find corresponding time indices
                            xy_hd_freq_locers = np.array([
                                np.abs(base_time - positions[idx, 0]).argmin()
                                for idx in xy_hd_freq_locs
                            ])

                            xy_hd_occupancy_map[i, j, k] = len(xy_hd_freq_locers)
                            
                            l = min(500, len(xy_hd_freq_locers))
                            raw_visits[i, j, k, :l, :] = freq_maps[xy_hd_freq_locers[:l], 1, :]

    runtime = (time.time() - start_time) / 60
    print(f"Runtime: {runtime:.2f} minutes")

    return raw_visits

def extract_matlab_obj(obj):
    """
    Recursively extract MATLAB structs into Python dictionaries and arrays.
    """
    if not (isinstance(obj, mat_struct) or isinstance(obj, dict)):
        return obj

    out = {}
    obj = obj.__dict__ if isinstance(obj, mat_struct) else obj

    for key, val in obj.items():
        # Skip metadata
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
    parser = argparse.ArgumentParser(description="Compute XY-HD rate maps from spike and position data.")
    parser.add_argument("--dataset", required=True, type=int, help="Dataset number (required).")
    parser.add_argument("--rat", required=True, type=str, help="Rat ID (required).")
    parser.add_argument("--n_hd_bins", default=17, type=int, 
                        help="Number of head-direction bins. (default: 17)")
    parser.add_argument("--n_x_bins", default=11, type=int, 
                        help="Number of X spatial bins. (default: 11)")
    parser.add_argument("--n_y_bins", default=11, type=int, 
                        help="Number of Y spatial bins. (default: 11)")
    parser.add_argument("--vel_threshold", required=False, type=int, help="Velocity threshold (default: 2 cm/s).", default=2)

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
    
    # Normalize positions
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

    # collect the excitatory neurons which are also hd neurons for analysis later
    excit_not_hd = (excit_list == 1) & (hd_list == 0)
    excit_not_hd = excit_not_hd.astype(int)

    print('excit_not_hd', np.where(excit_not_hd))
    
    # Load spike data
    spike_data = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/SpikeData.mat", 
                         struct_as_record=False, squeeze_me=True)
    spike_data = extract_matlab_obj(spike_data['S'])

    # Build spike array
    arr = []
    for a in spike_data['C']:
        arr.append(a['tsd']['t'])
    full_arr = []
    for i, a in enumerate(arr):
        for aa in a:
            full_arr.append([aa, i])
    spike_times = np.array(full_arr)
    spike_times = spike_times[np.argsort(spike_times[:, 0])]
    
    # Filter spikes by epoch
    spike_mask = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[spike_mask]

    # Load in velocity data
    velocity_data = []
    vel_path = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Velocity.mat"
    velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
        
    velocity_data = extract_matlab_obj(velocity_data['vel'])
    velocity_data = np.dstack((velocity_data['t'], velocity_data['data']))[0]
    mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
    velocity_data = velocity_data[mask]

    # Compute rate maps
    raw_visits = get_rate_maps(
        spike_times, 
        positions, 
        np.where(hd_list == 1)[0],
        n_hd_bins=args.n_hd_bins,
        n_x_bins=args.n_x_bins,
        n_y_bins=args.n_y_bins,
        velocity_threshold=args.vel_threshold,
        velocity_data=velocity_data
    )

    # Save results
    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    np.save(f"{save_path}/xy_hd_raw_visits_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}", raw_visits)
    print(f"Saved XY-HD rate maps with shape: {raw_visits.shape}")

if __name__ == "__main__":
    main()