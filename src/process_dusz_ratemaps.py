import argparse
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy.io.matlab._mio5_params import mat_struct
import time
from tqdm import tqdm
from joblib import Parallel, delayed

def get_rate_maps(spike_times, positions, excit_list, 
                  n_hd_bins=17, filter_vel=False, n_spatial_bins=11):
    """
    Build spatial and head-direction rate maps.

    spike_times: (N x 2) array of [timestamp, neuron_id]
    positions: (M x 4) array of [timestamp, x, y, head_direction]
    excit_list: indices of excitatory neurons
    n_hd_bins: number of bins to use for head-direction
    filter_vel: whether or not to filter by velocity threshold
    n_spatial_bins: number of bins to use for x-y space
    """
    seed = 111
    np.random.seed(seed=seed)
    
    timebin_tol = 1e-4
    tau = 0.120  # approx 8Hz
    interpolation_points = 2
    velocity_threshold = 5  # cm/s

    env_len = max(np.max(positions[:, 1]), np.max(positions[:, 2]))
    hd_max = np.max(positions[:, 3])

    bin_count = n_spatial_bins
    bin_width = env_len / bin_count

    start_time = time.time()
    print(f"Seed {seed}")

    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0], tau)

    neuron_idxs = set(spike_times[:, 1])
    spike_mat = []
    for i in tqdm(neuron_idxs, desc="Separating spike data by neuron"):
        if int(i) in excit_list:
            spike_mat.append(spike_times[spike_times[:, 1] == i])
    n_neurons = len(spike_mat)
    print(f"# neurons: {n_neurons}")

    # --- VELOCITY FILTERING (currently disabled) ---
    fast_bins = base_time 

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
        return current_map

    # Parallel processing for each neuron
    results = Parallel(n_jobs=-1)(
        delayed(compute_freq_map_for_neuron)(i, spike_mat[i], base_time, tau) 
        for i in tqdm(range(n_neurons), desc="Computing freq maps")
    )
        
    # Put the results into freq_maps
    print("Unpacking frequency maps...")
    for i, current_map in enumerate(results):
        freq_maps[:, :, i] = current_map

    # --- BUILD SPATIAL + HEAD-DIRECTION RATE MAPS ---
    spatial_bins = np.linspace(0, env_len, bin_count + 1)
    rate_maps = np.zeros((bin_count, bin_count, n_neurons))

    hd_bins = np.linspace(0, hd_max, n_hd_bins + 1)
    hd_rate_map = np.zeros((bin_count, bin_count, n_hd_bins, n_neurons))
    hd_occupancy_map = np.zeros((bin_count, bin_count, n_hd_bins))
    hd_rate_map_fold0 = np.zeros((bin_count, bin_count, n_hd_bins, n_neurons))
    hd_rate_map_fold1 = np.zeros((bin_count, bin_count, n_hd_bins, n_neurons))
    raw_hd_rate_maps = np.zeros((bin_count, bin_count, 13959, n_hd_bins, n_neurons))  # max visits?

    raw_hd_rate_maps[:] = np.nan
    hd_rate_map[:] = np.nan
    hd_rate_map_fold0[:] = np.nan
    hd_rate_map_fold1[:] = np.nan

    print("Creating rate maps...")
    for i in tqdm(range(len(spatial_bins) - 1), desc="Building rate maps (outer loop)"):
        for j in range(len(spatial_bins) - 1):
            freq_locs = np.where(
                (positions[:, 1] <= spatial_bins[i + 1]) &
                (positions[:, 1] >= spatial_bins[i]) &
                (positions[:, 2] <= spatial_bins[j + 1]) &
                (positions[:, 2] >= spatial_bins[j])
            )[0]

            if freq_locs.size > 5:
                freq_locers = np.array([
                    np.abs(base_time - positions[idx, 0]).argmin()
                    for idx in freq_locs
                ])

                # Filter by velocity threshold (currently disabled)
                mask = np.any(
                    np.isclose(base_time[freq_locers, None], fast_bins, atol=timebin_tol),
                    axis=1
                )
                freq_locers = freq_locers[mask]

                all_hd_freq_locs = []
                for l in range(len(hd_bins) - 1):
                    hd_freq_locs = np.where(
                        (positions[freq_locs, 3] >= hd_bins[l]) &
                        (positions[freq_locs, 3] < hd_bins[l + 1])
                    )[0]
                    hd_freq_locs = freq_locs[hd_freq_locs]

                    hd_occupancy_map[i, j, l] = len(hd_freq_locs)
                    if hd_freq_locs.size > 2:
                        all_hd_freq_locs.append(hd_freq_locs)
                        hd_freq_locers = np.array([
                            np.abs(base_time - positions[idx, 0]).argmin()
                            for idx in hd_freq_locs
                        ])
                        mask = np.any(
                            np.isclose(base_time[hd_freq_locers, None], fast_bins, atol=timebin_tol),
                            axis=1
                        )
                        hd_freq_locers = hd_freq_locers[mask]

                        hd_rate_map[i, j, l] = np.mean(freq_maps[hd_freq_locers, 1, :], axis=0)

                        np.random.shuffle(hd_freq_locers)
                        half = len(hd_freq_locers) // 2
                        hd_rate_map_fold0[i, j, l] = np.mean(freq_maps[hd_freq_locers[:half], 1, :], axis=0)
                        hd_rate_map_fold1[i, j, l] = np.mean(freq_maps[hd_freq_locers[half:], 1, :], axis=0)

                        n = hd_freq_locers.shape[0]
                        raw_hd_rate_maps[i, j, :n, l, :] = freq_maps[hd_freq_locers, 1, :]

                # Compute the spatial rate map by combining all HD bins
                if len(all_hd_freq_locs) == 0:
                    rate_maps[i, j, :] = np.nan
                else:
                    freq_locs_concat = np.concatenate(all_hd_freq_locs)
                    freq_locers_concat = np.array([
                        np.abs(base_time - positions[idx, 0]).argmin()
                        for idx in freq_locs_concat
                    ])
                    rate_maps[i, j] = np.nanmean(freq_maps[freq_locers_concat, 1, :], axis=0)

    runtime = (time.time() - start_time) / 60
    print(f"Runtime: {runtime:.2f} minutes")

    return rate_maps, hd_rate_map, hd_rate_map_fold0, hd_rate_map_fold1, hd_occupancy_map, raw_hd_rate_maps

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
    parser = argparse.ArgumentParser(description="Compute rate maps and HD maps from spike and position data.")
    parser.add_argument("--dataset", required=True, type=int, help="Dataset number (required).")
    parser.add_argument("--rat", required=True, type=str, help="Rat ID (required).")
    parser.add_argument("--n_hd_bins", default=17, type=int, 
                        help="Number of head-direction bins. (default: 17)")
    parser.add_argument("--n_spatial_bins", default=11, type=int, 
                        help="Number of spatial bins in x-y space. (default: 11)")
    parser.add_argument("--filter_vel", default=False, action="store_true", 
                        help="Apply velocity filtering (off by default).")

    args = parser.parse_args()

    # Load data
    epochs = loadmat(f"/home/mag/Space/data/dusz/Dataset_{args.dataset}/{args.rat}/Data/BehavEpochs.mat", 
                     struct_as_record=False, squeeze_me=True)
    epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    
    tracking_data = pd.read_csv(f"/home/mag/Space/data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Tracking_data.csv")
    tracking_data = tracking_data.set_axis(['timestamp', 'x', 'y', 'hd_mystery', 'mystery', 'another_hd_mystery'], axis=1)
    
    positions = tracking_data[['timestamp', 'x', 'y', 'another_hd_mystery']].to_numpy()
    position_mask = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[position_mask]
    
    # Normalize positions
    positions[:, 3] += abs(np.min(positions[:, 3]))
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))
    
    # Load cell types
    cell_types = loadmat(f"/home/mag/Space/data/dusz/Dataset_{args.dataset}/{args.rat}/Data/CellTypes.mat", 
                         struct_as_record=False, squeeze_me=True)
    hd_list = extract_matlab_obj(cell_types['hd'])
    #good_list = extract_matlab_obj(cell_types['gd'])
    excit_list = extract_matlab_obj(cell_types['ex'])
    
    # collect the excitatory neurons which are also hd neurons for analysis later
    hd_list = hd_list[np.where(excit_list == 1)[0]]
    print('??', hd_list)
    hd_neuron_idx = np.where(hd_list == 1)[0]
    
    # Load spike data
    spike_data = loadmat(f"/home/mag/Space/data/dusz/Dataset_{args.dataset}/{args.rat}/Data/SpikeData.mat", 
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

    # Compute rate maps
    rate_maps, hd_rate_map, hd_rate_map_fold0, hd_rate_map_fold1, hd_occupancy_map, raw_hd_rate_maps = get_rate_maps(
        spike_times, 
        positions, 
        np.where(excit_list == 1)[0],
        n_hd_bins=args.n_hd_bins,
        filter_vel=args.filter_vel,
        n_spatial_bins=args.n_spatial_bins
    )

    # Save results
    save_path = f"/home/mag/Space/results/dusz/Dataset_{args.dataset}/{args.rat}"
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    np.save(f"{save_path}/rate_maps_n_hd_bins_{args.n_hd_bins}", rate_maps)
    np.save(f"{save_path}/hd_rate_map_n_hd_bins_{args.n_hd_bins}", hd_rate_map)
    np.save(f"{save_path}/hd_rate_map_fold0_n_hd_bins_{args.n_hd_bins}", hd_rate_map_fold0)
    np.save(f"{save_path}/hd_rate_map_fold1_n_hd_bins_{args.n_hd_bins}", hd_rate_map_fold1)
    np.save(f"{save_path}/hd_occupancy_map_n_hd_bins_{args.n_hd_bins}", hd_occupancy_map)
    np.save(f"{save_path}/raw_hd_rate_maps_n_hd_bins_{args.n_hd_bins}", raw_hd_rate_maps)
    np.save(f"{save_path}/hd_neuron_indexes", hd_neuron_idx)

if __name__ == "__main__":
    main()
