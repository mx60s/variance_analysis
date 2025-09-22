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

# ok so this needs to be a binning function right?
# you can assume youll always be given spikes and times
# and then also behavioral variables which come in lists with times
# maybe then you want to be outputting separate lists of bin visits

# I want behavioral_vars to be timestamps x whatever other columns

# here's how I was doing this before
'''
    time_step_gap = positions[3, 0] - positions[2, 0]
    point_steps = np.diff(positions[:, 1:3], axis=0) # the x and y coordinates, leaving behind timestamps diff between consecutive steps
    point_steps = np.vstack((point_steps, point_steps[-1])) # make it the same length
    point_dist = np.linalg.norm(point_steps, axis=1) / time_step_gap

    bin_indices = np.searchsorted(base_time, positions[:, 0]) - 1
    valid_mask = (bin_indices >= 0) & (bin_indices < len(base_time) - 1)
    bin_indices = bin_indices[valid_mask]
    point_dist_valid = point_dist[valid_mask]

    sum_dist = np.bincount(bin_indices, weights=point_dist_valid)
    count_dist = np.bincount(bin_indices)
    with np.errstate(divide='ignore', invalid='ignore'):
        vel_list = sum_dist / count_dist
    vel_list[np.isnan(vel_list)] = 0  

    if len(vel_list) < len(base_time):
        vel_list = np.pad(vel_list, (0, len(base_time) - len(vel_list)), 'constant')

    fast_indices = np.where(vel_list >= velocity_threshold)[0]
    fast_bins = base_time[fast_indices] if filter_vel else base_time
'''

# you could just pass into this function the spike and the bin it falls into lol
# so then you could do binning first

# ok so that's what neds to be done

def get_binned_spike_freqs(binned_spikes, behavioral_vars, excit_list,
                      n_bins=(17,17), visit_cutoff = 4, behav_filter=None):
    """
    Computes neuronal frequencies per time bin, then bins according to whatever
    behavioral variables you want.
    This assumes that behavioral_vars[0] is timesteps
    binned_spikes (N_timebins, 2). Col 0 is time bin index, col 1 is spike index
    Spike binning occurs earlier so you can filter (like by velocity)
    """
    timebin_tol = 1e-4
    tau = 0.120

    assert(behavioral_vars.shape[1] == len(n_bins))

    behavioral_max_vars = np.max(behavioral_vars[:, 1:], axis=0)
    bin_widths = [bin_max / n for bin_max, n in zip(behavioral_max_vars, n_bins)]
    bin_edges = [
        np.linspace(0, bin_max, bin_count + 1) 
        for bin_max, bin_count in zip(behavioral_max_vars, bin_widths)
    ]
    
    start_time = time.time()
    
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0], tau)

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
        return current_map

    results = Parallel(n_jobs=-1)(
        delayed(compute_freq_map_for_neuron)(i, spike_mat[i], base_time, tau) 
        for i in tqdm(range(n_neurons), desc="Computing freq maps")
    )
        
    print("Unpacking frequency maps...")
    for i, current_map in enumerate(results):
        freq_maps[:, :, i] = current_map

    # split everything off into bins and collect the timepoints that fall into each
    bins_per_dim = []
    for dim in range(behavioral_vars.shape[1] - 1):
        bins_per_dim.append(np.digitize(behavioral_vars[:, dim + 1], bin_edges[dim]) - 1)
        
    bins_per_dim = np.array(bins_per_dim).T  # shape (timepoints, N)

    bin_dict = defaultdict(list)
        
    for t in range(len(bins_per_dim)):
        bin_indices = tuple(bins_per_dim[t])
        bin_dict[bin_indices].append(t)

    # remove low-visit bins 
    for bin_idxs, visits in bin_dict.items():
        if len(visits) < visit_cutoff:
            del bin_dict[bin_idxs]
            
    return bin_dict
        

def get_md_from_xy(positions, smoothing_window=75):
    def rolling_mean(positions):
        kernel = np.ones(smoothing_window) / smoothing_window
        return np.convolve(positions, kernel, mode='same')
        
    x = rolling_mean(positions[:, 1])
    y = rolling_mean(positions[:, 2])
    
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    direction = np.degrees(np.arctan2(dy, dx))
    direction = (direction + 360) % 360

    direction = np.concatenate([direction, [direction[-1]]])

    return direction
    

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
    velocity_threshold = 5  # cm/s

    env_len = max(np.max(positions[:, 1]), np.max(positions[:, 2]))
    hd_max = np.max(positions[:, 3])

    #print(np.any(np.isnan(positions[:, 2])))
    md = get_md_from_xy(positions)
    positions[:, 3] = md
    #print(positions[:3, 3])

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

    results = Parallel(n_jobs=-1)(
        delayed(compute_freq_map_for_neuron)(i, spike_mat[i], base_time, tau) 
        for i in tqdm(range(n_neurons), desc="Computing freq maps")
    )
        
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
            freq_locs = np.where( # find all the positions in the bin
                (positions[:, 1] < spatial_bins[i + 1]) &
                (positions[:, 1] >= spatial_bins[i]) &
                (positions[:, 2] < spatial_bins[j + 1]) &
                (positions[:, 2] >= spatial_bins[j])
            )[0]

            if freq_locs.size > 0:
                freq_locers = np.array([ # find all of the time windows in there 
                    np.abs(base_time - positions[idx, 0]).argmin()
                    for idx in freq_locs
                ])

                all_hd_freq_locs = []
                for l in range(len(hd_bins) - 1):
                    hd_freq_locs = np.where(
                        (positions[freq_locs, 3] >= hd_bins[l]) &
                        (positions[freq_locs, 3] < hd_bins[l + 1])
                    )[0]
                    hd_freq_locs = freq_locs[hd_freq_locs]

                    hd_occupancy_map[i, j, l] = len(hd_freq_locs)
                    if hd_freq_locs.size > 30:
                        all_hd_freq_locs.append(hd_freq_locs)
                        hd_freq_locers = np.array([
                            np.abs(base_time - positions[idx, 0]).argmin()
                            for idx in hd_freq_locs
                        ])
                        #mask = np.any(
                        #    np.isclose(base_time[hd_freq_locers, None], fast_bins, atol=timebin_tol),
                        #    axis=1
                        #)
                        #hd_freq_locers = hd_freq_locers[mask]

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
    positions[:, 3] += abs(np.min(positions[:, 3]))
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))
    
    # Load cell types
    cell_types = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/CellTypes.mat", 
                         struct_as_record=False, squeeze_me=True)
    hd_list = extract_matlab_obj(cell_types['hd'])
    #good_list = extract_matlab_obj(cell_types['gd'])
    excit_list = extract_matlab_obj(cell_types['ex'])
    
    # collect the excitatory neurons which are also hd neurons for analysis later
    hd_list = hd_list[np.where(excit_list == 1)[0]]
    hd_neuron_idx = np.where(hd_list == 1)[0]
    
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
    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"
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
