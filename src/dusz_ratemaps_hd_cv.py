import argparse
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy.io.matlab._mio5_params import mat_struct
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

current_file_path = Path().resolve()

def get_md_from_xy(positions, smoothing_window=200):
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
        
    # Using x and y as-is
    x = positions[:, 1]
    y = positions[:, 2]
    
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    direction = np.degrees(np.arctan2(-dy, dx))
    direction = (direction + 360) % 360
    direction = circular_smooth(direction, smoothing_window=smoothing_window)

    direction = np.concatenate([direction, [direction[-1]]])
    return direction

def times_to_bin_indices(times, edges):
    idx = np.searchsorted(edges, times, side='right') - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    return idx[valid]

def _gaussian_filter(rate_map, smooth_sigma):
    V = rate_map.copy()
    V[np.isnan(rate_map)] = 0
    U = 0*rate_map.copy() + 1
    U[np.isnan(rate_map)] = 0
    VV = gaussian_filter(V, sigma=smooth_sigma, axes=(0, 1, 2))
    UU = gaussian_filter(U, sigma=smooth_sigma, axes=(0, 1, 2))
    return VV/UU

def get_rate_maps_cv(spike_times, positions, excit_list, 
                     n_hd_bins=17, n_md_bins=17, 
                     velocity_threshold=2, velocity_data=None,
                     n_spatial_bins=11, smooth_window=200, cv_count=4, shuffle_md=False, save_path=None):
    """
    Compute frequency maps then generate cross validation (CV) splits for rate maps.
    For each md/hd bin (after shuffling and taking 100 sample indices), we do the following:
      (a) Global: Use all sample indices in that bin and then use a CV mask (on base_time indices)
          to create a train/test average firing rate.
      (b) Folded: First split the 100 samples into two halves (fold0 and fold1, as before) and then
          for each fold, use the CV mask to generate train/test averages.
    This function returns four lists (length = cv_count):
       global_train_maps, global_test_maps, folded_train_maps, folded_test_maps.
    The global maps have shape (n_md_bins, n_hd_bins, n_neurons) and the folded maps have
    shape (n_md_bins, n_hd_bins, 2, n_neurons) where the 3rd axis represents the two folds.
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

    md_max = 360. #np.max(md)
    hd_max = 360. #np.max(hd)

    start_time = time.time()

    # Build base_time vector.
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + tau, tau)

    # Separate spike data by neuron (only if in excit_list)
    neuron_idxs = np.array(sorted(np.unique(spike_times[:, 1]).astype(int)))

    # Build spike_mat and remember neuron order:
    spike_mat = []
    kept_neuron_ids = []
    for i in tqdm(neuron_idxs, desc="Separating spike data by neuron"):
        if int(i) in excit_list:
            spike_mat.append(spike_times[spike_times[:, 1] == i])
            kept_neuron_ids.append(int(i))
    kept_neuron_ids = np.array(kept_neuron_ids, dtype=int)
    
    np.save(f"{save_path}/neuron_id_order_{n_hd_bins}_{n_md_bins}.npy", kept_neuron_ids)
    n_neurons = len(spike_mat)
    print(f"# neurons: {n_neurons}")

    # Create frequency maps (for each time bin and each neuron)
    # freq_maps: shape (len(base_time), 2, n_neurons) where column 0 = time, col 1 = frequency
    freq_maps = np.empty((len(base_time) - 1, 2, n_neurons))
    for idx, i in enumerate(tqdm(range(n_neurons), desc="Computing freq maps")):
        current_spikes = spike_mat[idx]
        current_map = np.empty((len(base_time) - 1, 2))
        current_map[:, 0] = base_time[:-1]
        for j in range(len(base_time) - 1):
            running_freq = np.sum((current_spikes[:, 0] <= base_time[j + 1]) & 
                                  (current_spikes[:, 0] > base_time[j])) / tau
            current_map[j, 1] = running_freq
        # Handle last bin:
        #current_map[-1, 1] = np.sum(current_spikes[:, 0] > base_time[-1]) / tau
        freq_maps[:, 1, idx] = current_map[:, 1]
    print("Finished computing frequency maps.")

    # (Optional) Apply velocity filter if velocity_data is provided.
    if velocity_data is not None:
        avg_vel = np.empty(len(base_time) - 1)
        for i in range(len(base_time) - 1):
            mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i + 1])
            if np.any(mask):
                avg_vel[i] = np.nanmean(velocity_data[mask, 1])
            else:
                avg_vel[i] = np.nan
        #avg_vel[-1] = np.nan
        low_vel_bins = np.where((np.isnan(avg_vel)) | (avg_vel < velocity_threshold))[0]
        ratio = (len(low_vel_bins) / (len(base_time) - 1)) * 100
        print(f"Dropping {ratio:.2f}% of time bins due to low velocity")
        freq_maps[low_vel_bins, 1, :] = np.nan # but ultimately freq_maps is still of shape len(base_time)

    # Define bin edges for md and hd
    md_bins = np.linspace(0, md_max, n_md_bins + 1)
    hd_bins = np.linspace(0, hd_max, n_hd_bins + 1)

    # Create CV masks. Each element is a boolean array (length = len(base_time))
    # where True indicates a train time bin and False indicates test.
    cv_masks = []
    n_points = len(base_time) - 1
    folds = np.array_split(np.arange(n_points), cv_count)
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)

    # Preallocate output arrays for each CV split.
    # Global rate maps: shape (n_md_bins, n_hd_bins, n_neurons)
    global_train_maps = [np.full((n_md_bins, n_hd_bins, n_neurons), np.nan) for _ in range(cv_count)]
    global_test_maps = [np.full((n_md_bins, n_hd_bins, n_neurons), np.nan) for _ in range(cv_count)]
    # Folded maps: shape (n_md_bins, n_hd_bins, 2, n_neurons) where 2 corresponds to fold0 and fold1.
    folded_train_maps = [np.full((n_md_bins, n_hd_bins, 2, n_neurons), np.nan) for _ in range(cv_count)]
    folded_test_maps = [np.full((n_md_bins, n_hd_bins, 2, n_neurons), np.nan) for _ in range(cv_count)]

    print("Creating CV rate maps...")
    # Loop over md bins
    for i in tqdm(range(len(md_bins) - 1), desc="MD bins"):
        # Get indices (from positions) where the md falls in the current bin.
        freq_locs = np.where((md >= md_bins[i]) & (md < md_bins[i + 1]))[0]
        if freq_locs.size > 2:
            # For each position index in this md bin, find the nearest index in base_time.
            #freq_locers = np.array([np.abs(base_time - positions[idx, 0]).argmin() for idx in freq_locs])
            freq_locers = times_to_bin_indices(positions[freq_locs, 0], base_time)
            
            # Loop over hd bins
            for j in tqdm(range(len(hd_bins) - 1), desc="HD bins", leave=False):
                # Select those positions (from freq_locs) with head direction in the current hd bin.
                hd_mask = (hd[freq_locs] >= hd_bins[j]) & (hd[freq_locs] < hd_bins[j + 1])
                if np.any(hd_mask):
                    hd_freq_locs = freq_locs[hd_mask]
                    # Get corresponding base_time indices.
                    #hd_freq_locers = np.array([np.abs(base_time - positions[idx, 0]).argmin() for idx in hd_freq_locs])
                    hd_freq_locers = times_to_bin_indices(positions[hd_freq_locs, 0], base_time)
                    
                    # Only process if we have enough samples in this bin.
                    #if hd_freq_locers.size < 100:
                    #    continue
                        
                    # Shuffle and take a sample of 100 indices.
                    #np.random.shuffle(hd_freq_locers)
                    #hd_freq_locers = hd_freq_locers[:100]
                    
                    # Precompute the folded indices (for the folded method).
                    half = len(hd_freq_locers) // 2
                    # For an odd number, we drop the last sample (as in your original code)
                    if len(hd_freq_locers) % 2 == 0:
                        fold0_idxs = hd_freq_locers[:half]
                        fold1_idxs = hd_freq_locers[half:]
                    else:
                        fold0_idxs = hd_freq_locers[:half]
                        fold1_idxs = hd_freq_locers[half:-1]
                    
                    # For each CV split, calculate the train and test averages.
                    for cv in range(cv_count):
                        mask_cv = cv_masks[cv]
                        train_idx_global = hd_freq_locers[mask_cv[hd_freq_locers]]
                        test_idx_global = hd_freq_locers[~mask_cv[hd_freq_locers]]
                        
                        if train_idx_global.size > 0:
                            global_train_rate = np.nanmean(freq_maps[train_idx_global, 1, :], axis=0)
                        else:
                            global_train_rate = np.full(n_neurons, np.nan)
                        if test_idx_global.size > 0:
                            global_test_rate = np.nanmean(freq_maps[test_idx_global, 1, :], axis=0)
                        else:
                            global_test_rate = np.full(n_neurons, np.nan)
                        
                        global_train_maps[cv][i, j, :] = global_train_rate
                        global_test_maps[cv][i, j, :] = global_test_rate
                        
                        # Folded method: do the same for each fold.
                        # Fold0:
                        train_idx_fold0 = fold0_idxs[mask_cv[fold0_idxs]]
                        test_idx_fold0 = fold0_idxs[~mask_cv[fold0_idxs]]
                        if train_idx_fold0.size > 0:
                            fold0_train_rate = np.nanmean(freq_maps[train_idx_fold0, 1, :], axis=0)
                        else:
                            fold0_train_rate = np.full(n_neurons, np.nan)
                        if test_idx_fold0.size > 0:
                            fold0_test_rate = np.nanmean(freq_maps[test_idx_fold0, 1, :], axis=0)
                        else:
                            fold0_test_rate = np.full(n_neurons, np.nan)
                        # Fold1:
                        train_idx_fold1 = fold1_idxs[mask_cv[fold1_idxs]]
                        test_idx_fold1 = fold1_idxs[~mask_cv[fold1_idxs]]
                        if train_idx_fold1.size > 0:
                            fold1_train_rate = np.nanmean(freq_maps[train_idx_fold1, 1, :], axis=0)
                        else:
                            fold1_train_rate = np.full(n_neurons, np.nan)
                        if test_idx_fold1.size > 0:
                            fold1_test_rate = np.nanmean(freq_maps[test_idx_fold1, 1, :], axis=0)
                        else:
                            fold1_test_rate = np.full(n_neurons, np.nan)
                        
                        # Save the two folds together.
                        folded_train_maps[cv][i, j, 0, :] = fold0_train_rate
                        folded_train_maps[cv][i, j, 1, :] = fold1_train_rate
                        folded_test_maps[cv][i, j, 0, :] = fold0_test_rate
                        folded_test_maps[cv][i, j, 1, :] = fold1_test_rate
                        
    runtime = (time.time() - start_time) / 60
    print(f"Runtime: {runtime:.2f} minutes")
    return global_train_maps, global_test_maps, folded_train_maps, folded_test_maps, cv_masks

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
    parser = argparse.ArgumentParser(description="Compute CV rate maps and HD maps from spike and position data.")
    parser.add_argument("--dataset", required=True, type=int, help="Dataset number (required).")
    parser.add_argument("--rat", required=True, type=str, help="Rat ID (required).")
    parser.add_argument("--n_hd_bins", default=17, type=int, help="Number of head-direction bins. (default: 17)")
    parser.add_argument("--n_md_bins", default=17, type=int, help="Number of movement-direction bins. (default: 17)")
    parser.add_argument("--n_spatial_bins", default=11, type=int, help="Number of spatial bins in x-y space. (default: 11)")
    parser.add_argument("--smooth_window", required=True, type=int, help="Smoothing window for md calculation (default: 200).")
    parser.add_argument("--vel_threshold", required=False, type=int, help="Velocity threshold (default: 2 cm/s).", default=2)
    parser.add_argument("--shuffle_md", required=False, action='store_true', 
                       help="Shuffle MD for use in downstream decoding analysis.")
    parser.add_argument("--smooth_sigma", type=float, default=1, help="Sigma for use in Gaussian smoothing of rate maps")
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
    
    # For later analysis, compute excitatory but non-HD cells.
    excit_not_hd = (excit_list == 1) & (hd_list == 0)
    excit_not_hd = excit_not_hd.astype(int)
    print('excit_not_hd', np.where(excit_not_hd))
    
    # Load spike data and build spike array.
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
    
    # Load velocity data if available.
    velocity_data = None
    vel_path = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Velocity.mat"
    if os.path.exists(vel_path):
        velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
        velocity_data = extract_matlab_obj(velocity_data['vel'])
        velocity_data = np.dstack((velocity_data['t'], velocity_data['data']))[0]
        mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
        velocity_data = velocity_data[mask]

    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"

    # Compute CV rate maps.
    (global_train_maps, global_test_maps, 
     folded_train_maps, folded_test_maps, cv_masks) = get_rate_maps_cv(
        spike_times, 
        positions, 
        np.where(hd_list == 1)[0],
        n_hd_bins=args.n_hd_bins,
        n_md_bins=args.n_md_bins,
        n_spatial_bins=args.n_spatial_bins,
        smooth_window=args.smooth_window,
        velocity_threshold=args.vel_threshold,
        velocity_data=velocity_data,
        cv_count=4,
        shuffle_md=args.shuffle_md,
         save_path=save_path
    )

    for cv in range(4):
        global_train_maps[cv] = _gaussian_filter(global_train_maps[cv], args.smooth_sigma)
        global_test_maps[cv] = _gaussian_filter(global_test_maps[cv], args.smooth_sigma)
        folded_train_maps[cv] = _gaussian_filter(folded_train_maps[cv], args.smooth_sigma)
        folded_test_maps[cv] = _gaussian_filter(folded_test_maps[cv], args.smooth_sigma)
    
    # Save results.
    os.makedirs(save_path, exist_ok=True)

    suffix = "_shuffled_md" if args.shuffle_md else ""
    
    # Save for each CV split:
    # For each CV split, we save two pairs of files (global and folded, each with train/test),
    # resulting in 4 files per CV split x 4 CV splits = 16 save files.
    for cv in range(4):
        print(f"Saving at {save_path}/hd_md_rate_map_global_train_cv{cv}_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy")
        np.save(f"{save_path}/hd_md_rate_map_global_train_cv{cv}_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", global_train_maps[cv])
        np.save(f"{save_path}/hd_md_rate_map_global_test_cv{cv}_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", global_test_maps[cv])
        np.save(f"{save_path}/hd_md_rate_map_fold_train_cv{cv}_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", folded_train_maps[cv])
        np.save(f"{save_path}/hd_md_rate_map_fold_test_cv{cv}_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", folded_test_maps[cv])
        np.save(f"{save_path}/cv_mask{cv}_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", cv_masks[cv])
    
if __name__ == "__main__":
    main()
