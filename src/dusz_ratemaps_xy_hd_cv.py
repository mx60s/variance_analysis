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

def get_rate_maps_cv(spike_times, positions, excit_list, 
                     n_x_bins=17, n_y_bins=17, n_hd_bins=17, 
                     velocity_threshold=2, velocity_data=None,
                     cv_count=4):
    """
    Compute cross-validation rate maps for head direction (HD) and spatial position (X, Y).
    
    Creates train/test splits using cross-validation for rate maps binned by:
    - Spatial position (X, Y)
    - Head direction (HD)
    
    Returns:
        global_train_maps: List of train rate maps (n_x_bins, n_y_bins, n_hd_bins, n_neurons)
        global_test_maps: List of test rate maps (n_x_bins, n_y_bins, n_hd_bins, n_neurons)
        folded_train_maps: List of folded train rate maps (n_x_bins, n_y_bins, n_hd_bins, 2, n_neurons)
        folded_test_maps: List of folded test rate maps (n_x_bins, n_y_bins, n_hd_bins, 2, n_neurons)
        cv_masks: List of CV masks for each split
    """
    seed = 111
    np.random.seed(seed)
    
    tau = 0.120  # ~8Hz sampling rate
    
    # Extract x, y, and head direction
    x = positions[:, 1]
    y = positions[:, 2]
    hd = positions[:, 3]
    hd = (hd + 360) % 360  # Normalize to [0, 360)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    hd_max = np.max(hd)

    start_time = time.time()
    print(f"Processing {len(excit_list)} excitatory neurons...")

    # Create time bins
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + tau, tau)

    # Separate spike data by excitatory neurons
    neuron_idxs = set(spike_times[:, 1])
    spike_mat = []
    for i in tqdm(neuron_idxs, desc="Processing neurons"):
        if int(i) in excit_list:
            spike_mat.append(spike_times[spike_times[:, 1] == i])
    
    n_neurons = len(spike_mat)
    print(f"Number of neurons: {n_neurons}")

    # Compute firing rate for each time bin and neuron
    freq_maps = np.empty((len(base_time) - 1, 2, n_neurons))
    
    for idx, i in enumerate(tqdm(range(n_neurons), desc="Computing firing rates")):
        current_spikes = spike_mat[idx]
        current_map = np.empty((len(base_time) - 1, 2))
        current_map[:, 0] = base_time[:-1]
        
        for j in range(len(base_time) - 1):
            spike_count = np.sum((current_spikes[:, 0] > base_time[j]) & 
                               (current_spikes[:, 0] <= base_time[j + 1]))
            current_map[j, 1] = spike_count / tau
        
        freq_maps[:, 1, idx] = current_map[:, 1]
    
    print("Finished computing firing rate maps.")

    # Apply velocity threshold if velocity data provided
    if velocity_data is not None:
        print(f"Applying velocity threshold: {velocity_threshold} cm/s")
        avg_vel = np.empty(len(base_time) - 1)
        
        for i in range(len(base_time) - 1):
            mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i + 1])
            if np.any(mask):
                avg_vel[i] = np.nanmean(velocity_data[mask, 1])
            else:
                avg_vel[i] = np.nan

        np.save('/home/mag/variance-analysis/vel_test.npy', avg_vel)
        low_vel_bins = np.where((np.isnan(avg_vel)) | (avg_vel < velocity_threshold))[0]
        ratio = (len(low_vel_bins) / (len(base_time) - 1)) * 100
        print(f"Excluding {ratio:.2f}% of time bins due to low velocity")
        freq_maps[low_vel_bins, 1, :] = np.nan

    # Define bin edges
    x_bins = np.linspace(x_min, x_max, n_x_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_y_bins + 1)
    hd_bins = np.linspace(0, hd_max, n_hd_bins + 1)

    # Create cross-validation masks
    cv_masks = []
    n_points = len(base_time) - 1
    folds = np.array_split(np.arange(n_points), cv_count)
    
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)

    # Initialize output arrays with x, y, hd dimensions
    global_train_maps = [np.full((n_x_bins, n_y_bins, n_hd_bins, n_neurons), np.nan) for _ in range(cv_count)]
    global_test_maps = [np.full((n_x_bins, n_y_bins, n_hd_bins, n_neurons), np.nan) for _ in range(cv_count)]
    folded_train_maps = [np.full((n_x_bins, n_y_bins, n_hd_bins, 2, n_neurons), np.nan) for _ in range(cv_count)]
    folded_test_maps = [np.full((n_x_bins, n_y_bins, n_hd_bins, 2, n_neurons), np.nan) for _ in range(cv_count)]

    print("Creating cross-validation rate maps...")
    
    # Process each X bin
    for i in tqdm(range(len(x_bins) - 1), desc="X bins"):
        # Find position indices in current X bin
        x_indices = np.where((x >= x_bins[i]) & (x < x_bins[i + 1]))[0]
        
        if len(x_indices) > 2:  # Need sufficient samples
            # Process each Y bin
            for j in range(len(y_bins) - 1):
                # Find positions in current Y bin (within X bin)
                y_mask = (y[x_indices] >= y_bins[j]) & (y[x_indices] < y_bins[j + 1])
                
                if np.any(y_mask):
                    xy_indices = x_indices[y_mask]
                    
                    # Map position indices to time bin indices
                    time_indices = np.array([np.abs(base_time - positions[idx, 0]).argmin() 
                                           for idx in xy_indices])
                    
                    # Process each HD bin
                    for k in range(len(hd_bins) - 1):
                        # Select positions within current HD bin
                        hd_mask = (hd[xy_indices] >= hd_bins[k]) & (hd[xy_indices] < hd_bins[k + 1])
                        
                        if np.any(hd_mask):
                            hd_xy_indices = xy_indices[hd_mask]
                            hd_time_indices = np.array([np.abs(base_time - positions[idx, 0]).argmin() 
                                                      for idx in hd_xy_indices])
                            
                            # Shuffle indices for random sampling
                            np.random.shuffle(hd_time_indices)
                            
                            # Split into two folds for folded analysis
                            half = len(hd_time_indices) // 2
                            if len(hd_time_indices) % 2 == 0:
                                fold0_idx = hd_time_indices[:half]
                                fold1_idx = hd_time_indices[half:]
                            else:
                                fold0_idx = hd_time_indices[:half]
                                fold1_idx = hd_time_indices[half:-1]
                            
                            # Process each CV split
                            for cv in range(cv_count):
                                mask_cv = cv_masks[cv]
                                
                                # Global method: use all samples with CV mask
                                train_idx_global = hd_time_indices[mask_cv[hd_time_indices]]
                                test_idx_global = hd_time_indices[~mask_cv[hd_time_indices]]
                                
                                # Calculate global rates
                                if len(train_idx_global) > 0:
                                    global_train_rate = np.nanmean(freq_maps[train_idx_global, 1, :], axis=0)
                                else:
                                    global_train_rate = np.full(n_neurons, np.nan)
                                
                                if len(test_idx_global) > 0:
                                    global_test_rate = np.nanmean(freq_maps[test_idx_global, 1, :], axis=0)
                                else:
                                    global_test_rate = np.full(n_neurons, np.nan)
                                
                                global_train_maps[cv][i, j, k, :] = global_train_rate
                                global_test_maps[cv][i, j, k, :] = global_test_rate
                                
                                # Folded method: process each fold separately
                                for fold_num, fold_idx in enumerate([fold0_idx, fold1_idx]):
                                    train_idx_fold = fold_idx[mask_cv[fold_idx]]
                                    test_idx_fold = fold_idx[~mask_cv[fold_idx]]
                                    
                                    if len(train_idx_fold) > 0:
                                        fold_train_rate = np.nanmean(freq_maps[train_idx_fold, 1, :], axis=0)
                                    else:
                                        fold_train_rate = np.full(n_neurons, np.nan)
                                    
                                    if len(test_idx_fold) > 0:
                                        fold_test_rate = np.nanmean(freq_maps[test_idx_fold, 1, :], axis=0)
                                    else:
                                        fold_test_rate = np.full(n_neurons, np.nan)
                                    
                                    folded_train_maps[cv][i, j, k, fold_num, :] = fold_train_rate
                                    folded_test_maps[cv][i, j, k, fold_num, :] = fold_test_rate

    runtime = (time.time() - start_time) / 60
    print(f"Processing completed in {runtime:.2f} minutes")
    
    return global_train_maps, global_test_maps, folded_train_maps, folded_test_maps, cv_masks

def extract_matlab_obj(obj):
    """
    Recursively convert MATLAB struct objects to Python dictionaries.
    """
    if not (isinstance(obj, mat_struct) or isinstance(obj, dict)):
        return obj

    out = {}
    obj = obj.__dict__ if isinstance(obj, mat_struct) else obj

    for key, val in obj.items():
        if key in ['_fieldnames', 'type', 'time_unit', 'data_unit', 'name']:
            continue
        
        if isinstance(val, (mat_struct, dict)):
            out[key] = extract_matlab_obj(val)
        elif isinstance(val, np.ndarray):
            arr = [extract_matlab_obj(v) for v in val]
            out[key] = np.array(arr)
        else:
            out[key] = val

    return out

def load_data(dataset, rat):
    """
    Load and preprocess all required data files.
    
    Returns:
        positions: Position data array
        spike_times: Spike timing data
        excit_list: List of excitatory neuron indices
        velocity_data: Velocity data (if available)
        epoch_data: Behavioral epoch information
    """
    base_path = current_file_path / f"data/dusz/Dataset_{dataset}/{rat}/Data"
    
    # Load behavioral epochs
    epochs = loadmat(base_path / "BehavEpochs.mat", 
                     struct_as_record=False, squeeze_me=True)
    if dataset == 6:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    else:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    
    # Load and process position data
    tracking_data = pd.read_csv(base_path / "Tracking_data.csv")
    tracking_data = tracking_data.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    positions = tracking_data[['timestamp', 'x', 'y', 'hd']].to_numpy()
    
    # Filter by behavioral epoch
    position_mask = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[position_mask]
    
    # Normalize positions to start from origin
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))
    
    # Load cell type classifications
    cell_types = loadmat(base_path / "CellTypes.mat", 
                         struct_as_record=False, squeeze_me=True)
    hd_list = extract_matlab_obj(cell_types['hd'])
    excit_list = extract_matlab_obj(cell_types['ex'])
    good_list = extract_matlab_obj(cell_types['gd'])
    fs_list = extract_matlab_obj(cell_types['fs'])
    
    print(f"HD cells: {np.where(hd_list)[0]}")
    print(f"Excitatory cells: {np.where(excit_list)[0]}")
    print(f"Good cells: {np.where(good_list)[0]}")
    print(f"Fast-spiking cells: {np.where(fs_list)[0]}")
    
    # Load and process spike data
    spike_data = loadmat(base_path / "SpikeData.mat", 
                         struct_as_record=False, squeeze_me=True)
    spike_data = extract_matlab_obj(spike_data['S'])
    
    # Build spike times array
    spike_times_list = []
    for cell_idx, cell_data in enumerate(spike_data['C']):
        spike_times_cell = cell_data['tsd']['t']
        for spike_time in spike_times_cell:
            spike_times_list.append([spike_time, cell_idx])
    
    spike_times = np.array(spike_times_list)
    spike_times = spike_times[np.argsort(spike_times[:, 0])]  # Sort by time
    
    # Filter by behavioral epoch
    spike_mask = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[spike_mask]
    
    # Load velocity data if available
    velocity_data = None
    vel_path = base_path / "Velocity.mat"
    if os.path.exists(vel_path):
        velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
        velocity_data = extract_matlab_obj(velocity_data['vel'])
        velocity_data = np.column_stack((velocity_data['t'], velocity_data['data']))
        
        # Filter by behavioral epoch
        mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
        velocity_data = velocity_data[mask]
    
    return positions, spike_times, np.where(hd_list == 1)[0], velocity_data, epoch_data

def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-validation HD-XY rate maps from neural spike and position data."
    )
    parser.add_argument("--dataset", required=True, type=int, 
                       help="Dataset number")
    parser.add_argument("--rat", required=True, type=str, 
                       help="Rat identifier")
    parser.add_argument("--n_x_bins", default=17, type=int, 
                       help="Number of X spatial bins (default: 17)")
    parser.add_argument("--n_y_bins", default=17, type=int, 
                       help="Number of Y spatial bins (default: 17)")
    parser.add_argument("--n_hd_bins", default=17, type=int, 
                       help="Number of head-direction bins (default: 17)")
    parser.add_argument("--vel_threshold", default=2, type=int, 
                       help="Velocity threshold in cm/s (default: 2)")
    parser.add_argument("--cv_count", default=4, type=int,
                       help="Number of cross-validation folds (default: 4)")
    
    args = parser.parse_args()
    
    print(f"Processing Dataset {args.dataset}, Rat {args.rat}")
    print(f"X bins: {args.n_x_bins}, Y bins: {args.n_y_bins}, HD bins: {args.n_hd_bins}")
    print(f"Velocity threshold: {args.vel_threshold} cm/s")
    print(f"CV folds: {args.cv_count}")
    
    # Load all data
    positions, spike_times, excit_list, velocity_data, epoch_data = load_data(args.dataset, args.rat)
    
    # Compute cross-validation rate maps
    (global_train_maps, global_test_maps, 
     folded_train_maps, folded_test_maps, cv_masks) = get_rate_maps_cv(
        spike_times=spike_times,
        positions=positions,
        excit_list=excit_list,
        n_x_bins=args.n_x_bins,
        n_y_bins=args.n_y_bins,
        n_hd_bins=args.n_hd_bins,
        velocity_threshold=args.vel_threshold,
        velocity_data=velocity_data,
        cv_count=args.cv_count
    )

    # Create output directory
    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"
    os.makedirs(save_path, exist_ok=True)
    
    # Save results for each CV fold
    print(f"Saving results to {save_path}")
    for cv in range(args.cv_count):
        base_name = f"x_y_hd_rate_map_cv{cv}_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}"
        
        np.save(f"{save_path}/{base_name}_global_train.npy", global_train_maps[cv])
        np.save(f"{save_path}/{base_name}_global_test.npy", global_test_maps[cv])
        np.save(f"{save_path}/{base_name}_fold_train.npy", folded_train_maps[cv])
        np.save(f"{save_path}/{base_name}_fold_test.npy", folded_test_maps[cv])
        np.save(f"{save_path}/cv_mask{cv}_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}.npy", cv_masks[cv])
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()