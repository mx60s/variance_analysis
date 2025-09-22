# rate map script:

import argparse
import os
import numpy as np
from scipy.io import loadmat, savemat
import scipy.signal as sci
import time
from tqdm import tqdm
from pathlib import Path

def match_times(time_list, search_time):
    """Find the closest time match in time_list."""
    times = time_list[:, 0]
    array_row = np.searchsorted(times, search_time, side="left")
    if array_row > 0 and (array_row == len(times) or 
                          abs(search_time - times[array_row - 1]) < abs(search_time - times[array_row])):
        return time_list[array_row - 1, 1]
    else:
        return time_list[array_row, 1]

def compute_theta_phase(lfp):
    """Compute theta phase from LFP data."""
    sample_rate = 1 / np.diff(lfp[:, 0])[0]
    
    # Bandpass filter for theta (6-12 Hz)
    b, a = sci.butter(2, [6, 12], btype='bandpass', fs=sample_rate)
    filt_band = sci.lfilter(b, a, lfp[:, 1])
    
    # Find peaks and troughs
    peaks = sci.find_peaks(-filt_band, distance=50)[0]
    troughs = sci.find_peaks(filt_band, distance=50)[0]
    
    # Find zero crossings
    crossings = np.where(np.diff(np.sign(filt_band)) != 0)[0] + 1
    crossings = np.delete(crossings, np.where(crossings < peaks[0])[0][1:])
    crossings = np.delete(crossings, np.where((crossings > peaks[0]) & (crossings < troughs[0]))[0][1:])
    
    if peaks[0] > troughs[0]:
        cross_270 = crossings[::2].copy()
        cross_90 = crossings[1::2].copy()
    else:
        cross_270 = crossings[1::2].copy()
        cross_90 = crossings[::2].copy()
    
    # Prepare phase extrema
    peaks = np.c_[peaks, np.zeros(np.shape(peaks))]
    peaks[::2, 1] = 360
    troughs = np.c_[troughs, np.ones(np.shape(troughs)) * 180]
    cross_90 = np.c_[cross_90, np.ones(np.shape(cross_90)) * 90]
    cross_270 = np.c_[cross_270, np.ones(np.shape(cross_270)) * 270]
    
    phase_extrema = np.r_[peaks, cross_90, troughs, cross_270]
    sort_extrema = np.argsort(phase_extrema[:, 0])
    phase_extrema = phase_extrema[sort_extrema]
    
    # Interpolate phase
    mis_interpolate = []
    for i in range(len(phase_extrema[:, 0]) - 1):
        if phase_extrema[i + 1, 1] == 0:
            interp_line = np.linspace(phase_extrema[i, 1], 360, 
                                    int(np.abs(phase_extrema[i + 1, 0] - phase_extrema[i, 0])))
        elif phase_extrema[i, 1] == 360:
            interp_line = np.linspace(0, phase_extrema[i + 1, 1], 
                                    int(np.abs(phase_extrema[i + 1, 0] - phase_extrema[i, 0])))
        else:
            interp_line = np.linspace(phase_extrema[i, 1], phase_extrema[i + 1, 1], 
                                    int(np.abs(phase_extrema[i + 1, 0] - phase_extrema[i, 0])))
        mis_interpolate.append(interp_line)
    
    interpolated_phase = [item for sublist in mis_interpolate for item in sublist]
    pad_end = np.linspace(phase_extrema[-1, 1], False, int(len(filt_band) - phase_extrema[-1, 0]))
    pad_start = np.ones(int(phase_extrema[0, 0])) * phase_extrema[0, 1]
    interpolated_phase = np.r_[interpolated_phase, pad_end]
    interpolated_phase = np.r_[pad_start, interpolated_phase]
    
    return np.asarray([lfp[:, 0], interpolated_phase]).T

def compute_velocity(locations, base_time):
    """Compute velocity from position data."""
    time_step_gap = locations[3, 0] - locations[2, 0]
    point_steps = np.diff(locations[:, 1:3], axis=0)
    point_steps = np.vstack((point_steps, point_steps[-1]))
    point_dist = np.linalg.norm(point_steps, axis=1) / time_step_gap
    
    vel_list = np.zeros((np.shape(base_time)))
    for t in tqdm(range(len(base_time) - 1), desc="Computing velocity"):
        spikes_in_window = np.where((locations[:, 0] < base_time[t + 1]) & 
                                   (locations[:, 0] >= base_time[t]))[0]
        if np.size(spikes_in_window) == 0:
            dister = 0
        else:
            dister = np.mean(point_dist[spikes_in_window])
        vel_list[t] = dister
    
    return vel_list

def get_rate_maps_cv(spike_mat, locations, base_time, freq_maps,
                     env_len=200, n_bins=17, n_hd_bins=17,
                     velocity_threshold=5, filter_vel=True,
                     cv_count=4, seed=777):
    """
    Compute cross-validation rate maps for spatial position and head direction.
    
    Returns:
        global_train_maps: List of train rate maps (n_bins, n_bins, n_neurons, n_hd_bins)
        global_test_maps: List of test rate maps (n_bins, n_bins, n_neurons, n_hd_bins)
        folded_train_maps: List of folded train rate maps (n_bins, n_bins, n_neurons, n_hd_bins, 2)
        folded_test_maps: List of folded test rate maps (n_bins, n_bins, n_neurons, n_hd_bins, 2)
        occupancy_maps: List of occupancy maps for each CV fold
        cv_masks: List of CV masks for each split
    """
    np.random.seed(seed)
    
    tau = 0.120
    timebin_tol = 1e-4
    n_neurons = len(spike_mat)
    
    # Compute velocity and filter time bins
    vel_list = compute_velocity(locations, base_time)
    fast_indices = np.where(vel_list >= velocity_threshold)[0]
    fast_bins = base_time[fast_indices] if filter_vel else base_time
    #np.save('/home/mag/variance-analysis/test_vel.npy', vel_list[fast_indices])
    
    print(f"Filtering {100 - len(fast_indices)/len(base_time)*100:.1f}% of bins due to low velocity")
    
    # Define bin edges
    spatial_bins = np.linspace(0, env_len, n_bins + 1)
    hd_bins = np.linspace(0, 360.01, n_hd_bins + 1)
    
    # Create cross-validation masks
    cv_masks = []
    n_points = len(base_time) - 1
    folds = np.array_split(np.arange(n_points), cv_count) 
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)
    
    # Initialize output arrays - Note: dimensions ordered as (x, y, hd, neurons) for compatibility
    global_train_maps = [np.full((n_bins, n_bins, n_hd_bins, n_neurons), np.nan) 
                        for _ in range(cv_count)]
    global_test_maps = [np.full((n_bins, n_bins, n_hd_bins, n_neurons), np.nan) 
                       for _ in range(cv_count)]
    folded_train_maps = [np.full((n_bins, n_bins, n_hd_bins, 2, n_neurons), np.nan) 
                        for _ in range(cv_count)]
    folded_test_maps = [np.full((n_bins, n_bins, n_hd_bins, 2, n_neurons), np.nan) 
                       for _ in range(cv_count)]
    occupancy_maps = [np.zeros((n_bins, n_bins, n_hd_bins)) for _ in range(cv_count)]
    
    print("Creating cross-validation rate maps...")
    
    # Process each spatial bin
    for i in tqdm(range(len(spatial_bins) - 1), desc="Processing spatial bins"):
        for j in range(len(spatial_bins) - 1):
            # Find positions in current spatial bin
            freq_locs = np.where((locations[:, 1] < spatial_bins[i + 1]) & 
                                (locations[:, 1] >= spatial_bins[i]) &
                                (locations[:, 2] < spatial_bins[j + 1]) & 
                                (locations[:, 2] >= spatial_bins[j]))[0]
            
            if freq_locs.size > 2:
                # Map to time bins and apply velocity filter
                freq_locers = np.array([np.abs(base_time - locations[freq_locs[k], 0]).argmin() 
                                      for k in range(len(freq_locs))])
                mask = np.any(np.isclose(base_time[freq_locers, None], fast_bins, atol=timebin_tol), axis=1)
                freq_locers = freq_locers[mask]
                
                # Process each HD bin
                for l in range(len(hd_bins) - 1):
                    hd_freq_locs = np.where((locations[freq_locs, 3] >= hd_bins[l]) & 
                                          (locations[freq_locs, 3] < hd_bins[l + 1]))[0]
                    hd_freq_locs = freq_locs[hd_freq_locs]
                    
                    if hd_freq_locs.size > 0:
                        # Map HD positions to time bins
                        hd_freq_locers = np.array([np.abs(base_time - locations[hd_freq_locs[m], 0]).argmin() 
                                                 for m in range(len(hd_freq_locs))])
                        mask = np.any(np.isclose(base_time[hd_freq_locers, None], fast_bins, atol=timebin_tol), axis=1)
                        hd_freq_locers = hd_freq_locers[mask]
                        
                        if hd_freq_locers.size > 0:
                            # Shuffle for random sampling
                            np.random.shuffle(hd_freq_locers)
                            
                            # Split into two folds
                            half = len(hd_freq_locers) // 2
                            fold0_idx = hd_freq_locers[:half] if half > 0 else []
                            fold1_idx = hd_freq_locers[half:] if half > 0 else []
                            
                            # Process each CV split
                            for cv in range(cv_count):
                                mask_cv = cv_masks[cv]
                                
                                # Global method: use all samples with CV mask
                                valid_indices = hd_freq_locers[hd_freq_locers < len(mask_cv)]
                                train_idx_global = valid_indices[mask_cv[valid_indices]]
                                test_idx_global = valid_indices[~mask_cv[valid_indices]]
                                
                                # Calculate global rates - store with dimensions (x, y, hd, neurons)
                                if len(train_idx_global) > 0:
                                    global_train_maps[cv][i, j, l, :] = np.nanmean(freq_maps[train_idx_global, 1, :], axis=0)
                                
                                if len(test_idx_global) > 0:
                                    global_test_maps[cv][i, j, l, :] = np.nanmean(freq_maps[test_idx_global, 1, :], axis=0)
                                
                                occupancy_maps[cv][i, j, l] = len(hd_freq_locs)
                                
                                # Folded method: process each fold separately
                                # this fold is for EV calculations, the other is for cross validation
                                for fold_num, fold_idx in enumerate([fold0_idx, fold1_idx]):
                                    if len(fold_idx) > 0:
                                        valid_fold = fold_idx[fold_idx < len(mask_cv)]
                                        train_idx_fold = valid_fold[mask_cv[valid_fold]]
                                        test_idx_fold = valid_fold[~mask_cv[valid_fold]]
                                        
                                        if len(train_idx_fold) > 0:
                                            folded_train_maps[cv][i, j, l, fold_num, :] = np.nanmean(
                                                freq_maps[train_idx_fold, 1, :], axis=0)
                                        
                                        if len(test_idx_fold) > 0:
                                            folded_test_maps[cv][i, j, l, fold_num, :] = np.nanmean(
                                                freq_maps[test_idx_fold, 1, :], axis=0)
    
    return (global_train_maps, global_test_maps, 
            folded_train_maps, folded_test_maps, 
            occupancy_maps, cv_masks)

def smooth_rate_maps(rate_maps, env_len=200):
    """Apply Gaussian smoothing to rate maps."""
    n_bins = rate_maps.shape[0]
    sigma = 6
    
    spatial_bins = np.linspace(0, env_len, n_bins + 1)
    filter_range = spatial_bins - n_bins
    filter_x, filter_y = np.meshgrid(filter_range, filter_range)
    smooth_maps = np.zeros(rate_maps.shape)
    
    for i in range(len(spatial_bins)):
        for j in range(len(spatial_bins)):
            gauss_filter = (np.exp(-(((filter_x - spatial_bins[j]) + n_bins) ** 2 + 
                                    ((filter_y - spatial_bins[i]) + n_bins) ** 2) / 
                                  (2 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
            
            if len(rate_maps.shape) == 4:  # Has HD and neuron dimensions (x, y, hd, neurons)
                for l in range(rate_maps.shape[2]):  # HD bins
                    smooth_maps[i, j, l, :] = [np.nansum(gauss_filter * rate_maps[:, :, l, n]) 
                                              for n in range(rate_maps.shape[3])]
            else:
                smooth_maps[i, j, :] = [np.nansum(gauss_filter * rate_maps[:, :, n]) 
                                       for n in range(rate_maps.shape[2])]
    
    # Adjust magnitude
    with np.errstate(divide='ignore', invalid='ignore'):
        if len(rate_maps.shape) == 4:
            for l in range(rate_maps.shape[2]):  # HD bins
                max_raw = np.nanmax(rate_maps[:, :, l, :], axis=(0, 1))
                max_smooth = np.nanmax(smooth_maps[:, :, l, :], axis=(0, 1))
                scale = np.where(max_smooth > 0, max_raw / max_smooth, 1)
                smooth_maps[:, :, l, :] *= scale
        else:
            max_raw = np.nanmax(rate_maps, axis=(0, 1))
            max_smooth = np.nanmax(smooth_maps, axis=(0, 1))
            scale = np.where(max_smooth > 0, max_raw / max_smooth, 1)
            smooth_maps *= scale[np.newaxis, np.newaxis, :]
    
    return smooth_maps

def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-validation rate maps from Pfeiffer data."
    )
    parser.add_argument("--data_path", default="/home/mag/variance-analysis/data/pfeiffer", 
                       type=str, help="Path to data directory")
    parser.add_argument('--rat', default='1', type=str, help='Which rat')
    parser.add_argument("--save_path", default="/home/mag/variance-analysis/results/pfeiffer", 
                       type=str, help="Path to save results")
    parser.add_argument("--n_bins", default=17, type=int, 
                       help="Number of spatial bins (default: 17)")
    parser.add_argument("--n_hd_bins", default=17, type=int, 
                       help="Number of head-direction bins (default: 17)")
    parser.add_argument("--vel_threshold", default=5, type=float, 
                       help="Velocity threshold in cm/s (default: 5)")
    parser.add_argument("--cv_count", default=4, type=int,
                       help="Number of cross-validation folds (default: 4)")
    parser.add_argument("--filter_vel", action='store_true', default=True,
                       help="Filter low velocity bins (default: True)")
    parser.add_argument("--seed", default=777, type=int,
                       help="Random seed (default: 777)")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)

    data_path = args.data_path + f'/Rat{args.rat}/Open1/'
    save_path = args.save_path + f'/Rat{args.rat}'
    
    print(f"Processing data from: {data_path}")
    print(f"Spatial bins: {args.n_bins}, HD bins: {args.n_hd_bins}")
    print(f"Velocity threshold: {args.vel_threshold} cm/s")
    print(f"CV folds: {args.cv_count}, Seed: {args.seed}")
    
    start_time = time.time()
    
    # Load data
    print("Loading data...")
    locations = loadmat(f'{data_path}/Position_Data.mat')['Position_Data']
    spike_data = loadmat(f'{data_path}/Spike_Data.mat')
    spike_times = spike_data['Spike_Data']
    excit_list = spike_data['Excitatory_Neurons']
    lfp = loadmat(f'{data_path}/LFP_Data.mat')['LFP_Data']
    
    # Parameters
    tau = 0.120
    interpolation_points = 1
    env_len = 200
    
    # Compute theta phase
    print("Computing theta phase...")
    theta_time = compute_theta_phase(lfp)
    
    # Add theta phase to spike times
    position_sample = np.diff(locations[:, 0])[1]
    #dec_tau = position_sample / interpolation_points # consider removing this entirely and using tau
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + tau, tau)
    
    stapler = [match_times(theta_time, spike_times[k, 0]) for k in range(len(spike_times[:, 0]))]
    spike_times = np.c_[spike_times, stapler]
    
    # Extract excitatory neurons
    spike_mat = []
    for neuron_id in excit_list:
        neuron_spikes = spike_times[spike_times[:, 1] == neuron_id]
        spike_mat.append(neuron_spikes) 
    
    n_neurons = len(spike_mat)
    print(f"Found {n_neurons} excitatory neurons")
    
    # Compute firing rate maps
    print("Computing firing rate maps...")
    freq_maps = np.empty((len(base_time) - 1, 2, n_neurons))
    
    for i in tqdm(range(n_neurons), desc="Processing neurons"):
        current_map = np.vstack((base_time[:-1], np.empty(len(base_time) - 1))).T
        for j in range(len(base_time) - 1):
            running_freq = len(np.where((spike_mat[i][:, 0] <= base_time[j + 1]) &
                                       (spike_mat[i][:, 0] > base_time[j]))[0]) / tau
            current_map[j, 1] = running_freq
        freq_maps[:, :, i] = current_map
    
    # Compute cross-validation rate maps
    print("Computing cross-validation rate maps...")
    (global_train_maps, global_test_maps, 
     folded_train_maps, folded_test_maps, 
     occupancy_maps, cv_masks) = get_rate_maps_cv(
        spike_mat=spike_mat,
        locations=locations,
        base_time=base_time,
        freq_maps=freq_maps,
        env_len=env_len,
        n_bins=args.n_bins,
        n_hd_bins=args.n_hd_bins,
        velocity_threshold=args.vel_threshold,
        filter_vel=args.filter_vel,
        cv_count=args.cv_count,
        seed=args.seed
    )
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)

    np.save(f'{save_path}/freq_maps_for_rate_maps.npy', freq_maps)
    
    # Save results for each CV fold
    print(f"Saving results to {save_path}...")
    for cv in range(args.cv_count):
        # Use naming convention compatible with decoder script
        base_name = f"x_y_hd_rate_map_cv{cv}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}"
        
        # Save raw rate maps
        np.save(f"{save_path}/{base_name}_global_train.npy", global_train_maps[cv])
        np.save(f"{save_path}/{base_name}_global_test.npy", global_test_maps[cv])
        np.save(f"{save_path}/{base_name}_fold_train.npy", folded_train_maps[cv])
        np.save(f"{save_path}/{base_name}_fold_test.npy", folded_test_maps[cv])
        
        # Save smoothed rate maps
        #np.save(f"{save_path}/smooth_{base_name}_global_train.npy", smooth_global_train_maps[cv])
        #np.save(f"{save_path}/smooth_{base_name}_global_test.npy", smooth_global_test_maps[cv])
        
        # Save occupancy and masks with compatible naming
        np.save(f"{save_path}/hd_occupancy_map_cv{cv}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}.npy", 
                occupancy_maps[cv])
        np.save(f"{save_path}/cv_mask{cv}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}.npy", cv_masks[cv])
    
    # Save general data
    np.save(f'{save_path}/locations.npy', locations)
    np.save(f'{save_path}/time_bins.npy', base_time)
    
    runtime = (time.time() - start_time) / 60
    print(f"Processing completed in {runtime:.2f} minutes")

if __name__ == "__main__":
    main()

# decoding script: 

import argparse
import os
import numpy as np
from scipy.io import loadmat
import time
from tqdm import tqdm
from pathlib import Path

def bin_spike_counts(spike_times, base_time, excit_list):
    """
    Bin spike times into counts for each excitatory neuron and time bin.
    
    Parameters:
        spike_times: (N x 3) array with columns [timestamp, neuron_id, theta_phase]
        base_time: 1D array of time bin edges
        excit_list: List of excitatory neuron indices
    
    Returns:
        spike_counts: 2D array of shape (n_time_bins, n_excit_neurons)
    """
    n_bins = len(base_time) - 1
    n_neurons = len(excit_list)
    spike_counts = np.zeros((n_bins, n_neurons))
    
    # Map excitatory neuron IDs to indices
    neuron_map = {neuron_id: idx for idx, neuron_id in enumerate(excit_list)}
    
    # Use numpy histogram for efficient binning
    for idx, neuron_id in enumerate(excit_list):
        neuron_spikes = spike_times[spike_times[:, 1] == neuron_id, 0]
        if len(neuron_spikes) > 0:
            counts, _ = np.histogram(neuron_spikes, bins=base_time)
            spike_counts[:, idx] = counts
    
    return spike_counts

def joint_bayesian_decoding(rate_map, spike_counts, tau, x_bins, y_bins, hd_bins, 
                           avg_vel, vel_threshold, eps=1e-10):
    """
    Joint Bayesian decoding of spatial position (x, y) and head direction using Poisson likelihood.
    
    Parameters:
        rate_map: 4D array (n_x_bins, n_y_bins, n_hd_bins, n_neurons) - tuning curves
        spike_counts: 2D array (n_time_bins, n_neurons) - observed spike counts
        tau: time bin size in seconds
        x_bins: x position bin edges
        y_bins: y position bin edges  
        hd_bins: head direction bin edges
        avg_vel: 1D array of average velocity per time bin
        vel_threshold: minimum velocity threshold for decoding
        eps: small constant to avoid log(0)
    
    Returns:
        decoded_x: decoded x position (bin centers) for each time bin
        decoded_y: decoded y position (bin centers) for each time bin
        decoded_hd: decoded head direction (bin centers) for each time bin
        posterior_all: list of posterior probability arrays (one per time bin)
    """
    n_time_bins = spike_counts.shape[0]
    n_x_bins = rate_map.shape[0]
    n_y_bins = rate_map.shape[1]
    n_hd_bins = rate_map.shape[2]
    
    decoded_x = np.full(n_time_bins, np.nan)
    decoded_y = np.full(n_time_bins, np.nan)
    decoded_hd = np.full(n_time_bins, np.nan)
    posterior_all = []
    
    print(f"Decoding {n_time_bins} time bins using {rate_map.shape[3]} neurons...")
    
    # Process each time bin
    for t in tqdm(range(n_time_bins), desc="Decoding time bins"):
        # Skip if velocity is too low or undefined
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            decoded_x[t] = np.nan
            decoded_y[t] = np.nan
            decoded_hd[t] = np.nan
            posterior_all.append(np.full((n_x_bins, n_y_bins, n_hd_bins), np.nan))
            continue
        
        # Poisson log-likelihood: k*log(λ*τ) - λ*τ
        spike_data = spike_counts[t][None, None, None, :]  # Shape: (1, 1, 1, n_neurons)
        rate_data = rate_map * tau  # Expected spike count
        
        # Compute log-likelihood for each (x, y, hd) bin
        with np.errstate(divide='ignore', invalid='ignore'):
            log_likelihood = np.nansum(
                spike_data * np.log(rate_data + eps) - rate_data, 
                axis=3
            )
        
        # Skip if all likelihoods are NaN
        if np.all(np.isnan(log_likelihood)):
            decoded_x[t] = np.nan
            decoded_y[t] = np.nan
            decoded_hd[t] = np.nan
            posterior_all.append(np.full((n_x_bins, n_y_bins, n_hd_bins), np.nan))
            continue
        
        # Convert to posterior probabilities (normalize for numerical stability)
        max_log_likelihood = np.nanmax(log_likelihood)
        posterior = np.exp(log_likelihood - max_log_likelihood)
        posterior = posterior / np.nansum(posterior)
        posterior_all.append(posterior)
        
        # Find bin with maximum posterior probability
        x_idx, y_idx, hd_idx = np.unravel_index(np.nanargmax(posterior), posterior.shape)
        
        # Convert bin indices to bin centers
        decoded_x[t] = (x_bins[x_idx] + x_bins[x_idx + 1]) / 2
        decoded_y[t] = (y_bins[y_idx] + y_bins[y_idx + 1]) / 2
        decoded_hd[t] = (hd_bins[hd_idx] + hd_bins[hd_idx + 1]) / 2
    
    return decoded_x, decoded_y, decoded_hd, posterior_all

def compute_velocity(locations, base_time):
    """Compute velocity from position data for each time bin."""
    time_step_gap = locations[3, 0] - locations[2, 0]
    point_steps = np.diff(locations[:, 1:3], axis=0)
    point_steps = np.vstack((point_steps, point_steps[-1]))
    point_dist = np.linalg.norm(point_steps, axis=1) / time_step_gap
    
    vel_list = np.zeros(len(base_time) - 1)
    for t in range(len(base_time) - 1):
        pos_in_window = np.where((locations[:, 0] >= base_time[t]) & 
                                (locations[:, 0] < base_time[t + 1]))[0]
        if np.size(pos_in_window) == 0:
            vel_list[t] = 0
        else:
            vel_list[t] = np.mean(point_dist[pos_in_window])
    
    return vel_list

def load_pfeiffer_data(data_path):
    """
    Load Pfeiffer format data for decoding.
    
    Returns:
        spike_times: Spike timing data with theta phase
        locations: Position data array
        excit_list: List of excitatory neuron indices
    """
    print("Loading Pfeiffer data...")
    
    # Load position data
    locations = loadmat(f'{data_path}/Position_Data.mat')['Position_Data']
    
    # Load spike data
    spike_data = loadmat(f'{data_path}/Spike_Data.mat')
    spike_times = spike_data['Spike_Data']
    excit_list = spike_data['Excitatory_Neurons'].flatten()
    
    print(f"Loaded {len(spike_times)} spikes from {len(excit_list)} excitatory neurons")
    print(f"Position data shape: {locations.shape}")
    
    return spike_times, locations, excit_list

def main():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian decoding for Pfeiffer data format."
    )
    parser.add_argument("--data_path", default="/home/mag/variance-analysis/data/pfeiffer", 
                       type=str, help="Path to data directory")
    parser.add_argument('--rat', default='1', type=str, help='Which rat')
    parser.add_argument("--save_path", default="/home/mag/variance-analysis/results/pfeiffer", 
                       type=str, help="Path to save results")
    parser.add_argument("--n_bins", default=17, type=int, 
                       help="Number of spatial bins (default: 17)")
    parser.add_argument("--n_hd_bins", default=17, type=int, 
                       help="Number of head-direction bins (default: 17)")
    parser.add_argument("--tau", default=0.120, type=float, 
                       help="Time bin size in seconds (default: 0.120)")
    parser.add_argument("--vel_threshold", default=5, type=float, 
                       help="Velocity threshold in cm/s (default: 5)")
    parser.add_argument("--cv_index", default=-1, type=int, 
                       help="Cross-validation fold index (-1 for no CV)")
    
    args = parser.parse_args()

    # gotta make this a path
    data_path = args.data_path + f'/Rat{args.rat}/Open1'
    save_path = args.save_path + f'/Rat{args.rat}'
    
    print(f"Joint X-Y-HD Bayesian Decoding for Pfeiffer Data")
    print(f"Data path: {data_path}")
    print(f"Spatial bins: {args.n_bins}, HD bins: {args.n_hd_bins}")
    print(f"Time bin size: {args.tau}s, Velocity threshold: {args.vel_threshold} cm/s")
    if args.cv_index >= 0:
        print(f"Using cross-validation fold: {args.cv_index}")
    
    # Load precomputed rate maps
    print("Loading rate maps...")
    if args.cv_index < 0:
        rate_map_file = f"{save_path}/x_y_hd_rate_map_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}.npy"
    else:
        rate_map_file = f"{save_path}/x_y_hd_rate_map_cv{args.cv_index}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}_global_train.npy"
    
    if not os.path.exists(rate_map_file):
        print(f"Error: Rate map file not found at {rate_map_file}")
        print("Please run the rate map analysis script first.")
        return
    
    rate_map = np.load(rate_map_file)  # shape: (n_x_bins, n_y_bins, n_hd_bins, n_neurons)
    print(f"Loaded rate map with shape: {rate_map.shape}")
    
    # Load Pfeiffer data
    spike_times, locations, excit_list = load_pfeiffer_data(data_path)
    
    # Create time bins for spike counting (matching the rate map script)
    interpolation_points = 1
    position_sample = np.diff(locations[:, 0])[1]
    #dec_tau = position_sample / interpolation_points
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0]+ args.tau, args.tau)
    
    # Bin spike counts for excitatory neurons only
    spike_counts = bin_spike_counts(spike_times, base_time, excit_list)
    print(f"Spike counts shape: {spike_counts.shape}")

    freq_maps_original = np.load(f'{save_path}/freq_maps_for_rate_maps.npy')
    base_time_original = np.load(f'{save_path}/time_bins.npy')
    
    # Extract spike counts from freq_maps (converting rates back to counts)
    spike_counts = freq_maps_original[:, 1, :] * args.tau  # rates * tau = counts
    
    print(f"Using spike counts from rate map generation")
    print(f"Shape: {spike_counts.shape}")
    
    # Extract spatial features
    x = locations[:, 1]
    y = locations[:, 2]
    hd = locations[:, 3]  # Already in degrees for Pfeiffer data
    
    # Create bin edges matching those used in rate map creation
    env_len = 200  # Environment length for Pfeiffer data
    x_bins = np.linspace(0, env_len, args.n_bins + 1)
    y_bins = np.linspace(0, env_len, args.n_bins + 1)
    hd_bins = np.linspace(0, 360.01, args.n_hd_bins + 1)
    
    # Compute velocity for each time bin
    print("Computing velocity for each time bin...")
    avg_vel = compute_velocity(locations, base_time)

    print(f"First 5 neurons in rate map - mean rates: {np.nanmean(rate_map, axis=(0,1,2))[:5]}")
    print(f"First 5 neurons in spike_counts - mean rates: {spike_counts.mean(axis=0)[:5] / args.tau}")
    
    # Apply cross-validation filtering if specified
    if args.cv_index >= 0:
        print(f"Applying CV fold {args.cv_index} filtering...")
        cv_mask_file = f"{save_path}/cv_mask{args.cv_index}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}.npy"
        
        if not os.path.exists(cv_mask_file):
            print(f"Error: CV mask file not found at {cv_mask_file}")
            return
        
        cv_mask = np.load(cv_mask_file)
        
        # Ensure mask length matches data
        if len(cv_mask) > len(spike_counts):
            cv_mask = cv_mask[:len(spike_counts)]
        elif len(cv_mask) < len(spike_counts):
            print(f"Warning: CV mask shorter than data, padding with False")
            cv_mask = np.concatenate([cv_mask, np.zeros(len(spike_counts) - len(cv_mask), dtype=bool)])
        
        test_indices = np.where(cv_mask == False)[0]
        
        # Filter to test data only
        spike_counts = spike_counts[test_indices, :]
        avg_vel = avg_vel[test_indices]
        
        # Calculate time centers for test bins
        time_centers = (base_time[test_indices] + base_time[np.minimum(test_indices + 1, len(base_time) - 1)]) / 2
        
        print(f"Decoding on {len(test_indices)} test time bins")
    else:
        # Calculate time centers for all bins
        time_centers = (base_time[:-1] + base_time[1:]) / 2
        if len(time_centers) > len(spike_counts):
            time_centers = time_centers[:len(spike_counts)]
    
    # Perform joint Bayesian decoding
    print("Starting joint Bayesian decoding...")
    start_time = time.time()
    
    decoded_x, decoded_y, decoded_hd, posterior_all = joint_bayesian_decoding(
        rate_map=rate_map,
        spike_counts=spike_counts,
        tau=args.tau,
        x_bins=x_bins,
        y_bins=y_bins,
        hd_bins=hd_bins,
        avg_vel=avg_vel,
        vel_threshold=args.vel_threshold
    )
    
    runtime = time.time() - start_time
    print(f"Decoding completed in {runtime:.2f} seconds")
    
    # Save results
    print(f"Saving results to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    if args.cv_index >= 0:
        # Cross-validation results
        base_name = f"decoded_cv{args.cv_index}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}"
        np.save(f"{save_path}/{base_name}_x.npy", decoded_x)
        np.save(f"{save_path}/{base_name}_y.npy", decoded_y)
        np.save(f"{save_path}/{base_name}_hd.npy", decoded_hd)
        np.save(f"{save_path}/{base_name}_posterior.npy", np.array(posterior_all))
        np.save(f"{save_path}/{base_name}_time_centers.npy", time_centers)
    else:
        # Full dataset results
        base_name = f"decoded_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}"
        np.save(f"{save_path}/{base_name}_x.npy", decoded_x)
        np.save(f"{save_path}/{base_name}_y.npy", decoded_y)
        np.save(f"{save_path}/{base_name}_hd.npy", decoded_hd)
        np.save(f"{save_path}/{base_name}_posterior.npy", np.array(posterior_all))
        np.save(f"{save_path}/{base_name}_time_centers.npy", time_centers)
    
    # Print summary statistics
    valid_x = decoded_x[~np.isnan(decoded_x)]
    valid_y = decoded_y[~np.isnan(decoded_y)]
    valid_hd = decoded_hd[~np.isnan(decoded_hd)]
    
    print(f"\nDecoding Summary:")
    print(f"Total time bins: {len(decoded_hd)}")
    print(f"Successfully decoded: {len(valid_hd)} ({100*len(valid_hd)/len(decoded_hd):.1f}%)")
    if len(valid_x) > 0:
        print(f"X position range: {np.min(valid_x):.2f} - {np.max(valid_x):.2f} cm")
        print(f"Y position range: {np.min(valid_y):.2f} - {np.max(valid_y):.2f} cm")
        print(f"Head direction range: {np.min(valid_hd):.1f} - {np.max(valid_hd):.1f} degrees")
    print("Decoding analysis completed successfully!")

if __name__ == "__main__":
    main()