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

def load_pfeiffer_data(data_path, save_path):
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
    spike_times = np.load(f'{save_path}/spike_times.npy') #spike_data['Spike_Data']
    excit_list = spike_data['Excitatory_Neurons'].flatten()
    
    print(f"Loaded {len(spike_times)} spikes from {len(excit_list)} excitatory neurons")
    print(f"Position data shape: {locations.shape}")
    
    return spike_times, locations, excit_list

def main():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian decoding using half-split rate maps."
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
    parser.add_argument("--fold", default=0, type=int, choices=[0, 1],
                       help="Which fold to use for decoding (0 or 1, default: 0)")
    parser.add_argument("--decode_with_other_fold", action='store_true',
                       help="If set, decode using the OTHER fold's rate map (cross-decoding)")
    
    args = parser.parse_args()

    data_path = args.data_path + f'/Rat{args.rat}/Open1'
    save_path = args.save_path + f'/Rat{args.rat}'
    
    print(f"Joint X-Y-HD Bayesian Decoding using Half-Split Rate Maps")
    print(f"Data path: {data_path}")
    print(f"Spatial bins: {args.n_bins}, HD bins: {args.n_hd_bins}")
    print(f"Time bin size: {args.tau}s, Velocity threshold: {args.vel_threshold} cm/s")
    
    # Determine which fold's rate map to use
    if False: #args.decode_with_other_fold:
        # Use the opposite fold for cross-decoding
        rate_map_fold = 1 - args.fold
        print(f"Cross-decoding: Using fold {rate_map_fold} rate map to decode fold {args.fold} data")
    else:
        # Use the same fold for standard decoding
        rate_map_fold = args.fold
        print(f"Standard decoding: Using fold {rate_map_fold} for both rate map and decoding")
    
    # Load precomputed rate maps
    print(f"Loading rate map from fold {rate_map_fold}...")
    rate_map_file = f"{save_path}/x_y_hd_rate_map_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}_fold{rate_map_fold}.npy"
    
    if not os.path.exists(rate_map_file):
        print(f"Error: Rate map file not found at {rate_map_file}")
        print("Please run the half-split rate map analysis script first.")
        return
    
    rate_map = np.load(rate_map_file)  # shape: (n_x_bins, n_y_bins, n_hd_bins, n_neurons)
    print(f"Loaded rate map with shape: {rate_map.shape}")
    # rate maps look totally fine
    
    # Load Pfeiffer data
    spike_times, locations, excit_list = load_pfeiffer_data(data_path, save_path)
    
    # Create time bins for spike counting (matching the rate map script)
    #interpolation_points = 1
    #position_sample = np.diff(locations[:, 0])[1]
    #dec_tau = position_sample / interpolation_points
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + args.tau, args.tau)
    
    # Bin spike counts for excitatory neurons only
    spike_counts = bin_spike_counts(spike_times, base_time, excit_list)
    print(f"Spike counts shape: {spike_counts.shape}")

    print(f"Spike count stats:")
    print(f"  Mean firing rate: {spike_counts.mean() / args.tau:.2f} Hz")
    print(f"  Max firing rate: {spike_counts.max() / args.tau:.2f} Hz")
    print(f"  Fraction of bins with no spikes: {(spike_counts.sum(axis=1) == 0).mean():.3f}")
    print(f"  Active neurons per bin: {(spike_counts > 0).sum(axis=1).mean():.1f}")
    
    # Compare a specific time bin's activity pattern
    test_bin = 100  # Or any bin that should have activity
    active_neurons = np.where(spike_counts[test_bin] > 0)[0]
    print(f"\nTime bin {test_bin}: {len(active_neurons)} active neurons")
    if len(active_neurons) > 0:
        print(f"  Neuron indices: {active_neurons[:5]}...")  # First 5
        print(f"  Spike counts: {spike_counts[test_bin, active_neurons[:5]]}")


    print(f"First 5 neurons in rate map - mean rates: {np.nanmean(rate_map, axis=(0,1,2))[:5]}")
    print(f"First 5 neurons in spike_counts - mean rates: {spike_counts.mean(axis=0)[:5] / args.tau}")
    
    # Check if there's any correlation
    test_neuron = 0
    rate_map_profile = np.nanmean(rate_map[:,:,:,test_neuron])  # Average rate for neuron 0
    spike_rate = spike_counts[:, test_neuron].mean() / args.tau
    print(f"Neuron {test_neuron}: rate_map avg={rate_map_profile:.2f}, spike avg={spike_rate:.2f}")

    print(f"First 10 excit_list values: {excit_list[:10]}")
    print(f"Max neuron ID in excit_list: {excit_list.max()}")
    print(f"Min neuron ID in excit_list: {excit_list.min()}")

    print(f"Spike times range: {spike_times[0, 0]:.3f} to {spike_times[-1, 0]:.3f}")
    print(f"Base_time range: {base_time[0]:.3f} to {base_time[-1]:.3f}")
    print(f"Number of base_time bins: {len(base_time) - 1}")
    
    # Check if they're 0-indexed or 1-indexed
    if excit_list.min() == 0:
        print("Neurons are 0-indexed")
    else:
        print(f"Neurons start at index {excit_list.min()}")

    # checked spike counts, they look fine

    #np.save('/home/mag/variance-analysis/spike_test.npy', spike_counts)
    
    x = locations[:, 1]
    y = locations[:, 2]
    hd = (locations[:, 3] + 360) % 360
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    hd_max = np.max(hd)

    x_bins = np.linspace(x_min, x_max, args.n_bins + 1)
    y_bins = np.linspace(y_min, y_max, args.n_bins + 1)
    hd_bins = np.linspace(0, hd_max, args.n_hd_bins + 1)
    
    # Compute velocity for each time bin
    print("Computing velocity for each time bin...")
    avg_vel = compute_velocity(locations, base_time)
    
    # Calculate time centers for all bins
    time_centers = (base_time[:-1] + base_time[1:]) / 2
    if len(time_centers) > len(spike_counts):
        time_centers = time_centers[:len(spike_counts)]

    test_spike_counts = np.zeros((1, rate_map.shape[3]))
    test_x, test_y, test_hd = 5, 5, 5  # Middle bins
    test_spike_counts[0, :] = rate_map[test_x, test_y, test_hd, :] * args.tau
    
    # Decode this synthetic data
    decoded = joint_bayesian_decoding(
        rate_map, test_spike_counts, args.tau, x_bins, y_bins, hd_bins,
        np.array([10]),  # High velocity to pass filter
        args.vel_threshold
    )
    print(f"Test decode: x={decoded[0][0]}, y={decoded[1][0]}, hd={decoded[2][0]}")
    print(f"Expected approx: x={x_bins[test_x]}-{x_bins[test_x+1]}, y={y_bins[test_y]}-{y_bins[test_y+1]}")
    
    # Perform joint Bayesian decoding
    print("Starting joint Bayesian decoding...")
    start_time_decode = time.time()
    
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
    
    runtime = time.time() - start_time_decode
    print(f"Decoding completed in {runtime:.2f} seconds")
    
    # Save results
    print(f"Saving results to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # Create filename based on decoding type
    if args.decode_with_other_fold:
        base_name = f"decoded_cross_fold{args.fold}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}"
    else:
        base_name = f"decoded_fold{args.fold}_{args.n_bins}_{args.n_bins}_{args.n_hd_bins}"
    
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