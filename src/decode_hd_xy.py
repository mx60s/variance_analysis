import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5_params import mat_struct
import time
from tqdm import tqdm
from pathlib import Path

def extract_matlab_obj(obj):
    """
    Recursively convert MATLAB struct objects to Python dictionaries.
    """
    if not (isinstance(obj, mat_struct) or isinstance(obj, dict)):
        return obj

    out = {}
    obj = obj.__dict__ if isinstance(obj, mat_struct) else obj

    for key, val in obj.items():
        # Skip metadata fields
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

def bin_spike_counts(spike_times, base_time):
    """
    Bin spike times into counts for each neuron and time bin.
    
    Parameters:
        spike_times: (N x 2) array with columns [timestamp, neuron_id]
        base_time: 1D array of time bin edges
    
    Returns:
        spike_counts: 2D array of shape (n_time_bins, n_neurons)
    """
    n_neurons = int(np.max(spike_times[:, 1])) + 1
    n_bins = len(base_time) - 1
    spike_counts = np.zeros((n_bins, n_neurons))
    
    # Use numpy histogram for efficient binning
    for neuron_id in range(n_neurons):
        neuron_spikes = spike_times[spike_times[:, 1] == neuron_id, 0]
        if len(neuron_spikes) > 0:
            counts, _ = np.histogram(neuron_spikes, bins=base_time)
            spike_counts[:, neuron_id] = counts

    return spike_counts

def joint_bayesian_decoding(rate_map, spike_counts, tau, x_bins, y_bins, hd_bins, 
                           avg_vel, vel_threshold, eps=1e-10):
    """
    Joint Bayesian decoding of spatial position (x, y) and head direction using Poisson likelihood.
    
    For each time bin, computes the Poisson log-likelihood for every (x, y, head direction)
    bin combination, then selects the bin with highest posterior probability.
    
    Time bins with velocity below threshold are skipped (decoded as NaN).
    
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
        # where k = spike_counts[t], λ = rate_map, τ = tau
        spike_data = spike_counts[t][None, None, None, :]  # Shape: (1, 1, 1, n_neurons)
        rate_data = rate_map * tau  # Expected spike count
        
        # Compute log-likelihood for each (x, y, hd) bin
        log_likelihood = np.nansum(
            spike_data * np.log(rate_data + eps) - rate_data, 
            axis=3
        )
        
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

def load_data(dataset, rat, current_file_path):
    """
    Load and preprocess all data files needed for decoding.
    
    Returns:
        spike_times: Filtered spike timing data
        positions: Position data array
        velocity_data: Velocity measurements
        epoch_data: Behavioral epoch information
        excit_list: Indices of excitatory neurons
    """
    base_path = current_file_path / f"data/dusz/Dataset_{dataset}/{rat}/Data"
    
    # Load spike data
    print("Loading spike data...")
    spike_data = loadmat(base_path / "SpikeData.mat", struct_as_record=False, squeeze_me=True)
    spike_data = extract_matlab_obj(spike_data['S'])
    
    # Build spike times array
    spike_times_list = []
    for cell_idx, cell_data in enumerate(spike_data['C']):
        spike_times_cell = cell_data['tsd']['t']
        for spike_time in spike_times_cell:
            spike_times_list.append([spike_time, cell_idx])
    
    spike_times = np.array(spike_times_list)
    spike_times = spike_times[np.argsort(spike_times[:, 0])]  # Sort by time
    
    # Load behavioral epochs
    print("Loading behavioral epochs...")
    epochs = loadmat(base_path / "BehavEpochs.mat", struct_as_record=False, squeeze_me=True)
    if dataset == 6:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    else:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    
    # Filter spikes by behavioral epoch
    spike_mask = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[spike_mask]
    
    # Load cell type classifications
    print("Loading cell classifications...")
    cell_types = loadmat(base_path / "CellTypes.mat", struct_as_record=False, squeeze_me=True)
    hd_list = extract_matlab_obj(cell_types['hd'])
    excit_list = extract_matlab_obj(cell_types['ex'])
    excit_indices = np.where(hd_list == 1)[0]
    
    # Load position data
    print("Loading position data...")
    tracking_data = pd.read_csv(base_path / "Tracking_data.csv")
    tracking_data = tracking_data.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    positions = tracking_data[['timestamp', 'x', 'y', 'hd']].to_numpy()
    
    # Filter positions by behavioral epoch
    position_mask = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[position_mask]
    
    # Normalize positions to start from origin
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))
    
    # Load velocity data
    print("Loading velocity data...")
    velocity_data = None
    vel_path = base_path / "Velocity.mat"
    if vel_path.exists():
        velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
        velocity_data = extract_matlab_obj(velocity_data['vel'])
        velocity_data = np.column_stack((velocity_data['t'], velocity_data['data']))
        
        # Filter by behavioral epoch
        mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
        velocity_data = velocity_data[mask]
    
    return spike_times, positions, velocity_data, epoch_data, excit_indices

def main():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian decoding of spatial position (x, y) and head direction."
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
    parser.add_argument("--tau", default=0.120, type=float, 
                       help="Time bin size in seconds (default: 0.120)")
    parser.add_argument("--vel_threshold", default=2, type=int, 
                       help="Velocity threshold in cm/s (default: 2)")
    parser.add_argument("--cv_index", default=-1, type=int, 
                       help="Cross-validation fold index (-1 for no CV)")
    
    args = parser.parse_args()
    
    print(f"Joint X-Y-HD Bayesian Decoding")
    print(f"Dataset {args.dataset}, Rat {args.rat}")
    print(f"X bins: {args.n_x_bins}, Y bins: {args.n_y_bins}, HD bins: {args.n_hd_bins}")
    print(f"Time bin size: {args.tau}s, Velocity threshold: {args.vel_threshold} cm/s")
    if args.cv_index >= 0:
        print(f"Using cross-validation fold: {args.cv_index}")

    current_file_path = Path().resolve()
    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"

    # Load precomputed rate maps
    print("Loading rate maps...")
    if args.cv_index < 0:
        rate_map_file = save_path / f"x_y_hd_rate_map_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}.npy"
    else:
        rate_map_file = save_path / f"x_y_hd_rate_map_cv{args.cv_index}_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}_global_train.npy"
    
    if not rate_map_file.exists():
        print(f"Error: Rate map file not found at {rate_map_file}")
        print("Please run the rate map analysis script first.")
        return
    
    rate_map = np.load(rate_map_file)  # shape: (n_x_bins, n_y_bins, n_hd_bins, n_neurons)
    print(f"Loaded rate map with shape: {rate_map.shape}")

    # Load all required data
    spike_times, positions, velocity_data, epoch_data, excit_indices = load_data(
        args.dataset, args.rat, current_file_path
    )

    # Create time bins for spike counting
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + args.tau, args.tau)
    spike_counts = bin_spike_counts(spike_times, base_time)
    
    # Filter spike counts to only include excitatory neurons
    spike_counts = spike_counts[:, excit_indices]
    print(f"Using {len(excit_indices)} excitatory neurons for decoding")

    # Extract spatial features
    x = positions[:, 1]
    y = positions[:, 2]
    hd = (positions[:, 3] + 360) % 360  # Normalize head direction
    
    # Create bin edges matching those used in rate map creation
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    hd_max = np.max(hd)
    
    x_bins = np.linspace(x_min, x_max, args.n_x_bins + 1)
    y_bins = np.linspace(y_min, y_max, args.n_y_bins + 1)
    hd_bins = np.linspace(0, hd_max, args.n_hd_bins + 1)

    # Compute average velocity per time bin
    print("Computing velocity for each time bin...")
    avg_vel = np.full(len(base_time) - 1, np.nan)
    
    if velocity_data is not None:
        for i in range(len(base_time) - 1):
            mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i + 1])
            if np.any(mask):
                avg_vel[i] = np.nanmean(velocity_data[mask, 1])

    # Apply cross-validation filtering if specified
    if args.cv_index >= 0:
        print(f"Applying CV fold {args.cv_index} filtering...")
        cv_mask_file = save_path / f"cv_mask{args.cv_index}_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}.npy"
        
        if not cv_mask_file.exists():
            print(f"Error: CV mask file not found at {cv_mask_file}")
            return
        
        cv_mask = np.load(cv_mask_file)
        test_indices = np.where(cv_mask == False)[0]
        
        # Filter to test data only
        spike_counts = spike_counts[test_indices, :]
        avg_vel = avg_vel[test_indices]
        
        # Calculate time centers for test bins
        # Use the center of each test time bin
        time_centers = (base_time[test_indices] + base_time[test_indices + 1]) / 2
        
        print(f"Decoding on {len(test_indices)} test time bins")
    else:
        # Calculate time centers for all bins
        time_centers = (base_time[:-1] + base_time[1:]) / 2

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
        base_name = f"decoded_cv{args.cv_index}_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}"
        np.save(save_path / f"{base_name}_x.npy", decoded_x)
        np.save(save_path / f"{base_name}_y.npy", decoded_y)
        np.save(save_path / f"{base_name}_hd.npy", decoded_hd)
        np.save(save_path / f"{base_name}_posterior.npy", np.array(posterior_all))
        np.save(save_path / f"{base_name}_time_centers.npy", time_centers)
    else:
        # Full dataset results
        base_name = f"decoded_{args.n_x_bins}_{args.n_y_bins}_{args.n_hd_bins}"
        np.save(save_path / f"{base_name}_x.npy", decoded_x)
        np.save(save_path / f"{base_name}_y.npy", decoded_y)
        np.save(save_path / f"{base_name}_hd.npy", decoded_hd)
        np.save(save_path / f"{base_name}_posterior.npy", np.array(posterior_all))
        np.save(save_path / f"{base_name}_time_centers.npy", time_centers)

    # Print summary statistics
    valid_x = decoded_x[~np.isnan(decoded_x)]
    valid_y = decoded_y[~np.isnan(decoded_y)]
    valid_hd = decoded_hd[~np.isnan(decoded_hd)]
    
    print(f"\nDecoding Summary:")
    print(f"Total time bins: {len(decoded_hd)}")
    print(f"Successfully decoded: {len(valid_hd)} ({100*len(valid_hd)/len(decoded_hd):.1f}%)")
    print(f"X position range: {np.min(valid_x):.2f} - {np.max(valid_x):.2f} units")
    print(f"Y position range: {np.min(valid_y):.2f} - {np.max(valid_y):.2f} units")
    print(f"Head direction range: {np.min(valid_hd):.1f} - {np.max(valid_hd):.1f} degrees")
    print("Decoding analysis completed successfully!")

if __name__ == "__main__":
    main()