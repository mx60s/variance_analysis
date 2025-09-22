# very old

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

# --- Functions for decoding ---

def bin_spike_counts(spike_times, base_time, tau):
    """
    Bin the spike times into counts for each neuron for each time bin.
    
    Parameters:
      spike_times : (N x 2) numpy array, with columns [timestamp, neuron_id]
      base_time   : 1D array of bin edges (should be created with np.arange(start, stop+tau, tau))
      tau         : time bin size (in seconds)
    
    Returns:
      spike_counts : 2D array of shape (n_time_bins, n_neurons)
    """
    n_neurons = int(np.max(spike_times[:, 1])) + 1
    n_bins = len(base_time) - 1
    spike_counts = np.zeros((n_bins, n_neurons))
    
    # Bin spike times for each neuron
    for n in range(n_neurons):
        neuron_spikes = spike_times[spike_times[:,1] == n, 0]
        counts, _ = np.histogram(neuron_spikes, bins=base_time)
        spike_counts[:, n] = counts

    return spike_counts

def per_neuron_bayesian_decoding(rate_map, spike_counts, tau, hd_bins, md_bins, avg_vel, vel_threshold, eps=1e-10):
    """
    For each time bin, compute the Poisson log-likelihood for every (movement direction,
    head direction) bin combination for each neuron independently, and then choose the bin
    with the highest posterior probability per neuron.
    
    Parameters:
      rate_map     : 3D array (n_md_bins, n_hd_bins, n_neurons) from tuning curves
      spike_counts : 2D array (n_time_bins, n_neurons) of spike counts
      tau          : time bin size (s)
      hd_bins      : array of head-direction bin edges
      md_bins      : array of movement-direction bin edges
      eps          : small constant to avoid log(0)
    
    Returns:
      decoded_hd   : 2D array of shape (n_time_bins, n_neurons) with decoded head direction (bin center)
      decoded_md   : 2D array of shape (n_time_bins, n_neurons) with decoded movement direction (bin center)
      posterior_all: list of posterior arrays (one per time bin; each array has shape (n_md_bins, n_hd_bins, n_neurons))
    """
    n_time_bins = spike_counts.shape[0]
    n_md_bins = rate_map.shape[0]
    n_hd_bins = rate_map.shape[1]
    n_neurons = spike_counts.shape[1]
    decoded_hd = np.zeros((n_time_bins, n_neurons))
    decoded_md = np.zeros((n_time_bins, n_neurons))
    posterior_all = []
    
    for t in tqdm(range(n_time_bins), desc="Decoding time bins (per neuron)"):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            decoded_hd[t, :] = np.full(n_neurons, np.nan)
            decoded_md[t, :] = np.full(n_neurons, np.nan)
            posterior_all.append(np.full((n_md_bins, n_hd_bins, n_neurons), np.nan))
            continue
            
        logL = spike_counts[t][None, None, :] * np.log(rate_map * tau + eps) - rate_map * tau
        
        max_logL = np.max(logL, axis=(0,1))  # shape: (n_neurons,)
        posterior = np.exp(logL - max_logL[None, None, :])
        sum_posterior = np.sum(posterior, axis=(0,1))  # shape: (n_neurons,)
        posterior = posterior / sum_posterior[None, None, :]
        posterior_all.append(posterior)
        
        flat_idx = np.argmax(posterior.reshape(n_md_bins * n_hd_bins, n_neurons), axis=0)  # shape: (n_neurons,)
        md_idx = flat_idx // n_hd_bins
        hd_idx = flat_idx % n_hd_bins
        
        # Determine bin centers for md and hd for each neuron
        decoded_md[t, :] = (md_bins[md_idx] + md_bins[md_idx + 1]) / 2
        decoded_hd[t, :] = (hd_bins[hd_idx] + hd_bins[hd_idx + 1]) / 2

    return decoded_hd, decoded_md, posterior_all


def main():
    parser = argparse.ArgumentParser(
        description="Per-neuron Bayesian decoding of head direction and movement direction."
    )
    parser.add_argument("--dataset", required=True, type=int, help="Dataset number (required).")
    parser.add_argument("--rat", required=True, type=str, help="Rat ID (required).")
    parser.add_argument("--n_hd_bins", default=17, type=int, help="Number of head-direction bins (default: 17).")
    parser.add_argument("--n_md_bins", default=17, type=int, help="Number of movement-direction bins (default: 17).")
    parser.add_argument("--tau", default=0.120, type=float, help="Time bin size in seconds (default: 0.120).")
    parser.add_argument("--smooth_window", required=True, type=int, help="Smoothing window for md calculation (default: 75).")
    parser.add_argument("--vel_threshold", required=False, type=int, help="Velocity threshold (default: 2 cm/s).", default=2)
    parser.add_argument("--cv_index", required=False, type=int, help="Cross validation index", default=-1)
    args = parser.parse_args()

    current_file_path = Path().resolve()
    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"

    # --- Load precomputed rate maps ---
    if args.cv_index < 0:
        rate_map_file = save_path / f"hd_md_rate_map_{args.n_hd_bins}_{args.n_md_bins}.npy"
    else:
        rate_map_file = save_path / f"hd_md_rate_map_global_train_cv{args.cv_index}_{args.n_hd_bins}_{args.n_md_bins}.npy"
    
    if not rate_map_file.exists():
        print("Rate map file not found! Check that the rate-map script has run and saved the results.")
        return
    rate_map = np.load(rate_map_file)  # shape: (n_md_bins, n_hd_bins, n_neurons)

    # --- Load spike data ---
    spike_data_file = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/SpikeData.mat"
    spike_data = loadmat(spike_data_file, struct_as_record=False, squeeze_me=True)
    spike_data = extract_matlab_obj(spike_data['S'])
    
    # Build spike_times: each row is [timestamp, neuron_id]
    arr = []
    for a in spike_data['C']:
        arr.append(a['tsd']['t'])
    full_arr = []
    for i, a in enumerate(arr):
        for aa in a:
            full_arr.append([aa, i])
    spike_times = np.array(full_arr)
    spike_times = spike_times[np.argsort(spike_times[:, 0])]

    # --- Load epoch data to filter spikes ---
    epochs_file = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/BehavEpochs.mat"
    epochs = loadmat(epochs_file, struct_as_record=False, squeeze_me=True)
    if args.dataset == 6:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    else:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    spike_mask = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[spike_mask]

    # --- Create base time bins using tau ---
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + args.tau, args.tau)
    spike_counts = bin_spike_counts(spike_times, base_time, args.tau)
   # print(spike_counts.shape)

    # --- Load tracking data to compute bin edges ---
    tracking_data_file = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Tracking_data.csv"
    tracking_data = pd.read_csv(tracking_data_file)
    tracking_data = tracking_data.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    positions = tracking_data[['timestamp', 'x', 'y', 'hd']].to_numpy()
    position_mask = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[position_mask]
    
    # Adjust positions (similar to the rate-map script)
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))

    # Load cell types
    cell_types = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/CellTypes.mat", 
                         struct_as_record=False, squeeze_me=True)
    print('total neurons', spike_counts.shape[1])
    hd_list = extract_matlab_obj(cell_types['hd'])
    excit_list = extract_matlab_obj(cell_types['ex'])
    good_list = extract_matlab_obj(cell_types['gd'])
    fs_list = extract_matlab_obj(cell_types['fs'])

    # collect the excitatory neurons which are also hd neurons for analysis later
    excit_not_hd = (excit_list == 1) & (hd_list == 0)
    excit_not_hd = excit_not_hd.astype(int)

    # --- Filter out neurons as well before decoding ---
    neuron_filter = np.where(excit_list == 1)[0]
    spike_counts = spike_counts[:, neuron_filter]

    # --- Load velocity data ---
    vel_path = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Velocity.mat"
    velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
    velocity_data = extract_matlab_obj(velocity_data['vel'])
    velocity_data = np.dstack((velocity_data['t'], velocity_data['data']))[0]
    mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
    velocity_data = velocity_data[mask]
    
    # Compute movement direction (md) and adjust head direction (hd)
    md = get_md_from_xy(positions, smoothing_window=args.smooth_window)
    hd = positions[:, 3] % 360
    # Get the maximum values used for binning in the original rate map creation
    hd_max = np.max(hd)
    md_max = np.max(md)
    
    # Create bin edges
    hd_bins = np.linspace(0, hd_max, args.n_hd_bins + 1)
    md_bins = np.linspace(0, md_max, args.n_md_bins + 1)
    
    # --- Compute average velocity per time bin using velocity_data ---
    n_bins = len(base_time) - 1
    avg_vel = np.empty(n_bins)
    for i in range(n_bins):
        mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i+1])
        if np.any(mask):
            avg_vel[i] = np.mean(velocity_data[mask, 1])
        else:
            avg_vel[i] = np.nan

    # --- Filter time bins with cross-validation indexes
    if args.cv_index > -1:
        cv_mask = np.load(save_path / f"cv_mask{args.cv_index}_{args.n_hd_bins}_{args.n_md_bins}.npy")
        test_idxs = np.where(cv_mask[:-1] == False)[0]
        
        # Filter spike_counts, avg_vel, and update time centers so that you only decode on test bins.
        spike_counts = spike_counts[test_idxs, :]
        avg_vel = avg_vel[test_idxs]
        time_centers = (base_time[:-1] + base_time[1:]) / 2
        time_centers = time_centers[test_idxs]

    print("Starting joint Bayesian decoding...")
    start_time = time.time()
    decoded_hd, decoded_md, posterior_all = per_neuron_bayesian_decoding(
        rate_map, spike_counts, args.tau, hd_bins, md_bins, avg_vel, args.vel_threshold
    )
    runtime = time.time() - start_time
    print(f"Decoding completed in {runtime:.2f} seconds")
    
    # --- Save decoded results ---
    if args.cv_index > -1:
        np.save(f"{save_path}/decoded_hd_per_neuron_{args.n_hd_bins}_{args.n_md_bins}_cv{args.cv_index}.npy", decoded_hd)
        np.save(f"{save_path}/decoded_md_per_neuron_{args.n_hd_bins}_{args.n_md_bins}_cv{args.cv_index}.npy", decoded_md)
        np.save(f"{save_path}/posterior_per_neuron_{args.n_hd_bins}_{args.n_md_bins}_cv{args.cv_index}.npy", np.array(posterior_all))
        # Save the time bin centers
        np.save(f"{save_path}/time_centers_cv{args.cv_index}.npy", time_centers)
        
        # Write CSV files summarizing the per-neuron decoding per time bin
        n_neurons = decoded_hd.shape[1]
        hd_columns = [f'neuron_{i}' for i in range(n_neurons)]
        md_columns = [f'neuron_{i}' for i in range(n_neurons)]
        
        decoded_hd_df = pd.DataFrame(decoded_hd, columns=hd_columns)
        decoded_hd_df.insert(0, 'time', time_centers)
        decoded_hd_df.to_csv(f"{save_path}/decoded_hd_per_neuron_cv{args.cv_index}.csv", index=False)
        
        decoded_md_df = pd.DataFrame(decoded_md, columns=md_columns)
        decoded_md_df.insert(0, 'time', time_centers)
        decoded_md_df.to_csv(f"{save_path}/decoded_md_per_neuron_cv{args.cv_index}.csv", index=False)
    else:
        np.save(f"{save_path}/decoded_hd_per_neuron_{args.n_hd_bins}_{args.n_md_bins}.npy", decoded_hd)
        np.save(f"{save_path}/decoded_md_per_neuron_{args.n_hd_bins}_{args.n_md_bins}.npy", decoded_md)
        np.save(f"{save_path}/posterior_per_neuron_{args.n_hd_bins}_{args.n_md_bins}.npy", np.array(posterior_all))
    
        # Save the time bin centers
        np.save(f"{save_path}/time_centers.npy", time_centers)
        
        # Write CSV files summarizing the per-neuron decoding per time bin
        n_neurons = decoded_hd.shape[1]
        hd_columns = [f'neuron_{i}' for i in range(n_neurons)]
        md_columns = [f'neuron_{i}' for i in range(n_neurons)]
        
        decoded_hd_df = pd.DataFrame(decoded_hd, columns=hd_columns)
        decoded_hd_df.insert(0, 'time', time_centers)
        decoded_hd_df.to_csv(f"{save_path}/decoded_hd_per_neuron.csv", index=False)
        
        decoded_md_df = pd.DataFrame(decoded_md, columns=md_columns)
        decoded_md_df.insert(0, 'time', time_centers)
        decoded_md_df.to_csv(f"{save_path}/decoded_md_per_neuron.csv", index=False)
        
        print(f"Decoded results saved to {save_path}")

if __name__ == "__main__":
    main()
