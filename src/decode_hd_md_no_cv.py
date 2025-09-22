# Simplified decoding without cross-validation
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

    direction = np.degrees(np.arctan2(-dy, dx))
    direction = (direction + 360) % 360

    direction = np.concatenate([direction, [direction[-1]]])

    return direction

def bin_spike_counts(spike_times, base_time, tau):
    """
    Bin the spike times into counts for each neuron for each time bin.
    """
    n_neurons = int(np.max(spike_times[:, 1])) + 1
    n_bins = len(base_time) - 1
    spike_counts = np.zeros((n_bins, n_neurons))
    
    for n in range(n_neurons):
        neuron_spikes = spike_times[spike_times[:,1] == n, 0]
        counts, _ = np.histogram(neuron_spikes, bins=base_time)
        spike_counts[:n_bins, n] = counts

    return spike_counts

def joint_bayesian_decoding(rate_map, spike_counts, tau, hd_bins, md_bins, avg_vel, vel_threshold, eps=1e-10):
    """
    Bayesian decoding for each time bin.
    """
    n_time_bins = spike_counts.shape[0]
    n_md_bins = rate_map.shape[0]
    n_hd_bins = rate_map.shape[1]
    decoded_hd = np.zeros(n_time_bins)
    decoded_md = np.zeros(n_time_bins)
    posterior_all = [] 
    
    for t in tqdm(range(n_time_bins), desc="Decoding time bins"):
        # Skip decoding if velocity is low
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            decoded_hd[t] = np.nan
            decoded_md[t] = np.nan
            posterior_all.append(np.full((n_md_bins, n_hd_bins), np.nan))
            continue

        # Poisson log-likelihood
        logL = np.nansum(spike_counts[t][None, None, :] * np.log(rate_map * tau + eps) - rate_map * tau, axis=2)
        
        # Compute posterior
        max_logL = np.max(logL)
        posterior = np.exp(logL - max_logL)
        posterior = posterior / np.nansum(posterior)
        posterior_all.append(posterior)
        
        # Find maximum posterior bin
        md_idx, hd_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
        decoded_md[t] = (md_bins[md_idx] + md_bins[md_idx + 1]) / 2
        decoded_hd[t] = (hd_bins[hd_idx] + hd_bins[hd_idx + 1]) / 2

    return decoded_hd, decoded_md, posterior_all

def main():
    parser = argparse.ArgumentParser(
        description="Simplified Bayesian decoding without CV."
    )
    parser.add_argument("--dataset", required=True, type=int, help="Dataset number.")
    parser.add_argument("--rat", required=True, type=str, help="Rat ID.")
    parser.add_argument("--n_hd_bins", default=17, type=int, help="Number of HD bins.")
    parser.add_argument("--n_md_bins", default=17, type=int, help="Number of MD bins.")
    parser.add_argument("--tau", default=0.120, type=float, help="Time bin size (s).")
    parser.add_argument("--smooth_window", required=True, type=int, help="Smoothing window.")
    parser.add_argument("--vel_threshold", required=False, type=int, help="Velocity threshold.", default=2)
    parser.add_argument("--shuffle_md", action='store_true', help="Use shuffled MD rate maps.")
    
    args = parser.parse_args()

    current_file_path = Path().resolve()
    save_path = current_file_path / f"results/dusz/Dataset_{args.dataset}/{args.rat}"

    # Load rate map
    suffix = "_shuffled_md" if args.shuffle_md else ""
    rate_map_file = save_path / f"hd_md_rate_map_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy"
    base_time_file = save_path / f"base_time_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy"
    
    if not rate_map_file.exists():
        print(f"Rate map file not found at {rate_map_file}!")
        return
    
    rate_map = np.load(rate_map_file)
    base_time = np.load(base_time_file)
    
    # Load spike data
    spike_data_file = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/SpikeData.mat"
    spike_data = loadmat(spike_data_file, struct_as_record=False, squeeze_me=True)
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

    # Load epoch data
    epochs_file = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/BehavEpochs.mat"
    epochs = loadmat(epochs_file, struct_as_record=False, squeeze_me=True)
    if args.dataset == 6:
        epoch_data = extract_matlab_obj(epochs['wakeEp'])
    else:
        epoch_data = extract_matlab_obj(epochs['wake1Ep'])
    spike_mask = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[spike_mask]

    # Bin spike counts
    spike_counts = bin_spike_counts(spike_times, base_time, args.tau)

    # Filter neurons
    cell_types = loadmat(current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/CellTypes.mat", 
                         struct_as_record=False, squeeze_me=True)
    hd_list = extract_matlab_obj(cell_types['hd'])
    neuron_filter = np.where(hd_list == 1)[0]
    spike_counts = spike_counts[:, neuron_filter]
    
    # Load tracking data
    tracking_data_file = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Tracking_data.csv"
    tracking_data = pd.read_csv(tracking_data_file)
    tracking_data = tracking_data.set_axis(['timestamp', 'x', 'y', 'hd', 'pitch', 'roll'], axis=1)
    positions = tracking_data[['timestamp', 'x', 'y', 'hd']].to_numpy()
    position_mask = (positions[:, 0] > epoch_data['start']) & (positions[:, 0] < epoch_data['stop'])
    positions = positions[position_mask]
    positions[:, 1] += abs(np.min(positions[:, 1]))
    positions[:, 2] += abs(np.min(positions[:, 2]))

    # Load velocity data
    vel_path = current_file_path / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Velocity.mat"
    velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
    velocity_data = extract_matlab_obj(velocity_data['vel'])
    velocity_data = np.dstack((velocity_data['t'], velocity_data['data']))[0]
    mask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
    velocity_data = velocity_data[mask]
    
    # Compute MD and HD
    md = get_md_from_xy(positions, smoothing_window=args.smooth_window)
    if args.shuffle_md:
        np.random.shuffle(md)
    
    hd = (positions[:, 3] + 360) % 360
    hd_max = np.max(hd)
    md_max = np.max(md)
    
    # Create bin edges
    hd_bins = np.linspace(0, hd_max, args.n_hd_bins + 1)
    md_bins = np.linspace(0, md_max, args.n_md_bins + 1)
    
    # Compute average velocity per time bin
    avg_vel = np.empty(len(base_time))
    for i in range(len(base_time) - 1):
        mask = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i+1])
        if np.any(mask):
            avg_vel[i] = np.nanmean(velocity_data[mask, 1])
        else:
            avg_vel[i] = np.nan
    avg_vel[-1] = np.nan

    time_centers = (base_time[:-1] + base_time[1:]) / 2

    # Decode
    print("Starting Bayesian decoding...")
    if args.shuffle_md:
        print("(Using MD-shuffled rate maps)")
    start_time = time.time()
    
    decoded_hd, decoded_md, posterior_all = joint_bayesian_decoding(
        rate_map, spike_counts, args.tau, hd_bins, md_bins, avg_vel, args.vel_threshold
    )
    
    runtime = time.time() - start_time
    print(f"Decoding completed in {runtime:.2f} seconds")

    # Save results
    suffix = "_shuffled" if args.shuffle_md else ""
    np.save(f"{save_path}/time_centers_simple{suffix}.npy", time_centers)
    np.save(f"{save_path}/decoded_hd_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", decoded_hd)
    np.save(f"{save_path}/decoded_md_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", decoded_md)
    np.save(f"{save_path}/posterior_simple_{args.n_hd_bins}_{args.n_md_bins}{suffix}.npy", np.array(posterior_all))
    
    print(f"Decoded results saved to {save_path}")

if __name__ == "__main__":
    main()