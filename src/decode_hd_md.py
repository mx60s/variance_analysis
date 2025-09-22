import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5_params import mat_struct
import time
from pathlib import Path

# -----------------------------
# Helper functions
# -----------------------------

def extract_matlab_obj(obj):
    """Recursively extract MATLAB structs into Python dictionaries and arrays."""
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


def get_md_from_xy(positions, smoothing_window=200):
    def circular_smooth(angles, smoothing_window):
        angles_rad = np.radians(angles)
        sin_vals = np.sin(angles_rad)
        cos_vals = np.cos(angles_rad)
        kernel = np.ones(smoothing_window) / smoothing_window
        smooth_sin = np.convolve(sin_vals, kernel, mode='same')
        smooth_cos = np.convolve(cos_vals, kernel, mode='same')
        smooth_angles = np.degrees(np.arctan2(smooth_sin, smooth_cos)) % 360
        return smooth_angles
    x = positions[:, 1]
    y = positions[:, 2]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    direction = np.degrees(np.arctan2(-dy, dx))
    direction = (direction + 360) % 360
    direction = circular_smooth(direction, smoothing_window=smoothing_window)
    direction = np.concatenate([direction, [direction[-1]]])
    return direction


def bin_spike_counts(spike_times, base_time):
    """Return (n_bins, n_neurons_full) spike count matrix for given bin edges."""
    n_neurons_full = int(np.max(spike_times[:, 1])) + 1
    n_bins = len(base_time) - 1
    spike_counts = np.zeros((n_bins, n_neurons_full), dtype=float)
    for n in range(n_neurons_full):
        neuron_spikes = spike_times[spike_times[:, 1] == n, 0]
        if neuron_spikes.size == 0:
            continue
        counts, _ = np.histogram(neuron_spikes, bins=base_time)
        spike_counts[:, n] = counts
    return spike_counts


# -----------------------------
# Decoders
# -----------------------------

def solo_bayesian_decoding(rate_map_md, spike_counts_sel, tau, md_bins, avg_vel, vel_threshold, eps=1e-10):
    """
    Solo MD decoding using Poisson likelihood.
      rate_map_md: (n_md_bins, n_neurons)
      spike_counts_sel: (n_time_bins, n_neurons) aligned with neuron order in rate_map_md
    Returns decoded_md (n_time_bins,) and list of posteriors (each (n_md_bins,)).
    """
    n_time_bins = spike_counts_sel.shape[0]
    n_md_bins, n_neurons = rate_map_md.shape
    decoded_md = np.full(n_time_bins, np.nan)
    posterior_all = []

    # Precompute log(rate*tau) and rate*tau for speed; guard NaNs
    lam = np.nan_to_num(rate_map_md * tau, nan=0.0, posinf=0.0, neginf=0.0)
    loglam = np.log(lam + eps)

    for t in range(n_time_bins):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full(n_md_bins, np.nan))
            continue
        k_t = spike_counts_sel[t]  # (n_neurons,)
        # logL over md bins: sum_n [ k_n * log(lam_mn) - lam_mn ]
        logL = (k_t[None, :] * loglam - lam).sum(axis=1)
        # normalize
        max_logL = np.max(logL)
        post = np.exp(logL - max_logL)
        s = np.nansum(post)
        if s <= 0:
            posterior_all.append(np.full(n_md_bins, np.nan))
            continue
        post /= s
        posterior_all.append(post)
        md_idx = int(np.nanargmax(post))
        decoded_md[t] = (md_bins[md_idx] + md_bins[md_idx + 1]) / 2.0

    return decoded_md, posterior_all


# -----------------------------
# I/O helpers for SOLO mode
# -----------------------------

def load_md_products(save_path, n_md_bins, use_shuffled):
    """Load saved artifacts from md_marginals.py."""
    suffix = "_shuffled_md" if use_shuffled else ""
    md_edges_fp = Path(save_path) / f"md_bin_edges_{n_md_bins}{suffix}.npy"
    neuron_order_fp = Path(save_path) / f"neuron_id_order_md_{n_md_bins}.npy"  # neuron order has no shuffle suffix
    base_edges_fp = Path(save_path) / f"cv_base_time_edges{suffix}.npy"
    if not md_edges_fp.exists():
        raise FileNotFoundError(f"Missing MD bin edges at {md_edges_fp}")
    if not neuron_order_fp.exists():
        raise FileNotFoundError(f"Missing neuron order at {neuron_order_fp}")
    md_bins = np.load(md_edges_fp)
    kept_neuron_ids = np.load(neuron_order_fp)
    base_time_edges = np.load(base_edges_fp) if base_edges_fp.exists() else None
    return md_bins, kept_neuron_ids, base_time_edges


def load_md_rate_map(save_path, n_md_bins, cv_index, use_shuffled):
    """Return (rate_map_md, file_used). If cv_index<0, average across available CV splits."""
    suffix = "_shuffled_md" if use_shuffled else ""
    if cv_index >= 0:
        fp = Path(save_path) / f"md_marginal_train_cv{cv_index}_{n_md_bins}{suffix}.npy"
        if not fp.exists():
            raise FileNotFoundError(f"MD marginal train map not found: {fp}")
        return np.load(fp), str(fp)
    # average across cv folds if present
    maps = []
    for cv in range(8):  # try up to 8; break when miss
        cand = Path(save_path) / f"md_marginal_train_cv{cv}_{n_md_bins}{suffix}.npy"
        if cand.exists():
            maps.append(np.load(cand))
    if not maps:
        # fallback to test maps if only those exist
        for cv in range(8):
            cand = Path(save_path) / f"md_marginal_test_cv{cv}_{n_md_bins}{suffix}.npy"
            if cand.exists():
                maps.append(np.load(cand))
    if not maps:
        raise FileNotFoundError(f"No MD marginal maps found in {save_path} for n_md_bins={n_md_bins}")
    stacked = np.stack(maps, axis=0)
    return np.nanmean(stacked, axis=0), "avg_of_cv_maps"


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Bayesian decoding with optional SOLO (MD-only) mode.")
    parser.add_argument("--dataset", required=True, type=int)
    parser.add_argument("--rat", required=True, type=str)
    parser.add_argument("--n_md_bins", default=17, type=int)
    parser.add_argument("--tau", default=0.120, type=float, help="Bin size in seconds; should match marginals.")
    parser.add_argument("--smooth_window", required=True, type=int)
    parser.add_argument("--vel_threshold", default=2, type=float)
    parser.add_argument("--cv_index", default=-1, type=int, help="If >=0, decode only on held-out bins of this CV split.")
    parser.add_argument("--shuffle_md", action='store_true', help="Use shuffled-md products from training script.")
    parser.add_argument("--solo", action='store_true', help="Use MD-only marginals (from md_marginals.py) for decoding.")

    args = parser.parse_args()

    cwd = Path().resolve()
    save_path = cwd / f"results/dusz/Dataset_{args.dataset}/{args.rat}"

    # -----------------
    # Load SOLO artifacts
    # -----------------
    if not args.solo:
        raise SystemExit("This script is configured for --solo MD decoding only in this version.")

    md_bins, kept_neuron_ids, base_time_edges_saved = load_md_products(save_path, args.n_md_bins, args.shuffle_md)
    rate_map_md, src_used = load_md_rate_map(save_path, args.n_md_bins, args.cv_index, args.shuffle_md)
    if rate_map_md.ndim != 2:
        raise ValueError(f"Expected MD rate map (n_md_bins, n_neurons). Got shape {rate_map_md.shape} from {src_used}")

    # -----------------
    # Load spikes & epoch
    # -----------------
    spike_data_file = cwd / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/SpikeData.mat"
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

    epochs_file = cwd / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/BehavEpochs.mat"
    epochs = loadmat(epochs_file, struct_as_record=False, squeeze_me=True)
    epoch_data = extract_matlab_obj(epochs['wakeEp'] if args.dataset == 6 else epochs['wake1Ep'])
    mask = (spike_times[:, 0] > epoch_data['start']) & (spike_times[:, 0] < epoch_data['stop'])
    spike_times = spike_times[mask]

    # -----------------
    # Base-time edges (align with marginals if possible)
    # -----------------
    if base_time_edges_saved is not None:
        base_time = base_time_edges_saved
        # sanity check tau
        diffs = np.diff(base_time)
        est_tau = float(np.median(diffs))
        if abs(est_tau - args.tau) > 1e-6:
            print(f"[warn] tau mismatch: marginals tau≈{est_tau:.6f}s vs arg tau={args.tau:.6f}s. Using saved edges.")
    else:
        base_time = np.arange(spike_times[0, 0], spike_times[-1, 0] + args.tau, args.tau)

    # Spike counts in full space then select neurons
    spike_counts_full = bin_spike_counts(spike_times, base_time)
    # Align to kept neuron IDs
    spike_counts = spike_counts_full[:, kept_neuron_ids]

    # -----------------
    # Load velocity & compute per-bin avg velocity
    # -----------------
    avg_vel = np.empty(len(base_time) - 1)
    avg_vel[:] = np.nan
    vel_path = cwd / f"data/dusz/Dataset_{args.dataset}/{args.rat}/Data/Velocity.mat"
    if vel_path.exists():
        velocity_data = loadmat(vel_path, struct_as_record=False, squeeze_me=True)
        velocity_data = extract_matlab_obj(velocity_data['vel'])
        velocity_data = np.dstack((velocity_data['t'], velocity_data['data']))[0]
        vmask = (velocity_data[:, 0] > epoch_data['start']) & (velocity_data[:, 0] < epoch_data['stop'])
        velocity_data = velocity_data[vmask]
        for i in range(len(base_time) - 1):
            m = (velocity_data[:, 0] >= base_time[i]) & (velocity_data[:, 0] < base_time[i + 1])
            if np.any(m):
                avg_vel[i] = np.nanmean(velocity_data[m, 1])
    else:
        print("[warn] Velocity.mat not found; decoding will skip bins due to NaN velocities. Set --vel_threshold 0 to decode anyway.")
        avg_vel[:] = np.nan  # user can set threshold 0 to avoid skipping

    # -----------------
    # If CV specified, restrict decoding to test bins only
    # -----------------
    time_centers = (base_time[:-1] + base_time[1:]) / 2.0
    if args.cv_index >= 0:
        cv_mask_fp = Path(save_path) / f"cv_mask{args.cv_index}_md_{args.n_md_bins}{('_shuffled_md' if args.shuffle_md else '')}.npy"
        if not cv_mask_fp.exists():
            # also support legacy name without _md_
            alt = Path(save_path) / f"cv_mask{args.cv_index}_{args.n_md_bins}_{args.n_md_bins}{('_shuffled_md' if args.shuffle_md else '')}.npy"
            cv_mask_fp = alt if alt.exists() else cv_mask_fp
        cv_mask = np.load(cv_mask_fp)
        test_idx = np.where(~cv_mask)[0]
        spike_counts = spike_counts[test_idx, :]
        avg_vel = avg_vel[test_idx]
        time_centers = time_centers[test_idx]

    # -----------------
    # Decode (SOLO)
    # -----------------
    print("Starting SOLO MD Bayesian decoding from marginals...")
    t0 = time.time()
    decoded_md, posterior_all = solo_bayesian_decoding(rate_map_md, spike_counts, args.tau, md_bins, avg_vel, args.vel_threshold)
    dt = time.time() - t0
    print(f"Decoding completed in {dt:.2f} s")

    # -----------------
    # Save outputs
    # -----------------
    suffix = "_shuffled_md" if args.shuffle_md else ""
    out_base = Path(save_path)
    np.save(out_base / f"decoded_md_{args.n_md_bins}{suffix}_solo.npy" if args.cv_index < 0 else out_base / f"decoded_md_{args.n_md_bins}_cv{args.cv_index}{suffix}_solo.npy", decoded_md)
    # store posteriors as ragged object if needed
    try:
        np.save(out_base / f"posterior_{args.n_md_bins}{suffix}_solo.npy" if args.cv_index < 0 else out_base / f"posterior_{args.n_md_bins}_cv{args.cv_index}{suffix}_solo.npy", np.array(posterior_all, dtype=object))
    except Exception:
        # fallback to large 2D if all posteriors valid
        valid = [p for p in posterior_all if p is not None and not np.any(np.isnan(p))]
        if valid:
            P = np.vstack(valid)
            np.save(out_base / f"posterior_{args.n_md_bins}{suffix}_solo_valid.npy", P)

    np.save(out_base / f"time_centers{'' if args.cv_index < 0 else f'_cv{args.cv_index}'}{suffix}_solo.npy", time_centers)
    print(f"Saved decoded outputs to {out_base}")


if __name__ == "__main__":
    main()
