from scipy.io import loadmat, savemat
import scipy.io
import scipy.signal as sci
import numpy as np
import time
import math
from tqdm import tqdm

def match_times(time_list, search_time):
    times = time_list[:, 0]
    array_row = np.searchsorted(times, search_time, side="left")
    if array_row > 0 and (array_row == len(times) or abs(search_time - times[array_row - 1]) < abs(search_time - times[array_row])):
        return time_list[array_row - 1, 1]
    else:
        return time_list[array_row, 1]

def adj_rate_maps(smooth_maps, rate_maps, time_window):
    adjusted_map = np.asarray([
        smooth_maps[:, :, k] * (np.nanmax(rate_maps[:, :, k]) / np.nanmax(smooth_maps[:, :, k]))
        for k in range(len(rate_maps[0, 0, :]))
    ])
    smoothed_magnitude_map = np.zeros(np.shape(smooth_maps))
    for k in range(len(smooth_maps[0, 0, :])):
        smoothed_magnitude_map[:, :, k] = adjusted_map[k, :, :]
    return smoothed_magnitude_map


seed = 777
np.random.seed(seed=seed)

filter_vel = True
timebin_tol = 1e-4
tau = 0.120
interpolation_points = 2
velocity_threshold = 5 # cm/s
env_len = 200
bin_count = 17
hd_bin_counts = [17]
bin_width = env_len / bin_count
start_time = time.time()

save_path = '/home/mag/Space/results/pfeiffer'
data_path = '/home/mag/Space/data/PfeifferData/Rat1/Open1'

print(f"Seed {seed}")

for hd_bin_count in hd_bin_counts:
    print(f"bin count {hd_bin_count}")
            
    locations = loadmat(f'{data_path}/Position_Data.mat')['Position_Data']
    spike_data = loadmat(f'{data_path}/Spike_Data.mat')
    spike_times = spike_data['Spike_Data']
    lfp = loadmat(f'{data_path}/LFP_Data.mat')['LFP_Data']

    sample_rate = 1 / np.diff(lfp[:, 0])[0]
    position_sample = np.diff(locations[:, 0])[1]
    dec_tau = position_sample / interpolation_points
    b, a = sci.butter(2, [6, 12], btype='bandpass', fs=sample_rate)
    filt_band = sci.lfilter(b, a, lfp[:, 1])

    peaks = sci.find_peaks(-filt_band, distance=50)[0]
    troughs = sci.find_peaks(filt_band, distance=50)[0]

    crossings = np.where(np.diff(np.sign(filt_band)) != 0)[0] + 1
    crossings = np.delete(crossings, np.where(crossings < peaks[0])[0][1:])
    crossings = np.delete(crossings, np.where((crossings > peaks[0]) & (crossings < troughs[0]))[0][1:])
    cross_270 = crossings[1::2].copy()
    cross_90 = crossings[::2].copy()
    if peaks[0] > troughs[0]:
        cross_270 = crossings[::2].copy()
        cross_90 = crossings[1::2].copy()
    else:
        cross_270 = crossings[1::2].copy()
        cross_90 = crossings[::2].copy()

    peaks = np.c_[peaks, np.zeros(np.shape(peaks))]
    peaks[::2, 1] = 360
    troughs = np.c_[troughs, np.ones(np.shape(troughs)) * 180]
    cross_90 = np.c_[cross_90, np.ones(np.shape(cross_90)) * 90]
    cross_270 = np.c_[cross_270, np.ones(np.shape(cross_270)) * 270]

    phase_extrema = np.r_[peaks, cross_90, troughs, cross_270]
    sort_extrema = np.argsort(phase_extrema[:, 0])
    phase_extrema = phase_extrema[sort_extrema]

    mis_interpolate = []
    print('interpolate')
    for i in tqdm(range(len(phase_extrema[:, 0]) - 1)):
        if phase_extrema[i + 1, 1] == 0:
            interp_line = np.linspace(phase_extrema[i, 1], 360, int(np.abs(phase_extrema[i + 1, 0] - phase_extrema[i, 0])))
        elif phase_extrema[i, 1] == 360:
            interp_line = np.linspace(0, phase_extrema[i + 1, 1], int(np.abs(phase_extrema[i + 1, 0] - phase_extrema[i, 0])))
        else:
            interp_line = np.linspace(phase_extrema[i, 1], phase_extrema[i + 1, 1], int(np.abs(phase_extrema[i + 1, 0] - phase_extrema[i, 0])))
        mis_interpolate.append(interp_line)

    interpolated_phase = [item for sublist in mis_interpolate for item in sublist]
    pad_end = np.linspace(phase_extrema[-1, 1], False, int(len(filt_band) - phase_extrema[-1, 0]))
    pad_start = np.ones(int(phase_extrema[0, 0])) * phase_extrema[0, 1]
    interpolated_phase = np.r_[interpolated_phase, pad_end]
    interpolated_phase = np.r_[pad_start, interpolated_phase]

    theta_phase = interpolated_phase
    theta_time = np.asarray([lfp[:, 0], theta_phase]).T
    base_time = np.arange(spike_times[0, 0], spike_times[-1, 0], dec_tau)
    stapler = [match_times(theta_time, spike_times[k, 0]) for k in range(len(spike_times[:, 0]))]
    spike_times = np.c_[spike_times, stapler]
    excit_list = spike_data['Excitatory_Neurons']
    spike_mat = []

    for i in tqdm(range(int(max(spike_times[:, 1])))):
        if i in excit_list:
            spike_mat.append(spike_times[spike_times[:, 1] == i])

    time_step_gap = locations[3, 0] - locations[2, 0]
    point_steps = np.diff(locations[:, 1:3], axis=0)
    point_steps = np.vstack((point_steps, point_steps[-1]))
    point_dist = np.linalg.norm(point_steps, axis=1) / time_step_gap

    vel_list = np.zeros((np.shape(base_time)))
    for t in tqdm(range(len(base_time) - 1)):
        spikes_in_window = np.where((locations[:, 0] < base_time[t + 1]) & (locations[:, 0] >= base_time[t]))[0]
        if np.size(spikes_in_window) == 0:
            dister = 0
        else:
            dister = np.mean(point_dist[spikes_in_window])
        vel_list[t] = dister

    fast_indices = np.where(vel_list >= velocity_threshold)[0]
    fast_bins = base_time[fast_indices] if filter_vel else base_time

    freq_maps = np.empty((len(base_time), 2, len(spike_mat)))

    for i in tqdm(range(len(spike_mat))):
        current_map = np.vstack((base_time, np.empty(len(base_time)))).T
        for j in range(len(base_time) - 1):
            running_freq = len(np.where((spike_mat[i][:, 0] <= base_time[j + 1]) &
                                        (spike_mat[i][:, 0] > base_time[j]))[0]) / tau
            current_map[j, 1] = running_freq
        freq_maps[:, :, i] = current_map

    spatial_bins = np.linspace(0, env_len, bin_count)
    rate_maps = np.zeros((bin_count, bin_count, len(spike_mat)))

    hd_bins = np.linspace(0, 360.01, hd_bin_count)
    hd_rate_map = np.zeros((bin_count, bin_count, len(spike_mat), hd_bin_count))
    hd_occupancy_map = np.zeros((bin_count, bin_count, hd_bin_count))
    hd_rate_map_fold0 = np.zeros((bin_count, bin_count, len(spike_mat), hd_bin_count))
    hd_rate_map_fold1 = np.zeros((bin_count, bin_count, len(spike_mat), hd_bin_count))

    hd_rate_map[:] = np.nan
    hd_rate_map_fold0[:] = np.nan
    hd_rate_map_fold1[:] = np.nan

    hd_occ = np.zeros((bin_count, bin_count, hd_bin_count))

    for i in tqdm(range(len(spatial_bins) - 1)):
        for j in range(len(spatial_bins) - 1):
            freq_locs = np.where((locations[:, 1] < spatial_bins[i + 1]) & (locations[:, 1] >= spatial_bins[i]) &
                                    (locations[:, 2] < spatial_bins[j + 1]) & (locations[:, 2] >= spatial_bins[j]))[0]

            if freq_locs.size > 5:
                freq_locers = np.array([np.abs(base_time - locations[freq_locs[k], 0]).argmin() for k in range(len(freq_locs))])
                mask = np.any(np.isclose(base_time[freq_locers, None], fast_bins, atol=timebin_tol), axis=1)
                freq_locers = freq_locers[mask]

                all_hd_freq_locs = []

                for l in range(len(hd_bins) - 1):
                    hd_freq_locs = np.where((locations[freq_locs, 3] >= hd_bins[l]) & (locations[freq_locs, 3] < hd_bins[l + 1]))[0]
                    hd_freq_locs = freq_locs[hd_freq_locs]

                    hd_occ[i, j, l] = len(hd_freq_locs)

                    if hd_freq_locs.size < 5:
                        hd_rate_map[i, j, :, l] = np.nan
                        hd_rate_map_fold0[i, j, :, l] = np.nan
                        hd_rate_map_fold1[i, j, :, l] = np.nan
                        hd_occupancy_map[i, j, l] = 0
                    else:
                        all_hd_freq_locs.append(hd_freq_locs)

                        hd_freq_locers = np.array([np.abs(base_time - locations[hd_freq_locs[m], 0]).argmin() for m in range(len(hd_freq_locs))])
                        mask = np.any(np.isclose(base_time[hd_freq_locers, None], fast_bins, atol=timebin_tol), axis=1)
                        hd_freq_locers = hd_freq_locers[mask]

                        hd_rate_map[i, j, :, l] = np.mean(freq_maps[hd_freq_locers, 1, :], axis=0)
                        hd_occupancy_map[i, j, l] = len(hd_freq_locs)

                        np.random.shuffle(hd_freq_locers)
                        hd_rate_map_fold0[i, j, :, l] = np.mean(freq_maps[hd_freq_locers, 1, :][:len(hd_freq_locers) // 2], axis=0)
                        hd_rate_map_fold1[i, j, :, l] = np.mean(freq_maps[hd_freq_locers, 1, :][len(hd_freq_locers) // 2:], axis=0)

                if len(all_hd_freq_locs) == 0:
                    rate_maps[i, j, :] = np.nan
                else:
                    freq_locs = np.concatenate(all_hd_freq_locs)
                    freq_locers = np.array([np.abs(base_time - locations[freq_locs[k], 0]).argmin() for k in range(len(freq_locs))])
                    rate_maps[i, j, :] = np.nanmean(freq_maps[freq_locers, 1, :], axis=0)

    sigma = 6
    filter_range = spatial_bins - bin_count
    filter_x, filter_y = np.meshgrid(filter_range, filter_range)
    smooth_rate_maps = np.zeros(np.shape(rate_maps))

    for i in range(len(spatial_bins)):
        for j in range(len(spatial_bins)):
            gauss_filter = (np.exp(-(((filter_x - spatial_bins[j]) + bin_count) ** 2 + ((filter_y - spatial_bins[i]) + bin_count) ** 2)
                                        / (2 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
            smooth_rate_maps[i, j, :] = [np.nansum(gauss_filter * rate_maps[:, :, n]) for n in range(len(spike_mat))]

    smooth_rate_maps = adj_rate_maps(smooth_rate_maps, rate_maps, tau)

    np.save(f'{save_path}/rate_maps_seed_{seed}_hd_bin_count_{hd_bin_count}', rate_maps)
    np.save(f'{save_path}/smooth_rate_maps_seed_{seed}_hd_bin_count_{hd_bin_count}', smooth_rate_maps)
    np.save(f'{save_path}/hd_rate_map_seed_{seed}_hd_bin_count_{hd_bin_count}', hd_rate_map)
    np.save(f'{save_path}/hd_rate_map_seed_{seed}_fold0_hd_bin_count_{hd_bin_count}', hd_rate_map_fold0)
    np.save(f'{save_path}/hd_rate_map_seed_{seed}_fold1_hd_bin_count_{hd_bin_count}', hd_rate_map_fold1)
    np.save(f'{save_path}/hd_occupancy_map_seed_{seed}_hd_bin_count_{hd_bin_count}', hd_occupancy_map)

    np.save(f'{save_path}/locations', locations)
    np.save(f'{save_path}/time_bins', base_time)

runtime = (time.time() - start_time) / 60
print(f"Runtime: {runtime:.2f} minutes")
