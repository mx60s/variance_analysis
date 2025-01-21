import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def quantify_variance(heatmap_test, heatmap_train, axis):
    '''
    Quantify the variance due to a certain variable in a cross-validated manner.
    Dim=[0, 1]
    heatmaps should be 2D
    Negative values indicate dominance of inherent variance over dimension-explained variance.
    '''
    test_mean = np.nanmean(heatmap_test, axis=axis)
    test_mean_expanded = test_mean[np.newaxis, :] if axis == 0 else test_mean[:, np.newaxis]
    n_bins = heatmap_test.shape[axis]
    test_mean_expanded = np.repeat(test_mean_expanded, n_bins, axis=axis)

    train_mean = np.nanmean(heatmap_train, axis=axis)
    train_mean_expanded = train_mean[np.newaxis, :] if axis == 0 else train_mean[:, np.newaxis]
    train_mean_expanded = np.repeat(train_mean_expanded, n_bins, axis=axis)

    test_total_mean = np.nanmean(heatmap_test)
    train_total_mean = np.nanmean(heatmap_train)

    estimator = 0.5 * (np.nansum((heatmap_test - test_mean_expanded)**2)
                       + np.nansum((heatmap_train - train_mean_expanded)**2)
                       - np.nansum((heatmap_test - heatmap_train)**2)
                       + n_bins * ((test_total_mean - train_total_mean)**2))

    return estimator

def calculate_rate_maps(visits, split):
    np.random.seed(split)

    hd_rate_map_test = np.zeros((n_bins, n_bins, n_hd_bins, n_neurons))
    hd_rate_map_train = np.zeros((n_bins, n_bins, n_hd_bins, n_neurons))

    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_hd_bins):
                bin_visits = visits[i, j, :, k] # n_max_visits, n_neurons
                mask = ~np.isnan(bin_visits).any(axis=1) # or the issue could be here I guess
                bin_visits = bin_visits[mask].copy()
                np.random.shuffle(bin_visits)

                half = len(bin_visits) // 2
                hd_rate_map_test[i, j, k, :] = np.nanmean(bin_visits[:half], axis=0)
                hd_rate_map_train[i, j, k, :] = np.nanmean(bin_visits[half:], axis=0)

    return hd_rate_map_test, hd_rate_map_train

def process_split(split):
    hd_rate_map_test, hd_rate_map_train = calculate_rate_maps(visits, split)

    split_results = {
        'xy': np.zeros(n_neurons),
        'hd': np.zeros(n_neurons),
    }

    for idx in range(n_neurons):
        heatmap_test_flat = hd_rate_map_test[:, :, :, idx].reshape(-1, n_hd_bins)
        heatmap_train_flat = hd_rate_map_train[:, :, :, idx].reshape(-1, n_hd_bins)

        var_xy = quantify_variance(heatmap_test_flat, heatmap_train_flat, axis=0)
        var_hd = quantify_variance(heatmap_test_flat, heatmap_train_flat, axis=1)

        split_results['hd'][idx] = var_hd
        split_results['xy'][idx] = var_xy

    return split_results

if __name__ == "__main__":
    print("Loading visits (big file)")
    try:
        visits = np.load('/home/mag/Space/results/pfeiffer/raw_hd_rate_maps_hd_bin_count_17.npy')
        print(f"File loaded with shape: {visits.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

    # check visit calculation next
    n_bins, _, n_max_visits, n_hd_bins, n_neurons = visits.shape
    n_splits = 100

    final_results = {
        'xy': np.zeros((n_splits, n_neurons)),
        'hd': np.zeros((n_splits, n_neurons)),
    }

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_split, split) for split in range(n_splits)]
        for split, future in enumerate(tqdm(futures, desc="Calculating V_signal")):
            split_results = future.result()
            final_results['hd'][split] = split_results['hd']
            final_results['xy'][split] = split_results['xy']

    np.save(f'/home/mag/Space/results/pfeiffer/hd_variances', final_results['hd'])
    np.save(f'/home/mag/Space/results/pfeiffer/xy_variances', final_results['xy'])

    print("Done <3")
