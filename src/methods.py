import numpy as np

def get_splits(
        visits, # s1_dim, s2_dim, n_max_visits, n_neurons
        min_visits=2,
        cap=False,
        seed=None
):
    """
    Randomly halves the samples in each bin → train / test splits.

    Returns
    -------
    train_mean, test_mean : mean spike count in each bin (nan if <min_visits)
    train_occ,  test_occ  : visit counts assigned to each split
    """
    rng = np.random.default_rng(seed)
    num_bins = visits.shape[0]

    # half = cap // 2
    n_neurons = visits.shape[3]

    total_mean = np.full((num_bins, num_bins, n_neurons), np.nan)
    train_mean = np.full((num_bins, num_bins, n_neurons), np.nan)
    test_mean  = np.full((num_bins, num_bins, n_neurons), np.nan)
    train_occ  = np.full((num_bins, num_bins), 0)
    test_occ   = np.full((num_bins, num_bins), 0)

    for s1 in range(visits.shape[0]):
        for s2 in range(visits.shape[1]):
            for n in range(visits.shape[3]):
                n_visits = np.count_nonzero(~np.isnan(visits[s1, s2, :, n]))
                if n_visits >= min_visits:
                    take_n = min_visits if cap else n_visits
                    visit_split = visits[s1, s2, :take_n, n]#[s1, s2, :cap, n]
                    rng.shuffle(visit_split)
                    half = len(visit_split) // 2
                    train, test = visit_split[:half], visit_split[half:]
                    
                    train_mean[s1, s2, n] = train.mean()
                    test_mean[s1,  s2, n] = test.mean()
                    total_mean[s1, s2, n] = visit_split.mean()
                    train_occ = half
                    test_occ = half
                else:
                    train_occ[s1, s2, n] = 0
                    test_occ[s1, s2, n] = 0

    return total_mean, train_mean, test_mean, train_occ, test_occ

def quantify_variance(heatmap_test, heatmap_train, axis=0, verbose=False):
    """
    Quantify the variance due to a certain variable in a cross-validated manner.
    heatmaps are 2D, and are average spikes/sec across visits to each behavioral bin.
    Negative values indicate dominance of inherent variance over dimension-explained variance.
    """

    test_mean = np.nanmean(heatmap_test, axis=axis, keepdims=True)
    train_mean = np.nanmean(heatmap_train, axis=axis, keepdims=True)

    n_bins = heatmap_test.shape[axis]

    # Now compute the variance terms with occupancy weighting
    inter_var_test = np.nansum((heatmap_test - test_mean)**2)
    inter_var_train = np.nansum((heatmap_train - train_mean)**2)

    intra_var_all = np.nansum((heatmap_test - heatmap_train)**2)
    intra_var_means = n_bins * np.nansum((test_mean - train_mean)**2)

    if verbose:
        print(f"axis={axis} sum((test - test_mean)^2)={inter_var_test:.4f}")
        print(f"axis={axis} sum((train - train_mean)^2)={inter_var_train:.4f}")
        print(f"axis={axis} sum((test - train)^2)={intra_var_all:.4f}")
        print(f"axis={axis} n_bins * sum((test_mean - train_mean)^2)={intra_var_means:.4f}")
        print()
    
    estimator = 0.5 * (
        inter_var_test +
        inter_var_train -
        intra_var_all +
        intra_var_means
    )
    
    return estimator, inter_var_test, inter_var_train, intra_var_all, intra_var_means
