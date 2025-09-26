import numpy as np
from scipy.ndimage import gaussian_filter

def nan_aware_gaussian_filter(rate_map: np.ndarray, sigma: int) -> np.ndarray:
    V = rate_map.copy()
    isnan = np.isnan(V)
    V[isnan] = 0.0
    U = np.ones_like(rate_map, dtype=float)
    U[isnan] = 0.0

    VV = gaussian_filter(V, sigma=sigma, mode="nearest")
    UU = gaussian_filter(U, sigma=sigma, mode="nearest")

    out = VV / UU
    out[UU <= 0] = np.nan
    return out

def get_splits(
        visits, # s1_dim, s2_dim, n_max_visits, n_neurons
        min_visits=2,
        cap=False,
        seed=None
):
    """
    Randomly halves the samples in each bin → fold0 / fold1 splits.

    Returns
    -------
    fold0_mean, fold1_mean : mean spike count in each bin (nan if <min_visits)
    fold0_occ,  fold1_occ  : visit counts assigned to each split
    """
    rng = np.random.default_rng(seed)
    s1_bins = visits.shape[0]
    s2_bins = visits.shape[1]

    n_neurons = visits.shape[3]

    total_mean = np.full((s1_bins, s2_bins, n_neurons), np.nan)
    fold0_mean = np.full((s1_bins, s2_bins, n_neurons), np.nan)
    fold1_mean  = np.full((s1_bins, s2_bins, n_neurons), np.nan)
    fold0_occ  = np.full((s1_bins, s2_bins), 0)
    fold1_occ   = np.full((s1_bins, s2_bins), 0)

    for s1 in range(s1_bins):
        for s2 in range(s2_bins):
            for n in range(n_neurons):
                n_visits = np.count_nonzero(~np.isnan(visits[s1, s2, :, n]))
                if n_visits >= min_visits:
                    take_n = min_visits if cap else n_visits
                    visit_split = visits[s1, s2, :take_n, n]
                    rng.shuffle(visit_split)
                    half = len(visit_split) // 2
                    fold0, fold1 = visit_split[:half], visit_split[half:]
                    
                    fold0_mean[s1, s2, n] = np.nanmean(fold0)
                    fold1_mean[s1,  s2, n] = np.nanmean(fold1)
                    total_mean[s1, s2, n] = np.nanmean(visit_split)
                    fold0_occ[s1, s2] = half
                    fold1_occ[s1, s2] = half

    return total_mean, fold0_mean, fold1_mean, fold0_occ, fold1_occ

def quantify_variance(heatmap_fold1, heatmap_fold0, axis=0, verbose=False):
    """
    Quantify the variance due to a certain variable in a cross-validated manner.
    heatmaps are 2D, and are average spikes/sec across visits to each behavioral bin.
    Negative values indicate dominance of inherent variance over dimension-explained variance.
    """

    fold1_mean = np.nanmean(heatmap_fold1, axis=axis, keepdims=True)
    fold0_mean = np.nanmean(heatmap_fold0, axis=axis, keepdims=True)

    n_bins = heatmap_fold1.shape[axis]

    # Now compute the variance terms with occupancy weighting
    inter_var_fold1 = np.nansum((heatmap_fold1 - fold1_mean)**2)
    inter_var_fold0 = np.nansum((heatmap_fold0 - fold0_mean)**2)

    intra_var_all = np.nansum((heatmap_fold1 - heatmap_fold0)**2)
    intra_var_means = n_bins * np.nansum((fold1_mean - fold0_mean)**2)

    if verbose:
        print(f"axis={axis} sum((fold1 - fold1_mean)^2)={inter_var_fold1:.4f}")
        print(f"axis={axis} sum((fold0 - fold0_mean)^2)={inter_var_fold0:.4f}")
        print(f"axis={axis} sum((fold1 - fold0)^2)={intra_var_all:.4f}")
        print(f"axis={axis} n_bins * sum((fold1_mean - fold0_mean)^2)={intra_var_means:.4f}")
        print()
    
    estimator = 0.5 * (
        inter_var_fold1 +
        inter_var_fold0 -
        intra_var_all +
        intra_var_means
    )

    return estimator, inter_var_fold1, inter_var_fold0, intra_var_all, intra_var_means
