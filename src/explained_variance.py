import numpy as np
from scipy.ndimage import gaussian_filter, zoom

def nan_aware_gaussian_filter(rate_map, sigma, circular_axes):
    V = rate_map.astype(float, copy=True)
    isnan = np.isnan(V)
    V[isnan] = 0.0

    U = np.ones_like(V, dtype=float)
    U[isnan] = 0.0

    ndim = V.ndim
    mode = ["nearest"] * ndim
    for ax in circular_axes:
        mode[ax] = "wrap"

    VV = gaussian_filter(V, sigma=sigma, mode=mode)
    UU = gaussian_filter(U, sigma=sigma, mode=mode)

    out = VV / UU
    out[UU <= 0] = np.nan
    return out
    
    
def masked_zoom(arr, zoom_factor, order=1):
    mask = np.isnan(arr)
    arr_filled = np.nan_to_num(arr, nan=0.0)
    zoomed = zoom(arr_filled, zoom=zoom_factor, order=order)
    zoomed = np.maximum(zoomed, 0.0)
    mask_zoomed = zoom(mask.astype(float), zoom=zoom_factor, order=0)
    zoomed[mask_zoomed > 0.5] = np.nan

    return zoomed


def get_splits(visits, min_visits=2, cap=False, seed=None):
  rng = np.random.default_rng(seed)
  s1_bins = visits.shape[0]
  s2_bins = visits.shape[1]
  n_neurons = visits.shape[3]

  total_mean = np.full((s1_bins, s2_bins, n_neurons), np.nan)
  fold0_mean = np.full((s1_bins, s2_bins, n_neurons), np.nan)
  fold1_mean = np.full((s1_bins, s2_bins, n_neurons), np.nan)
  fold0_occ = np.full((s1_bins, s2_bins), 0)
  fold1_occ = np.full((s1_bins, s2_bins), 0)

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
          fold1_mean[s1, s2, n] = np.nanmean(fold1)
          total_mean[s1, s2, n] = np.nanmean(visit_split)
          fold0_occ[s1, s2] = half
          fold1_occ[s1, s2] = half


  return total_mean, fold0_mean, fold1_mean, fold0_occ, fold1_occ


def quantify_variance(heatmap_fold0, heatmap_fold1, occupancy, axis=0, verbose=False):
    occ = occupancy / 2
    est_spikes0 = heatmap_fold0 * occ
    est_spikes1 = heatmap_fold1 * occ

    total_time   = np.nansum(occ, axis=axis, keepdims=True)
    total_spikes_0 = np.nansum(est_spikes0, axis=axis, keepdims=True)
    fold0_mean = np.divide(total_spikes_0, total_time,
                          out=np.full_like(total_spikes_0, np.nan),
                          where=total_time > 0)

    total_spikes_1 = np.nansum(est_spikes1, axis=axis, keepdims=True)
    fold1_mean = np.divide(total_spikes_1, total_time,
                          out=np.full_like(total_spikes_1, np.nan),
                          where=total_time > 0)

    n_bins = heatmap_fold1.shape[axis]

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
