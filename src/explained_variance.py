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


from numba import njit

@njit
def _corr_matrix(fold0, fold1):
    """Compute pairwise cross-fold correlation matrix with per-pair validity."""
    n_slices = fold0.shape[0]
    n_bins = fold0.shape[1]
    corr_matrix = np.full((n_slices, n_slices), np.nan)

    for i in range(n_slices):
        for j in range(n_slices):
            # Find mutually valid bins
            sum_a = 0.0
            sum_b = 0.0
            count = 0
            for k in range(n_bins):
                ai = fold0[i, k]
                bj = fold1[j, k]
                if ai == ai and bj == bj:  # NaN check
                    sum_a += ai
                    sum_b += bj
                    count += 1
            if count < 3:
                continue

            mean_a = sum_a / count
            mean_b = sum_b / count

            num = 0.0
            ss_a = 0.0
            ss_b = 0.0
            for k in range(n_bins):
                ai = fold0[i, k]
                bj = fold1[j, k]
                if ai == ai and bj == bj:
                    da = ai - mean_a
                    db = bj - mean_b
                    num += da * db
                    ss_a += da * da
                    ss_b += db * db

            denom = (ss_a * ss_b) ** 0.5
            if denom > 0:
                corr_matrix[i, j] = num / denom

    return corr_matrix


def quantify_variance_corr(heatmap_fold0, heatmap_fold1, occupancy, axis=0, min_occ=0, verbose=False):
    occ = occupancy / 2
    mask = occ > min_occ
    fold0 = np.where(mask, heatmap_fold0, np.nan)
    fold1 = np.where(mask, heatmap_fold1, np.nan)

    fold0 = np.moveaxis(fold0, axis, 0).reshape(fold0.shape[axis], -1).copy()
    fold1 = np.moveaxis(fold1, axis, 0).reshape(fold1.shape[axis], -1).copy()

    corr_matrix = _corr_matrix(fold0, fold1)

    reliability = np.nanmean(np.diag(corr_matrix))
    off_diag = ~np.eye(corr_matrix.shape[0], dtype=bool)
    variation = np.nanmean(corr_matrix[off_diag])

    if verbose:
        print(f"axis={axis}  reliability={reliability:.4f}  "
              f"variation={variation:.4f}")

    return variation, reliability
