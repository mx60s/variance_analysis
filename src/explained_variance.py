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

    out = np.divide(VV, UU, out=np.full_like(VV, np.nan), where=UU > 0)
    out[UU <= 0] = np.nan
    return out
    
    
def masked_zoom(arr, zoom_factor, order=1, *, clip_nonnegative=False):
    """Interpolate valid values; preserve signed activations unless explicitly clipped."""
    mask = np.isnan(arr)
    arr_filled = np.nan_to_num(arr, nan=0.0)
    zoomed = zoom(arr_filled, zoom=zoom_factor, order=order)
    if clip_nonnegative:
        zoomed = np.maximum(zoomed, 0.0)
    mask_zoomed = zoom(mask.astype(float), zoom=zoom_factor, order=0)
    zoomed[mask_zoomed > 0.5] = np.nan

    return zoomed


def get_splits(visits, min_visits=2, cap=False, seed=None):
    """Split finite visits without mutating input; occupancy is actual per-fold count.

    Visits have shape (s1, s2, visits, neurons). A shared validity mask across
    neurons is required because the returned occupancy arrays have no neuron axis.
    """
    visits = np.asarray(visits)
    if visits.ndim != 4 or min_visits < 2:
        raise ValueError("Expected 4-D visits and min_visits >= 2")
    valid = np.isfinite(visits)
    if visits.shape[-1] == 0:
        raise ValueError("At least one neuron is required")
    if not np.all(valid == valid[..., :1]):
        raise ValueError("Neuron-specific missingness needs neuron-specific occupancy")
    rng = np.random.default_rng(seed)
    shape = visits.shape[:2] + (visits.shape[-1],)
    total_mean, fold0_mean, fold1_mean = [np.full(shape, np.nan) for _ in range(3)]
    fold0_occ, fold1_occ = [np.zeros(visits.shape[:2], dtype=int) for _ in range(2)]
    for s1, s2 in np.ndindex(visits.shape[:2]):
        indices = np.flatnonzero(valid[s1, s2, :, 0])
        if len(indices) < min_visits:
            continue
        for n in range(visits.shape[-1]):
            values = visits[s1, s2, indices, n].copy()
            rng.shuffle(values)
            if cap:
                values = values[:min_visits]
            half = len(values) // 2
            fold0_mean[s1, s2, n] = values[:half].mean()
            fold1_mean[s1, s2, n] = values[half:].mean()
            total_mean[s1, s2, n] = values.mean()
        fold0_occ[s1, s2], fold1_occ[s1, s2] = half, len(values)-half
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
            min_a = min_b = np.inf
            max_a = max_b = -np.inf
            for k in range(n_bins):
                ai = fold0[i, k]
                bj = fold1[j, k]
                if ai == ai and bj == bj:  # NaN check
                    min_a = min(min_a, ai)
                    max_a = max(max_a, ai)
                    min_b = min(min_b, bj)
                    max_b = max(max_b, bj)
                    sum_a += ai
                    sum_b += bj
                    count += 1
            if count < 3 or min_a == max_a or min_b == max_b:
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
    """Return off-diagonal correlation and reliability (not 1 - correlation).

    occupancy is the total across the two folds; min_occ is a per-fold
    threshold under the equal-fold approximation. Undefined slices remain NaN.
    """
    if heatmap_fold0.shape != heatmap_fold1.shape or np.shape(occupancy) != heatmap_fold0.shape:
        raise ValueError("Fold maps and occupancy must have matching shapes")
    occ = np.asarray(occupancy) / 2
    mask = occ > min_occ
    fold0 = np.where(mask, heatmap_fold0, np.nan)
    fold1 = np.where(mask, heatmap_fold1, np.nan)

    fold0 = np.moveaxis(fold0, axis, 0).reshape(fold0.shape[axis], -1).copy()
    fold1 = np.moveaxis(fold1, axis, 0).reshape(fold1.shape[axis], -1).copy()

    corr_matrix = _corr_matrix(fold0, fold1)

    diag = np.diag(corr_matrix)
    reliability = np.mean(diag[np.isfinite(diag)]) if np.any(np.isfinite(diag)) else np.nan
    off_diag = ~np.eye(corr_matrix.shape[0], dtype=bool)
    values = corr_matrix[off_diag]
    variation = np.mean(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else np.nan

    if verbose:
        print(f"axis={axis}  reliability={reliability:.4f}  "
              f"variation={variation:.4f}")

    return variation, reliability


def smooth_location_hd(flat_map, sigma=(1, 1), grid_shape=None):
    """Smooth true X/Y neighbors and circular HD, then restore (locations, HD).

    sigma is (spatial, angular), in bins. Square grids can be inferred; rectangular
    grids require grid_shape. Flattening before spatial smoothing joins row ends
    to unrelated row starts and omits the second spatial dimension.
    """
    flat_map = np.asarray(flat_map)
    if flat_map.ndim != 2 or len(sigma) != 2:
        raise ValueError('Expected a (locations, HD) map and two smoothing widths')
    if grid_shape is None:
        side = int(np.sqrt(flat_map.shape[0]))
        grid_shape = (side, side)
    if np.prod(grid_shape) != flat_map.shape[0]:
        raise ValueError('grid_shape does not match the number of locations')
    grid = flat_map.reshape(*grid_shape, flat_map.shape[1])
    return nan_aware_gaussian_filter(grid, (sigma[0], sigma[0], sigma[1]), (2,)).reshape(flat_map.shape)
