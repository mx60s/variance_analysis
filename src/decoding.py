"""Bayesian and chance-level decoding, plus error evaluation, shared by the Pfeiffer and Dusz pipelines."""
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# cross-validation and preprocessing

def make_cv_masks(n_points: int, cv_count: int) -> List[np.ndarray]:
    """Contiguous folds; mask is True for training bins, False for the held-out fold."""
    folds = np.array_split(np.arange(n_points), cv_count)
    cv_masks: List[np.ndarray] = []
    for cv in range(cv_count):
        mask = np.ones(n_points, dtype=bool)
        mask[folds[cv]] = False
        cv_masks.append(mask)
    return cv_masks


def test_spike_counts(spike_rates: np.ndarray, test_idx: np.ndarray, tau: float,
                      smooth_spikes_sigma_s: float = 0.0) -> np.ndarray:
    sc = (spike_rates[test_idx, :] * tau).astype(float)
    if smooth_spikes_sigma_s and smooth_spikes_sigma_s > 0:
        sigma_bins_t = max(1e-9, smooth_spikes_sigma_s / tau)
        sc = gaussian_filter1d(sc, sigma=sigma_bins_t, axis=0, mode="nearest")
    return sc


def gaussian_filter_hd(rate_curve: np.ndarray, smooth_sigma: Optional[float]) -> np.ndarray:
    """NaN-aware circular 1D Gaussian smoothing across head/MD bins."""
    if smooth_sigma is None or smooth_sigma <= 0:
        return rate_curve
    V = rate_curve.copy()
    isnan = np.isnan(V)
    V[isnan] = 0.0
    U = np.ones_like(rate_curve, dtype=float)
    U[isnan] = 0.0
    VV = gaussian_filter1d(V, sigma=smooth_sigma, axis=0, mode="wrap")
    UU = gaussian_filter1d(U, sigma=smooth_sigma, axis=0, mode="wrap")
    with np.errstate(invalid='ignore', divide='ignore'):
        out = VV / UU
    out[UU <= 0] = np.nan
    return out

# decoding

def hd_bayesian_decoding(
    rate_map_hd: np.ndarray,
    spike_counts: np.ndarray,
    tau: float,
    hd_bins: np.ndarray,
    avg_vel: np.ndarray,
    vel_threshold: float = 0.0,
    eps: float = 1e-10,
    rate_offset: float = 0.0,
    progress: bool = False,
):
    """Poisson decoder over angle bins. Non-finite rates are zeroed; rate_offset is added to every
    expected count (Dusz uses 1e-12, Pfeiffer 0)."""
    T = spike_counts.shape[0]
    H, N = rate_map_hd.shape

    decoded_hd = np.full(T, np.nan)
    posterior_all: List[np.ndarray] = []

    rate_tau = rate_map_hd * tau
    valid_h = np.any(np.isfinite(rate_tau), axis=1)
    rate_tau[~np.isfinite(rate_tau)] = 0.0
    if rate_offset:
        rate_tau += rate_offset

    for t in tqdm(range(T), desc="Decoding HD time bins", disable=not progress):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((H,), np.nan))
            continue

        k = spike_counts[t][None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k * np.log(rate_tau + eps) - rate_tau, axis=1)
        log_like[~valid_h] = -np.inf

        if not np.any(np.isfinite(log_like)):
            posterior_all.append(np.full((H,), np.nan))
            continue

        m = np.nanmax(log_like)
        post = np.exp(log_like - m)
        post_sum = np.nansum(post)
        if post_sum <= 0 or not np.isfinite(post_sum):
            posterior_all.append(np.full((H,), np.nan))
            continue
        post /= post_sum
        posterior_all.append(post)

        h_idx = int(np.nanargmax(post))
        decoded_hd[t] = 0.5 * (hd_bins[h_idx] + hd_bins[h_idx + 1])

    return decoded_hd, posterior_all


def xy_bayesian_decoding(
    rate_map_xy: np.ndarray,
    spike_counts: np.ndarray,
    tau: float,
    avg_vel: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    vel_threshold: float = 0.0,
    eps: float = 1e-10,
    nan_to_zero: bool = True,
    progress: bool = False,
):
    """Poisson decoder over XY bins. With nan_to_zero=False (Dusz), NaN rates are left in and
    dropped neuron-wise by nansum instead of being treated as zero rate (Pfeiffer)."""
    T, N = spike_counts.shape
    Y, X, N_map = rate_map_xy.shape
    assert N_map == N, "rate_map_xy last dim must match spike_counts."

    decoded_xy = np.full((T, 2), np.nan)
    posterior_all: List[np.ndarray] = []

    rate_tau_flat = (rate_map_xy * tau).reshape(Y * X, N)
    if nan_to_zero:
        rate_tau_flat = np.nan_to_num(rate_tau_flat, nan=0.0)

    for t in tqdm(range(T), desc="Decoding XY time bins", disable=not progress):
        if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
            posterior_all.append(np.full((Y, X), np.nan))
            continue

        k = spike_counts[t]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_like = np.nansum(k[None, :] * np.log(rate_tau_flat + eps) - rate_tau_flat, axis=1)

        if np.all(~np.isfinite(log_like)):
            posterior_all.append(np.full((Y, X), np.nan))
            continue

        m = np.nanmax(log_like)
        post = np.exp(log_like - m)
        s = np.nansum(post)
        if s <= 0 or not np.isfinite(s):
            posterior_all.append(np.full((Y, X), np.nan))
            continue
        post = (post / s).reshape(Y, X)
        posterior_all.append(post)

        max_idx_flat = int(np.nanargmax(post))
        iy, ix = divmod(max_idx_flat, X)
        decoded_xy[t, 0] = 0.5 * (x_edges[ix] + x_edges[ix + 1])
        decoded_xy[t, 1] = 0.5 * (y_edges[iy] + y_edges[iy + 1])

    return decoded_xy, posterior_all


# ==================== CHANCE-LEVEL DECODING ====================
# These functions decode based purely on occupancy (the prior), ignoring neural activity.
# This provides a baseline: "how well would we do just guessing based on where the animal spends time?"


def hd_chance_decoding(
    occupancy: np.ndarray,
    n_test_bins: int,
    hd_edges: np.ndarray,
    avg_vel: np.ndarray,
    vel_threshold: float = 0.0,
    mode: str = "mode",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chance-level decoding for HD (or MD) based on occupancy distribution.

    Parameters
    ----------
    occupancy : np.ndarray
        Occupancy counts per HD bin (from training data).
    n_test_bins : int
        Number of test time bins to decode.
    hd_edges : np.ndarray
        Bin edges for HD.
    avg_vel : np.ndarray
        Average velocity per test bin.
    vel_threshold : float
        Velocity threshold for valid bins.
    mode : str
        'mode' - always predict most occupied bin (deterministic baseline)
        'sample' - sample from occupancy distribution (stochastic baseline)
    rng : np.random.Generator, optional
        Random generator for 'sample' mode.

    Returns
    -------
    decoded_hd : np.ndarray
        Chance-decoded HD values.
    occupancy_prior : np.ndarray
        Normalized occupancy distribution used as prior.
    """
    # Normalize occupancy to get prior P(HD)
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total_occ = occ.sum()
    if total_occ > 0:
        prior = occ / total_occ
    else:
        prior = np.ones_like(occ) / len(occ)  # uniform if no occupancy

    decoded_hd = np.full(n_test_bins, np.nan)

    if mode == "mode":
        # Always predict the most occupied bin
        mode_idx = int(np.argmax(prior))
        mode_hd = 0.5 * (hd_edges[mode_idx] + hd_edges[mode_idx + 1])
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            decoded_hd[t] = mode_hd
    elif mode == "sample":
        # Sample from occupancy distribution
        if rng is None:
            rng = np.random.default_rng()
        bin_centers = 0.5 * (hd_edges[:-1] + hd_edges[1:])
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            idx = rng.choice(len(prior), p=prior)
            decoded_hd[t] = bin_centers[idx]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mode' or 'sample'.")

    return decoded_hd, prior


def xy_chance_decoding(
    occupancy: np.ndarray,
    n_test_bins: int,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    avg_vel: np.ndarray,
    vel_threshold: float = 0.0,
    mode: str = "mode",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chance-level decoding for XY based on occupancy distribution.

    Parameters
    ----------
    occupancy : np.ndarray
        Occupancy counts per XY bin (shape: n_y_bins x n_x_bins, from training data).
    n_test_bins : int
        Number of test time bins to decode.
    x_edges, y_edges : np.ndarray
        Bin edges for X and Y.
    avg_vel : np.ndarray
        Average velocity per test bin.
    vel_threshold : float
        Velocity threshold for valid bins.
    mode : str
        'mode' - always predict most occupied bin (deterministic baseline)
        'sample' - sample from occupancy distribution (stochastic baseline)
    rng : np.random.Generator, optional
        Random generator for 'sample' mode.

    Returns
    -------
    decoded_xy : np.ndarray
        Chance-decoded XY values (shape: n_test_bins x 2).
    occupancy_prior : np.ndarray
        Normalized occupancy distribution used as prior.
    """
    Y, X = occupancy.shape
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total_occ = occ.sum()
    if total_occ > 0:
        prior = occ / total_occ
    else:
        prior = np.ones_like(occ) / occ.size

    decoded_xy = np.full((n_test_bins, 2), np.nan)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    if mode == "mode":
        mode_idx_flat = int(np.argmax(prior))
        iy, ix = divmod(mode_idx_flat, X)
        mode_x = x_centers[ix]
        mode_y = y_centers[iy]
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            decoded_xy[t, 0] = mode_x
            decoded_xy[t, 1] = mode_y
    elif mode == "sample":
        if rng is None:
            rng = np.random.default_rng()
        prior_flat = prior.ravel()
        for t in range(n_test_bins):
            if np.isnan(avg_vel[t]) or avg_vel[t] < vel_threshold:
                continue
            idx = rng.choice(len(prior_flat), p=prior_flat)
            iy, ix = divmod(idx, X)
            decoded_xy[t, 0] = x_centers[ix]
            decoded_xy[t, 1] = y_centers[iy]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mode' or 'sample'.")

    return decoded_xy, prior


def compute_chance_error_analytical_hd(
    occupancy: np.ndarray,
    hd_edges: np.ndarray,
) -> Dict[str, float]:
    """
    Compute expected error analytically for HD chance decoding.

    This computes E[|error|] = sum_i P(true=i) * |angle_diff(i, mode)|
    For mode decoding: always guess the most occupied bin.

    Returns expected mean absolute error under chance model.
    """
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total = occ.sum()
    if total <= 0:
        return {'expected_mae_deg': np.nan, 'mode_bin_probability': np.nan}

    prior = occ / total
    n_bins = len(prior)
    bin_centers = 0.5 * (hd_edges[:-1] + hd_edges[1:])

    # Mode prediction
    mode_idx = int(np.argmax(prior))
    mode_angle = bin_centers[mode_idx]

    # Expected error: sum over true positions weighted by their probability
    expected_error = 0.0
    for i in range(n_bins):
        err = np.abs(angular_distance_deg(np.array([bin_centers[i]]), np.array([mode_angle]))[0])
        expected_error += prior[i] * err

    return {
        'expected_mae_deg': float(expected_error),
        'mode_bin_probability': float(prior[mode_idx]),
        'mode_bin_center_deg': float(mode_angle),
    }


def compute_chance_error_analytical_xy(
    occupancy: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    key_suffix: str = "_cm",
) -> Dict[str, float]:
    """
    Compute expected error analytically for XY chance decoding.

    Returns expected mean absolute error (Euclidean distance) under chance model,
    stored as f'expected_mae{key_suffix}'.
    """
    Y, X = occupancy.shape
    occ = occupancy.copy()
    occ[~np.isfinite(occ)] = 0.0
    total = occ.sum()
    if total <= 0:
        return {f'expected_mae{key_suffix}': np.nan, 'mode_bin_probability': np.nan}

    prior = occ / total
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    mode_idx_flat = int(np.argmax(prior))
    iy_mode, ix_mode = divmod(mode_idx_flat, X)
    mode_x = x_centers[ix_mode]
    mode_y = y_centers[iy_mode]

    expected_error = 0.0
    for iy in range(Y):
        for ix in range(X):
            dist = np.sqrt((x_centers[ix] - mode_x)**2 + (y_centers[iy] - mode_y)**2)
            expected_error += prior[iy, ix] * dist

    return {
        f'expected_mae{key_suffix}': float(expected_error),
        'mode_bin_probability': float(prior[iy_mode, ix_mode]),
        'mode_x': float(mode_x),
        'mode_y': float(mode_y),
    }


def compute_no_prior_chance_hd() -> Dict[str, float]:
    """
    Compute expected error for HD/MD decoding with no prior (uniform random guessing).

    If we guess uniformly at random from [0, 360), the expected absolute
    circular error is exactly 90 degrees.

    Derivation: The absolute circular error is uniformly distributed on [0, 180],
    so E[|error|] = 90.
    """
    return {
        'no_prior_expected_mae_deg': 90.0,
        'description': 'Uniform random guessing over [0, 360) degrees'
    }


def compute_no_prior_chance_xy(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> Dict[str, float]:
    """
    Compute expected error for XY decoding with no prior (uniform random guessing).

    For a square environment of side L, the expected Euclidean distance between
    two uniformly random points is approximately 0.5214 * L.

    Exact formula: L * (2 + sqrt(2) + 5*log(1+sqrt(2))) / 15
    """
    L_x = x_edges[-1] - x_edges[0]
    L_y = y_edges[-1] - y_edges[0]

    # For a square, use the exact formula
    # For a rectangle, use average of the two dimensions as approximation
    L = (L_x + L_y) / 2.0

    # Exact coefficient for square: (2 + sqrt(2) + 5*ln(1+sqrt(2))) / 15 ≈ 0.5214
    sqrt2 = math.sqrt(2)
    coeff = (2 + sqrt2 + 5 * math.log(1 + sqrt2)) / 15

    expected_error = coeff * L

    return {
        'no_prior_expected_mae_cm': float(expected_error),
        'environment_size_x_cm': float(L_x),
        'environment_size_y_cm': float(L_y),
        'description': 'Uniform random guessing over environment'
    }

# error analysis helpers

def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angular distance in degrees, wrapped to [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    rad_unwrapped = np.unwrap(rad, discont=np.pi)
    return np.rad2deg(rad_unwrapped)


def interp_circular_deg(t_src: np.ndarray, t_tgt: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    t_src = t_src[order]
    angles_deg = angles_deg[order]
    ang_unwrapped = unwrap_deg(angles_deg)
    ang_interp_unwrapped = np.interp(t_tgt, t_src, ang_unwrapped)
    ang_interp = np.mod(ang_interp_unwrapped, 360.0)
    return ang_interp


def interp_linear_series(t_src: np.ndarray, t_tgt: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    order = np.argsort(t_src)
    return np.interp(t_tgt, t_src[order], v_src[order])


def _fast_time_centers(time_centers: np.ndarray, velocity_series: np.ndarray, velocity_filter: float,
                       window: float) -> np.ndarray:
    """True where the mean speed within +/- window of the time center is >= velocity_filter."""
    v_times = velocity_series[:, 0]
    v_vals = velocity_series[:, 1]
    velocity_interp = np.full(len(time_centers), np.nan)
    for i, tc in enumerate(time_centers):
        mask = (v_times >= tc - window) & (v_times <= tc + window)
        if np.any(mask):
            velocity_interp[i] = float(np.nanmean(v_vals[mask]))
    return (velocity_interp >= velocity_filter) & (~np.isnan(velocity_interp))


def compute_ground_truth_hd(positions: np.ndarray, time_centers: np.ndarray, velocity_series: np.ndarray,
                            velocity_filter: float, window: float = 0.06,
                            angles_deg: Optional[np.ndarray] = None) -> np.ndarray:
    """HD at the time centers (or another per-sample angle series, e.g. MD), NaN where too slow."""
    angles = positions[:, 3] if angles_deg is None else angles_deg
    hd_interp = interp_circular_deg(positions[:, 0], time_centers, angles)
    valid_velocity = _fast_time_centers(time_centers, velocity_series, velocity_filter, window)
    return np.where(valid_velocity, hd_interp, np.nan)


def compute_ground_truth_xy(positions: np.ndarray, time_centers: np.ndarray, velocity_series: np.ndarray,
                            velocity_filter: float, window: float = 0.06) -> np.ndarray:
    t = positions[:, 0]
    x_interp = interp_linear_series(t, time_centers, positions[:, 1])
    y_interp = interp_linear_series(t, time_centers, positions[:, 2])
    valid = _fast_time_centers(time_centers, velocity_series, velocity_filter, window)
    gt_xy = np.column_stack([x_interp, y_interp])
    gt_xy[~valid, :] = np.nan
    return gt_xy


def evaluate_errors_hd(decoded_hd: np.ndarray, time_centers: np.ndarray, positions: np.ndarray,
                       velocity_series: np.ndarray, vel_threshold: float,
                       angles_deg: Optional[np.ndarray] = None, name: str = "hd",
                       std_of_abs: bool = True) -> Dict[str, Any]:
    """Signed circular errors against compute_ground_truth_hd. Stats keys are prefixed with name;
    the std is taken over |error| if std_of_abs (Pfeiffer) and over the signed error otherwise (Dusz)."""
    gt = compute_ground_truth_hd(positions, time_centers, velocity_series, vel_threshold, angles_deg=angles_deg)
    err = angular_distance_deg(decoded_hd, gt)
    err_abs = np.abs(err)
    total = int(err.size)
    valid = int(np.sum(~np.isnan(err)))
    stats = {
        f'{name}_mean_abs_error_deg': float(np.nanmean(err_abs)),
        f'{name}_median_abs_error_deg': float(np.nanmedian(err_abs)),
        f'{name}_std_abs_error_deg': float(np.nanstd(err_abs if std_of_abs else err)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors_deg': err}


def evaluate_errors_xy(decoded_xy: np.ndarray, time_centers: np.ndarray, positions: np.ndarray,
                       velocity_series: np.ndarray, vel_threshold: float,
                       key_suffix: str = "_cm") -> Dict[str, Any]:
    gt_xy = compute_ground_truth_xy(positions, time_centers, velocity_series, vel_threshold)
    diffs = decoded_xy - gt_xy  # (T,2)
    dists = np.sqrt(np.nansum(diffs**2, axis=1))
    total = int(dists.size)
    valid = int(np.sum(~np.isnan(dists)))
    stats = {
        f'xy_mean_abs_error{key_suffix}': float(np.nanmean(dists)),
        f'xy_median_abs_error{key_suffix}': float(np.nanmedian(dists)),
        f'xy_std_abs_error{key_suffix}': float(np.nanstd(dists)),
        'total_samples': total,
        'valid_samples': valid,
        'success_rate_pct': float(100.0 * valid / max(total, 1)),
    }
    return {'stats': stats, 'errors' + key_suffix: dists}
