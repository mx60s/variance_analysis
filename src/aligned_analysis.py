"""Explicit one-state-per-spike-bin analysis helpers.

Nearest tracking sample to each spike-bin center; no extrapolation across recording
edges or tracking gaps. Positive MD lag selects a later tracking sample, with NaN
at the end instead of wrapping the recording.
"""

import numpy as np


def align_tracking(positions, edges):
    t = positions[:, 0]
    centers = (edges[:-1] + edges[1:]) / 2
    if np.any(np.diff(t) <= 0):
        raise ValueError("Tracking timestamps must be strictly increasing")
    j = np.clip(np.searchsorted(t, centers), 1, len(t) - 1)
    left = j - 1
    idx = np.where(abs(centers - t[left]) <= abs(t[j] - centers), left, j)
    valid = (
        (centers >= t[0])
        & (centers <= t[-1])
        & (abs(t[idx] - centers) <= 1.5 * np.median(np.diff(t)))
    )
    out = positions[idx].copy()
    out[:, 0] = centers
    out[~valid, 1:] = np.nan
    return out, idx, valid


def average_in_bins(times, values, edges):
    idx = np.searchsorted(edges, times, side="right") - 1
    valid = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(values)
    count = np.bincount(idx[valid], minlength=len(edges) - 1)
    total = np.bincount(idx[valid], weights=values[valid], minlength=len(edges) - 1)
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def lag_without_wrap(values, lag):
    if not isinstance(lag, int) or lag < 0:
        raise ValueError("lag must be a nonnegative sample count")
    out = np.full_like(values, np.nan, dtype=float)
    if lag == 0:
        return np.array(values, copy=True)
    if lag < len(values):
        out[:-lag] = values[lag:]
    return out


def labels(values, edges):
    i = np.searchsorted(edges, values, side="right") - 1
    i[(~np.isfinite(values)) | (i < 0) | (i >= len(edges) - 1)] = -1
    return i


def grouped_maps(rates, row, col, nrows, ncols, valid, floor=5, seed=0):
    """Equal-size disjoint folds; discard at most one sample per occupied state."""
    rng = np.random.RandomState(seed)
    good = valid & (row >= 0) & (col >= 0)
    joint = row * ncols + col
    idx = np.flatnonzero(good)
    idx = idx[np.argsort(joint[idx], kind="stable")]
    split = np.split(idx, np.flatnonzero(np.diff(joint[idx])) + 1) if len(idx) else []
    shape = (nrows * ncols, rates.shape[1])
    a = np.full(shape, np.nan)
    b = a.copy()
    occ = np.zeros(nrows * ncols)
    for members in split:
        if len(members) < floor:
            continue
        members = members.copy()
        rng.shuffle(members)
        half = len(members) // 2
        j = joint[members[0]]
        a[j] = rates[members[:half]].mean(0)
        b[j] = rates[members[half : 2 * half]].mean(0)
        occ[j] = 2 * half
    return (
        a.reshape(nrows, ncols, -1),
        b.reshape(nrows, ncols, -1),
        occ.reshape(nrows, ncols),
    )


def poisson_argmax(rate, counts, tau, eps=1e-10, offset=0, chunk=256):
    """MAP under uniform state prior, equivalent to the existing decoder.

    Non-finite neuron rates are dropped from each state's likelihood; wholly
    unsupported states are excluded. Does not allocate/save full posteriors.
    """
    rate = np.asarray(rate) * tau
    valid = np.isfinite(rate).any(1)
    if not valid.any():
        return np.full(len(counts), -1, dtype=int)
    lograte = np.where(np.isfinite(rate), np.log(np.nan_to_num(rate) + offset + eps), 0)
    penalty = np.nansum(rate + offset, axis=1)
    result = []
    for start in range(0, len(counts), chunk):
        ll = counts[start : start + chunk] @ lograte.T - penalty
        ll[:, ~valid] = -np.inf
        result.append(np.argmax(ll, axis=1))
    return np.concatenate(result) if result else np.empty(0, dtype=int)


def coalesce_tracking_timestamps(positions):
    """Merge duplicate timestamps using XY means and a circular HD mean.

    Reject time reversals rather than silently sorting a reset clock.
    """
    positions = np.asarray(positions)
    if np.any(np.diff(positions[:, 0]) < 0):
        raise ValueError("Tracking clock reverses")
    times, first, count = np.unique(
        positions[:, 0], return_index=True, return_counts=True
    )
    if len(times) == len(positions):
        return positions.copy(), 0
    out = np.empty((len(times), 4))
    out[:, 0] = times
    for j in [1, 2]:
        out[:, j] = np.add.reduceat(positions[:, j], first) / count
    angles = np.deg2rad(positions[:, 3])
    s = np.add.reduceat(np.sin(angles), first)
    c = np.add.reduceat(np.cos(angles), first)
    out[:, 3] = np.rad2deg(np.arctan2(s, c)) % 360
    out[np.hypot(s, c) < 1e-12, 3] = np.nan
    return out, int(len(positions) - len(times))
