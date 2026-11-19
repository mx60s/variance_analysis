"""Gap-aware offline movement estimates from centered local linear XY fits.

Directions use the curated NWB convention: clockwise from positive Y. Speed and
direction derive from the same velocity vector. No velocity window bridges a
gap or nonfinite position, and estimates require a complete centered window.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def estimate_movement(positions, window_s=0.2, gap_factor=1.5, min_speed=1e-6):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 3 or len(positions) < 3:
        raise ValueError("Expected at least three timestamped XY positions")
    t, xy = positions[:, 0], positions[:, 1:3]
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if window_s <= 0 or gap_factor <= 1:
        raise ValueError("Positive window and gap_factor > 1 required")
    dt = float(np.median(np.diff(t)))
    half = max(1, int(round(window_s / (2 * dt))))
    width = 2 * half + 1
    finite = np.isfinite(xy).all(1)
    breaks = (np.diff(t) > gap_factor * dt) | ~finite[:-1] | ~finite[1:]
    boundaries = np.r_[0, np.flatnonzero(breaks) + 1, len(t)]
    velocity = np.full((len(t), 2), np.nan)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if hi - lo < width or not finite[lo:hi].all():
            continue
        # Work in blocks to bound temporary memory on long recordings. Centering
        # timestamps and coordinates prevents loss of precision at large offsets.
        tw = sliding_window_view(t[lo:hi], width)
        xw = sliding_window_view(xy[lo:hi], width, axis=0)
        for start in range(0, len(tw), 8192):
            stop = min(start + 8192, len(tw))
            tr = tw[start:stop] - tw[start:stop, half, None]
            tr -= tr.mean(1, keepdims=True)
            xr = xw[start:stop] - xw[start:stop, :, half, None]
            denom = (tr * tr).sum(1)
            velocity[lo + half + start : lo + half + stop] = (
                (xr * tr[:, None, :]).sum(2) / denom[:, None]
            )
    speed = np.linalg.norm(velocity, axis=1)
    valid = np.isfinite(speed) & (speed > min_speed)
    direction = np.full(len(t), np.nan)
    direction[valid] = np.degrees(np.arctan2(velocity[valid, 0], velocity[valid, 1])) % 360
    return dict(
        velocity=velocity,
        speed=speed,
        direction=direction,
        valid=valid,
        window_samples=width,
        window_duration_s=2 * half * dt,
        tracking_dt=dt,
        gap_count=int(np.sum(np.diff(t) > gap_factor * dt)),
        excluded_incomplete_window=int(np.sum(~np.isfinite(speed))),
    )
