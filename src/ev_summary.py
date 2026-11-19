"""Fast pairwise-valid correlation summaries for regeneration.

Algebraically equivalent to explained_variance._corr_matrix, without retaining
its matrix when every bin is finite. Sparse support uses pairwise sufficient
statistics; per-row centering improves numerical conditioning.
"""

import numpy as np
from .explained_variance import _corr_matrix
from numba import njit


@njit
def _repair_pairs(a, b, pairs, c):
    for q in range(len(pairs)):
        i, j = pairs[q]
        c[i, j] = _corr_matrix(a[i : i + 1], b[j : j + 1])[0, 0]


def correlation_summary(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Expected matching 2-D maps")
    original_a, original_b = a.copy(), b.copy()
    ma, mb = np.isfinite(a), np.isfinite(b)
    if a.shape[1] < 3:
        return np.nan, np.nan
    if ma.all() and mb.all():
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        va, vb = np.ptp(original_a, axis=1) > 0, np.ptp(original_b, axis=1) > 0
        a = np.divide(a, na[:, None], out=np.zeros_like(a), where=va[:, None])
        b = np.divide(b, nb[:, None], out=np.zeros_like(b), where=vb[:, None])
        valid_diag = va & vb
        diag = (a * b).sum(axis=1)
        nd = valid_diag.sum()
        count = va.sum() * vb.sum() - nd
        off = (
            (np.dot(a.sum(axis=0), b.sum(axis=0)) - diag[valid_diag].sum()) / count
            if count
            else np.nan
        )
        rel = diag[valid_diag].mean() if nd else np.nan
        return off, rel
    # Center each slice before evaluating pairwise-complete sums.
    sa = np.divide(
        np.where(ma, a, 0).sum(1),
        ma.sum(1),
        out=np.zeros(a.shape[0]),
        where=ma.sum(1) > 0,
    )
    sb = np.divide(
        np.where(mb, b, 0).sum(1),
        mb.sum(1),
        out=np.zeros(b.shape[0]),
        where=mb.sum(1) > 0,
    )
    a, b = np.where(ma, a - sa[:, None], 0), np.where(mb, b - sb[:, None], 0)
    ma, mb = ma.astype(float), mb.astype(float)
    n = ma @ mb.T
    x, y = a @ mb.T, ma @ b.T
    with np.errstate(invalid="ignore", divide="ignore"):
        second_a, second_b = (a * a) @ mb.T, ma @ (b * b).T
        va = second_a - x * x / n
        vb = second_b - y * y / n
        cov = a @ b.T - x * y / n
        denom = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
        c = np.divide(
            cov,
            denom,
            out=np.full_like(cov, np.nan),
            where=(n >= 3) & (va > 0) & (vb > 0),
        )
    # Numerically ambiguous, nearly constant pair support: use the reference.
    ambiguous = (n >= 3) & (
        (va <= 1e-6 * np.maximum(second_a, 1e-30))
        | (vb <= 1e-6 * np.maximum(second_b, 1e-30))
    )
    varying_a = np.max(np.where(ma, original_a, -np.inf), axis=1) > np.min(
        np.where(ma, original_a, np.inf), axis=1
    )
    varying_b = np.max(np.where(mb, original_b, -np.inf), axis=1) > np.min(
        np.where(mb, original_b, np.inf), axis=1
    )
    valid_pairs = varying_a[:, None] & varying_b[None, :]
    c[~valid_pairs] = np.nan
    ambiguous &= valid_pairs
    # Sufficient-statistic subtraction loses precision on low-variance pair
    # support. Repair those pairs with the centered two-pass reference, using
    # the original values (not the globally centered intermediates).
    aa, bb = np.where(ma, original_a, np.nan), np.where(mb, original_b, np.nan)
    _repair_pairs(aa, bb, np.argwhere(ambiguous), c)
    diag = np.diag(c)
    good = np.isfinite(diag)
    rel = diag[good].mean() if good.any() else np.nan
    np.fill_diagonal(c, np.nan)
    good = np.isfinite(c)
    return c[good].mean() if good.any() else np.nan, rel


def quantify_fast(f0, f1, occupancy, axis=0, min_occ=0):
    mask = np.asarray(occupancy) / 2 > min_occ
    a, b = np.where(mask, f0, np.nan), np.where(mask, f1, np.nan)
    return correlation_summary(np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0))
