from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Tuple
from copy import deepcopy
from scipy import ndimage

'''
Everything marked with _old is the Luo et al implementation.
_new is mine, and fully implements the Tanni methodology for place cell classification.
'''

def _is_dead_neuron(heatmap):
    """
    Given a neuron's 2D heatmap, check if it is a dead neuron.
    """
    # neuron is dead if less than 1% of the heatmap is active
    return np.sum(heatmap > 0) < 0.01 * heatmap.shape[0] * heatmap.shape[1]

def _compute_single_heatmap_border_scores_old(
    activation_map: np.ndarray, db: int = 3
) -> float:
    """Banino‑style border score (2018)."""
    num_bins = activation_map.shape[0]
    c = np.mean([
        activation_map[i, j]
        for i in range(db, num_bins - db)
        for j in range(db, num_bins - db)
    ])
    wall_scores = []
    for i in range(4):
        if i == 0:
            activations = activation_map[:db, :]
        elif i == 1:
            activations = activation_map[:, -db:]
        elif i == 2:
            activations = activation_map[-db:, :]
        else:
            activations = activation_map[:, :db]
        bi = np.mean(activations)
        wall_scores.append((bi - c) / (bi + c))
    return float(np.max(wall_scores))

# border score method is the same
_compute_single_heatmap_border_scores_new = _compute_single_heatmap_border_scores_old 


def _compute_single_heatmap_directional_scores_old(
    activation_maps: np.ndarray,
) -> Tuple[float, np.ndarray]:
    num_bins = activation_maps.shape[1]
    alphas = np.linspace(0, 2 * np.pi, num=num_bins, endpoint=False)
    betas = np.mean(activation_maps, axis=0)
    coords = np.stack([betas * np.cos(alphas), betas * np.sin(alphas)], axis=-1)
    r_normed = np.sum(coords, axis=0) / np.sum(betas)
    mean_vector_length = float(np.linalg.norm(r_normed))
    return mean_vector_length, np.linalg.norm(coords, axis=-1)

def _compute_single_heatmap_directional_scores_new(
    activation_maps: np.ndarray,
) -> Tuple[float, np.ndarray]:
    rng = np.ptp(activation_maps)
    normalised = np.zeros_like(activation_maps) if rng == 0 else (
        (activation_maps - np.min(activation_maps)) / rng
    )
    return _compute_single_heatmap_directional_scores_old(normalised)


def _compute_single_heatmap_fields_info_old(
    heatmap: np.ndarray,
    pixel_min_threshold: int,
    pixel_max_threshold: int,
):
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap_gray = (heatmap_norm * 255).astype(np.uint8)
    
    activity_thr = np.mean(heatmap_gray)
    _, thresh = cv2.threshold(heatmap_gray, activity_thr, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    mask = (stats[:, cv2.CC_STAT_AREA] >= pixel_min_threshold) & (
        stats[:, cv2.CC_STAT_AREA] <= pixel_max_threshold
    )
    mask[0] = False  # drop background
    filtered_stats = stats[mask]
    filtered_labels = np.where(np.isin(labels, np.nonzero(mask)[0]), labels, 0)
    num_clusters = np.array([filtered_stats.shape[0]])
    num_pixels_in_clusters = filtered_stats[:, cv2.CC_STAT_AREA]
    max_vals, mean_vals, var_vals = [], [], []
    
    for label in np.unique(filtered_labels):
        if label == 0:
            continue
        vals = heatmap[filtered_labels == label]
        max_vals.append(np.round(np.max(vals), 1))
        mean_vals.append(np.round(np.mean(vals), 1))
        var_vals.append(np.round(np.var(vals), 1))
        
    if num_clusters[0] == 0:
        num_pixels_in_clusters = max_vals = mean_vals = var_vals = np.array([0])
        
    return (
        num_clusters,
        num_pixels_in_clusters,
        np.asarray(max_vals),
        np.asarray(mean_vals),
        np.asarray(var_vals)
    )

# --------------------------
# supporting methods for the new, total Tanni place cell classification
# --------------------------

def _spatial_correlation(map_1, map_2, *, min_included_value=0.01, mask=None, min_bins=4):
    if map_1.ndim == 2:
        map_1 = map_1[:, :, None]
    if map_2.ndim == 2:
        map_2 = map_2[:, :, None]
    if mask is not None:
        mask = mask[:, :, None]
    ignore = np.logical_or(np.isnan(map_1), np.isnan(map_2))
    ignore |= (map_1 < min_included_value) | (map_2 < min_included_value)
    inc = ~ignore if mask is None else np.logical_and(~ignore, mask)
    if np.count_nonzero(inc) < min_bins:
        return np.nan, np.nan
    from scipy.stats import pearsonr
    return pearsonr(map_1[inc], map_2[inc])


def _compute_field_stability(field_ratemap, ratemap1, ratemap2, *, min_included_value, min_bins):
    mask = ~np.isnan(field_ratemap)
    return _spatial_correlation(ratemap1, ratemap2, min_included_value=min_included_value, mask=mask, min_bins=min_bins)[0]


def _primary_filter(ratemap, *, min_area_bins=0, min_peak_value=0):
    return (np.count_nonzero(ratemap) >= min_area_bins) and (np.nanmax(ratemap) >= min_peak_value)


def _secondary_filter(ratemap, *, max_area_bins, stability_ratemaps, min_stability, stability_kwargs):
    if np.count_nonzero(ratemap) > max_area_bins:
        return False
    st = _compute_field_stability(ratemap, *stability_ratemaps, **stability_kwargs)
    return (not np.isnan(st)) and (st >= min_stability)


def _label_fields(binmap):
    return ndimage.label(binmap > 0)[0]


def _extract_subfields(ratemap, *, filter_kwargs):
    field_map = _label_fields(ratemap)
    sub_ratemaps = []
    for field_nr in np.unique(field_map)[1:]:
        sub = np.where(field_map == field_nr, ratemap, 0)
        if _primary_filter(sub, **filter_kwargs):
            sub_ratemaps.append(sub)
    return sub_ratemaps


def _detect_fields(ratemap, stability_ratemaps, *, base_threshold, threshold_step, _primary_filter_kwargs, _secondary_filter_kwargs):
    ratemap = ratemap.astype(float).copy()
    ratemap[np.isnan(ratemap)] = 0
    ratemap[ratemap < base_threshold] = 0
    # iterative splitting
    fields = _extract_subfields(ratemap, filter_kwargs=_primary_filter_kwargs)
    final_fields = []
    for field in fields:
        # secondary filter
        kwargs = deepcopy(_secondary_filter_kwargs)
        kwargs['stability_ratemaps'] = stability_ratemaps
        kwargs['max_area_bins'] = int(np.sum(~np.isnan(ratemap)) * kwargs.pop('max_relative_bins'))
        if _secondary_filter(field, **kwargs):
            final_fields.append(field)
    return final_fields


def _compute_single_heatmap_fields_info_new(
    heatmap: np.ndarray,
    env_x_max: int,
    env_y_max: int,
    env_x_min: int,
    env_y_min: int,
    pixel_min_threshold: int,
    pixel_max_threshold: int,
):
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    # Split hd sequence into two halves for stability check
    split = heatmap.shape[1] // 2
    ratemap1 = np.sum(heatmap[:, :split], axis=1)
    ratemap2 = np.sum(heatmap[:, split:], axis=1)
    totalmap = ratemap1 + ratemap2
    shape = (
        env_x_max - env_x_min + 1,
        env_y_max - env_y_min + 1,
    )
    ratemap1 = ratemap1.reshape(shape)
    ratemap2 = ratemap2.reshape(shape)
    totalmap = totalmap.reshape(shape)

    base_th = float(np.mean(totalmap))
    step = 0.1 * float(np.std(totalmap))
    prim_kw = {'min_area_bins': 15, 'min_peak_value': np.max(totalmap) * 0.5}
    sec_kw = {
        'max_relative_bins': 0.5,
        'min_stability': 0.3,
        'stability_kwargs': {'min_included_value': base_th / 2, 'min_bins': 9},
    }
    fields = _detect_fields(
        totalmap,
        (ratemap1, ratemap2),
        base_threshold=base_th,
        threshold_step=step,
        _primary_filter_kwargs=prim_kw,
        _secondary_filter_kwargs=sec_kw,
    )
    num_clusters = np.array([len(fields)])
    if len(fields) == 0:
        zeros = np.array([0])
        return num_clusters, zeros, zeros, zeros, zeros
    num_pixels = np.array([np.count_nonzero(~np.isnan(f)) for f in fields])
    max_vals = np.array([np.nanmax(f) for f in fields])
    mean_vals = np.array([np.nanmean(f) for f in fields])
    var_vals = np.array([np.nanvar(f) for f in fields])
    return (
        num_clusters, 
        num_pixels, 
        max_vals, 
        mean_vals, 
        var_vals
    )

# public wrappers

def _compute_single_heatmap_border_scores(
    activation_map: np.ndarray,
    db: int = 3,
    *,
    use_old_place_method: bool = True,
) -> float:
    if use_old_place_method:
        return _compute_single_heatmap_border_scores_old(activation_map, db=db)
    return _compute_single_heatmap_border_scores_new(activation_map, db=db)


def _compute_single_heatmap_directional_scores(
    activation_maps: np.ndarray,
    *,
    use_old_place_method: bool = True,
):
    if use_old_place_method:
        return _compute_single_heatmap_directional_scores_old(activation_maps)
    return _compute_single_heatmap_directional_scores_new(activation_maps)


def _compute_single_heatmap_fields_info(
    *args: Any,
    use_old_place_method: bool = True,
    **kwargs: Any,
):
    if use_old_place_method:
        return _compute_single_heatmap_fields_info_old(*args, **kwargs)
    return _compute_single_heatmap_fields_info_new(*args, **kwargs)

__all__ = [
    "_compute_single_heatmap_border_scores",
    "_compute_single_heatmap_directional_scores",
    "_compute_single_heatmap_fields_info",
    "_is_dead_neuron"
]
