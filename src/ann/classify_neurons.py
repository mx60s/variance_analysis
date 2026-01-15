from pathlib import Path
import os
from tqdm.auto import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import logging
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .score_utils import *


def classify_neurons(neuron_chart_info):
    """
    Given a neuron_chart_info, classify the neurons into different types,
    and return the indices of neurons by type or combo of types.
    """
    dead_neurons_indices = []
    max_num_clusters = np.max(neuron_chart_info[:, 1])  # global max used for setting xaxis.
    num_clusters = np.zeros(max_num_clusters+1)
    cluster_sizes = []
    cluster_peaks = []
    border_cell_indices = []
    place_cells_indices = []
    direction_cell_indices = []
    active_no_type_indices = []

    for neuron_index in range(neuron_chart_info.shape[0]):
        if neuron_chart_info[neuron_index, 0] == 0:
            dead_neurons_indices.append(neuron_index)
        else:
            num_clusters[int(neuron_chart_info[neuron_index, 1])] += 1
            cluster_sizes.extend(neuron_chart_info[neuron_index, 2])
            cluster_peaks.extend(neuron_chart_info[neuron_index, 3])

            if neuron_chart_info[neuron_index, 1] > 0:
                place_cells_indices.append(neuron_index)
                is_place_cell = True
            else:
                is_place_cell = False

            if neuron_chart_info[neuron_index, 10] > 0.47:
                direction_cell_indices.append(neuron_index)
                is_direction_cell = True
            else:
                is_direction_cell = False

            if neuron_chart_info[neuron_index, 9] > 0.5:
                border_cell_indices.append(neuron_index)
                is_border_cell = True
            else:
                is_border_cell = False

            if not (is_place_cell or is_direction_cell or is_border_cell):
                active_no_type_indices.append(neuron_index)

    # plot
    n_dead_neurons = len(dead_neurons_indices)
    n_active_neurons = neuron_chart_info.shape[0] - n_dead_neurons

    # Collect the indices of neurons that are all three types
    # (place + border + direction)
    place_border_direction_cells_indices = \
        list(set(place_cells_indices) & set(border_cell_indices) & set(direction_cell_indices))
    
    # Collect the indices of neurons that are two types (inc. three types)
    # (place + border cells)
    # (place + direction cells)
    # (border + direction cells)
    place_and_border_cells_indices = \
        list(set(place_cells_indices) & set(border_cell_indices))
    place_and_direction_cells_indices = \
        list(set(place_cells_indices) & set(direction_cell_indices))
    border_and_direction_cells_indices = \
        list(set(border_cell_indices) & set(direction_cell_indices))
    
    # Collect the indices of neurons that are only two types
    # (place  + border - direction),
    # (place  + direction   - border),
    # (border + direction   - place)
    place_and_border_not_direction_cells_indices = \
        list(set(place_and_border_cells_indices) - set(place_border_direction_cells_indices))
    place_and_direction_not_border_cells_indices = \
        list(set(place_and_direction_cells_indices) - set(place_border_direction_cells_indices))
    border_and_direction_not_place_cells_indices = \
        list(set(border_and_direction_cells_indices) - set(place_border_direction_cells_indices))
    
    # Collect the indices of neurons that are exclusive 
    # place cells, 
    # border cells, 
    # direction cells
    exclusive_place_cells_indices = \
        list(set(place_cells_indices) - (set(place_and_border_cells_indices) | set(place_and_direction_cells_indices)))
    exclusive_border_cells_indices = \
        list(set(border_cell_indices) - (set(place_and_border_cells_indices) | set(border_and_direction_cells_indices)))
    exclusive_direction_cells_indices = \
        list(set(direction_cell_indices) - (set(place_and_direction_cells_indices) | set(border_and_direction_cells_indices)))

    results =  {
        'dead_neurons_indices': dead_neurons_indices,
        'place_border_direction_cells_indices': place_border_direction_cells_indices,
        'place_and_border_not_direction_cells_indices': place_and_border_not_direction_cells_indices,
        'place_and_direction_not_border_cells_indices': place_and_direction_not_border_cells_indices,
        'border_and_direction_not_place_cells_indices': border_and_direction_not_place_cells_indices,
        'exclusive_place_cells_indices': exclusive_place_cells_indices,
        'exclusive_border_cells_indices': exclusive_border_cells_indices,
        'exclusive_direction_cells_indices': exclusive_direction_cells_indices,
        'active_no_type_indices': active_no_type_indices,
    }
    
    assert neuron_chart_info.shape[0] == sum([len(v) for v in results.values()])

    # Check all values are mutually exclusive
    for key, value in results.items():
        for key2, value2 in results.items():
            if key != key2:
                assert len(set(value) & set(value2)) == 0, f'{key} and {key2} have common elements'

    return results

def score_neurons(
        model_reps_paths,
        config,
        random_seed=None,
        sample_partial=False,
        use_old_place_method=True,
        pixel_min_threshold = 20,
        base_dir='/',
        n_visits=0
    ):
    """
    Score each individual neuron with information about place field stability etc.
    """
    
    env_x_min, env_x_max, env_y_min, env_y_max = config['env_bounds']
    n_rotations = config['n_rotations']

    # load model outputs
    model_reps = []
    for model_reps_path in model_reps_paths:
        model_reps.append(np.load(model_reps_path)['activations']) # in shape num_locs, num_rots, num_features

    model_reps = np.concatenate(model_reps)

    if n_visits > 0:
        xy_shape = env_x_max - env_x_min + 1
        model_reps = model_reps.reshape(xy_shape, xy_shape, n_rotations, n_visits, -1)
        model_reps = model_reps.mean(axis=3)
        n_neurons = model_reps.shape[3]
        model_reps = model_reps.reshape(xy_shape * xy_shape, n_rotations, n_neurons)
    else:
        n_neurons = model_reps.shape[1]
        model_reps = np.reshape(model_reps, 
                                    (model_reps.shape[0] // n_rotations, 
                                     n_rotations, model_reps.shape[1]))

    model_name = config['model_name'] 
    output_layer = config['output_layer']

    results_path = (
        base_dir / "data/ann_activations" / f"{model_name}_{output_layer}"
    )
    results_path.mkdir(parents=True, exist_ok=True)

    sample_path = None
    if sample_partial:
        print("Sample partial")
        np.random.seed(random_seed)

        if n_neurons > 8000:
            sampled_indices = np.random.choice(n_neurons, 8000, replace=False)
            model_reps = model_reps[:, :, sampled_indices]
            print(f'Sampled 8000 neurons out of {n_neurons}')

            sample_path = results_path / f'{model_name}_{output_layer}_sampled_idx.npy'

            np.save(sample_path, sampled_indices)
            
        else:
            print(f'Less than 8000 neurons ({n_neurons}), using all neurons')
        
    # ken: vit uses gelu, unfortunately, it produces negative values
    # in model_reps which messes up border and direction scores (led to huge scores),
    # a workaround is to relu the model_reps.
    if 'vit' in model_name or 'post_pred' in output_layer:
        print('[Check] transformer-based model detected, relu model_reps.')
        model_reps[model_reps < 0] = 0.

    # charted info:
    charted_info = [
                    'dead',
                    # --- fields info
                    'num_clusters', 
                    'num_pixels_in_clusters', 
                    'max_value_in_clusters', 
                    'mean_value_in_clusters', 
                    'var_value_in_clusters',
                    'entire_map_mean',
                    'entire_map_var',
                    # ---
                    'gridness',
                    # ---
                    'borderness',
                    # --- directioness
                    'mean_vector_length',
                    'per_rotation_vector_length',
                    # ---
                ]
    
    # shape \in (total_n_neurons, len(charted_info))
    # init from zero so as we first check if a neuron is dead
    # if dead, we continue to next neuron so the rest of the info re this neuron 
    # will be kept as zero
    
    neuron_chart_info = np.zeros(
        (model_reps.shape[2], len(charted_info)), dtype=object # num_features, num_charted_info
    )
    model_reps_summed = np.sum( # num_locs, num_rots, num_features
        model_reps, axis=1, keepdims=True
    )

    for neuron_index in tqdm(
        range(n_neurons),
        desc="Charting neurons",
        unit="neuron",
    ):
        # reshape to (n_locations, n_rotations, n_features)
        # but rotation dimension is 1 after summing, so 
        # we just hard code 0.
        heatmap = model_reps_summed[:, 0, neuron_index].reshape(
            (env_x_max - env_x_min + 1, 
            env_y_max - env_y_min + 1) )

        ###### Go thru each required info, maybe modularise later.
        # these are the same in oumc and umc
        if _is_dead_neuron(heatmap):
            #print(f'neuron {neuron_index} dead.')
            neuron_chart_info[neuron_index, 0] = np.array([0])
            continue
        else:
            neuron_chart_info[neuron_index, 0] = np.array([1])
            # compute, collect and save neuron chart info
            # 1. fields info
            if use_old_place_method:
                args = [
                    heatmap,
                    pixel_min_threshold,
                    int(heatmap.shape[0]*heatmap.shape[1]*0.5)
                ]
            else:
                args = [
                    model_reps[:, :, neuron_index], # now passing in the unsummed model_reps, so hd is preserved
                    env_x_max,
                    env_y_max,
                    env_x_min,
                    env_y_min,
                    pixel_min_threshold,
                    int(heatmap.shape[0]*heatmap.shape[1]*0.5)
                ]

            num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
                mean_value_in_clusters, var_value_in_clusters = \
                    _compute_single_heatmap_fields_info(
                        *args,
                        use_old_place_method=use_old_place_method
                        )
                    
            neuron_chart_info[neuron_index, 1] = num_clusters
            neuron_chart_info[neuron_index, 2] = num_pixels_in_clusters
            neuron_chart_info[neuron_index, 3] = max_value_in_clusters
            neuron_chart_info[neuron_index, 4] = mean_value_in_clusters
            neuron_chart_info[neuron_index, 5] = var_value_in_clusters
            neuron_chart_info[neuron_index, 6] = np.array([np.mean(heatmap)])
            neuron_chart_info[neuron_index, 7] = np.array([np.var(heatmap)])

            # there was also gridness, but leaving it behind because we don't need it and it was erroring out

            # 3. borderness
            neuron_chart_info[neuron_index, 9] = \
                _compute_single_heatmap_border_scores(
                    heatmap, 
                    use_old_place_method=use_old_place_method
                )

            # 4. directioness (use model_reps instead of model_reps_summed)
            directional_score, per_rotation_vector_length = \
                _compute_single_heatmap_directional_scores(
                    activation_maps=model_reps[:, :, neuron_index],
                    use_old_place_method=use_old_place_method
                )
                    
            neuron_chart_info[neuron_index, 10] = directional_score
            neuron_chart_info[neuron_index, 11] = per_rotation_vector_length


    fpath = results_path / "neuron_chart.npy"
    np.save(fpath, neuron_chart_info)
    print(f"[Saved] {fpath}")

    return (fpath, sample_path)
    