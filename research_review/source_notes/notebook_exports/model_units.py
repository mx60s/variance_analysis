# CELL 0
# Run in a fresh runtime, and use %pip (targets the notebook kernel)
%pip uninstall -y torch torchvision torchaudio xformers pillow
%pip cache purge -q

%pip install --no-cache-dir \
  torch torchvision pillow \
  --index-url https://download.pytorch.org/whl/cu126

%pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu126


# CELL 1
from google.colab import drive
import os

drive.mount('/content/drive/')
current_file_path = '/content/drive/MyDrive/ve/variance_analysis'

os.chdir(current_file_path)

%pip install -e .

!mkdir -p /usr/share/fonts/truetype/helvetica
!ttx -y 0 -o /usr/share/fonts/truetype/helvetica/Helvetica-Regular.ttf fonts/Helvetica.ttc
!ttx -y 1 -o /usr/share/fonts/truetype/helvetica/Helvetica-Oblique.ttf fonts/Helvetica.ttc
!ttx -y 2 -o /usr/share/fonts/truetype/helvetica/Helvetica-Bold.ttf fonts/Helvetica.ttc
!ttx -y 3 -o /usr/share/fonts/truetype/helvetica/Helvetica-BoldOblique.ttf fonts/Helvetica.ttc
!fc-cache -f

from pathlib import Path
current_file_path = Path(current_file_path)

# CELL 3
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
import shutil

!mkdir -p ~/.fonts
!cp fonts/Helvetica.ttc ~/.fonts/

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'Helvetica'

# CELL 5
import numpy as np

model_name = 'predictive_coder'  # predictive_coder | resnet50 | vit_b16 | resnet50_untrained | vgg16

# Environment constants
ENV_X_MIN = 120
ENV_X_MAX = 136
ENV_Y_MIN = 197
ENV_Y_MAX = 213

# Grid dimensions
N_POS_X = ENV_X_MAX - ENV_X_MIN + 1  # 17
N_POS_Y = ENV_Y_MAX - ENV_Y_MIN + 1  # 17

N_V2_PANORAMA = 90

print(f"Grid: {N_POS_X} x {N_POS_Y} locations")

# CELL 7
# predictive code latents are pre-prediction/post-encoding ('pre_pred'), or post-prediction/pre-decoding residual stream ('post_pred')
# resnet can take layer1, layer3, layer4, or avgpool
# vit can take layer_{3, 6, 9, 12}
# vgg16 can take features.9, features.16, features.30, or classifier.3

from src.ann.get_activations import load_model, get_activations

layer_name = 'pre_pred'

config = {
    'model_name': model_name,
    'output_layer': layer_name,
    'env_bounds': [ENV_X_MIN, ENV_X_MAX, ENV_Y_MIN, ENV_Y_MAX],
    'n_rotations': N_V2_PANORAMA  # Using 90 V2 samples from panorama
}

model, preprocess = load_model(model_name, base_path=current_file_path, output_layer=layer_name)

# CELL 8
# Panorama data configuration
# --------------------------
# panorama-90yaw: 90 V2 samples at 0° pitch (for unit classification)
# panorama-90yaw-pitch: 90 V2 × 4 pitch samples (for EV with pitch visits)

# Pitch configuration
N_PITCHES = 4
PITCHES = [-3, -1, 1, 3]  # degrees

# Interleaved fold assignment to balance positive/negative pitches
# fold0 gets pitches at indices [0, 2] -> [-3°, +1°]
# fold1 gets pitches at indices [1, 3] -> [-1°, +3°]
FOLD0_PITCH_INDICES = [0, 2]
FOLD1_PITCH_INDICES = [1, 3]

device = 'cuda'

# CELL 9
# Load panorama-90yaw data (clean, 0° pitch) for unit classification
import glob
from tqdm.auto import tqdm

def load_panorama_images(panorama_dir, with_pitch=False):
    """
    Load panorama images from npz files.

    Args:
        panorama_dir: Path to directory containing loc_*.npz files
        with_pitch: If True, expects 360 images per location (90 V2 * 4 pitch)
                   If False, expects 90 images per location (90 V2 only)

    Returns:
        images: (n_locs, n_images_per_loc, H, W, C) array
        locations: (n_locs, 2) array of (x, y) coordinates
    """
    npz_files = sorted(glob.glob(str(panorama_dir / 'loc_*.npz')))
    print(f"Found {len(npz_files)} location files in {panorama_dir}")

    images_list = []
    locations_list = []

    for npz_file in tqdm(npz_files, desc="Loading panorama"):
        data = np.load(npz_file)
        imgs = data['rgb']  # (n_images, H, W, C) or similar

        # Parse location from filename: loc_+XXX.X_+YYY.Y.npz
        fname = Path(npz_file).stem
        parts = fname.replace('loc_', '').split('_')
        x = float(parts[0])
        y = float(parts[1])

        images_list.append(imgs)
        locations_list.append([x, y])

    images = np.stack(images_list, axis=0)  # (n_locs, n_images, H, W, C)
    locations = np.array(locations_list)     # (n_locs, 2)

    expected_per_loc = 360 if with_pitch else 90
    print(f"Loaded images shape: {images.shape}")
    print(f"Expected {expected_per_loc} images per location, got {images.shape[1]}")

    return images, locations


# Load clean panorama (90 V2, 0° pitch)
panorama_clean_dir = current_file_path / 'data/minecraft/panorama-90yaw'
panorama_clean_images, panorama_clean_locs = load_panorama_images(panorama_clean_dir, with_pitch=False)

n_locs_clean = panorama_clean_images.shape[0]
print(f"\nClean panorama: {n_locs_clean} locations, {panorama_clean_images.shape[1]} V2 samples each")

# Show an example image
plt.imshow(panorama_clean_images[0, 0])
plt.title(f"Location {panorama_clean_locs[0]}, V2=0")
plt.show()

# CELL 10
import torch

def get_panorama_activations(model, preprocess_fn, panorama_images, model_name,
                              device='cuda', batch_size=32, max_units=8000, sample_seed=413):
    """
    Compute model activations for panorama images.

    Args:
        model: PyTorch model
        preprocess_fn: Preprocessing function from load_model (expects numpy arrays)
        panorama_images: (n_locs, n_images_per_loc, H, W, C) array
        model_name: Name of the model (needed to handle predictive_coder specially)
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        max_units: Maximum number of units to keep (subsamples if exceeded)
        sample_seed: Random seed for unit subsampling

    Returns:
        activations: (n_locs, n_images_per_loc, n_units) array
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device).eval()

    n_locs, n_imgs_per_loc = panorama_images.shape[:2]

    # Flatten for batch processing
    flat_images = panorama_images.reshape(-1, *panorama_images.shape[2:])  # (n_total, H, W, C)
    n_total = flat_images.shape[0]

    # Check if predictive coder (needs sequence dimension)
    is_predictive = model_name.startswith("predictive_coder")
    seq_len = 10 if is_predictive else None

    selected_idx = None
    activations_list = []

    with torch.no_grad():
        for i in tqdm(range(0, n_total, batch_size), desc="Computing activations"):
            batch_np = flat_images[i:i+batch_size]  # (B, H, W, C)

            # For predictive coder, add sequence dimension
            if is_predictive:
                batch_np = np.repeat(batch_np[:, None, ...], seq_len, axis=1)  # (B, T, H, W, C)

            # Apply preprocessing and move to device
            x = preprocess_fn(batch_np).to(device)

            # Forward pass
            out = model(x)

            # Handle different output formats
            if isinstance(out, dict):  # torchvision feature extractor
                out = next(iter(out.values()))
            if is_predictive:
                out = out[:, -1]  # Take last timestep

            # Flatten spatial dimensions
            out = out.flatten(1)

            # Subsample units if needed
            if out.shape[1] > max_units:
                if selected_idx is None:
                    g = torch.Generator(device=out.device).manual_seed(sample_seed)
                    selected_idx = torch.randperm(out.shape[1], generator=g, device=out.device)[:max_units]
                out = out.index_select(1, selected_idx)

            activations_list.append(out.cpu().numpy())

    activations_flat = np.concatenate(activations_list, axis=0)  # (n_total, n_units)
    activations = activations_flat.reshape(n_locs, n_imgs_per_loc, -1)  # (n_locs, n_imgs, n_units)

    return activations


# Compute or load clean activations
clean_acts_path = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_clean/acts.npz'

if clean_acts_path.exists():
    print(f"Loading cached clean activations from {clean_acts_path}")
    clean_acts_data = np.load(clean_acts_path)
    clean_acts = clean_acts_data['activations']
else:
    print("Computing clean activations from panorama-90yaw...")
    clean_acts = get_panorama_activations(
        model, preprocess, panorama_clean_images,
        model_name=model_name,
        device=device, batch_size=64
    )

    # Cache for later
    clean_acts_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(clean_acts_path, activations=clean_acts, locations=panorama_clean_locs)
    print(f"Saved clean activations to {clean_acts_path}")

print(f"Clean activations shape: {clean_acts.shape}")
# Shape: (n_locs, n_v2, n_units) = (289, 90, n_units)

n_locs, n_v2, n_units = clean_acts.shape
print(f"  {n_locs} locations, {n_v2} V2 samples, {n_units} units")

# CELL 11
from src.ann.classify_units import score_units, classify_units

# For score_units, we need activations in the expected format
# Reshape clean_acts from (n_locs, n_v2, n_units) to (n_locs * n_v2, n_units)
#clean_acts_flat = clean_acts.reshape(-1, n_units)

# Save in expected format temporarily
temp_acts_path = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_clean/acts_flat.npz'
#np.savez(temp_acts_path, activations=clean_acts_flat)

# Update config for panorama data
config_classify = {
    'model_name': model_name,
    'output_layer': layer_name,
    'env_bounds': [ENV_X_MIN, ENV_X_MAX, ENV_Y_MIN, ENV_Y_MAX],
    'n_rotations': N_V2_PANORAMA
}

unit_scores_fpath, sampled_path = score_units(
    [temp_acts_path],
    config_classify,
    base_dir=current_file_path,
    sample_partial=False,
    pixel_min_threshold=1,
    use_old_place_method=False,
    n_visits=1,
)

unit_score_charts = np.load(unit_scores_fpath, allow_pickle=True)
classifications = classify_units(unit_score_charts)
print(classifications.keys())

# CELL 12
from src.ann.fig import plot_unit_type_pie

fig = plot_unit_type_pie(classifications)
fig.savefig(current_file_path / f'figs/{model_name}_{layer_name}_breakdown.pdf')
plt.show()

# CELL 13
def get_splits_deterministic(visits, fold0_indices, fold1_indices):
    """
    Split visits into folds using deterministic index assignment.

    This is used for pitch-based visits where we want balanced pitch assignment:
    - fold0 gets pitches at fold0_indices (e.g., [-3°, +1°])
    - fold1 gets pitches at fold1_indices (e.g., [-1°, +3°])

    Args:
        visits: (s1_bins, s2_bins, n_visits, n_units) array
                For pitch mode: (n_locs, n_v2, n_pitches, n_units)
        fold0_indices: list of visit indices for fold0
        fold1_indices: list of visit indices for fold1

    Returns:
        fold0_mean, fold1_mean: (s1_bins, s2_bins, n_units) arrays
        fold0_occ, fold1_occ: (s1_bins, s2_bins) occupancy arrays
    """
    s1_bins, s2_bins, n_visits, n_units = visits.shape

    # Extract visits for each fold
    fold0_visits = visits[:, :, fold0_indices, :]  # (s1, s2, n_fold0, units)
    fold1_visits = visits[:, :, fold1_indices, :]  # (s1, s2, n_fold1, units)

    # Compute means across visits dimension
    fold0_mean = np.nanmean(fold0_visits, axis=2)  # (s1, s2, units)
    fold1_mean = np.nanmean(fold1_visits, axis=2)  # (s1, s2, units)

    # Occupancy is uniform since we always have the same number of visits per bin
    fold0_occ = np.full((s1_bins, s2_bins), len(fold0_indices), dtype=float)
    fold1_occ = np.full((s1_bins, s2_bins), len(fold1_indices), dtype=float)

    return fold0_mean, fold1_mean, fold0_occ, fold1_occ


print(f"Pitch visit mode configuration:")
print(f"  Pitches: {PITCHES}")
print(f"  Fold 0 indices: {FOLD0_PITCH_INDICES} -> pitches {[PITCHES[i] for i in FOLD0_PITCH_INDICES]}")
print(f"  Fold 1 indices: {FOLD1_PITCH_INDICES} -> pitches {[PITCHES[i] for i in FOLD1_PITCH_INDICES]}")

# CELL 15
# Load the pitch-varied panorama data
panorama_pitch_dir = current_file_path / 'data/minecraft/panorama-90yaw-pitch'
panorama_pitch_images, panorama_pitch_locs = load_panorama_images(panorama_pitch_dir, with_pitch=True)

n_locs_pitch = panorama_pitch_images.shape[0]
print(f"\nPitch panorama: {n_locs_pitch} locations")
print(f"Images per location: {panorama_pitch_images.shape[1]} (should be {N_V2_PANORAMA * N_PITCHES} = {N_V2_PANORAMA * N_PITCHES})")

# Verify locations match
assert n_locs_pitch == n_locs, f"Location count mismatch: pitch={n_locs_pitch}, clean={n_locs}"

# CELL 16
pitch_acts_path = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_pitch/acts.npz'

if pitch_acts_path.exists():
    print(f"Loading cached pitch activations from {pitch_acts_path}")
    pitch_acts = np.load(pitch_acts_path)['activations']
else:
    print("Computing pitch activations (this may take a while)...")
    pitch_acts = get_panorama_activations(
        model, preprocess, panorama_pitch_images,
        model_name=model_name,
        device=device, batch_size=64
    )

    # Cache for later
    pitch_acts_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pitch_acts_path, activations=pitch_acts, locations=panorama_pitch_locs)
    print(f"Saved pitch activations to {pitch_acts_path}")

print(f"Pitch activations shape: {pitch_acts.shape}")

# CELL 18
from tqdm import tqdm
from src.explained_variance import nan_aware_gaussian_filter, masked_zoom, quantify_variance_corr

units = np.asarray(classifications['exclusive_units_indices'])
is_place = np.isin(np.arange(n_units), units)
print(f"Found {len(units)} units out of {n_units} units")

sigma = (1, 1)

pitch_acts_4d = pitch_acts.reshape(n_locs, N_V2_PANORAMA, N_PITCHES, n_units)
print(f"Pitch activations reshaped: {pitch_acts_4d.shape}")
print(f"  = (n_locs={n_locs}, n_v2={N_V2_PANORAMA}, n_pitch={N_PITCHES}, n_units={n_units})")

fold0_mean, fold1_mean, fold0_occ, fold1_occ = get_splits_deterministic(
    pitch_acts_4d,
    FOLD0_PITCH_INDICES,
    FOLD1_PITCH_INDICES
)

fold0 = fold0_mean
fold1 = fold1_mean
occ = fold0_occ + fold1_occ

zoom_factor = (1, N_V2_PANORAMA // N_POS_X)

print(f"fold0 shape: {fold0.shape}")
print(f"fold1 shape: {fold1.shape}")
print(f"occ shape: {occ.shape}")
print(f"zoom_factor: {zoom_factor}")


# CELL 19
from joblib import Parallel, delayed

n_using = min(1000, n_units)
units = list(range(n_units))
np.random.seed(42)
np.random.shuffle(units)

def process_unit_pitch(n, fold0, fold1, occ, zoom_factor, sigma):
    f0 = nan_aware_gaussian_filter(fold0[:, :, n], sigma=sigma, circular_axes=(1,))
    f1 = nan_aware_gaussian_filter(fold1[:, :, n], sigma=sigma, circular_axes=(1,))
    f0 = masked_zoom(f0, zoom_factor=zoom_factor)
    f1 = masked_zoom(f1, zoom_factor=zoom_factor)
    occ_z = masked_zoom(occ, zoom_factor=zoom_factor)
    ev_v1 = 1 - quantify_variance_corr(f0, f1, occ_z, axis=0)[0]
    ev_v2 = 1 - quantify_variance_corr(f0, f1, occ_z, axis=1)[0]
    return ev_v1, ev_v2

results = Parallel(n_jobs=-1)(
    delayed(process_unit_pitch)(units[i], fold0, fold1, occ, zoom_factor, sigma)
    for i in tqdm(range(n_using))
)

s1_var = np.array([r[0] for r in results])
s2_var = np.array([r[1] for r in results])
is_place_trunc = np.array([is_place[units[i]] for i in range(n_using)])

print(f"Computed EV for {n_using} units")


# CELL 20
import numpy as np
import matplotlib.pyplot as plt

n_sample_other = 200
n_sample_place = 100
rng = np.random.default_rng(42)

other_idx = np.where(~is_place_trunc)[0]
place_idx = np.where(is_place_trunc)[0]

if len(other_idx) > n_sample_other:
    other_idx = rng.choice(other_idx, n_sample_other, replace=False)
if len(place_idx) > n_sample_place:
    place_idx = rng.choice(place_idx, n_sample_place, replace=False)

plt.scatter(s1_var[other_idx], s2_var[other_idx], s=30,
            label=f"Other units (n={len(other_idx)})")
plt.scatter(s1_var[place_idx], s2_var[place_idx], color="red", s=30,
            label=f"Exclusive units (n={len(place_idx)})")

vals = np.concatenate([s1_var, s2_var])
vmin, vmax = 0, 1 #np.nanmin(vals), np.nanmax(vals)
plt.plot([vmin, vmax], [vmin, vmax], "--", lw=1, color="k", alpha=0.6)

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("EV (V1)")
plt.ylabel("EV (V2)")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(current_file_path / f'figs/{model_name}_{layer_name}_ev_pitch.pdf')
plt.show()

x_all = np.asarray(s1_var, dtype=float).ravel()
y_all = np.asarray(s2_var, dtype=float).ravel()
valid = np.isfinite(x_all) & np.isfinite(y_all)
xv, yv = x_all[valid], y_all[valid]

v1_dominant = np.sum(xv > yv) / len(xv)
v2_dominant = np.sum(yv > xv) / len(yv)
ratio = yv / (xv + yv + 1e-8)
print(f"\nSummary for pitch visit mode (all {len(xv)} units):")
print(f"  V1-dominant: {v1_dominant:.1%}")
print(f"  V2-dominant: {v2_dominant:.1%}")
print(f"  Mean V2/(V1+V2): {np.mean(ratio):.3f}")
print(f"  Median V2/(V1+V2): {np.median(ratio):.3f}")


# CELL 22
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from src.ann.get_activations import load_model
from src.ann.classify_units import score_units, classify_units
from src.explained_variance import nan_aware_gaussian_filter, masked_zoom, quantify_variance_corr

models_and_layers = {
    "resnet50": ["layer1", "layer3", "layer4", "avgpool"],
    "resnet50_untrained": ["avgpool"],
    "vgg16": ["features.9"],
    "predictive_coder": ["conv1", "down1", "pre_pred", "post_pred"],
}

def process_unit_loop(n, fold0, fold1, occ, zoom_factor, sigma):
    f0 = nan_aware_gaussian_filter(fold0[:, :, n], sigma=sigma, circular_axes=(1,))
    f1 = nan_aware_gaussian_filter(fold1[:, :, n], sigma=sigma, circular_axes=(1,))
    f0 = masked_zoom(f0, zoom_factor=zoom_factor)
    f1 = masked_zoom(f1, zoom_factor=zoom_factor)
    occ_z = masked_zoom(occ, zoom_factor=zoom_factor)
    ev_v1 = 1 - quantify_variance_corr(f0, f1, occ_z, axis=0)[0]
    ev_v2 = 1 - quantify_variance_corr(f0, f1, occ_z, axis=1)[0]
    return ev_v1, ev_v2

for model_name, layer_names in models_and_layers.items():
    for layer_name in layer_names:
        print(f"\n{'='*60}")
        print(f"=== {model_name} / {layer_name} ===")
        print(f"{'='*60}")

        model_curr, preprocess_curr = load_model(
            model_name, base_path=current_file_path, output_layer=layer_name
        )

        clean_acts_path = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_clean/acts.npz'

        if clean_acts_path.exists():
            print(f"  Loading cached clean activations")
            clean_acts_curr = np.load(clean_acts_path)['activations']
        else:
            print(f"  Computing clean activations...")
            clean_acts_curr = get_panorama_activations(
                model_curr, preprocess_curr, panorama_clean_images,
                model_name=model_name, device=device, batch_size=64
            )
            clean_acts_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(clean_acts_path, activations=clean_acts_curr, locations=panorama_clean_locs)

        n_locs_curr, n_v2_curr, n_units_curr = clean_acts_curr.shape
        print(f"  n_units={n_units_curr}")

        clean_acts_flat = clean_acts_curr.reshape(-1, n_units_curr)
        temp_acts_path = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_clean/acts_flat.npz'
        np.savez(temp_acts_path, activations=clean_acts_flat)

        config_curr = {
            "model_name": model_name,
            "output_layer": layer_name,
            "env_bounds": [ENV_X_MIN, ENV_X_MAX, ENV_Y_MIN, ENV_Y_MAX],
            "n_rotations": N_V2_PANORAMA,
        }

        unit_scores_fpath, _ = score_units(
            [temp_acts_path], config_curr, base_dir=current_file_path,
            sample_partial=False, pixel_min_threshold=1,
            use_old_place_method=False, n_visits=1,
        )
        unit_score_charts = np.load(unit_scores_fpath, allow_pickle=True)
        classifications_curr = classify_units(unit_score_charts)
        units_curr = np.asarray(classifications_curr["exclusive_units_indices"])
        is_place_curr = np.isin(np.arange(n_units_curr), units_curr)
        print(f"  {len(units_curr)} units")

        pitch_acts_path = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_pitch/acts.npz'

        if pitch_acts_path.exists():
            print(f"  Loading cached pitch activations")
            pitch_acts_curr = np.load(pitch_acts_path)['activations']
        else:
            print(f"  Computing pitch activations...")
            pitch_acts_curr = get_panorama_activations(
                model_curr, preprocess_curr, panorama_pitch_images,
                model_name=model_name, device=device, batch_size=64
            )
            pitch_acts_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pitch_acts_path, activations=pitch_acts_curr, locations=panorama_pitch_locs)

        sigma = (1, 1)
        pitch_acts_4d = pitch_acts_curr.reshape(n_locs_curr, N_V2_PANORAMA, N_PITCHES, n_units_curr)

        fold0_curr, fold1_curr, fold0_occ_curr, fold1_occ_curr = get_splits_deterministic(
            pitch_acts_4d, FOLD0_PITCH_INDICES, FOLD1_PITCH_INDICES
        )
        occ_curr = fold0_occ_curr + fold1_occ_curr
        zoom_factor_curr = (1, N_V2_PANORAMA // N_POS_X)

        results = Parallel(n_jobs=-1)(
            delayed(process_unit_loop)(n, fold0_curr, fold1_curr, occ_curr, zoom_factor_curr, sigma)
            for n in tqdm(range(n_units_curr), desc=f"  EV per unit")
        )

        ev_v1 = np.array([r[0] for r in results])
        ev_v2 = np.array([r[1] for r in results])

        ev_out_dir = current_file_path / f"results/ann/{model_name}_{layer_name}"
        ev_out_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            ev_out_dir / "ev_all_units_pitch.npz",
            ev_v1=ev_v1, ev_v2=ev_v2,
            is_place=is_place_curr, n_units=n_units_curr,
        )
        print(f"  Saved EVs for {n_units_curr} units")

        valid = np.isfinite(ev_v1) & np.isfinite(ev_v2)
        x, y = ev_v1[valid], ev_v2[valid]
        v1_dominant = np.sum(x > y) / len(x)
        ratio = y / (x + y + 1e-8)
        print(f"  V1-dominant: {v1_dominant:.1%}, Mean V2/(V1+V2): {np.mean(ratio):.3f}")


# CELL 24
n_sample_other = 250
n_sample_place = 75
rng_vis = np.random.default_rng(42)

for model_name_plot, layer_names_plot in models_and_layers.items():
    for layer_name_plot in layer_names_plot:
        print(f"\n=== Plotting {model_name_plot} / {layer_name_plot} ===")

        ev_path = current_file_path / f"results/ann/{model_name_plot}_{layer_name_plot}/ev_all_units_pitch.npz"
        if not ev_path.exists():
            print(f"  Skipping: EV file not found at {ev_path}")
            continue

        data = np.load(ev_path)
        ev_v1, ev_v2 = data["ev_v1"], data["ev_v2"]
        is_place_plot = data["is_place"]
        n_units_plot = int(data["n_units"])
        print(f"  Loaded EVs for {n_units_plot} units ({is_place_plot.sum()} units)")

        other_idx = np.where(~is_place_plot)[0]
        place_idx = np.where(is_place_plot)[0]

        if len(other_idx) > n_sample_other:
            other_idx = rng_vis.choice(other_idx, n_sample_other, replace=False)
        if len(place_idx) > n_sample_place:
            place_idx = rng_vis.choice(place_idx, n_sample_place, replace=False)

        plt.scatter(ev_v1[other_idx], ev_v2[other_idx], s=30,
                    label=f"Other units (n={len(other_idx)})")
        plt.scatter(ev_v1[place_idx], ev_v2[place_idx], color="red", s=30,
                    label=f"Exclusive units (n={len(place_idx)})")

        vals = np.concatenate([ev_v1.ravel(), ev_v2.ravel()])
        vmin, vmax = 0, 1 #np.nanmin(vals), np.nanmax(vals)

        plt.plot([vmin, vmax], [vmin, vmax], "--", lw=1, color="k", alpha=0.6)

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("EV (V1)")
        plt.ylabel("EV (V2)")
        plt.ylim(0., 1.)
        plt.xlim(0., 1.)
        plt.title(f"{model_name_plot} / {layer_name_plot}")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()

        scatter_out_path = current_file_path / f"figs/{model_name_plot}_{layer_name_plot}_ev_pitch.pdf"
        scatter_out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(scatter_out_path)
        plt.show()
        plt.close()

        valid = np.isfinite(ev_v1) & np.isfinite(ev_v2)
        xv, yv = ev_v1[valid], ev_v2[valid]
        v1_dominant = np.sum(xv > yv) / len(xv)
        ratio = yv / (xv + yv + 1e-8)
        print(f"  V1-dominant: {v1_dominant:.1%}, Mean V2/(V1+V2): {np.mean(ratio):.3f}")
        print(f"  Saved scatter to {scatter_out_path}")


# CELL 26
from src.ann.fig import plot_unit_2d_3d

def reshape_for_3d_panorama(acts, n_locs, n_v2):
    """
    Reshape panorama activations for 3D plotting.

    Input: (n_locs, n_v2, n_units)
    Output: (n_units, sqrt(n_locs), sqrt(n_locs), n_v2) assuming square grid
    """
    n_units = acts.shape[-1]
    grid_size = int(np.sqrt(n_locs))
    assert grid_size * grid_size == n_locs, f"n_locs={n_locs} is not a perfect square"

    # Reshape: (n_locs, n_v2, n_units) -> (grid_x, grid_y, n_v2, n_units)
    acts_4d = acts.reshape(grid_size, grid_size, n_v2, n_units)

    # Reorder to (n_units, grid_x, grid_y, n_v2)
    return np.moveaxis(acts_4d, -1, 0)


# Use clean activations for 3D visualization (0° pitch only, 90 V2)
clean_acts_3d = reshape_for_3d_panorama(clean_acts, n_locs, N_V2_PANORAMA)
print(f"Clean activations for 3D: {clean_acts_3d.shape}")
print(f"  = (n_units={n_units}, grid_x={N_POS_X}, grid_y={N_POS_Y}, n_v2={N_V2_PANORAMA})")

# CELL 27
# Plot example place-like units based on pitch-mode EV results
ev_path = current_file_path / f'results/ann/{model_name}_{layer_name}/ev_all_units_pitch.npz'

if ev_path.exists():
    ev_data = np.load(ev_path)
    ev_v1_viz = ev_data['ev_v1']
    ev_v2_viz = ev_data['ev_v2']

    # Find units with high V1 selectivity (place-like)
    v1_ratio = ev_v1_viz / (ev_v1_viz + ev_v2_viz + 1e-8)
    valid_viz = np.isfinite(v1_ratio)
    top_place_units = np.argsort(v1_ratio)[::-1]
    top_place_units = [n for n in top_place_units if valid_viz[n]][:5]

    print(f"\n=== Top place-like units (high V1/total ratio) ===")
    for unit_idx in top_place_units[:3]:
        print(f"Unit {unit_idx}: V1 ratio = {v1_ratio[unit_idx]:.3f}")
        fig = plot_unit_2d_3d(
            clean_acts_3d,
            unit_idx,
            axes_names=("X", "Y", "V2"),
            overall_title=f"Unit {unit_idx} (V1 ratio: {v1_ratio[unit_idx]:.2f})",
            n_slices=4
        )
        fig.write_image(str(current_file_path / f'figs/{model_name}_{layer_name}_unit_{unit_idx}_3d_pitch.pdf'))
        fig.show()

    # Also show V2-dominant units for comparison
    v2_ratio = ev_v2_viz / (ev_v1_viz + ev_v2_viz + 1e-8)
    top_v2_units = np.argsort(v2_ratio)[::-1]
    top_v2_units = [n for n in top_v2_units if valid_viz[n]][:3]

    print(f"\n=== Top V2-like units (high V2/total ratio) ===")
    for unit_idx in top_v2_units[:2]:
        print(f"Unit {unit_idx}: V2 ratio = {v2_ratio[unit_idx]:.3f}")
        fig = plot_unit_2d_3d(
            clean_acts_3d,
            unit_idx,
            axes_names=("X", "Y", "V2"),
            overall_title=f"Unit {unit_idx} (V2 ratio: {v2_ratio[unit_idx]:.2f})",
            n_slices=4
        )
        fig.write_image(str(current_file_path / f'figs/{model_name}_{layer_name}_unit_{unit_idx}_v2_3d_pitch.pdf'))
        fig.show()
else:
    print(f"EV data not found at {ev_path}")

# CELL 29
# Intermediate Layer Analysis - Unit 1: Visualize conv1 filters

from src.ann.fig import visualize_conv1_filters
from src.ann.get_activations import load_model

# Load the predictive coder model (any layer works, we just need the weights)
model_conv1, preprocess_conv1 = load_model(
    'predictive_coder',
    base_path=current_file_path,
    output_layer='conv1',
    device=device
)

# Visualize the 64 learned 7x7 filters
fig = visualize_conv1_filters(model_conv1, ncols=8, normalize=True)
fig.savefig(current_file_path / 'figs/predictive_coder_conv1_filters.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print filter statistics
weights = model_conv1.core.encoder.conv1.weight.detach().cpu().numpy()
print(f"Conv1 filters shape: {weights.shape}")
print(f"Filter range: [{weights.min():.3f}, {weights.max():.3f}]")

# CELL 30
# Intermediate Layer Analysis - Unit 2: Extract encoder layer activations

encoder_layers = ['conv1', 'down1', 'down2']
encoder_clean_activations = {}
encoder_pitch_activations = {}

for layer in encoder_layers:
    print(f"\n=== Extracting {layer} activations ===")

    model_enc, preprocess_enc = load_model(
        'predictive_coder',
        base_path=current_file_path,
        output_layer=layer,
        device=device
    )

    # Clean activations (for classification and 3D viz)
    clean_acts_path = current_file_path / f'data/ann_activations/predictive_coder_{layer}_panorama_clean/acts.npz'

    if clean_acts_path.exists():
        print(f"  Loading cached clean activations")
        encoder_clean_activations[layer] = np.load(clean_acts_path)['activations']
    else:
        print(f"  Computing clean activations...")
        encoder_clean_activations[layer] = get_panorama_activations(
            model_enc, preprocess_enc, panorama_clean_images,
            model_name='predictive_coder',
            device=device, batch_size=64
        )
        clean_acts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(clean_acts_path, activations=encoder_clean_activations[layer])

    print(f"  Clean shape: {encoder_clean_activations[layer].shape}")

    # Pitch activations (for EV calculation)
    pitch_acts_path = current_file_path / f'data/ann_activations/predictive_coder_{layer}_panorama_pitch/acts.npz'

    if pitch_acts_path.exists():
        print(f"  Loading cached pitch activations")
        encoder_pitch_activations[layer] = np.load(pitch_acts_path)['activations']
    else:
        print(f"  Computing pitch activations...")
        encoder_pitch_activations[layer] = get_panorama_activations(
            model_enc, preprocess_enc, panorama_pitch_images,
            model_name='predictive_coder',
            device=device, batch_size=64
        )
        pitch_acts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pitch_acts_path, activations=encoder_pitch_activations[layer])

    print(f"  Pitch shape: {encoder_pitch_activations[layer].shape}")

# CELL 31
# Intermediate Layer Analysis - Unit 2: Extract encoder layer activations

encoder_layers = ['conv1', 'down1', 'down2']
encoder_clean_activations = {}
encoder_pitch_activations = {}

for layer in encoder_layers:
    print(f"\n=== Extracting {layer} activations ===")

    model_enc, preprocess_enc = load_model(
        'predictive_coder',
        base_path=current_file_path,
        output_layer=layer,
        device=device
    )

    # Clean activations (for classification and 3D viz)
    clean_acts_path = current_file_path / f'data/ann_activations/predictive_coder_{layer}_panorama_clean/acts.npz'

    if clean_acts_path.exists():
        print(f"  Loading cached clean activations")
        encoder_clean_activations[layer] = np.load(clean_acts_path)['activations']
    else:
        print(f"  Computing clean activations...")
        encoder_clean_activations[layer] = get_panorama_activations(
            model_enc, preprocess_enc, panorama_clean_images,
            model_name='predictive_coder',
            device=device, batch_size=64
        )
        clean_acts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(clean_acts_path, activations=encoder_clean_activations[layer])

    print(f"  Clean shape: {encoder_clean_activations[layer].shape}")

    # Pitch activations (for EV calculation)
    pitch_acts_path = current_file_path / f'data/ann_activations/predictive_coder_{layer}_panorama_pitch/acts.npz'

    if pitch_acts_path.exists():
        print(f"  Loading cached pitch activations")
        encoder_pitch_activations[layer] = np.load(pitch_acts_path)['activations']
    else:
        print(f"  Computing pitch activations...")
        encoder_pitch_activations[layer] = get_panorama_activations(
            model_enc, preprocess_enc, panorama_pitch_images,
            model_name='predictive_coder',
            device=device, batch_size=64
        )
        pitch_acts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pitch_acts_path, activations=encoder_pitch_activations[layer])

    print(f"  Pitch shape: {encoder_pitch_activations[layer].shape}")

# CELL 32
from src.explained_variance import nan_aware_gaussian_filter, masked_zoom, quantify_variance_corr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def process_unit_encoder(n, fold0, fold1, occ, zoom_factor, sigma):
    f0 = nan_aware_gaussian_filter(fold0[:, :, n], sigma=sigma, circular_axes=(1,))
    f1 = nan_aware_gaussian_filter(fold1[:, :, n], sigma=sigma, circular_axes=(1,))
    f0 = masked_zoom(f0, zoom_factor=zoom_factor)
    f1 = masked_zoom(f1, zoom_factor=zoom_factor)
    occ_z = masked_zoom(occ, zoom_factor=zoom_factor)
    ev_v1 = 1 - quantify_variance_corr(f0, f1, occ_z, axis=0)[0]
    ev_v2 = 1 - quantify_variance_corr(f0, f1, occ_z, axis=1)[0]
    return ev_v1, ev_v2

encoder_ev_results = {}

for layer_name in encoder_layers:
    print(f"\n=== Computing EV for {layer_name} ===")

    pitch_acts = encoder_pitch_activations[layer_name]
    n_locs_enc, n_imgs_enc, n_units_enc = pitch_acts.shape
    print(f"  {n_units_enc} units")

    pitch_acts_4d = pitch_acts.reshape(n_locs_enc, N_V2_PANORAMA, N_PITCHES, n_units_enc)

    fold0_enc, fold1_enc, fold0_occ_enc, fold1_occ_enc = get_splits_deterministic(
        pitch_acts_4d, FOLD0_PITCH_INDICES, FOLD1_PITCH_INDICES
    )
    occ_enc = fold0_occ_enc + fold1_occ_enc
    zoom_factor_enc = (1, N_V2_PANORAMA // N_POS_X)

    sigma = (1, 1)

    results = Parallel(n_jobs=-1)(
        delayed(process_unit_encoder)(n, fold0_enc, fold1_enc, occ_enc, zoom_factor_enc, sigma)
        for n in tqdm(range(n_units_enc), desc=f"  EV ({layer_name})", leave=False)
    )

    ev_v1 = np.array([r[0] for r in results])
    ev_v2 = np.array([r[1] for r in results])

    encoder_ev_results[layer_name] = (ev_v1, ev_v2)

    ev_out_dir = current_file_path / f'results/ann/predictive_coder_{layer_name}'
    ev_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ev_out_dir / 'ev_all_units_pitch.npz', ev_v1=ev_v1, ev_v2=ev_v2)
    print(f"  Saved to {ev_out_dir}")


# CELL 33
# Intermediate Layer Analysis - Unit 4: Visualize EV progression across encoder layers

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

print("=== Layer Summary Statistics ===")
for ax, (layer, (ev_v1, ev_v2)) in zip(axes, encoder_ev_results.items()):
    valid = np.isfinite(ev_v1) & np.isfinite(ev_v2) & (ev_v1 > 0) & (ev_v2 > 0)
    x, y = ev_v1[valid], ev_v2[valid]

    # Subsample for visualization
    n_show = min(500, len(x))
    idx = np.random.choice(len(x), n_show, replace=False)

    ax.scatter(x[idx], y[idx], s=20, alpha=0.5)

    vmin, vmax = 0, 1
    ax.plot([vmin, vmax], [vmin, vmax], '--', color='k', alpha=0.5, label='x=y')

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel('EV (V1)')
    ax.set_ylabel('EV (V2)')
    ax.set_title(layer)
    ax.set_aspect('equal')

    # Summary stats
    v1_dominant = np.sum(x > y) / len(x)
    denom = x + y
    denom[denom == 0] = 1e-8
    mean_v1_ratio = np.mean(x / denom)

    print(f"{layer:8s}: {v1_dominant:.1%} V1-dominant, mean V1 ratio: {mean_v1_ratio:.3f}, n={len(x)}")

plt.tight_layout()
plt.savefig(current_file_path / 'figs/predictive_coder_encoder_ev_scatter_pitch.pdf', bbox_inches='tight')
plt.show()

print("\nInterpretation:")
print("- Early layers (conv1) should show ~50% V1-dominant (no spatial specificity)")
print("- Deeper layers should show increasing V1-dominance (emerging place-like selectivity)")
print("- This progression supports the argument that spatial selectivity emerges from visual features")

# CELL 34
# Intermediate Layer Analysis - Unit 6: Plot conv1 units (early layer)

# Pick units with highest activation variance (most "interesting" filters)
conv1_var = np.var(conv1_3d, axis=(1, 2, 3))
top_conv1_units = np.argsort(conv1_var)[::-1]

print("=== Conv1 Units (Early Layer - Visual Features) ===")
print("These should NOT look like units or V2 units\n")

for i, unit_idx in enumerate(top_conv1_units[:3]):
    fig = plot_unit_2d_3d(
        conv1_3d,
        unit_idx,
        axes_names=("X", "Y", "V2"),
        overall_title=f"Conv1 Unit {unit_idx} (Early Visual Feature)",
        n_slices=4
    )
    fig.write_image(str(current_file_path / f'figs/conv1_unit_{unit_idx}_3d_pitch.pdf'))
    fig.show()

# CELL 35
# Intermediate Layer Analysis - Unit 7: Plot pre_pred units (late layer) for comparison

# Load the EV results to find high-V1 units (place-like)
ev_path = current_file_path / 'results/ann/predictive_coder_pre_pred/ev_all_units_pitch.npz'
if ev_path.exists():
    ev_data = np.load(ev_path)
    ev_v1_prepred = ev_data['ev_v1']
    ev_v2_prepred = ev_data['ev_v2']

    # Find units with high V1 selectivity (place-like)
    v1_ratio = ev_v1_prepred / (ev_v1_prepred + ev_v2_prepred + 1e-8)
    valid_ratio = np.isfinite(v1_ratio)
    top_place_units = np.argsort(v1_ratio)[::-1]
    top_place_units = [n for n in top_place_units if valid_ratio[n]]

    print("\n=== Pre_pred Units (Late Layer - Place-like) ===")
    print("These SHOULD look like units with localized activity fields\n")

    for i, unit_idx in enumerate(top_place_units[:3]):
        fig = plot_unit_2d_3d(
            prepred_3d,
            unit_idx,
            axes_names=("X", "Y", "V2"),
            overall_title=f"Pre_pred Unit {unit_idx} (Place-like, V1 ratio: {v1_ratio[unit_idx]:.2f})",
            n_slices=4
        )
        fig.write_image(str(current_file_path / f'figs/prepred_place_unit_{unit_idx}_3d_pitch.pdf'))
        fig.show()

    # Also show V2-dominant units
    v2_ratio = ev_v2_prepred / (ev_v1_prepred + ev_v2_prepred + 1e-8)
    top_v2_units = np.argsort(v2_ratio)[::-1]
    top_v2_units = [n for n in top_v2_units if valid_ratio[n]]

    print("\n=== Pre_pred Units (Late Layer - V2-like) ===")
    print("These might show V2-like tuning\n")

    for i, unit_idx in enumerate(top_v2_units[:2]):
        fig = plot_unit_2d_3d(
            prepred_3d,
            unit_idx,
            axes_names=("X", "Y", "V2"),
            overall_title=f"Pre_pred Unit {unit_idx} (V2-like, V2 ratio: {v2_ratio[unit_idx]:.2f})",
            n_slices=4
        )
        fig.write_image(str(current_file_path / f'figs/prepred_v2_unit_{unit_idx}_3d_pitch.pdf'))
        fig.show()
else:
    print("EV results for pre_pred not found - run the main analysis first")