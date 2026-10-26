#!/usr/bin/env python
"""
Standalone runner for the deep-net EV pipeline (Task 1), reusing model_neurons.ipynb's
exact functions. Colab bits stripped; device=MPS. Lazy model-loading so validating a
cached layer doesn't download weights.

Usage:
  EV_MODELS='{"resnet50":["layer3"]}' .venv/bin/python notebooks/run_model_layers.py
Env:
  EV_MODELS  JSON dict {model: [layers]}   (default: resnet50/layer3)
  EV_DEVICE  mps | cpu | cuda              (default: mps)
"""
import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from src.ann.get_activations import load_model
from src.ann.classify_neurons import score_neurons, classify_neurons
from src.explained_variance import nan_aware_gaussian_filter, masked_zoom, quantify_variance_corr

current_file_path = Path(ROOT)
DEVICE = os.environ.get("EV_DEVICE", "mps")

# ---- constants (verbatim from notebook) ----
ENV_X_MIN, ENV_X_MAX, ENV_Y_MIN, ENV_Y_MAX = 120, 136, 197, 213
N_POS_X = ENV_X_MAX - ENV_X_MIN + 1          # 17
N_HD_PANORAMA, N_PITCHES = 90, 4
FOLD0_PITCH_INDICES, FOLD1_PITCH_INDICES = [0, 2], [1, 3]

# ---- functions (verbatim from notebook) ----
def load_panorama_images(panorama_dir, with_pitch=False):
    npz_files = sorted(glob.glob(str(panorama_dir / 'loc_*.npz')))
    imgs_list, locs_list = [], []
    for f in tqdm(npz_files, desc="Loading panorama"):
        d = np.load(f); imgs_list.append(d['rgb'])
        parts = Path(f).stem.replace('loc_', '').split('_')
        locs_list.append([float(parts[0]), float(parts[1])])
    return np.stack(imgs_list, axis=0), np.array(locs_list)

def get_panorama_activations(model, preprocess_fn, panorama_images, model_name,
                             device='cuda', batch_size=32, max_units=8000, sample_seed=413):
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device).eval()
    n_locs, n_imgs = panorama_images.shape[:2]
    flat = panorama_images.reshape(-1, *panorama_images.shape[2:])
    is_pred = model_name.startswith("predictive_coder"); seq_len = 10 if is_pred else None
    selected_idx, acts = None, []
    with torch.no_grad():
        for i in tqdm(range(0, flat.shape[0], batch_size), desc=f"  acts[{model_name}/{n_imgs}]"):
            batch = flat[i:i+batch_size]
            if is_pred:
                batch = np.repeat(batch[:, None, ...], seq_len, axis=1)
            x = preprocess_fn(batch).to(device)
            out = model(x)
            if isinstance(out, dict): out = next(iter(out.values()))
            if is_pred: out = out[:, -1]
            out = out.flatten(1)
            if out.shape[1] > max_units:
                if selected_idx is None:
                    g = torch.Generator().manual_seed(sample_seed)  # CPU generator, so the same units are picked on any device
                    selected_idx = torch.randperm(out.shape[1], generator=g)[:max_units].to(out.device)
                out = out.index_select(1, selected_idx)
            acts.append(out.cpu().numpy())
    return np.concatenate(acts, axis=0).reshape(n_locs, n_imgs, -1)

def get_splits_deterministic(visits, f0i, f1i):
    s1, s2 = visits.shape[0], visits.shape[1]
    fold0 = np.nanmean(visits[:, :, f0i, :], axis=2)
    fold1 = np.nanmean(visits[:, :, f1i, :], axis=2)
    return fold0, fold1, np.full((s1, s2), len(f0i), float), np.full((s1, s2), len(f1i), float)

def process_neuron_loop(n, fold0, fold1, occ, zoom_factor, sigma):
    f0 = nan_aware_gaussian_filter(fold0[:, :, n], sigma=sigma, circular_axes=(1,))
    f1 = nan_aware_gaussian_filter(fold1[:, :, n], sigma=sigma, circular_axes=(1,))
    f0 = masked_zoom(f0, zoom_factor=zoom_factor); f1 = masked_zoom(f1, zoom_factor=zoom_factor)
    occ_z = masked_zoom(occ, zoom_factor=zoom_factor)
    ev_xy = 1 - quantify_variance_corr(f0, f1, occ_z, axis=0)[0]
    ev_hd = 1 - quantify_variance_corr(f0, f1, occ_z, axis=1)[0]
    return ev_xy, ev_hd

# ---- lazy panorama loaders ----
_pano = {}
def clean():
    if 'c' not in _pano: _pano['c'] = load_panorama_images(current_file_path/'data/minecraft/panorama-90yaw', False)
    return _pano['c']
def pitch():
    if 'p' not in _pano: _pano['p'] = load_panorama_images(current_file_path/'data/minecraft/panorama-90yaw-pitch', True)
    return _pano['p']

def scatter(model_name, layer_name, ev_xy, ev_hd, is_place):
    v = np.isfinite(ev_xy) & np.isfinite(ev_hd); x, y, ip = ev_xy[v], ev_hd[v], is_place[v]
    rng = np.random.default_rng(42)
    oth = np.where(~ip)[0]; pl = np.where(ip)[0]
    if len(oth) > 250: oth = rng.choice(oth, 250, replace=False)
    if len(pl) > 75:  pl = rng.choice(pl, 75, replace=False)
    plt.figure(figsize=(4, 4))
    plt.scatter(x[oth], y[oth], s=22, alpha=0.7, label=f"other (n={len(oth)})")
    if len(pl): plt.scatter(x[pl], y[pl], s=22, color="red", label=f"place (n={len(pl)})")
    plt.plot([0, 1], [0, 1], "--", lw=1, color="k", alpha=0.6)
    plt.gca().set_aspect("equal", "box"); plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("EV (XY)"); plt.ylabel("EV (HD)"); plt.title(f"{model_name}/{layer_name}", fontsize=9)
    plt.legend(frameon=False, fontsize=7); plt.tight_layout()
    out = current_file_path / f"figs/{model_name}_{layer_name}_ev_pitch.pdf"; plt.savefig(out); plt.close()
    return out

def run_layer(model_name, layer_name):
    print(f"\n{'='*60}\n=== {model_name} / {layer_name} ===\n{'='*60}")
    clean_p = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_clean/acts.npz'
    pitch_p = current_file_path / f'data/ann_activations/{model_name}_{layer_name}_panorama_pitch/acts.npz'
    model = preprocess = None
    def get_model():
        nonlocal model, preprocess
        if model is None:
            print(f"  loading model on {DEVICE} ...")
            model, preprocess = load_model(model_name, base_path=current_file_path,
                                           output_layer=layer_name, device=DEVICE)
        return model, preprocess

    # ---- clean acts -> classify place cells ----
    if clean_p.exists():
        clean_acts = np.load(clean_p)['activations']; print("  clean acts: cached")
    else:
        m, pp = get_model(); imgs, locs = clean()
        clean_acts = get_panorama_activations(m, pp, imgs, model_name, DEVICE, batch_size=64)
        clean_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(clean_p, activations=clean_acts, locations=locs)
    n_loc, n_hd, n_neu = clean_acts.shape; print(f"  n_neurons={n_neu}")
    flat_p = clean_p.parent / 'acts_flat.npz'; np.savez(flat_p, activations=clean_acts.reshape(-1, n_neu))
    cfg = {"model_name": model_name, "output_layer": layer_name,
           "env_bounds": [ENV_X_MIN, ENV_X_MAX, ENV_Y_MIN, ENV_Y_MAX], "n_rotations": N_HD_PANORAMA}
    scores_fp, _ = score_neurons([flat_p], cfg, base_dir=current_file_path, sample_partial=False,
                                 pixel_min_threshold=1, use_old_place_method=False, n_visits=1)
    chart = np.load(scores_fp, allow_pickle=True)
    classes = classify_neurons(chart)
    place = np.asarray(classes["exclusive_place_cells_indices"])
    is_place = np.isin(np.arange(n_neu), place); print(f"  {len(place)} place cells")

    # ---- pitch acts -> EV ----
    if pitch_p.exists():
        pitch_acts = np.load(pitch_p)['activations']; print("  pitch acts: cached")
    else:
        m, pp = get_model(); imgs, locs = pitch()
        pitch_acts = get_panorama_activations(m, pp, imgs, model_name, DEVICE, batch_size=64)
        pitch_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pitch_p, activations=pitch_acts, locations=locs)
    v4 = pitch_acts.reshape(n_loc, N_HD_PANORAMA, N_PITCHES, n_neu)
    f0, f1, o0, o1 = get_splits_deterministic(v4, FOLD0_PITCH_INDICES, FOLD1_PITCH_INDICES)
    occ = o0 + o1; zoom = (1, N_HD_PANORAMA // N_POS_X); sigma = (1, 1)
    res = np.array(Parallel(n_jobs=-1)(
        delayed(process_neuron_loop)(n, f0, f1, occ, zoom, sigma) for n in tqdm(range(n_neu), desc="  EV")))
    ev_xy, ev_hd = res[:, 0], res[:, 1]
    od = current_file_path / f"results/ann/{model_name}_{layer_name}"; od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "ev_all_neurons_pitch.npz", ev_xy=ev_xy, ev_hd=ev_hd, is_place=is_place, n_neurons=n_neu)
    out = scatter(model_name, layer_name, ev_xy, ev_hd, is_place)
    v = np.isfinite(ev_xy) & np.isfinite(ev_hd)
    print(f"  RESULT EV_xy_med={np.median(ev_xy[v]):.3f} EV_hd_med={np.median(ev_hd[v]):.3f} "
          f"xy_dominant={np.mean(ev_xy[v]>ev_hd[v]):.1%} -> {out}")

if __name__ == "__main__":
    MODELS = json.loads(os.environ.get("EV_MODELS", '{"resnet50":["layer3"]}'))
    for m, layers in MODELS.items():
        for l in layers:
            run_layer(m, l)
    print("\nDONE")
