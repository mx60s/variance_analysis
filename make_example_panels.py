"""Example-cell panels for the manuscript.

Draws the joint tuning of single cells from the saved rate maps and activation caches:

  pfeiffer RAT NEURON               location x HD map of a place cell
  dusz DATASET SESSION NEURON       location x HD map of a head direction cell
  dusz-hdmd DATASET SESSION NEURON  HD x MD map of a head direction cell
  model MODEL LAYER UNIT            location x HD map of a network unit (0-degree-pitch panoramas)
  paper                             the place cell and head direction cell example panels of the manuscript

NEURON is the index within the session's saved rate map (the N in the figure file names); UNIT is the
index within the layer's activation cache. The ResNet example panels in the supplement were made from an
earlier activation run than the current caches, and their unit numbers do not select the same units here. Figures go to figs/examples/ unless --out is given.

Examples (from the repo root):
  python make_example_panels.py pfeiffer 2 133
  python make_example_panels.py dusz-hdmd 1 A3701-191119 28
  python make_example_panels.py model resnet50 layer3 94
  python make_example_panels.py paper --out figs
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
from src.explained_variance import nan_aware_gaussian_filter
from src.ann.fig import plot_neuron_2d_3d

PFEIFFER = ROOT / "results/rodents/pfeiffer"
DUSZ = ROOT / "results/rodents/dusz"
ACTS = ROOT / "data/ann_activations"

# panels used in the manuscript: (kind, args, output name)
PAPER_PANELS = [
    ("pfeiffer", (4, 63), "pfeiffer_example_xy_rank1_rat4_N63"),
    ("pfeiffer", (2, 133), "pfeiffer_example_xy_rank0_rat2_N133"),
    ("dusz", (1, "A3702-191126", 13), "dusz_example_hd_rank0_A3702-191126_N13"),
    ("dusz", (1, "A3710-200609", 57), "dusz_example_hd_rank1_A3710-200609_N57"),
    ("dusz-hdmd", (1, "A3701-191119", 28), "dusz_hdmd_D1_A3701-191119_28_joint"),
    ("dusz-hdmd", (1, "A3711-200810b", 16), "dusz_hdmd_D1_A3711-200810b_16_joint"),
]


def location_hd_figure(rate_map, neuron, sigma=(1, 1, 4), clip_percentile=99):
    """rate_map: (nx, ny, n_hd, n_neurons). Smooths (HD circular), clips outliers, draws the 3-D panel."""
    m = nan_aware_gaussian_filter(rate_map[:, :, :, neuron], sigma, circular_axes=(0, 0, 1))
    m = np.clip(m, None, float(np.nanpercentile(m, clip_percentile)))
    return plot_neuron_2d_3d(
        m[None], 0, axes_names=("X", "Y", "HD"), n_slices=4, z_spacing=1.0,
        surf_opacity=1.0, grid_color="white", grid_width=1, z_label_degrees=True, firing_rate=True,
    )


def pfeiffer(rat, neuron):
    rate_map = np.load(PFEIFFER / f"Rat{rat}/ratemap_xyhd_20x20x60.npz")["rate_map"]
    return location_hd_figure(rate_map, int(neuron))


def dusz(dataset, session, neuron):
    rate_map = np.load(DUSZ / f"Dataset_{dataset}/{session}/fold_rate_xyhd_20x20x90.npz")["rate_map"]
    return location_hd_figure(rate_map, int(neuron))


def dusz_hdmd(dataset, session, neuron):
    rate_map = np.load(DUSZ / f"Dataset_{dataset}/{session}/fold_rate_hdmd_90x90.npz")["rate_map"]
    m = nan_aware_gaussian_filter(rate_map[:, :, int(neuron)], (1, 1), circular_axes=(0, 1))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(m.T, origin="lower", extent=[0, 360, 0, 360], cmap="viridis")
    ax.set_xlabel("HD (°)")
    ax.set_ylabel("MD (°)")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Firing rate (spikes/s)")
    fig.tight_layout()
    return fig


def model(model_name, layer, unit):
    z = np.load(ACTS / f"{model_name}_{layer}_panorama_clean/acts.npz")
    acts, locs = z["activations"], z["locations"]  # (n_locations, n_hd, n_units), (n_locations, 2)
    xs, ys = np.unique(locs[:, 0]), np.unique(locs[:, 1])
    grid = np.full((len(xs), len(ys), acts.shape[1]), np.nan)
    grid[np.searchsorted(xs, locs[:, 0]), np.searchsorted(ys, locs[:, 1])] = acts[:, :, int(unit)]
    return plot_neuron_2d_3d(
        grid[None], 0, axes_names=("X", "Y", "HD"), n_slices=4, z_spacing=1.0,
        surf_opacity=1.0, grid_color="white", grid_width=1, z_label_degrees=True,
    )


KINDS = {"pfeiffer": pfeiffer, "dusz": dusz, "dusz-hdmd": dusz_hdmd, "model": model}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "write_image"):
        fig.write_image(str(path))
    else:
        fig.savefig(path)
        plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=list(KINDS) + ["paper"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out", default="figs/examples", help="output directory (default: figs/examples)")
    a = ap.parse_args()
    out = ROOT / a.out

    if a.kind == "paper":
        for kind, args, name in PAPER_PANELS:
            try:
                save(KINDS[kind](*args), out / f"{name}.pdf")
            except FileNotFoundError as e:
                print(f"Skipping {name}: missing {e.filename}")
        return

    fig = KINDS[a.kind](*a.args)
    save(fig, out / f"{a.kind}_{'_'.join(map(str, a.args))}.pdf")


if __name__ == "__main__":
    main()
