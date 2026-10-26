# Place Cell Variance Analysis

Code for the manuscript *Invariance and selectivity to spatial variables in deep neural networks and the brain* (source in `Spatial_maps_in_CNNs/`).

The project asks how much each stimulus variable (location, head direction, movement direction) contributes to a neuron's firing, without assuming the variables act independently. Marginal tuning curves and additive GLMs can mislead when variables are sampled together; `notes_marginals_and_glm.md` explains why. The method builds a joint rate map from two independent halves of the data (folds). For a variable *x*, it then measures how similar the tuning to the other variables stays as *x* changes, using cross-fold Pearson correlations.

The same analysis is applied to:

- place cells from rat CA1 (Pfeiffer & Foster 2013)
- head-direction cells from mouse postsubiculum (Duszkiewicz et al. 2024)
- units in deep networks (ResNet-50, VGG16 and a predictive-coding network) shown first-person views from a Minecraft environment

## The metric

For a variable *x*, EV(*x*) = 1 − ⟨C_ij⟩ over i ≠ j. Here C_ij is the Pearson correlation between the fold-0 rate map at value *i* of *x* and the fold-1 rate map at value *j*. EV is roughly in [0, 1] and can go slightly above 1 when the off-diagonal correlations are negative.

`src/explained_variance.py:quantify_variance_corr(fold0, fold1, occupancy, axis)` returns the **raw** mean off-diagonal correlation, plus the mean diagonal (same-value) correlation, which measures reliability. Callers apply `1 −` themselves:

```python
ev_xy = 1 - quantify_variance_corr(fold0, fold1, occ, axis=0)[0]   # location
ev_hd = 1 - quantify_variance_corr(fold0, fold1, occ, axis=1)[0]   # head direction
```

Rate maps are laid out with axis 0 = location and axis 1 = head direction, for both biological and model data. Before September 2026 the code used an unbounded sum-of-squares estimator. Any figure whose axes run to 1e4–1e7 is from that version and is out of date.

## Installation

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .            # rodent analyses
pip install -e ".[ann]"     # plus the deep-network analyses (torch etc.)
```

On a CUDA machine you can also install `xformers`. `src/ann/predictive_coder.py` falls back to PyTorch attention when it isn't installed, which is the case on Apple Silicon.

## Data

None of the data is in the repository. The shared Google Drive folder has exactly the files below; copy its `data/` folder into the repo root. Only these files are read:

```
data/
  rodents/
    pfeiffer/
      Rat1/Open1/{Position_Data.mat, Spike_Data.mat}     # also Rat2, Rat3, Rat4
    dusz/
      Dataset_metadata.xlsx
      Dataset_1/<session>/Data/{BehavEpochs.mat, Tracking_data.csv, CellTypes.mat, SpikeData.mat, Velocity.mat}
      Dataset_2/<session>/Data/...
  minecraft/
    panorama-90yaw/         loc_+X_+Y.npz, 289 files (17x17 grid), 90 headings at 0 pitch
    panorama-90yaw-pitch/   loc_+X_+Y.npz, 289 files, 90 headings x 4 pitches
  weights/
    predictive_coder.ckpt
```

- **Pfeiffer & Foster 2013** (Nature 497:74–79): CA1 place cells from 4 rats navigating an open arena. The data aren't public; we got them directly from the authors' lab. Only the `Open1` session is used.
- **Duszkiewicz et al. 2024** (Nat Neurosci 27:782–792): postsubiculum recordings in mice. The data is public; see the paper's data availability statement. Only Datasets 1 and 2 are used, and within them only the square-arena sessions, chosen from `Dataset_metadata.xlsx`.
- **Minecraft panoramas**: first-person views recorded with the Malmo platform at every block of a 17x17 area. The recording script is not in this repository.
- **Predictive coder checkpoint**: our own training of the architecture from Gornet & Thomson 2024 ([jgornet/predictive-coding-recovers-maps](https://github.com/jgornet/predictive-coding-recovers-maps)). The training code is not in this repository. `load_model` raises an error if the checkpoint is missing. ResNet-50 and VGG16 use the torchvision ImageNet weights, which download automatically.

Generated files go to `results/` (rate maps, EV values, decoding output) and `data/ann_activations/` (cached network activations; the pitch set is about 3 GB per layer). The Drive folder includes the rodent `results/` and the activation caches for the layers in the paper. Everything there can be regenerated, but the network activations take a while.

The rate-map builders split each bin's samples into the two folds at random, with a fixed seed (`seed=0`), so a rerun gives identical results.

For network layers with more than 8,000 units, the activation code keeps a random subset of 8,000 units. The cached activations on the Drive were made on a CUDA GPU in Colab, where that subset came out differently from the one behind the paper's figures, because PyTorch's seeded random numbers differ between devices. The scatters therefore show different individual units but the same distribution. The code now draws the subset on the CPU, so new runs pick the same units on any machine. That subset is different again from both of those, so delete a layer's cache and rerun it if you need it to match the code exactly.

## Running

The notebooks were written in Google Colab, so their first cells mount Google Drive and install packages. To run one locally without Jupyter, use `notebooks/run_nb.py`. It skips the Colab cells, runs from the repo root and uses MPS instead of CUDA:

```bash
python notebooks/run_nb.py notebooks/bio_neurons.ipynb                 # whole notebook
python notebooks/run_nb.py notebooks/bio_neurons.ipynb --only 15-20    # selected cells
```

Opening the notebooks in Jupyter also works if you skip the setup cells at the top. Each notebook also tries to install Helvetica from `fonts/Helvetica.ttc`, which isn't in the repo; `run_nb.py` substitutes DejaVu Sans.

| What | Produces (in `figs/`) |
|---|---|
| `notebooks/bio_neurons.ipynb` | Rate maps and EV for the Pfeiffer place cells and the Duszkiewicz HD cells: `*_ev_total_best_fit-corr.pdf`, `hdmd_occupancy_*-corr.pdf` |
| `notebooks/model_neurons.ipynb` | Network activations on the panoramas, unit classification, EV per layer: `*_ev_pitch.pdf`, `predictive_coder_conv1_filters.pdf` |
| `notebooks/run_model_layers.py` | The same EV pipeline as a script, for batch runs: `EV_MODELS='{"resnet50":["layer3"]}' EV_DEVICE=mps python notebooks/run_model_layers.py` |
| `notebooks/1d_decoding.ipynb` | Bayesian decoding of location, HD and MD vs chance: `*_decoding_vs_chance.pdf` |
| `notebooks/rodent_summary.py` | Prints the rodent numbers quoted in the manuscript (EV fractions and medians, finite-sampling floors, HD–MD coupling, mean decoding errors) from `results/rodents/`. Run it after the two notebooks above. |
| `notebooks/binning_bias_fig.ipynb` | Simulations of the estimator and its finite-sampling floor: `smoothed_binning_fig_corr.pdf`, `binning_noise_floor_sweep.*` |
| `notebooks/marginals_example.ipynb`, `notebooks/glm.ipynb` | Motivating examples for marginal tuning and GLMs: `fig2.pdf`, `glm*.pdf` |
| `make_method_figure.py`, `make_decoding_figure.py` | Schematics: `method_schematic.*`, `decoding_schematic.*` |
| `make_example_panels.py` | Example-cell panels (joint location × HD and HD × MD maps of single cells, or of single network units). `python make_example_panels.py paper` redraws the place cell and HD cell examples of the manuscript into `figs/examples/`. |

The rate-map builders and decoders are in `src/` (`pfeiffer_ratemaps.py`, `dusz_ratemaps.py`, `*_decode_solo.py`). Each can also be run from the command line; use `--help` for the options.

## Rate-map parameters

The results only reproduce with these settings. Don't change them without rechecking the HD-cell scatter.

- **Pfeiffer, location x HD:** 20x20 location bins, 60 HD bins, tau = 0.25 s, speed > 5 cm/s, occupancy floor 5. Smoothed with σ = (1, 2), wrapping only the HD axis.
- **Duszkiewicz, location x HD:** 20x20 location bins, 90 HD bins, `vel_threshold=2`, `floor=5`. A higher floor (e.g. 10) mostly keeps stationary periods, which makes HD cells look location-tuned.
- **Duszkiewicz, HD x MD:** 90 x 90 bins, `vel_threshold=1`, `floor=10`. Both axes are circular.

## Current results

These are the numbers in the manuscript:

- Place cells: EV_xy > EV_hd for 99% of cells (n = 551).
- HD cells: EV_hd > EV_xy for 94% of cells (n = 1,489, 30 sessions).
- HD vs MD: EV_hd > EV_md for 100% of cells (n = 1,489).
- Trained network layers lie on the diagonal (EV_xy ≈ EV_hd ≈ 0.9). The untrained ResNet-50 is the exception.

Each EV comes with a finite-sampling floor, 1 − ⟨C_ii⟩: the value EV takes for a neuron that is invariant to that variable. Median floors and reliability-normalized contributions, (⟨C_ii⟩ − ⟨C_ij⟩)/⟨C_ii⟩, with the pipeline settings above:

- Place cells: EV_hd 0.49 with a floor of 0.19; normalized 0.37 for HD and 0.93 for location. So place fields keep their location across head directions, while their shape changes modestly with head direction.
- HD cells: EV_xy 0.67 with a floor of 0.32; normalized 0.52 for location and 0.85 for HD.
- HD vs MD: EV_md 0.18 with a floor of 0.04; normalized 0.14.

The measure ignores gain changes, because Pearson correlation is invariant to scale and offset. Head-direction modulation of place-field peak rate therefore doesn't show up in EV_hd.

In the Duszkiewicz sessions, HD and movement direction are only weakly coupled. The HD x MD joint map is therefore well populated, so EV_hd ≫ EV_md is not an artifact of sampling.

## Known gaps

- The ResNet example panels in the supplement (`resnet50_layer*_N*.pdf`) were drawn from a different 8,000-unit subset than the one in the current activation caches, so `make_example_panels.py model` can't redraw those same units.

## Repository layout

```
src/                  estimator, rate-map builders, decoders; src/ann/ for model loading and unit classification
notebooks/            analysis notebooks and the two runner scripts
figs/                 figures used in the manuscript and their current siblings
Spatial_maps_in_CNNs/ manuscript source (compile with `tectonic manuscript.tex`)
research_review/      planning and review notes for next steps (not part of the pipeline)
```
