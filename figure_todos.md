# Figure TODOs from Overleaf Red-Text Notes

## Fig. 2 (GLM) — `Figures/fig2_glm.pdf`
**Source notebook:** `notebooks/glm.ipynb`

| TODO (line) | What's needed | Status |
|---|---|---|
| **L182**: *"Plot the marginal tuning in addition to the GLM fit."* | Currently the notebook plots true 2D heatmap, GLM-fit heatmap, and 1D cross-sections. Need to add the **empirical marginal tuning curves** (bin-averaged) alongside the GLM-predicted marginals so the reader can compare both. | The GLM notebook already computes marginals — just needs an extra line series on the existing 1D panels. |

---

## Fig. 1 (Correlated Sampling) — `Figures/CNN_fig1.pdf`
**Source notebook:** `notebooks/marginals_example.ipynb`

No red-text TODOs on the figure itself. The figure currently exists and shows 3 correlation levels. The only related TODO is the **methods text** (L553), which was covered in `methods_snippets.md`.

---

## Fig. 3 (Estimation Method Validation) — `Figures/fig_3_method.pdf`
**Source notebook:** `notebooks/binning_bias_fig.ipynb`

| TODO (line) | What's needed | Status |
|---|---|---|
| **L253**: *"Was the simulation based on assuming a continuous tuning function? If it is, it might also be useful to show a version with a discretized tuning function (i.e., a look up table)."* | The notebook already implements **both** continuous and discrete versions (`get_rot_counts_continuous` vs `get_rot_counts_discrete`), saving to `figs/smoothed_binning_fig_corr.pdf` and `figs/discrete_binning_fig.pdf`. Need to either add the discrete version as extra panels (d-g) in the main figure, or include it as a supplementary figure. | Code exists — needs to be composed into the final PDF. |

---

## Fig. 4 (Predictive Coder) — `Figures/Fig_pred coder_ver2.pdf`
**Source notebook:** `notebooks/model_neurons.ipynb`

| TODO (line) | What's needed | Status |
|---|---|---|
| **L284** (commented): *"SI Fig. XX for the performance of the agent in this prediction task."* | Generate an SI figure showing the predictive coder's training loss / MSE over epochs, and example predicted vs actual frames. | Would need to add logging during training or load training logs. |
| **L292**: *"To see if our results generalize to other layers... (see SI Fig. XX)."* | Generate EV scatter plots for **all layers** (not just attention layers). The notebook already loops over layers — need to save per-layer scatter plots as SI figures. | `model_neurons.ipynb` can do this; just need to run it for `conv1`, `down1`, `down2` in addition to attention layers. |
| **L304** (commented): *"Add a figure showing the environment we constructed. Also, which equation did we use to quantify the contribution? I think we should use the noise-free version."* | (1) Add a screenshot/render of the Minecraft environment. (2) Clarify in the caption that Eq. 1 (noise-free version) was used for model neurons (since model activations are deterministic). | Environment screenshot needs to be captured from Minecraft/Malmo. The equation clarification is a caption edit. |

---

## Fig. 5 (CNN / ResNet) — `Figures/resnet_fig_2.pdf`
**Source notebook:** `notebooks/model_neurons.ipynb`

| TODO (line) | What's needed | Status |
|---|---|---|
| **L323**: *"Show some 3-D plots as those in Fig. 4?"* | Add example 3D joint tuning plots (location x HD) for individual CNN units, similar to the predictive coder examples. | `src/ann/fig.py` has `plot_neuron_2d_3d()` which generates these via Plotly. Need to select example CNN units and export to PDF. |
| **L328** (commented): *"Add some example neurons tuning curves, and the quantification using our method."* | Same as above — pick representative CNN units showing entangled tuning, plot their joint tuning maps alongside the EV scatter. | Plotting code exists; needs curation of example units. |

---

## Fig. 6 (Pfeiffer Place Cells) — `Figures/fig_4_pfeiffer.pdf`
**Source notebook:** `notebooks/bio_neurons.ipynb`

| TODO (line) | What's needed | Status |
|---|---|---|
| **L381**: *"Add the chance level for decoding error of XY and HD in the last panel."* | Add horizontal dashed lines showing chance-level MAE on the decoding error plots. | `1d_decoding.ipynb` already computes chance decoding (`run_chance_decoding=True`, mode="mode"). The values exist in the decode output — just need to overlay them on panels (d) and (e). |
| **L396**: *"(MAE= 20.7 cm)"* — red-flagged value | Verify this number from decoding output and un-red it, or update if incorrect. | Check `pfeiffer_decode_solo.py` output. |
| **L385** (commented): *"Add a figure on 3-D tuning curves of place cells, and add a figure to quantify variance explained by place and HD."* | The main figure already has example joint tuning (panels a,b) and EV quantification (panel c). This comment appears to be from an earlier draft and may be resolved. Verify panels match the caption. | Likely already done — verify. |

---

## Fig. 7 (Duszkiewicz HD Cells) — `Figures/fig_5_dusz.pdf`
**Source notebook:** `notebooks/bio_neurons.ipynb`

| TODO (line) | What's needed | Status |
|---|---|---|
| **L398**: *"brain area XX of rats"* and *"XXX"* | Fill in the brain area name: **postsubiculum (PoSub)**. The code (`dusz_ratemaps.py`) filters for PoSub excitatory HD cells. Also note these are **mice**, not rats. | Text fix only. |
| **L417**: *"Rescale the middle panels and add separate colorbar to the middle panels? Would it be useful to show the HD tuning curve for different part of the space?"* | (1) Add individual colorbars to the joint location x HD heatmap panels (b,d,f). (2) Consider adding an extra panel or SI figure showing HD tuning curves conditioned on different spatial quadrants. | Plotting change in the notebook that generates this figure. |
| **L421** (commented): *"Head direction cells contain little information about space?"* | This is a rhetorical note, not an action item. The EV analysis already shows this. | No action needed. |
| **L423** (commented): *"Add a figure on the HD decoding results."* | Add XY decoding results from HD cell population showing high error. | `1d_decoding.ipynb` already generates `figs/dusz_xy_decoding_vs_chance.pdf` and `figs/dusz_hd_decoding_vs_chance.pdf`. These need to be incorporated into the main figure or referenced as SI. |

---

## Fig. 8 (MD vs HD Decoding) — `Figures/mdhd_decoding.pdf`
**Source notebook:** `notebooks/bio_neurons.ipynb` + `notebooks/1d_decoding.ipynb`

| TODO (line) | What's needed | Status |
|---|---|---|
| **L446**: *"Add colorbar to b. Also, when I opened the pdf in Adobe Illustrator, panel b got lost. You can update a pdf for panel b separately. Also, add a panel showing the joint distribution of MD and HD."* | Three items: (1) Add colorbar to the joint HD x MD heatmap panel. (2) Fix the PDF export so panel b renders in Illustrator (likely a Plotly-to-PDF rendering issue — may need to export as a separate file or use matplotlib instead). (3) Add a new panel showing the **behavioral joint distribution** of MD and HD — `bio_neurons.ipynb` already generates this as `figs/hdmd_occupancy_all-corr.pdf`. | Occupancy plot exists. Colorbar and PDF fix are plotting tweaks. |

---

## Supplementary Figures (not yet created)

| TODO | What's needed | Source |
|---|---|---|
| **SI: Predictive coder performance** (L284) | Training curve + example next-frame predictions | Needs training logs |
| **SI: Per-layer EV for predictive coder** (L292-293) | EV scatter for conv1, down1, down2 layers | `model_neurons.ipynb` — run for extra layers |
| **SI: CNN 3D tuning examples** (L323, L328, L358) | 3D joint tuning plots for CNN units | `fig.py:plot_neuron_2d_3d()` exists |
| **SI: HD decoding from HD cell population** (L423) | Show low XY decode error, high HD decode accuracy | `1d_decoding.ipynb` outputs exist |
| **SI: Minecraft environment screenshot** (L304) | Render of the 17x17 block arena | Capture from Malmo/Minecraft |

---

## Non-Figure Red Text (text-only fixes)

- **L396**: Fill in MAE values for HD decoding ("XX deg for n=XX session")
- **L398**: Replace "brain area XX" with "postsubiculum (PoSub)" and "rats" with "mice"
- **L400**: Fill in MAE values for location and HD decoding from HD cells
- **L435**: Fill in MAE for decoded MD ("XXX")
- **L642**: Fill in MSE for predictive coder training
