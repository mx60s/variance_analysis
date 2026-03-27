# Methods Snippets for Overleaf Paper

These snippets address the `\textcolor{red}{...}` TODO items in the overleaf manuscript, drawn from the codebase.

---

## 1. Simulations and Parameters for Fig. 1 (Marginal Tuning Under Correlated Sampling)

> **Overleaf TODO (line 553):** *"Maggie to add a section describing the simulations and parameters used in Fig. 1-2."*

**Source:** `notebooks/marginals_example.ipynb`

### LaTeX snippet

```latex
\subsubsection*{Simulations for Fig.~\ref{fig:fig1}}

To illustrate the effect of correlated stimulus sampling on marginal tuning curves (Fig.~\ref{fig:fig1}), we simulated a hypothetical neuron whose firing rate depends only on variable $x$:
\[
r(x, y) = r_{\mathrm{peak}} \exp\!\Bigl(-\frac{x^2}{2\sigma_x^2}\Bigr),
\]
with $r_{\mathrm{peak}} = 1.0$ (arbitrary units) and $\sigma_x = 1.0$.
The neuron is therefore completely invariant to~$y$.

We sampled $N = 300$ stimulus pairs $(x_i, y_i)$ under three correlation conditions.
In each condition, $x_i$ was drawn uniformly from $[-4, 4]$.
The corresponding $y_i$ was drawn from a truncated normal distribution
$y_i \sim \mathcal{N}(x_i, \sigma_{\mathrm{cond}}^2)$ clipped to $[-4, 4]$
(via rejection sampling), where the conditional standard deviation was
\[
\sigma_{\mathrm{cond}} = \sigma_{\max}(1-\rho) + \sigma_{\min}\,\rho,
\]
with $\sigma_{\max} = 1.20 \times 4 = 4.8$, $\sigma_{\min} = 0.10 \times 4 = 0.4$,
and $\rho \in \{0.0, 0.6, 0.9\}$.
Higher $\rho$ produces stronger correlation between $x$ and $y$.

Marginal tuning curves for $y$ were estimated by binning $y$ into 30 equally spaced
bins and computing the mean firing rate per bin.
Rates were smoothed with a 1-D Gaussian kernel ($\sigma = 1$~bin).
Bins with fewer than 5 samples were excluded.
All simulations used a fixed random seed (numpy seed 42) for reproducibility.
```

---

## 2. Simulations and Parameters for Fig. 2 (GLM Failure)

**Source:** `notebooks/glm.ipynb`

### LaTeX snippet

```latex
\subsubsection*{Simulations for Fig.~\ref{fig:glm}}

To demonstrate the failure of GLM to capture non-multiplicative joint tuning
(Fig.~\ref{fig:glm}), we simulated neurons whose response depends on a
rotated coordinate $u = x\cos\theta - y\sin\theta$:
\[
r(x,y) = \exp\!\biggl(\alpha + \frac{-(u - \mu)^2}{2\sigma_u^2}\biggr),
\]
with baseline log-rate $\alpha = \ln 5$, $\mu = 0$, $\sigma_u = 1.2$, and
rotation angles $\theta = 60\degree$ (Fig.~\ref{fig:glm}a--c) and
$\theta = 45\degree$ (Fig.~\ref{fig:glm}d--f).
Stimulus samples $(x_i, y_i)$ were drawn independently from $\mathcal{N}(0, 1)$
($N = 20{,}000$ samples per condition), and spike counts were generated from a
Poisson distribution with rate $r(x_i, y_i)$.

A Poisson GLM with quadratic polynomial features,
$\log r = \beta_0 + \beta_1 x + \beta_2 y + \beta_3 x^2 + \beta_4 y^2$,
was fit to the simulated spikes.
Marginal tuning curves $\hat{f}(x)$ and $\hat{g}(y)$ were evaluated from the GLM fit by fixing the other variable at its mean.
The joint tuning and marginal slices were visualized on a
$160\times160$ grid spanning $[-3, 3]$ in both dimensions.
```

---

## 3. Simulations and Parameters for Fig. 3 (Explained Variance Estimation Method)

> **Overleaf TODO (line 607):** *"Maggie to add a section describing the simulations and the parameters used in Fig. 3"*

**Source:** `notebooks/binning_bias_fig.ipynb`, `src/explained_variance.py`

### LaTeX snippet

```latex
\subsubsection*{Simulations for Fig.~\ref{fig:estimation}}

To validate the split-half variance estimation procedure (Fig.~\ref{fig:estimation}),
we simulated neurons with a rotated 2-D Gaussian tuning curve.
The firing rate of the simulated neuron is
\[
r(s_1, s_2) = \exp\!\biggl(\beta_0 + A\,\exp\!\Bigl(-\frac{u^2}{2\sigma^2}\Bigr)\biggr),
\]
where $u = (s_1 - c)\cos\theta - (s_2 - c)\sin\theta$ is the stimulus projected onto the preferred axis rotated by angle~$\theta$,
$c = 0.4$ is the tuning center, $\sigma = 0.1$ is the tuning width,
$\beta_0 = \ln(5)$ sets a baseline rate of 5~spikes/s,
and $A = \ln(35) - \ln(5)$ yields a peak rate of 35~spikes/s.
Spike counts were drawn from $\mathrm{Poisson}(r)$.

We tested three rotation angles, $\theta \in \{0\degree, 25\degree, 45\degree\}$.
At $\theta = 0\degree$, the neuron is selective to $s_1$ and invariant to $s_2$;
at $\theta = 45\degree$, both variables contribute equally.
Stimuli $(s_1, s_2)$ were drawn uniformly from $[0, 1]^2$ ($N = 1{,}000$ samples per simulation)
and discretized into a $15 \times 15$ spatial grid.
For each rotation angle, we ran 600 independent simulations (different random seeds).

In each simulation, spike counts were accumulated per bin and randomly split into
two equal folds.
The mean firing rate in each fold served as the fold-specific rate map estimate.
Pairwise Pearson correlations were computed between 1-D slices
(tuning curves conditioned on different values of the other variable)
across the two folds.
The mean off-diagonal correlation quantifies the variance explained by a given variable,
and the mean diagonal correlation (reliability) captures noise-induced variability.
The fraction of variance explained by each variable was computed as the ratio
of its off-diagonal correlation to the sum across both variables.
Ground truth fractions were computed from the noiseless discrete rate map using the
same correlation-based metric.
```

---

## 4. Predictive Coding Network Details

> **Overleaf TODO (line 640):** *"Describe the extension we did compared to Gornet et al. We allowed the rotation of the head."*

**Source:** `src/ann/predictive_coder.py`, `src/ann/get_activations.py`

### LaTeX snippet

```latex
We re-implemented the predictive coding framework of Gornet and Thomson~\cite{Gornet2024automated}
using a UNet-style encoder--decoder architecture.
The encoder consists of an initial $7\times7$ convolution (64 channels) followed by
batch normalization and max pooling, then two downsampling residual blocks
(64 and 128 channels, with 2 residual blocks per stage).
The decoder mirrors the encoder with upsampling blocks and skip connections.
Between encoder and decoder, two self-attention layers (128 channels, 8 heads each)
with causal masking operate on the latent representation, interleaved with
feed-forward networks and layer normalization.
The model receives sequences of 10 consecutive first-person RGB images
(preprocessed to $224 \times 224$ pixels) and is trained to predict the next frame.

The model was trained for 200 epochs using SGD with Nesterov momentum,
a weight decay of $5 \times 10^{-6}$, a learning rate of $10^{-1}$,
and OneCycle learning rate scheduling~\cite{SmithTopin2019}.

A key difference from the original implementation in~\cite{Gornet2024automated}
is that we decoupled the agent's head direction from its movement direction,
allowing the head to rotate independently.
This enabled us to collect model responses across all combinations of
2-D location and head direction.
In the original model, head direction was fixed along the movement direction,
precluding analysis of invariance to head direction.

For activation extraction, we collected the output of the last attention layer
for each location--head direction pair.
When a layer produced more than 8{,}000 units, a fixed random subset of 8{,}000 units
was selected (seed = 413) and used consistently across all stimulus conditions.
```

---

## 5. CNN Activation Collection Details

**Source:** `src/ann/get_activations.py`

### LaTeX snippet

```latex
For the CNN analysis, we evaluated VGG-16 (with ImageNet-pretrained and
randomly initialized weights) using the same Minecraft environment.
First-person perspective images ($224 \times 224$ pixels) were collected for
each combination of 2-D location and head direction.
Images were preprocessed with ImageNet normalization
(mean $= [0.485, 0.456, 0.406]$, std $= [0.229, 0.224, 0.225]$).
Activations were extracted in batches of 64 images.
When the number of units in a given layer exceeded 8{,}000,
a fixed random subset was selected for analysis.
```

---

## 6. Place Cell Classification Criteria (Tanni et al.)

> **Overleaf line 632 area:** partially written but could use code-level detail

**Source:** `src/ann/score_utils.py`, `src/ann/classify_neurons.py`

### LaTeX snippet

```latex
Place cell classification followed the criteria of Tanni et al.~\cite{Tanni2022}
with one omission: the pyramidal-waveform requirement was excluded as it has
no analogue for model units.

\textit{Spatial stability.}
For each unit, rate maps were computed separately from odd and even
head-direction samples. A unit was required to have a Pearson correlation $r > 0.3$
between the two half-maps to be considered spatially stable.

\textit{Place field detection.}
We implemented the iterative thresholding procedure of~\cite{Tanni2022}.
An initial binary mask was created by thresholding the rate map at
the map mean. Connected components (4-neighbor connectivity) were identified.
The threshold was then raised in steps of $0.1 \times \mathrm{std}(\text{rate map})$,
and candidate fields were re-identified at each step.
Candidates were required to have a minimum area of 15 spatial bins,
a peak activation $\geq 50\%$ of the map maximum,
and to occupy no more than 50\% of the total map area.

\textit{Additional cell type classification.}
Border cells were identified using the Banino et al.~\cite{Banino2018} criterion:
for each wall, the mean activation in a 3-bin border strip was compared to
the center activation, yielding a border score
$= \max_{\text{walls}} (\bar{r}_{\text{border}} - \bar{r}_{\text{center}}) / (\bar{r}_{\text{border}} + \bar{r}_{\text{center}})$.
Units with a border score $> 0.5$ were classified as border cells.
Head direction cells were identified by computing the mean resultant vector length
of the circular activation profile across head direction bins;
units with a mean vector length $> 0.47$ were classified as direction cells.
```

---

## 7. Pfeiffer Place Cell Data Analysis Parameters

**Source:** `src/pfeiffer_ratemaps.py`, `src/pfeiffer_decode_solo.py`

### LaTeX snippet (to supplement existing Methods section, ~line 657)

```latex
Position was discretized into a $100 \times 100$ grid over the $201 \times 201$~cm arena
(~2~cm bins).
Head direction was binned into 90 bins ($4\degree$ each).
Neural activity was binned into 250~ms time windows.
Time bins in which the animal's speed was below 5~cm/s were excluded.
Spatial bins with fewer than 5 visits were discarded.

For each neuron, we constructed 3-D rate maps (X $\times$ Y $\times$ HD)
by dividing the spike count in each bin by the occupancy.
For the split-half variance analysis, visits to each bin were randomly shuffled
and divided into two equal folds, from which separate rate maps were computed.

For decoding, we used a memoryless Bayesian decoder with independent Poisson neurons
and a uniform prior.
Rate maps were computed on a finer grid:
$100 \times 100$ position bins and $360$ HD bins ($1\degree$ each),
smoothed with a Gaussian kernel ($\sigma = 2$~bins for both XY and HD).
Four-fold cross-validation was used: rate maps were estimated from three folds
and decoding was performed on the held-out fold.
The decoding window was $\tau = 250$~ms.
```

---

## 8. Duszkiewicz Head Direction Cell Data Analysis Parameters

**Source:** `src/dusz_ratemaps.py`, `src/dusz_decode_solo.py`

### LaTeX snippet (to supplement existing Methods section, ~line 684)

```latex
For the head direction cell analysis, position was discretized into
a $20 \times 20$ grid, and head direction into 90 bins ($4\degree$ each).
Neural activity was binned into 50~ms windows.
Time bins with speed below 2~cm/s were excluded.
Spatial bins with fewer than 5 visits were discarded.

Movement direction was estimated from the 2-D positional trace by computing
frame-to-frame displacements and converting to angles via $\operatorname{atan2}$.
The resulting angles were wrapped to $[0\degree, 360\degree)$ and smoothed with a
circular boxcar filter (window = 20 samples).
A temporal shift of 3 samples was applied to synchronize the movement direction
estimate with neural activity.
For the HD $\times$ MD analysis, both head direction and movement direction were
binned into 90 bins ($4\degree$ each).

For decoding, rate maps were computed on finer grids:
$360$ bins for HD and MD ($1\degree$ each), and $32 \times 32$ bins for XY position.
Rate maps were smoothed with a Gaussian kernel ($\sigma = 3\degree$, with circular
wrapping for angular variables).
Spike trains were temporally smoothed with a Gaussian kernel
($\sigma = 100$~ms) prior to decoding, matching the analysis in~\cite{Duszkiewicz2024}.
Four-fold cross-validation was used with a decoding window of $\tau = 50$~ms.
Bins with fewer than 10 occupancy samples were excluded from rate map estimation.
```

---

## Quick Reference: Parameters Summary

| Parameter | Pfeiffer (Place Cells) | Duszkiewicz (HD Cells) |
|-----------|----------------------|----------------------|
| XY bins (ratemap) | 100 × 100 | 20 × 20 |
| HD bins (ratemap) | 90 (4°) | 90 (4°) |
| MD bins (ratemap) | — | 90 (4°) |
| Time bin (τ) | 250 ms | 50 ms |
| Speed threshold | 5.0 cm/s | 2.0 cm/s |
| Min occupancy | 5 | 5 |
| XY bins (decode) | 100 × 100 | 32 × 32 |
| HD bins (decode) | 360 (1°) | 360 (1°) |
| MD bins (decode) | — | 360 (1°) |
| Spike smoothing σ | 0 s | 100 ms |
| Rate map smoothing σ | 2 bins | 3° |
| CV folds | 4 | 4 |
| Decode window | 250 ms | 50 ms |
