#!/usr/bin/env python3
"""
Generate a schematic figure explaining the 1D Bayesian decoding analysis.
"""

import os
import numpy as np

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

plt.rcParams['mathtext.fontset'] = 'cm'

np.random.seed(42)
rng = np.random.default_rng(42)

# --- Synthetic data: HD decoding from place cells ---
n_hd_bins = 60
n_neurons = 8
tau = 0.25  # time bin in seconds

hd_centers = np.linspace(0, 360, n_hd_bins, endpoint=False)

# Create diverse HD tuning curves for a population
def von_mises_rate(theta, mu, kappa, peak):
    rad = np.deg2rad(theta - mu)
    return peak * np.exp(kappa * (np.cos(rad) - 1))

neuron_params = [
    (45,  1.5, 8),
    (120, 2.0, 12),
    (200, 1.0, 6),
    (280, 1.8, 10),
    (350, 0.8, 5),
    (160, 2.5, 15),
    (90,  1.2, 7),
    (240, 1.5, 9),
]

# Build rate maps (tuning curves)
rate_maps = np.zeros((n_hd_bins, n_neurons))
for n, (mu, kappa, peak) in enumerate(neuron_params):
    rate_maps[:, n] = von_mises_rate(hd_centers, mu, kappa, peak)
    rate_maps[:, n] += 0.5  # baseline firing

# Smooth
for n in range(n_neurons):
    rate_maps[:, n] = gaussian_filter1d(rate_maps[:, n], sigma=1.5, mode='wrap')

# Simulate a true HD trajectory
n_time = 200
time_axis = np.arange(n_time) * tau
true_hd = np.cumsum(rng.normal(0, 8, n_time)) % 360
true_hd = gaussian_filter1d(true_hd, sigma=2)
true_hd = true_hd % 360

# Generate spike counts from the true HD
true_bin_idx = np.clip((true_hd / 360 * n_hd_bins).astype(int), 0, n_hd_bins - 1)
spike_counts = np.zeros((n_time, n_neurons))
for t in range(n_time):
    for n in range(n_neurons):
        lam = rate_maps[true_bin_idx[t], n] * tau
        spike_counts[t, n] = rng.poisson(max(lam, 0.01))

# Decode
decoded_hd = np.full(n_time, np.nan)
posteriors = []
eps = 1e-10
rate_tau = rate_maps * tau

for t in range(n_time):
    k = spike_counts[t]
    log_like = np.nansum(k[None, :] * np.log(rate_tau + eps) - rate_tau, axis=1)
    m = np.nanmax(log_like)
    post = np.exp(log_like - m)
    s = post.sum()
    if s > 0:
        post /= s
    posteriors.append(post)
    decoded_hd[t] = hd_centers[np.argmax(post)]

posteriors = np.array(posteriors)

# Pick an example time bin for the detailed view
t_example = 80
example_post = posteriors[t_example]
example_spikes = spike_counts[t_example]
example_true_hd = true_hd[t_example]

# --- Figure ---
fig = plt.figure(figsize=(16, 12), facecolor='white')

gs_outer = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.5, 1.0],
                             left=0.06, right=0.94, top=0.90, bottom=0.04,
                             hspace=0.35)

# ===== ROW 1: Training — rate maps =====
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0],
                                          width_ratios=[1.5, 0.1, 1.5], wspace=0.05)

# Panel A: Tuning curves for all neurons
ax_tc = fig.add_subplot(gs_top[0, 0])
colors_n = plt.cm.tab10(np.arange(n_neurons))
for n in range(n_neurons):
    ax_tc.plot(hd_centers, rate_maps[:, n], color=colors_n[n], lw=1.8,
               label=f'Neuron {n+1}')
ax_tc.set_xlabel('Head direction (deg)', fontsize=10)
ax_tc.set_ylabel('Firing rate (Hz)', fontsize=10)
ax_tc.set_title('A. Training: 1D tuning curves from training fold',
                fontsize=12, fontweight='bold', loc='left')
ax_tc.set_xlim(0, 360)
ax_tc.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.8)
ax_tc.spines['top'].set_visible(False)
ax_tc.spines['right'].set_visible(False)
ax_tc.text(-0.02, 1.08, r'$\lambda_n(\theta) = $ mean rate of neuron $n$ at bin $\theta$',
           transform=ax_tc.transAxes, fontsize=10, fontstyle='italic', color='#555555')

# Panel B: Heatmap view of rate maps
ax_hm = fig.add_subplot(gs_top[0, 2])
im = ax_hm.imshow(rate_maps.T, aspect='auto', cmap='viridis', origin='lower',
                   extent=[0, 360, -0.5, n_neurons - 0.5])
ax_hm.set_xlabel('Head direction (deg)', fontsize=10)
ax_hm.set_ylabel('Neuron', fontsize=10)
ax_hm.set_yticks(range(n_neurons))
ax_hm.set_yticklabels([f'{n+1}' for n in range(n_neurons)])
ax_hm.set_title('B. Population rate map',
                fontsize=12, fontweight='bold', loc='left')
cb = fig.colorbar(im, ax=ax_hm, shrink=0.8, pad=0.02)
cb.set_label('Rate (Hz)', fontsize=8)

# ===== ROW 2: Decoding at one time bin =====
gs_mid = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1],
                                          width_ratios=[1.0, 1.2, 1.2, 1.2],
                                          wspace=0.3)

# Panel C: Observed spikes
ax_spk = fig.add_subplot(gs_mid[0, 0])
bars = ax_spk.barh(range(n_neurons), example_spikes, color=colors_n, height=0.7)
ax_spk.set_xlabel('Spike count', fontsize=10)
ax_spk.set_ylabel('Neuron', fontsize=10)
ax_spk.set_yticks(range(n_neurons))
ax_spk.set_yticklabels([f'{n+1}' for n in range(n_neurons)])
ax_spk.set_title(f'C. Observed spikes\n(time bin $t={t_example}$, '
                 f'$\\tau$={tau}s)',
                 fontsize=11, fontweight='bold', loc='left')
ax_spk.spines['top'].set_visible(False)
ax_spk.spines['right'].set_visible(False)
ax_spk.invert_yaxis()

# Panel D: Log-likelihood
ax_ll = fig.add_subplot(gs_mid[0, 1])
log_like_ex = np.nansum(example_spikes[None, :] * np.log(rate_tau + eps) - rate_tau, axis=1)
ax_ll.plot(hd_centers, log_like_ex, 'k-', lw=1.5)
ax_ll.axvline(example_true_hd, color='#2ca02c', ls='--', lw=1.5, label='True HD')
ax_ll.set_xlabel('Head direction (deg)', fontsize=10)
ax_ll.set_ylabel('Log-likelihood', fontsize=10)
ax_ll.set_title('D. Log-likelihood over bins',
                fontsize=11, fontweight='bold', loc='left')
ax_ll.set_xlim(0, 360)
ax_ll.legend(fontsize=8)
ax_ll.spines['top'].set_visible(False)
ax_ll.spines['right'].set_visible(False)

# Panel E: Posterior
ax_post = fig.add_subplot(gs_mid[0, 2])
ax_post.fill_between(hd_centers, example_post, alpha=0.4, color='#1f77b4')
ax_post.plot(hd_centers, example_post, color='#1f77b4', lw=2)
ax_post.axvline(example_true_hd, color='#2ca02c', ls='--', lw=1.5, label='True HD')
decoded_ex = hd_centers[np.argmax(example_post)]
ax_post.axvline(decoded_ex, color='#d62728', ls='-', lw=2, label='Decoded (MAP)')
ax_post.set_xlabel('Head direction (deg)', fontsize=10)
ax_post.set_ylabel(r'$P(\theta \mid \mathbf{k})$', fontsize=11)
ax_post.set_title('E. Posterior (normalized)',
                  fontsize=11, fontweight='bold', loc='left')
ax_post.set_xlim(0, 360)
ax_post.legend(fontsize=8)
ax_post.spines['top'].set_visible(False)
ax_post.spines['right'].set_visible(False)

# Panel F: Error histogram (from full decoding)
ax_err = fig.add_subplot(gs_mid[0, 3])
errors = (decoded_hd - true_hd + 180) % 360 - 180
ax_err.hist(errors, bins=40, color='#8c564b', alpha=0.7, edgecolor='white', density=True)
mae = np.mean(np.abs(errors))
median_ae = np.median(np.abs(errors))
ax_err.axvline(0, color='k', ls='--', lw=1, alpha=0.5)
ax_err.set_xlabel('Decoding error (deg)', fontsize=10)
ax_err.set_ylabel('Density', fontsize=10)
ax_err.set_title(f'F. Error distribution\nMAE={mae:.1f}deg, '
                 f'median={median_ae:.1f}deg',
                 fontsize=11, fontweight='bold', loc='left')
ax_err.spines['top'].set_visible(False)
ax_err.spines['right'].set_visible(False)

# ===== ROW 3: Decoded trajectory + equations =====
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[2],
                                          width_ratios=[2.0, 1.3], wspace=0.15)

# Panel G: Decoded vs true trajectory
ax_traj = fig.add_subplot(gs_bot[0, 0])
ax_traj.plot(time_axis, true_hd, 'k-', lw=1.5, label='True HD', alpha=0.7)
ax_traj.scatter(time_axis, decoded_hd, c='#d62728', s=8, alpha=0.6, label='Decoded HD', zorder=3)
ax_traj.axvspan(time_axis[t_example] - tau/2, time_axis[t_example] + tau/2,
                color='#ffdd57', alpha=0.5, label=f'Example bin (t={t_example})')
ax_traj.set_xlabel('Time (s)', fontsize=10)
ax_traj.set_ylabel('Head direction (deg)', fontsize=10)
ax_traj.set_title('G. Decoded trajectory vs ground truth',
                  fontsize=12, fontweight='bold', loc='left')
ax_traj.set_ylim(0, 360)
ax_traj.legend(fontsize=8, loc='upper right')
ax_traj.spines['top'].set_visible(False)
ax_traj.spines['right'].set_visible(False)

# Panel H: Equations
ax_eq = fig.add_subplot(gs_bot[0, 1])
ax_eq.axis('off')

eq_text = (
    r"$\mathbf{Poisson\ likelihood{:}}$" "\n"
    r"$\log P(\mathbf{k} \mid \theta) = \sum_{n=1}^{N} \left[ k_n \log(\lambda_n(\theta)\,\tau) - \lambda_n(\theta)\,\tau \right]$"
    "\n\n"
    r"$\mathbf{Posterior{:}}$" "\n"
    r"$P(\theta \mid \mathbf{k}) = \frac{P(\mathbf{k} \mid \theta)\, P(\theta)}{\sum_{\theta'} P(\mathbf{k} \mid \theta')\, P(\theta')}$"
    "\n\n"
    r"$\mathbf{Decoded\ stimulus{:}}$" "\n"
    r"$\hat{\theta} = \arg\max_\theta \; P(\theta \mid \mathbf{k})$"
    "\n\n"
    r"where $k_n$ = spike count of neuron $n$,"
    "\n"
    r"$\lambda_n(\theta)$ = tuning curve rate,"
    "\n"
    r"$\tau$ = time bin width, $P(\theta)$ = flat prior"
)

ax_eq.text(0.05, 0.95, eq_text, transform=ax_eq.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5ff',
                     edgecolor='#8888cc', alpha=0.95),
           linespacing=1.6)

fig.suptitle('1D Bayesian Population Decoding',
             fontsize=15, fontweight='bold', y=0.96)

fig.savefig(os.path.join(FIGS, 'decoding_schematic.pdf'),
            dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'decoding_schematic.png'),
            dpi=200, bbox_inches='tight', facecolor='white')

print("Saved to figs/decoding_schematic.pdf and .png")
plt.close()
