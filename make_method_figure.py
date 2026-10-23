#!/usr/bin/env python3
"""
Generate a schematic figure explaining the cross-fold correlation
explained variance method, with equations.
"""

import os
import numpy as np

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

plt.rcParams['mathtext.fontset'] = 'cm'

np.random.seed(42)

# --- Synthetic data ---
n_s1, n_s2 = 20, 20
s1 = np.linspace(0, 1, n_s1)
s2 = np.linspace(0, 1, n_s2)
S1, S2 = np.meshgrid(s1, s2, indexing='ij')

def make_neuron(cx, cy, sx, sy):
    return np.exp(-((S1 - cx)**2 / (2*sx**2) + (S2 - cy)**2 / (2*sy**2)))

true_map = make_neuron(0.4, 0.6, 0.15, 0.12)
true_map += 0.3 * make_neuron(0.75, 0.3, 0.1, 0.1)

n_visits = 800
rng = np.random.default_rng(42)
pos_s1 = rng.beta(2, 2, n_visits)
pos_s2 = rng.beta(2, 2, n_visits)
bin_s1 = np.clip((pos_s1 * n_s1).astype(int), 0, n_s1-1)
bin_s2 = np.clip((pos_s2 * n_s2).astype(int), 0, n_s2-1)

rates = true_map[bin_s1, bin_s2] + rng.normal(0, 0.15, n_visits)
rates = np.clip(rates, 0, None)

idx = np.arange(n_visits)
rng.shuffle(idx)
half = n_visits // 2
fold0_idx, fold1_idx = idx[:half], idx[half:]

def build_rate_map(visit_indices):
    rate_map = np.full((n_s1, n_s2), np.nan)
    occ = np.zeros((n_s1, n_s2))
    rate_sum = np.zeros((n_s1, n_s2))
    for i in visit_indices:
        occ[bin_s1[i], bin_s2[i]] += 1
        rate_sum[bin_s1[i], bin_s2[i]] += rates[i]
    valid = occ >= 2
    rate_map[valid] = rate_sum[valid] / occ[valid]
    return rate_map, occ

fold0_map, occ0 = build_rate_map(fold0_idx)
fold1_map, occ1 = build_rate_map(fold1_idx)

def nan_smooth(rm, sigma=1.0):
    V = rm.copy()
    mask = np.isnan(V)
    V[mask] = 0.0
    U = np.ones_like(V)
    U[mask] = 0.0
    VV = gaussian_filter(V, sigma=sigma)
    UU = gaussian_filter(U, sigma=sigma)
    out = np.where(UU > 0, VV/UU, np.nan)
    return out

fold0_smooth = nan_smooth(fold0_map, sigma=1.2)
fold1_smooth = nan_smooth(fold1_map, sigma=1.2)

def cross_corr_matrix(f0, f1, axis=0):
    if axis == 1:
        f0 = f0.T
        f1 = f1.T
    n_slices = f0.shape[0]
    C = np.full((n_slices, n_slices), np.nan)
    for i in range(n_slices):
        for j in range(n_slices):
            a = f0[i]
            b = f1[j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() < 3:
                continue
            a_v = a[valid]
            b_v = b[valid]
            a_v = a_v - a_v.mean()
            b_v = b_v - b_v.mean()
            denom = np.sqrt((a_v**2).sum() * (b_v**2).sum())
            if denom > 0:
                C[i, j] = (a_v * b_v).sum() / denom
    return C

corr_A = cross_corr_matrix(fold0_smooth, fold1_smooth, axis=0)

# --- Figure ---
fig = plt.figure(figsize=(16, 11), facecolor='white')

# Two rows: top row = schematic panels, bottom row = equations
gs_outer = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1.0],
                             left=0.04, right=0.96, top=0.88, bottom=0.02,
                             hspace=0.18)

gs_main = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_outer[0],
                                           width_ratios=[1.1, 0.15, 1.1, 1.1, 1.2, 1.0],
                                           wspace=0.15)

cmap = 'viridis'
vmin, vmax = 0, np.nanmax(fold0_smooth) * 1.05

# --- Step 1: Joint tuning map ---
ax1 = fig.add_subplot(gs_main[0, 0])
im1 = ax1.imshow(true_map.T, origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
ax1.set_xlabel('Stimulus A (e.g. position)', fontsize=9)
ax1.set_ylabel('Stimulus B (e.g. head direction)', fontsize=9)
ax1.set_title('1. Joint tuning map\n$f(A, B)$', fontsize=11, fontweight='bold', pad=10)
ax1.set_xticks([])
ax1.set_yticks([])

# --- Arrow 1->2 ---
ax_arrow1 = fig.add_subplot(gs_main[0, 1])
ax_arrow1.axis('off')
ax_arrow1.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#333333'),
                   xycoords='axes fraction')
ax_arrow1.text(0.5, 0.62, 'split\nhalf', ha='center', va='bottom', fontsize=8,
               color='#333333', fontstyle='italic')

# --- Step 2: Two folds (stacked) ---
gs_folds = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 2],
                                            hspace=0.35)
ax2a = fig.add_subplot(gs_folds[0])
ax2a.imshow(fold0_smooth.T, origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
ax2a.set_title('2. Split & smooth', fontsize=11, fontweight='bold', pad=8)
ax2a.set_ylabel('Fold 0', fontsize=9, color='#d62728')
ax2a.set_xticks([])
ax2a.set_yticks([])
for spine in ax2a.spines.values():
    spine.set_color('#d62728')
    spine.set_linewidth(2)

ax2b = fig.add_subplot(gs_folds[1])
ax2b.imshow(fold1_smooth.T, origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
ax2b.set_ylabel('Fold 1', fontsize=9, color='#1f77b4')
ax2b.set_xlabel('Stimulus A', fontsize=8)
ax2b.set_xticks([])
ax2b.set_yticks([])
for spine in ax2b.spines.values():
    spine.set_color('#1f77b4')
    spine.set_linewidth(2)

# --- Step 3: Slice along axis -> tuning curves ---
gs_curves = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 3],
                                             hspace=0.35)
ax3a = fig.add_subplot(gs_curves[0])
ax3b = fig.add_subplot(gs_curves[1])

colors_slices = plt.cm.plasma(np.linspace(0.15, 0.85, 5))
slice_indices = [3, 7, 10, 14, 17]
x_axis = np.arange(n_s2)

for k, si in enumerate(slice_indices):
    label = f'$a_{{{si}}}$'
    ax3a.plot(x_axis, fold0_smooth[si, :], color=colors_slices[k], lw=1.5, alpha=0.85, label=label)
    ax3b.plot(x_axis, fold1_smooth[si, :], color=colors_slices[k], lw=1.5, alpha=0.85, label=label)

ax3a.set_title('3. Slice along A\n(tuning curves over B)', fontsize=11, fontweight='bold', pad=8)
ax3a.set_ylabel('Rate (fold 0)', fontsize=8, color='#d62728')
ax3a.set_xticks([])
ax3a.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.7)

ax3b.set_ylabel('Rate (fold 1)', fontsize=8, color='#1f77b4')
ax3b.set_xlabel('Stimulus B bins', fontsize=8)

for ax in [ax3a, ax3b]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)

# --- Step 4: Cross-fold correlation matrix ---
ax4 = fig.add_subplot(gs_main[0, 4])
im4 = ax4.imshow(corr_A, cmap='RdBu_r', vmin=-0.3, vmax=1.0, aspect='equal', origin='lower')
ax4.set_xlabel('$j$ (A bin, fold 1)', fontsize=9)
ax4.set_ylabel('$i$ (A bin, fold 0)', fontsize=9)
ax4.set_title('4. Cross-fold\ncorrelation matrix $C$', fontsize=11, fontweight='bold', pad=8)
cb = fig.colorbar(im4, ax=ax4, shrink=0.7, pad=0.03)
cb.set_label('Pearson $r$', fontsize=8)

n = corr_A.shape[0]
for i in range(n):
    ax4.plot(i, i, 's', ms=2.5, color='#2ca02c', markeredgewidth=0)

# --- Step 5: Result explanation ---
ax5 = fig.add_subplot(gs_main[0, 5])
ax5.axis('off')

diag_vals = np.diag(corr_A)
off_mask = ~np.eye(n, dtype=bool)
off_vals = corr_A[off_mask]

reliability = np.nanmean(diag_vals)
variation = np.nanmean(off_vals)
ev_A = 1 - variation

result_text = (
    "5. Read off the matrix\n"
    "------------------------\n\n"
    f"Diagonal (green dots):\n"
    f"  mean r = {reliability:.3f}\n"
    f"  = Reliability\n"
    f"  (same-bin cross-fold r)\n\n"
    f"Off-diagonal mean:\n"
    f"  mean r = {variation:.3f}\n"
    f"  = Similarity of tuning\n"
    f"    across A positions\n\n"
    f"Reported EV(A):\n"
    f"  1 - off-diag = {ev_A:.3f}\n\n"
    "High EV(A):\n"
    "  off-diag << diag\n"
    "  -> A modulates firing\n\n"
    "Low EV(A):\n"
    "  off-diag ~ diag\n"
    "  -> A has little effect"
)

ax5.text(0.05, 0.95, result_text, transform=ax5.transAxes,
         fontsize=9.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                   edgecolor='#cccccc', alpha=0.9))

# --- Equation strip across the bottom ---
ax_eq = fig.add_subplot(gs_outer[1])
ax_eq.axis('off')

eq_text = (
    r"$\mathbf{Step\ 3{:}}$  For each bin $a_i$ of stimulus A, extract the tuning curve over B:  "
    r"$\mathbf{f}^{(0)}_i = f_{\mathrm{fold\,0}}(a_i,\, \cdot\,)$"
    r"$\quad$and$\quad$"
    r"$\mathbf{f}^{(1)}_j = f_{\mathrm{fold\,1}}(a_j,\, \cdot\,)$"
    "\n\n"
    r"$\mathbf{Step\ 4{:}}$  Cross-fold correlation matrix:  "
    r"$C_{ij} \;=\; \mathrm{corr}\!\left(\,\mathbf{f}^{(0)}_i,\;\mathbf{f}^{(1)}_j\,\right)$"
    "\n\n"
    r"$\mathbf{Step\ 5{:}}$  "
    r"$\mathrm{Reliability} = \langle C_{ii} \rangle_i$"
    r"$\qquad\qquad$"
    r"$\mathrm{EV}(A) \;=\; 1 \;-\; \langle C_{ij} \rangle_{i \neq j}$"
)

ax_eq.text(0.5, 0.65, eq_text, transform=ax_eq.transAxes,
           fontsize=12, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#fafafa',
                     edgecolor='#999999', alpha=0.95),
           linespacing=1.8)

# --- Main title ---
fig.suptitle('Cross-Fold Correlation Method for Decomposing Stimulus Contributions',
             fontsize=14, fontweight='bold', y=0.95)

fig.savefig(os.path.join(FIGS, 'method_schematic.pdf'),
            dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'method_schematic.png'),
            dpi=200, bbox_inches='tight', facecolor='white')

print("Saved to figs/method_schematic.pdf and .png")
plt.close()
