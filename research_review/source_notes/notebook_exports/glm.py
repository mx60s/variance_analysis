# CELL 0
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

import os
os.chdir('/content/drive/MyDrive/ve/variance_analysis')

# CELL 1
!pip install -e /content/drive/MyDrive/ve/variance_analysis

# CELL 2

!pip -q install fonttools

!ttx -l fonts/Helvetica.ttc

!sudo mkdir -p /usr/share/fonts/truetype/helvetica
!ttx -y 0 -o /usr/share/fonts/truetype/helvetica/Helvetica-Regular.ttf fonts/Helvetica.ttc
!ttx -y 1 -o /usr/share/fonts/truetype/helvetica/Helvetica-Oblique.ttf fonts/Helvetica.ttc
!ttx -y 2 -o /usr/share/fonts/truetype/helvetica/Helvetica-Bold.ttf fonts/Helvetica.ttc
!ttx -y 3 -o /usr/share/fonts/truetype/helvetica/Helvetica-BoldOblique.ttf fonts/Helvetica.ttc

!fc-cache -fv


# CELL 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.font_manager as fm
import shutil

from pathlib import Path
current_file_path = Path().resolve()
current_file_path

!mkdir -p ~/.fonts
!cp fonts/Helvetica.ttc ~/.fonts/

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'Helvetica'

# CELL 5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

rng = np.random.default_rng(0)
n = 20000
x = rng.normal(0, 1.0, size=n)
y = rng.normal(0, 1.0, size=n)

def rotated_coord(x, y, c, theta, wrap=False):
    if wrap:
        d1 = ((x - c + 0.5) % 1.0) - 0.5
        d2 = ((y - c + 0.5) % 1.0) - 0.5
    else:
        d1 = x - c
        d2 = y - c
    ct, st = np.cos(theta), np.sin(theta)
    u = d1 * ct - d2 * st
    return u

mu = 0.0
theta = 60
theta_r = np.deg2rad(theta)
sigma_u = 1.2

u = rotated_coord(x, y, mu, theta_r, wrap=False)

alpha = np.log(5.0)
eta = alpha - ((u - mu) ** 2) / (2 * sigma_u ** 2)
lam = np.exp(eta)
events = rng.poisson(lam)
df = pd.DataFrame(dict(x=x, y=y, events=events))

# CELL 6
modelA = smf.glm("events ~ x + y + I(x**2) + I(y**2)", data=df, family=sm.families.Poisson()).fit()
print(modelA.summary())

def pseudo_r2(counts, mu):
    def ll(y, m):
        m = np.clip(m, 1e-12, None)
        return np.sum(y * np.log(m) - m)
    mu_null = np.full_like(counts, np.mean(counts))
    return 1.0 - ll(counts, mu) / ll(counts, mu_null)

muA = modelA.fittedvalues.values
print(f"\\nMcFadden pseudo-R^2 (Model A): {pseudo_r2(events, muA):.3f}")

# CELL 7
grid = np.linspace(-2.5, 2.5, 160)
Xg, Yg = np.meshgrid(grid, grid)
Ug = rotated_coord(Xg, Yg, mu, theta, wrap=False)
eta_true_grid = alpha - ((Ug - mu) ** 2) / (2 * sigma_u ** 2)
lam_true_grid = np.exp(eta_true_grid)

x_grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 300)
y_grid = np.linspace(np.percentile(y, 1), np.percentile(y, 99), 300)
x_mean = float(np.mean(x))
y_mean = float(np.mean(y))

df_x = pd.DataFrame({
    "x": x_grid,
    "y": y_mean,
    "x2": x_grid**2,
    "y2": np.full_like(x_grid, y_mean**2),
})
pred_x = modelA.predict(df_x[["x","y","x2","y2"]]).to_numpy()

df_y = pd.DataFrame({
    "x": x_mean,
    "y": y_grid,
    "x2": np.full_like(y_grid, x_mean**2),
    "y2": y_grid**2,
})
pred_y = modelA.predict(df_y[["x","y","x2","y2"]]).to_numpy()

fig = plt.figure(figsize=(12, 4))

ax1 = plt.subplot(1, 3, 1)
ax1.set_aspect('equal')
im = ax1.imshow(lam_true_grid, origin="lower",
                extent=[grid.min(), grid.max(), grid.min(), grid.max()])
ax1.set_xlabel("x"); ax1.set_ylabel("y")

ax2 = plt.subplot(1, 3, 2)
ax2.plot(x_grid, pred_x)
ax2.set_xlabel("x")
ax2.set_box_aspect(1)

ax3 = plt.subplot(1, 3, 3)
ax3.plot(y_grid, pred_y)
ax3.set_xlabel("y")
ax3.set_box_aspect(1)

plt.subplots_adjust(wspace=0.5)
cbar_ax = fig.add_axes([ax1.get_position().x1 + 0.01,
                        ax1.get_position().y0,
                        0.02,
                        ax1.get_position().height])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Rates")
plt.savefig(current_file_path / f'figs/glm_theta_{theta}.pdf')

plt.show()

# CELL 8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import Normalize

# -------------------------
# Helpers
# -------------------------
def rotated_coord(x, y, c, theta_rad, wrap=False):
    if wrap:
        d1 = ((x - c + 0.5) % 1.0) - 0.5
        d2 = ((y - c + 0.5) % 1.0) - 0.5
    else:
        d1 = x - c
        d2 = y - c
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    return d1 * ct - d2 * st

def true_rate_on_grid(X, Y, *, mu=0.0, theta_deg=60, sigma_u=1.2, alpha_log=np.log(5.0)):
    theta_r = np.deg2rad(theta_deg)
    U = rotated_coord(X, Y, mu, theta_r, wrap=False)
    eta = alpha_log - ((U - mu) ** 2) / (2 * sigma_u ** 2)
    return np.exp(eta)

def fit_glm_quadratic(df):
    return smf.glm(
        "events ~ x + y + I(x**2) + I(y**2)",
        data=df,
        family=sm.families.Poisson()
    ).fit()

def predict_on_grid(model, Xg, Yg):
    dfg = pd.DataFrame({"x": Xg.ravel(), "y": Yg.ravel()})
    return model.predict(dfg).to_numpy().reshape(Xg.shape)

# -------------------------
# Precompute row content
# -------------------------
def compute_row(
    *,
    seed,
    theta_deg,
    sigma_u=1.2,
    alpha_log=np.log(5.0),
    n=20000,
    grid_min=-2.5,
    grid_max=2.5,
    grid_n=160,
    slice_n=300,
    mu=0.0,
):
    rng = np.random.default_rng(seed)

    x = rng.normal(0, 1.0, size=n)
    y = rng.normal(0, 1.0, size=n)

    lam = true_rate_on_grid(x, y, mu=mu, theta_deg=theta_deg, sigma_u=sigma_u, alpha_log=alpha_log)
    events = rng.poisson(lam)
    df = pd.DataFrame(dict(x=x, y=y, events=events))

    model = fit_glm_quadratic(df)

    grid = np.linspace(grid_min, grid_max, grid_n)
    Xg, Yg = np.meshgrid(grid, grid)
    lam_true_grid = true_rate_on_grid(Xg, Yg, mu=mu, theta_deg=theta_deg, sigma_u=sigma_u, alpha_log=alpha_log)
    lam_glm_grid  = predict_on_grid(model, Xg, Yg)

    # slice domains (kept per-row; we’ll unify limits later)
    x_grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), slice_n)
    y_grid = np.linspace(np.percentile(y, 1), np.percentile(y, 99), slice_n)

    pred_fx = model.predict(pd.DataFrame({"x": x_grid, "y": np.zeros_like(x_grid)})).to_numpy()
    pred_gy = model.predict(pd.DataFrame({"x": np.zeros_like(y_grid), "y": y_grid})).to_numpy()

    # "True" displayed as flat lines (mean slice level)
    lam_true_fx = true_rate_on_grid(x_grid, np.zeros_like(x_grid), mu=mu, theta_deg=theta_deg, sigma_u=sigma_u, alpha_log=alpha_log)
    lam_true_gy = true_rate_on_grid(np.zeros_like(y_grid), y_grid, mu=mu, theta_deg=theta_deg, sigma_u=sigma_u, alpha_log=alpha_log)
    true_fx_flat = np.full_like(x_grid, lam_true_fx.mean())
    true_gy_flat = np.full_like(y_grid, lam_true_gy.mean())

    return dict(
        grid=grid,
        Xg=Xg, Yg=Yg,
        lam_true_grid=lam_true_grid,
        lam_glm_grid=lam_glm_grid,
        x_grid=x_grid, y_grid=y_grid,
        true_fx=true_fx_flat, true_gy=true_gy_flat,
        pred_fx=pred_fx, pred_gy=pred_gy
    )

# -------------------------
# Compute both rows first
# -------------------------
row_a = compute_row(seed=0, theta_deg=60, sigma_u=1.2, alpha_log=np.log(5.0))
row_b = compute_row(seed=1, theta_deg=45, sigma_u=1.2, alpha_log=np.log(5.0))

rows = [row_a, row_b]

# -------------------------
# Global bounds for comparison
# -------------------------
# Heatmaps: shared vmin/vmax across ALL 4 heatmaps
all_heatmaps = [
    row_a["lam_true_grid"], row_a["lam_glm_grid"],
    row_b["lam_true_grid"], row_b["lam_glm_grid"],
]
vmin = 0.0
vmax = float(np.max([np.nanmax(hm) for hm in all_heatmaps]))
norm = Normalize(vmin=vmin, vmax=vmax)

# 1D plots: shared ylim across ALL f(x)/g(y)
all_1d = [
    row_a["true_fx"], row_a["pred_fx"], row_a["true_gy"], row_a["pred_gy"],
    row_b["true_fx"], row_b["pred_fx"], row_b["true_gy"], row_b["pred_gy"],
]
ymin = float(np.min([np.nanmin(arr) for arr in all_1d]))
ymax = float(np.max([np.nanmax(arr) for arr in all_1d]))

# Optional: shared x-lims too (recommended for easy comparison)
xlim_fx = (
    float(min(row_a["x_grid"].min(), row_b["x_grid"].min())),
    float(max(row_a["x_grid"].max(), row_b["x_grid"].max())),
)
xlim_gy = (
    float(min(row_a["y_grid"].min(), row_b["y_grid"].min())),
    float(max(row_a["y_grid"].max(), row_b["y_grid"].max())),
)

# -------------------------
# Plot
# -------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)

cmap_heat = "viridis"

def plot_row(ax_row, row, row_tag):
    ax_h, ax_glm, ax_fx, ax_gy = ax_row
    grid = row["grid"]
    extent = [grid.min(), grid.max(), grid.min(), grid.max()]

    ax_h.set_aspect("equal")
    im1 = ax_h.imshow(row["lam_true_grid"], origin="lower", extent=extent,
                      cmap=cmap_heat, norm=norm)
    ax_h.set_title("h(x, y)")
    ax_h.set_xlabel("x"); ax_h.set_ylabel("y")
    ax_h.text(-0.18, 0.5, row_tag, transform=ax_h.transAxes,
              fontsize=18, fontweight="bold", va="center", ha="right")

    ax_glm.set_aspect("equal")
    im2 = ax_glm.imshow(row["lam_glm_grid"], origin="lower", extent=extent,
                        cmap=cmap_heat, norm=norm)
    ax_glm.set_title("GLM fit")
    ax_glm.set_xlabel("x"); ax_glm.set_ylabel("y")

    # per-axis colorbars (like your original layout)
    plt.colorbar(im1, ax=ax_h, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax_glm, fraction=0.046, pad=0.04)

    ax_fx.plot(row["x_grid"], row["true_fx"], linewidth=3, label="True")
    ax_fx.plot(row["x_grid"], row["pred_fx"], linestyle="--", linewidth=2.5, label="GLM")
    ax_fx.set_title("f(x)")
    ax_fx.set_xlabel("x"); ax_fx.set_ylabel("Rate")
    ax_fx.set_ylim(ymin, ymax)
    ax_fx.set_xlim(*xlim_fx)
    ax_fx.legend(frameon=False, loc="upper right")
    ax_fx.set_box_aspect(1)

    ax_gy.plot(row["y_grid"], row["true_gy"], linewidth=3, label="True")
    ax_gy.plot(row["y_grid"], row["pred_gy"], linestyle="--", linewidth=2.5, label="GLM")
    ax_gy.set_title("g(y)")
    ax_gy.set_xlabel("y"); ax_gy.set_ylabel("Rate")
    ax_gy.set_ylim(ymin, ymax)
    ax_gy.set_xlim(*xlim_gy)
    ax_gy.legend(frameon=False, loc="upper right")
    ax_gy.set_box_aspect(1)

plot_row(axes[0], row_a, "(a)")
plot_row(axes[1], row_b, "(b)")

plt.savefig(current_file_path / f'figs/glm.pdf')
plt.show()


# CELL 9
