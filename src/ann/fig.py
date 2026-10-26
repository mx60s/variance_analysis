import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Mapping, Sequence
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple
import re

def plot_cell_type_pie(
    classifications: Mapping[str, Sequence[int]],
    *,
    threshold: float = 0.01,
    explode_offset: float = 0.1,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Draw a pie chart of neuron‑type counts.

    Parameters
    ----------
    classifications
        Dict whose keys are `<type>_indices` (exact keys used below)
        and whose values are index lists or arrays.
    threshold : float, default 0.01
        Minimum fractional prevalence (1 %=0.01) for a slice to be shown.
        Rarer types are omitted from the chart.
    explode_offset : float, default 0.1
        How far to 'explode' each slice.
    ax : matplotlib.axes.Axes or None
        If given, draw on this axes; otherwise create a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    
    keys = {
        "exclusive_place_cells_indices"            : "P",
        "exclusive_border_cells_indices"           : "B",
        "exclusive_direction_cells_indices"        : "D",
        "place_and_border_not_direction_cells_indices": "P+B",
        "place_and_direction_not_border_cells_indices": "P+D",
        "border_and_direction_not_place_cells_indices": "B+D",
        "place_border_direction_cells_indices"     : "P+B+D",
        "active_no_type_indices"                   : "Active (no type)",
    }

    counts = {
        label: len(classifications[k])
        for k, label in keys.items()
    }
    counts["Inactive"] = len(classifications["dead_neurons_indices"])

    total = sum(counts.values())
    if total == 0:
        raise ValueError("No cells found – nothing to plot.")

    labels, n_cells = zip(*[
        (lbl, n) for lbl, n in counts.items()
        if n / total >= threshold
    ])

    pastel = cm.Pastel1.colors
    colour_map = {
        "P"              : pastel[1],
        "B"              : pastel[0],
        "D"              : pastel[2],
        "P+B"            : pastel[3],
        "P+D"            : pastel[4],
        "B+D"            : pastel[5],
        "P+B+D"          : pastel[6],
        "Active (no type)": pastel[7],
        "Inactive"       : "grey",
    }
    colours = [colour_map[lbl] for lbl in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    explode = [explode_offset] * len(labels)
    ax.pie(
        n_cells,
        labels=labels,
        colors=colours,
        explode=explode,
        autopct=lambda p: f"{p:.0f}%" if p >= 1 else "",
        textprops={"fontsize": 14},
    )
    ax.set_aspect("equal")  # keep it circular

    return fig

def _max_cartesian_axis_index(layout):
    pat = re.compile(r"^xaxis(\d*)$")
    m = 1
    for k in layout:
        mm = pat.match(k)
        if mm:
            idx = int(mm.group(1)) if mm.group(1) else 1
            m = max(m, idx)
    return m

def _axis_keys(n):
    return (f"xaxis{n}" if n>1 else "xaxis", f"yaxis{n}" if n>1 else "yaxis")

def _axis_refs(n):
    return (f"x{n}" if n>1 else "x", f"y{n}" if n>1 else "y")

def _col_domains(ncols, column_widths, hspace):
    total_gap = (ncols - 1) * hspace
    avail = 1.0 - total_gap
    norm = sum(column_widths)
    widths = [avail * (w / norm) for w in column_widths]
    domains = []
    start = 0.0
    for i, w in enumerate(widths):
        end = start + w
        domains.append((start, end))
        if i < ncols - 1:
            start = end + hspace
    return domains

def plot_neuron_2d_3d(
    activations: np.ndarray,
    neuron_idx: int,
    axes_names: Tuple[str, str, str] = ("X", "Y", "HD"),
    *,
    n_slices: int = 4,
    z_spacing: float = 1.5,
    cmap: str = "Jet",
    surf_opacity: float = 0.9,
    grid_color: str = "black",
    grid_width: float = 1,
    fig_size: Tuple[int, int] = (1800, 600),
    overall_title: str | None = None,
    z_label_degrees: bool = True,
    firing_rate: bool = False,
    show_slice_grid: bool = False
):
    if activations.ndim != 4:
        raise ValueError("`activations` must be 4-D: (N_neurons, Nx, Ny, Nz)")

    data = activations[neuron_idx]
    nx, ny, nz = data.shape

    slice_indices = np.linspace(0, nz - 1, n_slices + 1, dtype=int).tolist()
    slice_indices = sorted(set(slice_indices))

    def idx_to_deg(z_idx: int) -> float:
        return (360.0 / nz) * (z_idx % nz)

    cmin, cmax = float(np.nanmin(data)), float(np.nanmax(data))

    column_widths = [0.35, 0.33, 0.32]
    hspace = 0.07
    x_domains = _col_domains(3, column_widths, hspace)
    xdom3 = x_domains[2]
    ydom3 = (0.0, 1.0)

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=column_widths,
        specs=[[{"type": "scene"}, {"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=hspace,
        subplot_titles=(
            f"3D heatmap ({axes_names[0]}, {axes_names[1]}, {axes_names[2]})",
            f"Marginal across {axes_names[2]}",
            f"Slices conditioned on {axes_names[2]}"
        ),
    )

    X = np.repeat(np.arange(nx), ny * nz)
    Y = np.tile(np.repeat(np.arange(ny), nz), nx)
    Z = np.tile(np.arange(nz), nx * ny)

    fig.add_trace(
        go.Volume(
            x=X, y=Y, z=Z,
            value=data.flatten(),
            opacity=0.10,
            surface_count=25,
            colorscale=cmap,
            cmin=cmin, cmax=cmax,
            showscale=False,
            coloraxis="coloraxis"
        ),
        row=1, col=1
    )
    fig.update_scenes(
        dict(
            xaxis=dict(title=axes_names[0], showticklabels=False),
            yaxis=dict(title=axes_names[1], showticklabels=False),
            zaxis=dict(title=axes_names[2], showticklabels=False),
            aspectmode="cube",
        ),
        row=1, col=1
    )

    marginal_xy = np.nanmean(data, axis=2)
    fig.add_trace(
        go.Heatmap(
            z=marginal_xy.T,
            colorscale=cmap,
            zmin=cmin, zmax=cmax,
            showscale=False,
            hovertemplate=(
                f"{axes_names[0]}=%{{x}}<br>"
                f"{axes_names[1]}=%{{y}}<br>"
                f"{'Firing rate' if firing_rate else 'Activation'}=%{{z}}"
                "<extra></extra>"
            ),
            coloraxis="coloraxis",
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text=axes_names[0], row=1, col=2, showticklabels=False)
    fig.update_yaxes(title_text=axes_names[1], row=1, col=2, showticklabels=False)

    start_idx = _max_cartesian_axis_index(fig.layout) + 1
    gap = 0.02
    total_height = ydom3[1] - ydom3[0]
    slot_height = (total_height - gap * (n_slices - 1)) / n_slices

    next_idx = start_idx

    for si in range(len(slice_indices) - 1):
        y0 = ydom3[0] + si * (slot_height + gap)
        y1 = y0 + slot_height
        x0, x1 = xdom3

        xk, yk = _axis_keys(next_idx)
        xr, yr = _axis_refs(next_idx)
        next_idx += 1

        fig.layout[xk] = dict(
            domain=[x0, x1],
            matches=None,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain="domain",
            scaleratio=1
        )
        fig.layout[yk] = dict(
            domain=[y0, y1],
            matches=None,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor=xr,
        )

        z_lo, z_hi = slice_indices[si], slice_indices[si+1]
        z2d = np.nanmean(data[:, :, z_lo:z_hi], axis=2).T
        #z2d = data[:, :, z_idx].T
        fig.add_trace(
            go.Heatmap(
                z=z2d,
                colorscale=cmap,
                zmin=cmin, zmax=cmax,
                showscale=False,
                hovertemplate=(
                    f"{'Firing rate' if firing_rate else 'Activation'}=%{{z}}"
                    "<extra></extra>"
                ),
                coloraxis="coloraxis",
                xaxis=xr,
                yaxis=yr,
            )
        )

        #slice_centers.append((0.5 * (y0 + y1), z_idx))
    '''
    label_x = xdom3[1] + 0.015
    for ymid, z_idx in slice_centers:
        label_txt = f"{idx_to_deg(z_idx):.0f}°" if z_label_degrees else str(z_idx)
        fig.add_annotation(
            x=label_x, y=ymid,
            xref="paper", yref="paper",
            text=label_txt,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    '''

    fig.update_xaxes(title_text=axes_names[0], row=1, col=2)
    fig.update_yaxes(title_text=axes_names[1], row=1, col=2)

    fig.update_layout(
        coloraxis=dict(
            cmin=cmin, cmax=cmax, colorscale=cmap,
            colorbar=dict(
                title=dict(
                    text="Firing rate<br>(spikes/s)" if firing_rate else "Activation"
                ),
                x=1.02, y=0.5, len=0.9, thickness=20
            )
        ),
        width=fig_size[0], height=fig_size[1],
        margin=dict(l=60, r=100, t=80, b=60),
        title=dict(text=overall_title, x=0.5) if overall_title else None,
    )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=3)

    return fig


def visualize_conv1_filters(
    model,
    *,
    figsize: Tuple[int, int] = (12, 8),
    ncols: int = 8,
    normalize: bool = True,
    title: str = "Conv1 Filters (7x7)",
) -> plt.Figure:
    """
    Visualize the learned conv1 filters from a PredictiveCoder encoder.

    These 7x7 filters typically learn Gabor-like oriented edge detectors,
    similar to V1 simple cells - demonstrating these are visual features.

    Parameters
    ----------
    model : torch.nn.Module
        PredictiveCoder or wrapper with encoder.conv1 weights
    figsize : tuple
        Figure size in inches
    ncols : int
        Number of columns in the filter grid
    normalize : bool
        Whether to normalize each filter to [0, 1] for display
    title : str
        Figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract conv1 weights: (out_channels, in_channels, H, W) = (64, 3, 7, 7)
    if hasattr(model, 'encoder'):
        weights = model.encoder.conv1.weight.detach().cpu().numpy()
    elif hasattr(model, 'core') and hasattr(model.core, 'encoder'):
        weights = model.core.encoder.conv1.weight.detach().cpu().numpy()
    else:
        raise ValueError("Cannot find encoder.conv1 in model")

    n_filters = weights.shape[0]  # 64
    nrows = int(np.ceil(n_filters / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_filters):
        ax = axes[i]

        # Get filter: (in_channels, H, W) e.g. (3, 7, 7)
        filt = weights[i]

        # Show as RGB if 3 input channels
        if filt.shape[0] == 3:
            # Transpose to (H, W, C) for RGB display
            filt_rgb = np.transpose(filt, (1, 2, 0))
            if normalize:
                filt_rgb = (filt_rgb - filt_rgb.min()) / (filt_rgb.max() - filt_rgb.min() + 1e-8)
            ax.imshow(filt_rgb)
        else:
            # Grayscale: average across input channels
            filt_gray = filt.mean(axis=0)
            if normalize:
                filt_gray = (filt_gray - filt_gray.min()) / (filt_gray.max() - filt_gray.min() + 1e-8)
            ax.imshow(filt_gray, cmap='gray')

        ax.axis('off')

    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    return fig
