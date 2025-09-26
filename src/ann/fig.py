import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List

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
    ax.set_title("Neuron‑type composition", fontsize=16)
    ax.set_aspect("equal")  # keep it circular

    return fig


def plot_neuron_2d_3d(
    activations: np.ndarray,
    neuron_idx: int,
    axes_names: Tuple[str, str, str] = ("X", "Y", "Z"),
    *,
    slice_indices: List[int] | None = None,
    n_slices: int = 4,
    z_spacing: float = 1.5,
    cmap: str = "Hot",
    surf_opacity: float = 0.9,
    grid_color: str = "black",
    grid_width: float = 1,
    fig_size: Tuple[int, int] = (1800, 600),
    overall_title: str | None = None,
    z_label_degrees: bool = False,
):
    if activations.ndim != 4:
        raise ValueError("`activations` must be 4-D: (N_neurons, Nx, Ny, Nz)")

    data = activations[neuron_idx]  # (Nx, Ny, Nz)
    nx, ny, nz = data.shape

    if slice_indices is None:
        slice_indices = np.linspace(0, nz - 1, n_slices, dtype=int).tolist()
    slice_indices = sorted(set(slice_indices))

    if z_label_degrees:
        deg_per_idx = 360.0 / nz
        tickvals = [idx * z_spacing for idx in slice_indices]
        ticktext = [f"{idx * deg_per_idx:.0f}°" for idx in slice_indices]

    cmin, cmax = np.nanmin(data), np.nanmax(data)

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.3, 0.35, 0.35],
        specs=[[{"type": "heatmap"}, {"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.07,
    )

    marginal_xy = data.mean(axis=2)
    fig.add_trace(
        go.Heatmap(
            z=marginal_xy.T,
            colorscale=cmap,
            zmin=cmin,
            zmax=cmax,
            showscale=False,
            hovertemplate=f"{axes_names[0]}=%{{x}}<br>{axes_names[1]}=%{{y}}"
                          "<br>Activation=%{z}<extra></extra>",
            coloraxis="coloraxis", 
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text=axes_names[0], row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text=axes_names[1], row=1, col=1, showticklabels=False)

    x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    for k, z_idx in enumerate(slice_indices):
        z_plane = z_idx * z_spacing
        z_label = f"{(360.0 / nz) * z_idx:.0f}°" if z_label_degrees else str(z_idx)

        # Surface
        fig.add_trace(
            go.Surface(
                x=x2d,
                y=y2d,
                z=np.full_like(x2d, z_plane),
                surfacecolor=data[:, :, z_idx],
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax,
                opacity=surf_opacity,
                showscale=False,
                hovertemplate=f"{axes_names[2]}={z_label}<extra></extra>",
                coloraxis="coloraxis",
            ),
            row=1, col=2
        )

        # Grid
        for xi in range(nx):
            fig.add_trace(
                go.Scatter3d(
                    x=[xi] * ny, y=np.arange(ny), z=[z_plane] * ny,
                    mode="lines",
                    line=dict(color=grid_color, width=grid_width),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=2
            )
        for yi in range(ny):
            fig.add_trace(
                go.Scatter3d(
                    x=np.arange(nx), y=[yi] * nx, z=[z_plane] * nx,
                    mode="lines",
                    line=dict(color=grid_color, width=grid_width),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=2
            )

    zaxis_kwargs_stack = dict(title=axes_names[2])
    if z_label_degrees:
        zaxis_kwargs_stack.update(tickvals=tickvals, ticktext=ticktext)

    fig.update_scenes(
        dict(
            xaxis=dict(title=axes_names[0], showticklabels=False),
            yaxis=dict(title=axes_names[1], showticklabels=False),
            zaxis=zaxis_kwargs_stack,
            aspectmode="cube",
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Volume(
            x=np.repeat(np.arange(nx), ny * nz),
            y=np.tile(np.repeat(np.arange(ny), nz), nx),
            z=np.tile(np.arange(nz), nx * ny),
            value=data.flatten(),
            opacity=0.1,
            surface_count=25,
            colorscale=cmap,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            colorbar=dict(title="Activation"),
            coloraxis="coloraxis",
        ),
        row=1, col=3
    )

    fig.update_scenes(
        dict(
            xaxis=dict(title=axes_names[0], showticklabels=False),
            yaxis=dict(title=axes_names[1], showticklabels=False),
            zaxis=dict(title=axes_names[2]),
            aspectmode="cube",
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        coloraxis=dict(
            cmin=cmin,
            cmax=cmax,
            colorscale=cmap,
            colorbar=dict(
                title="Activation",
                x=1.02,           # just outside the last subplot
                y=0.5,
                len=0.9,
                thickness=20
            )
        ),
        width=fig_size[0],
        height=fig_size[1],
        margin=dict(l=60, r=60, t=80, b=60),
        title=dict(text=overall_title, x=0.5) if overall_title else None,
    )

    return fig

