#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np

from src.rodent_data import (
    load_dusz_data, build_spike_rates, compute_avg_velocity_per_bin, get_md_from_xy,
    select_time_indices_in_angle_bin, select_time_indices_in_xy_bin, split_half_rate_map,
)


def hdmd_fold_rate_map(spike_rates: np.ndarray, positions: np.ndarray, base_time: np.ndarray,
                       n_hd_bins: int, n_md_bins: int, md_series_deg: np.ndarray,
                       avg_vel_bins: np.ndarray, velocity_threshold: float,
                       floor: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    fast_mask = (avg_vel_bins >= velocity_threshold) if velocity_threshold > 0 else None
    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)
    md_edges = np.linspace(0.0, 360.0 + 1e-6, n_md_bins + 1)

    idx_hd = [select_time_indices_in_angle_bin(positions, base_time, positions[:, 3], (hd_edges[ih], hd_edges[ih + 1]), fast_mask)
              for ih in range(n_hd_bins)]
    idx_md = [select_time_indices_in_angle_bin(positions, base_time, md_series_deg, (md_edges[im], md_edges[im + 1]), fast_mask)
              for im in range(n_md_bins)]
    fold0_map, fold1_map, rate_map, occupancy = split_half_rate_map(spike_rates, idx_hd, idx_md, floor)

    return fold0_map, fold1_map, rate_map, occupancy, (hd_edges, md_edges)


def xyhd_fold_rate_map(spike_rates: np.ndarray, locations: np.ndarray, base_time: np.ndarray,
                       n_loc_bins: int, n_hd_bins: int,
                       avg_vel_bins: np.ndarray, velocity_threshold: float,
                       arena_pad: float = 0.0,
                       floor: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fast_mask = (avg_vel_bins >= velocity_threshold) if velocity_threshold > 0 else None

    xmin, xmax = float(np.nanmin(locations[:, 1])), float(np.nanmax(locations[:, 1]))
    ymin, ymax = float(np.nanmin(locations[:, 2])), float(np.nanmax(locations[:, 2]))
    if arena_pad and arena_pad != 0.0:
        xmin -= arena_pad; xmax += arena_pad
        ymin -= arena_pad; ymax += arena_pad
    x_edges = np.linspace(xmin, xmax + 1e-6, n_loc_bins + 1)
    y_edges = np.linspace(ymin, ymax + 1e-6, n_loc_bins + 1)
    hd_edges = np.linspace(0.0, 360.0 + 1e-6, n_hd_bins + 1)

    idx_xy = [select_time_indices_in_xy_bin(locations, base_time, (x_edges[ix], x_edges[ix + 1]), (y_edges[iy], y_edges[iy + 1]), fast_mask)
              for iy in range(n_loc_bins) for ix in range(n_loc_bins)]
    idx_hd = [select_time_indices_in_angle_bin(locations, base_time, locations[:, 3], (hd_edges[ih], hd_edges[ih + 1]), fast_mask)
              for ih in range(n_hd_bins)]
    maps = split_half_rate_map(spike_rates, idx_xy, idx_hd, floor)
    fold0_map, fold1_map, rate_map, occupancy = (m.reshape(n_loc_bins, n_loc_bins, *m.shape[1:]) for m in maps)

    return fold0_map, fold1_map, rate_map, occupancy, (x_edges, y_edges, hd_edges)


def get_ratemaps(args: Optional[argparse.Namespace] = None, /, **overrides: Any) -> Dict[str, Any]:
    if args is None:
        args = argparse.Namespace(**overrides)
    else:
        for k, v in overrides.items():
            setattr(args, k, v)

    d = args.__dict__
    d.setdefault("data_path", "./data/rodents/dusz")
    d.setdefault("save_path", "./results/rodents/dusz")
    d.setdefault("dataset", None)
    d.setdefault("rat", None)
    d.setdefault("mode", "xyhd")
    d.setdefault("n_hd_bins", 90)
    d.setdefault("n_md_bins", 90)
    d.setdefault("n_loc_bins", 20)
    d.setdefault("tau", 0.05)
    d.setdefault("vel_threshold", 2.0)
    d.setdefault("md_smooth_window", 20)
    d.setdefault("save_intermediates", False)
    d.setdefault("md_roll", 3)
    d.setdefault("floor", 5)
    d.setdefault("seed", 0)  # fixes the random fold split
    np.random.seed(d["seed"])

    if d["dataset"] is None or d["rat"] is None:
        raise ValueError("Both 'dataset' (int) and 'rat' (str) must be provided.")

    data_path = Path(d["data_path"])
    save_root = Path(d["save_path"]) / f"Dataset_{d['dataset']}" / d["rat"]
    os.makedirs(save_root, exist_ok=True)

    print(f"\n=== Rate Map Builder ===")
    print(f"Data path : {data_path}")
    print(f"Save path : {save_root}")
    print(f"tau: {d['tau']:.3f}s | vel_th: {d['vel_threshold']}")
    t0 = time.time()

    print("\nLoading data…")
    spike_times_raw, positions, cell_types, velocity_series = load_dusz_data(data_path, d["dataset"], d["rat"])

    hd = np.asarray(cell_types['hd']).astype(int).ravel()
    excit = np.asarray(cell_types["ex"], dtype=int).ravel()
    excit_ids = np.where((hd == 1) & (excit == 1))[0]

    print(f"Spikes: {len(spike_times_raw)} | Excitatory neurons: {len(excit_ids)} | Position samples: {positions.shape[0]}")

    if spike_times_raw.size == 0 or positions.size == 0 or excit_ids.size == 0:
        raise RuntimeError("Insufficient data: check dataset/rat paths and cell types.")

    base_time = np.arange(spike_times_raw[0, 0], spike_times_raw[-1, 0] + d["tau"], d["tau"])
    print(f"Time bins: {len(base_time)-1} bins covering {base_time[0]:.3f}–{base_time[-1]:.3f}s")

    print("Binning spikes into rates…")
    spike_rates = build_spike_rates(spike_times_raw, base_time, d["tau"], excit_ids)  # (T, N)
    avg_vel_bins = compute_avg_velocity_per_bin(base_time, velocity_series, positions)

    out: Dict[str, Any] = {}
    edges: Dict[str, Any] = {}

    if d["mode"] == "hdmd":
        md_series_deg = get_md_from_xy(positions, smoothing_window=d["md_smooth_window"])
        md_series_deg = np.roll(md_series_deg, -d["md_roll"])
        fold0, fold1, rmean, occ, (hd_edges, md_edges) = hdmd_fold_rate_map(
            spike_rates, positions, base_time,
            d["n_hd_bins"], d["n_md_bins"], md_series_deg,
            avg_vel_bins, d["vel_threshold"],
            floor=d["floor"]
        )
        out = dict(fold0_map=fold0, fold1_map=fold1, rate_map=rmean, occupancy=occ)
        edges = {"hd": hd_edges, "md": md_edges}
        base = f"fold_rate_hdmd_{d['n_hd_bins']}x{d['n_md_bins']}"
    else:  # xyhd
        fold0, fold1, rmean, occ, (x_edges, y_edges, hd_edges) = xyhd_fold_rate_map(
            spike_rates, positions, base_time,
            d["n_loc_bins"], d["n_hd_bins"],
            avg_vel_bins, d["vel_threshold"],
            floor=d["floor"]
        )
        out = dict(fold0_map=fold0, fold1_map=fold1, rate_map=rmean, occupancy=occ)
        edges = {"x": x_edges, "y": y_edges, "hd": hd_edges}
        base = f"fold_rate_xyhd_{d['n_loc_bins']}x{d['n_loc_bins']}x{d['n_hd_bins']}"

    out_npz = save_root / f"{base}.npz"
    np.savez_compressed(out_npz, **out, **{k+"_edges": v for k, v in edges.items()})
    print(f"\nSaved: {out_npz}")

    meta_path = save_root / f"{base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "mode": d["mode"],
            "tau": d["tau"],
            "vel_threshold": d["vel_threshold"],
            "n_hd_bins": d["n_hd_bins"],
            "n_md_bins": d["n_md_bins"],
            "n_loc_bins": d["n_loc_bins"],
            "md_smooth_window": d["md_smooth_window"],
            "floor": d["floor"],
            "seed": d["seed"],
        }, f, indent=2)
    print("Saved meta json.")

    elapsed_min = (time.time() - t0) / 60.0
    print(f"Done in {elapsed_min:.2f} min.")

    summary = {
        "out_npz": str(out_npz),
        "out_meta": str(meta_path),
        "mode": d["mode"],
        "shapes": {k: (np.array(v).shape if isinstance(v, np.ndarray) else None)
                   for k, v in out.items()},
        "edges_keys": list(edges.keys()),
        "elapsed_min": float(elapsed_min),
    }
    return summary

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make rate maps: XY×HD or HD×MD."
    )
    parser.add_argument("--data_path", type=str, default="./data/rodents/dusz",
                        help="Root data dir containing Dataset_{id}/{rat}/Data")
    parser.add_argument("--save_path", type=str, default="./results/rodents/dusz",
                        help="Directory to write outputs")
    parser.add_argument("--dataset", type=int, required=True, help="Dataset number (int)")
    parser.add_argument("--rat", type=str, required=True, help="Rat ID (e.g., 'R1')")

    parser.add_argument("--mode", type=str, choices=["xyhd", "hdmd"], default="xyhd",
                        help="Rate map type: 3D XY×HD or 2D HD×MD")

    # Bin params
    parser.add_argument("--n_hd_bins", type=int, default=90, help="Number of head-direction bins")
    parser.add_argument("--n_md_bins", type=int, default=90, help="Number of movement-direction bins (hdmd)")
    parser.add_argument("--n_loc_bins", type=int, default=20, help="Number of X (and Y) bins (xyhd)")

    parser.add_argument("--tau", type=float, default=0.050, help="Time bin size (s)")
    parser.add_argument("--vel_threshold", type=float, default=2.0, help="Velocity threshold (units/s)")

    parser.add_argument("--md_smooth_window", type=int, default=20,
                        help="Smoothing window (samples) for MD from XY (hdmd)")
                        
    parser.add_argument("--md_roll", type=int, default=1) # tbh maybe doesn't need to be here anyway
    parser.add_argument("--floor", type=int, default=5)

    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    get_ratemaps(args)

if __name__ == "__main__":
    main()