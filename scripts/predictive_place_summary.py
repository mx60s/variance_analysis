"""Reproduce the exclusive-place subset summaries quoted in the manuscript."""
from pathlib import Path
import json
import numpy as np
root = Path(__file__).resolve().parents[1]
summary = {}
for layer in ['conv1', 'down1', 'pre_pred', 'post_pred']:
    path = root / 'outputs/paper_recipe/analysis' / layer
    ev = np.load(path / 'ev.npz')['ev']
    classes = json.loads((path / 'classifications.json').read_text())
    indices = np.asarray(classes['exclusive_place_cells_indices'], dtype=int)
    values = ev[indices]
    values = values[np.isfinite(values).all(axis=1)]
    summary[layer] = dict(classified=len(indices), defined=len(values),
                          median=np.median(values, axis=0).tolist(),
                          location_greater=float(np.mean(values[:, 0] > values[:, 1])))
print(json.dumps(summary, indent=2))
