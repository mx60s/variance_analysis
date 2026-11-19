"""Authors' two-head configuration with per-frame causal normalization."""

from src.ann.predictive_coder import PredictiveCoder


def make_model(architecture=None):
    config = dict(
        architecture
        or {
            "layers": [2, 2, 2, 2],
            "seq_len": 10,
            "causal_normalization": True,
            "num_heads": 2,
        }
    )
    heads = config.pop("num_heads")
    if heads != 2 or not config["causal_normalization"]:
        raise ValueError("Expected causal two-head recipe")
    model = PredictiveCoder(3, 3, **config)
    # Projection dimensions do not change; only the partition into heads does.
    for attention in [model.attention_1, model.attention_2]:
        attention.num_heads = heads
        attention.d_head = 128 // heads
    return model
