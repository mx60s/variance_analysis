from __future__ import annotations

import os, warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import (
    AutoImageProcessor,
    ViTConfig,
    ViTModel,
    logging as hf_logging,
)

from .predictive_coder import PredictiveCoder

hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(
    "ignore",
    message=r"Failed to load image Python extension",
    category=UserWarning,
    module=r"torchvision\.io\.image",
)

MC_MEAN = torch.tensor([121.6697, 149.3242, 154.9510]) / 255.0
MC_STD  = torch.tensor([40.7521,  47.7267, 103.2739]) / 255.0


def _make_tensor_preprocess(
    img_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std : tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std  = torch.tensor(std).view(1, 3, 1, 1)

    def _preprocess(batch):
        x = torch.as_tensor(batch)
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:                # 0‑255 → 0‑1
            x = x / 255.0
        x = x.permute(0, 3, 1, 2)        # NHWC → NCHW

        h, w = x.shape[2:]
        if h != img_size or w != img_size:
            scale = img_size / min(h, w)
            new_h, new_w = round(h * scale), round(w * scale)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                              align_corners=False)
            y0 = (new_h - img_size) // 2
            x0 = (new_w - img_size) // 2
            x = x[:, :, y0 : y0 + img_size, x0 : x0 + img_size]

        return (x - mean.to(x.device)) / std.to(x.device)

    return _preprocess


def _make_predictive_coder(
    *,
    seq_len: int = 8,
    in_channels: int = 3,
    out_channels: int = 3,
    layers        = (2, 2),
    device: str   = "cpu",
    weights_path: Path | None = None,
):
    net = PredictiveCoder(
        in_channels=in_channels,
        out_channels=out_channels,
        layers=list(layers),
        seq_len=seq_len,
    ).to(device)

    if weights_path is not None:
        if not weights_path.exists():
            raise FileNotFoundError(f"Predictive coder checkpoint not found: {weights_path}")
        net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    return net


def load_model(
    model_name   : str,
    base_path         : Path, 
    output_layer : str | None = None,
    device       : str = "cpu",
):
    """
    Returns
    -------
    (model, preprocess_fn)
        • `model(x)` yields ONLY the requested layer.
        • `preprocess_fn` turns raw images → tensor batch on `device`.
    """
    if model_name.startswith("vit"):
        repo_id   = "google/vit-base-patch16-224-in21k"
        cache_dir = base_path / "data/weights/vit_b16"

        if model_name == "vit_b16":
            core = ViTModel.from_pretrained(repo_id, cache_dir=cache_dir)
        elif model_name == "vit_b16_untrained":
            core = ViTModel(ViTConfig())
        else:
            raise ValueError(f"Unknown model {model_name}")

        core.to(device).eval()

        processor = AutoImageProcessor.from_pretrained(repo_id, cache_dir=cache_dir)
        preprocess_fn = lambda imgs: processor(imgs, return_tensors="pt")[
            "pixel_values"
        ].to(device)

        if output_layer in {None, "last_hidden_state"}:
            idx = -1
        elif output_layer.startswith("layer_"):
            idx = int(output_layer.split("_")[1]) + 1
        else:
            raise ValueError(
                "For ViT use output_layer='layer_k' or 'last_hidden_state'"
            )

        class ViTHiddenLayer(torch.nn.Module):
            def __init__(self, hf_model, layer_idx):
                super().__init__()
                self.hf_model = hf_model
                self.layer_idx = layer_idx

            @torch.no_grad()
            def forward(self, x):
                out = self.hf_model(
                    x,
                    output_hidden_states=True,
                    return_dict=True,
                )
                return out.hidden_states[self.layer_idx]

        return ViTHiddenLayer(core, idx).to(device), preprocess_fn

    if model_name.startswith("predictive_coder"):
        ckpt_path = base_path / "data/weights/predictive_coder.ckpt"

        model = _make_predictive_coder(
            seq_len=10,
            layers=(2, 2),
            device=device,
            weights_path=ckpt_path,
        )

        def preprocess_fn(batch):
            x = torch.as_tensor(batch).float()
            if x.max() > 1.5:
                x = x / 255.0
            x = x.permute(0, 1, 4, 2, 3)
            return (x - MC_MEAN.view(1, 1, 3, 1, 1)) / MC_STD.view(1, 1, 3, 1, 1)

        encoder_layers = {'conv1', 'down1', 'down2'}
        attention_layers = {'pre_pred', 'post_pred'}

        if output_layer in encoder_layers:
            class EncoderWrapper(torch.nn.Module):
                def __init__(self, core, layer_name):
                    super().__init__()
                    self.core = core
                    self.layer_name = layer_name

                def forward(self, x):
                    return self.core.get_encoder_latents(x, layer_name=self.layer_name)

            return EncoderWrapper(model, output_layer).to(device), preprocess_fn

        elif output_layer in attention_layers:
            which = 0 if output_layer == "pre_pred" else -1

            class ResidualWrapper(torch.nn.Module):
                def __init__(self, core, which):
                    super().__init__()
                    self.core = core
                    self.which = which

                def forward(self, x):
                    return self.core.get_latents(x, which=self.which)

            return ResidualWrapper(model, which).to(device), preprocess_fn

        else:
            raise ValueError(
                f"Predictive coder requires output_layer in {encoder_layers | attention_layers}, "
                f"got '{output_layer}'"
            )

    name_map = {
        "vgg16"              : (models.vgg16,   models.VGG16_Weights.IMAGENET1K_V1),
        "vgg16_untrained"    : (models.vgg16,   None),
        "resnet50"           : (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        "resnet50_untrained" : (models.resnet50, None),
    }
    if model_name not in name_map:
        raise ValueError(f"Unknown model {model_name}")

    constructor, weights_enum = name_map[model_name]
    model = constructor(weights=weights_enum).to(device).eval()

    preprocess_fn = _make_tensor_preprocess()

    if output_layer is None:
        output_layer = "fc" if "resnet" in model_name else "classifier.6"

    model = create_feature_extractor(model, {output_layer: "out"}).to(device)
    return model, preprocess_fn
