"""Inference-only construction for the official EMA-VFI Small architecture."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .feature_extractor import feature_extractor
from .flow_estimation import MultiScaleFlow


def _small_config():
    channels = 16
    depths = [2, 2, 2, 2, 2]
    embed_dims = [channels, 2 * channels, 4 * channels,
                  8 * channels, 16 * channels]
    motion_dims = [
        0, 0, 0,
        8 * channels // depths[-2],
        16 * channels // depths[-1],
    ]
    backbone = {
        "embed_dims": embed_dims,
        "motion_dims": motion_dims,
        "num_heads": [8 * channels // 32, 16 * channels // 32],
        "mlp_ratios": [4, 4],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "depths": depths,
        "window_sizes": [7, 7],
    }
    flow = {
        "embed_dims": embed_dims,
        "motion_dims": motion_dims,
        "depths": depths,
        "num_heads": [8 * channels // 32, 16 * channels // 32],
        "window_sizes": [7, 7],
        "scales": [4, 8, 16],
        "hidden_dims": [4 * channels, 4 * channels],
        "c": channels,
    }
    return backbone, flow


class EMAVFISmall(nn.Module):
    def __init__(self):
        super().__init__()
        backbone_config, flow_config = _small_config()
        self.net = MultiScaleFlow(
            feature_extractor(**backbone_config), **flow_config)

    def _infer(self, images, timesteps, down_scale):
        frame0, frame1 = images[:, :3], images[:, 3:6]
        appearance, motion = self.net.feature_bone(frame0, frame1)
        flow_images = images
        flow_appearance, flow_motion = appearance, motion
        if down_scale != 1.0:
            flow_images = F.interpolate(
                images, scale_factor=down_scale, mode="bilinear",
                align_corners=False, recompute_scale_factor=False)
            flow_appearance, flow_motion = self.net.feature_bone(
                flow_images[:, :3], flow_images[:, 3:6])

        predictions = []
        for timestep in timesteps:
            flow, mask = self.net.calculate_flow(
                flow_images, float(timestep),
                flow_appearance, flow_motion)
            if down_scale != 1.0:
                flow = F.interpolate(
                    flow, size=images.shape[-2:], mode="bilinear",
                    align_corners=False) / down_scale
                mask = F.interpolate(
                    mask, size=images.shape[-2:], mode="bilinear",
                    align_corners=False)
            predictions.append(self.net.coraseWarp_and_Refine(
                images, appearance, flow, mask))
        return predictions

    def forward(self, frame0, frame1, timesteps,
                down_scale=1.0, tta=False):
        images = torch.cat((frame0, frame1), dim=1)
        if not tta:
            return self._infer(images, timesteps, down_scale)

        augmented = torch.cat(
            (images, images.flip(2).flip(3)), dim=0)
        predictions = self._infer(augmented, timesteps, down_scale)
        return [
            (prediction[:1] +
             prediction[1:2].flip(2).flip(3)) * 0.5
            for prediction in predictions
        ]

    def load_official_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        converted = {}
        for key, value in state.items():
            if "attn_mask" in key or key.endswith("HW"):
                continue
            converted[key[7:] if key.startswith("module.") else key] = value
        self.net.load_state_dict(converted, strict=True)
