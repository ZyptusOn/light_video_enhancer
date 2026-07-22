"""Correct, bounded-memory CUDA block-matching interpolation."""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .base import FrameInterpolationEngine


class TorchFlowEngine(FrameInterpolationEngine):
    _PRESETS = {
        "ultra": (8, 2),
        "fast": (8, 3),
        "balanced": (4, 4),
        "quality": (2, 5),
    }

    def __init__(self, quality: str = "balanced"):
        if not torch.cuda.is_available():
            raise ImportError("GPU 光流需要当前 Python 环境中的 CUDA PyTorch")
        self._quality = quality if quality in self._PRESETS else "balanced"
        self._device = torch.device("cuda")
        self._width = self._height = 0
        self._small_w = self._small_h = 0
        self._multiplier = 2
        self._grid = None

    @property
    def name(self) -> str:
        return "CUDA Block Flow (%s)" % self._quality

    def initialize(self, width: int, height: int, multiplier: int = 2) -> None:
        scale_div, _ = self._PRESETS[self._quality]
        self._width, self._height = width, height
        self._small_w = max(1, (width + scale_div - 1) // scale_div)
        self._small_h = max(1, (height + scale_div - 1) // scale_div)
        self._multiplier = multiplier
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self._device),
            torch.arange(width, dtype=torch.float32, device=self._device), indexing="ij")
        self._grid = torch.stack((x, y), dim=0).unsqueeze(0)

    def _tensor(self, frame: np.ndarray):
        return (torch.from_numpy(np.ascontiguousarray(frame)).to(self._device, non_blocking=True)
                .permute(2, 0, 1).unsqueeze(0).float())

    @staticmethod
    def _gray(frame):
        return (frame[:, 0:1] * 0.114 + frame[:, 1:2] * 0.587 + frame[:, 2:3] * 0.299)

    def _flow(self, frame0, frame1):
        scale_div, radius = self._PRESETS[self._quality]
        g0 = F.interpolate(self._gray(frame0), (self._small_h, self._small_w),
                           mode="bilinear", align_corners=False)
        g1 = F.interpolate(self._gray(frame1), (self._small_h, self._small_w),
                           mode="bilinear", align_corners=False)
        g0 = F.avg_pool2d(g0, 3, stride=1, padding=1)
        g1 = F.avg_pool2d(g1, 3, stride=1, padding=1)
        best = torch.full_like(g0, float("inf"))
        flow_x = torch.zeros_like(g0)
        flow_y = torch.zeros_like(g0)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = torch.roll(g1, shifts=(-dy, -dx), dims=(2, 3))
                error = F.avg_pool2d((g0 - shifted).abs(), 5, stride=1, padding=2)
                if dy < 0:
                    error[:, :, : -dy, :] = float("inf")
                elif dy > 0:
                    error[:, :, -dy:, :] = float("inf")
                if dx < 0:
                    error[:, :, :, : -dx] = float("inf")
                elif dx > 0:
                    error[:, :, :, -dx:] = float("inf")
                improve = error < best
                best = torch.where(improve, error, best)
                flow_x = torch.where(improve, torch.as_tensor(float(dx), device=self._device), flow_x)
                flow_y = torch.where(improve, torch.as_tensor(float(dy), device=self._device), flow_y)
        flow = torch.cat((flow_x, flow_y), dim=1)
        flow = F.interpolate(flow, (self._height, self._width), mode="bilinear", align_corners=False)
        return flow * float(scale_div)

    def _warp(self, frame, offset):
        coords = self._grid + offset
        x = 2.0 * coords[:, 0] / max(self._width - 1, 1) - 1.0
        y = 2.0 * coords[:, 1] / max(self._height - 1, 1) - 1.0
        grid = torch.stack((x, y), dim=-1)
        return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray) -> List[np.ndarray]:
        with torch.inference_mode():
            f0, f1 = self._tensor(frame0), self._tensor(frame1)
            flow = self._flow(f0, f1)
            result = []
            for index in range(1, self._multiplier):
                t = index / self._multiplier
                a = self._warp(f0, -flow * t)
                b = self._warp(f1, flow * (1.0 - t))
                out = (a * (1.0 - t) + b * t).clamp_(0, 255).byte()
                result.append(out[0].permute(1, 2, 0).cpu().numpy())
            return result

    def release(self) -> None:
        self._grid = None
        torch.cuda.empty_cache()
