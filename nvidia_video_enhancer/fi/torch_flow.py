import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from .base import FrameInterpolationEngine


class TorchFlowEngine(FrameInterpolationEngine):
    """
    GPU 光流插帧（PyTorch CUDA 加速）。

    利用 torch 直接在 GPU 上做:
      - 灰度转换 (矩阵乘法)
      - 光流计算 (Farneback → 等效近邻搜索)
      - warp (grid_sample, 硬件双线性采样)

    预设:
      ultra:   1/8 分辨率 + 最小核
      fast:    1/8 分辨率
      balanced: 1/4 分辨率
      quality: 1/2 分辨率
    """

    _PRESETS = {
        "ultra":    dict(scale_div=8,  search_radius=4),
        "fast":     dict(scale_div=8,  search_radius=6),
        "balanced": dict(scale_div=4,  search_radius=8),
        "quality":  dict(scale_div=2,  search_radius=12),
    }

    def __init__(self, quality: str = "balanced"):
        if not torch.cuda.is_available():
            raise ImportError("PyTorch CUDA 不可用")

        self._full_w = 0
        self._full_h = 0
        self._multiplier = 2
        self._quality_name = quality
        self._preset = self._PRESETS.get(quality, self._PRESETS["balanced"])
        self._device = torch.device("cuda")
        self._grid = None
        self._stream = torch.cuda.Stream()

    @property
    def name(self) -> str:
        return f"GPU Optical Flow ({self._quality_name})"

    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None:
        p = self._preset
        d = p["scale_div"]
        self._full_w = width
        self._full_h = height
        self._src_w = max(1, (width  + d - 1) // d)
        self._src_h = max(1, (height + d - 1) // d)
        self._radius = p["search_radius"]
        self._multiplier = multiplier

        gx = torch.arange(width,  dtype=torch.float32, device=self._device)
        gy = torch.arange(height, dtype=torch.float32, device=self._device)
        gy, gx = torch.meshgrid(gy, gx, indexing="ij")
        self._grid = torch.stack([gx, gy], dim=0)

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(bgr).to(self._device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float()
        return t

    def _bgr_to_gray(self, bgr_tensor: torch.Tensor) -> torch.Tensor:
        b, g, r = bgr_tensor[:, 0], bgr_tensor[:, 1], bgr_tensor[:, 2]
        return 0.114 * b + 0.587 * g + 0.299 * r

    def _patch_match(self, g0: torch.Tensor, g1: torch.Tensor,
                     radius: int) -> torch.Tensor:
        """SVP 等价块匹配 (GPU + unfold, O(1) kernel launches)"""
        B, H, W = g0.shape
        g0_patches = g0[0].unsqueeze(0).unsqueeze(0)

        g1_4d = g1[0].unsqueeze(0).unsqueeze(0)
        win = 2 * radius + 1
        g1_unfolded = F.unfold(g1_4d, (win, win), padding=radius)
        g1_unfolded = g1_unfolded.view(1, win * win, H, W)

        g0_broadcast = g0_patches.view(1, 1, H, W)
        error = (g1_unfolded - g0_broadcast).abs()

        best_idx = error.argmin(dim=0)
        dy = best_idx // win - radius
        dx = best_idx % win - radius

        return torch.stack([dx.float(), dy.float()], dim=0)

    def _calc_flow(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        p = self._preset
        d = p["scale_div"]
        r = p["search_radius"]

        g0 = self._bgr_to_gray(f0)
        g1 = self._bgr_to_gray(f1)

        if d > 1:
            g0 = F.interpolate(g0.unsqueeze(0), size=(self._src_h, self._src_w),
                               mode="bilinear", align_corners=False).squeeze(0)
            g1 = F.interpolate(g1.unsqueeze(0), size=(self._src_h, self._src_w),
                               mode="bilinear", align_corners=False).squeeze(0)

        flow_small = self._patch_match(g0.unsqueeze(0), g1.unsqueeze(0), r)

        if d > 1:
            flow_small = flow_small.unsqueeze(0).float()
            flow_small = F.interpolate(flow_small, size=(self._full_h, self._full_w),
                                       mode="bilinear", align_corners=False)
            flow_small = flow_small.squeeze(0) * float(d)

        flow = flow_small.permute(1, 2, 0)
        flow = -flow
        return flow

    def _warp(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        _, _, H, W = frame.shape
        flow_grid = flow.permute(2, 0, 1).unsqueeze(0)
        flow_grid = flow_grid.float()
        norm = flow_grid.clone()
        norm[:, 0] = 2.0 * norm[:, 0] / max(W - 1, 1) - 1.0
        norm[:, 1] = 2.0 * norm[:, 1] / max(H - 1, 1) - 1.0
        normalized_xy = norm.permute(0, 2, 3, 1)
        return F.grid_sample(frame.float(), normalized_xy,
                             mode="bilinear", padding_mode="border",
                             align_corners=True)

    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]:
        f0 = self._to_tensor(frame0)
        f1 = self._to_tensor(frame1)
        flow = self._calc_flow(f0, f1)

        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            w0 = self._warp(f0, -flow * t)
            w1 = self._warp(f1, flow * (1 - t))
            blended = w0 * (1 - t) + w1 * t
            blended = blended.clamp(0, 255).byte()
            blended = blended[0].permute(1, 2, 0)
            results.append(blended.cpu().numpy())

        return results

    def release(self) -> None:
        pass
