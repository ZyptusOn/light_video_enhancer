import cv2
import numpy as np
from typing import List
from .base import FrameInterpolationEngine


class DISFlowEngine(FrameInterpolationEngine):
    """
    Dense Inverse Search 光流插帧（SVP 等效快速路径）。

    DIS 是 cv2 contrib 模块中的快速稠密光流算法，特点:
      - 预设 PRESET_FAST: 类似 SVP 的 pel=1 + 粗搜索策略
      - PRESET_MEDIUM: 类似 SVP balanced
      - PRESET_ULTRAFAST: 最快，适合实时需求

    对比 Farneback:
      - DIS 基于梯度下降 + 变分细化，速度快 5-15×
      - 内部使用多尺度框架（类似 SVP 的金字塔层）

    pip install opencv-contrib-python 以启用 DIS 引擎。
    """

    _PRESET_MAP = {
        "ultra":    cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast":     cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "balanced": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        "quality":  cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }

    _FALLBACK_PRESETS = {
        "ultra":    dict(pyr_scale=0.9, levels=1, winsize=5,  iterations=1, poly_n=1, poly_sigma=0.0, scale_div=8),
        "fast":     dict(pyr_scale=0.9, levels=1, winsize=7,  iterations=1, poly_n=3, poly_sigma=1.0, scale_div=8),
        "balanced": dict(pyr_scale=0.5, levels=2, winsize=15, iterations=2, poly_n=5, poly_sigma=1.2, scale_div=4),
        "quality":  dict(pyr_scale=0.5, levels=3, winsize=15, iterations=4, poly_n=7, poly_sigma=1.5, scale_div=2),
    }

    def __init__(self, quality: str = "balanced"):
        self._full_w = 0
        self._full_h = 0
        self._multiplier = 2
        self._quality_name = quality
        self._dis = None
        self._has_dis = False
        self._grid_x = None
        self._grid_y = None
        self._fb_preset = self._FALLBACK_PRESETS.get(quality, self._FALLBACK_PRESETS["balanced"])
        self._fb_src_w = 0
        self._fb_src_h = 0

    @property
    def name(self) -> str:
        return f"DenseInvSearch ({self._quality_name})"

    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None:
        self._full_w = width
        self._full_h = height
        self._multiplier = multiplier
        self._grid_y, self._grid_x = np.mgrid[0:height, 0:width].astype(np.float32)

        try:
            preset = self._PRESET_MAP.get(self._quality_name, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self._dis = cv2.DISOpticalFlow_create(preset)
            self._has_dis = True
        except (AttributeError, cv2.error):
            self._has_dis = False
            fd = self._fb_preset["scale_div"]
            self._fb_src_w = max(1, (width  + fd - 1) // fd)
            self._fb_src_h = max(1, (height + fd - 1) // fd)

    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]:
        if self._has_dis:
            return self._interpolate_dis(frame0, frame1)
        return self._interpolate_fallback(frame0, frame1)

    def _interpolate_dis(self, f0, f1):
        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        flow = self._dis.calc(g0, g1, None)

        gx, gy = self._grid_x, self._grid_y
        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            w0 = cv2.remap(f0, gx - flow[..., 0] * t, gy - flow[..., 1] * t,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(f1, gx + flow[..., 0] * (1 - t), gy + flow[..., 1] * (1 - t),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            results.append(cv2.addWeighted(w0, 1 - t, w1, t, 0))
        return results

    def _interpolate_fallback(self, f0, f1):
        p = self._fb_preset
        fd = p["scale_div"]

        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

        if fd > 1:
            g0s = cv2.resize(g0, (self._fb_src_w, self._fb_src_h), interpolation=cv2.INTER_LINEAR)
            g1s = cv2.resize(g1, (self._fb_src_w, self._fb_src_h), interpolation=cv2.INTER_LINEAR)
        else:
            g0s, g1s = g0, g1

        flow = cv2.calcOpticalFlowFarneback(
            g0s, g1s, None,
            p["pyr_scale"], p["levels"], p["winsize"],
            p["iterations"], p["poly_n"], p["poly_sigma"], flags=0)

        if fd > 1:
            flow = cv2.resize(flow * float(fd), (self._full_w, self._full_h),
                              interpolation=cv2.INTER_LINEAR)

        gx, gy = self._grid_x, self._grid_y
        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            w0 = cv2.remap(f0, gx - flow[..., 0] * t, gy - flow[..., 1] * t,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(f1, gx + flow[..., 0] * (1 - t), gy + flow[..., 1] * (1 - t),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            results.append(cv2.addWeighted(w0, 1 - t, w1, t, 0))
        return results

    def release(self) -> None:
        self._dis = None
