import cv2
import numpy as np
from typing import List
from .base import FrameInterpolationEngine


class OpticalFlowEngine(FrameInterpolationEngine):
    """
    基于 OpenCV Farneback 光流的帧插值引擎。

    预设（SVP 启发：pel=1 + block + 跳过精细搜索）:

      ultra:  1/8 分辨率 + 1层金字塔 + 最小窗口 (~1ms @1080p)
      fast:   1/8 分辨率 + 1层金字塔 + 小窗口    (~2ms)
      balanced: 1/4 分辨率 + 2层金字塔 + 中窗口  (~5ms)
      quality: 1/2 分辨率 + 3层金字塔 + 大窗口   (~20ms)

    所有档位 warp 都在全分辨率原始彩色帧上进行。
    GPU warp via cv2.UMat (OpenCL)。
    """

    _PRESETS = {
        "ultra":    dict(pyr_scale=0.9, levels=1, winsize=5,  iterations=1, poly_n=1, poly_sigma=0.0, scale_div=8),
        "fast":     dict(pyr_scale=0.9, levels=1, winsize=7,  iterations=1, poly_n=3, poly_sigma=1.0, scale_div=8),
        "balanced": dict(pyr_scale=0.5, levels=2, winsize=15, iterations=2, poly_n=5, poly_sigma=1.2, scale_div=4),
        "quality":  dict(pyr_scale=0.5, levels=3, winsize=15, iterations=4, poly_n=7, poly_sigma=1.5, scale_div=2),
    }

    _UMAT_AVAIL = None

    def __init__(self, quality: str = "balanced"):
        self._full_w = 0
        self._full_h = 0
        self._src_w = 0
        self._src_h = 0
        self._multiplier = 2
        self._preset = self._PRESETS.get(quality, self._PRESETS["balanced"])
        self._quality_name = quality
        self._grid_x = None
        self._grid_y = None
        self._use_umat = False

    @property
    def name(self) -> str:
        return f"Optical Flow ({self._quality_name})"

    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None:
        p = self._preset
        self._full_w = width
        self._full_h = height
        self._src_w = max(1, (width  + p["scale_div"] - 1) // p["scale_div"])
        self._src_h = max(1, (height + p["scale_div"] - 1) // p["scale_div"])
        self._multiplier = multiplier
        self._grid_y, self._grid_x = np.mgrid[0:height, 0:width].astype(np.float32)

        if OpticalFlowEngine._UMAT_AVAIL is None:
            try:
                cv2.UMat(1, 1, cv2.CV_8UC1)
                OpticalFlowEngine._UMAT_AVAIL = True
            except Exception:
                OpticalFlowEngine._UMAT_AVAIL = False
        self._use_umat = OpticalFlowEngine._UMAT_AVAIL

    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]:
        p = self._preset
        fd = p["scale_div"]

        g0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        if fd > 1:
            g0s = cv2.resize(g0, (self._src_w, self._src_h), interpolation=cv2.INTER_LINEAR)
            g1s = cv2.resize(g1, (self._src_w, self._src_h), interpolation=cv2.INTER_LINEAR)
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

        if self._use_umat:
            return self._warp_umat(frame0, frame1, flow, gx, gy)
        return self._warp_cpu(frame0, frame1, flow, gx, gy)

    def _warp_cpu(self, f0, f1, flow, gx, gy):
        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            w0 = cv2.remap(f0, gx - flow[..., 0] * t, gy - flow[..., 1] * t,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(f1, gx + flow[..., 0] * (1 - t), gy + flow[..., 1] * (1 - t),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            results.append(cv2.addWeighted(w0, 1 - t, w1, t, 0))
        return results

    def _warp_umat(self, f0, f1, flow, gx, gy):
        u0 = cv2.UMat(f0)
        u1 = cv2.UMat(f1)
        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            w0 = cv2.remap(u0, gx - flow[..., 0] * t, gy - flow[..., 1] * t,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(u1, gx + flow[..., 0] * (1 - t), gy + flow[..., 1] * (1 - t),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            results.append(cv2.addWeighted(w0, 1 - t, w1, t, 0).get())
        return results

    def release(self) -> None:
        pass
