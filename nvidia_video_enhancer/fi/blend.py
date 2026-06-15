from typing import List
import numpy as np
import cv2

from .base import FrameInterpolationEngine


class BlendFIEngine(FrameInterpolationEngine):
    """
    基本帧混合插帧：对相邻帧做加权混合，生成中间帧。
    同时使用简单的运动补偿（光流扭曲）来提升效果。
    """

    def __init__(self, device: str = "cuda"):
        self._multiplier = 2
        self._width = 0
        self._height = 0

    @property
    def name(self) -> str:
        return "Optical Flow Blend"

    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None:
        self._multiplier = multiplier
        self._width = width
        self._height = height

    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]:
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        flow_fwd = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_bwd = cv2.calcOpticalFlowFarneback(
            gray1, gray0, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        h, w = gray0.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            map_fwd = np.stack([
                x_coords - flow_fwd[..., 0] * t,
                y_coords - flow_fwd[..., 1] * t,
            ], axis=-1).astype(np.float32)
            map_bwd = np.stack([
                x_coords + flow_bwd[..., 0] * (1 - t),
                y_coords + flow_bwd[..., 1] * (1 - t),
            ], axis=-1).astype(np.float32)

            warped0 = cv2.remap(frame0, map_fwd, None, cv2.INTER_LINEAR)
            warped1 = cv2.remap(frame1, map_bwd, None, cv2.INTER_LINEAR)
            blended = (warped0 * (1 - t) + warped1 * t).astype(np.uint8)
            results.append(blended)

        return results

    def release(self) -> None:
        pass
