"""Very fast dependency-free temporal blending fallback."""

from typing import List

import cv2
import numpy as np

from .base import FrameInterpolationEngine


class BlendFIEngine(FrameInterpolationEngine):
    def __init__(self, device: str = "auto", quality: str = "balanced"):
        self._multiplier = 2

    @property
    def name(self) -> str:
        return "Temporal Blend"

    def initialize(self, width: int, height: int, multiplier: int = 2) -> None:
        self._multiplier = multiplier

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray) -> List[np.ndarray]:
        return [cv2.addWeighted(frame0, 1.0 - index / self._multiplier,
                                frame1, index / self._multiplier, 0)
                for index in range(1, self._multiplier)]

    def release(self) -> None:
        pass
