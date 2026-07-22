"""Portable non-AI resize fallbacks."""

from typing import Tuple

import numpy as np

from light_video_enhancer.sr.base import SuperResolutionEngine


class BicubicEngine(SuperResolutionEngine):
    def __init__(self):
        self._dst_size: Tuple[int, int] = (0, 0)

    @property
    def name(self) -> str:
        return "Bicubic"

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        self._dst_size = (dst_width, dst_height)

    def process(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(frame, self._dst_size, interpolation=cv2.INTER_CUBIC)

    def release(self) -> None:
        pass


class LanczosEngine(SuperResolutionEngine):
    def __init__(self):
        self._dst_size: Tuple[int, int] = (0, 0)

    @property
    def name(self) -> str:
        return "Lanczos"

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        self._dst_size = (dst_width, dst_height)

    def process(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(frame, self._dst_size, interpolation=cv2.INTER_LANCZOS4)

    def release(self) -> None:
        pass
