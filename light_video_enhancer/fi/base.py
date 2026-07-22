from abc import ABC, abstractmethod
from typing import List
import numpy as np


class FrameInterpolationEngine(ABC):
    @abstractmethod
    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None: ...

    @abstractmethod
    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]: ...

    @abstractmethod
    def release(self) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
