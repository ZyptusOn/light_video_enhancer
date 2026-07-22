from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class SuperResolutionEngine(ABC):
    @abstractmethod
    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None: ...

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def release(self) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
