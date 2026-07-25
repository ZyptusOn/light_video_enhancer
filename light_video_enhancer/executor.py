"""Common contract for persistent frame-batch accelerators."""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class FrameBatchExecutor(ABC):
    """A ready-to-run accelerator that transforms an overlapped frame batch.

    Executors own their worker process and runtime resources.  VideoEnhancer
    only needs this small interface, regardless of whether the implementation
    uses CUDA, Vulkan, shared memory, or another transport.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @abstractmethod
    def process(self, frames: List[np.ndarray],
                skip_first: bool = False) -> List[Any]: ...

    @abstractmethod
    def release(self) -> None: ...
