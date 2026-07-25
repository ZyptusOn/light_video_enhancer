from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ncnn_contract import NcnnSuperResolutionStage


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

    @property
    def supports_batch(self) -> bool:
        return False

    @property
    def supports_directory_batch(self) -> bool:
        return False

    @property
    def batch_output_pixels(self) -> int:
        return 0

    @property
    def batch_output_size(self) -> Optional[Tuple[int, int]]:
        return None

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        return [self.process(frame) for frame in frames]

    def process_directory(self, input_dir: str, output_dir: str,
                          input_count: int) -> int:
        raise NotImplementedError(
            "%s does not support directory batches" % type(self).__name__)

    def native_ncnn_stage(self) -> Optional["NcnnSuperResolutionStage"]:
        """Describe this initialized engine for a persistent NCNN worker."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
