from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..ncnn_contract import NcnnInterpolationStage


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

    @property
    def supports_batch(self) -> bool:
        return False

    @property
    def supports_directory_batch(self) -> bool:
        return False

    def interpolate_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if len(frames) < 2:
            return list(frames)
        output = [frames[0]]
        for frame0, frame1 in zip(frames, frames[1:]):
            output.extend(self.interpolate(frame0, frame1))
            output.append(frame1)
        return output

    def process_directory(self, input_dir: str, output_dir: str,
                          input_count: int) -> int:
        raise NotImplementedError(
            "%s does not support directory batches" % type(self).__name__)

    def native_ncnn_stage(self) -> Optional["NcnnInterpolationStage"]:
        """Describe this initialized engine for a persistent NCNN worker."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
