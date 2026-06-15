import numpy as np
from typing import Tuple

from .base import SuperResolutionEngine


class ESRGANEngine(SuperResolutionEngine):
    """
    使用 Real-ESRGAN (ncnn/Vulkan 后端) 进行 AI 超分。

    pip install realesrgan-ncnn-vulkan
    """

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._upsampler = None
        self._tile_size = 400
        self._dst_width = 0
        self._dst_height = 0

    @property
    def name(self) -> str:
        return "Real-ESRGAN"

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        self._dst_width = dst_width
        self._dst_height = dst_height
        scale = max(dst_width / src_width, dst_height / src_height)

        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError:
            raise ImportError(
                "请安装 Real-ESRGAN: pip install realesrgan basicsr"
            )

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        self._upsampler = RealESRGANer(
            scale=4,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model=model,
            tile=self._tile_size,
            half=True,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._upsampler is None:
            raise RuntimeError("引擎未初始化")

        output, _ = self._upsampler.enhance(frame, outscale=1)
        out_h, out_w = output.shape[:2]

        if out_w != self._dst_width or out_h != self._dst_height:
            import cv2
            output = cv2.resize(output, (self._dst_width, self._dst_height),
                                interpolation=cv2.INTER_LANCZOS4)
        return output

    def release(self) -> None:
        self._upsampler = None


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
