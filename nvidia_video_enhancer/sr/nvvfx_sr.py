import numpy as np
from .base import SuperResolutionEngine


class NVVFX_SR_Engine(SuperResolutionEngine):
    def __init__(self, device: int = 0, quality: str = "HIGH"):
        self._device = device
        self._quality_name = quality
        self._sr = None
        self._stream_ptr = None
        self._debug_done = False

    @property
    def name(self) -> str:
        return f"NVIDIA VFX VSR ({self._quality_name})"

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "nvidia-vfx 引擎需要 CUDA 版 PyTorch。\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
        from nvvfx import VideoSuperRes

        quality = getattr(VideoSuperRes.QualityLevel, self._quality_name, None)
        if quality is None:
            ratio = max(dst_width / src_width, dst_height / src_height)
            quality = VideoSuperRes.QualityLevel.HIGH if ratio >= 2 else VideoSuperRes.QualityLevel.DENOISE_HIGH

        self._sr = VideoSuperRes(device=self._device, quality=quality)
        self._sr.input_width = src_width
        self._sr.input_height = src_height
        self._sr.output_width = dst_width
        self._sr.output_height = dst_height
        self._sr.load()
        self._stream_ptr = torch.cuda.current_stream().cuda_stream

    def process(self, frame: np.ndarray) -> np.ndarray:
        import torch

        bgr = np.ascontiguousarray(frame.copy())
        t = torch.from_numpy(bgr).to(f"cuda:{self._device}")
        t = t.permute(2, 0, 1).contiguous().float().div_(255.0)

        out = self._sr.run(t, stream_ptr=self._stream_ptr)
        del t

        r = torch.from_dlpack(out.image)
        del out

        r = r.permute(1, 2, 0).contiguous()
        r = r.mul_(255.0).clamp_(0.0, 255.0).to(torch.uint8)
        n = r.cpu().numpy()
        del r

        return n

    def release(self) -> None:
        self._sr = None
        import torch
        torch.cuda.empty_cache()
