"""Official SPAN models executed by the persistent NCNN/Vulkan worker."""

import math
import os
from typing import Optional

import numpy as np

from light_video_enhancer._logging import get_logger
from light_video_enhancer._paths import get_model_dir
from light_video_enhancer.ncnn_contract import NcnnSuperResolutionStage
from light_video_enhancer.sr.base import SuperResolutionEngine

_log = get_logger(__name__)


class SPANNcnnEngine(SuperResolutionEngine):
    """Fast real-time-oriented 2x/4x SR without a PyTorch dependency."""

    def __init__(self, device: str = "auto", gpu_id: Optional[int] = None,
                 quality: str = "quality"):
        self._gpu_id = gpu_id
        self._quality = quality if quality in {
            "fast", "balanced", "quality", "ultra"} else "quality"
        self._src_w = self._src_h = 0
        self._dst_w = self._dst_h = 0
        self._scale = 2
        self._channels = 48
        self._param = ""
        self._model = ""

    @property
    def name(self) -> str:
        target = "auto GPU" if self._gpu_id is None else "GPU %d" % self._gpu_id
        return "SPAN ncnn (%dx, ch%d, %s, %s)" % (
            self._scale, self._channels, self._quality, target)

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        if self._gpu_id is not None and self._gpu_id < 0:
            raise RuntimeError("SPAN NCNN 仅支持 Vulkan GPU")
        self._src_w, self._src_h = src_width, src_height
        self._dst_w, self._dst_h = dst_width, dst_height
        ratio = max(float(dst_width) / src_width, float(dst_height) / src_height)
        target = max(2, min(4, int(math.ceil(ratio))))
        # 4x-to-2x oversampling is intentionally reserved for Ultra.
        self._scale = 4 if target > 2 or self._quality == "ultra" else 2
        self._channels = 48 if self._quality in {"fast", "balanced"} else 52
        stem = "spanx%d_ch%d" % (self._scale, self._channels)
        models = get_model_dir("ncnn", "span")
        self._param = os.path.join(models, stem + ".param")
        self._model = os.path.join(models, stem + ".bin")
        missing = [path for path in (self._param, self._model)
                   if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "SPAN 模型资源不完整: %s" % ", ".join(missing))
        _log.info("SPAN NCNN 就绪: %dx%d -> %dx%d, model=%s",
                  src_width, src_height, dst_width, dst_height, stem)

    def process(self, frame: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "SPAN 需要原生 NCNN Worker；请恢复 lve-ncnn-worker.exe")

    def native_ncnn_stage(self) -> NcnnSuperResolutionStage:
        return NcnnSuperResolutionStage(
            kind="span",
            param_path=self._param,
            model_path=self._model,
            scale=self._scale,
        )

    def release(self) -> None:
        pass
