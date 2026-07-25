"""Real-CUGAN ncnn-vulkan with reusable directory batch processing."""

import math
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from light_video_enhancer._image_batch import (
    make_directory, ncnn_jobs, ncnn_tile, read_frames, validate_outputs,
    write_frames)
from light_video_enhancer._logging import get_logger
from light_video_enhancer._paths import get_model_dir, get_pkg_dir
from light_video_enhancer.ncnn_contract import NcnnSuperResolutionStage
from light_video_enhancer.sr.base import SuperResolutionEngine

_log = get_logger(__name__)


class RealCUGANEngine(SuperResolutionEngine):
    _QUALITY = {
        "fast": (-1, False, "no-denoise"),
        "balanced": (0, False, "conservative"),
        "quality": (3, False, "denoise3x"),
        "ultra": (3, True, "denoise3x"),
    }

    def __init__(self, device: str = "auto", gpu_id: Optional[int] = None,
                 quality: str = "quality"):
        self._gpu_id = gpu_id
        self._quality = quality if quality in self._QUALITY else "quality"
        self._src_w = self._src_h = 0
        self._dst_w = self._dst_h = 0
        self._scale = 2
        self._exe = ""
        self._models = ""

    @property
    def name(self) -> str:
        target = "auto GPU" if self._gpu_id is None else (
            "CPU" if self._gpu_id < 0 else "GPU %d" % self._gpu_id)
        return "Real-CUGAN ncnn (%dx, %s, %s)" % (
            self._scale, self._quality, target)

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def supports_directory_batch(self) -> bool:
        return True

    @property
    def batch_output_pixels(self) -> int:
        return ((self._src_w * self._scale) *
                (self._src_h * self._scale))

    @property
    def batch_output_size(self):
        return (self._src_w * self._scale,
                self._src_h * self._scale)

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        base = os.path.join(get_pkg_dir(), "ncnn", "realcugan")
        self._exe = os.path.join(base, "realcugan-ncnn-vulkan.exe")
        self._models = get_model_dir("ncnn", "realcugan", "models-se")
        ratio = max(float(dst_width) / src_width, float(dst_height) / src_height)
        self._scale = max(2, min(4, int(math.ceil(ratio))))
        _noise, _tta, model_kind = self._QUALITY[self._quality]
        required = [
            self._exe,
            os.path.join(self._models, "up%dx-%s.param" %
                         (self._scale, model_kind)),
            os.path.join(self._models, "up%dx-%s.bin" %
                         (self._scale, model_kind)),
        ]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Real-CUGAN 资源不完整: %s" %
                                    ", ".join(missing))
        self._src_w, self._src_h = src_width, src_height
        self._dst_w, self._dst_h = dst_width, dst_height
        _log.info("Real-CUGAN 批处理就绪: %dx%d -> %dx%d (%s)",
                  src_width, src_height, dst_width, dst_height, self._quality)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.process_batch([frame])[0]

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        with tempfile.TemporaryDirectory(prefix="lve_cugan_") as work:
            input_dir = os.path.join(work, "input")
            output_dir = os.path.join(work, "output")
            write_frames(frames, input_dir, "Real-CUGAN")
            self.process_directory(input_dir, output_dir, len(frames))
            return read_frames(output_dir, len(frames),
                               (self._dst_w, self._dst_h), "Real-CUGAN")

    def process_directory(self, input_dir: str, output_dir: str,
                          input_count: int) -> int:
        """Upscale a directory without decoding results back into Python."""
        if input_count < 1:
            return 0
        make_directory(output_dir)
        noise, tta, _model_kind = self._QUALITY[self._quality]
        command = [
            self._exe, "-i", input_dir.replace("\\", "/"),
            "-o", output_dir.replace("\\", "/"),
            "-s", str(self._scale), "-n", str(noise),
            "-m", self._models.replace("\\", "/"),
            "-j", ncnn_jobs(self._src_w, self._src_h,
                            self._src_w * self._scale,
                            self._src_h * self._scale, engine="realcugan"),
            "-f", "png",
        ]
        tile = ncnn_tile("realcugan")
        if tile:
            command.extend(["-t", str(tile)])
        if tta:
            command.append("-x")
        if self._gpu_id is not None:
            command.extend(["-g", str(self._gpu_id)])
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=max(120, input_count * 45),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt" else 0)
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError("Real-CUGAN 批处理失败: %s" %
                               (error or result.returncode))
        validate_outputs(output_dir, input_count, "Real-CUGAN")
        return input_count

    def native_ncnn_stage(self) -> NcnnSuperResolutionStage:
        noise, tta, model_kind = self._QUALITY[self._quality]
        stem = "up%dx-%s" % (self._scale, model_kind)
        return NcnnSuperResolutionStage(
            kind="realcugan",
            param_path=os.path.join(self._models, stem + ".param"),
            model_path=os.path.join(self._models, stem + ".bin"),
            scale=self._scale,
            tta=tta,
            noise=noise,
            syncgap=3,
            tile=ncnn_tile("realcugan") or 0)

    def release(self) -> None:
        pass
