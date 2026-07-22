"""Classic ESRGAN and Real-ESRGAN NCNN/Vulkan super-resolution engines."""

import math
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np

from light_video_enhancer._image_batch import (
    make_directory, ncnn_jobs, read_frames, validate_outputs, write_frames)
from light_video_enhancer._logging import get_logger
from light_video_enhancer._paths import get_pkg_dir
from light_video_enhancer.sr.base import SuperResolutionEngine

_log = get_logger(__name__)


def _find_realesrgan_exe() -> str:
    base = os.path.join(get_pkg_dir(), "ncnn", "realesrgan")
    exe = os.path.join(base, "realesrgan-ncnn-vulkan.exe")
    if os.path.isfile(exe):
        return exe
    raise FileNotFoundError("未找到 realesrgan-ncnn-vulkan.exe: %s" % base)


class _NcnnESRGANBase(SuperResolutionEngine):
    def __init__(self, classic: bool, device: str = "auto",
                 gpu_id: Optional[int] = None, quality: str = "quality"):
        self._classic = classic
        self._gpu_id = gpu_id
        self._quality = quality if quality in {
            "fast", "balanced", "quality", "ultra"} else "quality"
        self._src_w = self._src_h = 0
        self._dst_w = self._dst_h = 0
        self._target_scale = 2
        self._native_scale = 4
        self._model_name = ""
        self._exe = ""
        self._models_dir = ""

    @property
    def name(self) -> str:
        target = "auto GPU" if self._gpu_id is None else (
            "CPU" if self._gpu_id < 0 else "GPU %d" % self._gpu_id)
        family = "ESRGAN classic" if self._classic else "Real-ESRGAN"
        return "%s ncnn (%s, %s)" % (family, self._quality, target)

    @property
    def batch_output_pixels(self) -> int:
        return ((self._src_w * self._native_scale) *
                (self._src_h * self._native_scale))

    @property
    def batch_output_size(self):
        return (self._src_w * self._native_scale,
                self._src_h * self._native_scale)

    def _select_model(self) -> Tuple[str, int]:
        if self._classic:
            return "esrgan-x4", 4
        if self._quality == "balanced":
            return "realesrgan-x4plus-anime", 4
        if self._quality in {"quality", "ultra"}:
            return "realesrgan-x4plus", 4
        return "realesr-animevideov3", self._target_scale

    def _model_files(self) -> Tuple[str, str]:
        stem = self._model_name
        if stem == "realesr-animevideov3":
            stem += "-x%d" % self._native_scale
        return (os.path.join(self._models_dir, stem + ".param"),
                os.path.join(self._models_dir, stem + ".bin"))

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        self._src_w, self._src_h = src_width, src_height
        self._dst_w, self._dst_h = dst_width, dst_height
        ratio = max(float(dst_width) / src_width, float(dst_height) / src_height)
        self._target_scale = max(2, min(4, int(math.ceil(ratio))))
        self._exe = _find_realesrgan_exe()
        self._models_dir = os.path.join(os.path.dirname(self._exe), "models")
        self._model_name, self._native_scale = self._select_model()
        required = [self._exe] + list(self._model_files())
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("ESRGAN 模型资源不完整: %s" %
                                    ", ".join(missing))
        _log.info("%s 就绪: %dx%d -> %dx%d, model=%s, native=%dx",
                  "ESRGAN" if self._classic else "Real-ESRGAN",
                  src_width, src_height, dst_width, dst_height,
                  self._model_name, self._native_scale)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.process_batch([frame])[0]

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        with tempfile.TemporaryDirectory(prefix="lve_esrgan_") as work:
            input_dir = os.path.join(work, "input")
            output_dir = os.path.join(work, "output")
            write_frames(frames, input_dir, self.name)
            self.process_directory(input_dir, output_dir, len(frames))
            return read_frames(output_dir, len(frames),
                               (self._dst_w, self._dst_h), self.name)

    def process_directory(self, input_dir: str, output_dir: str,
                          input_count: int) -> int:
        if input_count < 1:
            return 0
        make_directory(output_dir)
        native_w = self._src_w * self._native_scale
        native_h = self._src_h * self._native_scale
        command = [
            self._exe, "-i", input_dir.replace("\\", "/"),
            "-o", output_dir.replace("\\", "/"),
            "-s", str(self._native_scale),
            "-m", self._models_dir.replace("\\", "/"),
            "-n", self._model_name,
            "-j", ncnn_jobs(self._src_w, self._src_h, native_w, native_h),
            "-f", "png",
        ]
        if self._quality == "ultra":
            command.append("-x")
        if self._gpu_id is not None:
            command.extend(["-g", str(self._gpu_id)])
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.path.dirname(self._exe),
            timeout=max(120, input_count * 60),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt" else 0)
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError("%s 批处理失败: %s" %
                               (self.name, error or result.returncode))
        validate_outputs(output_dir, input_count, self.name)
        return input_count

    def release(self) -> None:
        pass


class RealESRGANEngine(_NcnnESRGANBase):
    """Real-ESRGAN: fast AnimeVideo-v3 or high-quality x4plus."""

    def __init__(self, device: str = "auto", gpu_id: Optional[int] = None,
                 quality: str = "quality"):
        super().__init__(False, device=device, gpu_id=gpu_id, quality=quality)


class ESRGANEngine(_NcnnESRGANBase):
    """Original ESRGAN x4 perceptual model deployed through NCNN/Vulkan."""

    def __init__(self, device: str = "auto", gpu_id: Optional[int] = None,
                 quality: str = "quality"):
        super().__init__(True, device=device, gpu_id=gpu_id, quality=quality)
