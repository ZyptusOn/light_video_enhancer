"""RIFE ncnn-vulkan using the executable's efficient directory batch mode."""

import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from light_video_enhancer.fi.base import FrameInterpolationEngine
from light_video_enhancer._image_batch import (
    make_directory, ncnn_jobs, read_frames, validate_outputs, write_frames)
from light_video_enhancer._logging import get_logger
from light_video_enhancer._paths import get_model_dir, get_pkg_dir

_log = get_logger(__name__)


def _no_window_flag() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class RIFENcnnEngine(FrameInterpolationEngine):
    """Portable Vulkan RIFE with directory-to-directory chaining support."""

    def __init__(self, quality: str = "balanced", gpu_id: Optional[int] = None):
        self._quality = quality
        self._gpu_id = gpu_id
        self._width = 0
        self._height = 0
        self._multiplier = 2
        self._exe = ""
        self._model_dir = ""

    @property
    def name(self) -> str:
        target = "auto GPU" if self._gpu_id is None else (
            "CPU" if self._gpu_id < 0 else "GPU %d" % self._gpu_id)
        return "RIFE ncnn-vulkan (batch, %s)" % target

    def initialize(self, width: int, height: int, multiplier: int = 2) -> None:
        if multiplier < 2:
            raise ValueError("RIFE 插帧倍率至少为 2")
        base = os.path.join(get_pkg_dir(), "ncnn", "rife")
        self._exe = os.path.join(base, "rife-ncnn-vulkan.exe")
        self._model_dir = get_model_dir("ncnn", "rife", "rife-v4.6")
        required = [self._exe, os.path.join(self._model_dir, "flownet.param"),
                    os.path.join(self._model_dir, "flownet.bin")]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("RIFE ncnn 资源不完整: %s" % ", ".join(missing))
        self._width, self._height = width, height
        self._multiplier = multiplier
        _log.info("RIFE ncnn 批处理就绪: %dx%d, %dx", width, height, multiplier)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray) -> List[np.ndarray]:
        sequence = self.interpolate_batch([frame0, frame1])
        return sequence[1:-1]

    def interpolate_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        if len(frames) == 1:
            return [frames[0].copy()]
        target_count = (len(frames) - 1) * self._multiplier + 1
        with tempfile.TemporaryDirectory(prefix="lve_rife_") as work:
            input_dir = os.path.join(work, "input")
            output_dir = os.path.join(work, "output")
            write_frames(frames, input_dir, "RIFE ncnn")
            self.process_directory(input_dir, output_dir, len(frames))
            return read_frames(output_dir, target_count,
                               (self._width, self._height), "RIFE ncnn")

    def process_directory(self, input_dir: str, output_dir: str,
                          input_count: int) -> int:
        """Run one directory job and leave images for a following NCNN stage."""
        if input_count < 2:
            raise ValueError("RIFE ncnn 目录批处理至少需要 2 帧")
        target_count = (input_count - 1) * self._multiplier + 1
        make_directory(output_dir)
        command = [
            self._exe, "-i", input_dir.replace("\\", "/"),
            "-o", output_dir.replace("\\", "/"),
            "-n", str(target_count), "-m", self._model_dir.replace("\\", "/"),
            "-j", ncnn_jobs(self._width, self._height), "-f", "%08d.png",
        ]
        if self._gpu_id is not None:
            command.extend(["-g", str(self._gpu_id)])
        if self._width * self._height > 1920 * 1080:
            command.append("-u")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=max(120, input_count * 30), creationflags=_no_window_flag())
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError("RIFE ncnn 批处理失败: %s" %
                               (error or result.returncode))
        validate_outputs(output_dir, target_count, "RIFE ncnn")
        return target_count

    def release(self) -> None:
        pass
