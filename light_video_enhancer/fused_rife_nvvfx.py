"""Windows 10/11 fused RIFE -> NV-VFX CUDA subprocess."""

import os
import subprocess
import sys
import threading
from typing import List, Optional

import numpy as np

from ._logging import get_logger
from ._paths import get_pkg_file
from ._shared_frames import (
    FramedPipeReader, SharedNDArray, close_process_pipes, write_framed)
from .executor import FrameBatchExecutor
from .fi._scene_detect import classify_pair
from .fi.rife import _find_weight_file
from .sr.nvvfx_sr import _GUI_QUALITY_LEVELS, _NVVFX_QUALITY_LEVELS

_log = get_logger(__name__)


def modern_windows_available() -> bool:
    if os.environ.get("LVE_DISABLE_FUSED_CUDA", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.name != "nt":
        return False
    try:
        return sys.getwindowsversion().major >= 10
    except AttributeError:
        return False


class YUV420Frame:
    """Owned planar I420 frame produced by the CUDA fast path."""

    __slots__ = ("data", "width", "height", "is_yuv420")

    def __init__(self, data: np.ndarray, width: int, height: int):
        self.data = data
        self.width = int(width)
        self.height = int(height)
        self.is_yuv420 = True


class FusedRifeNvvfxEngine(FrameBatchExecutor):
    """Keep RIFE intermediate tensors on CUDA before NV-VFX inference."""

    def __init__(self, torch_python: Optional[str], quality: str = "quality"):
        requested = str(quality or "quality")
        self._quality_name = _GUI_QUALITY_LEVELS.get(requested.lower(), requested.upper())
        if self._quality_name not in _NVVFX_QUALITY_LEVELS:
            self._quality_name = "HIGH"
        self._quality_value = _NVVFX_QUALITY_LEVELS[self._quality_name]
        self._torch_python = torch_python
        self._proc = None
        self._reader = None
        self._stderr_thread = None
        self._stderr_lines: List[str] = []
        self._shared_input = None
        self._shared_output = None
        self._max_input_frames = 3
        self._max_output_frames = 5
        self._multiplier = 2
        self._src_w = self._src_h = self._dst_w = self._dst_h = 0

    @property
    def name(self) -> str:
        return "RIFE v4.25 + NVIDIA Video Effects VSR (%s, fused CUDA/shm)" % self._quality_name

    @property
    def batch_size(self) -> int:
        return self._max_input_frames

    @staticmethod
    def output_count(input_count: int, multiplier: int, skip_first: bool = False) -> int:
        if input_count <= 0:
            return 0
        return max(0, (input_count - 1) * multiplier + (0 if skip_first else 1))

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int, multiplier: int) -> None:
        if not modern_windows_available():
            raise RuntimeError("融合 CUDA 快速路径只在 Windows 10/11 启用")
        if multiplier < 2:
            raise ValueError("RIFE 插帧倍率至少为 2")
        self._src_w, self._src_h = src_width, src_height
        self._dst_w, self._dst_h = dst_width, dst_height
        self._multiplier = multiplier
        output_bytes = dst_width * dst_height * 3 // 2
        budget = 96 * 1024 * 1024
        affordable_outputs = max(multiplier + 1, budget // max(1, output_bytes))
        affordable_pairs = max(1, (affordable_outputs - 1) // multiplier)
        self._max_input_frames = min(3, affordable_pairs + 1)
        self._max_output_frames = self.output_count(
            self._max_input_frames, multiplier, skip_first=False)
        self._shared_input = SharedNDArray.create(
            (self._max_input_frames, src_height, src_width, 3))
        self._shared_output = SharedNDArray.create(
            (self._max_output_frames, output_bytes))

        area = src_width * src_height
        scale = 0.25 if area > 3840 * 2160 else (0.5 if area > 1920 * 1080 * 2 else 1.0)
        alignment = max(128, int(128 / scale))
        pad_w = ((src_width + alignment - 1) // alignment) * alignment - src_width
        pad_h = ((src_height + alignment - 1) // alignment) * alignment - src_height
        model_path = _find_weight_file()
        if not model_path:
            self.release()
            raise FileNotFoundError("缺少 RIFE 权重: light_video_enhancer/fi/flownet.pkl")

        python_executable = self._torch_python or sys.executable
        script = get_pkg_file("_fused_rife_nvvfx_infer.py")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(
                [python_executable, "-u", script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
            self._reader = FramedPipeReader(self._proc.stdout, "lve-fused-cuda-read")
            self._stderr_thread = threading.Thread(
                target=self._read_stderr, args=(self._proc.stderr,), daemon=True)
            self._stderr_thread.start()
            self._write({
                "model_path": model_path,
                "quality": self._quality_value,
                "src_w": src_width, "src_h": src_height,
                "dst_w": dst_width, "dst_h": dst_height,
                "multiplier": multiplier,
                "scale": scale, "pad_w": pad_w, "pad_h": pad_h,
                "shared_input": self._shared_input.descriptor(),
                "shared_output": self._shared_output.descriptor(),
            })
            reply = self._read(120.0)
            if isinstance(reply, dict) and "error" in reply:
                raise RuntimeError(reply["error"])
        except Exception:
            details = self._stderr_text()
            self.release()
            if details:
                _log.debug("融合 Worker 错误输出:\n%s", details)
            raise
        _log.info("融合 CUDA Worker 就绪: %dx%d -> %dx%d, %dx, batch=%d",
                  src_width, src_height, dst_width, dst_height,
                  multiplier, self._max_input_frames)

    def process(self, frames: List[np.ndarray], skip_first: bool = False) -> List[np.ndarray]:
        if not self._proc or self._shared_input is None or self._shared_output is None:
            raise RuntimeError("融合 CUDA Worker 尚未初始化")
        if not frames:
            return []
        if len(frames) > self._max_input_frames:
            raise ValueError("融合批次过大: %d > %d" % (len(frames), self._max_input_frames))
        for index, frame in enumerate(frames):
            if frame.shape != (self._src_h, self._src_w, 3):
                raise ValueError("融合 Worker 输入尺寸不一致")
            np.copyto(self._shared_input.array[index], frame)
        pair_modes = [classify_pair(left, right)
                      for left, right in zip(frames, frames[1:])]
        self._write({"command": "process", "count": len(frames),
                     "skip_first": bool(skip_first), "pair_modes": pair_modes})
        reply = self._read(60.0)
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError("融合 CUDA 推理失败: %s" % reply["error"])
        count = int(reply.get("count", -1)) if isinstance(reply, dict) else -1
        expected = self.output_count(len(frames), self._multiplier, skip_first)
        if count != expected or count > self._max_output_frames:
            raise RuntimeError("融合 Worker 返回帧数错误: %d，预期 %d" % (count, expected))
        return [YUV420Frame(self._shared_output.array[index].copy(),
                            self._dst_w, self._dst_h) for index in range(count)]

    def _write(self, value) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("融合 CUDA Worker 未运行")
        write_framed(self._proc.stdin, value)

    def _read(self, timeout: float):
        if self._reader is None:
            raise RuntimeError("融合 CUDA Worker 未运行")
        try:
            return self._reader.read(timeout)
        except TimeoutError as exc:
            raise TimeoutError("融合 CUDA Worker 在 %.0f 秒内没有响应" % timeout) from exc
        except EOFError as exc:
            raise RuntimeError("融合 CUDA Worker 已退出: %s\n%s" % (
                exc, self._stderr_text())) from exc

    def _read_stderr(self, pipe) -> None:
        try:
            for line in pipe:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_lines.append(text)
        except Exception:
            pass

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-30:])

    def release(self) -> None:
        process, self._proc = self._proc, None
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=5)
            except Exception:
                process.kill()
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        close_process_pipes(process)
        for name in ("_shared_input", "_shared_output"):
            value = getattr(self, name, None)
            if value is not None:
                try:
                    value.close()
                except Exception:
                    pass
                setattr(self, name, None)
