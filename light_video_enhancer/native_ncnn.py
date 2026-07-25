"""Persistent NCNN/Vulkan execution through a native shared-memory worker.

The legacy NCNN command-line tools remain the compatibility fallback.  This
module removes their per-batch process startup and PNG directory hand-offs
when the selected interpolation/SR engines can run in one native process.
"""

import os
import queue
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._logging import get_logger
from ._paths import get_pkg_file
from ._shared_frames import SharedNDArray
from .executor import FrameBatchExecutor
from .fi._scene_detect import classify_pair
from .ncnn_contract import (
    NcnnInterpolationStage, NcnnSuperResolutionStage)

_log = get_logger(__name__)

_MAGIC = 0x4E45564C
_VERSION = 1
_COMMAND_PROCESS = 1
_COMMAND_STOP = 2
_REQUEST = struct.Struct("<IIIII")
_REPLY = struct.Struct("<IIiIId")


def native_worker_available() -> bool:
    disabled = os.environ.get("LVE_DISABLE_FUSED_NCNN", "").strip().lower()
    if disabled in {"1", "true", "yes"} or os.name != "nt":
        return False
    try:
        if sys.getwindowsversion() < (6, 1):
            return False
    except AttributeError:
        return False
    return os.path.isfile(get_pkg_file(
        "ncnn", "lve_worker", "lve-ncnn-worker.exe"))


@dataclass(frozen=True)
class NativeNcnnSpec:
    """Model and runtime options understood by the native worker."""

    gpu_id: int = -2
    rife_model: str = "none"
    rife_tta: bool = False
    rife_uhd: bool = False
    sr_kind: str = "none"
    sr_param: str = ""
    sr_model: str = ""
    sr_scale: int = 1
    sr_tta: bool = False
    sr_noise: int = -1
    sr_syncgap: int = 3
    sr_tile: int = 0

    @property
    def has_rife(self) -> bool:
        return self.rife_model != "none"

    @property
    def has_sr(self) -> bool:
        return self.sr_kind != "none"


def spec_from_engines(sr_engine, fi_engine,
                      gpu_id: Optional[int]) -> Optional[NativeNcnnSpec]:
    """Translate initialized legacy engine objects into one native job.

    Returning ``None`` means that the selected combination is not representable
    by the fused worker and must continue through the normal engine interfaces.
    """
    if sr_engine is None and fi_engine is None:
        return None
    try:
        fi_stage = (fi_engine.native_ncnn_stage()
                    if fi_engine is not None else None)
        sr_stage = (sr_engine.native_ncnn_stage()
                    if sr_engine is not None else None)
    except (AttributeError, NotImplementedError):
        return None
    if (fi_engine is not None and
            not isinstance(fi_stage, NcnnInterpolationStage)):
        return None
    if (sr_engine is not None and
            not isinstance(sr_stage, NcnnSuperResolutionStage)):
        return None

    values = {
        "gpu_id": -2 if gpu_id is None else int(gpu_id),
    }
    if fi_stage is not None:
        values.update(
            rife_model=fi_stage.model_dir,
            rife_tta=fi_stage.tta,
            rife_uhd=fi_stage.uhd,
        )
    if sr_stage is not None:
        values.update(
            sr_kind=sr_stage.kind,
            sr_param=sr_stage.param_path,
            sr_model=sr_stage.model_path,
            sr_scale=sr_stage.scale,
            sr_tta=sr_stage.tta,
            sr_noise=sr_stage.noise,
            sr_syncgap=sr_stage.syncgap,
            sr_tile=sr_stage.tile,
        )
    return NativeNcnnSpec(**values)


def _read_exact(pipe, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = pipe.read(length - len(data))
        if not chunk:
            raise EOFError("native NCNN worker pipe closed")
        data.extend(chunk)
    return bytes(data)


class _ReplyReader:
    def __init__(self, pipe):
        self._pipe = pipe
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="lve-native-ncnn-read", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                header = _read_exact(self._pipe, _REPLY.size)
                magic, version, status, count, size, elapsed = (
                    _REPLY.unpack(header))
                if magic != _MAGIC or version != _VERSION:
                    raise RuntimeError("native NCNN worker protocol mismatch")
                message = _read_exact(self._pipe, size).decode(
                    "utf-8", errors="replace") if size else ""
                self._queue.put((True, (status, count, message, elapsed)))
        except BaseException as exc:
            self._queue.put((False, exc))

    def read(self, timeout: float):
        try:
            ok, value = self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("native NCNN worker response timed out") from exc
        if not ok:
            raise EOFError(str(value)) from value
        return value

    def join(self, timeout: float = 1.0) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout)


class NativeNcnnEngine(FrameBatchExecutor):
    """One persistent Vulkan process for RIFE, SR, or the fused chain."""

    def __init__(self, spec: NativeNcnnSpec):
        self._spec = spec
        self._proc = None
        self._reader = None
        self._stderr_thread = None
        self._stderr_lines: List[str] = []
        self._shared_input = None
        self._shared_output = None
        self._src_w = self._src_h = self._dst_w = self._dst_h = 0
        self._multiplier = 1
        self._max_input_frames = 1
        self._max_output_frames = 1
        self._gpu_name = ""

    @property
    def name(self) -> str:
        stages = []
        if self._spec.has_rife:
            stages.append("RIFE ncnn")
        if self._spec.has_sr:
            stages.append(
                "Real-CUGAN" if self._spec.sr_kind == "realcugan"
                else ("ESRGAN" if self._spec.sr_kind == "esrgan"
                      else "Real-ESRGAN"))
        return "%s (%s, native Vulkan/shm)" % (
            " + ".join(stages), self._gpu_name or "GPU")

    @property
    def batch_size(self) -> int:
        return self._max_input_frames

    @staticmethod
    def output_count(input_count: int, multiplier: int,
                     skip_first: bool = False) -> int:
        if input_count <= 0:
            return 0
        return max(0, (input_count - 1) * multiplier +
                   (0 if skip_first else 1))

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int,
                   multiplier: int = 1) -> None:
        if not native_worker_available():
            raise RuntimeError("原生 NCNN Worker 不可用")
        if self._spec.gpu_id < -1:
            gpu_id = -2
        else:
            gpu_id = self._spec.gpu_id
        if gpu_id == -1:
            raise RuntimeError("原生 NCNN Worker 只支持 Vulkan GPU")
        self._src_w, self._src_h = int(src_width), int(src_height)
        self._dst_w, self._dst_h = int(dst_width), int(dst_height)
        self._multiplier = int(multiplier if self._spec.has_rife else 1)
        if self._spec.has_rife and self._multiplier < 2:
            raise ValueError("RIFE NCNN 插帧倍率至少为 2")

        input_bytes = self._src_w * self._src_h * 3
        output_bytes = self._dst_w * self._dst_h * 3
        input_budget = 96 * 1024 * 1024
        output_budget = 192 * 1024 * 1024
        affordable_inputs = max(1, input_budget // max(1, input_bytes))
        if self._spec.has_rife:
            affordable_outputs = max(
                self._multiplier + 1,
                output_budget // max(1, output_bytes))
            affordable_pairs = max(
                1, (affordable_outputs - 1) // self._multiplier)
            self._max_input_frames = min(
                5, affordable_inputs, affordable_pairs + 1)
            self._max_input_frames = max(2, self._max_input_frames)
        else:
            affordable_outputs = max(
                1, output_budget // max(1, output_bytes))
            self._max_input_frames = max(
                1, min(8, affordable_inputs, affordable_outputs))
        self._max_output_frames = self.output_count(
            self._max_input_frames, self._multiplier, False)
        self._shared_input = SharedNDArray.create(
            (self._max_input_frames, self._src_h, self._src_w, 3))
        self._shared_output = SharedNDArray.create(
            (self._max_output_frames, self._dst_h, self._dst_w, 3))

        command = [
            get_pkg_file("ncnn", "lve_worker", "lve-ncnn-worker.exe"),
            "--input-shm", self._shared_input.memory.name,
            "--output-shm", self._shared_output.memory.name,
            "--src-w", str(self._src_w),
            "--src-h", str(self._src_h),
            "--dst-w", str(self._dst_w),
            "--dst-h", str(self._dst_h),
            "--max-input", str(self._max_input_frames),
            "--max-output", str(self._max_output_frames),
            "--multiplier", str(self._multiplier),
            "--gpu", str(gpu_id),
            "--rife-model", self._spec.rife_model,
            "--rife-tta", "1" if self._spec.rife_tta else "0",
            "--rife-uhd", "1" if self._spec.rife_uhd else "0",
            "--sr-kind", self._spec.sr_kind,
        ]
        if self._spec.has_sr:
            command.extend([
                "--sr-param", self._spec.sr_param,
                "--sr-model", self._spec.sr_model,
                "--sr-scale", str(self._spec.sr_scale),
                "--sr-tta", "1" if self._spec.sr_tta else "0",
                "--sr-noise", str(self._spec.sr_noise),
                "--sr-syncgap", str(self._spec.sr_syncgap),
                "--sr-tile", str(self._spec.sr_tile),
            ])

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, creationflags=flags)
            self._reader = _ReplyReader(self._proc.stdout)
            self._stderr_thread = threading.Thread(
                target=self._read_stderr, args=(self._proc.stderr,),
                name="lve-native-ncnn-stderr", daemon=True)
            self._stderr_thread.start()
            status, _count, message, _elapsed = self._read(180.0)
            if status != 0:
                raise RuntimeError(message or "原生 NCNN Worker 初始化失败")
            self._gpu_name = message
        except Exception:
            details = self._stderr_text()
            self.release()
            if details:
                _log.debug("原生 NCNN Worker 错误输出:\n%s", details)
            raise
        _log.info("原生 NCNN Worker 就绪: %s, batch=%d",
                  self.name, self._max_input_frames)

    def process(self, frames: List[np.ndarray],
                skip_first: bool = False) -> List[np.ndarray]:
        if not self._proc or self._shared_input is None:
            raise RuntimeError("原生 NCNN Worker 尚未初始化")
        if not frames:
            return []
        if len(frames) > self._max_input_frames:
            raise ValueError("原生 NCNN 批次过大: %d > %d" % (
                len(frames), self._max_input_frames))
        for index, frame in enumerate(frames):
            expected = (self._src_h, self._src_w, 3)
            if frame.shape != expected:
                raise ValueError("原生 NCNN 输入尺寸不一致: %s != %s" % (
                    frame.shape, expected))
            np.copyto(self._shared_input.array[index], frame)

        pair_modes = bytes(classify_pair(left, right)
                           for left, right in zip(frames, frames[1:]))
        request = _REQUEST.pack(
            _MAGIC, _VERSION, _COMMAND_PROCESS, len(frames),
            1 if skip_first else 0)
        try:
            self._proc.stdin.write(request)
            self._proc.stdin.write(pair_modes)
            self._proc.stdin.flush()
        except (AttributeError, BrokenPipeError, OSError) as exc:
            raise RuntimeError("无法向原生 NCNN Worker 发送数据: %s\n%s" % (
                exc, self._stderr_text())) from exc

        timeout = max(
            120.0, len(frames) * self._multiplier *
            (90.0 if self._spec.has_sr else 45.0))
        status, count, message, elapsed = self._read(timeout)
        if status != 0:
            raise RuntimeError("原生 NCNN 推理失败: %s" % (
                message or status))
        expected_count = self.output_count(
            len(frames), self._multiplier, skip_first)
        if count != expected_count or count > self._max_output_frames:
            raise RuntimeError("原生 NCNN 返回帧数错误: %d，预期 %d" % (
                count, expected_count))
        _log.debug("原生 NCNN 批次: %d -> %d 帧, %.1f ms",
                   len(frames), count, elapsed)
        return [self._shared_output.array[index].copy()
                for index in range(count)]

    def _read(self, timeout: float):
        if self._reader is None:
            raise RuntimeError("原生 NCNN Worker 未运行")
        try:
            return self._reader.read(timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                "原生 NCNN Worker 在 %.0f 秒内没有响应" % timeout) from exc
        except EOFError as exc:
            raise RuntimeError("原生 NCNN Worker 已退出: %s\n%s" % (
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
                    process.stdin.write(_REQUEST.pack(
                        _MAGIC, _VERSION, _COMMAND_STOP, 0, 0))
                    process.stdin.flush()
                    process.stdin.close()
                process.wait(timeout=5)
            except Exception:
                process.kill()
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
        if self._reader is not None:
            self._reader.join()
            self._reader = None
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        self._stderr_thread = None
        for name in ("_shared_input", "_shared_output"):
            value = getattr(self, name, None)
            if value is not None:
                try:
                    value.close()
                except Exception:
                    pass
                setattr(self, name, None)
