"""Optional SeedVR2 video-restoration adapter for Windows 10/11."""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional

import numpy as np

from .base import SuperResolutionEngine
from .._image_batch import read_frames, write_frames
from .._logging import get_logger
from .._paths import get_model_dir, get_pkg_file
from .._shared_frames import FramedPipeReader, close_process_pipes, write_framed

_log = get_logger(__name__)


class SeedVR2Engine(SuperResolutionEngine):
    """Persistent low-VRAM SeedVR2 process with temporal chunk context."""

    _MODEL_VARIANTS = {
        "3b": (
            "seedvr2-3b-fp8",
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "3B FP8", 8.0),
        "7b": (
            "seedvr2-7b-q4",
            "seedvr2_ema_7b-Q4_K_M.gguf",
            "7B Q4", 11.0),
        "7b-sharp": (
            "seedvr2-7b-sharp-q4",
            "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
            "7B Sharp Q4", 11.0),
    }
    _BATCHES = {
        "fast": 5,
        "balanced": 9,
        "quality": 13,
        "ultra": 21,
    }

    def __init__(self, device: str = "auto",
                 torch_python: Optional[str] = None,
                 quality: str = "balanced"):
        self._torch_python = torch_python
        self._quality = quality if quality in self._BATCHES else "balanced"
        self.preferred_batch_size = self._BATCHES[self._quality]
        self._src_w = self._src_h = self._dst_w = self._dst_h = 0
        self._proc = None
        self._reader = None
        self._stderr_thread = None
        self._stderr_lines: List[str] = []
        self._history: List[np.ndarray] = []
        self._gpu_name = ""

        self._model_label = "3B FP8"
        self._dit_model = ""
        self._model_family = "3b"
        self._min_vram_gib = 8.0

    @property
    def name(self) -> str:
        return "SeedVR2 %s (%s, %s, experimental restoration)" % (
            self._model_label, self._quality, self._gpu_name or "CUDA")

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def batch_output_pixels(self) -> int:
        return self._dst_w * self._dst_h * self.preferred_batch_size

    @property
    def batch_output_size(self):
        return self._dst_w, self._dst_h

    def _select_model(self):
        base_dir = get_model_dir("seedvr2-3b-fp8")
        vae_path = os.path.join(base_dir, "ema_vae_fp16.safetensors")
        if self._quality == "ultra":
            order = ("7b-sharp", "7b", "3b")
        elif self._quality == "quality":
            order = ("7b", "3b")
        else:
            order = ("3b",)
        for key in order:
            pack, filename, label, min_vram = self._MODEL_VARIANTS[key]
            model_path = os.path.join(get_model_dir(pack), filename)
            if os.path.isfile(model_path) and os.path.isfile(vae_path):
                self._model_label = label
                self._dit_model = os.path.abspath(model_path)
                self._model_family = "7b" if key.startswith("7b") else "3b"
                self._min_vram_gib = min_vram
                if self._model_family == "7b":
                    # Consumer GPUs need maximum block swapping. Larger temporal
                    # batches can exceed VRAM even when the Q4 weights fit.
                    self.preferred_batch_size = 5
                return base_dir
        required = [vae_path]
        preferred = self._MODEL_VARIANTS[order[0]]
        required.append(os.path.join(
            get_model_dir(preferred[0]), preferred[1]))
        raise FileNotFoundError(
            "SeedVR2 runtime/model files are missing: " +
            ", ".join(path for path in required if not os.path.isfile(path)))

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        if os.name != "nt" or sys.getwindowsversion() < (10, 0):
            raise RuntimeError("SeedVR2 is supported only on Windows 10/11")
        self._src_w, self._src_h = int(src_width), int(src_height)
        self._dst_w, self._dst_h = int(dst_width), int(dst_height)
        runtime = get_pkg_file("external", "seedvr2_runtime.zip")
        model_dir = self._select_model()
        if not os.path.isfile(runtime):
            raise FileNotFoundError("SeedVR2 runtime is missing: " + runtime)

        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        # SeedVR2 configures cudaMallocAsync during import. Our worker imports
        # torch first, so the allocator must be fixed before the child starts;
        # changing it after CUDA loads triggers a PyTorch internal assertion.
        child_env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        self._proc = subprocess.Popen(
            [self._torch_python or sys.executable, "-u",
             get_pkg_file("sr", "_seedvr2_infer.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._reader = FramedPipeReader(
            self._proc.stdout, "lve-seedvr2-reader")
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()
        write_framed(self._proc.stdin, {
            "runtime": runtime,
            "model_dir": model_dir,
            "dit_model": self._dit_model,
            "model_family": self._model_family,
            "min_vram_gib": self._min_vram_gib,
            "quality": self._quality,
            "dst_width": self._dst_w,
            "dst_height": self._dst_h,
        })
        try:
            reply = self._reader.read(timeout=180)
        except Exception as exc:
            error = self._stderr_text()
            self.release()
            raise RuntimeError(
                "SeedVR2 subprocess failed to start:\n%s" % error) from exc
        if not isinstance(reply, dict) or not reply.get("ready"):
            error = reply.get("error", "invalid startup reply") if isinstance(
                reply, dict) else "invalid startup reply"
            self.release()
            raise RuntimeError("SeedVR2 startup failed: %s" % error)
        self._gpu_name = str(reply.get("gpu_name", "CUDA"))
        _log.info(
            "SeedVR2 ready: %dx%d -> %dx%d (%s, %s, experimental)",
            src_width, src_height, dst_width, dst_height,
            self._model_label, self._quality)

    @staticmethod
    def _allowed_count_at_least(count: int) -> int:
        return max(5, ((int(count) - 1 + 3) // 4) * 4 + 1)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.process_batch([frame])[0]

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError(
                "SeedVR2 subprocess exited:\n%s" % self._stderr_text())

        prefix = self._history[-5:-1] if self._history else []
        logical = list(prefix) + list(frames)
        target_count = self._allowed_count_at_least(len(logical))
        padded = logical + [logical[-1]] * (target_count - len(logical))
        work = tempfile.mkdtemp(prefix="lve_seedvr2_")
        try:
            input_dir = os.path.join(work, "input")
            output_dir = os.path.join(work, "output")
            write_frames(padded, input_dir, "SeedVR2")
            write_framed(self._proc.stdin, {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "count": target_count,
            })
            try:
                reply = self._reader.read(timeout=7200)
            except Exception as exc:
                raise RuntimeError(
                    "SeedVR2 inference communication failed:\n%s" %
                    self._stderr_text()) from exc
            if not isinstance(reply, dict) or reply.get("count") != target_count:
                error = reply.get("error", "invalid reply") if isinstance(
                    reply, dict) else "invalid reply"
                detail = self._stderr_text()
                if detail:
                    error += "\n" + detail
                raise RuntimeError("SeedVR2 inference failed: %s" % error)
            outputs = read_frames(
                output_dir, target_count,
                (self._dst_w, self._dst_h), "SeedVR2")
            start = len(prefix)
            result = outputs[start:start + len(frames)]
        finally:
            shutil.rmtree(work, ignore_errors=True)

        timeline = list(frames) if not self._history else (
            self._history + list(frames[1:]))
        self._history = timeline[-5:]
        return result

    def _read_stderr(self, pipe) -> None:
        try:
            for line in pipe:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_lines.append(text)
        except Exception:
            pass

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-40:])

    def release(self) -> None:
        process, self._proc = self._proc, None
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=30)
            except Exception:
                process.kill()
                try:
                    process.wait(timeout=3)
                except Exception:
                    pass
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        close_process_pipes(process)
        self._history = []
