"""OSDEnhancer joint space-time video super-resolution adapter."""

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


class OSDEnhancerEngine(SuperResolutionEngine):
    """Isolated 4x spatial and 2x temporal one-step diffusion adapter."""

    preferred_batch_size = 5
    temporal_multiplier = 2
    _FILES = (
        "prompt_embeddings/empty.safetensors",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )

    def __init__(self, device: str = "auto",
                 torch_python: Optional[str] = None,
                 quality: str = "quality"):
        self._torch_python = torch_python
        self._src_w = self._src_h = self._dst_w = self._dst_h = 0
        self._proc = None
        self._reader = None
        self._stderr_thread = None
        self._stderr_lines: List[str] = []
        self._gpu_name = ""

    @property
    def name(self) -> str:
        return "OSDEnhancer joint 4x/2x (%s, experimental)" % (
            self._gpu_name or "CUDA")

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def batch_output_pixels(self) -> int:
        return self._dst_w * self._dst_h * 9

    @property
    def batch_output_size(self):
        return self._dst_w, self._dst_h

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        if os.name != "nt" or sys.getwindowsversion() < (10, 0):
            raise RuntimeError("OSDEnhancer is supported only on Windows 10/11")
        self._src_w, self._src_h = int(src_width), int(src_height)
        self._dst_w, self._dst_h = int(dst_width), int(dst_height)
        if (self._dst_w, self._dst_h) != (
                self._src_w * 4, self._src_h * 4):
            raise ValueError(
                "OSDEnhancer is a native joint 4x/2x model; select exactly 4x")

        runtime = get_pkg_file("external", "osdenhancer_runtime.zip")
        model_dir = get_model_dir("osdenhancer-v1")
        required = [runtime]
        required.extend(os.path.join(model_dir, name) for name in self._FILES)
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "OSDEnhancer runtime/model files are missing: " +
                ", ".join(missing))

        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        self._proc = subprocess.Popen(
            [self._torch_python or sys.executable, "-u",
             get_pkg_file("sr", "_osdenhancer_infer.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=child_env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._reader = FramedPipeReader(
            self._proc.stdout, "lve-osdenhancer-reader")
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()
        write_framed(self._proc.stdin, {
            "runtime": runtime,
            "checkpoint_path": model_dir,
        })
        try:
            reply = self._reader.read(timeout=900)
        except Exception as exc:
            error = self._stderr_text()
            self.release()
            raise RuntimeError(
                "OSDEnhancer subprocess failed to start:\n%s" % error) from exc
        if not isinstance(reply, dict) or not reply.get("ready"):
            error = reply.get("error", "invalid startup reply") if isinstance(
                reply, dict) else "invalid startup reply"
            detail = self._stderr_text()
            self.release()
            if detail:
                error += "\n" + detail
            raise RuntimeError("OSDEnhancer startup failed: %s" % error)
        self._gpu_name = str(reply.get("gpu_name", "CUDA"))
        _log.info(
            "OSDEnhancer ready: %dx%d@1x -> %dx%d@2x (%s, experimental)",
            src_width, src_height, dst_width, dst_height, self._gpu_name)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.process_batch([frame])[0]

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError(
                "OSDEnhancer subprocess exited:\n%s" % self._stderr_text())
        work = tempfile.mkdtemp(prefix="lve_osdenhancer_")
        try:
            input_dir = os.path.join(work, "input")
            output_dir = os.path.join(work, "output")
            write_frames(frames, input_dir, "OSDEnhancer")
            write_framed(self._proc.stdin, {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "input_count": len(frames),
            })
            expected = (len(frames) - 1) * 2 + 1
            try:
                reply = self._reader.read(timeout=7200)
            except Exception as exc:
                raise RuntimeError(
                    "OSDEnhancer inference communication failed:\n%s" %
                    self._stderr_text()) from exc
            if not isinstance(reply, dict) or reply.get("count") != expected:
                error = reply.get("error", "invalid reply") if isinstance(
                    reply, dict) else "invalid reply"
                detail = self._stderr_text()
                if detail:
                    error += "\n" + detail
                raise RuntimeError("OSDEnhancer inference failed: %s" % error)
            return read_frames(
                output_dir, expected,
                (self._dst_w, self._dst_h), "OSDEnhancer")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _read_stderr(self, pipe) -> None:
        try:
            for line in pipe:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_lines.append(text)
        except Exception:
            pass

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-50:])

    def release(self) -> None:
        process, self._proc = self._proc, None
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=10)
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

    def __del__(self):
        self.release()
