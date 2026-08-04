"""SparkVSR Stage-2 keyframe-guided video super-resolution adapter."""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional, Sequence

import numpy as np

from .base import SuperResolutionEngine
from .._image_batch import read_frames, write_frames
from .._logging import get_logger
from .._paths import get_model_dir, get_pkg_file
from .._shared_frames import FramedPipeReader, close_process_pipes, write_framed

_log = get_logger(__name__)


def parse_reference_indices(value) -> List[int]:
    """Parse a comma/space separated index list without accepting ambiguity."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = list(value)
    try:
        indices = [int(item) for item in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SparkVSR reference indices must be non-negative integers") from exc
    if any(item < 0 for item in indices):
        raise ValueError("SparkVSR reference indices cannot be negative")
    if indices != sorted(set(indices)):
        raise ValueError(
            "SparkVSR reference indices must be unique and ascending")
    if any(b - a < 4 for a, b in zip(indices, indices[1:])):
        raise ValueError(
            "SparkVSR reference indices must be at least 4 frames apart")
    return indices


class SparkVSREngine(SuperResolutionEngine):
    """Persistent, isolated official SparkVSR Stage-2 adapter."""

    _FILES = (
        "model_index.json", "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model", "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
        "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
        "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    )
    _BATCH = {"fast": 9, "balanced": 17, "quality": 33, "ultra": 49}

    def __init__(self, device: str = "auto",
                 torch_python: Optional[str] = None,
                 quality: str = "quality",
                 reference_path: Optional[str] = None,
                 reference_indices: Optional[Sequence[int]] = None,
                 reference_guidance: float = 1.0):
        self._torch_python = torch_python
        self._quality = quality if quality in self._BATCH else "quality"
        self.preferred_batch_size = self._BATCH[self._quality]
        self._reference_path = os.path.abspath(reference_path) if reference_path else ""
        self._reference_indices = parse_reference_indices(reference_indices)
        self._reference_guidance = float(reference_guidance)
        self._src_w = self._src_h = self._dst_w = self._dst_h = 0
        self._proc = None
        self._reader = None
        self._stderr_thread = None
        self._stderr_lines: List[str] = []
        self._gpu_name = ""
        self._next_start = 0

    @property
    def name(self) -> str:
        mode = "keyframe" if self._reference_path else "no-ref"
        return "SparkVSR Stage-2 (%s, %s, manual)" % (
            mode, self._gpu_name or "CUDA")

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def batch_output_pixels(self) -> int:
        return self._dst_w * self._dst_h * self.preferred_batch_size

    @property
    def batch_output_size(self):
        return self._dst_w, self._dst_h

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        if os.name != "nt" or sys.getwindowsversion() < (10, 0):
            raise RuntimeError("SparkVSR is supported only on Windows 10/11")
        self._src_w, self._src_h = int(src_width), int(src_height)
        self._dst_w, self._dst_h = int(dst_width), int(dst_height)
        if (self._dst_w, self._dst_h) != (self._src_w * 4, self._src_h * 4):
            raise ValueError("SparkVSR is a native 4x model; select exactly 4x")
        if not 0.0 <= self._reference_guidance <= 4.0:
            raise ValueError("SparkVSR reference guidance must be between 0 and 4")
        if bool(self._reference_path) != bool(self._reference_indices):
            raise ValueError(
                "SparkVSR keyframe mode requires both a reference path and indices")
        if self._reference_path and not os.path.exists(self._reference_path):
            raise FileNotFoundError(
                "SparkVSR reference path does not exist: " + self._reference_path)

        runtime = get_pkg_file("external", "sparkvsr_runtime.zip")
        model_dir = get_model_dir("sparkvsr-stage2")
        required = [runtime]
        required.extend(os.path.join(model_dir, name) for name in self._FILES)
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "SparkVSR runtime/model files are missing: " + ", ".join(missing[:5]))

        child_env = os.environ.copy()
        child_env.update({
            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        })
        self._proc = subprocess.Popen(
            [self._torch_python or sys.executable, "-u",
             get_pkg_file("sr", "_sparkvsr_infer.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=child_env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._reader = FramedPipeReader(self._proc.stdout, "lve-sparkvsr-reader")
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()
        write_framed(self._proc.stdin, {
            "runtime": runtime, "model_path": model_dir,
            "quality": self._quality,
            "reference_path": self._reference_path,
            "reference_indices": self._reference_indices,
            "reference_guidance": self._reference_guidance,
            "dst_width": self._dst_w, "dst_height": self._dst_h,
        })
        try:
            reply = self._reader.read(timeout=1800)
        except Exception as exc:
            error = self._stderr_text()
            self.release()
            raise RuntimeError("SparkVSR subprocess failed to start:\n%s" % error) from exc
        if not isinstance(reply, dict) or not reply.get("ready"):
            error = reply.get("error", "invalid startup reply") if isinstance(reply, dict) else "invalid startup reply"
            detail = self._stderr_text()
            self.release()
            raise RuntimeError("SparkVSR startup failed: %s%s" % (
                error, "\n" + detail if detail else ""))
        self._gpu_name = str(reply.get("gpu_name", "CUDA"))
        _log.info(
            "SparkVSR ready: %dx%d -> %dx%d (%s, %s), batch=%d",
            src_width, src_height, dst_width, dst_height,
            "keyframe" if self._reference_path else "no-ref",
            self._gpu_name, self.preferred_batch_size)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.process_batch([frame])[0]

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("SparkVSR subprocess exited:\n%s" % self._stderr_text())
        work = tempfile.mkdtemp(prefix="lve_sparkvsr_")
        try:
            input_dir, output_dir = os.path.join(work, "input"), os.path.join(work, "output")
            write_frames(frames, input_dir, "SparkVSR")
            write_framed(self._proc.stdin, {
                "input_dir": input_dir, "output_dir": output_dir,
                "input_count": len(frames), "batch_start": self._next_start,
            })
            try:
                reply = self._reader.read(timeout=14400)
            except Exception as exc:
                raise RuntimeError(
                    "SparkVSR inference communication failed:\n%s" % self._stderr_text()) from exc
            if not isinstance(reply, dict) or reply.get("count") != len(frames):
                error = reply.get("error", "invalid reply") if isinstance(reply, dict) else "invalid reply"
                detail = self._stderr_text()
                raise RuntimeError("SparkVSR inference failed: %s%s" % (
                    error, "\n" + detail if detail else ""))
            self._next_start += max(0, len(frames) - 1)
            return read_frames(output_dir, len(frames),
                               (self._dst_w, self._dst_h), "SparkVSR")
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
        return "\n".join(self._stderr_lines[-80:])

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
