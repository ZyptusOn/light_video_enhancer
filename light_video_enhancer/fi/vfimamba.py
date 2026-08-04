"""VFIMamba interpolation in an isolated CUDA PyTorch process."""
import os
import subprocess
import sys
import threading
from typing import List, Optional
import numpy as np
from ._scene_detect import PAIR_NORMAL, classify_pair, skipped_intermediates
from .base import FrameInterpolationEngine
from .._logging import get_logger
from .._paths import get_model_dir, get_pkg_file
from .._shared_frames import SharedNDArray, close_process_pipes, read_framed, write_framed

_log = get_logger(__name__)


class VFIMambaEngine(FrameInterpolationEngine):
    """Official VFIMamba S/full models with arbitrary-timestep reuse."""
    _QUALITY = {
        "fast": ("small", 0.5, False),
        "balanced": ("small", 0.0, False),
        "quality": ("full", 0.5, False),
        "ultra": ("full", 0.0, True),
    }

    def __init__(self, device: str = "auto", quality: str = "balanced",
                 torch_python: Optional[str] = None):
        self._requested_device = device
        self._quality = quality if quality in self._QUALITY else "balanced"
        self._torch_python = torch_python
        self._proc = None
        self._stderr_thread = None
        self._stderr_lines: List[str] = []
        self._shared_input = None
        self._shared_output = None
        self._width = self._height = 0
        self._multiplier = 2
        self._scan_backend = "unknown"

    @property
    def name(self) -> str:
        variant, flow_scale, tta = self._QUALITY[self._quality]
        detail = "S" if variant == "small" else "Full"
        if flow_scale:
            detail += ", flow %.2fx" % flow_scale
        if tta:
            detail += ", TTA"
        return "VFIMamba %s (%s, subprocess-shm)" % (detail, self._scan_backend)

    def initialize(self, width: int, height: int, multiplier: int = 2) -> None:
        if multiplier < 2:
            raise ValueError("VFIMamba interpolation multiplier must be at least 2")
        if self._requested_device == "cpu":
            raise RuntimeError("VFIMamba currently requires a CUDA GPU")
        self._width, self._height = int(width), int(height)
        self._multiplier = int(multiplier)
        variant, flow_scale, tta = self._QUALITY[self._quality]
        model_dir = get_model_dir("vfimamba")
        model_name = "VFIMamba_S.pkl" if variant == "small" else "VFIMamba.pkl"
        model_path = os.path.join(model_dir, model_name)
        runtime = get_pkg_file("external", "vfimamba_runtime.zip")
        if not os.path.isfile(runtime):
            raise FileNotFoundError("VFIMamba runtime is missing")
        if not os.path.isfile(model_path):
            raise FileNotFoundError("VFIMamba model is missing: vfimamba/%s" % model_name)
        try:
            self._shared_input = SharedNDArray.create((2, height, width, 3))
            self._shared_output = SharedNDArray.create((self._multiplier - 1, height, width, 3))
        except (OSError, RuntimeError, ValueError):
            self._release_shared()
            raise RuntimeError("VFIMamba shared memory could not be allocated")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        script = get_pkg_file("fi", "_vfimamba_infer.py")
        self._proc = subprocess.Popen(
            [self._torch_python or sys.executable, "-u", script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=flags)
        self._stderr_thread = threading.Thread(target=self._read_stderr,
                                                args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()
        write_framed(self._proc.stdin, {
            "runtime": runtime, "model_path": model_path, "variant": variant,
            "flow_scale": flow_scale, "tta": tta,
            "shared_input": self._shared_input.descriptor(),
            "shared_output": self._shared_output.descriptor(),
        })
        try:
            reply = read_framed(self._proc.stdout)
        except EOFError as exc:
            error = self._stderr_text()
            self.release()
            raise RuntimeError("VFIMamba subprocess failed to start:\n%s" % error) from exc
        if not isinstance(reply, dict) or not reply.get("ready"):
            error = reply.get("error", "invalid startup reply") if isinstance(reply, dict) else "invalid startup reply"
            self.release()
            raise RuntimeError("VFIMamba subprocess failed to start: %s" % error)
        self._scan_backend = str(reply.get("scan_backend", "unknown"))
        if self._scan_backend == "PyTorch reference":
            _log.warning("VFIMamba is using the safe PyTorch selective-scan fallback; install a compatible mamba_ssm CUDA extension for full speed")
        _log.info("VFIMamba ready: %dx%d, %dx, quality=%s, scan=%s",
                  width, height, multiplier, self._quality, self._scan_backend)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray) -> List[np.ndarray]:
        if frame0.shape != frame1.shape:
            raise ValueError("VFIMamba input frame dimensions do not match")
        pair_mode = classify_pair(frame0, frame1)
        if pair_mode != PAIR_NORMAL:
            return skipped_intermediates(frame0, frame1, self._multiplier, pair_mode)
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("VFIMamba subprocess exited:\n%s" % self._stderr_text())
        np.copyto(self._shared_input.array[0], frame0)
        np.copyto(self._shared_input.array[1], frame1)
        try:
            write_framed(self._proc.stdin, {"timesteps": [index / self._multiplier for index in range(1, self._multiplier)]})
            reply = read_framed(self._proc.stdout)
        except (EOFError, BrokenPipeError) as exc:
            raise RuntimeError("VFIMamba communication failed:\n%s" % self._stderr_text()) from exc
        if not isinstance(reply, dict) or "count" not in reply:
            error = reply.get("error", "invalid reply") if isinstance(reply, dict) else "invalid reply"
            raise RuntimeError("VFIMamba inference failed: %s" % error)
        return [self._shared_output.array[index].copy() for index in range(int(reply["count"]))]

    def _read_stderr(self, pipe) -> None:
        try:
            for line in pipe:
                value = line.decode("utf-8", errors="replace").rstrip()
                if value:
                    self._stderr_lines.append(value)
        except Exception:
            pass

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-20:])

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
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        close_process_pipes(process)
        self._release_shared()

    def _release_shared(self) -> None:
        for name in ("_shared_input", "_shared_output"):
            value = getattr(self, name, None)
            if value is not None:
                try:
                    value.close()
                except Exception:
                    pass
                setattr(self, name, None)
