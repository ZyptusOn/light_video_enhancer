"""RIFE v4.25 PyTorch interpolation (in-process or persistent subprocess)."""

import os
import subprocess
import threading
from typing import List, Optional

import numpy as np

from .base import FrameInterpolationEngine
from ._scene_detect import PAIR_NORMAL, classify_pair, skipped_intermediates
from .._logging import get_logger
from .._paths import get_model_file, get_pkg_file
from .._shared_frames import SharedNDArray, read_framed, write_framed

_log = get_logger(__name__)

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    F = None
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from ._rife_model import FlownetCas


def _find_weight_file() -> Optional[str]:
    names = ["flownet.pkl", "rife_v4.25.pth", "rife_v4.26.pth"]
    for name in names:
        path = get_model_file("fi", name)
        if os.path.isfile(path):
            return path
    for name in names:
        path = os.path.join(os.getcwd(), name)
        if os.path.isfile(path):
            return path
    return None


def _pickle_write(pipe, obj) -> None:
    write_framed(pipe, obj)


def _pickle_read(pipe):
    return read_framed(pipe)


def _pack_array(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    return {"__lve_array__": 1, "shape": tuple(int(part) for part in array.shape),
            "dtype": array.dtype.str, "data": array.tobytes(order="C")}


def _unpack_array(value) -> np.ndarray:
    if not isinstance(value, dict) or value.get("__lve_array__") != 1:
        raise TypeError("无效的 RIFE 数组消息")
    shape = tuple(int(part) for part in value["shape"])
    result = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
    expected = int(np.prod(shape, dtype=np.int64))
    if result.size != expected:
        raise ValueError("RIFE 数组消息长度不匹配")
    return result.reshape(shape).copy()


class RIFEEngine(FrameInterpolationEngine):
    def __init__(self, device: str = "auto", torch_python: Optional[str] = None):
        self._requested_device = device
        self._torch_python = torch_python
        self._use_subprocess = False
        self._subproc = None
        self._stderr_thread = None
        self._stderr_lines: List[str] = []
        self._model = None
        self._device = None
        self._fp16 = False
        self._scale = 1.0
        self._multiplier = 2
        self._width = self._height = 0
        self._pad_w = self._pad_h = 0
        self._model_path: Optional[str] = None
        self._shared_input = None
        self._shared_output = None

    @property
    def name(self) -> str:
        precision = "FP16" if self._fp16 else "FP32"
        if self._shared_input is not None:
            mode = "subprocess-shm"
        else:
            mode = "subprocess" if self._use_subprocess else "in-process"
        return "RIFE v4.25 (%s, %s)" % (precision, mode)

    def initialize(self, src_width: int, src_height: int, multiplier: int = 2) -> None:
        if multiplier < 2:
            raise ValueError("RIFE 插帧倍率至少为 2")
        self._width, self._height = src_width, src_height
        self._multiplier = multiplier
        area = src_width * src_height
        self._scale = 0.25 if area > 3840 * 2160 else (0.5 if area > 1920 * 1080 * 2 else 1.0)
        alignment = max(128, int(128 / self._scale))
        self._pad_w = ((src_width + alignment - 1) // alignment) * alignment - src_width
        self._pad_h = ((src_height + alignment - 1) // alignment) * alignment - src_height
        self._model_path = _find_weight_file()
        if not self._model_path:
            raise FileNotFoundError("缺少 RIFE 权重: light_video_enhancer/fi/flownet.pkl")

        current_cuda = bool(_TORCH_AVAILABLE and torch.cuda.is_available())
        allow_cpu = self._requested_device == "cpu"
        if _TORCH_AVAILABLE and (current_cuda or allow_cpu):
            self._init_inprocess(use_cuda=current_cuda and not allow_cpu)
        elif self._torch_python:
            self._init_subprocess()
        else:
            raise RuntimeError("RIFE 需要 CUDA PyTorch；也可选择 RIFE ncnn-vulkan")
        _log.info("RIFE 就绪: %dx%d, %dx, scale=%.2f", src_width, src_height,
                  multiplier, self._scale)

    def _init_inprocess(self, use_cuda: bool) -> None:
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._fp16 = use_cuda
        torch.set_grad_enabled(False)
        if use_cuda:
            # Avoid a multi-second algorithm search before the first frame.
            torch.backends.cudnn.benchmark = False
        self._model = FlownetCas().to(self._device).eval()
        state = torch.load(self._model_path, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if any(key.startswith("module.") for key in state):
            state = {key.replace("module.", "", 1): value for key, value in state.items()}
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing:
            _log.warning("RIFE 权重缺少 %d 个键", len(missing))
        if unexpected:
            _log.warning("RIFE 权重含 %d 个未使用键", len(unexpected))
        if self._fp16:
            self._model.half()

    def _read_stderr(self, pipe) -> None:
        try:
            for line in pipe:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_lines.append(text)
        except Exception:
            pass

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-20:])

    def _init_subprocess(self) -> None:
        script = get_pkg_file("fi", "_rife_infer.py")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        shared_args = {}
        try:
            self._shared_input = SharedNDArray.create((2, self._height, self._width, 3))
            self._shared_output = SharedNDArray.create(
                (self._multiplier - 1, self._height, self._width, 3))
            shared_args = {
                "ipc": "shared_v1",
                "shared_input": self._shared_input.descriptor(),
                "shared_output": self._shared_output.descriptor(),
            }
        except (OSError, RuntimeError, ValueError):
            self._release_shared()
            _log.warning("RIFE 共享内存不可用，回退到管道传输", exc_info=True)

        self._subproc = subprocess.Popen(
            [self._torch_python, "-u", script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, args=(self._subproc.stderr,), daemon=True)
        self._stderr_thread.start()
        arguments = {"model_path": self._model_path, "fp16": True}
        arguments.update(shared_args)
        _pickle_write(self._subproc.stdin, arguments)
        _pickle_write(self._subproc.stdin, [])
        try:
            reply = _pickle_read(self._subproc.stdout)
        except EOFError as exc:
            self.release()
            raise RuntimeError("RIFE 子进程启动失败\n%s" % self._stderr_text()) from exc
        if isinstance(reply, dict) and "error" in reply:
            error = reply["error"]
            self.release()
            raise RuntimeError("RIFE 子进程启动失败: %s" % error)
        self._use_subprocess = True
        self._fp16 = True

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray) -> List[np.ndarray]:
        if frame0.shape != frame1.shape:
            raise ValueError("RIFE 输入帧尺寸不一致")
        pair_mode = classify_pair(frame0, frame1)
        if pair_mode != PAIR_NORMAL:
            return skipped_intermediates(
                frame0, frame1, self._multiplier, pair_mode)
        if self._use_subprocess:
            return self._interpolate_subprocess(frame0, frame1)
        return self._interpolate_inprocess(frame0, frame1)

    def _to_tensor(self, bgr: np.ndarray):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        tensor = torch.from_numpy(rgb).to(self._device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.half().div_(255.0) if self._fp16 else tensor.float().div_(255.0)

    def _interpolate_inprocess(self, frame0, frame1) -> List[np.ndarray]:
        i0, i1 = self._to_tensor(frame0), self._to_tensor(frame1)
        i0 = F.pad(i0, (0, self._pad_w, 0, self._pad_h))
        i1 = F.pad(i1, (0, self._pad_w, 0, self._pad_h))
        output = []
        with torch.inference_mode():
            for index in range(1, self._multiplier):
                pred = self._model.inference(i0, i1, index / self._multiplier, self._scale)
                pred = pred[0, :, :self._height, :self._width]
                rgb = pred.float().permute(1, 2, 0).clamp_(0, 1).mul_(255).byte().cpu().numpy()
                output.append(np.ascontiguousarray(rgb[:, :, ::-1]))
        return output

    def _interpolate_subprocess(self, frame0, frame1) -> List[np.ndarray]:
        if self._subproc is None or self._subproc.poll() is not None:
            raise RuntimeError("RIFE 子进程已退出\n%s" % self._stderr_text())
        if self._shared_input is not None and self._shared_output is not None:
            np.copyto(self._shared_input.array[0], frame0)
            np.copyto(self._shared_input.array[1], frame1)
            request = {
                "protocol": 3,
                "timesteps": [i / self._multiplier for i in range(1, self._multiplier)],
                "pad_w": self._pad_w,
                "pad_h": self._pad_h,
                "scale": self._scale,
            }
        else:
            request = {
                "protocol": 2,
                "frame0": _pack_array(frame0),
                "frame1": _pack_array(frame1),
                "timesteps": [i / self._multiplier for i in range(1, self._multiplier)],
                "pad_w": self._pad_w,
                "pad_h": self._pad_h,
                "scale": self._scale,
            }
        try:
            _pickle_write(self._subproc.stdin, request)
            result = _pickle_read(self._subproc.stdout)
        except (EOFError, BrokenPipeError) as exc:
            raise RuntimeError("RIFE 子进程通信失败:\n%s" % self._stderr_text()) from exc
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError("RIFE 推理失败: %s" % result["error"])
        if self._shared_output is not None and isinstance(result, dict) and result.get("shared"):
            count = int(result.get("count", 0))
            return [self._shared_output.array[index].copy() for index in range(count)]
        try:
            return [_unpack_array(frame) for frame in result]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("RIFE 子进程返回了无效数据: %s" % exc) from exc

    def release(self) -> None:
        process, self._subproc = self._subproc, None
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
        self._release_shared()
        self._model = None
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_shared(self) -> None:
        for name in ("_shared_input", "_shared_output"):
            value = getattr(self, name, None)
            if value is not None:
                try:
                    value.close()
                except Exception:
                    pass
                setattr(self, name, None)
