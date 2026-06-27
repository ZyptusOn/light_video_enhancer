"""
RIFE ncnn-vulkan 插帧引擎。

优先使用 ncnn Python binding (pip install ncnn) 进行进程内推理，
如果 ncnn 不可用则回退到 rife-ncnn-vulkan.exe CLI 模式。

模型结构 (rife-v4.6):
  in0: frame0 BGR [0,1] float32  (padded HxWx3)
  in1: frame1 BGR [0,1] float32  (padded HxWx3)
  in2: timestep scalar
  out0 → interpolated frame BGR [0,1] float32 (padded HxWx3)
"""

import os
import subprocess
import tempfile
import shutil
from typing import List, Optional
import numpy as np
import cv2

from .base import FrameInterpolationEngine
from .._paths import get_pkg_dir, is_frozen
from .._logging import get_logger

_log = get_logger(__name__)

_MODULE = 32


def _find_rife_ncnn_exe() -> str:
    if is_frozen():
        base = os.path.join(get_pkg_dir(), "ncnn", "rife")
    else:
        base = os.path.join(get_pkg_dir(), "ncnn", "rife")
    exe = os.path.join(base, "rife-ncnn-vulkan.exe")
    if os.path.exists(exe):
        return exe
    raise FileNotFoundError(
        "未找到 rife-ncnn-vulkan.exe\n"
        f"请确保 ncnn/rife/rife-ncnn-vulkan.exe 存在\n"
        f"搜索路径: {base}"
    )


def _pad_size(w: int, h: int) -> tuple:
    pw = ((w - 1) // _MODULE + 1) * _MODULE
    ph = ((h - 1) // _MODULE + 1) * _MODULE
    return pw, ph


def _pad_frame(frame: np.ndarray, pw: int, ph: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == pw and h == ph:
        return frame
    padded = np.zeros((ph, pw, 3), dtype=frame.dtype)
    padded[:h, :w, :] = frame
    return padded


def _crop_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    return frame[:h, :w]


# ========== ncnn Python API 模式 ==========

_NCNN_AVAILABLE = False
try:
    import ncnn as _ncnn
    _NCNN_AVAILABLE = True
except ImportError:
    pass


class _RIFENcnnNet:
    """封装 ncnn Python API 推理。"""

    def __init__(self, param_path: str, bin_path: str):
        self.net = _ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        self.pw = 0
        self.ph = 0

    def forward(self, f0: np.ndarray, f1: np.ndarray,
                timestep: float) -> np.ndarray:
        h, w = f0.shape[:2]
        self.pw, self.ph = _pad_size(w, h)

        p0 = _pad_frame(f0, self.pw, self.ph)
        p1 = _pad_frame(f1, self.pw, self.ph)

        p0_f = p0.astype(np.float32) / 255.0
        p1_f = p1.astype(np.float32) / 255.0

        mat0 = _ncnn.Mat(p0_f.tobytes(), self.pw, self.ph, 3, 4 * self.pw)
        mat1 = _ncnn.Mat(p1_f.tobytes(), self.pw, self.ph, 3, 4 * self.pw)

        ts_data = np.full((self.ph, self.pw, 1), timestep, dtype=np.float32)
        mat_ts = _ncnn.Mat(ts_data.tobytes(), self.pw, self.ph, 1, 4 * self.pw)

        ex = self.net.create_extractor()
        ex.input("in0", mat0)
        ex.input("in1", mat1)
        ex.input("in2", mat_ts)

        ret, out = ex.extract("out0")
        if ret != 0:
            raise RuntimeError(f"ncnn extract out0 失败, ret={ret}")

        out_arr = np.array(out, copy=False)
        if out_arr.size == 0:
            raise RuntimeError("ncnn 推理输出为空")

        nchw = out_arr.reshape(3, self.ph, self.pw)
        chw = nchw.transpose(1, 2, 0)
        chw = np.clip(chw * 255.0, 0, 255).astype(np.uint8)

        return _crop_frame(chw, w, h)


class RIFENcnnEngine(FrameInterpolationEngine):
    """
    RIFE ncnn-vulkan 插帧引擎。

    优先使用 ncnn Python binding (进程内 Vulkan 推理, ~30ms/帧)，
    不可用时自动回退到 CLI 子进程模式 (~150ms/帧)。
    """

    def __init__(self, quality: str = "balanced"):
        self._width = 0
        self._height = 0
        self._multiplier = 2
        self._model: Optional[_RIFENcnnNet] = None
        self._use_ncnn = True

        self._exe = ""
        self._exe_dir = ""
        self._model_dir = ""
        self._model_rel = ""
        self._gpu_id = 0
        self._tmpdir = ""
        self._p0 = ""
        self._p1 = ""
        self._po = ""
        self._fallback_info = ""

    @property
    def name(self) -> str:
        if self._use_ncnn and _NCNN_AVAILABLE and self._model is not None:
            return "RIFE ncnn-vulkan (Python API, Vulkan)"
        return "RIFE ncnn-vulkan (CLI fallback)"

    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None:
        self._width = width
        self._height = height
        self._multiplier = multiplier

        model_dir = os.path.join(get_pkg_dir(), "ncnn", "rife")
        for candidate in ["rife-v4.6", "rife-v4", "rife-v4.0"]:
            test = os.path.join(model_dir, candidate)
            if os.path.isfile(os.path.join(test, "flownet.param")):
                self._model_dir = test
                break
        else:
            self._model_dir = model_dir
        self._model_rel = os.path.basename(self._model_dir)
        self._exe_dir = model_dir

        if _NCNN_AVAILABLE:
            param = os.path.join(self._model_dir, "flownet.param")
            bin_file = os.path.join(self._model_dir, "flownet.bin")
            try:
                self._model = _RIFENcnnNet(param, bin_file)
                _log.info("RIFE ncnn 就绪 (%dx%d, %dx, model=%s, Python API)",
                          width, height, multiplier, self._model_rel)
                return
            except Exception as e:
                _log.warning("ncnn Python API 初始化失败: %s，回退到 CLI 模式", e)
                self._use_ncnn = False
                self._fallback_info = str(e)
        else:
            self._use_ncnn = False
            self._fallback_info = "ncnn 包未安装 (pip install ncnn)"

        self._init_cli_fallback(width, height, multiplier)

    def _init_cli_fallback(self, width, height, multiplier):
        self._exe = _find_rife_ncnn_exe()
        self._tmpdir = tempfile.mkdtemp(prefix="rife_ncnn_")
        self._p0 = os.path.join(self._tmpdir, "f0.png")
        self._p1 = os.path.join(self._tmpdir, "f1.png")
        self._po = os.path.join(self._tmpdir, "out.png")
        _log.info("RIFE ncnn CLI 回退就绪 (%dx%d, %dx, model=%s)",
                  width, height, multiplier, self._model_rel)

    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]:
        n = self._multiplier - 1
        if n <= 0:
            return []

        if self._model is not None:
            return self._interpolate_ncnn(frame0, frame1, n)
        return self._interpolate_cli(frame0, frame1, n)

    def _interpolate_ncnn(self, f0, f1, n) -> List[np.ndarray]:
        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            out = self._model.forward(f0, f1, t)
            results.append(out)
        return results

    def _interpolate_cli(self, f0, f1, n) -> List[np.ndarray]:
        cv2.imwrite(self._p0, f0)
        cv2.imwrite(self._p1, f1)

        results = []
        for i in range(1, self._multiplier):
            t = i / self._multiplier
            cmd = [
                self._exe, "-0", self._p0, "-1", self._p1, "-o", self._po,
                "-s", str(t),
                "-m", self._model_rel,
                "-g", str(self._gpu_id),
            ]
            result = subprocess.run(
                cmd, capture_output=True,
                cwd=self._exe_dir, timeout=60,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace") if result.stderr else ""
                raise RuntimeError(
                    f"rife-ncnn-vulkan 推理失败 (timestep={t:.3f}): {stderr}"
                )

            out = cv2.imread(self._po)
            if out is None:
                raise RuntimeError(f"rife-ncnn-vulkan 输出无法读取: {self._po}")
            if out.shape[:2] != (self._height, self._width):
                out = cv2.resize(out, (self._width, self._height))
            results.append(out)

        return results

    def release(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = ""
