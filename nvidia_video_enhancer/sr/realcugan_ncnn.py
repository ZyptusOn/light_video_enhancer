"""
Real-CUGAN ncnn-vulkan 超分引擎。

优先使用 ncnn Python binding (pip install ncnn) 进行进程内推理，
如果 ncnn 不可用则回退到 realcugan-ncnn-vulkan.exe CLI 模式。

模型: models-se (UpSample 专用，2×/3×/4×)
  in0 → out0  (BGR, [0,1] float32)
"""

import os
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple
import numpy as np
import cv2

from .base import SuperResolutionEngine
from .._paths import get_pkg_dir, is_frozen
from .._logging import get_logger

_log = get_logger(__name__)


def _find_realcugan_exe() -> str:
    if is_frozen():
        base = os.path.join(get_pkg_dir(), "ncnn", "realcugan")
    else:
        base = os.path.join(get_pkg_dir(), "ncnn", "realcugan")
    exe = os.path.join(base, "realcugan-ncnn-vulkan.exe")
    if os.path.exists(exe):
        return exe
    raise FileNotFoundError(
        "未找到 realcugan-ncnn-vulkan.exe\n"
        f"搜索路径: {base}"
    )


# ========== ncnn Python API 模式 ==========

_NCNN_CUGAN_AVAILABLE = False
try:
    import ncnn as _ncnn_cugan
    _NCNN_CUGAN_AVAILABLE = True
except ImportError:
    pass


class _CUGANNet:
    """封装 Real-CUGAN ncnn Python API 推理。"""

    def __init__(self, param_path: str, bin_path: str, scale: int):
        self.net = _ncnn_cugan.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        self.scale = scale

    def process(self, frame: np.ndarray,
                dst_w: int, dst_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        f = frame.astype(np.float32) / 255.0

        mat_in = _ncnn_cugan.Mat(f.tobytes(), w, h, 3, 4 * w)

        ex = self.net.create_extractor()
        ex.input("in0", mat_in)

        ret, out = ex.extract("out0")
        if ret != 0:
            raise RuntimeError(f"ncnn extract out0 失败, ret={ret}")

        out_arr = np.array(out, copy=False)
        if out_arr.size == 0:
            raise RuntimeError("ncnn 推理输出为空")

        out_ch = out_arr.size // (dst_h * dst_w)
        if out_ch >= 3:
            nchw = out_arr.reshape(out_ch, dst_h, dst_w)
            rgb = nchw[:3].transpose(1, 2, 0)
        else:
            nchw = out_arr.reshape(dst_h, dst_w)
            raise RuntimeError(f"cugan 输出通道异常: shape={out_arr.shape}")

        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb


class RealCUGANEngine(SuperResolutionEngine):
    """
    Real-CUGAN ncnn 超分引擎。

    B 站出品视频超分模型，conservative 档位保留细节最佳。
    模型目录: ncnn/realcugan/models-se/

    优先使用 ncnn Python binding 进程内推理，
    不可用时回退到 CLI 子进程模式。
    """

    _SCALE_MODEL_MAP = {
        1: "up1x-conservative",
        2: "up2x-conservative",
        3: "up3x-conservative",
        4: "up4x-conservative",
    }

    def __init__(self, device: str = "cuda"):
        self._src_w = 0
        self._src_h = 0
        self._dst_w = 0
        self._dst_h = 0
        self._scale = 2
        self._model: Optional[_CUGANNet] = None
        self._model_name = ""

        self._exe = ""
        self._models_dir = ""
        self._gpu_id = 0
        self._tmpdir = ""
        self._pi = ""
        self._po = ""
        self._fallback_info = ""

    @property
    def name(self) -> str:
        if self._model is not None:
            return f"Real-CUGAN ncnn ({self._scale}x, conservative, Python API)"
        return f"Real-CUGAN ncnn ({self._scale}x, conservative, CLI fallback)"

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        self._src_w = src_width
        self._src_h = src_height
        self._dst_w = dst_width
        self._dst_h = dst_height

        self._exe = _find_realcugan_exe()
        exe_dir = os.path.dirname(self._exe)
        self._models_dir = os.path.join(exe_dir, "models-se")
        if not os.path.isdir(self._models_dir):
            raise FileNotFoundError(f"Real-CUGAN 模型目录不存在: {self._models_dir}")

        ratio = max(dst_width / src_width, dst_height / src_height)
        self._scale = max(2, min(4, int(ratio + 0.5)))

        self._model_name = self._SCALE_MODEL_MAP[self._scale]
        model_file = os.path.join(self._models_dir, f"{self._model_name}.param")
        if not os.path.isfile(model_file):
            _log.warning("%s 模型不存在，回退到 2x", self._model_name)
            self._scale = 2
            self._model_name = "up2x-conservative"

        if _NCNN_CUGAN_AVAILABLE:
            param = os.path.join(self._models_dir, f"{self._model_name}.param")
            bin_file = os.path.join(self._models_dir, f"{self._model_name}.bin")
            try:
                self._model = _CUGANNet(param, bin_file, self._scale)
                _log.info("Real-CUGAN ncnn 就绪 (%dx%d→%dx%d, %dx, Python API)",
                          src_width, src_height, dst_width, dst_height, self._scale)
                return
            except Exception as e:
                _log.warning("ncnn Python API 初始化失败: %s，回退到 CLI 模式", e)
                self._model = None
                self._fallback_info = str(e)

        self._init_cli_fallback(src_width, src_height, dst_width, dst_height)

    def _init_cli_fallback(self, src_width, src_height, dst_width, dst_height):
        self._tmpdir = tempfile.mkdtemp(prefix="cugan_")
        self._pi = os.path.join(self._tmpdir, "in.png")
        self._po = os.path.join(self._tmpdir, "out.png")
        _log.info("Real-CUGAN CLI 回退就绪 (%dx%d→%dx%d, %dx)",
                  src_width, src_height, dst_width, dst_height, self._scale)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._model is not None:
            return self._process_ncnn(frame)

        cv2.imwrite(self._pi, frame)
        exe_dir = os.path.dirname(self._exe)
        cmd = [
            self._exe, "-i", self._pi, "-o", self._po,
            "-s", str(self._scale),
            "-m", "models-se",
            "-g", str(self._gpu_id),
        ]
        result = subprocess.run(
            cmd, capture_output=True,
            cwd=exe_dir, timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace") if result.stderr else ""
            raise RuntimeError(f"realcugan-ncnn-vulkan 超分失败: {stderr}")

        out = cv2.imread(self._po)
        if out is None:
            raise RuntimeError(f"realcugan-ncnn-vulkan 输出无法读取: {self._po}")

        if out.shape[1] != self._dst_w or out.shape[0] != self._dst_h:
            out = cv2.resize(out, (self._dst_w, self._dst_h),
                             interpolation=cv2.INTER_LANCZOS4)
        return out

    def _process_ncnn(self, frame: np.ndarray) -> np.ndarray:
        out = self._model.process(frame, self._dst_w, self._dst_h)
        if out.shape[1] != self._dst_w or out.shape[0] != self._dst_h:
            out = cv2.resize(out, (self._dst_w, self._dst_h),
                             interpolation=cv2.INTER_LANCZOS4)
        return out

    def release(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = ""
