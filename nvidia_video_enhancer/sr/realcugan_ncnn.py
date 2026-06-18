"""
Real-CUGAN ncnn-vulkan 超分引擎。

通过调用 realcugan-ncnn-vulkan.exe 实现。
模型: models-se (UpSample 专用，2×/3×/4×)
"""

import os
import subprocess
import tempfile
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


class RealCUGANEngine(SuperResolutionEngine):
    """
    Real-CUGAN ncnn 超分引擎。

    B 站出品视频超分模型，conservative 档位保留细节最佳。
    模型目录: ncnn/realcugan/models-se/
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
        self._exe = ""
        self._models_dir = ""
        self._gpu_id = 0
        self._scale = 2

    @property
    def name(self) -> str:
        return f"Real-CUGAN ncnn ({self._scale}x, conservative)"

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

        model_name = self._SCALE_MODEL_MAP[self._scale]
        model_file = os.path.join(self._models_dir, f"{model_name}.param")
        if not os.path.isfile(model_file):
            _log.warning("%s 模型不存在，回退到 2x", model_name)
            self._scale = 2
            model_name = "up2x-conservative"

        _log.info("Real-CUGAN 就绪 (%dx%d→%dx%d, %dx, models-se)",
                  src_width, src_height, dst_width, dst_height,
                  self._scale)

    def process(self, frame: np.ndarray) -> np.ndarray:
        tmpdir = tempfile.mkdtemp(prefix="cugan_")
        try:
            pi = os.path.join(tmpdir, "in.png")
            po = os.path.join(tmpdir, "out.png")
            cv2.imwrite(pi, frame)

            exe_dir = os.path.dirname(self._exe)
            cmd = [
                self._exe, "-i", pi, "-o", po,
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

            out = cv2.imread(po)
            if out is None:
                raise RuntimeError(f"realcugan-ncnn-vulkan 输出无法读取: {po}")

            if out.shape[1] != self._dst_w or out.shape[0] != self._dst_h:
                out = cv2.resize(out, (self._dst_w, self._dst_h),
                                 interpolation=cv2.INTER_LANCZOS4)
            return out
        finally:
            self._cleanup(tmpdir)

    def release(self) -> None:
        pass

    @staticmethod
    def _cleanup(tmpdir):
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
