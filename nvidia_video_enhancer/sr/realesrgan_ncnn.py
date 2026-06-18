"""
Real-ESRGAN ncnn-vulkan 超分引擎。

通过调用 realesrgan-ncnn-vulkan.exe 实现。
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


def _find_realesrgan_exe() -> str:
    if is_frozen():
        base = os.path.join(get_pkg_dir(), "ncnn", "realesrgan")
    else:
        base = os.path.join(get_pkg_dir(), "ncnn", "realesrgan")
    exe = os.path.join(base, "realesrgan-ncnn-vulkan.exe")
    if os.path.exists(exe):
        return exe
    raise FileNotFoundError(
        "未找到 realesrgan-ncnn-vulkan.exe\n"
        f"搜索路径: {base}"
    )


class RealESRGANEngine(SuperResolutionEngine):
    """
    Real-ESRGAN ncnn 超分引擎。

    通用图像/视频超分，realesr-animevideov3 模型。
    模型: ncnn/realesrgan/models/
    """

    def __init__(self, device: str = "cuda"):
        self._src_w = 0
        self._src_h = 0
        self._dst_w = 0
        self._dst_h = 0
        self._exe = ""
        self._models_dir = ""
        self._gpu_id = 0
        self._scale = 4

    @property
    def name(self) -> str:
        return f"Real-ESRGAN ncnn ({self._scale}x)"

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        self._src_w = src_width
        self._src_h = src_height
        self._dst_w = dst_width
        self._dst_h = dst_height

        self._exe = _find_realesrgan_exe()

        exe_dir = os.path.dirname(self._exe)
        self._models_dir = os.path.join(exe_dir, "models")
        if not os.path.isdir(self._models_dir):
            raise FileNotFoundError(
                "Real-ESRGAN 模型目录不存在。\n"
                "请下载模型并放置到:\n"
                f"  {self._models_dir}\n"
                "下载地址:\n"
                "  https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases"
            )

        ratio = max(dst_width / src_width, dst_height / src_height)
        self._scale = max(2, min(4, int(ratio + 0.5)))

        _log.info("Real-ESRGAN 就绪 (%dx%d→%dx%d, %dx)",
                  src_width, src_height, dst_width, dst_height,
                  self._scale)

    def process(self, frame: np.ndarray) -> np.ndarray:
        tmpdir = tempfile.mkdtemp(prefix="esrgan_")
        try:
            pi = os.path.join(tmpdir, "in.png")
            po = os.path.join(tmpdir, "out.png")
            cv2.imwrite(pi, frame)

            exe_dir = os.path.dirname(self._exe)
            cmd = [
                self._exe, "-i", pi, "-o", po,
                "-s", str(self._scale),
                "-m", "models",
                "-g", str(self._gpu_id),
            ]
            result = subprocess.run(
                cmd, capture_output=True,
                cwd=exe_dir, timeout=120,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace") if result.stderr else ""
                raise RuntimeError(f"realesrgan-ncnn-vulkan 超分失败: {stderr}")

            out = cv2.imread(po)
            if out is None:
                raise RuntimeError(f"realesrgan-ncnn-vulkan 输出无法读取: {po}")

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
