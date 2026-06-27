"""
RIFE ncnn-vulkan 插帧引擎。

通过调用 rife-ncnn-vulkan.exe 实现。
依赖: GitHub Releases 预编译的 rife-ncnn-vulkan windows 包
需要: Vulkan 驱动（GPU 自带即可）

性能注意:
  持久 tmpdir + 固定文件名，避免每帧创建/删除临时目录的开销。
  subprocess 进程启动开销（~50ms/次）无法消除（ncnn CLI 为单次调用模式），
  如需更高性能请使用 PyTorch RIFE 子进程模式。
"""

import os
import subprocess
import tempfile
import shutil
from typing import List
import numpy as np
import cv2

from .base import FrameInterpolationEngine
from .._paths import get_pkg_dir, is_frozen
from .._logging import get_logger

_log = get_logger(__name__)


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


class RIFENcnnEngine(FrameInterpolationEngine):
    """
    RIFE ncnn-vulkan 插帧引擎 (零 PyTorch 依赖)。

    Vulkan 加速推理，每对帧通过 PNG 临时文件与 ncnn CLI 通信。
    临时目录在 initialize() 时创建，release() 时清理。
    """

    def __init__(self, quality: str = "balanced"):
        self._width = 0
        self._height = 0
        self._multiplier = 2
        self._exe = ""
        self._exe_dir = ""
        self._model_dir = ""
        self._model_rel = ""
        self._gpu_id = 0
        self._tmpdir = ""
        self._p0 = ""
        self._p1 = ""
        self._po = ""

    @property
    def name(self) -> str:
        return "RIFE ncnn-vulkan (Vulkan)"

    def initialize(self, width: int, height: int,
                   multiplier: int = 2) -> None:
        self._width = width
        self._height = height
        self._multiplier = multiplier

        self._exe = _find_rife_ncnn_exe()
        self._exe_dir = os.path.dirname(self._exe)

        for candidate in ["rife-v4.6", "rife-v4", "rife-v4.0"]:
            test = os.path.join(self._exe_dir, candidate)
            if os.path.isfile(os.path.join(test, "flownet.param")):
                self._model_dir = test
                break
        else:
            self._model_dir = self._exe_dir
        self._model_rel = os.path.basename(self._model_dir)

        self._tmpdir = tempfile.mkdtemp(prefix="rife_ncnn_")
        self._p0 = os.path.join(self._tmpdir, "f0.png")
        self._p1 = os.path.join(self._tmpdir, "f1.png")
        self._po = os.path.join(self._tmpdir, "out.png")

        _log.info("RIFE ncnn 就绪 (%dx%d, %dx, model=%s)",
                  width, height, multiplier, self._model_rel)

    def interpolate(self, frame0: np.ndarray,
                    frame1: np.ndarray) -> List[np.ndarray]:
        n = self._multiplier - 1
        if n <= 0:
            return []

        cv2.imwrite(self._p0, frame0)
        cv2.imwrite(self._p1, frame1)

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
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = ""
