import os
import ctypes
import cv2
import numpy as np

from ._dxva_convert import bgr_to_nv12
from .base import SuperResolutionEngine
from .._paths import get_data_file


def _bgr_to_nv12(bgr: np.ndarray, align_w: int = 0, align_h: int = 0) -> np.ndarray:
    """BGR24 → NV12 (YUV444 cvt + 2×2 avg, 无布局歧义)

    align_w/align_h: 若 >0 则输出对齐到此尺寸（右侧/下侧补零），
    确保与 bridge DLL 期望的 alignSrcW×alignSrcH 一致。
    """
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

    y = yuv[:, :, 0].ravel()

    h2, w2 = h // 2, w // 2
    u = yuv[:, :, 1].reshape(h2, 2, w2, 2).mean(axis=(1, 3)).astype(np.uint8).ravel()
    v = yuv[:, :, 2].reshape(h2, 2, w2, 2).mean(axis=(1, 3)).astype(np.uint8).ravel()

    uv = np.empty(len(u) * 2, dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    nv12 = np.concatenate([y, uv])

    # 对齐到 alignSrcW×alignSrcH（bridge 期望的尺寸）
    if align_w > 0 and align_h > 0 and (align_w != w or align_h != h):
        aligned = np.zeros(align_w * align_h * 3 // 2, dtype=np.uint8)
        # Y 平面：numpy 切片批量拷贝（替代逐行 Python 循环）
        y_plane = y.reshape(h, w)
        aligned_y = aligned[:align_h * align_w].reshape(align_h, align_w)
        aligned_y[:h, :w] = y_plane
        # UV 平面：numpy 切片批量拷贝
        y_off = align_w * align_h
        uv_src = nv12[h * w:].reshape(h2, w)
        uv_dst = aligned[y_off:].reshape(h2, align_w)
        uv_dst[:h2, :w] = uv_src
        return aligned

    return nv12


class DXVA_VSR_Engine(SuperResolutionEngine):
    """
    通过 Direct3D 11 Video Processor API 调用 NVIDIA 驱动的 RTX Video Super Resolution。

    原理：
    NVIDIA RTX VSR 在驱动层拦截 D3D11 VideoProcessorBlt 调用，
    当检测到视频内容时自动应用 AI 超分。此引擎模拟媒体播放器的行为：
    1. 创建 D3D11Device + D3D11VideoDevice + D3D11VideoProcessor
    2. 将每帧作为 NV12 纹理送入 VideoProcessor
    3. 读回经过驱动增强（超分）后的帧

    前提条件：
    - NVIDIA RTX 30/40 系列显卡
    - NVIDIA 控制面板 → 调整视频图像设置 → 启用 RTX 视频增强
    - 编译了 dxva_vsr_bridge.dll (见 bridge/ 目录)
    """

    def __init__(self):
        self._dll = None
        self._handle = None
        self._src_width = 0
        self._src_height = 0
        self._dst_width = 0
        self._dst_height = 0
        self._initialized = False

    @property
    def name(self) -> str:
        return "NVIDIA RTX VSR (D3D11 Video Processor)"

    def _load_dll(self) -> ctypes.CDLL:
        if self._dll is not None:
            return self._dll
        dll_path = get_data_file("bridge", "dxva_vsr_bridge.dll")
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"未找到 {dll_path}。请在 bridge/ 目录运行 build.sh 编译。"
            )
        self._dll = ctypes.CDLL(dll_path)

        self._dll.dxva_vsr_create.argtypes = []
        self._dll.dxva_vsr_create.restype = ctypes.c_void_p

        self._dll.dxva_vsr_initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        self._dll.dxva_vsr_initialize.restype = ctypes.c_int

        self._dll.dxva_vsr_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self._dll.dxva_vsr_process.restype = ctypes.c_int

        self._dll.dxva_vsr_get_output.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
        ]
        self._dll.dxva_vsr_get_output.restype = ctypes.c_int

        self._dll.dxva_vsr_release.argtypes = [ctypes.c_void_p]
        self._dll.dxva_vsr_release.restype = None

        self._dll.dxva_vsr_get_output_size.argtypes = [ctypes.c_void_p]
        self._dll.dxva_vsr_get_output_size.restype = ctypes.c_int

        return self._dll

    def initialize(self, src_width: int, src_height: int,
                   dst_width: int, dst_height: int) -> None:
        if self._initialized:
            if (src_width != self._src_width or src_height != self._src_height or
                    dst_width != self._dst_width or dst_height != self._dst_height):
                raise ValueError(
                    f"DXVA VSR 引擎已初始化 ({self._src_width}x{self._src_height}"
                    f"→{self._dst_width}x{self._dst_height})，"
                    f"不支持重新指定尺寸 ({src_width}x{src_height}→{dst_width}x{dst_height})。"
                )
            return
        self._src_width = src_width
        self._src_height = src_height
        self._dst_width = dst_width
        self._dst_height = dst_height

        dll = self._load_dll()
        print(f"[dxva_vsr] 创建 D3D11 设备 + Video Processor ({src_width}x{src_height}→{dst_width}x{dst_height}) ...")
        self._handle = dll.dxva_vsr_create()
        result = dll.dxva_vsr_initialize(
            self._handle, src_width, src_height, dst_width, dst_height
        )
        if result != 0:
            err_map = {-1: "D3D11CreateDevice", -2: "CreateVideoProcessor", -3: "CreateTextures"}
            self.release()
            step = err_map.get(result, f"code={result}")
            raise RuntimeError(
                f"D3D11 Video Processor 初始化失败 ({step})。\n"
                "请确保:\n"
                "1. NVIDIA RTX 30/40/50 系列显卡\n"
                "2. NVIDIA 控制面板 → 视频 → 启用 RTX 视频增强\n"
                "3. 最新驱动 (Game Ready 或 Studio)\n"
                "如果 Bridge DLL 编译失败，用 --sr-engine nvvfx 代替。"
            )
        print(f"[dxva_vsr] D3D11 Video Processor 初始化完成")
        self._initialized = True

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")
        dll = self._dll

        h, w = frame.shape[:2]
        if w != self._src_width or h != self._src_height:
            raise ValueError(
                f"输入帧尺寸 {w}x{h} 与初始化尺寸 {self._src_width}x{self._src_height} 不匹配"
            )
        frame_bgr = frame if frame.shape[2] == 3 else frame[:, :, :3]

        align_w = ((self._src_width + 1) // 2) * 2
        align_h = ((self._src_height + 1) // 2) * 2
        nv12 = bgr_to_nv12(frame_bgr, align_w=align_w, align_h=align_h)
        nv12 = np.ascontiguousarray(nv12)

        in_ptr = nv12.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        result = dll.dxva_vsr_process(self._handle, in_ptr, w, h, 0)
        if result != 0:
            raise RuntimeError(f"VideoProcessor 处理失败 (错误码: {result})")

        out_size = dll.dxva_vsr_get_output_size(self._handle)
        out_buf = (ctypes.c_ubyte * out_size)()
        result = dll.dxva_vsr_get_output(self._handle, out_buf, out_size)
        if result != 0:
            raise RuntimeError(f"读取输出帧失败 (错误码: {result})")

        bgra = np.ctypeslib.as_array(out_buf).reshape(
            self._dst_height, self._dst_width, 4
        )
        return np.ascontiguousarray(bgra[:, :, :3])

    def release(self) -> None:
        if self._handle and self._dll:
            self._dll.dxva_vsr_release(self._handle)
        self._handle = None
        self._initialized = False
