"""
FFmpeg C 包装器 — 通过 ffmpeg_worker.dll 调用 FFmpeg。

架构:
  Python (ctypes) → ffmpeg_worker.dll (C) → avcodec/avformat/swscale DLLs
"""

import ctypes
import os
import sys
from typing import Optional, Iterator, Tuple
import numpy as np

from .._paths import get_bundle_dir, get_pkg_dir, get_data_file, is_frozen

_worker_dll = None
_ffmpeg_path_patched = False


def _get_dll_search_dirs():
    dirs = []
    if is_frozen():
        bundle = get_bundle_dir()
        dirs.append(os.path.join(bundle, "ffmpeg_dlls"))
        dirs.append(os.path.join(bundle, "bridge"))
    else:
        pkg = get_pkg_dir()
        dirs.append(os.path.join(pkg, "ffmpeg_dlls"))
        dirs.append(os.path.join(pkg, "bridge"))
    return dirs


def _setup_paths():
    global _ffmpeg_path_patched
    if _ffmpeg_path_patched:
        return

    for d in _get_dll_search_dirs():
        if os.path.isdir(d):
            if sys.platform == "win32":
                if d not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(d)
                    except OSError:
                        pass

    _ffmpeg_path_patched = True


def _load_worker():
    global _worker_dll
    if _worker_dll is not None:
        return _worker_dll

    _setup_paths()

    dll_path = get_data_file("ffmpeg_bridge", "ffmpeg_worker.dll")
    if not os.path.exists(dll_path):
        raise FileNotFoundError(
            f"找不到 {dll_path}\n"
            "请在 MSYS2 UCRT64 终端中编译 Worker:\n"
            "  cd nvidia_video_enhancer/ffmpeg_bridge\n"
            "  ./build_worker.sh"
        )

    dll = ctypes.CDLL(dll_path)

    dll.nve_decoder_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.nve_decoder_open.restype = ctypes.c_void_p

    dll.nve_decoder_get_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
    ]
    dll.nve_decoder_get_info.restype = ctypes.c_int

    dll.nve_decoder_read_frame.argtypes = [ctypes.c_void_p]
    dll.nve_decoder_read_frame.restype = ctypes.POINTER(ctypes.c_uint8)

    dll.nve_decoder_close.argtypes = [ctypes.c_void_p]
    dll.nve_decoder_close.restype = None

    dll.nve_encoder_open.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
    ]
    dll.nve_encoder_open.restype = ctypes.c_void_p

    dll.nve_encoder_write_frame.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
    ]
    dll.nve_encoder_write_frame.restype = ctypes.c_int

    dll.nve_encoder_write_yuv.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
    ]
    dll.nve_encoder_write_yuv.restype = ctypes.c_int

    dll.nve_encoder_close.argtypes = [ctypes.c_void_p]
    dll.nve_encoder_close.restype = None

    _worker_dll = dll
    return dll


# ====== 公共 API ======

class FFmpegVideoDecoder:
    def __init__(self, input_path: str, use_nvdec: bool = True):
        self._input_path = input_path
        self._use_nvdec = use_nvdec
        self._handle: Optional[int] = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._total_frames = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def open(self) -> "FFmpegVideoDecoder":
        dll = _load_worker()
        handle = dll.nve_decoder_open(
            self._input_path.encode("utf-8"),
            1 if self._use_nvdec else 0,
        )
        if not handle:
            raise RuntimeError(f"无法打开视频文件: {self._input_path}")
        self._handle = handle

        w = ctypes.c_int()
        hi = ctypes.c_int()
        fps = ctypes.c_double()
        frames = ctypes.c_int64()
        dll.nve_decoder_get_info(handle, w, hi, fps, frames)
        self._width = w.value
        self._height = hi.value
        self._fps = fps.value
        self._total_frames = frames.value
        return self

    def read_frame(self) -> Optional[np.ndarray]:
        if self._handle is None:
            raise RuntimeError("解码器未打开")
        dll = _load_worker()
        ptr = dll.nve_decoder_read_frame(self._handle)
        if not ptr:
            return None
        size = self._width * self._height * 3
        buf = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * size)).contents
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(self._height, self._width, 3).copy()
        return arr

    def probe(self) -> dict:
        dll = _load_worker()
        handle = dll.nve_decoder_open(self._input_path.encode("utf-8"), 0)
        if not handle:
            return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}
        w = ctypes.c_int()
        hi = ctypes.c_int()
        fps = ctypes.c_double()
        frames = ctypes.c_int64()
        dll.nve_decoder_get_info(handle, w, hi, fps, frames)
        dll.nve_decoder_close(handle)
        return {
            "width": w.value, "height": hi.value,
            "fps": fps.value, "total_frames": frames.value,
        }

    def close(self):
        if self._handle is not None:
            _load_worker().nve_decoder_close(self._handle)
            self._handle = None

    def __iter__(self):
        while True:
            f = self.read_frame()
            if f is None:
                break
            yield f

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FFmpegVideoEncoder:
    def __init__(self, output_path: str, width: int, height: int,
                 fps: float, codec: str = "h264_nvenc", crf: int = 18,
                 preset: str = "p7", source_path: Optional[str] = None):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._crf = crf
        self._preset = preset
        self._source_path = source_path
        self._handle: Optional[int] = None

    def open(self) -> "FFmpegVideoEncoder":
        dll = _load_worker()
        src = self._source_path.encode("utf-8") if self._source_path else None
        h = dll.nve_encoder_open(
            self._output_path.encode("utf-8"),
            self._width, self._height, self._fps,
            self._codec.encode("utf-8"),
            self._crf,
            self._preset.encode("utf-8"),
            src,
        )
        if not h:
            raise RuntimeError(f"无法创建编码器: {self._codec}")
        self._handle = h
        return self

    def encode(self, frame: np.ndarray) -> bool:
        if self._handle is None:
            raise RuntimeError("编码器未打开")
        dll = _load_worker()
        h, w = frame.shape[:2]
        frame = frame.astype(np.uint8).copy()
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ret = dll.nve_encoder_write_frame(self._handle, ptr, w, h)
        return ret == 0

    def encode_yuv(self, yuv: np.ndarray, w: int, h: int) -> bool:
        if self._handle is None:
            raise RuntimeError("编码器未打开")
        dll = _load_worker()
        yuv = np.ascontiguousarray(yuv, dtype=np.uint8)
        ptr = yuv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ret = dll.nve_encoder_write_yuv(self._handle, ptr, w, h)
        return ret == 0

    def close(self):
        if self._handle is not None:
            _load_worker().nve_encoder_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
