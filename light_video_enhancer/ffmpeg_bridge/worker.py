"""ctypes wrapper for the bundled FFmpeg worker DLL."""

import ctypes
import os
import sys
from typing import Iterator, Optional

import numpy as np

from .._paths import get_data_file, get_pkg_dir, is_frozen

_worker_dll = None
_worker_error: Optional[str] = None
_dll_dir_handles = []


def _get_dll_search_dirs():
    package = get_pkg_dir()
    result = [os.path.join(package, "ffmpeg_dlls"), os.path.join(package, "bridge")]
    if is_frozen():
        bundle = getattr(sys, "_MEIPASS", "")
        result.extend([os.path.join(bundle, "ffmpeg_dlls"), os.path.join(bundle, "bridge")])
    return list(dict.fromkeys(result))


def _setup_paths() -> None:
    if sys.platform != "win32":
        return
    for directory in _get_dll_search_dirs():
        if not os.path.isdir(directory):
            continue
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if directory not in path_parts:
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                # The returned handle must stay alive or Windows removes the directory.
                _dll_dir_handles.append(os.add_dll_directory(directory))
            except OSError:
                pass


def _load_worker():
    global _worker_dll, _worker_error
    if _worker_dll is not None:
        return _worker_dll
    _setup_paths()
    dll_path = get_data_file("ffmpeg_bridge", "ffmpeg_worker.dll")
    if not os.path.isfile(dll_path):
        raise FileNotFoundError("缺少 FFmpeg Worker: %s" % dll_path)
    try:
        dll = ctypes.CDLL(dll_path)
    except OSError as exc:
        _worker_error = str(exc)
        raise OSError("FFmpeg Worker 或其依赖 DLL 无法加载: %s" % exc) from exc

    dll.nve_decoder_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.nve_decoder_open.restype = ctypes.c_void_p
    dll.nve_decoder_get_info.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
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
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
    ]
    dll.nve_encoder_write_frame.restype = ctypes.c_int
    dll.nve_encoder_write_yuv.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
    ]
    dll.nve_encoder_write_yuv.restype = ctypes.c_int
    dll.nve_encoder_close.argtypes = [ctypes.c_void_p]
    dll.nve_encoder_close.restype = None

    if hasattr(dll, "nve_encoder_is_available"):
        dll.nve_encoder_is_available.argtypes = [ctypes.c_char_p]
        dll.nve_encoder_is_available.restype = ctypes.c_int
    if hasattr(dll, "nve_encoder_set_audio_range"):
        dll.nve_encoder_set_audio_range.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        dll.nve_encoder_set_audio_range.restype = None

    _worker_dll = dll
    _worker_error = None
    return dll


def worker_is_loadable() -> bool:
    try:
        _load_worker()
        return True
    except (FileNotFoundError, OSError, AttributeError):
        return False


def encoder_is_available(codec: str) -> bool:
    try:
        dll = _load_worker()
        if hasattr(dll, "nve_encoder_is_available"):
            return bool(dll.nve_encoder_is_available(codec.encode("ascii")))
        return True
    except Exception:
        return False


def _hardware_code(value: str) -> int:
    value = (value or "auto").lower()
    if value in ("none", "cpu", "software"):
        return 0
    if value == "cuda":
        return 1
    if value in ("d3d11", "d3d11va"):
        return 2
    if value == "auto":
        try:
            from ..capabilities import detect_gpus
            vendors = {gpu.vendor for gpu in detect_gpus()}
            return 1 if "nvidia" in vendors else 2
        except Exception:
            return 0
    return 0


class FFmpegVideoDecoder:
    def __init__(self, input_path: str, use_nvdec: bool = True,
                 hardware: Optional[str] = None):
        self._input_path = input_path
        self._hardware = hardware or ("cuda" if use_nvdec else "cpu")
        self._handle = None
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

    def _read_info(self, handle) -> dict:
        dll = _load_worker()
        width, height = ctypes.c_int(), ctypes.c_int()
        fps, frames = ctypes.c_double(), ctypes.c_int64()
        if dll.nve_decoder_get_info(handle, ctypes.byref(width), ctypes.byref(height),
                                    ctypes.byref(fps), ctypes.byref(frames)) != 0:
            raise RuntimeError("FFmpeg Worker 无法读取视频信息")
        return {"width": width.value, "height": height.value,
                "fps": fps.value, "total_frames": frames.value}

    def open(self) -> None:
        if self._handle:
            return
        if not os.path.isfile(self._input_path):
            raise FileNotFoundError("输入文件不存在: %s" % self._input_path)
        dll = _load_worker()
        handle = dll.nve_decoder_open(
            os.fsencode(os.path.abspath(self._input_path)), _hardware_code(self._hardware))
        if not handle:
            raise RuntimeError("FFmpeg Worker 无法打开视频: %s" % self._input_path)
        self._handle = handle
        info = self._read_info(handle)
        self._width, self._height = info["width"], info["height"]
        self._fps, self._total_frames = info["fps"], info["total_frames"]

    def probe(self) -> dict:
        if not os.path.isfile(self._input_path):
            raise FileNotFoundError("输入文件不存在: %s" % self._input_path)
        dll = _load_worker()
        handle = dll.nve_decoder_open(os.fsencode(os.path.abspath(self._input_path)), 0)
        if not handle:
            raise RuntimeError("无法探测视频: %s" % self._input_path)
        try:
            return self._read_info(handle)
        finally:
            dll.nve_decoder_close(handle)

    def read(self) -> Optional[np.ndarray]:
        if not self._handle:
            raise RuntimeError("解码器未打开")
        ptr = _load_worker().nve_decoder_read_frame(self._handle)
        if not ptr:
            return None
        size = self._width * self._height * 3
        return np.ctypeslib.as_array(ptr, shape=(size,)).reshape(
            self._height, self._width, 3).copy()

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def close(self) -> None:
        if self._handle:
            _load_worker().nve_decoder_close(self._handle)
            self._handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_args):
        self.close()


class FFmpegVideoEncoder:
    def __init__(self, output_path: str, width: int, height: int, fps: float,
                 codec: str = "h264_mf", crf: int = 23, preset: str = "balanced",
                 source_path: Optional[str] = None, audio_start: float = 0.0,
                 audio_duration: Optional[float] = None):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._crf = crf
        self._preset = preset
        self._source_path = source_path
        self._audio_start = max(0.0, audio_start)
        self._audio_duration = audio_duration
        self._handle = None

    def open(self) -> None:
        if self._handle:
            return
        dll = _load_worker()
        source = os.fsencode(os.path.abspath(self._source_path)) if self._source_path else None
        handle = dll.nve_encoder_open(
            os.fsencode(os.path.abspath(self._output_path)), self._width, self._height,
            self._fps, self._codec.encode("ascii"), self._crf,
            self._preset.encode("ascii"), source)
        if not handle:
            raise RuntimeError("编码器 %s 初始化失败；请在 GUI 中选择可用编码器或使用 auto" % self._codec)
        self._handle = handle
        if hasattr(dll, "nve_encoder_set_audio_range"):
            duration = self._audio_duration if self._audio_duration is not None else -1.0
            dll.nve_encoder_set_audio_range(handle, self._audio_start, duration)

    def encode(self, frame: np.ndarray) -> None:
        if not self._handle:
            raise RuntimeError("编码器未打开")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if _load_worker().nve_encoder_write_frame(
                self._handle, ptr, frame.shape[1], frame.shape[0]) != 0:
            raise RuntimeError("视频帧编码失败")

    def encode_yuv(self, yuv: np.ndarray, width: int, height: int) -> None:
        if not self._handle:
            raise RuntimeError("编码器未打开")
        yuv = np.ascontiguousarray(yuv, dtype=np.uint8)
        ptr = yuv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if _load_worker().nve_encoder_write_yuv(self._handle, ptr, width, height) != 0:
            raise RuntimeError("YUV 视频帧编码失败")

    def close(self) -> None:
        if self._handle:
            _load_worker().nve_encoder_close(self._handle)
            self._handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_args):
        self.close()
