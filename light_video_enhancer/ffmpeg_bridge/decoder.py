"""Decoder wrapper using the lossless packet-retention API when available."""

import ctypes
from typing import Optional

import numpy as np

from .worker import FFmpegVideoDecoder as _BaseDecoder
from .worker import _load_worker


class FFmpegVideoDecoder(_BaseDecoder):
    def read(self) -> Optional[np.ndarray]:
        if not self._handle:
            raise RuntimeError("解码器未打开")
        dll = _load_worker()
        read_frame = getattr(dll, "nve_decoder_read_frame2", dll.nve_decoder_read_frame)
        read_frame.argtypes = [ctypes.c_void_p]
        read_frame.restype = ctypes.POINTER(ctypes.c_uint8)
        ptr = read_frame(self._handle)
        if not ptr:
            return None
        size = self._width * self._height * 3
        return np.ctypeslib.as_array(ptr, shape=(size,)).reshape(
            self._height, self._width, 3
        ).copy()

    def close(self) -> None:
        if not self._handle:
            return
        dll = _load_worker()
        close_decoder = getattr(dll, "nve_decoder_close2", dll.nve_decoder_close)
        close_decoder.argtypes = [ctypes.c_void_p]
        close_decoder.restype = None
        close_decoder(self._handle)
        self._handle = None
