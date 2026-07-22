"""Resilient encoder wrapper with cross-vendor fallbacks."""

import ctypes
import os

from .._logging import get_logger
from ..encoding import canonical_codec, codec_candidates
from .worker import FFmpegVideoEncoder as _BaseEncoder
from .worker import _load_worker, encoder_is_available

_log = get_logger(__name__)


def _codec_preset(codec: str, requested: str) -> str:
    """Translate user-facing speed names to backend-specific values."""
    value = (requested or "balanced").lower()
    if codec.endswith("_mf"):
        return "medium"
    if codec.endswith("_amf"):
        return value if value in {"speed", "balanced", "quality"} else "balanced"
    if codec == "libaom-av1":
        if value.isdigit():
            return str(min(8, int(value)))
        return {
            "p1": "8", "p2": "7", "p3": "6", "p4": "5",
            "p5": "4", "p6": "2", "p7": "1",
            "ultrafast": "8", "superfast": "8", "veryfast": "7",
            "faster": "7", "fast": "6", "balanced": "5",
            "medium": "5", "slow": "3", "slower": "2",
            "veryslow": "1", "quality": "2",
        }.get(value, "5")
    if codec == "libsvtav1":
        if value.isdigit():
            return str(min(13, int(value)))
        return {
            "p1": "13", "p2": "11", "p3": "9", "p4": "8",
            "p5": "7", "p6": "5", "p7": "3",
            "ultrafast": "13", "superfast": "12", "veryfast": "11",
            "faster": "10", "fast": "9", "balanced": "8",
            "medium": "8", "slow": "6", "slower": "5",
            "veryslow": "4", "quality": "4",
        }.get(value, "8")
    if codec in {"libx264", "libx265"}:
        return {
            "p1": "ultrafast", "p2": "veryfast", "p3": "fast",
            "p4": "medium", "p5": "medium", "p6": "slow",
            "p7": "veryslow", "balanced": "medium", "quality": "slow",
        }.get(value, value)
    if "nvenc" not in codec:
        return value
    if value.startswith("p") and value[1:].isdigit():
        return value
    return {
        "ultrafast": "p1",
        "veryfast": "p2",
        "fast": "p3",
        "balanced": "p5",
        "medium": "p5",
        "slow": "p6",
        "quality": "p7",
    }.get(value, "p5")


class FFmpegVideoEncoder(_BaseEncoder):
    """Encoder that prefers the requested backend and falls back safely."""

    def __init__(self, *args, **kwargs):
        if len(args) < 5 and "codec" not in kwargs:
            kwargs["codec"] = "h264_mf"
        super().__init__(*args, **kwargs)

    def open(self) -> None:
        if self._handle:
            return

        requested = canonical_codec(self._codec)
        candidates = codec_candidates(requested)

        dll = _load_worker()
        source = (
            os.fsencode(os.path.abspath(self._source_path))
            if self._source_path else None
        )
        for codec in candidates:
            if not encoder_is_available(codec):
                continue
            preset = _codec_preset(codec, self._preset)
            handle = dll.nve_encoder_open(
                os.fsencode(os.path.abspath(self._output_path)),
                self._width,
                self._height,
                self._fps,
                codec.encode("ascii"),
                self._crf,
                preset.encode("ascii"),
                source,
            )
            if not handle:
                continue

            self._handle = handle
            self._codec = codec
            if codec != requested:
                _log.warning("编码器 %s 不可用，已回退到 %s", requested, codec)

            if hasattr(dll, "nve_encoder_prepare"):
                prepare = dll.nve_encoder_prepare
                prepare.argtypes = [ctypes.c_void_p]
                prepare.restype = None
                prepare(handle)

            if hasattr(dll, "nve_encoder_set_audio_range"):
                duration = (
                    self._audio_duration
                    if self._audio_duration is not None else -1.0
                )
                dll.nve_encoder_set_audio_range(
                    handle, self._audio_start, duration
                )
            return

        raise RuntimeError(
            "编码器均不可用（已尝试：%s）" % ", ".join(candidates)
        )

    def close(self) -> None:
        if not self._handle:
            return

        dll = _load_worker()
        if hasattr(dll, "nve_encoder_finish3"):
            finish = dll.nve_encoder_finish3
        elif hasattr(dll, "nve_encoder_finish2"):
            finish = dll.nve_encoder_finish2
        elif hasattr(dll, "nve_encoder_finish"):
            finish = dll.nve_encoder_finish
        else:
            finish = dll.nve_encoder_close
        finish.argtypes = [ctypes.c_void_p]
        finish.restype = None
        finish(self._handle)
        self._handle = None
