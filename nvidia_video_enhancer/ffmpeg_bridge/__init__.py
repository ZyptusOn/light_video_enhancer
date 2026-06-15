import os
import sys
from .._paths import get_bundle_dir, is_frozen

_initialized = False


def init_dll_paths():
    global _initialized
    if _initialized:
        return

    bundle = get_bundle_dir()
    dll_dirs = []
    for sub in ("ffmpeg_dlls", "bridge"):
        d = os.path.join(bundle, sub)
        if os.path.isdir(d):
            dll_dirs.append(d)

    if sys.platform == "win32":
        for d in dll_dirs:
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
        path_env = os.environ.get("PATH", "")
        for d in dll_dirs:
            if d not in path_env:
                path_env = d + os.pathsep + path_env
        os.environ["PATH"] = path_env

    _initialized = True


init_dll_paths()

from .worker import FFmpegVideoDecoder, FFmpegVideoEncoder

__all__ = ["FFmpegVideoDecoder", "FFmpegVideoEncoder"]
