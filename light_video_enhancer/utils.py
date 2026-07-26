import os
import sys
from typing import Dict, Optional

from .capabilities import detect_gpus, quick_capabilities


def check_engine_availability(torch_python: Optional[str] = None,
                              deep: bool = False) -> Dict[str, object]:
    caps = dict(quick_capabilities())
    caps.update({"torch_cuda": False, "nvvfx": False, "torch_python": None})
    if deep:
        try:
            if torch_python:
                from ._env import check_python_env
                info = check_python_env(torch_python)
                caps["torch_cuda"] = bool(info.get("cuda"))
                caps["nvvfx"] = bool(info.get("nvvfx") and info.get("torch"))
                caps["torch_python"] = torch_python if caps["torch_cuda"] else None
            else:
                import torch
                caps["torch_cuda"] = bool(torch.cuda.is_available())
                try:
                    import nvvfx  # noqa: F401
                    caps["nvvfx"] = bool(caps["torch_cuda"])
                except (ImportError, OSError):
                    pass
        except (ImportError, OSError):
            pass
    return caps


def _safe_print(text: str = "") -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def print_system_info(torch_python: Optional[str] = None, deep: bool = False) -> None:
    caps = check_engine_availability(torch_python, deep=deep)
    _safe_print("=" * 58)
    _safe_print("  Light Video Enhancer - 系统能力")
    _safe_print("=" * 58)
    gpus = caps.get("gpus") or detect_gpus()
    if gpus:
        for gpu in gpus:
            _safe_print("  GPU: %s [%s]" % (gpu.name, gpu.vendor.upper()))
    else:
        _safe_print("  GPU: 未检测到活动显示设备")
    rows = [
        ("FFmpeg Worker", caps["worker"]),
        ("D3D11 VSR Bridge", caps["vsr_dll"]),
        ("RIFE PyTorch model", caps["rife_model"]),
        ("EMA-VFI Small model", caps["ema_vfi_model"]),
        ("RIFE ncnn-vulkan", caps["ncnn_rife"]),
        ("IFRNet ncnn-vulkan", caps["ncnn_ifrnet"]),
        ("SPAN ncnn-vulkan", caps["ncnn_span"]),
        ("Real-CUGAN ncnn", caps["ncnn_cugan"]),
        ("Real-ESRGAN ncnn", caps["ncnn_esrgan"]),
        ("ESRGAN classic ncnn", caps["ncnn_classic_esrgan"]),
    ]
    if deep:
        rows.extend([("PyTorch CUDA", caps["torch_cuda"]),
                     ("NVIDIA VFX package", caps["nvvfx"])])
    for label, available in rows:
        _safe_print("  %-24s: %s" % (label, "OK" if available else "--"))
    if not deep:
        _safe_print("  PyTorch: 未执行深度检测（按需检测）")
    _safe_print("=" * 58)
