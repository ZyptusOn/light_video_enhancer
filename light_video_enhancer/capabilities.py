"""Fast, side-effect-free hardware and engine capability discovery."""

import ctypes
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ._paths import model_file_exists, pkg_file_exists
from .encoding import CODEC_CHOICES, canonical_codec


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vendor: str
    device_id: str = ""


def _vendor_from_text(text: str) -> str:
    value = text.lower()
    if "10de" in value or "nvidia" in value:
        return "nvidia"
    if "8086" in value or "intel" in value:
        return "intel"
    if "1002" in value or "1022" in value or "amd" in value or "radeon" in value:
        return "amd"
    return "unknown"


def detect_gpus() -> List[GPUInfo]:
    """Use EnumDisplayDevices; unlike WMI this is fast and works on Windows 7."""
    if sys.platform != "win32":
        return []

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("DeviceName", ctypes.c_wchar * 32),
            ("DeviceString", ctypes.c_wchar * 128),
            ("StateFlags", ctypes.c_ulong),
            ("DeviceID", ctypes.c_wchar * 128),
            ("DeviceKey", ctypes.c_wchar * 128),
        ]

    result: List[GPUInfo] = []
    seen = set()
    enum_display = ctypes.windll.user32.EnumDisplayDevicesW
    for index in range(32):
        dev = DISPLAY_DEVICEW()
        dev.cb = ctypes.sizeof(dev)
        if not enum_display(None, index, ctypes.byref(dev), 0):
            break
        if not dev.DeviceString or dev.StateFlags & 0x00000008:
            continue
        key = (dev.DeviceString, dev.DeviceID)
        if key in seen:
            continue
        seen.add(key)
        result.append(GPUInfo(
            name=dev.DeviceString,
            vendor=_vendor_from_text(dev.DeviceID + " " + dev.DeviceString),
            device_id=dev.DeviceID,
        ))
    return result


def _has_current_module(name: str) -> bool:
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None and spec.loader is not None
    except (ImportError, AttributeError, ValueError):
        return False


def quick_capabilities() -> Dict[str, object]:
    """Return quickly; never imports Torch and never scans Python installs."""
    try:
        from .ffmpeg_bridge.worker import encoder_is_available, worker_is_loadable
        worker = worker_is_loadable()
        encoders = tuple(name for name in CODEC_CHOICES
                         if name != "auto" and encoder_is_available(name))
    except Exception:
        worker = False
        encoders = ()

    rife_model = model_file_exists("fi", "flownet.pkl")
    cugan_model = all(model_file_exists("ncnn", "realcugan", "models-se", name)
                      for name in ("up2x-conservative.param", "up2x-conservative.bin"))
    realesrgan_model = all(
        model_file_exists("ncnn", "realesrgan", "models", name)
        for name in ("realesr-animevideov3-x2.param",
                     "realesr-animevideov3-x2.bin",
                     "realesr-animevideov3-x3.param",
                     "realesr-animevideov3-x3.bin",
                     "realesr-animevideov3-x4.param",
                     "realesr-animevideov3-x4.bin",
                     "realesrgan-x4plus-anime.param",
                     "realesrgan-x4plus-anime.bin",
                     "realesrgan-x4plus.param", "realesrgan-x4plus.bin"))
    classic_esrgan_model = all(
        model_file_exists("ncnn", "realesrgan", "models", name)
        for name in ("esrgan-x4.param", "esrgan-x4.bin"))
    gpus = detect_gpus()
    vendors = {gpu.vendor for gpu in gpus}

    return {
        "worker": worker,
        "encoders": encoders,
        "vsr_dll": pkg_file_exists("bridge", "dxva_vsr_bridge.dll"),
        "gpus": gpus,
        "vendors": vendors,
        "torch_current": _has_current_module("torch"),
        "nvvfx_current": _has_current_module("nvvfx"),
        "rife_model": rife_model,
        "ncnn_rife": (
            pkg_file_exists("ncnn", "rife", "rife-ncnn-vulkan.exe")
            and all(model_file_exists("ncnn", "rife", "rife-v4.6", name)
                    for name in ("flownet.param", "flownet.bin"))
        ),
        "ncnn_cugan": (pkg_file_exists("ncnn", "realcugan", "realcugan-ncnn-vulkan.exe")
                        and cugan_model),
        "ncnn_esrgan": (pkg_file_exists("ncnn", "realesrgan", "realesrgan-ncnn-vulkan.exe")
                         and realesrgan_model),
        "ncnn_classic_esrgan": (
            pkg_file_exists("ncnn", "realesrgan", "realesrgan-ncnn-vulkan.exe")
            and classic_esrgan_model),
    }


def choose_codec(requested: str, gpus: Optional[List[GPUInfo]] = None) -> str:
    requested = canonical_codec(requested)
    if requested != "auto":
        return requested
    vendors = {gpu.vendor for gpu in (gpus if gpus is not None else detect_gpus())}
    if "nvidia" in vendors:
        return "h264_nvenc"
    if "amd" in vendors:
        return "h264_amf"
    # Media Foundation is available on Windows 7 and works across vendors. It
    # selects a hardware MFT when the installed driver exposes one and otherwise
    # uses a system encoder. The resilient encoder then tries libx264 and MPEG-4.
    return "h264_mf" if sys.platform == "win32" else "mpeg4"


def choose_engines(sr_engine: str, fi_engine: str) -> Tuple[str, str]:
    caps = quick_capabilities()
    vendors = caps["vendors"]
    if sr_engine == "auto":
        if caps["vsr_dll"] and ("nvidia" in vendors or "intel" in vendors):
            sr_engine = "dxva_vsr"
        elif caps["ncnn_cugan"]:
            sr_engine = "realcugan"
        else:
            sr_engine = "lanczos"
    if fi_engine == "auto":
        if caps["torch_current"] and caps["rife_model"]:
            fi_engine = "rife"
        else:
            try:
                import cv2
                fi_engine = "dis" if hasattr(cv2, "DISOpticalFlow_create") else "optical_flow"
            except ImportError:
                fi_engine = "blend"
    return sr_engine, fi_engine
