"""Fast, side-effect-free hardware and engine capability discovery."""

import ctypes
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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
    ema_vfi_model = model_file_exists(
        "fi", "ema_vfi", "ours_small_t.pkl")
    flashvsr_model = (
        pkg_file_exists("external", "flashvsr_runtime.zip")
        and all(model_file_exists("flashvsr-v1.1", name) for name in (
            "diffusion_pytorch_model_streaming_dmd.safetensors",
            "LQ_proj_in.ckpt", "TCDecoder.ckpt")))
    seedvr2_model = (
        pkg_file_exists("external", "seedvr2_runtime.zip")
        and all(model_file_exists("seedvr2-3b-fp8", name) for name in (
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "ema_vae_fp16.safetensors")))
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
    ifrnet_model = all(
        model_file_exists("ncnn", "ifrnet", variant, name)
        for variant in ("IFRNet_S_Vimeo90K", "IFRNet_Vimeo90K",
                        "IFRNet_L_Vimeo90K")
        for name in ("ifrnet.param", "ifrnet.bin"))
    span_model = all(
        model_file_exists("ncnn", "span", "spanx%d_ch%d.%s" %
                          (scale, channels, extension))
        for scale in (2, 4) for channels in (48, 52)
        for extension in ("param", "bin"))
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
        "ema_vfi_model": ema_vfi_model,
        "flashvsr_model": flashvsr_model,
        "seedvr2_model": seedvr2_model,
        "ncnn_rife": (
            pkg_file_exists("ncnn", "rife", "rife-ncnn-vulkan.exe")
            and all(model_file_exists("ncnn", "rife", "rife-v4.6", name)
                    for name in ("flownet.param", "flownet.bin"))
        ),
        "ncnn_ifrnet": (
            pkg_file_exists("ncnn", "lve_worker", "lve-ncnn-worker.exe")
            and ifrnet_model
        ),
        "ncnn_span": (
            pkg_file_exists("ncnn", "lve_worker", "lve-ncnn-worker.exe")
            and span_model
        ),
        "ncnn_cugan": (pkg_file_exists("ncnn", "realcugan", "realcugan-ncnn-vulkan.exe")
                        and cugan_model),
        "ncnn_esrgan": (pkg_file_exists("ncnn", "realesrgan", "realesrgan-ncnn-vulkan.exe")
                         and realesrgan_model),
        "ncnn_classic_esrgan": (
            pkg_file_exists("ncnn", "realesrgan", "realesrgan-ncnn-vulkan.exe")
            and classic_esrgan_model),
    }


@dataclass(frozen=True)
class EngineSelection:
    sr_engine: str
    fi_engine: str
    sr_score: int = 0
    fi_score: int = 0
    sr_reason_zh: str = ""
    sr_reason_en: str = ""
    fi_reason_zh: str = ""
    fi_reason_en: str = ""
    external_environment_verified: bool = False


def choose_codec(requested: str, gpus: Optional[List[GPUInfo]] = None,
                 available: Optional[Iterable[str]] = None) -> str:
    requested = canonical_codec(requested)
    if requested != "auto":
        return requested
    vendors = {gpu.vendor for gpu in (gpus if gpus is not None else detect_gpus())}
    if "nvidia" in vendors:
        preferred = "h264_nvenc"
    elif "amd" in vendors:
        preferred = "h264_amf"
    else:
        preferred = "h264_mf" if sys.platform == "win32" else "mpeg4"
    if available is None:
        return preferred

    usable = {canonical_codec(name) for name in available}
    candidates = [preferred]
    if sys.platform == "win32":
        candidates.append("h264_mf")
    candidates.extend(("libx264", "mpeg4"))
    for candidate in candidates:
        if candidate in usable:
            return candidate
    # Media Foundation is available on Windows 7 and works across vendors. It
    # selects a hardware MFT when the installed driver exposes one and otherwise
    # uses a system encoder. The resilient encoder then tries libx264 and MPEG-4.
    return "h264_mf" if sys.platform == "win32" else "mpeg4"


def _pick_engine(candidates):
    """Pick the highest score while preserving declaration order on ties."""
    return max(candidates, key=lambda item: item[1])


def select_engines(
        sr_engine: str, fi_engine: str, *,
        source_width: int = 0, source_height: int = 0,
        target_width: int = 0, target_height: int = 0,
        source_fps: float = 0.0, target_fps: Optional[float] = None,
        sr_quality: str = "quality", fi_quality: str = "balanced",
        fi_multiplier: int = 2, sr_first: bool = False,
        device: str = "auto", ncnn_gpu: Optional[int] = None,
        torch_python: Optional[str] = None,
        capabilities: Optional[Dict[str, object]] = None,
        python_environment: Optional[Dict[str, object]] = None,
        ) -> EngineSelection:
    """Select practical engines from fast checks and already-scanned state.

    The selector never scans Python installations and never imports PyTorch.
    Optional external features are trusted only when the selected executable
    has a fresh cache entry created by the explicit environment scan.
    """
    caps = capabilities if capabilities is not None else quick_capabilities()
    vendors = set(caps.get("vendors", ()))
    allow_vulkan = ncnn_gpu != -1

    environment = dict(python_environment or {})
    if not environment and torch_python:
        try:
            from ._env import get_cached_python_env
            environment = get_cached_python_env(torch_python) or {}
        except (ImportError, OSError, ValueError):
            environment = {}
    external_verified = bool(
        environment and environment.get("torch")
        and environment.get("cuda"))
    current_torch = bool(caps.get("torch_current")) and device != "cpu"
    torch_cuda = external_verified and device != "cpu"
    nvvfx_ready = (
        device != "cpu" and "nvidia" in vendors
        and ((torch_cuda and environment.get("nvvfx"))
             or (current_torch and caps.get("nvvfx_current"))))

    source_pixels = max(0, source_width) * max(0, source_height)
    target_pixels = max(0, target_width) * max(0, target_height)
    has_geometry = source_pixels > 0 and target_pixels > 0
    no_resize = (
        has_geometry and source_width == target_width
        and source_height == target_height)
    downscale = (
        has_geometry and
        (target_width < source_width or target_height < source_height))

    chosen_sr = sr_engine
    sr_score = 0
    sr_reason_zh = "用户明确指定"
    sr_reason_en = "explicitly selected"
    if sr_engine == "auto":
        sr_candidates = []
        if no_resize:
            sr_candidates.append((
                "none", 200, "目标尺寸与输入一致，跳过无效超分",
                "target size matches the input; redundant SR is skipped"))
        elif downscale:
            sr_candidates.append((
                "lanczos", 200, "目标尺寸更小，使用高质量缩放",
                "the target is smaller; use high-quality resampling"))
        else:
            sr_adjustments = {
                "fast": {"nvvfx": 0, "dxva_vsr": 22, "realesrgan": -8,
                         "span": 5, "realcugan": 0},
                "balanced": {"nvvfx": 7, "dxva_vsr": 14, "realesrgan": 4,
                             "span": 3, "realcugan": 1},
                "quality": {"nvvfx": 15, "dxva_vsr": 6, "realesrgan": 12,
                            "span": 4, "realcugan": 4},
                "ultra": {"nvvfx": 20, "dxva_vsr": 0, "realesrgan": 18,
                          "span": 8, "realcugan": 7},
            }
            quality_scores = sr_adjustments.get(
                sr_quality, sr_adjustments["quality"])
            if nvvfx_ready:
                fusion_candidate = (
                    fi_engine in {"auto", "rife"} and not sr_first
                    and caps.get("rife_model"))
                score = 94 + quality_scores["nvvfx"]
                if sr_quality != "fast" and fusion_candidate:
                    score += 10
                if target_pixels > 3840 * 2160:
                    score -= 15
                sr_candidates.append((
                    "nvvfx", score,
                    ("已验证 NVIDIA CUDA/NV-VFX 环境，可使用融合快速路径"
                     if fusion_candidate and torch_cuda else
                     "当前 Python 提供 PyTorch/NV-VFX，将尝试融合快速路径"
                     if fusion_candidate else
                     "NVIDIA Video Effects 可用，适合质量优先超分"),
                    ("verified NVIDIA CUDA/NV-VFX runtime enables the fused fast path"
                     if fusion_candidate and torch_cuda else
                     "the current Python provides PyTorch/NV-VFX; try the fused fast path"
                     if fusion_candidate else
                     "NVIDIA Video Effects is available for quality-oriented SR")))
            dxva_within_limit = (
                not has_geometry
                or (target_width <= 4096 and target_height <= 2160))
            if (caps.get("vsr_dll") and dxva_within_limit
                    and vendors.intersection({"nvidia", "intel"})):
                sr_candidates.append((
                    "dxva_vsr", 80 + quality_scores["dxva_vsr"],
                    "驱动 VSR 可用且目标尺寸在 4K 限制内",
                    "driver VSR is available and the target is within its 4K limit"))
            if allow_vulkan and caps.get("ncnn_esrgan"):
                score = 74 + quality_scores["realesrgan"]
                if target_pixels > 3840 * 2160:
                    score -= 12
                sr_candidates.append((
                    "realesrgan", score,
                    "便携 Vulkan 模型可用，且实测快于 SPAN",
                    "portable Vulkan models are available and benchmark faster than SPAN"))
            if allow_vulkan and caps.get("ncnn_span"):
                score = 52 + quality_scores["span"]
                if target_pixels > 3840 * 2160:
                    score -= 18
                sr_candidates.append((
                    "span", score,
                    "使用轻量 SPAN Vulkan 后备",
                    "use the lightweight SPAN Vulkan fallback"))
            if allow_vulkan and caps.get("ncnn_cugan"):
                sr_candidates.append((
                    "realcugan", 45 + quality_scores["realcugan"],
                    "使用可用的 Real-CUGAN Vulkan 后备",
                    "use the available Real-CUGAN Vulkan fallback"))
            sr_candidates.append((
                "lanczos", 18, "没有更合适的已验证 AI/驱动后端",
                "no more suitable verified AI or driver backend is available"))
        chosen_sr, sr_score, sr_reason_zh, sr_reason_en = _pick_engine(
            sr_candidates)

    chosen_fi = fi_engine
    fi_score = 0
    fi_reason_zh = "用户明确指定"
    fi_reason_en = "explicitly selected"
    needs_interpolation = fi_multiplier > 1 or bool(
        target_fps and source_fps and target_fps > source_fps * 1.01)
    if fi_engine == "auto":
        fi_candidates = []
        if not needs_interpolation:
            fi_candidates.append((
                "none", 200, "目标帧率不高于输入，跳过无效插帧",
                "target frame rate does not exceed the input; interpolation is skipped"))
        else:
            inference_pixels = (
                target_pixels if sr_first and chosen_sr != "none" else source_pixels)
            rife_score = 98 if torch_cuda else 80
            rife_score += {
                "fast": -14, "balanced": 0, "quality": 10, "ultra": 14,
            }.get(fi_quality, 0)
            if chosen_sr == "nvvfx" and not sr_first and nvvfx_ready:
                rife_score += 18
            if inference_pixels > 3840 * 2160:
                rife_score -= 48
            elif inference_pixels > 2560 * 1440:
                rife_score -= 30
            elif inference_pixels > 2500000:
                rife_score -= 18
            if sr_first and chosen_sr != "none":
                rife_score -= 5
            rife_score -= max(0, fi_multiplier - 2) * 2
            if (caps.get("rife_model") and device != "cpu"
                    and (torch_cuda or current_torch)):
                fi_candidates.append((
                    "rife", rife_score,
                    (("已验证 CUDA PyTorch；与 NV-VFX 共享融合路径"
                      if torch_cuda else
                      "当前 Python 提供 PyTorch/NV-VFX，将尝试融合路径")
                     if chosen_sr == "nvvfx" and not sr_first and nvvfx_ready
                     else
                     ("已验证 CUDA/PyTorch RIFE，优先保证插帧质量"
                      if torch_cuda else
                      "当前 Python 提供 PyTorch RIFE，优先保证插帧质量")),
                    (("verified CUDA PyTorch shares the fused NV-VFX path"
                      if torch_cuda else
                      "the current Python provides PyTorch/NV-VFX; try the fused path")
                     if chosen_sr == "nvvfx" and not sr_first and nvvfx_ready
                     else
                     ("verified CUDA/PyTorch RIFE is available for higher quality"
                      if torch_cuda else
                      "the current Python provides PyTorch RIFE for higher quality"))))
            if allow_vulkan and caps.get("ncnn_ifrnet"):
                score = 86 + {
                    "fast": 14, "balanced": 8, "quality": 2, "ultra": -5,
                }.get(fi_quality, 8)
                if inference_pixels > 3840 * 2160:
                    score -= 12
                elif inference_pixels > 2560 * 1440:
                    score -= 5
                fi_candidates.append((
                    "ifrnet_ncnn", score,
                    "IFRNet 常驻 Vulkan Worker 提供最佳实测吞吐",
                    "the persistent IFRNet Vulkan worker has the best measured throughput"))
            if allow_vulkan and caps.get("ncnn_rife"):
                score = 72 + {
                    "fast": 0, "balanced": 3, "quality": 8, "ultra": 10,
                }.get(fi_quality, 3)
                fi_candidates.append((
                    "rife_ncnn", score,
                    "使用便携 RIFE Vulkan 后备",
                    "use the portable RIFE Vulkan fallback"))
            if (torch_cuda and caps.get("ema_vfi_model")
                    and device != "cpu"):
                score = 68 + {
                    "fast": 2, "balanced": 5, "quality": 9, "ultra": 12,
                }.get(fi_quality, 5)
                fi_candidates.append((
                    "ema_vfi", score,
                    "使用已验证 CUDA 环境中的 EMA-VFI 后备",
                    "use EMA-VFI in the verified CUDA environment as a fallback"))
            try:
                import cv2
                if hasattr(cv2, "DISOpticalFlow_create"):
                    fi_candidates.append((
                        "dis", 35, "使用跨设备 DIS 光流后备",
                        "use the cross-device DIS optical-flow fallback"))
                else:
                    fi_candidates.append((
                        "optical_flow", 30, "使用跨设备 Farneback 光流后备",
                        "use the cross-device Farneback optical-flow fallback"))
            except ImportError:
                pass
            fi_candidates.append((
                "blend", 12, "没有可用的推理或光流后端",
                "no inference or optical-flow backend is available"))
        chosen_fi, fi_score, fi_reason_zh, fi_reason_en = _pick_engine(
            fi_candidates)

    return EngineSelection(
        chosen_sr, chosen_fi, sr_score, fi_score,
        sr_reason_zh, sr_reason_en, fi_reason_zh, fi_reason_en,
        external_verified)


def choose_engines(sr_engine: str, fi_engine: str, **context) -> Tuple[str, str]:
    """Backward-compatible tuple API for callers that do not need reasons."""
    selected = select_engines(sr_engine, fi_engine, **context)
    return selected.sr_engine, selected.fi_engine
