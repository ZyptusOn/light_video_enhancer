import subprocess
from typing import Optional, TypedDict
from ._paths import data_file_exists, pkg_file_exists
from ._logging import get_logger

_log = get_logger(__name__)


class EngineCapabilities(TypedDict):
    worker: bool
    vsr_dll: bool
    torch_cuda: bool
    nvvfx: bool
    ncnn_rife: bool
    ncnn_cugan: bool
    torch_python: Optional[str]


def win_ok(text: str, ok: bool):
    return f"  {text:20s}: {'✓' if ok else '✗'}"


def check_ffmpeg_worker() -> bool:
    return data_file_exists("ffmpeg_bridge", "ffmpeg_worker.dll")


def check_vsr_bridge() -> bool:
    return data_file_exists("bridge", "dxva_vsr_bridge.dll")


def check_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            _log.info("GPU: %s", result.stdout.strip())
            return True
        return False
    except FileNotFoundError:
        return False


def check_torch_cuda(torch_python: Optional[str] = None) -> bool:
    """
    检测 PyTorch + CUDA 是否可用。
    先尝试 in-process import，若失败则通过 torch_python 子进程检测。
    """
    # 1. in-process
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    # 2. 外部 Python
    if torch_python:
        try:
            from ._env import check_python_env
            info = check_python_env(torch_python, timeout=10)
            return info.get("cuda", False)
        except Exception:
            pass

    return False


def check_nvvfx(torch_python: Optional[str] = None) -> bool:
    try:
        from nvvfx import VideoSuperRes
        return True
    except ImportError:
        pass
    if torch_python:
        try:
            from ._env import check_python_env
            info = check_python_env(torch_python, timeout=5)
            return info.get("nvvfx", False) and info.get("torch", False)
        except Exception:
            pass
    return False


def check_engine_availability(torch_python: Optional[str] = None) -> EngineCapabilities:
    """
    检测所有引擎可用性。
    torch_python: 外部 Python 路径（含 torch+CUDA），用于 subprocess 模式。
    """
    has_torch_cuda = check_torch_cuda(torch_python)
    return {
        "worker":      check_ffmpeg_worker(),
        "vsr_dll":     check_vsr_bridge(),
        "torch_cuda":  has_torch_cuda,
        "nvvfx":       check_nvvfx(torch_python),
        "ncnn_rife":   pkg_file_exists("ncnn", "rife", "rife-ncnn-vulkan.exe"),
        "ncnn_cugan":  pkg_file_exists("ncnn", "realcugan", "realcugan-ncnn-vulkan.exe"),
        "torch_python": torch_python if has_torch_cuda else None,
    }


def print_system_info(torch_python: Optional[str] = None):
    print("=" * 50)
    print("  Video Enhancer — 系统检测")
    print("=" * 50)

    caps = check_engine_availability(torch_python)
    print(win_ok("FFmpeg Worker", caps["worker"]))
    print(win_ok("VSR Bridge DLL", caps["vsr_dll"]))
    print(win_ok("PyTorch CUDA", caps["torch_cuda"]))
    print(win_ok("nvidia-vfx SDK", caps["nvvfx"]))
    print(win_ok("ncnn RIFE", caps["ncnn_rife"]))
    print(win_ok("ncnn Real-CUGAN", caps["ncnn_cugan"]))
    if caps["torch_python"]:
        print(f"  {'torch Python':20s}: {caps['torch_python']}")

    print("=" * 50)

    if not caps["worker"]:
        print("[提示] FFmpeg Worker DLL 不可用，无法处理视频")

    if not caps["torch_cuda"] and not caps["ncnn_rife"]:
        print("[提示] 无可用插帧引擎 (需要 PyTorch CUDA 或 ncnn RIFE)")
    if not caps["nvvfx"]:
        print("[提示] pip install nvidia-vfx 获得 NVIDIA VFX SDK 超分")
    print()
