import subprocess
from ._paths import data_file_exists


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
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  GPU           : {result.stdout.strip()}")
            return True
        return False
    except FileNotFoundError:
        return False


def check_torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_nvvfx() -> bool:
    try:
        from nvvfx import VideoSuperRes
        return True
    except ImportError:
        return False


def check_engine_availability():
    return {
        "worker":    check_ffmpeg_worker(),
        "vsr_dll":   check_vsr_bridge(),
        "torch_cuda": check_torch_cuda(),
        "nvvfx":     check_nvvfx(),
    }


def print_system_info():
    print("=" * 50)
    print("  Video Enhancer — 系统检测")
    print("=" * 50)

    caps = check_engine_availability()
    print(win_ok("FFmpeg Worker", caps["worker"]))
    print(win_ok("VSR Bridge DLL", caps["vsr_dll"]))
    print(win_ok("PyTorch CUDA", caps["torch_cuda"]))
    print(win_ok("nvidia-vfx SDK", caps["nvvfx"]))

    print("=" * 50)

    if not caps["worker"]:
        print("[提示] FFmpeg Worker DLL 不可用，无法处理视频")

    if not caps["torch_cuda"]:
        print("[提示] PyTorch CUDA 不可用，nvvfx/RIFE 引擎不可用")
    if not caps["nvvfx"]:
        print("[提示] pip install nvidia-vfx 获得 NVIDIA VFX SDK 超分")
    print()
