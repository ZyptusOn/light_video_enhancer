#!/usr/bin/env python3
"""build_exe.py — 打包 nvidia_video_enhancer 为单个 .exe

用法:
    python nvidia_video_enhancer/build_exe.py

生成:
    视频增强.exe  (约 60-120 MB, 不含 PyTorch)
"""

import os
import sys
import subprocess
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(PROJECT_DIR, "nvidia_video_enhancer")
LAUNCHER = os.path.join(PROJECT_DIR, "launcher.py")
EXE_NAME = "视频增强"
OUTPUT_DIR = PROJECT_DIR

DATA_FILES = [
    ("nvidia_video_enhancer/ffmpeg_dlls", "ffmpeg_dlls"),
    ("nvidia_video_enhancer/ffmpeg_bridge/ffmpeg_worker.dll", "ffmpeg_bridge"),
    ("nvidia_video_enhancer/bridge/dxva_vsr_bridge.dll", "bridge"),
    ("nvidia_video_enhancer/fi/_rife_infer.py", "nvidia_video_enhancer/fi"),
    ("nvidia_video_enhancer/fi/_rife_model.py", "nvidia_video_enhancer/fi"),
    ("nvidia_video_enhancer/fi/warplayer.py", "nvidia_video_enhancer/fi"),
    ("nvidia_video_enhancer/fi/flownet.pkl", "nvidia_video_enhancer/fi"),
    ("nvidia_video_enhancer/sr/_nvvfx_infer.py", "nvidia_video_enhancer/sr"),
    ("nvidia_video_enhancer/ncnn", "nvidia_video_enhancer/ncnn"),
]

# 巨大的库, 不打包 (运行时按需 import)
# PyTorch 有 4GB+, 只有 RIFE 插帧和 nvvfx 超分才需要它
EXCLUDE_MODULES = [
    "torch", "torchvision", "torchaudio",
    "nvvfx",
    "tensorflow", "keras",
    "scipy", "pandas", "matplotlib",
    "PIL", "pillow",
    "jupyter", "ipykernel", "ipython",
    "tensorboard", "triton",
    "numba",
    "sympy", "networkx",
    "charset_normalizer.md__mypyc",
]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def main():
    if not shutil.which("pyinstaller"):
        print("PyInstaller 未安装。正在安装...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    add_data = []
    for src, dst in DATA_FILES:
        full = os.path.join(PROJECT_DIR, src)
        if os.path.isfile(full):
            add_data += ["--add-data", f"{full}{os.pathsep}{dst}"]
        elif os.path.isdir(full):
            for root, _, files in os.walk(full):
                for f in files:
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(os.path.dirname(fp), full)
                    if rel == ".":
                        target = dst
                    else:
                        target = os.path.join(dst, rel)
                    add_data += ["--add-data", f"{fp}{os.pathsep}{target}"]

    for d in ["build", "dist"]:
        path = os.path.join(PROJECT_DIR, d)
        if os.path.exists(path):
            shutil.rmtree(path)

    spec_file = os.path.join(PROJECT_DIR, f"{EXE_NAME}.spec")
    for f in [spec_file]:
        if os.path.exists(f):
            os.remove(f)

    exclude_flags = []
    for mod in EXCLUDE_MODULES:
        exclude_flags += ["--exclude-module", mod]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", EXE_NAME,
        "--distpath", OUTPUT_DIR,
        "--workpath", os.path.join(PROJECT_DIR, "build"),
        "--specpath", PROJECT_DIR,
        # 核心依赖
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "tkinter",
        "--hidden-import", "tqdm",
        "--hidden-import", "ctypes",
        "--hidden-import", "threading",
        # 项目模块
        "--hidden-import", "nvidia_video_enhancer",
        "--hidden-import", "nvidia_video_enhancer._paths",
        "--hidden-import", "nvidia_video_enhancer._env",
        "--hidden-import", "nvidia_video_enhancer.pipeline",
        "--hidden-import", "nvidia_video_enhancer.config",
        "--hidden-import", "nvidia_video_enhancer.cli",
        "--hidden-import", "nvidia_video_enhancer.gui",
        "--hidden-import", "nvidia_video_enhancer.utils",
        "--hidden-import", "nvidia_video_enhancer.sr",
        "--hidden-import", "nvidia_video_enhancer.sr.base",
        "--hidden-import", "nvidia_video_enhancer.sr.fallback",
        "--hidden-import", "nvidia_video_enhancer.sr.dxva_vsr",
        "--hidden-import", "nvidia_video_enhancer.fi",
        "--hidden-import", "nvidia_video_enhancer.fi.base",
        "--hidden-import", "nvidia_video_enhancer.fi.blend",
        "--hidden-import", "nvidia_video_enhancer.fi.optical_flow",
        "--hidden-import", "nvidia_video_enhancer.fi.dis_flow",
        "--hidden-import", "nvidia_video_enhancer.fi.rife_ncnn",
        "--hidden-import", "nvidia_video_enhancer.sr.realcugan_ncnn",
        "--hidden-import", "nvidia_video_enhancer.sr.realesrgan_ncnn",
        "--hidden-import", "nvidia_video_enhancer._logging",
        "--hidden-import", "nvidia_video_enhancer.ffmpeg_bridge",
        "--hidden-import", "nvidia_video_enhancer.ffmpeg_bridge.worker",
        "--hidden-import", "ncnn",
    ] + exclude_flags + add_data + [LAUNCHER]

    print("=" * 50)
    print("  PyInstaller 打包中 (不含 PyTorch/nvvfx)...")
    print("=" * 50)

    try:
        run(cmd, cwd=PROJECT_DIR)
    except subprocess.CalledProcessError:
        print("\n  ✗ 打包失败, 请检查上方错误信息")
        sys.exit(1)

    exe_path = os.path.join(OUTPUT_DIR, f"{EXE_NAME}.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n  ✓ 生成完成: {exe_path}")
        print(f"    大小: {size_mb:.1f} MB")
    else:
        print("\n  ✗ 未找到输出文件")
        sys.exit(1)

    for d in ["build"]:
        path = os.path.join(PROJECT_DIR, d)
        if os.path.exists(path):
            shutil.rmtree(path)
    for f in [spec_file]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
