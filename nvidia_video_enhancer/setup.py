from setuptools import setup, find_packages

setup(
    name="nvidia-video-enhancer",
    version="1.0.0",
    description="轻量级视频超分 & 插帧工具 (D3D11 VSR / NVENC / RIFE / ncnn)",
    long_description=(
        "轻量级视频超分 & 插帧工具。\n\n"
        "支持 DXVA VSR (NVIDIA RTX 视频增强)、NVIDIA NGX VSR、"
        "Real-CUGAN ncnn、Real-ESRGAN PyTorch 等超分引擎，"
        "以及 DIS 光流、RIFE AI (PyTorch/ncnn)、光流法 (Farneback) 等插帧引擎。\n\n"
        "注意: 完整功能需要编译 ffmpeg_worker.dll 和 dxva_vsr_bridge.dll。\n"
        "详见项目中的 bridge/ 和 ffmpeg_bridge/ 目录。"
    ),
    packages=find_packages(),
    package_data={
        "nvidia_video_enhancer": [
            "ffmpeg_bridge/*.dll",
            "ffmpeg_bridge/*.py",
            "ffmpeg_dlls/*.dll",
            "bridge/*.dll",
            "fi/*.py",
            "fi/*.pkl",
            "sr/*.py",
            "ncnn/**/*",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "opencv-contrib-python>=4.8",
        "tqdm>=4.65",
    ],
    extras_require={
        "nvvfx": ["nvidia-vfx"],
        "rife": ["torch>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "nve=nvidia_video_enhancer.__main__:main",
        ],
    },
)
