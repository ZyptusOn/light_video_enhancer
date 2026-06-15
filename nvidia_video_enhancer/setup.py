from setuptools import setup, find_packages

setup(
    name="nvidia-video-enhancer",
    version="1.0.0",
    description="视频超分 & 插帧工具 (D3D11 VSR / 光流 / RIFE)",
    packages=find_packages(),
    package_data={
        "nvidia_video_enhancer": [
            "ffmpeg_bridge/*.dll",
            "ffmpeg_dlls/*.dll",
            "bridge/*.dll",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24",
        "opencv-python>=4.8",
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
