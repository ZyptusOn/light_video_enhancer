# Light Video Enhancer

轻量级视频增强工具，利用 GPU 硬件加速实现超分辨率、帧插值和编码。
其中超分辨率RTX VSR方法为伪装d3d11播放器实现。

## 功能

- **超分辨率** — NVIDIA RTX VSR / Intel VSR / AMD D3D11 VP / 双三次 / Lanczos
- **帧插值** — DIS 光流 (SVP 风格) / Farneback 光流 / 混合 / RIFE AI，支持 2x/3x/4x 倍率
- **编码** — NVENC H.264 / HEVC / AV1 硬件编码 (NVIDIA)，软件编码回退
- **打包** — 可编译为单个 .exe，无需安装 Python 或 FFmpeg

推荐场景：720p 30fps → 2K 60fps (2x 超分 + 2x 插帧)，体感流畅。也支持 1080p → 4K 等高负载场景。

## 前提条件

- Windows 10/11
- 支持 Direct3D 11 Video Processor API 的显卡：
  - **NVIDIA** RTX 30/40/50 系列 — 完整 VSR AI 超分 + NVENC 硬件编码
  - **Intel** Arc / 核显 (11代+) — Intel VSR 超分
  - **AMD** — D3D11 Video Processor (无 AI 增强)，编码回退到软件
- NVIDIA 用户：控制面板 → 视频 → 启用 RTX 视频增强 (AI 超分前提)

## 快速开始

### 方式一：直接使用 .exe (推荐)

从release中下载 `.exe` 文件，双击启动 GUI，或拖拽视频文件到 .exe 上自动处理。

### 方式二：从源码运行

```bash
pip install -r requirements.txt
# GUI 模式
python launcher.py
# 命令行模式
python -m nvidia_video_enhancer input.mp4 output.mp4
```

### 编译 .exe

```bash
# 1. 编译 FFmpeg Worker DLL (MSYS2 UCRT64)
cd nvidia_video_enhancer/ffmpeg_bridge && ./build_worker.sh

# 2. 编译 D3D11 VSR Bridge DLL
cd nvidia_video_enhancer/bridge && ./build.sh

# 3. 打包 .exe
python nvidia_video_enhancer/build_exe.py
```

## FFmpeg 许可声明

本项目使用了 [FFmpeg](https://ffmpeg.org/) 多媒体框架，FFmpeg 以 **LGPL v2.1** 许可分发。  
本项目通过动态链接（.dll）方式使用 FFmpeg 库，符合 LGPL 条款。  
用户可以自行替换 `ffmpeg_dlls/` 目录下的 FFmpeg DLL 文件。

FFmpeg 源码：<https://git.ffmpeg.org/ffmpeg.git>

## AI 生成声明

本项目所有代码由 **DeepSeekV4Pro + TRAE Work** AI 辅助生成。

## 许可

本项目 Python/C++ 源码以 [MIT License](LICENSE) 许可分发。  
捆绑的 FFmpeg DLL 以 LGPL v2.1 许可分发，版权归 FFmpeg 项目所有。
