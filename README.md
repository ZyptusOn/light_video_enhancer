# Light Video Enhancer

轻量级视频增强工具。**通过模拟媒体播放器的 D3D11 渲染管线，欺骗 NVIDIA 驱动对任意视频帧应用 RTX Video Super Resolution AI 超分**——这正是 VLC/PotPlayer 等播放器实现实时视频增强的底层原理。

配合光流法插帧和 NVENC 硬件编码，一条龙将低分辨率视频升级为高分辨率高帧率视频。

## 工作原理

```
解码 (NVDEC) → 光流插帧 (DIS/Farneback) → 超分 (RTX VSR) → 编码 (NVENC)
                                     ↘ BGR→NV12 → D3D11 VideoProcessorBlt ↗
```

核心技巧：构造假的 D3D11 视频渲染管线，把每一帧伪装成"正在播放的视频"，喂给 Video Processor。NVIDIA 驱动在底层拦截 VideoProcessorBlt 调用，自动启用 RTX VSR AI 模型处理——不需要调用任何 NVIDIA 私有 SDK，不依赖 PyTorch/CUDA Toolkit。

## 功能

- **超分辨率** — NVIDIA RTX VSR (D3D11 欺骗法) / 双三次 / Lanczos，支持任意倍率
- **帧插值** — DIS 光流 (SVP 风格) / Farneback 光流 / 混合，支持 2x/3x/4x 倍率
- **硬件编码** — NVENC H.264 / HEVC / AV1
- **零依赖打包** — 编译为单个 .exe，无需安装 Python、FFmpeg、CUDA

推荐场景：720p 30fps → 2K 60fps (2x 超分 + 2x 插帧)，体感流畅。

## 前提条件（实用角度仅推荐 NVIDIA）

- Windows 10/11
- **NVIDIA RTX 30/40/50 系列显卡** — 同时具备 RTX VSR AI 超分 + NVENC 硬件编码，完整体验
- NVIDIA 控制面板 → 视频 → 启用 RTX 视频增强

> 代码层面兼容 Intel Arc VSR 和 AMD D3D11 VP，但无硬件编码（回退到软件），实用价值有限。

## 快速开始

### 方式一：直接使用 .exe (推荐)

从 release 中下载 `.exe` 文件，双击启动 GUI，或拖拽视频文件到 .exe 上自动处理。

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

本项目所有代码由 **DeepSeek V4 Pro + TRAE Work** AI 辅助生成。

## 许可

本项目 Python/C++ 源码以 [MIT License](LICENSE) 许可分发。  
捆绑的 FFmpeg DLL 以 LGPL v2.1 许可分发，版权归 FFmpeg 项目所有。
