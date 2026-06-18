# Light Video Enhancer

轻量级视频增强工具 — 超分辨率 + 帧插值 + NVENC 硬件编码，一条龙将低分辨率视频升级为高分辨率高帧率视频。

**核心亮点**：通过模拟媒体播放器的 D3D11 渲染管线，欺骗 NVIDIA 驱动对任意视频帧应用 RTX Video Super Resolution AI 超分——这正是 VLC/PotPlayer 等播放器实现实时视频增强的底层原理。

---

## 功能概览

### 超分辨率 (Super Resolution)

| 引擎 | 原理 | 依赖 |
|------|------|------|
| **DXVA VSR** | D3D11 VideoProcessorBlt 欺骗法，调用 NVIDIA RTX 视频增强 AI | FFmpeg Worker DLL 和 NVIDIA 显卡驱动 531.18 以后版本 |
| **NVIDIA NGX VSR** | NVIDIA VFX SDK 官方超分接口 | torch + nvidia-vfx |
| **Real-CUGAN ncnn** | 动漫优化 AI 超分 (ncnn-vulkan) | 无外部依赖 |
| **Real-ESRGAN** | 通用 AI 超分 (PyTorch) | torch |
| **双三次 / Lanczos** | 传统插值算法，零依赖 | 无 |

### 帧插值 (Frame Interpolation)

| 引擎 | 原理 | 依赖 |
|------|------|------|
| **RIFE AI (PyTorch)** | 最强 AI 插帧模型 | torch |
| **RIFE ncnn-vulkan** | RIFE 的 ncnn 实现，零外部依赖 | 无 |
| **DIS 光流** | SVP 同款稠密逆搜索光流法 | 无 (含于 OpenCV) |
| **GPU 光流** | CUDA 加速光流 (SVP 风格) | torch |
| **光流法 (Farneback)** | 经典 Farneback 光流 | 无 |
| **混合 (Blend)** | 简单帧混合 | 无 |

### 硬件编码

- NVENC H.264 / HEVC / AV1 硬件编码
- 通过 FFmpeg C API 内嵌调用，无需系统安装 FFmpeg
- 自动复制源文件音频流

---

## 工作原理

```
解码 (NVDEC) → 插帧 (RIFE / 光流) → 超分 (RTX VSR / ncnn) → 编码 (NVENC)
                                      ↘ BGR→NV12 → D3D11 VideoProcessorBlt ↗
```

核心技巧 (DXVA VSR)：构造假的 D3D11 视频渲染管线，把每一帧伪装成"正在播放的视频"，喂给 Video Processor。NVIDIA 驱动在底层拦截 VideoProcessorBlt 调用，自动启用 RTX VSR AI 模型处理——**不需要调用任何 NVIDIA 私有 SDK，不依赖 PyTorch/CUDA Toolkit**。

---

## 系统要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 64-bit |
| GPU | **NVIDIA RTX 30/40/50 系列** (推荐) — 完整体验：RTX VSR + NVENC |
| 驱动 | 最新 NVIDIA Game Ready / Studio 驱动 |
| 设置 | NVIDIA 控制面板 → 视频 → 启用 RTX 视频增强 |

> **非 NVIDIA GPU**：代码层面兼容 Intel Arc VSR 和 AMD D3D11 VP，但无硬件编码（回退到软件编码），可用的超分/插帧引擎也受限。推荐使用 ncnn 系列引擎。

---

## 快速开始

### 方式一：直接使用 .exe（推荐）

从 [Releases](https://github.com/ZyptusOn/light_video_enhancer/releases) 下载 `视频增强.exe`：

- **双击启动 GUI**，选择输入/输出文件，选择引擎，点击处理
- **拖拽视频文件到 .exe 上**，自动使用最佳引擎处理
- **命令行模式**：`视频增强.exe input.mp4 -o output.mp4 -s 2 --fi-engine rife_ncnn`

> .exe 已内置 ncnn 引擎 (RIFE / Real-CUGAN) 和 FFmpeg Worker DLL，无需安装 Python。

### 方式二：从源码运行

```bash
git clone https://github.com/ZyptusOn/light_video_enhancer.git
cd light_video_enhancer
pip install -r nvidia_video_enhancer/requirements.txt

# GUI 模式
python launcher.py

# 命令行模式
python -m nvidia_video_enhancer input.mp4 -o output.mp4 -s 2 --fi-engine rife_ncnn
```

### 命令行参数

```
nve input.mp4 -o output.mp4 [选项]

超分:
  -s, --scale FLOAT       超分倍率 (默认 2.0)
  -W, --width INT         输出宽度 (覆盖 --scale)
  -H, --height INT        输出高度 (覆盖 --scale)
  --sr-engine ENGINE      超分引擎: dxva_vsr, nvvfx, esrgan, realcugan, bicubic, lanczos, none

插帧:
  --fi-engine ENGINE      插帧引擎: dis, rife, rife_ncnn, torch_flow, optical_flow, blend, none
  --fi-multiplier INT     插帧倍率 (2/3/4, 默认 2)
  --fi-quality QUALITY    光流质量: ultra(极速), fast(快), balanced(均衡), quality(最佳)
  --sr-first              先超分再插帧 (默认先插帧再超分)

编码:
  --codec CODEC          编码器: h264_nvenc, hevc_nvenc, av1_nvenc
  --preset PRESET         NVENC preset: p1~p7 (默认 p7)
  --crf INT               质量 15~35 (默认 23, 越小越好)
  --container CONTAINER   容器: mp4, mkv, mov (默认 mp4)
  --fps FLOAT             覆盖输出帧率

其他:
  --start FLOAT           起始时间 (秒)
  --duration FLOAT        持续时长 (秒)
  --device {cuda,cpu}     计算设备 (默认 cuda)
```

**常用示例**：

```bash
# DXVA VSR 超分 2x + RIFE ncnn 插帧 2x (零依赖组合)
python -m nvidia_video_enhancer input.mp4 -o output.mp4 --sr-engine dxva_vsr --fi-engine rife_ncnn

# 仅超分 (双三次 2x)
python -m nvidia_video_enhancer input.mp4 -o output.mp4 --sr-engine bicubic --fi-engine none

# 仅插帧 (DIS 光流 2x → 60fps)
python -m nvidia_video_enhancer input.mp4 -o output.mp4 --sr-engine none --fi-engine dis

# Real-CUGAN ncnn 动漫超分 2x
python -m nvidia_video_enhancer input.mp4 -o output.mp4 --sr-engine realcugan --fi-engine none
```

---

## 编译

### 1. 编译 FFmpeg（从源码）

在 **MSYS2 UCRT64** 终端中运行:

```bash
cd ffmpeg_build
./build_ffmpeg.sh
```

> 需要: MSYS2 + MinGW-w64 UCRT64 gcc + ffnvcodec headers。  
> 产出: `ffmpeg/build/bin/*.dll` (avcodec, avformat, avutil, swscale 等)。

### 2. 编译 FFmpeg Worker DLL

```bash
cd nvidia_video_enhancer/ffmpeg_bridge
./build_worker.sh
```

> 产出: `ffmpeg_worker.dll`

### 3. 编译 D3D11 VSR Bridge DLL

```bash
cd nvidia_video_enhancer/bridge
./build.sh
```

> 产出: `dxva_vsr_bridge.dll`

### 4. 打包 .exe

```bash
python nvidia_video_enhancer/build_exe.py
```

> 产出: `视频增强.exe`

---

## 目录结构

```
light_video_enhancer/
├── launcher.py                    # PyInstaller 打包入口 + GUI/拖拽入口
├── ffmpeg_build/
│   └── build_ffmpeg.sh            # FFmpeg 源码编译脚本
├── nvidia_video_enhancer/
│   ├── __init__.py                # 包描述
│   ├── __main__.py                # CLI 入口 (python -m nvidia_video_enhancer)
│   ├── _env.py                    # 系统 Python/torch 环境检测
│   ├── _logging.py                # 统一日志模块
│   ├── _paths.py                  # 路径工具 (frozen/源码模式)
│   ├── pipeline.py                # 视频处理流水线
│   ├── config.py                  # 配置数据类
│   ├── cli.py                     # 命令行参数解析
│   ├── gui.py                     # Tkinter 图形界面
│   ├── utils.py                   # 引擎可用性检测
│   ├── build_exe.py               # PyInstaller 打包脚本
│   ├── setup.py                   # pip 安装
│   ├── requirements.txt           # Python 依赖
│   │
│   ├── ffmpeg_bridge/             # FFmpeg C API 包装器
│   │   ├── ffmpeg_worker.c        # C 源码 (解码/编码)
│   │   ├── ffmpeg_worker.dll      # 编译产物
│   │   ├── build_worker.sh        # 编译脚本
│   │   ├── worker.py              # Python ctypes 绑定
│   │   └── __init__.py
│   │
│   ├── ffmpeg_dlls/               # FFmpeg 共享库 (LGPL)
│   │   ├── avcodec-62.dll, avformat-62.dll, avutil-60.dll, swscale-9.dll
│   │   └── ... (运行时依赖库)
│   │
│   ├── bridge/                    # D3D11 VSR Bridge
│   │   ├── dxva_vsr_bridge.cpp    # C++ 源码
│   │   ├── dxva_vsr_bridge.dll    # 编译产物
│   │   └── build.sh
│   │
│   ├── sr/                        # 超分引擎
│   │   ├── base.py                # 抽象基类
│   │   ├── dxva_vsr.py            # DXVA VSR (D3D11 欺骗法)
│   │   ├── nvvfx_sr.py            # NVIDIA NGX VSR
│   │   ├── _nvvfx_infer.py        # nvvfx 子进程推理
│   │   ├── realcugan_ncnn.py      # Real-CUGAN ncnn
│   │   ├── realesrgan_ncnn.py     # Real-ESRGAN (PyTorch)
│   │   └── fallback.py            # 双三次/Lanczos
│   │
│   ├── fi/                        # 插帧引擎
│   │   ├── base.py                # 抽象基类
│   │   ├── rife.py                # RIFE AI (PyTorch)
│   │   ├── rife_ncnn.py           # RIFE ncnn-vulkan
│   │   ├── dis_flow.py            # DIS 光流
│   │   ├── torch_flow.py          # CUDA 光流
│   │   ├── optical_flow.py        # Farneback 光流
│   │   ├── blend.py               # 混合插帧
│   │   ├── _rife_infer.py         # RIFE PyTorch 推理
│   │   ├── _rife_model.py         # RIFE 模型定义
│   │   ├── warplayer.py           # 帧变形工具
│   │   └── flownet.pkl            # RIFE 模型权重
│   │
│   └── ncnn/                      # ncnn 引擎 + 模型文件
│       ├── rife/                   # RIFE ncnn-vulkan
│       ├── realcugan/              # Real-CUGAN ncnn-vulkan
│       └── realesrgan/             # Real-ESRGAN ncnn-vulkan
```

---

## PyTorch 环境配置（可选）

以下引擎需要 PyTorch + CUDA **外部 Python 环境**（.exe 不打包 torch）：

| 引擎 | torch 用途 |
|------|-----------|
| RIFE AI (PyTorch) | AI 插帧推理 |
| GPU 光流 (SVP 风格) | CUDA 加速光流计算 |
| NVIDIA NGX VSR | NVIDIA VFX SDK 超分 |
| Real-ESRGAN | AI 超分推理 |

安装方式（通过 conda）：

```bash
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia
pip install nvidia-vfx   # 可选: NVIDIA NGX VSR
```

程序启动时会**自动扫描系统上的 Python 环境**，优先选择带 torch+CUDA 的环境。无需设置环境变量。

---

## FFmpeg 许可声明

本项目使用了 [FFmpeg](https://ffmpeg.org/) 多媒体框架，FFmpeg 以 **LGPL v2.1** 许可分发。  
本项目通过动态链接（.dll）方式使用 FFmpeg 库，符合 LGPL 条款。  
用户可以自行替换 `ffmpeg_dlls/` 目录下的 FFmpeg DLL 文件。

FFmpeg 源码：<https://git.ffmpeg.org/ffmpeg.git>

---

## AI 生成声明

本项目所有代码由 **TRAE 工作台 (DeepSeek V4 Pro)** AI 辅助生成。

---

## 致谢

本项目使用或参考了以下开源项目：

| 项目 | 用途 | 许可 |
|------|------|------|
| [FFmpeg](https://ffmpeg.org/) | 多媒体解码/编码/muxer | LGPL v2.1 |
| [ffnvcodec](https://github.com/FFmpeg/nv-codec-headers) | NVENC/NVDEC 头文件 | MIT |
| [ncnn](https://github.com/Tencent/ncnn) | 高性能神经网络推理框架 | BSD-3-Clause |
| [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | RIFE ncnn-vulkan 移植 | MIT |
| [realcugan-ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan) | Real-CUGAN ncnn-vulkan 移植 | MIT |
| [realesrgan-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | Real-ESRGAN ncnn-vulkan 移植 | MIT |
| [RIFE](https://github.com/hzwer/ECCV2022-RIFE) | AI 插帧模型架构 | MIT |
| [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | RIFE 实用实现 | MIT |
| [Real-CUGAN](https://github.com/bilibili/ailab) | 动漫超分模型 (bilibili) | MIT |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 通用超分模型 | BSD-3-Clause |
| [NVIDIA Video Effects SDK](https://github.com/NVIDIA/DLSS-Native) | NGX VSR 超分 | NVIDIA EULA |
| [SVP](https://www.svp-team.com/) | 光流插帧思路参考 | — |
| [VLC](https://github.com/videolan/vlc) | D3D11 渲染管线参考 | LGPL v2.1 |
| [OpenCV](https://github.com/opencv/opencv) | 图像处理 / 光流算法 | Apache 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | AI 推理框架 | BSD-3-Clause |

---

## 许可

本项目 Python/C++ 源码以 [MIT License](LICENSE) 许可分发。  
捆绑的 FFmpeg DLL 以 LGPL v2.1 许可分发，版权归 FFmpeg 项目所有。  
`ncnn/` 目录下各子项目版权归原作者所有，遵循其各自的许可。
