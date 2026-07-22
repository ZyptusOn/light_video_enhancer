# Light Video Enhancer

[![Release](https://img.shields.io/github/v/release/ZyptusOn/light_video_enhancer?display_name=tag)](https://github.com/ZyptusOn/light_video_enhancer/releases/latest)
[![Windows](https://img.shields.io/badge/Windows-7%20SP1%20%7C%2010%20%7C%2011-0078D6)](#系统与硬件)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Light Video Enhancer 是一款面向 Windows 的视频超分、插帧与转码工具。它最初源于一个直接的想法：模仿 VLC 等播放器的 D3D11 视频呈现路径，让显卡驱动把离线视频帧当作正在播放的内容，从而调用驱动提供的视频增强能力。现在它已经发展为一条可组合的处理流水线，同时支持 NVIDIA Video Effects、RIFE、ncnn-vulkan、OpenCV 光流和多种软硬件编码器。

项目不再假设计算机一定安装 NVIDIA 显卡。NVIDIA 用户可以使用驱动 VSR、NV-VFX、CUDA RIFE 和 NVENC；Intel、AMD 或较旧系统可以选择 Vulkan NCNN、OpenCV、Media Foundation 或软件编码后端。

## 下载

| 系统 | 文件 | 说明 |
|---|---|---|
| Windows 10/11 x64 | [LightVideoEnhancer-Win10-11-x64.exe](https://github.com/ZyptusOn/light_video_enhancer/releases/download/v0.4.5/LightVideoEnhancer-Win10-11-x64.exe) | 推荐版本；支持现代 DPI、长路径和融合 CUDA 快速路径。 |
| Windows 7 SP1 x64 | [LightVideoEnhancer-Win7-x64.exe](https://github.com/ZyptusOn/light_video_enhancer/releases/download/v0.4.5/LightVideoEnhancer-Win7-x64.exe) | 使用 Python 3.8.10 与兼容版 PyInstaller 构建。 |

两个文件都是单文件 GUI，已内置 FFmpeg Worker、NCNN 命令行后端和模型，不需要安装系统 FFmpeg。首次启动会解压运行资源，杀毒软件扫描和 Vulkan 缓存建立可能使第一次使用稍慢。

### v0.4.5 校验值

| 文件 | SHA-256 |
|---|---|
| `LightVideoEnhancer-Win10-11-x64.exe` | `DE89B4DE6EBE8522EAFFE1F4F1A7D82161442C8EE92B9EEA25E01EBEAB339BD4` |
| `LightVideoEnhancer-Win7-x64.exe` | `B6EA4B2316FC78093EB348FE6F8732F66597A1274A6EDA38E37356079B596312` |

## v0.4.5 亮点

- 项目和 Python 包正式更名为 `light_video_enhancer`，命令行入口改为 `lve`。
- 新增 Real-ESRGAN AnimeVideo-v3、x4plus、x4plus-anime，以及原始 ESRGAN x4 感知模型。
- RIFE NCNN 与 NCNN 超分使用目录批处理和双向目录直连，避免中间帧反复进入 Python。
- 批量大小按分辨率、插帧倍率和模型原生输出动态计算；编码队列与临时目录清理可和后续 Vulkan 推理重叠。
- Windows 10/11 的 `RIFE PyTorch -> NV-VFX` 可进入融合 CUDA Worker，减少 GPU/CPU 往返。
- 完善 H.264、H.265/HEVC、AV1 的 NVIDIA、AMD、Media Foundation 和软件编码回退。
- 修复跨 NumPy 环境的 `No module named 'numpy._core'`、编码末帧丢失、帧率截断、音频片段时间戳和环境扫描遗漏等问题。
- 分离 Windows 7 与 Windows 10/11 发布构建，并保留传统缩放、OpenCV 插帧和兼容编码路径。

完整变化见 [CHANGELOG.md](CHANGELOG.md)。

## 处理流水线

```text
输入视频
   │
   ├─ 内嵌 FFmpeg 解码：CUDA / D3D11VA / dav1d / 软件回退
   │
   ├─ 插帧：RIFE PyTorch / RIFE NCNN / DIS / Farneback / CUDA 光流 / Blend
   │
   ├─ 超分：驱动 VSR / NV-VFX / Real-CUGAN / Real-ESRGAN / ESRGAN / Lanczos
   │
   └─ 编码：NVENC / AMF / Media Foundation / x264 / x265 / SVT-AV1 / libaom
          └─ 可复制原视频音频，失败时保留同格式优先的自动回退
```

默认顺序是先插帧、后超分，通常更快；勾选“先超分再插帧”会提高插帧输入分辨率，可能改善细节，也会显著增加计算量。连续的 NCNN 后端会在安全条件下自动直连目录。

## 后端

### 超分辨率

| GUI / CLI | 设备 | 适用场景 |
|---|---|---|
| 驱动 VSR / `dxva_vsr` | D3D11，NVIDIA / Intel / AMD | 模拟播放器 Video Processor 路径；速度快，增强效果取决于显卡和驱动。 |
| NVIDIA Video Effects / `nvvfx` | NVIDIA CUDA | NVIDIA SDK AI 超分；支持隔离进程、超时保护和融合 CUDA 管线。 |
| Real-CUGAN / `realcugan` | Vulkan | 动漫、线稿和压缩视频；2×/3×/4× NCNN 批处理。 |
| Real-ESRGAN / `realesrgan` | Vulkan | 通用或动漫视频；提供轻量视频模型和 x4plus 系列。 |
| ESRGAN classic / `esrgan` | Vulkan | 原始 ESRGAN x4 感知模型，偏锐利和纹理增强。 |
| Lanczos / `lanczos` | CPU | 稳定、清晰的传统缩放回退。 |
| Bicubic / `bicubic` | CPU | 平滑的传统缩放回退。 |

Real-ESRGAN 质量档位：

| 档位 | 模型 / 行为 |
|---|---|
| `fast` | AnimeVideo-v3 2×/3×/4×，按目标倍率选择原生模型。 |
| `balanced` | x4plus-anime，模型输出后按目标尺寸缩放。 |
| `quality` | 通用 x4plus。 |
| `ultra` | 通用 x4plus + TTA，最慢且显存占用最高。 |

### 帧插值

| GUI / CLI | 设备 | 说明 |
|---|---|---|
| RIFE / `rife` | PyTorch CUDA/CPU | 质量优先；持久子进程、共享内存和静帧/切镜检测。 |
| RIFE NCNN / `rife_ncnn` | Vulkan | 不依赖 PyTorch，目录批处理，适合便携发布。 |
| DIS / `dis` | CPU / OpenCV | 推荐的非 AI 通用方案，速度和运动补偿较均衡。 |
| Farneback / `optical_flow` | CPU / OpenCV | 经典稠密光流。 |
| CUDA 光流 / `torch_flow` | PyTorch CUDA | 轻量块匹配光流。 |
| 帧混合 / `blend` | CPU | 最快、最兼容，不进行运动估计。 |

“插帧质量”只在后端确实暴露可调策略时启用；RIFE 模型、RIFE NCNN 和帧混合使用固定或自动策略时，GUI 会禁用该控件。

### 编码

| 格式 | 硬件编码 | 内置软件编码 |
|---|---|---|
| H.264/AVC | `h264_nvenc`、`h264_amf`、`h264_mf` | `libx264`，别名 `x264` / `h264` |
| H.265/HEVC | `hevc_nvenc`、`hevc_amf`、`hevc_mf` | `libx265`，别名 `x265` / `h265` |
| AV1 | `av1_nvenc`、`av1_amf` | `libsvtav1`、`libaom-av1`，别名 `av1` / `svt-av1` / `aom` |
| 兼容回退 | — | `mpeg4` |

无法打开指定编码器时，程序先在同一格式内寻找其他实现。AV1 全部不可用时再依次尝试 HEVC、H.264 和 MPEG-4。CQ/CRF 越小质量越高；H.264/H.265 可从 18–28 起步，AV1 可从 24–36 起步。

## 系统与硬件

| 环境 | 建议 |
|---|---|
| Windows 10/11 x64 | 使用现代版 EXE；推荐具备 Vulkan 1.1 的 GPU。NVIDIA CUDA 用户可选 RIFE、NV-VFX 和 NVENC。 |
| Windows 7 SP1 x64 | 使用 Win7 专用 EXE；建议安装 KB2670838 和 KB2999226。稳定基线是传统缩放、OpenCV 插帧、Media Foundation/MPEG-4。 |
| NVIDIA | 驱动 VSR、NV-VFX、CUDA RIFE、NVENC 和 Vulkan NCNN 可按驱动/SDK能力组合。 |
| Intel / AMD | 可使用 Vulkan NCNN、OpenCV 和软件编码；支持的 D3D11/AMF/Media Foundation 能力取决于驱动。 |

某个后端出现在 GUI 中表示程序资源完整，不代表当前显卡驱动一定支持它。特别是新式 RTX VSR、CUDA、NV-VFX、AV1 硬编通常不适用于 Windows 7。

## 快速开始

### GUI

1. 从 [Releases](https://github.com/ZyptusOn/light_video_enhancer/releases/latest) 下载对应系统的 EXE。
2. 选择输入视频和输出路径。
3. 选择超分、插帧、编码器及各自质量档位。
4. 可选设置开始时间、处理时长、目标帧率和 NCNN GPU。
5. 点击“开始处理”；日志会显示实际管线和自动回退结果。

推荐组合：

- NVIDIA 高质量：`RIFE PyTorch -> NVIDIA Video Effects -> HEVC/AV1 NVENC`
- 跨厂商便携：`RIFE NCNN -> Real-CUGAN/Real-ESRGAN -> 软件或硬件编码`
- 低依赖稳定：`DIS -> Lanczos -> H.264 Media Foundation/x264`

### 从源码运行

```powershell
git clone https://github.com/ZyptusOn/light_video_enhancer.git
cd light_video_enhancer
python -m pip install -r requirements.txt
python -m pip install -e .
python -m light_video_enhancer
```

Windows 7 请使用 CPython 3.8.10：

```powershell
py -3.8 -m pip install -r requirements-win7.txt
py -3.8 -m pip install -e . --no-deps
py -3.8 -m light_video_enhancer
```

也可以双击 `启动GUI.bat`。

## 命令行示例

安装项目后使用 `lve`；也可以将 `lve` 替换为 `python -m light_video_enhancer`。

```powershell
# 自动选择后端，2× 超分 + 2× 插帧
lve input.mp4 -o output.mp4 -s 2 --fi-multiplier 2 -y

# RIFE NCNN + Real-ESRGAN + AV1 NVENC
lve input.mp4 -o output.mp4 --fi-engine rife_ncnn --sr-engine realesrgan --sr-quality quality --codec av1_nvenc -y

# 经典 ESRGAN x4 感知模型，输出目标仍为 2×
lve input.mp4 -o output.mp4 -s 2 --fi-engine none --sr-engine esrgan --sr-quality quality --codec x265 -y

# 只处理从第 60 秒开始的 30 秒，并输出 60 fps
lve input.mkv -o clip.mp4 --start 60 --duration 30 --fps 60 --sr-engine none --fi-engine blend -y

# 指定第二个 Vulkan GPU，并使用 CPU x264 编码
lve input.mp4 -o output.mp4 --ncnn-gpu 1 --sr-engine realcugan --fi-engine rife_ncnn --codec x264 -y
```

完整参数见：

```powershell
lve --help
```

## 外部 PyTorch / NVIDIA VFX 环境

单文件 EXE 不捆绑 PyTorch。RIFE PyTorch、CUDA 光流和 NV-VFX 会按需使用外部 Python 环境。GUI 启动时只做快速文件检查；在“环境与后端”页点击扫描后，程序会并行检查 PATH、Python Launcher、注册表、Conda、uv、pyenv 和 Poetry 等常见位置，并缓存结果。

如果机器有多个环境，可以在 GUI 中明确选择，也可以在 CLI 中指定：

```powershell
lve input.mp4 --fi-engine rife --torch-python C:\path\to\python.exe -o output.mp4 -y
```

跨环境 IPC 不序列化 NumPy 对象，因此允许主程序和外部 PyTorch 环境分别使用 NumPy 1.x/2.x，避免 `numpy._core` 反序列化错误。

## 性能说明

`rife_ncnn -> NCNN 超分` 的旧实现会把 RIFE 输出读回 Python，再写回磁盘交给超分进程。v0.4.5 改为目录直连，并增加动态批量、动态编码队列和异步临时目录清理。

本机 720p、33 帧合成基准：

| 路径 | 耗时 |
|---|---:|
| 旧的 Python 中间帧路径 | 22.256 秒 |
| v0.4.5 NCNN 目录直连 | 15.425 秒 |

该测试提升约 `1.443×`，总耗时减少约 `30.7%`。实际收益取决于分辨率、模型、磁盘、GPU 和编码器。任务管理器中仍可能看到周期性 GPU 波峰，因为 RIFE、超分和编码属于不同阶段；优化目标是缩短波峰之间的 CPU/磁盘空档，而不是让所有 GPU 引擎同时运行。

可复现实验：

```powershell
python benchmarks/benchmark_ncnn_chain.py --width 1280 --height 720 --frames 33 --scale 2 --gpu 0
```

## 构建与测试

安装测试依赖后运行：

```powershell
python -m unittest discover -s tests -v
```

现代单文件包：

```powershell
python -m pip install -r requirements-build.txt
python build_exe.py
```

Win7 单文件包必须使用 Python 3.8.10：

```powershell
py -3.8 -m pip install -r requirements-build-win7.txt
py -3.8 build_exe.py
```

FFmpeg、FFmpeg Worker 和 D3D11 Bridge 的重建说明见构建脚本：

```bash
./build_ffmpeg.sh
./light_video_enhancer/ffmpeg_bridge/build_worker.sh
./light_video_enhancer/bridge/build.sh
```

## 目录结构

```text
light_video_enhancer/       Python 包
  bridge/                   D3D11 Video Processor Bridge
  ffmpeg_bridge/            内嵌 FFmpeg C API Worker
  ffmpeg_dlls/              发布运行库
  fi/                       插帧后端
  sr/                       超分后端
  ncnn/                     便携 NCNN 程序、模型和各自许可证
benchmarks/                 GPU/NCNN 性能基准
tests/                      单元与真实编解码往返测试
build_exe.py                双目标 PyInstaller 构建脚本
launcher.py                 GUI 启动入口
```

## 从旧版迁移

- Python 包：`nvidia_video_enhancer` → `light_video_enhancer`
- 命令行：`nve` → `lve`
- 源码启动：`python -m light_video_enhancer`
- v0.4.5 的 GUI 配置项有所增加，建议重新检查超分质量、插帧质量、编码 preset 和 NCNN GPU。

## 已知限制

- 驱动 VSR 的增强效果由显卡、驱动设置和输入内容决定，程序无法保证所有设备都启用厂商 AI 模型。
- Vulkan NCNN 后端仍以独立进程分批执行；目录直连显著减少额外读写，但不会消除阶段切换。
- `ultra` 通常启用 TTA，速度和显存成本可能远高于 `quality`。
- Win7 构建兼容 GUI 和基础流水线，但现代 CUDA、RTX VSR、AV1 硬件编解码是否可用取决于厂商是否仍提供兼容驱动。
- 发布包较大，因为它同时包含 FFmpeg 运行库、多个 NCNN 可执行程序和模型。

## 致谢

项目使用或参考了 [FFmpeg](https://ffmpeg.org/)、[ncnn](https://github.com/Tencent/ncnn)、[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)、[Real-CUGAN](https://github.com/bilibili/ailab)、[realcugan-ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan)、[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)、[Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)、[ESRGAN](https://github.com/xinntao/ESRGAN)、[RIFE](https://github.com/hzwer/ECCV2022-RIFE)、[OpenCV](https://github.com/opencv/opencv) 和 [VLC](https://github.com/videolan/vlc)。

项目最初由 TRAE / DeepSeek V4 Pro 辅助创建，后续重构、兼容性修复、性能优化和发布同样使用了 AI 辅助。所有生成或修改内容均应以实际代码、测试和目标硬件验证为准。

## 许可证

项目自身代码采用 [MIT License](LICENSE)。捆绑的 FFmpeg、x264、x265、dav1d、SVT-AV1、libaom、RIFE、Real-CUGAN、Real-ESRGAN、ESRGAN、ncnn 以及厂商 SDK/驱动分别受各自许可证或使用条款约束。当前 FFmpeg 构建启用了 GPL 的 x264/x265，因此分发包需要按 GPL v2 或更高版本履行相应义务；当前构建未启用 `nonfree`。
